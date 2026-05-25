from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import yaml

try:
    from acceptance.tools.write_result import write_case_result
except ModuleNotFoundError:
    from write_result import write_case_result


REQUIRED_CASE_FIELDS = {
    "case_id",
    "title",
    "scope",
    "repo_target",
    "priority",
    "status",
    "golden",
    "actual",
    "compare",
    "evidence",
    "result",
}

REQUIRED_METADATA_FIELDS = {
    "case_id",
    "dtype",
    "shape",
    "tensor_rank",
    "layout",
    "seed",
    "distributed",
    "environment",
    "version",
    "epsilon",
    "gamma_shape",
    "artifact_key",
    "compared_keys",
    "optional_keys",
    "golden_source_kind",
    "actual_source_kind",
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip('"\'')


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_case(case: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_CASE_FIELDS - set(case))
    if missing:
        raise ValueError(f"Missing case fields: {missing}")
    for section in ("golden", "actual"):
        for field in ("source", "command", "artifact_dir"):
            if field not in case[section]:
                raise ValueError(f"Missing {section}.{field}")
    if "artifact" not in case["result"]:
        raise ValueError("Missing result.artifact")


def artifact_paths(case: dict[str, Any]) -> tuple[Path, Path, Path]:
    return (
        Path(case["golden"]["artifact_dir"]),
        Path(case["actual"]["artifact_dir"]),
        Path(case["result"]["artifact"]),
    )


def has_required_artifacts(case: dict[str, Any]) -> bool:
    golden_dir, actual_dir, _ = artifact_paths(case)
    return golden_dir.exists() and actual_dir.exists() and any(golden_dir.glob("*.npz")) and any(actual_dir.glob("*.npz"))


def load_npz_dir(path: Path) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for npz_path in sorted(path.glob("*.npz")):
        with np.load(npz_path) as loaded:
            for key in loaded.files:
                tensors[f"{npz_path.stem}.{key}"] = loaded[key]
    return tensors


def load_npz_file(path: Path, key: str) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        if key not in loaded.files:
            raise KeyError(f"Missing key {key!r} in {path}")
        return {key: loaded[key]}


def dtype_threshold(case: dict[str, Any], arrays: list[np.ndarray]) -> dict[str, float]:
    thresholds = case["compare"].get("threshold", {})
    dtype_key = "bf16" if any(array.dtype == np.dtype("float16") for array in arrays) else "fp32"
    return thresholds.get(dtype_key, thresholds.get("fp32", {"atol": 0.0, "rtol": 0.0, "pass_ratio": 1.0}))


def validate_rmsnorm_metadata(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_METADATA_FIELDS - set(metadata))
    if missing:
        errors.append(f"missing metadata fields: {missing}")
    if metadata.get("rank") == 3 and "tensor_rank" not in metadata:
        errors.append("rank=3 is invalid for tensor rank; use tensor_rank=3 and distributed.rank=0")
    expected = {
        "case_id": "OP-RMSNORM-001",
        "dtype": "fp32",
        "shape": [2, 16, 128],
        "tensor_rank": 3,
        "seed": 0,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"metadata {key} expected {value!r}, got {metadata.get(key)!r}")
    distributed = metadata.get("distributed", {})
    if distributed.get("rank") != 0:
        errors.append("metadata distributed.rank must be 0")
    if distributed.get("world_size") != 1:
        errors.append("metadata distributed.world_size must be 1")
    if metadata.get("golden_source_kind") != "torch_npu_runtime":
        errors.append("golden_source_kind must be torch_npu_runtime for migration pass")
    if metadata.get("actual_source_kind") != "mindspore_runtime":
        errors.append("actual_source_kind must be mindspore_runtime for migration pass")
    return errors


def metadata_result(case: dict[str, Any], result_path: Path, metadata_path: Path, errors: list[str]) -> dict[str, Any]:
    write_case_result(
        result_path,
        case_id=case["case_id"],
        status="blocked",
        stage="compare",
        evidence_grade="tool_contract_only",
        summary="Metadata is missing or invalid; compare cannot produce migration evidence.",
        evidence={"golden": str(Path(case["golden"]["artifact_dir"])), "actual": str(Path(case["actual"]["artifact_dir"])), "metadata": str(metadata_path)},
        reason="; ".join(errors),
        next_step="Fix OP-RMSNORM-001 metadata before rerunning compare_outputs.py.",
    )
    return json.loads(result_path.read_text(encoding="utf-8"))


def evidence_grade(metadata: dict[str, Any], status: str) -> str:
    if status == "pass" and metadata.get("golden_source_kind") == "torch_npu_runtime" and metadata.get("actual_source_kind") == "mindspore_runtime":
        return "migration_evidence"
    if metadata.get("golden_source_kind") in {"pt_reference", "manual_reference"}:
        return "provisional_reference"
    return "tool_contract_only"


def compare_arrays(
    case: dict[str, Any],
    golden: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    common: list[str],
) -> tuple[str, dict[str, Any], str]:
    arrays = [golden[key] for key in common] + [actual[key] for key in common]
    threshold = dtype_threshold(case, arrays)
    atol = float(threshold.get("atol", 0.0))
    rtol = float(threshold.get("rtol", 0.0))
    min_pass_ratio = float(threshold.get("pass_ratio", 1.0))
    finite = all(np.isfinite(array).all() for array in arrays)
    max_abs = 0.0
    max_rel = 0.0
    total = 0
    passed = 0
    for key in common:
        g = golden[key].astype(np.float64)
        a = actual[key].astype(np.float64)
        diff = np.abs(g - a)
        rel = diff / np.maximum(np.abs(g), 1e-12)
        ok = diff <= (atol + rtol * np.abs(g))
        max_abs = max(max_abs, float(diff.max(initial=0.0)))
        max_rel = max(max_rel, float(rel.max(initial=0.0)))
        total += int(ok.size)
        passed += int(ok.sum())
    pass_ratio = 1.0 if total == 0 else passed / total
    status = "pass" if finite and pass_ratio >= min_pass_ratio and math.isfinite(max_abs) and math.isfinite(max_rel) else "fail"
    metrics = {
        "finite_check": "pass" if finite else "fail",
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "pass_ratio": pass_ratio,
    }
    reason = "" if status == "pass" else "Tensor comparison did not satisfy finite/threshold checks."
    return status, metrics, reason


def compare_case(
    case: dict[str, Any],
    golden_path: str | Path | None = None,
    actual_path: str | Path | None = None,
    metadata_path_override: str | Path | None = None,
    key: str | None = None,
) -> dict[str, Any]:
    validate_case(case)
    golden_dir, actual_dir, result_path = artifact_paths(case)
    metadata_path = Path(metadata_path_override) if metadata_path_override else result_path.parent / "metadata.json"
    artifact_key = key or "output"

    if golden_path or actual_path:
        if not golden_path or not actual_path or not Path(golden_path).exists() or not Path(actual_path).exists():
            write_case_result(
                result_path,
                case_id=case["case_id"],
                status="not_run",
                stage="compare",
                evidence_grade="tool_contract_only",
                summary="Required explicit golden/actual artifacts are missing.",
                evidence={"golden": str(golden_path or ""), "actual": str(actual_path or ""), "metadata": str(metadata_path)},
                reason="Missing explicit golden or actual .npz artifact.",
                next_step="Collect torch_npu golden and MindSpore actual artifacts, then rerun compare_outputs.py.",
            )
            return json.loads(result_path.read_text(encoding="utf-8"))
        try:
            golden = load_npz_file(Path(golden_path), artifact_key)
            actual = load_npz_file(Path(actual_path), artifact_key)
        except KeyError as exc:
            write_case_result(
                result_path,
                case_id=case["case_id"],
                status="blocked",
                stage="compare",
                evidence_grade="tool_contract_only",
                summary="Artifact key is missing; compare cannot run.",
                evidence={"golden": str(golden_path), "actual": str(actual_path), "metadata": str(metadata_path)},
                reason=str(exc),
                next_step="Write matching artifact keys before rerunning compare_outputs.py.",
            )
            return json.loads(result_path.read_text(encoding="utf-8"))
        common = [artifact_key]
    else:
        if not has_required_artifacts(case):
            write_case_result(
                result_path,
                case_id=case["case_id"],
                status="not_run",
                stage="compare",
                evidence_grade="tool_contract_only",
                summary="Required golden/actual artifacts are missing; skeleton contract was not executed against real tensors.",
                evidence={"golden": str(golden_dir), "actual": str(actual_dir), "metadata": str(metadata_path)},
                reason="Missing golden or actual .npz artifacts.",
                next_step="Collect torch_npu/PT golden artifacts and MindSpore actual artifacts, then rerun compare_outputs.py.",
            )
            return json.loads(result_path.read_text(encoding="utf-8"))
        golden = load_npz_dir(golden_dir)
        actual = load_npz_dir(actual_dir)
        common = sorted(set(golden) & set(actual))
        if not common:
            write_case_result(
                result_path,
                case_id=case["case_id"],
                status="fail",
                stage="compare",
                evidence_grade="tool_contract_only",
                summary="No matching tensor keys found between golden and actual artifacts.",
                evidence={"golden": str(golden_dir), "actual": str(actual_dir), "metadata": str(metadata_path)},
                reason="Golden/actual .npz keys do not overlap.",
                next_step="Align artifact tensor key names before comparing.",
            )
            return json.loads(result_path.read_text(encoding="utf-8"))

    if not metadata_path.exists():
        return metadata_result(case, result_path, metadata_path, ["metadata.json is missing"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    errors = validate_rmsnorm_metadata(metadata) if case["case_id"] == "OP-RMSNORM-001" else []
    if errors:
        return metadata_result(case, result_path, metadata_path, errors)

    status, metrics, reason = compare_arrays(case, golden, actual, common)
    grade = evidence_grade(metadata, status)
    write_case_result(
        result_path,
        case_id=case["case_id"],
        status=status,
        stage="compare",
        evidence_grade=grade,
        summary=f"Compared {len(common)} tensor keys from golden and actual artifacts.",
        metrics=metrics,
        evidence={"golden": str(golden_path or golden_dir), "actual": str(actual_path or actual_dir), "metadata": str(metadata_path)},
        reason=reason,
        next_step="" if status == "pass" else "Inspect compared tensors and thresholds.",
    )
    return json.loads(result_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare acceptance golden/actual .npz artifacts.")
    parser.add_argument("--case", required=True, dest="case_path")
    parser.add_argument("--result", default="", help="Optional override for result artifact path.")
    parser.add_argument("--golden", default="", help="Explicit golden .npz path.")
    parser.add_argument("--actual", default="", help="Explicit actual .npz path.")
    parser.add_argument("--metadata", default="", help="Explicit metadata.json path.")
    parser.add_argument("--key", default="", help="Explicit .npz array key to compare.")
    args = parser.parse_args()
    case = load_simple_yaml(args.case_path)
    if args.result:
        case.setdefault("result", {})["artifact"] = args.result
    result = compare_case(
        case,
        golden_path=args.golden or None,
        actual_path=args.actual or None,
        metadata_path_override=args.metadata or None,
        key=args.key or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"pass", "not_run", "xfail"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
