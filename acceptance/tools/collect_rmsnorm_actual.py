from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from acceptance.tools.collect_rmsnorm_golden import ARTIFACT_NAME, GAMMA_NAME, INPUT_NAME, CASE_ID, normalize_output
    from acceptance.tools.write_result import write_case_result
except ModuleNotFoundError:
    from collect_rmsnorm_golden import ARTIFACT_NAME, GAMMA_NAME, INPUT_NAME, CASE_ID, normalize_output
    from write_result import write_case_result


def result_path(output_root: Path) -> Path:
    return output_root / "result.json"


def metadata_path(output_root: Path) -> Path:
    return output_root / "metadata.json"


def write_stage_result(output_root: Path, status: str, summary: str, reason: str, next_step: str = "") -> None:
    write_case_result(
        result_path(output_root),
        case_id=CASE_ID,
        status=status,
        stage="actual_collection",
        evidence_grade="tool_contract_only",
        summary=summary,
        reason=reason,
        next_step=next_step,
        evidence={"golden": str(output_root / "golden"), "actual": str(output_root / "actual"), "metadata": str(metadata_path(output_root))},
    )


def load_metadata(output_root: Path) -> dict[str, Any]:
    path = metadata_path(output_root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_metadata(output_root: Path, metadata: dict[str, Any]) -> None:
    metadata_path(output_root).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect OP-RMSNORM-001 MindSpore actual artifacts.")
    parser.add_argument("--case", default="acceptance/cases/OP-RMSNORM-001.yaml")
    parser.add_argument("--output-root", default="acceptance/artifacts/OP-RMSNORM-001")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    shared_dir = output_root / "shared"
    input_path = shared_dir / INPUT_NAME
    gamma_path = shared_dir / GAMMA_NAME
    if not input_path.exists() or not gamma_path.exists():
        write_stage_result(output_root, "not_run", "Shared input/gamma artifacts are missing; actual collection did not run.", "Run golden collection first to create shared input/gamma artifacts.", "Run collect_rmsnorm_golden.py, then rerun actual collection.")
        return 0

    try:
        import mindspore as ms
        from mindspore import Tensor, ops
    except Exception as exc:
        write_stage_result(output_root, "not_run", "MindSpore is unavailable; actual collection did not run.", str(exc), "Install MindSpore runtime, then rerun actual collection.")
        return 0

    rms_norm = getattr(ops, "rms_norm", None)
    if rms_norm is None:
        write_stage_result(output_root, "blocked", "MindSpore is present but ops.rms_norm is unavailable.", "Missing mindspore.ops.rms_norm runtime API.", "Confirm MindSpore version and API availability.")
        return 0

    input_np = np.load(input_path)["input"].astype(np.float32)
    gamma_np = np.load(gamma_path)["gamma"].astype(np.float32)
    metadata = load_metadata(output_root)
    epsilon = float(metadata.get("epsilon", 1e-6))
    output = normalize_output(rms_norm(Tensor(input_np, ms.float32), Tensor(gamma_np, ms.float32), epsilon)).asnumpy().astype(np.float32)

    actual_dir = output_root / "actual"
    actual_dir.mkdir(parents=True, exist_ok=True)
    np.savez(actual_dir / ARTIFACT_NAME, output=output)

    metadata.setdefault("environment", {}).setdefault("golden", {})
    metadata["environment"]["actual"] = {"platform": platform.platform(), "executable": sys.executable}
    metadata.setdefault("version", {})["python"] = metadata.get("version", {}).get("python") or sys.version.split()[0]
    metadata["version"]["mindspore"] = getattr(ms, "__version__", "unknown")
    metadata["version"]["numpy"] = np.__version__
    metadata["actual_source_kind"] = "mindspore_runtime"
    save_metadata(output_root, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
