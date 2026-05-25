from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from acceptance.tools.write_result import write_case_result
except ModuleNotFoundError:
    from write_result import write_case_result

CASE_ID = "OP-RMSNORM-001"
ARTIFACT_NAME = "rmsnorm_shape_2x16x128_fp32_seed0.npz"
INPUT_NAME = "input_shape_2x16x128_fp32_seed0.npz"
GAMMA_NAME = "gamma_shape_128_fp32_seed0.npz"


def parse_shape(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",")]


def normalize_output(result: Any) -> Any:
    return result[0] if isinstance(result, (tuple, list)) else result


def result_path(output_root: Path) -> Path:
    return output_root / "result.json"


def metadata_path(output_root: Path) -> Path:
    return output_root / "metadata.json"


def write_stage_result(output_root: Path, status: str, summary: str, reason: str, next_step: str = "") -> None:
    write_case_result(
        result_path(output_root),
        case_id=CASE_ID,
        status=status,
        stage="golden_collection",
        evidence_grade="tool_contract_only",
        summary=summary,
        reason=reason,
        next_step=next_step,
        evidence={"golden": str(output_root / "golden"), "actual": str(output_root / "actual"), "metadata": str(metadata_path(output_root))},
    )


def base_metadata(shape: list[int], dtype: str, seed: int, epsilon: float, torch_version: str, torch_npu_version: str) -> dict[str, Any]:
    return {
        "case_id": CASE_ID,
        "dtype": dtype,
        "shape": shape,
        "tensor_rank": len(shape),
        "layout": "contiguous",
        "seed": seed,
        "distributed": {"rank": 0, "world_size": 1, "group": "single"},
        "environment": {
            "golden": {"platform": platform.platform(), "executable": sys.executable},
            "actual": {},
        },
        "version": {
            "python": sys.version.split()[0],
            "torch": torch_version,
            "torch_npu": torch_npu_version,
            "mindspore": "",
            "numpy": np.__version__,
        },
        "epsilon": epsilon,
        "gamma_shape": [shape[-1]],
        "artifact_key": "output",
        "compared_keys": ["output"],
        "optional_keys": ["rstd"],
        "golden_source_kind": "torch_npu_runtime",
        "actual_source_kind": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect OP-RMSNORM-001 torch_npu golden artifacts.")
    parser.add_argument("--case", default="acceptance/cases/OP-RMSNORM-001.yaml")
    parser.add_argument("--shape", default="2,16,128")
    parser.add_argument("--dtype", default="fp32", choices=["fp32"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epsilon", default=1e-6, type=float)
    parser.add_argument("--output-root", default="acceptance/artifacts/OP-RMSNORM-001")
    parser.add_argument("--golden-source", default="torch_npu", choices=["torch_npu"])
    args = parser.parse_args()

    output_root = Path(args.output_root)
    shape = parse_shape(args.shape)

    try:
        import torch
    except Exception as exc:
        write_stage_result(output_root, "not_run", "torch is unavailable; golden collection did not run.", str(exc), "Install torch and torch_npu, then rerun golden collection.")
        return 0
    try:
        import torch_npu
    except Exception as exc:
        write_stage_result(output_root, "not_run", "torch_npu is unavailable; golden collection did not run.", str(exc), "Install torch_npu, then rerun golden collection.")
        return 0

    rms_norm = getattr(torch_npu, "npu_rms_norm", None)
    if rms_norm is None:
        rms_norm = getattr(getattr(torch, "ops", object()), "npu", object()).__dict__.get("npu_rms_norm") if hasattr(getattr(torch, "ops", object()), "npu") else None
    if rms_norm is None:
        write_stage_result(output_root, "blocked", "torch_npu is present but npu_rms_norm is unavailable.", "Missing torch_npu.npu_rms_norm runtime API.", "Confirm op-plugin version and API binding.")
        return 0

    torch.manual_seed(args.seed)
    tensor = torch.randn(*shape, dtype=torch.float32)
    gamma = torch.randn(shape[-1], dtype=torch.float32)
    npu_tensor = tensor.npu() if hasattr(tensor, "npu") else tensor.to("npu")
    npu_gamma = gamma.npu() if hasattr(gamma, "npu") else gamma.to("npu")
    output = normalize_output(rms_norm(npu_tensor, npu_gamma, args.epsilon)).detach().cpu().to(torch.float32).numpy()

    shared_dir = output_root / "shared"
    golden_dir = output_root / "golden"
    shared_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)
    np.savez(shared_dir / INPUT_NAME, input=tensor.cpu().numpy().astype(np.float32))
    np.savez(shared_dir / GAMMA_NAME, gamma=gamma.cpu().numpy().astype(np.float32))
    np.savez(golden_dir / ARTIFACT_NAME, output=output.astype(np.float32))

    torch_npu_version = getattr(torch_npu, "__version__", "unknown")
    metadata_path(output_root).write_text(
        json.dumps(base_metadata(shape, args.dtype, args.seed, args.epsilon, torch.__version__, torch_npu_version), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
