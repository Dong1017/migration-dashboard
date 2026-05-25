from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from acceptance.tools.collect_rmsnorm_golden import normalize_output, write_stage_result as write_golden_stage_result
from acceptance.tools.collect_rmsnorm_actual import write_stage_result as write_actual_stage_result
from acceptance.tools.compare_outputs import compare_case, load_simple_yaml, validate_case, validate_rmsnorm_metadata

CASE_PATH = Path("acceptance/cases/OP-RMSNORM-001.yaml")
REAL_RESULT_PATH = Path("acceptance/artifacts/OP-RMSNORM-001/result.json")
REQUIRED_FIELDS = {
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


def valid_metadata() -> dict:
    return {
        "case_id": "OP-RMSNORM-001",
        "dtype": "fp32",
        "shape": [2, 16, 128],
        "tensor_rank": 3,
        "layout": "contiguous",
        "seed": 0,
        "distributed": {"rank": 0, "world_size": 1, "group": "single"},
        "environment": {"golden": {}, "actual": {}},
        "version": {"python": "test", "torch": "test", "torch_npu": "test", "mindspore": "test", "numpy": "test"},
        "epsilon": 1e-6,
        "gamma_shape": [128],
        "artifact_key": "output",
        "compared_keys": ["output"],
        "optional_keys": ["rstd"],
        "golden_source_kind": "torch_npu_runtime",
        "actual_source_kind": "mindspore_runtime",
    }


def test_rmsnorm_case_schema_is_complete():
    case = load_simple_yaml(CASE_PATH)
    assert REQUIRED_FIELDS <= set(case)
    assert case["case_id"] == "OP-RMSNORM-001"
    assert {entry["dtype"] for entry in case["inputs"]} == {"fp32", "bf16"}
    assert [1, 1, 128] in [entry["shape"] for entry in case["inputs"]]
    validate_case(case)


def test_rmsnorm_missing_artifacts_write_not_run(tmp_path):
    case = load_simple_yaml(CASE_PATH)
    case["golden"]["artifact_dir"] = str(tmp_path / "missing_golden")
    case["actual"]["artifact_dir"] = str(tmp_path / "missing_actual")
    case["result"]["artifact"] = str(tmp_path / "result.json")

    result = compare_case(case)

    assert result["case_id"] == "OP-RMSNORM-001"
    assert result["status"] == "not_run"
    assert result["stage"] == "compare"
    assert result["evidence_grade"] == "tool_contract_only"
    assert result["metrics"]["max_abs_error"] is None
    assert Path(case["result"]["artifact"]).exists()


def test_collector_stage_results_never_write_pass(tmp_path):
    output_root = tmp_path / "case"

    write_golden_stage_result(output_root, "not_run", "torch_npu missing", "import failed")
    golden_result = json.loads((output_root / "result.json").read_text(encoding="utf-8"))
    assert golden_result["stage"] == "golden_collection"
    assert golden_result["status"] != "pass"

    write_actual_stage_result(output_root, "not_run", "mindspore missing", "import failed")
    actual_result = json.loads((output_root / "result.json").read_text(encoding="utf-8"))
    assert actual_result["stage"] == "actual_collection"
    assert actual_result["status"] != "pass"


def test_metadata_missing_prevents_pass(tmp_path):
    case = load_simple_yaml(CASE_PATH)
    golden = tmp_path / "golden.npz"
    actual = tmp_path / "actual.npz"
    np.savez(golden, output=np.ones((2, 16, 128), dtype=np.float32))
    np.savez(actual, output=np.ones((2, 16, 128), dtype=np.float32))
    case["result"]["artifact"] = str(tmp_path / "result.json")

    result = compare_case(case, golden_path=golden, actual_path=actual, metadata_path_override=tmp_path / "missing_metadata.json", key="output")

    assert result["status"] == "blocked"
    assert result["stage"] == "compare"
    assert result["evidence_grade"] == "tool_contract_only"


def test_rank_three_metadata_is_invalid():
    metadata = valid_metadata()
    metadata.pop("tensor_rank")
    metadata["rank"] = 3

    errors = validate_rmsnorm_metadata(metadata)

    assert any("rank=3" in error for error in errors)


def test_valid_metadata_requires_tensor_rank_and_distributed_rank():
    metadata = valid_metadata()

    assert validate_rmsnorm_metadata(metadata) == []

    metadata["distributed"]["rank"] = 3
    assert any("distributed.rank" in error for error in validate_rmsnorm_metadata(metadata))


def test_fake_npz_pass_uses_tmp_path_only(tmp_path):
    case = load_simple_yaml(CASE_PATH)
    golden = tmp_path / "golden.npz"
    actual = tmp_path / "actual.npz"
    metadata = tmp_path / "metadata.json"
    result_path = tmp_path / "result.json"
    np.savez(golden, output=np.ones((2, 16, 128), dtype=np.float32))
    np.savez(actual, output=np.ones((2, 16, 128), dtype=np.float32))
    metadata.write_text(json.dumps(valid_metadata()), encoding="utf-8")
    case["result"]["artifact"] = str(result_path)

    result = compare_case(case, golden_path=golden, actual_path=actual, metadata_path_override=metadata, key="output")

    assert result["status"] == "pass"
    assert result["stage"] == "compare"
    assert result["evidence_grade"] == "migration_evidence"
    assert result_path.exists()
    assert not REAL_RESULT_PATH.exists() or REAL_RESULT_PATH != result_path


def test_tuple_return_normalizes_to_output():
    assert normalize_output(("output", "rstd")) == "output"
    assert normalize_output(["output", "rstd"]) == "output"
    assert normalize_output("output") == "output"
