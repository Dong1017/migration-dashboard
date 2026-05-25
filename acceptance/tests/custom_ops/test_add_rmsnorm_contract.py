from __future__ import annotations

from pathlib import Path

from acceptance.tools.compare_outputs import compare_case, load_simple_yaml, validate_case

CASE_PATH = Path("acceptance/cases/OP-ADD-RMSNORM-001.yaml")
REQUIRED_ACTUAL_CANDIDATES = {
    "mindspore.ops.custom.add_rms_norm",
    "ms_custom_ops.add_rms_norm",
    "project-specific custom op wrapper",
}


def test_add_rmsnorm_case_schema_is_complete():
    case = load_simple_yaml(CASE_PATH)

    assert case["case_id"] == "OP-ADD-RMSNORM-001"
    assert case["title"] == "Add RMSNorm custom op equivalence evidence contract"
    assert case["repo_target"] == "ms_custom_ops"
    assert case["watch_type"] == "custom_op_acceptance"
    assert case["custom_op_required"] is True
    assert "torch_npu.npu_add_rms_norm" in case["apis"]["golden"]
    assert REQUIRED_ACTUAL_CANDIDATES <= set(case["apis"]["actual"])
    assert "blocked" in case["actual"]["command"].lower()
    validate_case(case)


def test_add_rmsnorm_missing_artifacts_write_not_run(tmp_path):
    case = load_simple_yaml(CASE_PATH)
    case["golden"]["artifact_dir"] = str(tmp_path / "missing_golden")
    case["actual"]["artifact_dir"] = str(tmp_path / "missing_actual")
    case["result"]["artifact"] = str(tmp_path / "result.json")

    result = compare_case(case)

    assert result["case_id"] == "OP-ADD-RMSNORM-001"
    assert result["status"] == "not_run"
    assert result["stage"] == "compare"
    assert result["evidence_grade"] == "tool_contract_only"
    assert Path(case["result"]["artifact"]).exists()
