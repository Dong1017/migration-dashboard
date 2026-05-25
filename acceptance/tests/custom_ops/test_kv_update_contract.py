from __future__ import annotations

from pathlib import Path

from acceptance.tools.compare_outputs import compare_case, load_simple_yaml, validate_case

CASE_PATH = Path("acceptance/cases/OP-KV-UPDATE-001.yaml")
REQUIRED_SCENARIOS = {"normal_update", "duplicate_index", "empty_update"}
REQUIRED_CHECKS = {
    "cache_after",
    "alias_view_after",
    "changed_positions",
    "unchanged_positions",
    "empty_update",
    "duplicate_index_behavior_from_pt_golden",
}


def test_kv_update_case_schema_is_complete():
    case = load_simple_yaml(CASE_PATH)
    assert case["case_id"] == "OP-KV-UPDATE-001"
    assert REQUIRED_SCENARIOS <= set(case["scenarios"])
    assert REQUIRED_CHECKS <= set(case["semantic_checks"])
    assert "torch_npu.npu_scatter_nd_update_" in case["apis"]["golden"]
    validate_case(case)


def test_kv_update_missing_artifacts_write_not_run(tmp_path):
    case = load_simple_yaml(CASE_PATH)
    case["golden"]["artifact_dir"] = str(tmp_path / "missing_golden")
    case["actual"]["artifact_dir"] = str(tmp_path / "missing_actual")
    case["result"]["artifact"] = str(tmp_path / "result.json")

    result = compare_case(case)

    assert result["case_id"] == "OP-KV-UPDATE-001"
    assert result["status"] == "not_run"
    assert "Missing golden or actual" in result["reason"]
    assert Path(case["result"]["artifact"]).exists()
