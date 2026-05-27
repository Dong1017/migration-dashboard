#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required for this self-check: {exc}")

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "acceptance" / "cases" / "vllm_omni_npu"
EXPECTED = json.loads((ROOT / "acceptance" / "manifests" / "vllm_omni_npu_expected_ids.json").read_text(encoding="utf-8"))
INDEX = json.loads((ROOT / "acceptance" / "manifests" / "vllm_omni_npu_78_case_index.json").read_text(encoding="utf-8"))

case_files = sorted(CASE_DIR.glob("*.yaml"))
if len(case_files) != 78:
    raise SystemExit(f"FAIL: expected 78 case files, got {len(case_files)}")

ids = []
errors = []
for path in case_files:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    did = data.get("dashboard_item_id")
    ids.append(did)
    for field in ["case_id", "dashboard_item_id", "title", "repo_target", "priority", "watch_type", "suggested_test", "required_evidence", "evidence_storage"]:
        if field not in data:
            errors.append(f"{path.name}: missing {field}")
    if data.get("repo_target") != "vllm + omni_npu":
        errors.append(f"{path.name}: repo_target must be vllm + omni_npu")
    if not str(did).startswith("2."):
        errors.append(f"{path.name}: dashboard_item_id must start with 2.x")
    if data.get("placeholder") and not data.get("material_status"):
        errors.append(f"{path.name}: placeholder case must have material_status")

if sorted(ids) != sorted(EXPECTED):
    missing = sorted(set(EXPECTED) - set(ids))
    extra = sorted(set(ids) - set(EXPECTED))
    dup = sorted({x for x in ids if ids.count(x) > 1})
    raise SystemExit(f"FAIL: coverage mismatch missing={missing} extra={extra} duplicate={dup}")

if len(INDEX.get("cases", [])) != 78 or INDEX.get("total_cases") != 78:
    errors.append("manifest total_cases/index cases must be 78")

if errors:
    raise SystemExit("FAIL:\n" + "\n".join(errors))

print("PASS: 78 vllm + omni_npu case files cover all 78 dashboard checkboxes exactly once.")
