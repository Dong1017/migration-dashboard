#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CHECKBOX_RE = re.compile(r"^\| \[ \] `([^`]+)` \| (P\d) \| (.*?) \|(?: (.*?) \| (\d+) \|)?$", re.M)
SECTION_RE = re.compile(r"^### E\.(\d+) (.+)$")
OP_ROW_RE = re.compile(r"^\| (\d+) \| `([^`]+)` \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|$")
CHECKBOX_ID_RE = re.compile(r"`(4\.\d+)`")
STATUS_KEYS = ["status", "owner", "evidence", "reason", "next", "notes", "updated_by"]
SOURCE_NAME = "migration-analysis-v0.5.md"
OP_GATE_MANIFEST = Path("acceptance/op_gate_cases.json")
OP_GATE_ARTIFACTS_DIR = Path("acceptance/artifacts")
ACCEPTANCE_EVIDENCE = Path("data/acceptance_evidence.json")
OP_GATE_STATUS_MAP = {
    "pass": "accepted",
    "blocked": "blocked",
    "fail": "blocked",
    "xfail": "blocked",
    "not_run": "open",
}


def classify(item_id: str, title: str) -> tuple[str, str, str]:
    if item_id.startswith("2."):
        repo = "vllm + omni_npu"
        lane = "Bridge/Engine/Serving"
    elif item_id.startswith("3."):
        repo = "omni_models"
        lane = "Pure MindSpore Model"
    else:
        repo = "ms_custom_ops"
        lane = "Custom Ops/API"

    prefixes = [
        ("2.1", "DLPack tensor conversion"),
        ("2.2", "forward_context proxy"),
        ("2.3", "KV cache adapter"),
        ("2.4", "MSBridgeWrapper"),
        ("2.5", "ACL Event & memory"),
        ("2.6", "MS comm groups"),
        ("2.7", "HCCL smoke"),
        ("2.8", "Engine regression"),
        ("2.9", "E2E & gates"),
        ("3.1", "MLA Attention"),
        ("3.2", "MoE"),
        ("3.3", "MTP"),
        ("3.4", "Embedding/LM Head"),
        ("3.5", "Weight/config"),
        ("3.6", "Parallel/distributed"),
        ("3.7", "Variants/unsupported"),
    ]
    group = next((name for prefix, name in prefixes if item_id.startswith(prefix)), title.replace("。", ""))
    return repo, lane, group


def strip_md(text: str) -> str:
    return text.replace("`", "").replace("。", "").strip()


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_status_overlays(status_dir: Path) -> dict:
    overlays = {"items": {}, "operator_rows": {}}
    if not status_dir.exists():
        return overlays
    for path in sorted(status_dir.glob("*.json")):
        data = read_json(path, {"items": {}, "operator_rows": {}})
        overlays["items"].update(data.get("items", {}))
        overlays["operator_rows"].update(data.get("operator_rows", {}))
    return overlays


def result_to_dashboard_status(status: str | None) -> str:
    return OP_GATE_STATUS_MAP.get(status or "", "open")


def load_acceptance_evidence(evidence_path: Path = ACCEPTANCE_EVIDENCE) -> dict[str, dict]:
    if not evidence_path.exists():
        return {}
    data = read_json(evidence_path, {"items": []})
    return {item["case_id"]: item for item in data.get("items", [])}


def load_op_gate_status(manifest_path: Path = OP_GATE_MANIFEST, artifacts_dir: Path = OP_GATE_ARTIFACTS_DIR) -> dict[int, dict]:
    if not manifest_path.exists():
        return {}

    evidence_by_case = load_acceptance_evidence()
    op_status = {}
    for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
        case_id = entry["case_id"]
        result_path = artifacts_dir / case_id / "result.json"
        result = evidence_by_case.get(case_id) or read_json(result_path, {"case_id": case_id})
        acceptance_status = result.get("status")
        status = {
            "case_id": case_id,
            "acceptance_status": acceptance_status or "not_run",
            "status": result_to_dashboard_status(acceptance_status),
            "evidence_grade": result.get("evidence_grade", ""),
            "stage": result.get("stage", ""),
            "reason": result.get("reason", ""),
            "next": result.get("next", ""),
            "artifact": result.get("artifact", str(result_path).replace("\\", "/")),
        }
        rows = [int(entry["row"])]
        if entry["row"] in {501, 502, 503, 504}:
            rows.append(int(entry["row"]) - 402)
        for row in rows:
            op_status[row] = status
    return op_status


def module_defaults(config_path: Path) -> dict:
    config = read_json(config_path, {"modules": {}})
    return {
        module.get("repo", key): module.get("display_owner", "")
        for key, module in config.get("modules", {}).items()
    }


def apply_status(target: dict, update: dict | None, default_owner: str = "") -> None:
    if default_owner and not target.get("owner"):
        target["owner"] = default_owner
    if not update:
        if target.get("status") == "todo":
            target["status"] = "open"
        return
    for key in STATUS_KEYS:
        if key in update:
            target[key] = update[key]


def apply_op_gate_status(target: dict, op_update: dict | None, explicit_update: dict | None) -> None:
    if not op_update:
        return

    target.update({key: value for key, value in op_update.items() if value})
    if explicit_update and "status" in explicit_update:
        return
    target["status"] = op_update["status"]


def best_row_update(overlays: dict, parents: list[str], row: str) -> dict | None:
    for parent_id in parents:
        update = overlays["operator_rows"].get(f"{parent_id}#{row}")
        if update:
            return update
    return None


def parse_operator_rows(markdown: str, overlays: dict, defaults: dict, op_gate_status: dict[int, dict]) -> list[dict]:
    rows = []
    in_ops_table = False
    for line in markdown.splitlines():
        if line.startswith("### 5.3 "):
            in_ops_table = True
            continue
        if in_ops_table and line.startswith("### 5.4 "):
            break
        if not in_ops_table:
            continue
        match = OP_ROW_RE.match(line)
        if not match:
            continue
        row, api, source_label, coverage, ms_candidate, handling = match.groups()
        row_number = int(row)
        parents = CHECKBOX_ID_RE.findall(coverage)
        if not parents:
            parents = ["4.14"]
        primary_parent = parents[0]
        child = {
            "id": f"{primary_parent}#{row}",
            "row": row_number,
            "api": api,
            "parent_id": primary_parent,
            "parent_ids": parents,
            "coverage": ", ".join(parents),
            "source_label": strip_md(source_label),
            "ms_candidate": strip_md(ms_candidate),
            "reference": strip_md(ms_candidate),
            "handling": strip_md(handling),
            "priority": "P0",
            "title": api,
            "repo": "ms_custom_ops",
            "section": "ms_custom_ops operator rows",
            "lane": "Custom Ops/API",
            "group": primary_parent,
            "status": "todo",
            "owner": "",
            "evidence": [],
            "notes": "",
            "kind": "operator_row",
        }
        row_update = best_row_update(overlays, parents, row)
        apply_status(child, row_update, defaults.get("ms_custom_ops", ""))
        apply_op_gate_status(child, op_gate_status.get(row_number), row_update)
        rows.append(child)
    return rows


def parse_checkbox_items(markdown: str, overlays: dict, defaults: dict) -> tuple[list[dict], dict[str, dict]]:
    items = []
    operator_groups = {}
    section = ""
    for line in markdown.splitlines():
        section_match = SECTION_RE.match(line)
        if section_match:
            section = section_match.group(2).strip()
            continue
        match = CHECKBOX_RE.match(line)
        if not match or not section:
            continue
        item_id, priority, title, rows, count = match.groups()
        repo, lane, group = classify(item_id, title)
        base = {
            "id": item_id,
            "priority": priority,
            "title": title.replace("。", ""),
            "repo": repo,
            "section": section,
            "lane": lane,
            "group": group,
            "status": "todo",
            "rows": rows or "",
            "item_count": int(count) if count else 1,
            "evidence": [],
            "owner": "",
            "notes": "",
            "kind": "checkbox",
        }
        if item_id.startswith("4."):
            operator_groups[item_id] = {**base, "status_counts": {}, "child_ids": []}
            continue
        apply_status(base, overlays["items"].get(item_id), defaults.get(repo, ""))
        items.append(base)
    return items, operator_groups


def attach_operator_groups(operator_rows: list[dict], operator_groups: dict[str, dict]) -> None:
    children_by_parent: dict[str, list[dict]] = defaultdict(list)
    for row in operator_rows:
        for parent_id in row["parent_ids"]:
            children_by_parent[parent_id].append(row)
    for group_id, group in operator_groups.items():
        children = sorted(children_by_parent.get(group_id, []), key=lambda child: child["row"])
        counts = Counter(child["status"] for child in children)
        group["children"] = children
        group["child_ids"] = [child["id"] for child in children]
        group["child_count"] = len(children)
        group["status_counts"] = dict(counts)
        if counts.get("blocked"):
            group["status"] = "blocked"
        elif children and counts.get("accepted") == len(children):
            group["status"] = "accepted"
        else:
            group["status"] = "open"


def read_existing_data(path: Path = Path("data.json")) -> dict:
    return read_json(path, {"items": [], "metrics": {}})


def preserve_non_operator_status(items: list[dict], existing_data: dict) -> None:
    existing_items = {item["id"]: item for item in existing_data.get("items", [])}
    for item in items:
        if item.get("repo") == "ms_custom_ops":
            continue
        existing = existing_items.get(item["id"])
        if not existing:
            continue
        for key in STATUS_KEYS:
            if key in existing:
                item[key] = existing[key]


def repo_details(items: list[dict]) -> dict[str, dict]:
    details = {}
    for repo in sorted({item["repo"] for item in items}):
        repo_items = [item for item in items if item["repo"] == repo]
        total = len(repo_items)
        accepted = sum(item["status"] == "accepted" for item in repo_items)
        blocked = sum(item["status"] in {"blocked", "xfail"} for item in repo_items)
        open_count = sum(item["status"] == "open" for item in repo_items)
        details[repo] = {
            "total": total,
            "accepted": accepted,
            "blocked": blocked,
            "open": open_count,
            "completion_rate": round(accepted * 100 / total, 1) if total else 0,
        }
    return details


def build_data(markdown: str, status_dir: Path = Path("data/status"), config_path: Path = Path("data/status_config.json")) -> dict:
    overlays = load_status_overlays(status_dir)
    defaults = module_defaults(config_path)
    existing_data = read_existing_data()
    op_gate_status = load_op_gate_status()
    operator_rows = parse_operator_rows(markdown, overlays, defaults, op_gate_status)
    checkbox_items, operator_groups = parse_checkbox_items(markdown, overlays, defaults)
    items = checkbox_items + operator_rows
    preserve_non_operator_status(items, existing_data)
    attach_operator_groups(operator_rows, operator_groups)

    status_counts = Counter(item["status"] for item in items)
    by_priority = Counter(item["priority"] for item in items)
    by_repo = Counter(item["repo"] for item in items)
    total = len(items)
    accepted = status_counts.get("accepted", 0)
    metrics = {
        "total": total,
        "accepted": accepted,
        "open": status_counts.get("open", 0),
        "blocked": status_counts.get("blocked", 0),
        "completion_rate": round(accepted * 100 / total, 1) if total else 0,
        "operator_rows": len(operator_rows),
        "operator_groups": len(operator_groups),
        "by_priority": dict(by_priority),
        "by_repo": dict(by_repo),
        "by_repo_detail": repo_details(items),
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "source": f"{SOURCE_NAME} Appendix E + Section 5.3",
    }
    return {
        "metrics": metrics,
        "items": items,
        "operator_rows": operator_rows,
        "operator_groups": operator_groups,
        "history": {"daily": []},
    }


def overlay_summary(status_dir: Path) -> tuple[int, list[str]]:
    overlay_count = 0
    blocked_items = []
    if not status_dir.exists():
        return overlay_count, blocked_items
    for path in sorted(status_dir.glob("*.json")):
        data = read_json(path, {"items": {}, "operator_rows": {}})
        for section in ("items", "operator_rows"):
            for item_id, update in data.get(section, {}).items():
                overlay_count += 1
                if update.get("status") == "blocked":
                    blocked_items.append(item_id)
    return overlay_count, blocked_items


def print_summary(data: dict, status_dir: Path) -> None:
    metrics = data["metrics"]
    status_counts = Counter(item["status"] for item in data["items"])
    overlay_count, blocked_items = overlay_summary(status_dir)
    print(f"total={metrics['total']}")
    print(f"by_repo={metrics['by_repo']}")
    print(f"by_priority={metrics['by_priority']}")
    print(f"by_status={dict(status_counts)}")
    print(f"operator_rows={metrics['operator_rows']}")
    print(f"operator_groups={metrics['operator_groups']}")
    print(f"source={metrics['source']}")
    print(f"overlay_count={overlay_count}")
    print(f"blocked_items={blocked_items}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE_NAME)
    parser.add_argument("--output", default="data.json")
    parser.add_argument("--status-dir", default="data/status")
    parser.add_argument("--config", default="data/status_config.json")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    data = build_data(source.read_text(encoding="utf-8"), Path(args.status_dir), Path(args.config))
    with output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {len(data['items'])} items and {len(data['operator_rows'])} operator rows to {output}")
    print_summary(data, Path(args.status_dir))


if __name__ == "__main__":
    main()
