#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path

CHECKBOX_RE = re.compile(r"^\| \[ \] `([^`]+)` \| (P\d) \| (.*?) \|(?: (.*?) \| (\d+) \|)?$", re.M)
SECTION_RE = re.compile(r"^### E\.(\d+) (.+)$")
OP_ROW_RE = re.compile(r"^\| (\d+) \| `([^`]+)` \| (.*?) \| (.*?) \| (.*?) \|$")
CHECKBOX_ID_RE = re.compile(r"`(4\.\d+)`")
STATUS_KEYS = ["status", "owner", "evidence", "reason", "next", "notes", "updated_by"]


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


def parse_operator_rows(markdown: str, overlays: dict, defaults: dict) -> list[dict]:
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
        row, api, coverage, reference, handling = match.groups()
        parents = CHECKBOX_ID_RE.findall(coverage)
        child = {
            "row": int(row),
            "api": api,
            "parent_ids": parents,
            "coverage": ", ".join(parents),
            "reference": strip_md(reference),
            "handling": strip_md(handling),
            "status": "todo",
            "owner": "",
            "evidence": [],
            "notes": "",
        }
        for parent_id in parents:
            apply_status(child, overlays["operator_rows"].get(f"{parent_id}#{row}"), defaults.get("ms_custom_ops", ""))
        rows.append(child)
    return rows


def build_data(markdown: str, status_dir: Path = Path("data/status"), config_path: Path = Path("data/status_config.json")) -> dict:
    overlays = load_status_overlays(status_dir)
    defaults = module_defaults(config_path)
    operator_rows = parse_operator_rows(markdown, overlays, defaults)
    children_by_parent: dict[str, list[dict]] = {}
    for row in operator_rows:
        for parent_id in row["parent_ids"]:
            children_by_parent.setdefault(parent_id, []).append(row)

    items = []
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
        children = sorted(children_by_parent.get(item_id, []), key=lambda child: child["row"])
        item = {
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
            "children": children,
            "child_count": len(children),
            "evidence": [],
            "owner": "",
            "notes": "",
        }
        apply_status(item, overlays["items"].get(item_id), defaults.get(repo, ""))
        items.append(item)

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
        "by_priority": dict(by_priority),
        "by_repo": dict(by_repo),
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "source": "migration-analysis-v0.3.md Appendix E + Section 5.3",
    }
    return {"metrics": metrics, "items": items, "operator_rows": operator_rows, "history": {"daily": []}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="../migration-analysis-v0.3.md")
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


if __name__ == "__main__":
    main()
