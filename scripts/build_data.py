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


def parse_operator_rows(markdown: str) -> list[dict]:
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
        rows.append({
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
        })
    return rows


def build_data(markdown: str) -> dict:
    operator_rows = parse_operator_rows(markdown)
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
        items.append({
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
        })

    by_priority = Counter(item["priority"] for item in items)
    by_repo = Counter(item["repo"] for item in items)
    metrics = {
        "total": len(items),
        "done": 0,
        "in_progress": 0,
        "todo": len(items),
        "blocked": 0,
        "completion_rate": 0,
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
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    data = build_data(source.read_text(encoding="utf-8"))
    with output.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {len(data['items'])} items and {len(data['operator_rows'])} operator rows to {output}")


if __name__ == "__main__":
    main()
