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


def build_data(markdown: str) -> dict:
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
        "by_priority": dict(by_priority),
        "by_repo": dict(by_repo),
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "source": "migration-analysis-v0.3.md Appendix E",
    }
    return {"metrics": metrics, "items": items, "history": {"daily": []}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="../migration-analysis-v0.3.md")
    parser.add_argument("--output", default="data.json")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    data = build_data(source.read_text(encoding="utf-8"))
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(data['items'])} items to {output}")


if __name__ == "__main__":
    main()
