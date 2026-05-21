#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VALID_STATUS = {"open", "blocked", "accepted"}
MODULE_ALIASES = {
    "vllm + omni_npu": "vllm_omni_npu",
    "vllm + omni-npu": "vllm_omni_npu",
    "vllm_omni_npu": "vllm_omni_npu",
    "omni_models": "omni_models",
    "ms_custom_ops": "ms_custom_ops",
}


def read_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_scalar_block(body: str) -> dict:
    result: dict[str, object] = {"evidence": []}
    current_list: str | None = None
    for raw in body.splitlines()[1:]:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_list:
            result.setdefault(current_list, []).append(line[4:].strip())
            continue
        current_list = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "evidence":
            result["evidence"] = []
            current_list = "evidence"
        else:
            result[key] = value
    return result


def parse_items_block(body: str) -> dict:
    module_match = re.search(r"^module:\s*(\S+)\s*$", body, re.M)
    if not module_match or "\nitems:" not in body:
        return {}
    module = module_match.group(1)
    items: list[dict] = []
    current: dict | None = None
    current_list: str | None = None
    for raw in body.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current:
                items.append(current)
            current = {"id": stripped.split(":", 1)[1].strip(), "evidence": []}
            current_list = None
            continue
        if current is None:
            continue
        if stripped.startswith("- ") and current_list:
            current.setdefault(current_list, []).append(stripped[2:].strip())
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "evidence":
            current["evidence"] = []
            current_list = "evidence"
        else:
            current[key] = value
            current_list = None
    if current:
        items.append(current)
    return {"module": module, "items": items}


def normalize_update(raw: dict) -> tuple[str, list[dict]]:
    module_raw = str(raw.get("module", "")).strip()
    module = MODULE_ALIASES.get(module_raw)
    if not module:
        raise ValueError(f"unknown module: {module_raw}")
    if raw.get("items"):
        updates = raw["items"]
    else:
        updates = [raw]
    return module, updates


def validate_author(author: str, module: str, config: dict) -> None:
    admins = set(config.get("global_admins", []))
    users = set(config["modules"][module].get("github_users", []))
    if author not in admins and author not in users:
        raise PermissionError(f"{author} cannot update {module}")


def validate_update(update: dict) -> str:
    status = str(update.get("status", "")).strip()
    if status not in VALID_STATUS:
        raise ValueError(f"invalid status: {status}")
    if status == "accepted" and not update.get("evidence"):
        raise ValueError("accepted requires at least one evidence entry")
    if status == "blocked" and not (update.get("reason") or update.get("next")):
        raise ValueError("blocked requires reason or next")
    if update.get("row"):
        if not update.get("parent"):
            raise ValueError("operator row updates require parent")
        return f"{update['parent']}#{update['row']}"
    item_id = str(update.get("id", "")).strip()
    if not item_id:
        raise ValueError("missing id")
    return item_id


def cleaned_update(update: dict, author: str) -> dict:
    allowed = ["status", "owner", "evidence", "reason", "next", "notes"]
    result = {key: update[key] for key in allowed if key in update and update[key] not in ("", [])}
    result["updated_by"] = f"@{author}"
    return result


def apply_updates(module: str, updates: list[dict], author: str, status_dir: Path) -> list[str]:
    path = status_dir / f"{module}.json"
    data = read_json(path, {"items": {}, "operator_rows": {}})
    applied = []
    for update in updates:
        target = validate_update(update)
        if "#" in target:
            data.setdefault("operator_rows", {})[target] = cleaned_update(update, author)
        else:
            data.setdefault("items", {})[target] = cleaned_update(update, author)
        applied.append(target)
    write_json(path, data)
    return applied


def event_comment(event_path: Path) -> tuple[str, str]:
    event = read_json(event_path)
    author = event.get("comment", {}).get("user", {}).get("login", "")
    body = event.get("comment", {}).get("body", "")
    return author, body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("data/status_config.json"))
    parser.add_argument("--status-dir", type=Path, default=Path("data/status"))
    parser.add_argument("--result", type=Path, default=Path("status_update_result.json"))
    args = parser.parse_args()

    author, body = event_comment(args.event)
    if not body.lstrip().startswith("/status-update"):
        print("ignored: comment does not start with /status-update")
        return
    raw = parse_items_block(body) or parse_scalar_block(body)
    config = read_json(args.config)
    module, updates = normalize_update(raw)
    validate_author(author, module, config)
    applied = apply_updates(module, updates, author, args.status_dir)
    write_json(args.result, {"module": module, "applied": applied, "author": author})
    print(f"applied {len(applied)} updates for {module}: {', '.join(applied)}")


if __name__ == "__main__":
    main()
