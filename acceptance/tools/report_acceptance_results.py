from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = ROOT / "acceptance" / "artifacts"
DEFAULT_MANIFEST = ROOT / "acceptance" / "op_gate_cases.json"
STATUSES = {"pass", "fail", "blocked", "not_run", "xfail"}


def load_operator_case_ids(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {entry["case_id"] for entry in manifest}


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_dashboard_rows(data_path: Path) -> dict[str, dict[str, Any]]:
    if not data_path.exists():
        return {}
    data = json.loads(data_path.read_text(encoding="utf-8"))
    return {
        item["case_id"]: item
        for item in data.get("operator_rows", [])
        if item.get("case_id")
    }


def load_evidence(evidence_path: Path) -> dict[str, dict[str, Any]]:
    if not evidence_path.exists():
        return {}
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    return {item["case_id"]: item for item in data.get("items", [])}


def load_results(artifacts_dir: Path, case_ids: set[str] | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if case_ids is not None and result.get("case_id") not in case_ids:
            continue
        result["_path"] = str(path)
        results.append(result)
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    counts = Counter(result.get("status", "unknown") for result in results)
    print(f"total: {len(results)}")
    for status in sorted(counts):
        print(f"{status}: {counts[status]}")


def print_results(results: list[dict[str, Any]], verbose: bool) -> None:
    for result in results:
        print(f"{result.get('case_id', '')}\t{result.get('status', '')}\t{result.get('stage', '')}\t{result.get('evidence_grade', '')}")
        if verbose:
            reason = result.get("reason", "")
            next_step = result.get("next", "")
            if reason:
                print(f"  reason: {reason}")
            if next_step:
                print(f"  next: {next_step}")
            print(f"  path: {result['_path']}")


def build_audit_rows(manifest_path: Path, artifacts_dir: Path, data_path: Path, evidence_path: Path) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    dashboard_rows = load_dashboard_rows(data_path)
    evidence = load_evidence(evidence_path)
    results = {result["case_id"]: result for result in load_results(artifacts_dir)}
    rows = []
    for entry in sorted(manifest, key=lambda item: int(item["row"])):
        case_id = entry["case_id"]
        result = evidence.get(case_id) or results.get(case_id) or {}
        dashboard = dashboard_rows.get(case_id, {})
        rows.append({
            "row": entry.get("row", ""),
            "api": entry.get("torch_api", ""),
            "case_id": case_id,
            "dashboard_status": dashboard.get("status", ""),
            "acceptance_status": result.get("status", "not_run"),
            "evidence_grade": result.get("evidence_grade", ""),
            "stage": result.get("stage", ""),
            "reason": result.get("reason", ""),
            "next": result.get("next", ""),
            "artifact": result.get("artifact", str(artifacts_dir / case_id / "result.json")),
            "reference_status": entry.get("reference_status", ""),
            "generation_mode": entry.get("generation_mode", ""),
            "operator_dir": entry.get("operator_dir", ""),
        })
    return rows


def write_audit_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report acceptance result.json statuses.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Acceptance artifacts directory.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Operator manifest used for default filtering.")
    parser.add_argument("--all-results", action="store_true", help="Include non-operator acceptance results such as bridge contracts.")
    parser.add_argument("--status", choices=sorted(STATUSES), help="Only show results with this status.")
    parser.add_argument("--summary", action="store_true", help="Print status counts.")
    parser.add_argument("--audit-csv", help="Write one CSV row per manifest case joined with dashboard and evidence status.")
    parser.add_argument("--data", default=str(ROOT / "data.json"), help="Dashboard data.json used for audit joins.")
    parser.add_argument("--evidence", default=str(ROOT / "data" / "acceptance_evidence.json"), help="Acceptance evidence JSON used for audit joins.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print reason, next step, and result path.")
    args = parser.parse_args()

    case_ids = None if args.all_results else load_operator_case_ids(Path(args.manifest))
    results = load_results(Path(args.artifacts_dir), case_ids)
    if args.status:
        results = [result for result in results if result.get("status") == args.status]

    if args.audit_csv:
        rows = build_audit_rows(Path(args.manifest), Path(args.artifacts_dir), Path(args.data), Path(args.evidence))
        if args.status:
            rows = [row for row in rows if row.get("acceptance_status") == args.status]
        write_audit_csv(rows, Path(args.audit_csv))
        print(f"wrote {len(rows)} audit rows to {args.audit_csv}")
    elif args.summary:
        print_summary(results)
    else:
        print_results(results, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
