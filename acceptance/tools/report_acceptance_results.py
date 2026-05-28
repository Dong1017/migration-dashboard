from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Report acceptance result.json statuses.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Acceptance artifacts directory.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Operator manifest used for default filtering.")
    parser.add_argument("--all-results", action="store_true", help="Include non-operator acceptance results such as bridge contracts.")
    parser.add_argument("--status", choices=sorted(STATUSES), help="Only show results with this status.")
    parser.add_argument("--summary", action="store_true", help="Print status counts.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print reason, next step, and result path.")
    args = parser.parse_args()

    case_ids = None if args.all_results else load_operator_case_ids(Path(args.manifest))
    results = load_results(Path(args.artifacts_dir), case_ids)
    if args.status:
        results = [result for result in results if result.get("status") == args.status]

    if args.summary:
        print_summary(results)
    else:
        print_results(results, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
