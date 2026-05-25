from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_STATUSES = {"pass", "fail", "blocked", "not_run", "xfail"}


def build_result(
    case_id: str,
    status: str,
    summary: str = "",
    metrics: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    reason: str = "",
    next_step: str = "",
    stage: str = "",
    evidence_grade: str = "",
) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported status {status!r}; expected one of {sorted(ALLOWED_STATUSES)}")
    default_metrics = {
        "finite_check": "",
        "max_abs_error": None,
        "max_rel_error": None,
        "pass_ratio": None,
    }
    default_evidence = {
        "golden": "",
        "actual": "",
        "metadata": "",
    }
    if metrics:
        default_metrics.update(metrics)
    if evidence:
        default_evidence.update(evidence)
    return {
        "case_id": case_id,
        "status": status,
        "stage": stage,
        "evidence_grade": evidence_grade,
        "summary": summary,
        "metrics": default_metrics,
        "evidence": default_evidence,
        "reason": reason,
        "next": next_step,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_result(path: str | Path, result: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def write_case_result(
    path: str | Path,
    case_id: str,
    status: str,
    summary: str = "",
    metrics: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    reason: str = "",
    next_step: str = "",
    stage: str = "",
    evidence_grade: str = "",
) -> Path:
    return write_result(
        path,
        build_result(
            case_id=case_id,
            status=status,
            summary=summary,
            metrics=metrics,
            evidence=evidence,
            reason=reason,
            next_step=next_step,
            stage=stage,
            evidence_grade=evidence_grade,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a normalized acceptance result JSON.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--status", required=True, choices=sorted(ALLOWED_STATUSES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--next", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--evidence-grade", default="")
    args = parser.parse_args()
    write_case_result(
        args.output,
        case_id=args.case_id,
        status=args.status,
        summary=args.summary,
        reason=args.reason,
        next_step=args.next,
        stage=args.stage,
        evidence_grade=args.evidence_grade,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
