from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_audit(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_gate(audit: dict[str, Any], *, mode: str) -> dict[str, Any]:
    summary = audit.get("summary") or {}
    reasons: list[str] = []
    if mode not in {"demo", "production"}:
        raise ValueError("mode must be demo or production")

    if not summary.get("claims_passed"):
        reasons.append("claims_failed")
    if not summary.get("demo_passed"):
        reasons.append("demo_failed")

    if mode == "production":
        if summary.get("pending_external_service", 0) > 0:
            reasons.append("pending_external_service")
        if summary.get("pending_evidence", 0) > 0:
            reasons.append("pending_evidence")
        if summary.get("verified", 0) != summary.get("total", 0):
            reasons.append("not_all_resume_items_verified")

    return {
        "mode": mode,
        "passed": not reasons,
        "status": audit.get("status"),
        "summary": summary,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate resume-derived acceptance results.")
    parser.add_argument("--audit", default="reports/acceptance_audit.json")
    parser.add_argument("--mode", choices=["demo", "production"], default="demo")
    args = parser.parse_args()
    result = evaluate_gate(load_audit(Path(args.audit)), mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
