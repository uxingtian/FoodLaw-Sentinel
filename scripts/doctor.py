from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from scripts.acceptance_gate import evaluate_gate, load_audit


REQUIRED_REPORTS = [
    "eval.json",
    "benchmark.json",
    "readiness.json",
    "demo_scenarios.json",
    "demo_report.md",
    "claim_verification.json",
    "resume_summary.md",
    "acceptance_audit.json",
    "acceptance_audit.md",
]


def build_doctor_report(*, reports_dir: Path | None = None) -> dict[str, Any]:
    reports_dir = reports_dir or settings.reports_dir
    audit_path = reports_dir / "acceptance_audit.json"
    gates = {
        "demo": {"passed": False, "reasons": ["missing_acceptance_audit"]},
        "production": {"passed": False, "reasons": ["missing_acceptance_audit"]},
    }
    if audit_path.exists():
        audit = load_audit(audit_path)
        gates = {
            "demo": evaluate_gate(audit, mode="demo"),
            "production": evaluate_gate(audit, mode="production"),
        }
    return {
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "config": {
            "data_dir": str(settings.data_dir),
            "reports_dir": str(reports_dir),
            "model_configured": settings.model_configured,
            "vector_backend": settings.vector_backend,
            "embedding_provider": settings.embedding_provider,
            "reranker_provider": settings.reranker_provider,
            "workflow_backend": settings.workflow_backend,
        },
        "reports": {name: _report_file_status(reports_dir / name) for name in REQUIRED_REPORTS},
        "gates": gates,
    }


def _report_file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "size_bytes": 0}
    return {"available": True, "size_bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local runtime, configuration, evidence reports, and acceptance gates.")
    parser.add_argument("--reports-dir", default=str(settings.reports_dir))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = build_doctor_report(reports_dir=Path(args.reports_dir))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
