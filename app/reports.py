from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.alignment import build_resume_alignment


REPORT_FILES = {
    "eval": "eval.json",
    "benchmark": "benchmark.json",
    "readiness": "readiness.json",
    "demo": "demo_scenarios.json",
    "claims": "claim_verification.json",
}

REPORT_ARTIFACTS = {
    "eval": ("eval.json", "application/json"),
    "benchmark": ("benchmark.json", "application/json"),
    "readiness": ("readiness.json", "application/json"),
    "demo_scenarios": ("demo_scenarios.json", "application/json"),
    "claim_verification": ("claim_verification.json", "application/json"),
    "resume_summary": ("resume_summary.md", "text/markdown; charset=utf-8"),
    "demo_report": ("demo_report.md", "text/markdown; charset=utf-8"),
    "acceptance_audit": ("acceptance_audit.md", "text/markdown; charset=utf-8"),
    "acceptance_audit_json": ("acceptance_audit.json", "application/json"),
    "doctor": ("doctor.json", "application/json"),
}


def load_report_summary(reports_dir: Path) -> dict[str, Any]:
    reports = {name: _read_report(reports_dir / filename) for name, filename in REPORT_FILES.items()}
    return {
        "available": any(report["available"] for report in reports.values()),
        "reports": reports,
        "summary": {
            "eval": _summarize_eval(reports["eval"].get("data") or {}),
            "benchmark": _summarize_benchmark(reports["benchmark"].get("data") or {}),
            "readiness": _summarize_readiness(
                (reports["claims"].get("data") or {}).get("readiness") or reports["readiness"].get("data") or {}
            ),
            "demo": _summarize_demo(reports["demo"].get("data") or {}),
            "claims": _summarize_claims(reports["claims"].get("data") or {}),
        },
        "alignment": build_resume_alignment(_claim_report_with_benchmark(reports)),
        "resume_summary": {
            "available": (reports_dir / "resume_summary.md").exists(),
            "path": str(reports_dir / "resume_summary.md"),
        },
        "demo_report": {
            "available": (reports_dir / "demo_report.md").exists(),
            "path": str(reports_dir / "demo_report.md"),
        },
        "acceptance_audit": {
            "available": (reports_dir / "acceptance_audit.md").exists(),
            "path": str(reports_dir / "acceptance_audit.md"),
        },
        "doctor": {
            "available": (reports_dir / "doctor.json").exists(),
            "path": str(reports_dir / "doctor.json"),
        },
    }


def _read_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "filename": path.name, "data": None}
    try:
        return {
            "available": True,
            "filename": path.name,
            "updated_at": path.stat().st_mtime,
            "data": json.loads(path.read_text(encoding="utf-8")),
        }
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "filename": path.name, "error": str(exc), "data": None}


def resolve_report_artifact(reports_dir: Path, artifact_name: str) -> tuple[Path, str] | None:
    artifact = REPORT_ARTIFACTS.get(artifact_name)
    if artifact is None:
        return None
    filename, media_type = artifact
    path = reports_dir / filename
    if not path.exists() or not path.is_file():
        return None
    return path, media_type


def _summarize_eval(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "cases": report.get("cases", 0),
        "role_accuracy": _round_metric(report.get("role_accuracy")),
        "answer_keyword_recall": _round_metric(report.get("answer_keyword_recall")),
        "source_keyword_recall": _round_metric(report.get("source_keyword_recall")),
        "avg_latency_ms": _round_metric(report.get("avg_latency_ms")),
        "tool_usage": report.get("tool_usage", {}),
        "graph_edges": (report.get("graph") or {}).get("edges", 0),
    }


def _summarize_benchmark(report: dict[str, Any]) -> dict[str, Any]:
    latency = report.get("latency_ms") or {}
    return {
        "requests": report.get("requests", 0),
        "concurrency": report.get("concurrency", 0),
        "success_rate": _round_metric(report.get("success_rate")),
        "throughput_qps": _round_metric(report.get("throughput_qps")),
        "p95_ms": _round_metric(latency.get("p95")),
        "max_ms": _round_metric(latency.get("max")),
    }


def _summarize_readiness(report: dict[str, Any]) -> dict[str, Any]:
    checks = report.get("checks") or []
    return {
        "production_ready": bool(report.get("production_ready")),
        "checks": [{"name": item.get("name"), "status": item.get("status"), "detail": item.get("detail")} for item in checks],
    }


def _summarize_claims(report: dict[str, Any]) -> dict[str, Any]:
    supported = report.get("supported_claims") or []
    unsupported = report.get("unsupported_claims") or []
    checks = report.get("checks") or []
    return {
        "passed": bool(report.get("passed")),
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "supported_claims": supported[:5],
        "unsupported_claims": unsupported[:5],
        "checks": [
            {
                "name": item.get("name"),
                "passed": bool(item.get("passed")),
                "observed": item.get("observed"),
                "threshold": item.get("threshold"),
            }
            for item in checks
        ],
    }


def _summarize_demo(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") or {}
    return {
        "passed": bool(summary.get("passed")),
        "total": summary.get("total", 0),
        "role_passed": summary.get("role_passed", 0),
        "with_sources": summary.get("with_sources", 0),
        "with_generation_guard": summary.get("with_generation_guard", 0),
    }


def _claim_report_with_benchmark(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    claim_report = dict(reports["claims"].get("data") or {})
    claim_report["benchmark"] = reports["benchmark"].get("data") or {}
    return claim_report


def _round_metric(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value
