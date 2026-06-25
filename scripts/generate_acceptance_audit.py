from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.reports import load_report_summary


NEXT_STEPS = {
    "qwen_vllm_inference": {
        "env": "OPENAI_API_KEY=...; OPENAI_BASE_URL=http://<vllm-host>:8000/v1; QA_MODEL=Qwen3-72B-Chat",
        "verify": "python scripts/check_readiness.py --output reports/readiness.json && python scripts/run_evidence_pipeline.py --reports-dir reports",
    },
    "chroma_langgraph_stack": {
        "env": "VECTOR_BACKEND=chroma; WORKFLOW_BACKEND=langgraph; install requirements-prod.txt",
        "verify": "python scripts/check_readiness.py --output reports/readiness.json",
    },
    "reranker_service": {
        "env": "RERANKER_PROVIDER=http; RERANKER_URL=http://<reranker-host>/rerank; RERANKER_MODEL=bge-reranker",
        "verify": "python scripts/check_readiness.py --output reports/readiness.json && python scripts/run_evidence_pipeline.py --reports-dir reports",
    },
    "high_concurrency_claim": {
        "env": "Use the same production service URLs and run from a machine with enough file descriptors and CPU/network headroom.",
        "verify": "python scripts/benchmark.py --base-url http://<host>:8000 --requests 5000 --concurrency 1000 --output reports/benchmark_1000.json",
    },
}


def build_acceptance_audit(report_summary: dict[str, Any]) -> dict[str, Any]:
    alignment = report_summary.get("alignment") or {}
    alignment_summary = alignment.get("summary") or {}
    items = alignment.get("items") or []
    pending = [_with_next_step(item) for item in items if item.get("status") != "verified"]
    verified = alignment_summary.get("verified", 0)
    total = alignment_summary.get("total", 0)
    status = "verified" if total > 0 and verified == total else "partially_verified"
    if not report_summary.get("available"):
        status = "missing_reports"
    return {
        "status": status,
        "summary": {
            "total": total,
            "verified": verified,
            "pending_external_service": alignment_summary.get("pending_external_service", 0),
            "pending_evidence": alignment_summary.get("pending_evidence", 0),
            "claims_passed": (report_summary.get("summary") or {}).get("claims", {}).get("passed", False),
            "demo_passed": (report_summary.get("summary") or {}).get("demo", {}).get("passed", False),
        },
        "verified_items": [item for item in items if item.get("status") == "verified"],
        "pending_items": pending,
        "artifacts": {
            "demo_report": report_summary.get("demo_report", {}),
            "resume_summary": report_summary.get("resume_summary", {}),
        },
    }


def generate_acceptance_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary") or {}
    lines = [
        "# 食品安全法律法规问答系统验收审计",
        "",
        f"- status: {audit.get('status')}",
        f"- verified: {summary.get('verified', 0)}/{summary.get('total', 0)}",
        f"- pending_external_service: {summary.get('pending_external_service', 0)}",
        f"- pending_evidence: {summary.get('pending_evidence', 0)}",
        f"- claims_passed: {str(bool(summary.get('claims_passed'))).lower()}",
        f"- demo_passed: {str(bool(summary.get('demo_passed'))).lower()}",
        "",
        "## 已验证项",
        "",
    ]
    lines.extend(_format_items(audit.get("verified_items") or []))
    lines.extend(["", "## 待补齐项", ""])
    lines.extend(_format_items(audit.get("pending_items") or []))
    lines.extend(
        [
            "",
            "## 交付物",
            "",
            f"- demo_report: {str(bool((audit.get('artifacts') or {}).get('demo_report', {}).get('available'))).lower()}",
            f"- resume_summary: {str(bool((audit.get('artifacts') or {}).get('resume_summary', {}).get('available'))).lower()}",
            "",
        ]
    )
    return "\n".join(lines)


def _format_items(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- 无"]
    lines = []
    for item in items:
        lines.append(f"- [{item.get('status')}] {item.get('label')} (`{item.get('id')}`)")
        next_step = item.get("next_step")
        if next_step:
            lines.append(f"  - env: {next_step.get('env')}")
            lines.append(f"  - verify: `{next_step.get('verify')}`")
    return lines


def _with_next_step(item: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(item)
    enriched["next_step"] = NEXT_STEPS.get(
        item.get("id"),
        {
            "env": "按该能力对应的生产配置补齐环境变量或服务地址。",
            "verify": "python scripts/run_evidence_pipeline.py --reports-dir reports",
        },
    )
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate acceptance audit from report summaries.")
    parser.add_argument("--reports-dir", default=str(settings.reports_dir))
    parser.add_argument("--json-output", default="reports/acceptance_audit.json")
    parser.add_argument("--markdown-output", default="reports/acceptance_audit.md")
    args = parser.parse_args()
    audit = build_acceptance_audit(load_report_summary(Path(args.reports_dir)))
    json_text = json.dumps(audit, ensure_ascii=False, indent=2)
    markdown = generate_acceptance_markdown(audit)
    print(json_text)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json_text, encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
