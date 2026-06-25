from __future__ import annotations

from collections import Counter
from typing import Any


RESUME_REQUIREMENTS = [
    {
        "id": "multi_agent_workflow",
        "label": "管理智能体 + 监管/消费者/生产者角色智能体",
        "category": "agent",
        "check": "required_agent_tools",
    },
    {
        "id": "source_cited_rag",
        "label": "混合检索问答并返回法规来源引用",
        "category": "rag",
        "check": "source_keyword_recall",
    },
    {
        "id": "response_latency",
        "label": "问答响应时间控制在 5 秒内",
        "category": "performance",
        "check": "benchmark_p95_ms",
    },
    {
        "id": "knowledge_graph",
        "label": "法规知识图谱/关系抽取增强检索",
        "category": "rag",
        "check": "knowledge_graph_edges",
    },
    {
        "id": "grounding_guard",
        "label": "模型回答来源引用与接地校验护栏",
        "category": "rag",
        "check": "grounding_guard_coverage",
    },
    {
        "id": "qwen_vllm_inference",
        "label": "Qwen/vLLM/OpenAI-compatible 大模型推理接入",
        "category": "production",
        "readiness": "llm_service",
    },
    {
        "id": "chroma_langgraph_stack",
        "label": "Chroma 向量库与 LangGraph 工作流生产栈",
        "category": "production",
        "readiness_all": ["chroma", "langgraph"],
    },
    {
        "id": "reranker_service",
        "label": "BGE/gte reranker 重排序服务",
        "category": "production",
        "readiness": "reranker_service",
    },
    {
        "id": "high_concurrency_claim",
        "label": "1000+ 并发用户访问声明",
        "category": "performance",
        "minimum_concurrency": 1000,
    },
]


def build_resume_alignment(claim_report: dict[str, Any]) -> dict[str, Any]:
    checks = {item.get("name"): item for item in claim_report.get("checks", [])}
    readiness_report = claim_report.get("readiness") or {}
    readiness = {item.get("name"): item for item in readiness_report.get("checks", [])}
    benchmark = claim_report.get("benchmark") or {}
    items = [
        _evaluate_requirement(requirement, checks=checks, readiness=readiness, benchmark=benchmark)
        for requirement in RESUME_REQUIREMENTS
    ]
    counts = Counter(item["status"] for item in items)
    return {
        "summary": {
            "total": len(items),
            "verified": counts["verified"],
            "implemented": counts["implemented"],
            "pending_external_service": counts["pending_external_service"],
            "pending_evidence": counts["pending_evidence"],
        },
        "items": items,
    }


def _evaluate_requirement(
    requirement: dict[str, Any],
    *,
    checks: dict[str, dict[str, Any]],
    readiness: dict[str, dict[str, Any]],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    status = "pending_evidence"
    if "check" in requirement:
        check = checks.get(requirement["check"], {})
        evidence = check
        status = "verified" if check.get("passed") else "pending_evidence"
    elif "readiness" in requirement:
        check = readiness.get(requirement["readiness"], {})
        evidence = check
        status = "verified" if check.get("status") == "ready" else "pending_external_service"
    elif "readiness_all" in requirement:
        selected = [readiness.get(name, {"name": name, "status": "missing"}) for name in requirement["readiness_all"]]
        evidence = {"checks": selected}
        status = "verified" if all(item.get("status") == "ready" for item in selected) else "pending_external_service"
    elif "minimum_concurrency" in requirement:
        observed = benchmark.get("concurrency", 0)
        evidence = {"observed": observed, "threshold": requirement["minimum_concurrency"]}
        status = "verified" if observed >= requirement["minimum_concurrency"] else "pending_evidence"

    return {
        "id": requirement["id"],
        "label": requirement["label"],
        "category": requirement["category"],
        "status": status,
        "evidence": evidence,
    }
