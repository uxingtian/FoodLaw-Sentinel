from __future__ import annotations

from typing import Any

from app.models import ChatRequest
from app.workflow import AgentWorkflow


DEFAULT_DEMO_SCENARIOS = [
    {
        "id": "regulator-inspection",
        "title": "监管抽检处置",
        "question": "监管部门抽检发现食品不符合安全标准时，应当如何依法处置？",
        "role": "auto",
        "expected_role": "regulator",
    },
    {
        "id": "consumer-claim",
        "title": "消费者维权索赔",
        "question": "消费者买到不符合食品安全标准的食品，应该如何投诉和索赔？",
        "role": "auto",
        "expected_role": "consumer",
    },
    {
        "id": "producer-compliance",
        "title": "生产经营者合规",
        "question": "食品生产经营企业如何建立进货查验记录和召回制度？",
        "role": "auto",
        "expected_role": "producer",
    },
]


async def run_demo_scenarios(
    workflow: AgentWorkflow,
    scenarios: list[dict[str, Any]] | None = None,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    scenarios = scenarios or DEFAULT_DEMO_SCENARIOS
    items = []
    for scenario in scenarios:
        response = await workflow.answer(
            ChatRequest(question=scenario["question"], role=scenario.get("role", "auto"), top_k=top_k)
        )
        guard = response.route.get("generation_guard") or {}
        items.append(
            {
                "id": scenario["id"],
                "title": scenario["title"],
                "question": scenario["question"],
                "expected_role": scenario["expected_role"],
                "actual_role": response.role,
                "role_ok": response.role == scenario["expected_role"],
                "confidence": response.confidence,
                "sources": len(response.sources),
                "fallback_used": response.fallback_used,
                "generation_guard": guard,
                "answer_preview": response.answer[:220],
                "route": {
                    "reason": response.route.get("reason"),
                    "workflow": response.route.get("workflow"),
                    "retriever_backend": response.route.get("retriever_backend"),
                    "reranker": response.route.get("reranker"),
                    "tools": response.route.get("tools", []),
                    "trace": response.route.get("trace", {}),
                },
            }
        )
    return {"summary": _summarize(items), "items": items}


def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    role_passed = sum(1 for item in items if item["role_ok"])
    with_sources = sum(1 for item in items if item["sources"] > 0)
    with_generation_guard = sum(1 for item in items if item.get("generation_guard"))
    return {
        "total": total,
        "role_passed": role_passed,
        "with_sources": with_sources,
        "with_generation_guard": with_generation_guard,
        "passed": total > 0 and role_passed == total and with_sources == total and with_generation_guard == total,
    }
