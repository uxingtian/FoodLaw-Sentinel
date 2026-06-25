from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.main import startup
from app.models import ChatRequest
from app.workflow import AgentWorkflow
import app.main as main


def load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


async def run_case(workflow: AgentWorkflow, case: dict) -> dict:
    response = await workflow.answer(
        ChatRequest(question=case["question"], role=case.get("role", "auto"), top_k=case.get("top_k", 5))
    )
    answer_text = response.answer
    sources_text = "\n".join(f"{source.title}\n{source.source}\n{source.excerpt}" for source in response.sources)
    must_include = case.get("must_include", [])
    source_keywords = case.get("expected_source_keywords", [])
    return {
        "id": case["id"],
        "role_ok": response.role == case["expected_role"],
        "answer_keyword_recall": ratio_present(must_include, answer_text),
        "source_keyword_recall": ratio_present(source_keywords, sources_text),
        "confidence": response.confidence,
        "sources": len(response.sources),
        "latency_ms": response.route["trace"]["total_ms"],
        "fallback_used": response.fallback_used,
        "generation_guard": response.route.get("generation_guard"),
        "tools": [tool["name"] for tool in response.route.get("tools", [])],
        "rewritten_query_changed": response.route.get("query", {}).get("rewritten")
        != response.route.get("query", {}).get("original"),
    }


def ratio_present(keywords: list[str], text: str) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword in text)
    return hits / len(keywords)


async def evaluate(cases: list[dict]) -> dict:
    startup()
    workflow = main.workflow
    results = [await run_case(workflow, case) for case in cases]
    total = len(results)
    role_accuracy = sum(1 for item in results if item["role_ok"]) / total if total else 0
    answer_recall = sum(item["answer_keyword_recall"] for item in results) / total if total else 0
    source_recall = sum(item["source_keyword_recall"] for item in results) / total if total else 0
    avg_latency = sum(item["latency_ms"] for item in results) / total if total else 0
    grounding_guard_coverage = sum(1 for item in results if item.get("generation_guard")) / total if total else 0
    tool_usage = {}
    for item in results:
        for tool_name in item["tools"]:
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
    return {
        "cases": total,
        "role_accuracy": round(role_accuracy, 4),
        "answer_keyword_recall": round(answer_recall, 4),
        "source_keyword_recall": round(source_recall, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "grounding_guard_coverage": round(grounding_guard_coverage, 4),
        "tool_usage": tool_usage,
        "results": results,
        "config": {
            "vector_backend": main.retriever.backend_name,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "reranker_provider": settings.reranker_provider,
            "reranker_model": settings.reranker_model,
            "workflow": main.workflow.backend_name,
            "model_configured": settings.model_configured,
        },
        "graph": main.retriever.graph.stats(),
    }


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the food safety legal QA system.")
    parser.add_argument("--cases", default="eval/golden_food_safety_qa.json")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = asyncio.run(evaluate(load_cases(Path(args.cases))))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main_cli()
