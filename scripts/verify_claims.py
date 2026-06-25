from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "role_accuracy": 0.9,
    "answer_keyword_recall": 0.9,
    "source_keyword_recall": 0.9,
    "benchmark_success_rate": 0.99,
    "benchmark_p95_ms": 5000,
    "grounding_guard_coverage": 1.0,
    "min_graph_edges": 1,
    "required_tools": ["query_rewrite", "compliance_checklist", "knowledge_graph_lookup"],
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(
    eval_report: dict,
    benchmark_report: dict,
    thresholds: dict | None = None,
    readiness_report: dict | None = None,
) -> dict:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    checks = [
        check_at_least("role_accuracy", eval_report.get("role_accuracy", 0), thresholds["role_accuracy"]),
        check_at_least(
            "answer_keyword_recall",
            eval_report.get("answer_keyword_recall", 0),
            thresholds["answer_keyword_recall"],
        ),
        check_at_least(
            "source_keyword_recall",
            eval_report.get("source_keyword_recall", 0),
            thresholds["source_keyword_recall"],
        ),
        check_at_least(
            "benchmark_success_rate",
            benchmark_report.get("success_rate", 0),
            thresholds["benchmark_success_rate"],
        ),
        check_at_most(
            "benchmark_p95_ms",
            benchmark_report.get("latency_ms", {}).get("p95", float("inf")),
            thresholds["benchmark_p95_ms"],
        ),
        check_at_least(
            "grounding_guard_coverage",
            eval_report.get("grounding_guard_coverage", 0),
            thresholds["grounding_guard_coverage"],
        ),
        check_at_least(
            "knowledge_graph_edges",
            eval_report.get("graph", {}).get("edges", 0),
            thresholds["min_graph_edges"],
        ),
        check_required_tools(eval_report.get("tool_usage", {}), thresholds["required_tools"]),
    ]
    passed = all(item["passed"] for item in checks)
    supported_claims = supported_claims_from_checks(checks, eval_report, benchmark_report, readiness_report)
    unsupported_claims = unsupported_claims_from_reports(eval_report, benchmark_report, readiness_report)
    return {
        "passed": passed,
        "checks": checks,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "thresholds": thresholds,
        "readiness": readiness_report,
    }


def check_at_least(name: str, observed: float, threshold: float) -> dict:
    return {
        "name": name,
        "passed": observed >= threshold,
        "observed": observed,
        "threshold": threshold,
        "operator": ">=",
    }


def check_at_most(name: str, observed: float, threshold: float) -> dict:
    return {
        "name": name,
        "passed": observed <= threshold,
        "observed": observed,
        "threshold": threshold,
        "operator": "<=",
    }


def check_required_tools(tool_usage: dict, required_tools: list[str]) -> dict:
    missing = [tool for tool in required_tools if tool_usage.get(tool, 0) <= 0]
    return {
        "name": "required_agent_tools",
        "passed": not missing,
        "observed": tool_usage,
        "threshold": required_tools,
        "operator": "contains",
        "missing": missing,
    }


def supported_claims_from_checks(
    checks: list[dict],
    eval_report: dict,
    benchmark_report: dict,
    readiness_report: dict | None = None,
) -> list[str]:
    by_name = {item["name"]: item for item in checks}
    claims = []
    if by_name["role_accuracy"]["passed"] and by_name["answer_keyword_recall"]["passed"]:
        claims.append(
            f"离线评测集角色识别准确率 {eval_report['role_accuracy']:.0%}，答案关键词召回 {eval_report['answer_keyword_recall']:.0%}"
        )
    if by_name["source_keyword_recall"]["passed"]:
        claims.append(f"来源引用召回 {eval_report['source_keyword_recall']:.0%}")
    if by_name["benchmark_success_rate"]["passed"] and by_name["benchmark_p95_ms"]["passed"]:
        claims.append(
            f"{benchmark_report['concurrency']} 并发压测成功率 {benchmark_report['success_rate']:.0%}，p95 {benchmark_report['latency_ms']['p95']}ms"
        )
    if by_name["knowledge_graph_edges"]["passed"]:
        graph = eval_report["graph"]
        claims.append(f"本地法规知识图谱抽取 {graph['edges']} 条关系，覆盖 {graph['subjects']} 类主体和 {graph['objects']} 类对象")
    if by_name["grounding_guard_coverage"]["passed"]:
        claims.append(f"模型输出接地校验覆盖率 {eval_report['grounding_guard_coverage']:.0%}，无引用或越界引用会回退到检索式答案")
    if by_name["required_agent_tools"]["passed"]:
        claims.append("多智能体工作流包含查询改写、合规清单、知识图谱查询工具调用")
    if readiness_report and readiness_report.get("production_ready"):
        claims.append("生产依赖就绪检查通过，可使用真实外部模型、向量库、重排序和工作流组件")
    return claims


def unsupported_claims_from_reports(
    eval_report: dict,
    benchmark_report: dict,
    readiness_report: dict | None = None,
) -> list[str]:
    unsupported = []
    config = eval_report.get("config", {})
    readiness_by_name = {}
    if readiness_report:
        readiness_by_name = {item["name"]: item for item in readiness_report.get("checks", [])}

    if not config.get("model_configured") and readiness_by_name.get("llm_service", {}).get("status") != "ready":
        unsupported.append("尚未证明真实 Qwen/vLLM 大模型推理服务已接入")
    if config.get("workflow") != "langgraph-stategraph" and readiness_by_name.get("langgraph", {}).get("status") != "ready":
        unsupported.append("当前报告未证明真实 LangGraph StateGraph 已启用")
    if "chroma" not in str(config.get("vector_backend", "")).lower() and readiness_by_name.get("chroma", {}).get("status") != "ready":
        unsupported.append("当前报告未证明真实 Chroma 向量库已启用")
    if config.get("reranker_provider") == "local" and readiness_by_name.get("reranker_service", {}).get("status") != "ready":
        unsupported.append("当前报告未证明真实 BGE/gte reranker 服务已启用")
    if benchmark_report.get("concurrency", 0) < 1000:
        unsupported.append("当前压测规模不足以支撑 1000+ 并发用户声明")
    return unsupported


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify resume-style claims from eval and benchmark reports.")
    parser.add_argument("--eval", default="reports/eval.json")
    parser.add_argument("--benchmark", default="reports/benchmark.json")
    parser.add_argument("--readiness", default="")
    parser.add_argument("--output", default="reports/claim_verification.json")
    args = parser.parse_args()
    readiness = load_json(Path(args.readiness)) if args.readiness else None
    report = verify(load_json(Path(args.eval)), load_json(Path(args.benchmark)), readiness_report=readiness)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
