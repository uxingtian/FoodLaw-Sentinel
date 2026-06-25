from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_TITLE = "食品安全法律法规多智能体问答系统"
TECH_STACK = [
    "FastAPI",
    "RAG",
    "BM25",
    "可插拔向量检索",
    "本地法规知识图谱",
    "MCP 风格工具调用",
    "OpenAI-compatible LLM/vLLM 适配",
    "可选 Chroma/LangGraph/Reranker 接入",
]


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_summary(report: dict) -> str:
    supported = report.get("supported_claims", [])
    unsupported = report.get("unsupported_claims", [])
    readiness = report.get("readiness") or {}
    lines = [
        f"# {PROJECT_TITLE}",
        "",
        "## 简历安全版描述",
        "",
        "构建食品安全法律法规多智能体问答系统，支持法规资料导入、角色化问答、混合检索、知识图谱增强、工具调用、来源引用、评测与压测报告生成；系统预留 Qwen/vLLM、Chroma、LangGraph、Embedding 和 Reranker 的生产接入能力。",
        "",
        "## 技术栈",
        "",
        "、".join(TECH_STACK),
        "",
        "## 已由当前报告证明的成果",
        "",
    ]
    lines.extend(f"- {claim}" for claim in supported)
    lines.extend(
        [
            "",
            "## 不建议直接写成已完成的生产成果",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in unsupported)
    lines.extend(
        [
            "",
            "## 当前生产依赖就绪状态",
            "",
            f"- production_ready: {str(readiness.get('production_ready', False)).lower()}",
        ]
    )
    for check in readiness.get("checks", []):
        lines.append(f"- {check['name']}: {check['status']}；{check['detail']}")
    lines.extend(
        [
            "",
            "## 推荐简历项目经历写法",
            "",
            "- 设计并实现食品安全法律法规多智能体问答系统，构建管理智能体、消费者/监管/生产经营者角色智能体和 MCP 风格工具调用链路，支持查询改写、合规清单生成与知识图谱查询。",
            "- 构建 BM25、向量检索、知识图谱增强与 reranker 融合的 RAG 检索链路，回答结果返回来源引用、置信度、路由 trace 和工具调用记录。",
            "- 建立离线评测、HTTP 压测、生产依赖 readiness 和简历 claim verifier 证据链，当前评测报告可证明角色识别、答案关键词召回、来源召回、图谱关系抽取和响应时间指标。",
            "- 预留 OpenAI-compatible Qwen/vLLM、Chroma、LangGraph、Embedding、BGE/gte reranker 接入配置，支持后续切换到生产组件并复用同一套验收报告。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a resume-safe project summary from verified claims.")
    parser.add_argument("--claims", default="reports/claim_verification.json")
    parser.add_argument("--output", default="reports/resume_summary.md")
    args = parser.parse_args()
    summary = generate_summary(load_report(Path(args.claims)))
    print(summary)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
