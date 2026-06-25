from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.graph_store import LegalKnowledgeGraph, edge_to_dict
from app.models import DocumentRole


@dataclass
class ToolResult:
    name: str
    summary: str
    data: dict


class AgentTool(Protocol):
    name: str

    def run(self, question: str, role: DocumentRole) -> ToolResult:
        ...


class QueryRewriteTool:
    name = "query_rewrite"

    ROLE_EXPANSIONS: dict[DocumentRole, list[str]] = {
        "regulator": ["监督检查", "风险处置", "抽样检验", "责令整改", "召回监督"],
        "consumer": ["消费者权益", "投诉举报", "赔偿损失", "食品安全标准", "标签说明"],
        "producer": ["生产经营者义务", "进货查验", "食品召回", "标签合规", "食品安全自查"],
        "general": ["食品安全法", "食品安全标准", "法律责任", "合规要求"],
    }

    def run(self, question: str, role: DocumentRole) -> ToolResult:
        additions = [term for term in self.ROLE_EXPANSIONS[role] if term not in question][:4]
        rewritten = question if not additions else f"{question} {' '.join(additions)}"
        return ToolResult(
            name=self.name,
            summary=f"扩展 {len(additions)} 个检索词",
            data={"rewritten_query": rewritten, "additions": additions},
        )


class ComplianceChecklistTool:
    name = "compliance_checklist"

    CHECKLISTS: dict[DocumentRole, list[str]] = {
        "regulator": ["核查许可资质", "固定证据", "判断风险等级", "依法采取整改/抽检/召回措施"],
        "consumer": ["保存购物凭证", "保留问题食品和标签照片", "向经营者主张赔偿", "向监管部门投诉举报"],
        "producer": ["停止风险产品生产经营", "通知相关方", "召回已上市食品", "记录召回和整改情况"],
        "general": ["确认事实场景", "检索现行法规", "核对适用主体", "保留来源依据"],
    }

    def run(self, question: str, role: DocumentRole) -> ToolResult:
        checklist = self.CHECKLISTS[role]
        return ToolResult(
            name=self.name,
            summary=f"生成 {len(checklist)} 项处置清单",
            data={"checklist": checklist},
        )


class GraphLookupTool:
    name = "knowledge_graph_lookup"

    def __init__(self, graph: LegalKnowledgeGraph) -> None:
        self.graph = graph

    def run(self, question: str, role: DocumentRole) -> ToolResult:
        hits = self.graph.search(question, role=role, top_k=3)
        edges = []
        for hit in hits:
            edges.extend(edge_to_dict(edge) for edge in hit.edges[:2])
        return ToolResult(
            name=self.name,
            summary=f"命中 {len(edges)} 条图谱关系",
            data={"edges": edges[:6]},
        )


class ToolRegistry:
    def __init__(self, tools: list[AgentTool] | None = None) -> None:
        self.tools = tools or [QueryRewriteTool(), ComplianceChecklistTool()]

    def run(self, question: str, role: DocumentRole) -> list[ToolResult]:
        return [tool.run(question, role) for tool in self.tools]


def tool_results_as_route(results: list[ToolResult]) -> list[dict]:
    return [{"name": item.name, "summary": item.summary, "data": item.data} for item in results]


def rewritten_query(question: str, results: list[ToolResult]) -> str:
    for result in results:
        if result.name == "query_rewrite":
            return str(result.data.get("rewritten_query") or question)
    return question
