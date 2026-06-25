from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.agents import confidence_from_results, fallback_answer, route_question, to_citations
from app.config import Settings
from app.grounding import validate_grounded_answer
from app.llm import generate_with_model
from app.models import ChatRequest, ChatResponse
from app.reranker import Reranker
from app.retrieval import HybridRetriever
from app.tools import ToolRegistry, rewritten_query, tool_results_as_route


@dataclass
class WorkflowTrace:
    route_ms: float = 0.0
    tool_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "route_ms": round(self.route_ms, 2),
            "tool_ms": round(self.tool_ms, 2),
            "retrieve_ms": round(self.retrieve_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "generate_ms": round(self.generate_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


class AgentWorkflow:
    """LangGraph-compatible orchestration boundary for the QA agents."""

    backend_name = "local-agent-graph"

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: Reranker | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.tool_registry = tool_registry or ToolRegistry()

    async def answer(self, request: ChatRequest) -> ChatResponse:
        started = perf_counter()
        question = request.question.strip()

        t0 = perf_counter()
        decision = route_question(question, request.role)
        route_ms = elapsed_ms(t0)

        top_k = request.top_k or self.settings.default_top_k
        recall_k = min(20, max(top_k * 3, top_k))

        t0 = perf_counter()
        tool_results = self.tool_registry.run(question, decision.role)
        search_query = rewritten_query(question, tool_results)
        tool_ms = elapsed_ms(t0)

        t0 = perf_counter()
        results = self.retriever.search(search_query, role=decision.role, top_k=recall_k)
        retrieve_ms = elapsed_ms(t0)

        t0 = perf_counter()
        if self.reranker is not None:
            results = self.reranker.rerank(search_query, decision.role, results, top_k=top_k)
        else:
            results = results[:top_k]
        rerank_ms = elapsed_ms(t0)

        confidence = confidence_from_results(results)
        t0 = perf_counter()
        model_answer = None
        if results and confidence >= 0.08:
            model_answer = await generate_with_model(
                settings=self.settings,
                question=question,
                role=decision.role,
                results=results,
            )
        generate_ms = elapsed_ms(t0)

        grounding = validate_grounded_answer(model_answer, len(results)) if model_answer is not None else None
        fallback_used = model_answer is None or not grounding.accepted
        answer = model_answer if model_answer is not None and grounding.accepted else fallback_answer(question, decision.role, results, confidence)
        route = decision.as_dict()
        route["workflow"] = self.backend_name
        route["retriever_backend"] = self.retriever.backend_name
        route["reranker"] = self.reranker.model_name if self.reranker else "none"
        route["query"] = {"original": question, "rewritten": search_query}
        route["tools"] = tool_results_as_route(tool_results)
        route["generation_guard"] = (
            grounding.as_dict()
            if grounding is not None
            else {"accepted": False, "violations": ["model_not_used"], "citations": []}
        )
        route["trace"] = WorkflowTrace(
            route_ms=route_ms,
            tool_ms=tool_ms,
            retrieve_ms=retrieve_ms,
            rerank_ms=rerank_ms,
            generate_ms=generate_ms,
            total_ms=elapsed_ms(started),
        ).as_dict()

        return ChatResponse(
            answer=answer,
            role=decision.role,
            confidence=confidence,
            sources=to_citations(results),
            route=route,
            fallback_used=fallback_used,
        )


def elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000


class LangGraphAgentWorkflow(AgentWorkflow):
    backend_name = "langgraph-stategraph"

    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: Reranker | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__(settings=settings, retriever=retriever, reranker=reranker, tool_registry=tool_registry)
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception as exc:
            raise RuntimeError("langgraph is not installed") from exc

        graph = StateGraph(dict)
        graph.add_node("answer", self._answer_node)
        graph.set_entry_point("answer")
        graph.add_edge("answer", END)
        return graph.compile()

    async def _answer_node(self, state: dict) -> dict:
        state["response"] = await super().answer(state["request"])
        return state

    async def answer(self, request: ChatRequest) -> ChatResponse:
        state = await self._graph.ainvoke({"request": request})
        return state["response"]


def build_workflow(
    settings: Settings,
    retriever: HybridRetriever,
    reranker: Reranker | None = None,
    tool_registry: ToolRegistry | None = None,
) -> AgentWorkflow:
    if settings.workflow_backend.lower() in {"langgraph", "auto"}:
        try:
            return LangGraphAgentWorkflow(
                settings=settings,
                retriever=retriever,
                reranker=reranker,
                tool_registry=tool_registry,
            )
        except Exception:
            if settings.workflow_backend.lower() == "langgraph":
                raise
    return AgentWorkflow(settings=settings, retriever=retriever, reranker=reranker, tool_registry=tool_registry)
