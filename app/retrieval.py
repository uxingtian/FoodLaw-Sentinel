from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.graph_store import LegalKnowledgeGraph
from app.models import DocumentRole, KnowledgeChunk
from app.retrieval_utils import normalize_scores, tokenize
from app.vector_store import LocalTfidfVectorStore, VectorStore


@dataclass
class SearchResult:
    chunk: KnowledgeChunk
    score: float
    bm25_score: float
    semantic_score: float
    graph_score: float = 0.0
    rerank_score: float = 0.0


class HybridRetriever:
    """Hybrid retrieval: BM25 keyword recall plus a pluggable vector store."""

    def __init__(self, vector_store: VectorStore | None = None, graph: LegalKnowledgeGraph | None = None) -> None:
        self.chunks: list[KnowledgeChunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self.vector_store = vector_store or LocalTfidfVectorStore()
        self.graph = graph or LegalKnowledgeGraph()

    @property
    def ready(self) -> bool:
        return bool(self.chunks)

    @property
    def backend_name(self) -> str:
        return f"{self.vector_store.name}+{self.graph.name}"

    def rebuild(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks
        self._tokenized = [tokenize(chunk.text) for chunk in chunks]
        self.vector_store.rebuild(chunks)
        self.graph.rebuild(chunks)
        self._bm25 = BM25Okapi(self._tokenized) if chunks else None

    def search(self, question: str, role: DocumentRole, top_k: int = 5) -> list[SearchResult]:
        if not self.chunks or not self._bm25:
            return []
        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        bm25_scores = normalize_scores(self._bm25.get_scores(query_tokens))
        vector_hits = self.vector_store.search(question, top_k=max(top_k * 4, 20))
        vector_scores = {hit.chunk_id: hit.score for hit in vector_hits}
        graph_hits = self.graph.search(question, role=role, top_k=max(top_k * 4, 20))
        graph_scores = {hit.chunk_id: hit.score for hit in graph_hits}

        results: list[SearchResult] = []
        for index, chunk in enumerate(self.chunks):
            semantic_score = vector_scores.get(chunk.id, 0.0)
            graph_score = graph_scores.get(chunk.id, 0.0)
            role_bonus = role_match_bonus(chunk.role, role)
            score = min(1.0, 0.48 * bm25_scores[index] + 0.34 * semantic_score + 0.18 * graph_score + role_bonus)
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=round(float(score), 4),
                    bm25_score=round(float(bm25_scores[index]), 4),
                    semantic_score=round(float(semantic_score), 4),
                    graph_score=round(float(graph_score), 4),
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return dedupe_by_document(results, top_k)


def role_match_bonus(chunk_role: DocumentRole, requested_role: DocumentRole) -> float:
    if chunk_role == requested_role:
        return 0.08
    if chunk_role == "general":
        return 0.03
    if requested_role == "general":
        return 0.02
    return 0.0


def dedupe_by_document(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    selected: list[SearchResult] = []
    per_document: dict[str, int] = {}
    for result in results:
        count = per_document.get(result.chunk.document_id, 0)
        if count >= 2:
            continue
        selected.append(result)
        per_document[result.chunk.document_id] = count + 1
        if len(selected) >= top_k:
            break
    return selected
