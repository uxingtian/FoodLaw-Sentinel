from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import requests

from app.models import DocumentRole
from app.retrieval import SearchResult
from app.retrieval_utils import tokenize


class Reranker(Protocol):
    model_name: str

    def rerank(self, question: str, role: DocumentRole, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        ...


class LocalReranker:
    """A deterministic reranker used when no cross-encoder service is available."""

    def __init__(self, model_name: str = "local-bge-compatible") -> None:
        self.model_name = model_name

    def rerank(self, question: str, role: DocumentRole, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        query_tokens = set(tokenize(question))
        reranked: list[SearchResult] = []
        for result in results:
            haystack = f"{result.chunk.title} {result.chunk.source} {result.chunk.text}"
            text_tokens = set(tokenize(haystack))
            overlap = len(query_tokens.intersection(text_tokens))
            coverage = overlap / max(1, len(query_tokens))
            role_score = 0.08 if result.chunk.role == role else 0.03 if result.chunk.role == "general" else 0.0
            rerank_score = min(1.0, 0.7 * result.score + 0.22 * coverage + role_score)
            reranked.append(
                replace(
                    result,
                    score=round(float(rerank_score), 4),
                    rerank_score=round(float(rerank_score), 4),
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]


class HttpReranker:
    """Adapter for BGE/gte style reranker HTTP services."""

    def __init__(self, url: str, model_name: str = "bge-reranker", timeout: float = 30.0) -> None:
        self.url = url
        self.model_name = model_name
        self.timeout = timeout

    def rerank(self, question: str, role: DocumentRole, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        payload = {
            "model": self.model_name,
            "query": question,
            "documents": [result.chunk.text for result in results],
            "top_k": top_k,
        }
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        scores = parse_reranker_scores(data, len(results))
        reranked = [
            replace(result, score=round(float(score), 4), rerank_score=round(float(score), 4))
            for result, score in zip(results, scores)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]


def parse_reranker_scores(data: dict, expected: int) -> list[float]:
    if "scores" in data:
        scores = [float(score) for score in data["scores"]]
    elif "results" in data:
        indexed = sorted(data["results"], key=lambda item: item.get("index", 0))
        scores = [float(item.get("score", item.get("relevance_score", 0.0))) for item in indexed]
    else:
        scores = []
    if len(scores) != expected:
        raise RuntimeError("reranker service returned an unexpected score count")
    return scores


def build_reranker(provider: str, model_name: str, url: str = "") -> Reranker | None:
    normalized = provider.lower()
    if normalized in {"", "none", "disabled"}:
        return None
    if normalized in {"http", "api", "bge", "gte"} and url:
        return HttpReranker(url=url, model_name=model_name)
    return LocalReranker(model_name=model_name)
