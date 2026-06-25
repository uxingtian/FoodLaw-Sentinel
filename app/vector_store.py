from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.embedding import EmbeddingClient
from app.models import KnowledgeChunk
from app.retrieval_utils import normalize_scores, tokenize


@dataclass
class VectorSearchHit:
    chunk_id: str
    score: float


class VectorStore:
    name = "base"

    @property
    def ready(self) -> bool:
        raise NotImplementedError

    def rebuild(self, chunks: list[KnowledgeChunk]) -> None:
        raise NotImplementedError

    def search(self, question: str, top_k: int) -> list[VectorSearchHit]:
        raise NotImplementedError


class LocalTfidfVectorStore(VectorStore):
    name = "local-tfidf"

    def __init__(self) -> None:
        self.chunks: list[KnowledgeChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self._vectorizer is not None and self._matrix is not None

    def rebuild(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in chunks])

    def search(self, question: str, top_k: int) -> list[VectorSearchHit]:
        if not self.ready or self._vectorizer is None or self._matrix is None:
            return []
        query_vector = self._vectorizer.transform([question])
        raw_scores = (self._matrix @ query_vector.T).toarray().ravel()
        scores = normalize_scores(raw_scores)
        hits = [
            VectorSearchHit(chunk_id=chunk.id, score=round(float(scores[index]), 4))
            for index, chunk in enumerate(self.chunks)
            if scores[index] > 0
        ]
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


class DenseEmbeddingVectorStore(VectorStore):
    name = "dense-embedding"

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embedding_client = embedding_client
        self.chunks: list[KnowledgeChunk] = []
        self._matrix: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return bool(self.chunks) and self._matrix is not None

    def rebuild(self, chunks: list[KnowledgeChunk]) -> None:
        self.chunks = chunks
        if not chunks:
            self._matrix = None
            return
        vectors = self.embedding_client.embed_texts([chunk.text for chunk in chunks])
        matrix = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def search(self, question: str, top_k: int) -> list[VectorSearchHit]:
        if not self.ready or self._matrix is None:
            return []
        query = np.asarray(self.embedding_client.embed_texts([question])[0], dtype=float)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm
        raw_scores = self._matrix @ query
        scores = normalize_scores(raw_scores)
        hits = [
            VectorSearchHit(chunk_id=chunk.id, score=round(float(scores[index]), 4))
            for index, chunk in enumerate(self.chunks)
            if scores[index] > 0
        ]
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


class ChromaVectorStore(VectorStore):
    name = "chroma"

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._chunks_by_id: dict[str, KnowledgeChunk] = {}
        try:
            import chromadb

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection("food_law_chunks")
        except Exception:
            self._client = None
            self._collection = None

    @property
    def ready(self) -> bool:
        return self._collection is not None and bool(self._chunks_by_id)

    def rebuild(self, chunks: list[KnowledgeChunk]) -> None:
        self._chunks_by_id = {chunk.id: chunk for chunk in chunks}
        if self._collection is None:
            return
        try:
            existing = self._collection.get(include=[])
            ids = existing.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
            if not chunks:
                return
            self._collection.add(
                ids=[chunk.id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "role": chunk.role,
                        "source": chunk.source,
                    }
                    for chunk in chunks
                ],
            )
        except Exception:
            self._collection = None

    def search(self, question: str, top_k: int) -> list[VectorSearchHit]:
        if self._collection is None:
            return []
        try:
            result = self._collection.query(query_texts=[question], n_results=top_k)
        except Exception:
            return []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else [0.0] * len(ids)
        hits: list[VectorSearchHit] = []
        for chunk_id, distance in zip(ids, distances):
            score = 1.0 / (1.0 + max(0.0, float(distance)))
            hits.append(VectorSearchHit(chunk_id=chunk_id, score=round(score, 4)))
        return hits


def build_vector_store(backend: str, persist_dir: Path, embedding_client: EmbeddingClient | None = None) -> VectorStore:
    normalized = backend.lower()
    if normalized in {"dense", "embedding", "local-embedding", "api-embedding"} and embedding_client is not None:
        return DenseEmbeddingVectorStore(embedding_client)
    if backend.lower() == "chroma":
        store = ChromaVectorStore(persist_dir)
        if store._collection is not None:
            return store
    return LocalTfidfVectorStore()
