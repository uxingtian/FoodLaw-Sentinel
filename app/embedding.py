from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import requests

from app.retrieval_utils import tokenize


class EmbeddingClient:
    name = "base"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass
class LocalHashEmbeddingClient(EmbeddingClient):
    dimensions: int = 384
    name: str = "local-hash-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=float)
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


@dataclass
class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    base_url: str
    api_key: str
    model: str
    timeout: float = 30.0
    name: str = "openai-compatible-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.base_url or not self.api_key:
            raise RuntimeError("embedding service is not configured")
        endpoint = self.base_url.rstrip("/") + "/embeddings"
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding", []) for item in data]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding service returned an unexpected vector count")
        return vectors


def build_embedding_client(provider: str, model: str, api_key: str, base_url: str) -> EmbeddingClient:
    if provider.lower() in {"openai", "openai-compatible", "api", "qwen"} and api_key and base_url:
        return OpenAICompatibleEmbeddingClient(base_url=base_url, api_key=api_key, model=model)
    return LocalHashEmbeddingClient(name=f"local-{model}")
