from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["auto", "regulator", "consumer", "producer", "general"]
DocumentRole = Literal["regulator", "consumer", "producer", "general"]

ROLE_LABELS: dict[str, str] = {
    "auto": "自动识别",
    "regulator": "监管机构",
    "consumer": "消费者",
    "producer": "生产经营者",
    "general": "通用咨询",
}


class DocumentMeta(BaseModel):
    id: str
    title: str
    role: DocumentRole
    source: str
    filename: str
    content_type: str = ""
    created_at: datetime
    chunk_count: int = 0


class KnowledgeChunk(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    title: str
    role: DocumentRole
    source: str
    text: str


class SourceCitation(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    title: str
    source: str
    role: DocumentRole
    excerpt: str
    score: float = Field(ge=0)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    role: Role = "auto"
    top_k: int | None = Field(default=None, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    role: DocumentRole
    confidence: float = Field(ge=0, le=1)
    sources: list[SourceCitation]
    route: dict[str, Any]
    fallback_used: bool


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_configured: bool
    index_ready: bool
    documents: int
    chunks: int
    vector_backend: str = "local-tfidf"
    embedding_provider: str = "local"
    embedding_model: str = ""
    reranker_provider: str = "local"
    reranker_model: str = ""
    workflow: str = "local-agent-graph"


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    roles: dict[str, int]
