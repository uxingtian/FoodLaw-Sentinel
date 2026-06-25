from __future__ import annotations

from dataclasses import dataclass, field

from app.models import DocumentRole, KnowledgeChunk
from app.retrieval_utils import tokenize


SUBJECT_TERMS = {
    "消费者": ["消费者", "购买者", "举报人"],
    "监管机构": ["监管", "市场监督", "部门", "监管部门", "主管部门"],
    "生产经营者": ["生产经营者", "企业", "生产企业", "经营者", "食品生产"],
}

OBLIGATION_TERMS = {
    "赔偿损失": ["赔偿", "赔偿损失", "索赔", "惩罚性赔偿"],
    "投诉举报": ["投诉", "举报", "投诉举报"],
    "监督检查": ["监督检查", "检查", "抽检", "抽样检验"],
    "风险处置": ["风险", "隐患", "查封", "扣押", "整改", "约谈"],
    "食品召回": ["召回", "停止生产经营", "通知消费者"],
    "进货查验": ["进货查验", "查验记录", "记录保存"],
    "标签合规": ["标签", "说明书", "疾病", "治疗功能", "预包装"],
    "食品安全自查": ["自查", "食品安全管理制度", "过程控制"],
}


@dataclass
class GraphEdge:
    subject: str
    relation: str
    object: str
    chunk_id: str
    document_id: str
    role: DocumentRole
    title: str
    evidence: str
    tokens: set[str] = field(default_factory=set, repr=False)


@dataclass
class GraphHit:
    chunk_id: str
    score: float
    edges: list[GraphEdge]


class LegalKnowledgeGraph:
    name = "local-legal-kg"

    def __init__(self, cache_size: int = 512) -> None:
        self.edges: list[GraphEdge] = []
        self._edges_by_chunk: dict[str, list[GraphEdge]] = {}
        self._search_cache: dict[tuple[str, DocumentRole, int], list[GraphHit]] = {}
        self._cache_size = cache_size

    @property
    def ready(self) -> bool:
        return bool(self.edges)

    def rebuild(self, chunks: list[KnowledgeChunk]) -> None:
        self.edges = []
        self._edges_by_chunk = {}
        self._search_cache = {}
        for chunk in chunks:
            for edge in extract_edges(chunk):
                edge.tokens = edge_tokens(edge)
                self.edges.append(edge)
                self._edges_by_chunk.setdefault(edge.chunk_id, []).append(edge)

    def search(self, question: str, role: DocumentRole, top_k: int = 10) -> list[GraphHit]:
        cache_key = (question, role, top_k)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        query_tokens = set(tokenize(question))
        scored: list[GraphHit] = []
        for chunk_id, edges in self._edges_by_chunk.items():
            score = 0.0
            matched_edges: list[GraphEdge] = []
            for edge in edges:
                overlap = len(query_tokens.intersection(edge.tokens))
                if overlap:
                    score += overlap / max(1, len(query_tokens))
                    matched_edges.append(edge)
                if edge.role == role:
                    score += 0.08
            if score > 0:
                scored.append(GraphHit(chunk_id=chunk_id, score=round(min(1.0, score), 4), edges=matched_edges or edges[:1]))
        scored.sort(key=lambda item: item.score, reverse=True)
        result = scored[:top_k]
        self._remember(cache_key, result)
        return result

    def stats(self) -> dict:
        subjects = {edge.subject for edge in self.edges}
        objects = {edge.object for edge in self.edges}
        roles = {"regulator": 0, "consumer": 0, "producer": 0, "general": 0}
        for edge in self.edges:
            roles[edge.role] += 1
        return {
            "name": self.name,
            "ready": self.ready,
            "edges": len(self.edges),
            "subjects": len(subjects),
            "objects": len(objects),
            "roles": roles,
            "cache_entries": len(self._search_cache),
        }

    def sample(self, limit: int = 20) -> list[dict]:
        return [edge_to_dict(edge) for edge in self.edges[:limit]]

    def _remember(self, key: tuple[str, DocumentRole, int], value: list[GraphHit]) -> None:
        if len(self._search_cache) >= self._cache_size:
            first_key = next(iter(self._search_cache))
            self._search_cache.pop(first_key, None)
        self._search_cache[key] = value


def extract_edges(chunk: KnowledgeChunk) -> list[GraphEdge]:
    text = chunk.text
    subjects = find_terms(text, SUBJECT_TERMS)
    objects = find_terms(text, OBLIGATION_TERMS)
    edges: list[GraphEdge] = []
    for subject in subjects:
        for obj in objects:
            edges.append(
                GraphEdge(
                    subject=subject,
                    relation=relation_for_object(obj),
                    object=obj,
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    role=chunk.role,
                    title=chunk.title,
                    evidence=text[:180],
                )
            )
    return dedupe_edges(edges)


def find_terms(text: str, taxonomy: dict[str, list[str]]) -> list[str]:
    hits = []
    for canonical, variants in taxonomy.items():
        if any(variant in text for variant in variants):
            hits.append(canonical)
    return hits


def relation_for_object(obj: str) -> str:
    if obj in {"赔偿损失", "投诉举报"}:
        return "享有权利"
    if obj in {"监督检查", "风险处置"}:
        return "依法履职"
    return "承担义务"


def edge_tokens(edge: GraphEdge) -> set[str]:
    return set(tokenize(f"{edge.subject} {edge.relation} {edge.object} {edge.evidence}"))


def dedupe_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[GraphEdge] = []
    for edge in edges:
        key = (edge.subject, edge.relation, edge.object, edge.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def edge_to_dict(edge: GraphEdge) -> dict:
    return {
        "subject": edge.subject,
        "relation": edge.relation,
        "object": edge.object,
        "chunk_id": edge.chunk_id,
        "document_id": edge.document_id,
        "role": edge.role,
        "title": edge.title,
        "evidence": edge.evidence,
    }
