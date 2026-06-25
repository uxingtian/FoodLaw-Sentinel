from __future__ import annotations

import re
from dataclasses import dataclass

from app.models import DocumentRole, Role, SourceCitation
from app.retrieval import SearchResult
from app.retrieval_utils import tokenize


ROLE_KEYWORDS: dict[DocumentRole, set[str]] = {
    "regulator": {"监管", "执法", "检查", "抽检", "处罚", "立案", "整改", "召回监督", "市场监督", "部门"},
    "consumer": {"消费者", "赔偿", "投诉", "举报", "维权", "标签", "过期", "买到", "退货", "索赔"},
    "producer": {
        "企业",
        "生产",
        "经营",
        "合规",
        "许可",
        "进货查验",
        "记录",
        "召回",
        "标签标识",
        "自查",
        "预包装",
        "疾病",
        "治疗功能",
    },
    "general": {"食品安全", "法规", "标准", "条款", "要求"},
}


ROLE_PROMPTS: dict[DocumentRole, str] = {
    "regulator": "你是食品安全监管辅助智能体，侧重监督检查、风险处置、证据固定和依法履职。",
    "consumer": "你是食品安全消费者咨询智能体，侧重投诉举报、赔偿救济、标签识别和消费风险提示。",
    "producer": "你是食品生产经营合规智能体，侧重许可资质、过程控制、记录留存、召回和标签合规。",
    "general": "你是食品安全法律法规问答智能体，侧重基于资料给出中立、可核验的法规信息。",
}


@dataclass
class RouteDecision:
    role: DocumentRole
    reason: str
    scores: dict[str, int]
    manager_agent: str = "管理智能体"

    def as_dict(self) -> dict:
        return {
            "manager_agent": self.manager_agent,
            "role_agent": f"{self.role}_agent",
            "role": self.role,
            "reason": self.reason,
            "scores": self.scores,
        }


def route_question(question: str, requested_role: Role) -> RouteDecision:
    if requested_role != "auto":
        return RouteDecision(role=requested_role, reason="用户指定角色", scores={requested_role: 1})

    scores: dict[str, int] = {}
    for role, keywords in ROLE_KEYWORDS.items():
        scores[role] = sum(1 for keyword in keywords if keyword in question)
    best_role = max(scores, key=scores.get)
    if scores[best_role] == 0:
        return RouteDecision(role="general", reason="未命中特定角色关键词，使用通用咨询智能体", scores=scores)
    return RouteDecision(role=best_role, reason=f"命中 {best_role} 场景关键词", scores=scores)


def build_llm_messages(question: str, role: DocumentRole, results: list[SearchResult]) -> list[dict[str, str]]:
    context = "\n\n".join(
        f"[{index}] 标题：{item.chunk.title}\n来源：{item.chunk.source}\n片段：{item.chunk.text}"
        for index, item in enumerate(results, start=1)
    )
    system = (
        f"{ROLE_PROMPTS[role]}\n"
        "只能依据给定资料回答。回答要结构清晰、标注引用编号；资料不足时要直接说明。"
        "不要声称替代律师、监管机关或法院意见。"
    )
    user = f"问题：{question}\n\n可用资料：\n{context}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def confidence_from_results(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    top = results[0].score
    support = min(0.18, 0.04 * max(0, len(results) - 1))
    return round(min(1.0, top + support), 2)


def to_citations(results: list[SearchResult]) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    for index, result in enumerate(results, start=1):
        citations.append(
            SourceCitation(
                index=index,
                chunk_id=result.chunk.id,
                document_id=result.chunk.document_id,
                title=result.chunk.title,
                source=result.chunk.source,
                role=result.chunk.role,
                excerpt=make_excerpt(result.chunk.text),
                score=result.score,
            )
        )
    return citations


def fallback_answer(question: str, role: DocumentRole, results: list[SearchResult], confidence: float) -> str:
    if not results or confidence < 0.08:
        return (
            "未在当前知识库中检索到足够直接的法规依据，无法给出精确结论。"
            "建议补充相关法律法规、监管文件或案件资料后再查询。"
        )

    selected_sentences = select_relevant_sentences(question, results)
    if not selected_sentences:
        selected_sentences = [make_excerpt(item.chunk.text, 160) for item in results[:3]]

    lines = ["基于当前知识库，可作如下检索式回答："]
    for index, sentence in enumerate(selected_sentences[:4], start=1):
        citation_index = min(index, len(results))
        lines.append(f"{index}. {sentence} [{citation_index}]")
    lines.append("以上为法规资料检索辅助结果，具体处理仍应结合完整原文、事实证据和主管部门要求。")
    return "\n".join(lines)


def select_relevant_sentences(question: str, results: list[SearchResult]) -> list[str]:
    query_tokens = set(tokenize(question))
    sentences: list[tuple[int, str, int]] = []
    for result_index, result in enumerate(results, start=1):
        for sentence in split_sentences(result.chunk.text):
            overlap = len(query_tokens.intersection(tokenize(sentence)))
            if overlap > 0:
                sentences.append((overlap, sentence, result_index))
    sentences.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence, _ in sentences:
        compact = re.sub(r"\s+", "", sentence)
        if compact in seen:
            continue
        seen.add(compact)
        selected.append(sentence)
        if len(selected) >= 4:
            break
    return selected


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[。！？；!?;])\s*|\n+", text) if part.strip()]


def make_excerpt(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"
