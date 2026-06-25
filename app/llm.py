from __future__ import annotations

from app.agents import build_llm_messages
from app.config import Settings
from app.models import DocumentRole
from app.retrieval import SearchResult


async def generate_with_model(
    *,
    settings: Settings,
    question: str,
    role: DocumentRole,
    results: list[SearchResult],
) -> str | None:
    if not settings.model_configured:
        return None
    try:
        from openai import AsyncOpenAI
    except Exception:
        return None
    client_kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        client_kwargs["base_url"] = settings.openai_base_url
    client = AsyncOpenAI(**client_kwargs)
    try:
        completion = await client.chat.completions.create(
            model=settings.qa_model,
            messages=build_llm_messages(question, role, results),
            temperature=0.2,
            max_tokens=900,
        )
    except Exception:
        return None
    message = completion.choices[0].message.content if completion.choices else ""
    return message.strip() if message else None
