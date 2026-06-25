from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass

import requests

from app.config import Settings


@dataclass
class ReadinessCheck:
    name: str
    status: str
    detail: str
    required_for_production_claim: bool = True


def check_readiness(settings: Settings, timeout: float = 3.0) -> dict:
    checks = [
        check_llm_service(settings, timeout),
        check_embedding_service(settings, timeout),
        check_reranker_service(settings, timeout),
        check_chroma(settings),
        check_langgraph(settings),
    ]
    production_ready = all(
        item.status == "ready" for item in checks if item.required_for_production_claim
    )
    return {
        "production_ready": production_ready,
        "checks": [asdict(item) for item in checks],
    }


def check_llm_service(settings: Settings, timeout: float) -> ReadinessCheck:
    if not settings.openai_api_key or not settings.openai_base_url:
        return ReadinessCheck("llm_service", "not_configured", "OPENAI_API_KEY 或 OPENAI_BASE_URL 未配置")
    return check_openai_compatible_models(
        name="llm_service",
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        timeout=timeout,
        detail_prefix=f"QA_MODEL={settings.qa_model}",
    )


def check_embedding_service(settings: Settings, timeout: float) -> ReadinessCheck:
    provider = settings.embedding_provider.lower()
    if provider in {"", "local"}:
        return ReadinessCheck("embedding_service", "local_fallback", "当前使用本地 embedding fallback")
    if not settings.embedding_api_key or not settings.embedding_base_url:
        return ReadinessCheck("embedding_service", "not_configured", "EMBEDDING_API_KEY 或 EMBEDDING_BASE_URL 未配置")
    return check_openai_compatible_models(
        name="embedding_service",
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        timeout=timeout,
        detail_prefix=f"EMBEDDING_MODEL={settings.embedding_model}",
    )


def check_reranker_service(settings: Settings, timeout: float) -> ReadinessCheck:
    provider = settings.reranker_provider.lower()
    if provider in {"", "local"}:
        return ReadinessCheck("reranker_service", "local_fallback", "当前使用本地 reranker fallback")
    if not settings.reranker_url:
        return ReadinessCheck("reranker_service", "not_configured", "RERANKER_URL 未配置")
    try:
        response = requests.post(
            settings.reranker_url,
            json={
                "model": settings.reranker_model,
                "query": "食品召回",
                "documents": ["食品生产经营者应当依法召回问题食品。"],
                "top_k": 1,
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        return ReadinessCheck("reranker_service", "unavailable", f"reranker 请求失败：{exc}")
    return ReadinessCheck("reranker_service", "ready", f"reranker 服务可用：{settings.reranker_model}")


def check_chroma(settings: Settings) -> ReadinessCheck:
    if settings.vector_backend.lower() != "chroma":
        return ReadinessCheck("chroma", "local_fallback", f"当前 VECTOR_BACKEND={settings.vector_backend}")
    if importlib.util.find_spec("chromadb") is None:
        return ReadinessCheck("chroma", "unavailable", "chromadb 未安装")
    return ReadinessCheck("chroma", "ready", "chromadb 已安装且 VECTOR_BACKEND=chroma")


def check_langgraph(settings: Settings) -> ReadinessCheck:
    if settings.workflow_backend.lower() not in {"langgraph", "auto"}:
        return ReadinessCheck("langgraph", "local_fallback", f"当前 WORKFLOW_BACKEND={settings.workflow_backend}")
    if importlib.util.find_spec("langgraph") is None:
        return ReadinessCheck("langgraph", "unavailable", "langgraph 未安装")
    return ReadinessCheck("langgraph", "ready", "langgraph 已安装")


def check_openai_compatible_models(
    *,
    name: str,
    base_url: str,
    api_key: str,
    timeout: float,
    detail_prefix: str,
) -> ReadinessCheck:
    try:
        response = requests.get(
            base_url.rstrip("/") + "/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        return ReadinessCheck(name, "unavailable", f"{detail_prefix}；/models 请求失败：{exc}")
    return ReadinessCheck(name, "ready", f"{detail_prefix}；OpenAI-compatible /models 可访问")
