from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load_env_file(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / os.getenv("DATA_DIR", "data")
    reports_dir: Path = ROOT_DIR / os.getenv("REPORTS_DIR", "reports")
    static_dir: Path = ROOT_DIR / "static"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    qa_model: str = os.getenv("QA_MODEL", "qwen-plus")
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    vector_backend: str = os.getenv("VECTOR_BACKEND", "local")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "qwen3-embedding-0.6b")
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
    reranker_provider: str = os.getenv("RERANKER_PROVIDER", "local")
    reranker_model: str = os.getenv("RERANKER_MODEL", "bge-reranker")
    reranker_url: str = os.getenv("RERANKER_URL", "")
    reranker_api_key: str = os.getenv("RERANKER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    workflow_backend: str = os.getenv("WORKFLOW_BACKEND", "auto")

    @property
    def model_configured(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vectors"


settings = Settings()
