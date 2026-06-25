from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion import chunk_text
from app.models import DocumentMeta, DocumentRole, KnowledgeChunk
from app.sample_corpus import SAMPLE_DOCUMENTS


class KnowledgeStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.upload_dir = data_dir / "uploads"
        self.documents_path = data_dir / "documents.json"
        self.chunks_path = data_dir / "chunks.json"
        self._lock = threading.RLock()

    def ensure_ready(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        if not self.documents_path.exists():
            self._write_json(self.documents_path, [])
        if not self.chunks_path.exists():
            self._write_json(self.chunks_path, [])

    def seed_if_empty(self) -> None:
        with self._lock:
            self.ensure_ready()
            if self.load_documents():
                return
            documents: list[DocumentMeta] = []
            chunks: list[KnowledgeChunk] = []
            for sample in SAMPLE_DOCUMENTS:
                doc_id = f"sample-{uuid.uuid4().hex[:8]}"
                pieces = chunk_text(sample["text"])
                documents.append(
                    DocumentMeta(
                        id=doc_id,
                        title=sample["title"],
                        role=sample["role"],
                        source=sample["source"],
                        filename=sample["filename"],
                        content_type="text/plain",
                        created_at=datetime.now(timezone.utc),
                        chunk_count=len(pieces),
                    )
                )
                chunks.extend(
                    KnowledgeChunk(
                        id=f"{doc_id}-{index}",
                        document_id=doc_id,
                        chunk_index=index,
                        title=sample["title"],
                        role=sample["role"],
                        source=sample["source"],
                        text=piece,
                    )
                    for index, piece in enumerate(pieces)
                )
            self.save_documents(documents)
            self.save_chunks(chunks)

    def load_documents(self) -> list[DocumentMeta]:
        self.ensure_ready()
        raw = self._read_json(self.documents_path)
        return [DocumentMeta.model_validate(item) for item in raw]

    def load_chunks(self) -> list[KnowledgeChunk]:
        self.ensure_ready()
        raw = self._read_json(self.chunks_path)
        return [KnowledgeChunk.model_validate(item) for item in raw]

    def save_documents(self, documents: list[DocumentMeta]) -> None:
        self._write_json(self.documents_path, [doc.model_dump(mode="json") for doc in documents])

    def save_chunks(self, chunks: list[KnowledgeChunk]) -> None:
        self._write_json(self.chunks_path, [chunk.model_dump(mode="json") for chunk in chunks])

    def add_document(
        self,
        *,
        title: str,
        role: DocumentRole,
        source: str,
        filename: str,
        content_type: str,
        text_chunks: list[str],
        original_content: bytes | None = None,
    ) -> DocumentMeta:
        with self._lock:
            documents = self.load_documents()
            chunks = self.load_chunks()
            doc_id = uuid.uuid4().hex
            clean_title = title.strip() or Path(filename).stem or "未命名资料"
            clean_source = source.strip() or filename
            document = DocumentMeta(
                id=doc_id,
                title=clean_title,
                role=role,
                source=clean_source,
                filename=filename,
                content_type=content_type,
                created_at=datetime.now(timezone.utc),
                chunk_count=len(text_chunks),
            )
            documents.append(document)
            chunks.extend(
                KnowledgeChunk(
                    id=f"{doc_id}-{index}",
                    document_id=doc_id,
                    chunk_index=index,
                    title=clean_title,
                    role=role,
                    source=clean_source,
                    text=text,
                )
                for index, text in enumerate(text_chunks)
            )
            if original_content is not None:
                safe_name = Path(filename).name or f"{doc_id}.txt"
                (self.upload_dir / f"{doc_id}-{safe_name}").write_bytes(original_content)
            self.save_documents(documents)
            self.save_chunks(chunks)
            return document

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            documents = self.load_documents()
            chunks = self.load_chunks()
            next_documents = [doc for doc in documents if doc.id != document_id]
            if len(next_documents) == len(documents):
                return False
            next_chunks = [chunk for chunk in chunks if chunk.document_id != document_id]
            self.save_documents(next_documents)
            self.save_chunks(next_chunks)
            for upload in self.upload_dir.glob(f"{document_id}-*"):
                if upload.is_file():
                    upload.unlink()
                elif upload.is_dir():
                    shutil.rmtree(upload)
            return True

    def stats_by_role(self) -> dict[str, int]:
        roles = {"regulator": 0, "consumer": 0, "producer": 0, "general": 0}
        for chunk in self.load_chunks():
            roles[chunk.role] += 1
        return roles

    @staticmethod
    def _read_json(path: Path) -> list[dict]:
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return json.loads(content)

    @staticmethod
    def _write_json(path: Path, payload: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
