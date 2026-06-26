from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.demo import DEFAULT_DEMO_SCENARIOS, run_demo_scenarios
from app.embedding import build_embedding_client
from app.ingestion import UnsupportedDocumentError, chunk_text, extract_text
from app.metrics import metrics_recorder
from app.readiness import check_readiness
from app.models import ChatRequest, ChatResponse, DocumentMeta, DocumentRole, HealthResponse, StatsResponse
from app.reranker import build_reranker
from app.reports import load_report_summary, resolve_report_artifact
from app.retrieval import HybridRetriever
from app.storage import KnowledgeStore
from app.tools import ComplianceChecklistTool, GraphLookupTool, QueryRewriteTool, ToolRegistry
from app.vector_store import build_vector_store
from app.workflow import build_workflow


app = FastAPI(title="食品安全法律法规多智能体问答系统", version="0.3.0")
store = KnowledgeStore(settings.data_dir)
embedding_client = build_embedding_client(
    provider=settings.embedding_provider,
    model=settings.embedding_model,
    api_key=settings.embedding_api_key,
    base_url=settings.embedding_base_url,
)
retriever = HybridRetriever(build_vector_store(settings.vector_backend, settings.vector_dir, embedding_client))
reranker = build_reranker(
    settings.reranker_provider,
    settings.reranker_model,
    settings.reranker_url,
    settings.reranker_api_key,
)
tool_registry = ToolRegistry([QueryRewriteTool(), ComplianceChecklistTool(), GraphLookupTool(retriever.graph)])
workflow = build_workflow(settings=settings, retriever=retriever, reranker=reranker, tool_registry=tool_registry)


def rebuild_index() -> None:
    retriever.rebuild(store.load_chunks())


def startup() -> None:
    global workflow
    store.seed_if_empty()
    rebuild_index()
    workflow = build_workflow(settings=settings, retriever=retriever, reranker=reranker, tool_registry=tool_registry)


@asynccontextmanager
async def lifespan(_: FastAPI):
    startup()
    yield


app.router.lifespan_context = lifespan


if settings.static_dir.exists():
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_path = settings.static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="前端页面尚未创建")
    return FileResponse(index_path)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    documents = store.load_documents()
    chunks = store.load_chunks()
    return HealthResponse(
        status="ok",
        model_configured=settings.model_configured,
        index_ready=retriever.ready,
        documents=len(documents),
        chunks=len(chunks),
        vector_backend=retriever.backend_name,
        embedding_provider=settings.embedding_provider,
        embedding_model=f"{settings.embedding_model} ({embedding_client.name})",
        reranker_provider=settings.reranker_provider,
        reranker_model=reranker.model_name if reranker else "none",
        workflow=workflow.backend_name,
    )


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    documents = store.load_documents()
    chunks = store.load_chunks()
    return StatsResponse(documents=len(documents), chunks=len(chunks), roles=store.stats_by_role())


@app.get("/api/metrics")
def metrics() -> dict:
    return metrics_recorder.snapshot()


@app.get("/api/readiness")
def readiness() -> dict:
    return check_readiness(settings)


@app.get("/api/reports")
def reports() -> dict:
    return load_report_summary(settings.reports_dir)


@app.get("/api/reports/artifacts/{artifact_name}")
def report_artifact(artifact_name: str) -> FileResponse:
    resolved = resolve_report_artifact(settings.reports_dir, artifact_name)
    if resolved is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    path, media_type = resolved
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/graph")
def graph() -> dict:
    return {"stats": retriever.graph.stats(), "sample": retriever.graph.sample(limit=20)}


@app.get("/api/demo/scenarios")
async def demo_scenarios() -> dict:
    return await run_demo_scenarios(workflow, DEFAULT_DEMO_SCENARIOS, top_k=settings.default_top_k)


@app.get("/api/documents", response_model=list[DocumentMeta])
def list_documents() -> list[DocumentMeta]:
    return sorted(store.load_documents(), key=lambda doc: doc.created_at, reverse=True)


@app.post("/api/documents", response_model=DocumentMeta)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(""),
    role: DocumentRole = Form("general"),
    source: str = Form(""),
) -> DocumentMeta:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    filename = Path(file.filename or "upload.txt").name
    try:
        text = extract_text(filename, content)
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pieces = chunk_text(text)
    if not pieces:
        raise HTTPException(status_code=400, detail="未能从文件中解析出有效文本")
    document = store.add_document(
        title=title,
        role=role,
        source=source,
        filename=filename,
        content_type=file.content_type or "",
        text_chunks=pieces,
        original_content=content,
    )
    rebuild_index()
    return document


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, bool]:
    deleted = store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="未找到文档")
    rebuild_index()
    return {"deleted": True}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    response = await workflow.answer(request)
    metrics_recorder.record_chat(response)
    return response
