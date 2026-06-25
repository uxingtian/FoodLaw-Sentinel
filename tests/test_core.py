from app.agents import confidence_from_results, fallback_answer, route_question
from app.ingestion import chunk_text
from app.models import KnowledgeChunk
from app.retrieval import HybridRetriever


def test_legal_knowledge_graph_extracts_and_searches_edges():
    from app.graph_store import LegalKnowledgeGraph

    chunks = [
        KnowledgeChunk(
            id="graph-c1",
            document_id="graph-d1",
            chunk_index=0,
            title="企业召回义务",
            role="producer",
            source="测试",
            text="食品生产经营者发现食品可能危害人体健康的，应当停止生产经营，通知消费者，并召回已上市食品。",
        )
    ]
    graph = LegalKnowledgeGraph()
    graph.rebuild(chunks)
    stats = graph.stats()
    assert stats["edges"] > 0
    hits = graph.search("生产经营者如何召回食品？", role="producer", top_k=3)
    assert hits
    assert hits[0].chunk_id == "graph-c1"
    graph.search("生产经营者如何召回食品？", role="producer", top_k=3)
    assert graph.stats()["cache_entries"] == 1


def test_agent_workflow_exposes_trace_and_reranker():
    import asyncio

    from app.config import settings
    from app.models import ChatRequest
    from app.reranker import build_reranker
    from app.workflow import AgentWorkflow

    chunks = [
        KnowledgeChunk(
            id="workflow-c1",
            document_id="workflow-d1",
            chunk_index=0,
            title="消费者赔偿",
            role="consumer",
            source="测试",
            text="消费者购买到不符合食品安全标准的食品，可以依法要求赔偿并投诉举报。",
        )
    ]
    retriever = HybridRetriever()
    retriever.rebuild(chunks)
    workflow = AgentWorkflow(settings=settings, retriever=retriever, reranker=build_reranker("local", "bge-reranker"))
    response = asyncio.run(workflow.answer(ChatRequest(question="消费者如何赔偿？", role="auto", top_k=1)))
    assert response.role == "consumer"
    assert response.route["workflow"] == "local-agent-graph"
    assert response.route["reranker"] == "bge-reranker"
    assert response.route["tools"]
    assert response.route["query"]["rewritten"] != response.route["query"]["original"]
    assert response.route["trace"]["tool_ms"] >= 0
    assert response.route["trace"]["total_ms"] >= 0


def test_agent_workflow_rejects_ungrounded_model_answer(monkeypatch):
    import asyncio
    from dataclasses import replace

    from app.config import settings
    from app.models import ChatRequest
    from app.workflow import AgentWorkflow

    chunks = [
        KnowledgeChunk(
            id="guard-c1",
            document_id="guard-d1",
            chunk_index=0,
            title="消费者索赔",
            role="consumer",
            source="测试",
            text="消费者购买到不符合食品安全标准的食品，可以依法要求赔偿并投诉举报。",
        )
    ]
    retriever = HybridRetriever()
    retriever.rebuild(chunks)

    async def ungrounded_answer(**_kwargs):
        return "消费者可以直接获得任意金额赔偿。"

    monkeypatch.setattr("app.workflow.generate_with_model", ungrounded_answer)
    workflow = AgentWorkflow(settings=replace(settings, openai_api_key="fake"), retriever=retriever)
    response = asyncio.run(workflow.answer(ChatRequest(question="消费者如何索赔？", role="auto", top_k=1)))

    assert response.fallback_used is True
    assert response.route["generation_guard"]["accepted"] is False
    assert "missing_citation" in response.route["generation_guard"]["violations"]
    assert "[1]" in response.answer


def test_demo_scenarios_cover_three_role_agents():
    import asyncio

    from app.config import settings
    from app.demo import DEFAULT_DEMO_SCENARIOS, run_demo_scenarios
    from app.main import startup
    import app.main as main

    startup()
    report = asyncio.run(run_demo_scenarios(main.workflow, DEFAULT_DEMO_SCENARIOS, top_k=settings.default_top_k))
    roles = {item["expected_role"] for item in report["items"]}
    assert {"regulator", "consumer", "producer"}.issubset(roles)
    assert report["summary"]["passed"] is True
    assert report["summary"]["with_sources"] == len(report["items"])
    assert report["summary"]["with_generation_guard"] == len(report["items"])


def test_demo_report_markdown_contains_role_evidence():
    from scripts.generate_demo_report import generate_demo_report

    report = {
        "summary": {
            "total": 3,
            "role_passed": 3,
            "with_sources": 3,
            "with_generation_guard": 3,
            "passed": True,
        },
        "items": [
            {
                "title": "监管抽检处置",
                "question": "监管部门抽检发现问题食品怎么办？",
                "expected_role": "regulator",
                "actual_role": "regulator",
                "role_ok": True,
                "sources": 2,
                "generation_guard": {"accepted": False, "violations": ["model_not_used"]},
                "answer_preview": "应当依法调查处置并固定证据。[1]",
            },
            {
                "title": "消费者维权索赔",
                "question": "消费者如何索赔？",
                "expected_role": "consumer",
                "actual_role": "consumer",
                "role_ok": True,
                "sources": 2,
                "generation_guard": {"accepted": False, "violations": ["model_not_used"]},
                "answer_preview": "可以投诉举报并依法要求赔偿。[1]",
            },
            {
                "title": "生产经营者合规",
                "question": "企业如何召回？",
                "expected_role": "producer",
                "actual_role": "producer",
                "role_ok": True,
                "sources": 2,
                "generation_guard": {"accepted": False, "violations": ["model_not_used"]},
                "answer_preview": "应当停止经营、通知消费者并召回。[1]",
            },
        ],
    }

    markdown = generate_demo_report(report)
    assert "# 食品安全法律法规问答系统演示报告" in markdown
    assert "监管抽检处置" in markdown
    assert "消费者维权索赔" in markdown
    assert "生产经营者合规" in markdown
    assert "来源数量：2" in markdown
    assert "护栏状态" in markdown


def test_acceptance_audit_summarizes_resume_alignment(tmp_path):
    from scripts.generate_acceptance_audit import build_acceptance_audit, generate_acceptance_markdown

    report_summary = {
        "available": True,
        "summary": {
            "claims": {"passed": True, "supported_count": 5, "unsupported_count": 4},
            "demo": {"passed": True, "total": 3, "role_passed": 3},
            "benchmark": {"success_rate": 1.0, "p95_ms": 800, "concurrency": 10},
        },
        "alignment": {
            "summary": {"total": 9, "verified": 5, "pending_external_service": 3, "pending_evidence": 1},
            "items": [
                {"id": "source_cited_rag", "label": "来源引用 RAG", "status": "verified"},
                {"id": "qwen_vllm_inference", "label": "Qwen/vLLM 推理接入", "status": "pending_external_service"},
                {"id": "chroma_langgraph_stack", "label": "Chroma/LangGraph 生产栈", "status": "pending_external_service"},
                {"id": "high_concurrency_claim", "label": "1000+ 并发声明", "status": "pending_evidence"},
            ],
        },
    }

    audit = build_acceptance_audit(report_summary)
    markdown = generate_acceptance_markdown(audit)
    assert audit["status"] == "partially_verified"
    assert audit["summary"]["verified"] == 5
    pending_by_id = {item["id"]: item for item in audit["pending_items"]}
    assert "OPENAI_BASE_URL" in pending_by_id["qwen_vllm_inference"]["next_step"]["env"]
    assert "scripts/check_readiness.py" in pending_by_id["qwen_vllm_inference"]["next_step"]["verify"]
    assert "VECTOR_BACKEND=chroma" in pending_by_id["chroma_langgraph_stack"]["next_step"]["env"]
    assert "scripts/benchmark.py" in pending_by_id["high_concurrency_claim"]["next_step"]["verify"]
    assert "Qwen/vLLM 推理接入" in markdown
    assert "1000+ 并发声明" in markdown
    assert "OPENAI_BASE_URL" in markdown
    assert "partially_verified" in markdown


def test_acceptance_gate_distinguishes_demo_and_production_modes():
    from scripts.acceptance_gate import evaluate_gate

    audit = {
        "status": "partially_verified",
        "summary": {
            "total": 9,
            "verified": 5,
            "pending_external_service": 3,
            "pending_evidence": 1,
            "claims_passed": True,
            "demo_passed": True,
        },
    }

    demo_result = evaluate_gate(audit, mode="demo")
    production_result = evaluate_gate(audit, mode="production")
    assert demo_result["passed"] is True
    assert production_result["passed"] is False
    assert "pending_external_service" in production_result["reasons"]


def test_doctor_report_includes_runtime_reports_and_gates(tmp_path):
    from scripts.doctor import build_doctor_report

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "acceptance_audit.json").write_text(
        '{"status":"partially_verified","summary":{"total":9,"verified":5,"pending_external_service":3,"pending_evidence":1,"claims_passed":true,"demo_passed":true}}',
        encoding="utf-8",
    )
    (reports_dir / "demo_report.md").write_text("# demo\n", encoding="utf-8")

    report = build_doctor_report(reports_dir=reports_dir)
    assert report["runtime"]["python"]
    assert report["config"]["vector_backend"]
    assert report["reports"]["acceptance_audit.json"]["available"] is True
    assert report["reports"]["demo_report.md"]["available"] is True
    assert report["gates"]["demo"]["passed"] is True
    assert report["gates"]["production"]["passed"] is False


def test_evidence_package_includes_only_expected_artifacts(tmp_path):
    import json
    import zipfile

    from scripts.package_evidence import build_evidence_package

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    root_dir = tmp_path / "project"
    root_dir.mkdir()
    (root_dir / "docs").mkdir()
    for name in ["demo_report.md", "resume_summary.md", "acceptance_audit.md", "doctor.json"]:
        (reports_dir / name).write_text(name, encoding="utf-8")
    (root_dir / ".env.production.example").write_text("OPENAI_BASE_URL=http://example/v1\n", encoding="utf-8")
    (root_dir / "docs" / "production_runbook.md").write_text("python scripts/acceptance_gate.py --mode production\n", encoding="utf-8")
    (reports_dir / "secret.txt").write_text("do not include", encoding="utf-8")
    output = tmp_path / "dist" / "evidence.zip"

    package_path = build_evidence_package(reports_dir=reports_dir, output_path=output, root_dir=root_dir)

    assert package_path == output
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
    assert "reports/demo_report.md" in names
    assert "reports/resume_summary.md" in names
    assert "reports/acceptance_audit.md" in names
    assert "reports/doctor.json" in names
    assert ".env.production.example" in names
    assert "docs/production_runbook.md" in names
    assert manifest["files"]["reports/demo_report.md"]["size_bytes"] == len("demo_report.md".encode("utf-8"))
    assert len(manifest["files"]["reports/demo_report.md"]["sha256"]) == 64
    assert ".env.production.example" in manifest["files"]
    assert "docs/production_runbook.md" in manifest["files"]
    assert "reports/secret.txt" not in names


def test_evidence_package_verifier_detects_hash_mismatch(tmp_path):
    from scripts.package_evidence import build_evidence_package
    from scripts.verify_evidence_package import verify_evidence_package

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "demo_report.md").write_text("original", encoding="utf-8")
    output = tmp_path / "evidence.zip"
    build_evidence_package(reports_dir=reports_dir, output_path=output)

    assert verify_evidence_package(output)["passed"] is True

    import zipfile

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(tampered, "w") as target:
        for name in source.namelist():
            if name == "reports/demo_report.md":
                target.writestr(name, "tampered")
            else:
                target.writestr(name, source.read(name))
    output = tampered
    result = verify_evidence_package(output)
    assert result["passed"] is False
    assert result["files"]["reports/demo_report.md"]["status"] == "hash_mismatch"


def test_production_template_and_runbook_cover_remaining_gaps():
    from pathlib import Path

    env_text = Path(".env.production.example").read_text(encoding="utf-8")
    runbook = Path("docs/production_runbook.md").read_text(encoding="utf-8")
    for token in [
        "OPENAI_BASE_URL",
        "QA_MODEL=Qwen3-72B-Chat",
        "VECTOR_BACKEND=chroma",
        "WORKFLOW_BACKEND=langgraph",
        "RERANKER_PROVIDER=http",
        "RERANKER_URL",
    ]:
        assert token in env_text
    for token in [
        "python scripts/check_readiness.py",
        "python scripts/run_evidence_pipeline.py",
        "python scripts/acceptance_gate.py --mode production",
        "--concurrency 1000",
    ]:
        assert token in runbook


def test_delivery_builder_runs_pipeline_gate_package_and_verify(tmp_path):
    from scripts.build_delivery import build_delivery

    calls = []

    def fake_pipeline(**kwargs):
        calls.append(("pipeline", kwargs["reports_dir"]))
        return {"passed": True, "acceptance_audit": str(kwargs["reports_dir"] / "acceptance_audit.md")}

    def fake_gate(audit, *, mode):
        calls.append(("gate", mode))
        return {"passed": True, "mode": mode, "reasons": []}

    def fake_package(**kwargs):
        calls.append(("package", kwargs["output_path"]))
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_path"].write_bytes(b"zip")
        return kwargs["output_path"]

    def fake_verify(path):
        calls.append(("verify", path))
        return {"passed": True, "files": {}}

    result = build_delivery(
        reports_dir=tmp_path / "reports",
        package_path=tmp_path / "dist" / "evidence.zip",
        run_pipeline_fn=fake_pipeline,
        evaluate_gate_fn=fake_gate,
        package_fn=fake_package,
        verify_package_fn=fake_verify,
    )

    assert result["passed"] is True
    assert result["package"] == str(tmp_path / "dist" / "evidence.zip")
    assert [item[0] for item in calls] == ["pipeline", "gate", "package", "verify"]


def test_dense_embedding_vector_store_can_retrieve_without_external_service():
    from app.embedding import LocalHashEmbeddingClient
    from app.vector_store import DenseEmbeddingVectorStore

    chunks = [
        KnowledgeChunk(
            id="dense-c1",
            document_id="dense-d1",
            chunk_index=0,
            title="进货查验",
            role="producer",
            source="测试",
            text="食品生产经营者应当建立进货查验记录制度。",
        ),
        KnowledgeChunk(
            id="dense-c2",
            document_id="dense-d2",
            chunk_index=0,
            title="投诉举报",
            role="consumer",
            source="测试",
            text="消费者发现食品安全问题可以投诉举报。",
        ),
    ]
    store = DenseEmbeddingVectorStore(LocalHashEmbeddingClient())
    retriever = HybridRetriever(store)
    retriever.rebuild(chunks)
    results = retriever.search("企业进货查验记录怎么做？", role="producer", top_k=1)
    assert results
    assert results[0].chunk.id == "dense-c1"


def test_parse_reranker_scores_accepts_common_shapes():
    from app.reranker import parse_reranker_scores

    assert parse_reranker_scores({"scores": [0.2, 0.8]}, 2) == [0.2, 0.8]
    assert parse_reranker_scores({"results": [{"index": 1, "score": 0.3}, {"index": 0, "score": 0.9}]}, 2) == [
        0.9,
        0.3,
    ]


def test_metrics_recorder_tracks_chat_response():
    from app.metrics import MetricsRecorder
    from app.models import ChatResponse

    recorder = MetricsRecorder()
    recorder.record_chat(
        ChatResponse(
            answer="ok",
            role="consumer",
            confidence=0.9,
            sources=[],
            fallback_used=True,
            route={
                "trace": {"total_ms": 12.5},
                "tools": [{"name": "query_rewrite"}, {"name": "compliance_checklist"}],
            },
        )
    )
    snapshot = recorder.snapshot()
    assert snapshot["total_requests"] == 1
    assert snapshot["fallback_requests"] == 1
    assert snapshot["role_counts"]["consumer"] == 1
    assert snapshot["tool_counts"]["query_rewrite"] == 1


def test_claim_verifier_separates_supported_and_unsupported_claims():
    from scripts.verify_claims import verify

    eval_report = {
        "role_accuracy": 0.95,
        "answer_keyword_recall": 0.92,
        "source_keyword_recall": 0.93,
        "tool_usage": {
            "query_rewrite": 3,
            "compliance_checklist": 3,
            "knowledge_graph_lookup": 3,
        },
        "graph": {"edges": 12, "subjects": 3, "objects": 5},
        "grounding_guard_coverage": 1.0,
        "config": {
            "model_configured": False,
            "workflow": "local-agent-graph",
            "vector_backend": "local-tfidf+local-legal-kg",
            "reranker_provider": "local",
        },
    }
    benchmark_report = {
        "concurrency": 10,
        "success_rate": 1.0,
        "latency_ms": {"p95": 800},
    }
    report = verify(eval_report, benchmark_report)
    assert report["passed"] is True
    assert any("来源引用召回" in claim for claim in report["supported_claims"])
    assert any("接地校验" in claim for claim in report["supported_claims"])
    assert any("1000+ 并发" in claim for claim in report["unsupported_claims"])


def test_readiness_reports_local_fallbacks_for_default_settings():
    from app.config import settings
    from app.readiness import check_readiness

    report = check_readiness(settings, timeout=0.1)
    checks = {item["name"]: item for item in report["checks"]}
    assert report["production_ready"] is False
    assert checks["embedding_service"]["status"] in {"local_fallback", "not_configured"}
    assert checks["reranker_service"]["status"] in {"local_fallback", "not_configured"}


def test_resume_summary_keeps_supported_and_unsupported_claims_separate():
    from scripts.generate_resume_summary import generate_summary

    report = {
        "supported_claims": ["来源引用召回 100%"],
        "unsupported_claims": ["当前报告未证明真实 Chroma 向量库已启用"],
        "readiness": {
            "production_ready": False,
            "checks": [{"name": "chroma", "status": "local_fallback", "detail": "当前 VECTOR_BACKEND=local"}],
        },
    }
    summary = generate_summary(report)
    assert "## 已由当前报告证明的成果" in summary
    assert "- 来源引用召回 100%" in summary
    assert "## 不建议直接写成已完成的生产成果" in summary
    assert "- 当前报告未证明真实 Chroma 向量库已启用" in summary
    assert "production_ready: false" in summary


def test_pipeline_writers_create_utf8_json_and_text(tmp_path):
    import json

    from scripts.run_evidence_pipeline import write_json, write_text

    json_path = tmp_path / "report.json"
    text_path = tmp_path / "summary.md"
    write_json(json_path, {"claim": "来源引用召回 100%"})
    write_text(text_path, "# 简历安全版描述\n")
    assert json.loads(json_path.read_text(encoding="utf-8"))["claim"] == "来源引用召回 100%"
    assert "简历安全版描述" in text_path.read_text(encoding="utf-8")


def test_report_summary_loads_evidence_dashboard_data(tmp_path):
    import json

    from app.reports import load_report_summary

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "eval.json").write_text(
        json.dumps(
            {
                "cases": 3,
                "role_accuracy": 1.0,
                "answer_keyword_recall": 0.95,
                "source_keyword_recall": 0.9,
                "avg_latency_ms": 12.34,
                "tool_usage": {"query_rewrite": 3},
                "graph": {"edges": 8},
            }
        ),
        encoding="utf-8",
    )
    (reports_dir / "benchmark.json").write_text(
        json.dumps({"requests": 20, "concurrency": 5, "success_rate": 1.0, "throughput_qps": 33.3, "latency_ms": {"p95": 120.5}}),
        encoding="utf-8",
    )
    (reports_dir / "claim_verification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "supported_claims": ["role routing >= 90%"],
                "unsupported_claims": ["1000+ concurrent users"],
                "checks": [{"name": "role_accuracy", "passed": True, "observed": 1.0, "threshold": 0.9}],
                "readiness": {"production_ready": False, "checks": [{"name": "llm_service", "status": "not_configured"}]},
            }
        ),
        encoding="utf-8",
    )

    summary = load_report_summary(reports_dir)
    assert summary["available"] is True
    assert summary["summary"]["eval"]["graph_edges"] == 8
    assert summary["summary"]["benchmark"]["p95_ms"] == 120.5
    assert summary["summary"]["claims"]["supported_count"] == 1
    assert summary["summary"]["claims"]["unsupported_count"] == 1
    assert summary["summary"]["readiness"]["production_ready"] is False


def test_resume_alignment_marks_verified_and_pending_requirements():
    from app.alignment import build_resume_alignment

    claim_report = {
        "checks": [
            {"name": "role_accuracy", "passed": True, "observed": 0.96, "threshold": 0.9},
            {"name": "source_keyword_recall", "passed": True, "observed": 0.94, "threshold": 0.9},
            {"name": "benchmark_p95_ms", "passed": True, "observed": 800, "threshold": 5000},
            {"name": "required_agent_tools", "passed": True, "observed": {"query_rewrite": 4, "compliance_checklist": 4, "knowledge_graph_lookup": 4}},
            {"name": "knowledge_graph_edges", "passed": True, "observed": 12, "threshold": 1},
        ],
        "readiness": {
            "production_ready": False,
            "checks": [
                {"name": "llm_service", "status": "not_configured", "detail": "missing model"},
                {"name": "chroma", "status": "local_fallback", "detail": "local backend"},
                {"name": "langgraph", "status": "unavailable", "detail": "not installed"},
            ],
        },
    }
    alignment = build_resume_alignment(claim_report)
    by_id = {item["id"]: item for item in alignment["items"]}
    assert by_id["multi_agent_workflow"]["status"] == "verified"
    assert by_id["source_cited_rag"]["status"] == "verified"
    assert by_id["qwen_vllm_inference"]["status"] == "pending_external_service"
    assert by_id["chroma_langgraph_stack"]["status"] == "pending_external_service"
    assert alignment["summary"]["verified"] >= 3
    assert alignment["summary"]["pending_external_service"] >= 2


def test_chunk_text_respects_size_window():
    text = "食品生产经营者应当建立食品安全管理制度。" * 120
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 900 for chunk in chunks)


def test_route_question_detects_roles():
    assert route_question("消费者买到过期食品如何投诉和索赔？", "auto").role == "consumer"
    assert route_question("监管部门抽检发现问题应如何处置？", "auto").role == "regulator"
    assert route_question("食品生产企业如何建立进货查验记录？", "auto").role == "producer"
    assert route_question("食品安全标准有什么要求？", "auto").role == "general"


def test_hybrid_retriever_returns_relevant_chunk():
    chunks = [
        KnowledgeChunk(
            id="c1",
            document_id="d1",
            chunk_index=0,
            title="消费者权益",
            role="consumer",
            source="示例",
            text="消费者购买到不符合食品安全标准的食品，可以投诉举报并依法要求赔偿。",
        ),
        KnowledgeChunk(
            id="c2",
            document_id="d2",
            chunk_index=0,
            title="企业合规",
            role="producer",
            source="示例",
            text="食品生产企业应当建立进货查验记录和食品安全自查制度。",
        ),
    ]
    retriever = HybridRetriever()
    retriever.rebuild(chunks)
    results = retriever.search("买到不符合食品安全标准的食品怎么赔偿？", role="consumer", top_k=2)
    assert results
    assert results[0].chunk.id == "c1"
    assert results[0].score > 0


def test_fallback_answer_handles_no_evidence():
    answer = fallback_answer("完全无关的问题", "general", [], confidence=0)
    assert "未在当前知识库中检索到足够直接的法规依据" in answer


def test_fallback_answer_includes_citations_when_supported():
    chunks = [
        KnowledgeChunk(
            id="c1",
            document_id="d1",
            chunk_index=0,
            title="召回义务",
            role="producer",
            source="示例",
            text="发现食品可能危害人体健康的，应当立即停止生产经营，通知消费者，并召回已经上市销售的食品。",
        )
    ]
    retriever = HybridRetriever()
    retriever.rebuild(chunks)
    results = retriever.search("食品召回义务是什么？", role="producer", top_k=1)
    confidence = confidence_from_results(results)
    answer = fallback_answer("食品召回义务是什么？", "producer", results, confidence)
    assert "[1]" in answer
    assert "召回" in answer
