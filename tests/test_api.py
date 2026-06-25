import asyncio
import json
from dataclasses import dataclass

import app.main as main
from app.retrieval import HybridRetriever
from app.storage import KnowledgeStore


@dataclass
class AsgiResponse:
    status_code: int
    body: bytes
    headers: dict[str, str]

    def json(self):
        return json.loads(self.body.decode("utf-8"))


def setup_app(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "store", KnowledgeStore(tmp_path / "data"))
    monkeypatch.setattr(main, "retriever", HybridRetriever())
    main.startup()


def test_health_and_seeded_corpus(monkeypatch, tmp_path):
    setup_app(monkeypatch, tmp_path)
    response = request("GET", "/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["documents"] == 3
    assert payload["chunks"] >= 3
    assert payload["index_ready"] is True
    graph = request("GET", "/api/graph")
    assert graph.status_code == 200
    graph_payload = graph.json()
    assert graph_payload["stats"]["edges"] > 0
    readiness = request("GET", "/api/readiness")
    assert readiness.status_code == 200
    readiness_payload = readiness.json()
    assert readiness_payload["production_ready"] is False
    reports = request("GET", "/api/reports")
    assert reports.status_code == 200
    reports_payload = reports.json()
    assert "summary" in reports_payload
    assert "claims" in reports_payload["summary"]
    assert "alignment" in reports_payload
    assert "items" in reports_payload["alignment"]
    assert "demo_report" in reports_payload
    assert "doctor" in reports_payload
    demo = request("GET", "/api/demo/scenarios")
    assert demo.status_code == 200
    demo_payload = demo.json()
    assert demo_payload["summary"]["passed"] is True
    assert {item["expected_role"] for item in demo_payload["items"]} >= {"regulator", "consumer", "producer"}
    artifact = request("GET", "/api/reports/artifacts/demo_report")
    assert artifact.status_code == 200
    assert "食品安全法律法规问答系统演示报告" in artifact.body.decode("utf-8")
    doctor = request("GET", "/api/reports/artifacts/doctor")
    assert doctor.status_code == 200
    assert "runtime" in doctor.json()
    blocked_artifact = request("GET", "/api/reports/artifacts/../../app/main.py")
    assert blocked_artifact.status_code == 404


def test_upload_chat_and_delete_document(monkeypatch, tmp_path):
    setup_app(monkeypatch, tmp_path)
    body, content_type = multipart_body(
        fields={"title": "消费者赔偿测试资料", "role": "consumer", "source": "测试来源"},
        files={
            "file": (
                "consumer.txt",
                "消费者购买到不符合食品安全标准的食品，可以依法要求赔偿并向监管部门投诉举报。".encode("utf-8"),
                "text/plain",
            )
        },
    )
    upload = request("POST", "/api/documents", body=body, headers={"content-type": content_type})
    assert upload.status_code == 200
    document = upload.json()
    assert document["chunk_count"] == 1

    chat = request_json(
        "POST",
        "/api/chat",
        {"question": "买到不符合食品安全标准的食品怎么赔偿？", "role": "auto", "top_k": 3},
    )
    assert chat.status_code == 200
    answer = chat.json()
    assert answer["role"] == "consumer"
    assert answer["sources"]
    assert answer["route"]["tools"]
    assert answer["route"]["query"]["rewritten"] != answer["route"]["query"]["original"]
    assert "赔偿" in answer["answer"]

    metrics = request("GET", "/api/metrics")
    assert metrics.status_code == 200
    metrics_payload = metrics.json()
    assert metrics_payload["total_requests"] >= 1
    assert metrics_payload["tool_counts"]["query_rewrite"] >= 1

    delete = request("DELETE", f"/api/documents/{document['id']}")
    assert delete.status_code == 200
    documents = request("GET", "/api/documents").json()
    assert all(item["id"] != document["id"] for item in documents)


def request_json(method: str, path: str, payload: dict) -> AsgiResponse:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return request(method, path, body=body, headers={"content-type": "application/json"})


def request(method: str, path: str, body: bytes = b"", headers: dict[str, str] | None = None) -> AsgiResponse:
    return asyncio.run(asgi_request(method, path, body, headers or {}))


async def asgi_request(method: str, path: str, body: bytes, headers: dict[str, str]) -> AsgiResponse:
    response_messages = []
    request_sent = False
    normalized_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in headers.items()]
    normalized_headers.append((b"content-length", str(len(body)).encode("latin-1")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": normalized_headers,
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }

    async def receive():
        nonlocal request_sent
        if request_sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        request_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        response_messages.append(message)

    await main.app(scope, receive, send)
    start = next(message for message in response_messages if message["type"] == "http.response.start")
    body_parts = [message.get("body", b"") for message in response_messages if message["type"] == "http.response.body"]
    response_headers = {key.decode("latin-1"): value.decode("latin-1") for key, value in start.get("headers", [])}
    return AsgiResponse(status_code=start["status"], body=b"".join(body_parts), headers=response_headers)


def multipart_body(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----codex-food-law-boundary"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(value.encode("utf-8"))
        chunks.append(b"\r\n")
    for name, (filename, content, content_type) in files.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
