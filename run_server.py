from __future__ import annotations

import argparse
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from app.main import app, startup


class AsgiHandler(BaseHTTPRequestHandler):
    server_version = "FoodLawQA/0.1"

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _handle(self) -> None:
        content_length = int(self.headers.get("content-length", "0") or 0)
        body = self.rfile.read(content_length) if content_length else b""
        response = asyncio.run(self._call_asgi(body))
        self.send_response(response["status"])
        sent_content_length = False
        for key, value in response["headers"]:
            header = key.decode("latin-1")
            if header.lower() == "content-length":
                sent_content_length = True
            self.send_header(header, value.decode("latin-1"))
        if not sent_content_length:
            self.send_header("Content-Length", str(len(response["body"])))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(response["body"])

    async def _call_asgi(self, body: bytes) -> dict:
        parsed = urlsplit(self.path)
        request_sent = False
        response_messages: list[dict] = []
        headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in self.headers.items()]
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": self.request_version.replace("HTTP/", ""),
            "method": self.command,
            "scheme": "http",
            "path": parsed.path,
            "raw_path": parsed.path.encode("utf-8"),
            "query_string": parsed.query.encode("utf-8"),
            "headers": headers,
            "client": self.client_address,
            "server": self.server.server_address,
        }

        async def receive() -> dict:
            nonlocal request_sent
            if request_sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict) -> None:
            response_messages.append(message)

        await app(scope, receive, send)
        start = next(message for message in response_messages if message["type"] == "http.response.start")
        body_parts = [message.get("body", b"") for message in response_messages if message["type"] == "http.response.body"]
        return {
            "status": start["status"],
            "headers": start.get("headers", []),
            "body": b"".join(body_parts),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the food safety legal QA app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    startup()
    server = ThreadingHTTPServer((args.host, args.port), AsgiHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
