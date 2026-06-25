from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.demo import DEFAULT_DEMO_SCENARIOS, run_demo_scenarios
from app.readiness import check_readiness
import app.main as app_main
from run_server import AsgiHandler
from scripts.benchmark import run as run_benchmark
from scripts.evaluate import evaluate, load_cases
from scripts.generate_acceptance_audit import build_acceptance_audit, generate_acceptance_markdown
from scripts.generate_demo_report import generate_demo_report
from scripts.generate_resume_summary import generate_summary
from scripts.doctor import build_doctor_report
from scripts.verify_claims import verify
from app.reports import load_report_summary


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ManagedServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> str:
        self.server = ThreadingHTTPServer((self.host, self.port), AsgiHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.2)
        return f"http://{self.host}:{self.port}"

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=3)


def run_pipeline(
    *,
    cases_path: Path,
    reports_dir: Path,
    host: str,
    port: int,
    requests_count: int,
    concurrency: int,
    timeout: float,
) -> dict:
    eval_report = asyncio.run(evaluate(load_cases(cases_path)))
    write_json(reports_dir / "eval.json", eval_report)

    demo_report = asyncio.run(run_demo_scenarios(app_main.workflow, DEFAULT_DEMO_SCENARIOS, top_k=settings.default_top_k))
    write_json(reports_dir / "demo_scenarios.json", demo_report)
    write_text(reports_dir / "demo_report.md", generate_demo_report(demo_report))

    with ManagedServer(host, port) as base_url:
        benchmark_report = run_benchmark(base_url, requests_count, concurrency, timeout)
    write_json(reports_dir / "benchmark.json", benchmark_report)

    readiness_report = check_readiness(settings, timeout=min(timeout, 3.0))
    write_json(reports_dir / "readiness.json", readiness_report)

    claim_report = verify(eval_report, benchmark_report, readiness_report=readiness_report)
    write_json(reports_dir / "claim_verification.json", claim_report)

    resume_summary = generate_summary(claim_report)
    write_text(reports_dir / "resume_summary.md", resume_summary)

    acceptance_audit = build_acceptance_audit(load_report_summary(reports_dir))
    write_json(reports_dir / "acceptance_audit.json", acceptance_audit)
    write_text(reports_dir / "acceptance_audit.md", generate_acceptance_markdown(acceptance_audit))
    doctor_report = build_doctor_report(reports_dir=reports_dir)
    write_json(reports_dir / "doctor.json", doctor_report)

    return {
        "eval": str(reports_dir / "eval.json"),
        "benchmark": str(reports_dir / "benchmark.json"),
        "readiness": str(reports_dir / "readiness.json"),
        "demo_scenarios": str(reports_dir / "demo_scenarios.json"),
        "demo_report": str(reports_dir / "demo_report.md"),
        "claim_verification": str(reports_dir / "claim_verification.json"),
        "resume_summary": str(reports_dir / "resume_summary.md"),
        "acceptance_audit": str(reports_dir / "acceptance_audit.md"),
        "doctor": str(reports_dir / "doctor.json"),
        "passed": claim_report["passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full evidence pipeline for resume-safe claims.")
    parser.add_argument("--cases", default="eval/golden_food_safety_qa.json")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    summary = run_pipeline(
        cases_path=Path(args.cases),
        reports_dir=Path(args.reports_dir),
        host=args.host,
        port=args.port,
        requests_count=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
