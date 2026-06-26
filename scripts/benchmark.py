from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


QUESTIONS = [
    "消费者买到不符合食品安全标准的食品怎么赔偿？",
    "监管部门发现食品安全隐患后可以采取哪些处置措施？",
    "食品生产企业发现产品可能危害人体健康时应如何召回？",
    "预包装食品标签能不能写疾病治疗功能？",
    "消费者发现食品安全问题可以向哪些部门投诉举报？",
]


def ask(base_url: str, question: str, timeout: float) -> dict:
    started = time.perf_counter()
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"question": question, "role": "auto", "top_k": 5},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return {"status": 0, "latency_ms": latency_ms, "ok": False, "error": str(exc)}
    latency_ms = (time.perf_counter() - started) * 1000
    return {"status": response.status_code, "latency_ms": latency_ms, "ok": response.ok}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(round((pct / 100) * (len(sorted_values) - 1))))
    return sorted_values[index]


def run(base_url: str, requests_count: int, concurrency: int, timeout: float) -> dict:
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(ask, base_url, QUESTIONS[index % len(QUESTIONS)], timeout)
            for index in range(requests_count)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    duration = time.perf_counter() - started
    latencies = [item["latency_ms"] for item in results]
    ok = sum(1 for item in results if item["ok"])
    return {
        "base_url": base_url,
        "requests": requests_count,
        "concurrency": concurrency,
        "success": ok,
        "success_rate": round(ok / requests_count, 4) if requests_count else 0,
        "duration_s": round(duration, 3),
        "throughput_qps": round(requests_count / duration, 2) if duration else 0,
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 2) if latencies else 0,
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "max": round(max(latencies), 2) if latencies else 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the QA HTTP API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = run(args.base_url, args.requests, args.concurrency, args.timeout)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
