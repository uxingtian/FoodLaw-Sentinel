from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.acceptance_gate import evaluate_gate, load_audit
from scripts.package_evidence import build_evidence_package
from scripts.run_evidence_pipeline import run_pipeline
from scripts.verify_evidence_package import verify_evidence_package


def build_delivery(
    *,
    reports_dir: Path,
    package_path: Path,
    run_pipeline_fn: Callable[..., dict[str, Any]] = run_pipeline,
    evaluate_gate_fn: Callable[..., dict[str, Any]] = evaluate_gate,
    package_fn: Callable[..., Path] = build_evidence_package,
    verify_package_fn: Callable[[Path], dict[str, Any]] = verify_evidence_package,
    cases_path: Path = Path("eval/golden_food_safety_qa.json"),
    host: str = "127.0.0.1",
    port: int = 8032,
    requests_count: int = 50,
    concurrency: int = 10,
    timeout: float = 10.0,
) -> dict[str, Any]:
    pipeline = run_pipeline_fn(
        cases_path=cases_path,
        reports_dir=reports_dir,
        host=host,
        port=port,
        requests_count=requests_count,
        concurrency=concurrency,
        timeout=timeout,
    )
    audit = load_audit(reports_dir / "acceptance_audit.json") if (reports_dir / "acceptance_audit.json").exists() else {}
    gate = evaluate_gate_fn(audit, mode="demo")
    package = package_fn(reports_dir=reports_dir, output_path=package_path)
    verification = verify_package_fn(package)
    passed = bool(pipeline.get("passed")) and bool(gate.get("passed")) and bool(verification.get("passed"))
    return {
        "passed": passed,
        "pipeline": pipeline,
        "gate": gate,
        "package": str(package),
        "verification": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local demo evidence delivery package in one command.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output", default="dist/food-law-qa-evidence.zip")
    parser.add_argument("--port", type=int, default=8032)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    result = build_delivery(
        reports_dir=Path(args.reports_dir),
        package_path=Path(args.output),
        port=args.port,
        requests_count=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
