# Production Verification Runbook

This runbook turns the current local demo into a production-verifiable resume claim. Use it only after the external services are actually reachable.

## 1. Configure Services

Copy `.env.production.example` to `.env` and replace the placeholders:

- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `QA_MODEL=Qwen3-72B-Chat`
- `EMBEDDING_PROVIDER=openai-compatible`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`
- `VECTOR_BACKEND=chroma`
- `RERANKER_PROVIDER=http`, `RERANKER_URL`, `RERANKER_MODEL`, `RERANKER_API_KEY`
- `WORKFLOW_BACKEND=langgraph`

Install production dependencies:

```bash
pip install -r requirements-prod.txt
```

## 2. Check Readiness

```bash
python scripts/check_readiness.py --output reports/readiness.json
python scripts/doctor.py --output reports/doctor.json
```

`reports/readiness.json` must show every production dependency as `ready`.

## 3. Run Evidence Pipeline

```bash
python scripts/run_evidence_pipeline.py --reports-dir reports --requests 200 --concurrency 20
python scripts/acceptance_gate.py --mode demo
```

The demo gate should pass before any larger benchmark is trusted.

## 4. Run 1000+ Concurrency Benchmark

Start the production app, then run the large benchmark from a machine with enough CPU, network, and file descriptor headroom:

```bash
python scripts/benchmark.py --base-url http://<host>:8000 --requests 5000 --concurrency 1000 --output reports/benchmark_1000.json
```

Keep `reports/benchmark_1000.json` with the final evidence package if you want to claim 1000+ concurrent access.

## 5. Production Gate

```bash
python scripts/acceptance_gate.py --mode production
python scripts/package_evidence.py --reports-dir reports --output dist/food-law-qa-evidence.zip
python scripts/verify_evidence_package.py dist/food-law-qa-evidence.zip
```

Only write the production-level resume wording when `acceptance_gate.py --mode production` passes.
