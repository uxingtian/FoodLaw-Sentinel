# 食品安全法律法规多智能体问答系统

本项目是一个本地可运行的食品安全法律法规 RAG 问答系统。它使用 FastAPI 提供 API 和静态前端，内置示例知识库，支持上传 `.txt`、`.md`、`.pdf`、`.docx` 资料，使用 BM25 + 可插拔向量检索 + reranker 重排序，并按监管、消费者、生产经营者等角色路由回答。

## 当前能力

- 管理智能体负责问题路由，消费者、监管机构、生产经营者、通用咨询角色智能体分别组织回答。
- MCP 风格工具注册表负责查询改写、角色化合规清单生成，并在回答 trace 中返回工具调用记录。
- 本地法规知识图谱会从资料片段抽取“主体-关系-义务/权利/措施”，用于检索加权、工具调用和 `/api/graph` 展示。
- 检索链路为 BM25 关键词召回 + 可插拔向量检索 + reranker 重排序。
- 默认使用本地 TF-IDF 向量后端和本地 reranker，无需外部服务即可运行。
- 支持本地 hash embedding、OpenAI-compatible embedding、Chroma、HTTP reranker、vLLM/OpenAI-compatible LLM 的配置入口。
- `WORKFLOW_BACKEND=langgraph` 且安装 LangGraph 后，可使用真实 LangGraph StateGraph；未安装时 `auto` 模式回落到本地工作流。
- 回答返回引用来源、置信度、路由信息和每一步耗时 trace。
- `/api/metrics` 暴露请求量、fallback 比例、角色分布、工具调用量和响应耗时。
- `/api/graph` 暴露图谱节点、关系、角色分布和示例关系。
- 检索层对分词和图谱查询做缓存，降低重复咨询和高并发场景下的响应时间。

## 运行

```bash
python -m uvicorn app.main:app --reload
```

访问 <http://127.0.0.1:8000>。

如果本机 `uvicorn` 缺少 HTTP 协议依赖，可以使用项目内置的标准库启动器：

```bash
python run_server.py
```

## 可选模型配置

复制 `.env.example` 为 `.env`，配置 OpenAI-compatible 服务：

```bash
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://你的兼容接口/v1
QA_MODEL=qwen-plus
VECTOR_BACKEND=local
RERANKER_PROVIDER=local
```

未配置模型时，系统会使用检索片段生成兜底回答，并保留来源引用。

## 评测与压测

一键生成完整证据链：

```bash
python scripts/run_evidence_pipeline.py --reports-dir reports --requests 50 --concurrency 10
```

该命令会生成 `eval.json`、`benchmark.json`、`readiness.json`、`claim_verification.json` 和 `resume_summary.md`。

离线评测角色路由、答案关键词召回、来源召回和平均耗时：

```bash
python scripts/evaluate.py --output reports/eval.json
```

启动服务后做 HTTP 并发压测：

```bash
python scripts/benchmark.py --requests 100 --concurrency 10
```

把评测和压测报告转换为简历成果验收结果：

```bash
python scripts/check_readiness.py --output reports/readiness.json
python scripts/verify_claims.py --readiness reports/readiness.json --output reports/claim_verification.json
python scripts/generate_resume_summary.py --output reports/resume_summary.md
python scripts/doctor.py --output reports/doctor.json
python scripts/acceptance_gate.py --mode demo
python scripts/acceptance_gate.py --mode production
python scripts/package_evidence.py --reports-dir reports --output dist/food-law-qa-evidence.zip
python scripts/verify_evidence_package.py dist/food-law-qa-evidence.zip
python scripts/build_delivery.py --reports-dir reports --output dist/food-law-qa-evidence.zip
```

`doctor.py` 用于交付前自检：输出 Python 版本、关键配置、报告文件是否存在，以及 demo/production 两种验收门禁状态。
`acceptance_gate.py --mode demo` 用于本地演示验收：要求离线 claim verifier 和三角色 demo 均通过。
`acceptance_gate.py --mode production` 用于生产级简历验收：要求所有简历对标项都变为 `verified`，仍有 Qwen/vLLM、Chroma/LangGraph、Reranker 或 1000+ 并发待补证时会失败。
完整交付物包括 `reports/demo_report.md`、`reports/resume_summary.md` 和 `reports/acceptance_audit.md`。
`package_evidence.py` 会把白名单报告打包为 zip，便于提交答辩材料或面试展示。
`verify_evidence_package.py` 会读取 zip 内的 `MANIFEST.json`，校验每个报告文件的大小和 SHA256。
`build_delivery.py` 会串联本地证据管线、demo gate、证据包生成和证据包校验，适合交付前一键刷新。

简历中的“准确率 90%+、响应 5 秒内、并发访问”等指标应以 `reports/eval.json` 和压测输出为准，不建议在未提供评测集和压测报告时直接声称生产级指标。
`reports/claim_verification.json` 会列出 `supported_claims` 和 `unsupported_claims`，用于区分当前可以写入简历的成果和仍需真实服务支撑的成果。
`reports/readiness.json` 会检查 Qwen/vLLM、Embedding、Reranker、Chroma、LangGraph 是否已经真实配置并可访问。
`reports/resume_summary.md` 会基于已验证成果生成简历安全版项目描述。

运行时指标：

```bash
curl http://127.0.0.1:8000/api/metrics
curl http://127.0.0.1:8000/api/readiness
curl http://127.0.0.1:8000/api/reports
```

`/api/reports` 会汇总 `reports/` 下的评测、压测、readiness 和 claim verification，并返回 `alignment` 字段：

- `verified`：当前报告已经证明的简历能力，例如多智能体工具链、来源引用召回、响应时间和知识图谱关系抽取。
- `pending_external_service`：代码已预留但需要真实外部服务证明的能力，例如 Qwen/vLLM、Chroma、LangGraph、Reranker。
- `pending_evidence`：需要扩大评测或压测规模后才能写入简历的能力，例如 1000+ 并发声明。

## 对齐简历描述的生产级增强

- 将 `VECTOR_BACKEND=chroma` 并安装 Chroma 后，可切换到 Chroma 持久化向量库；未安装时自动回落到本地 TF-IDF。
- 将 `VECTOR_BACKEND=dense`、`EMBEDDING_PROVIDER=openai-compatible` 指向 embedding 服务后，可使用真实 embedding 向量召回。
- 将 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`QA_MODEL` 指向 Qwen/DeepSeek/vLLM 的 OpenAI-compatible 接口，即可启用模型增强回答。
- 将 `RERANKER_PROVIDER=http`、`RERANKER_URL` 指向 BGE/gte reranker 服务后，可使用真实重排序模型。
- 将 `WORKFLOW_BACKEND=langgraph` 并安装 `requirements-prod.txt` 后，可使用 LangGraph StateGraph 工作流。

更多生产级说明见 [docs/architecture.md](docs/architecture.md)。
生产级验收步骤见 [docs/production_runbook.md](docs/production_runbook.md)，环境变量模板见 `.env.production.example`。

## Docker

```bash
docker compose up --build
```

`docker-compose.yml` 默认按 vLLM、embedding 服务、reranker 服务的生产接入方式配置环境变量。实际部署时把 URL 指向已有服务即可。

## 测试

```bash
python -m pytest
```
