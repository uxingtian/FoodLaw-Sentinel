# 生产级架构对齐说明

## 简历描述反推的目标

- 多智能体：管理智能体负责意图识别和路由，监管、消费者、生产经营者智能体负责角色化问答。
- MCP 工具：查询改写、法规要点抽取、合规处置清单等工具由智能体按角色调用。
- 知识图谱：围绕消费者、监管机构、生产经营者抽取权利、义务、监管措施等关系，作为检索增强信号。
- RAG 检索：文档解析、分块、Embedding、向量库、BM25 关键词召回、reranker 重排序。
- 模型推理：通过 OpenAI-compatible API 对接 Qwen/DeepSeek/vLLM。
- 证据链：离线评测准确率、来源召回、响应时间和并发压测报告。

## 当前代码对应关系

- `app/workflow.py`：本地工作流与可选 LangGraph `StateGraph` 适配。
- `app/tools.py`：MCP 风格工具注册表、查询改写工具、合规清单工具。
- `app/graph_store.py`：本地法律法规知识图谱抽取、查询、统计。
- `app/metrics.py`：运行时请求、fallback、角色分布、工具调用和耗时统计。

## 性能优化

- `app/retrieval_utils.py` 对分词结果使用 LRU 缓存，减少 BM25、图谱、reranker 的重复分词开销。
- `LegalKnowledgeGraph` 在构图时预计算每条边的 token，并缓存查询结果。
- `reports/benchmark.json` 记录当前本地 HTTP 压测结果，可用于对比后续 Chroma/vLLM 接入后的性能变化。
- `app/retrieval.py`：BM25 与向量检索融合。
- `app/vector_store.py`：本地 TF-IDF、dense embedding、可选 Chroma 后端。
- `app/embedding.py`：本地 hash embedding 与 OpenAI-compatible embedding 服务。
- `app/reranker.py`：本地 reranker 与 HTTP reranker 服务。
- `app/llm.py`：OpenAI-compatible chat completion，适配 vLLM。
- `scripts/evaluate.py`：离线评测。
- `scripts/benchmark.py`：HTTP 并发压测。

## 推荐生产配置

```env
VECTOR_BACKEND=dense
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_MODEL=qwen3-embedding-0.6b
EMBEDDING_BASE_URL=http://embedding-api:8000/v1
EMBEDDING_API_KEY=local-dev-key

RERANKER_PROVIDER=http
RERANKER_MODEL=bge-reranker
RERANKER_URL=http://reranker:8000/rerank
RERANKER_API_KEY=local-dev-key

OPENAI_BASE_URL=http://vllm:8000/v1
OPENAI_API_KEY=local-dev-key
QA_MODEL=Qwen3-72B-Chat
WORKFLOW_BACKEND=langgraph
```

## 简历表述边界

当前仓库已经具备可验证的多智能体 RAG 系统、可替换生产组件接口、评测和压测脚本。若简历写“Qwen3-72B、vLLM、Chroma、LangGraph、BGE reranker 已部署”，需要在目标机器上安装 `requirements-prod.txt` 并接入真实服务后，再用 `reports/` 下的评测与压测报告作为证据。

## 成果验收

运行以下命令生成证据链：

```bash
python scripts/run_evidence_pipeline.py --reports-dir reports --requests 50 --concurrency 10
```

也可以分步骤运行：

```bash
python scripts/evaluate.py --output reports/eval.json
python scripts/benchmark.py --base-url http://127.0.0.1:8000 --requests 100 --concurrency 10 --output reports/benchmark.json
python scripts/check_readiness.py --output reports/readiness.json
python scripts/verify_claims.py --readiness reports/readiness.json --output reports/claim_verification.json
python scripts/generate_resume_summary.py --output reports/resume_summary.md
```

`claim_verification.json` 中的 `supported_claims` 是当前报告已经证明的成果，`unsupported_claims` 是仍需接入真实生产组件或扩大压测规模后才能写入简历的内容。
`readiness.json` 用来证明真实 Qwen/vLLM、Embedding、Reranker、Chroma、LangGraph 是否已经可用。
`resume_summary.md` 只使用已验证成果生成简历安全版描述，并单独列出不建议直接写成已完成的生产成果。
