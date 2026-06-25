const roleLabels = {
  auto: "自动识别",
  regulator: "监管机构",
  consumer: "消费者",
  producer: "生产经营者",
  general: "通用咨询",
};

const state = {
  role: "auto",
  busy: false,
};

const healthText = document.querySelector("#healthText");
const modelBadge = document.querySelector("#modelBadge");
const docCount = document.querySelector("#docCount");
const chunkCount = document.querySelector("#chunkCount");
const roleStats = document.querySelector("#roleStats");
const claimBadge = document.querySelector("#claimBadge");
const evidenceSummary = document.querySelector("#evidenceSummary");
const documentList = document.querySelector("#documentList");
const uploadForm = document.querySelector("#uploadForm");
const chatForm = document.querySelector("#chatForm");
const chatLog = document.querySelector("#chatLog");
const routeText = document.querySelector("#routeText");
const questionInput = document.querySelector("#questionInput");
const topKInput = document.querySelector("#topKInput");
const sendButton = document.querySelector("#sendButton");
const refreshButton = document.querySelector("#refreshButton");

document.querySelector("#roleSwitch").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-role]");
  if (!button) return;
  state.role = button.dataset.role;
  document.querySelectorAll("#roleSwitch button").forEach((item) => item.classList.remove("selected"));
  button.classList.add("selected");
});

refreshButton.addEventListener("click", () => refreshAll());

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(uploadForm);
  setFormBusy(uploadForm, true);
  try {
    const response = await fetch("/api/documents", { method: "POST", body: formData });
    if (!response.ok) throw new Error(await readError(response));
    uploadForm.reset();
    await refreshAll();
  } catch (error) {
    alert(error.message);
  } finally {
    setFormBusy(uploadForm, false);
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || state.busy) return;
  appendMessage({ author: "你", meta: roleLabels[state.role], body: question, user: true });
  questionInput.value = "";
  setChatBusy(true);
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        role: state.role,
        top_k: Number(topKInput.value),
      }),
    });
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    const trace = data.route.trace || {};
    routeText.textContent = `${roleLabels[data.role]} · 置信度 ${Math.round(data.confidence * 100)}% · ${data.route.reason} · ${Math.round(trace.total_ms || 0)}ms`;
    appendMessage({
      author: "问答智能体",
      meta: data.fallback_used ? "本地检索回答" : "模型增强回答",
      body: data.answer,
      sources: data.sources,
      route: data.route,
    });
  } catch (error) {
    appendMessage({ author: "系统", meta: "请求失败", body: error.message });
  } finally {
    setChatBusy(false);
  }
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    chatForm.requestSubmit();
  }
});

async function refreshAll() {
  const [health, stats, documents, graph, reports] = await Promise.all([
    fetchJson("/api/health"),
    fetchJson("/api/stats"),
    fetchJson("/api/documents"),
    fetchJson("/api/graph"),
    fetchJson("/api/reports"),
  ]);
  renderHealth(health);
  renderStats(stats, graph.stats);
  renderReports(reports);
  renderDocuments(documents);
}

function renderHealth(health) {
  healthText.textContent = `${health.documents} 份资料，${health.chunks} 个片段 · ${health.workflow}`;
  modelBadge.textContent = health.model_configured ? "模型已配置" : "本地检索";
  modelBadge.className = `badge ${health.model_configured ? "ready" : "local"}`;
}

function renderStats(stats, graphStats) {
  docCount.textContent = stats.documents;
  chunkCount.textContent = stats.chunks;
  roleStats.innerHTML = "";
  Object.entries(stats.roles).forEach(([role, count]) => {
    const item = document.createElement("span");
    item.textContent = `${roleLabels[role]} ${count}`;
    roleStats.appendChild(item);
  });
  if (graphStats) {
    const graphItem = document.createElement("span");
    graphItem.textContent = `图谱关系 ${graphStats.edges}`;
    roleStats.appendChild(graphItem);
  }
}

function renderDocuments(documents) {
  documentList.innerHTML = "";
  if (!documents.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "暂无资料";
    documentList.appendChild(empty);
    return;
  }
  documents.forEach((documentItem) => {
    const item = document.createElement("article");
    item.className = "document-item";
    item.innerHTML = `
      <strong>${escapeHtml(documentItem.title)}</strong>
      <div class="document-meta">${escapeHtml(documentItem.source)} · ${documentItem.chunk_count} 片段</div>
      <div class="document-actions">
        <span class="role-pill">${roleLabels[documentItem.role]}</span>
        <button class="danger-button" type="button">删除</button>
      </div>
    `;
    item.querySelector("button").addEventListener("click", async () => {
      const response = await fetch(`/api/documents/${documentItem.id}`, { method: "DELETE" });
      if (!response.ok) {
        alert(await readError(response));
        return;
      }
      await refreshAll();
    });
    documentList.appendChild(item);
  });
}

function renderReports(reports) {
  const summary = reports.summary || {};
  const claims = summary.claims || {};
  const evaluation = summary.eval || {};
  const benchmark = summary.benchmark || {};
  const readiness = summary.readiness || {};
  const demo = summary.demo || {};
  const alignment = reports.alignment || {};
  const alignmentSummary = alignment.summary || {};
  const pendingItems = (alignment.items || []).filter((item) => item.status !== "verified").slice(0, 3);
  claimBadge.textContent = claims.passed ? "已验证" : "需补证";
  claimBadge.className = `mini-badge ${claims.passed ? "ready" : "warn"}`;
  evidenceSummary.innerHTML = "";
  if (!reports.available) {
    evidenceSummary.appendChild(evidenceRow("报告", "尚未生成"));
    return;
  }
  evidenceSummary.appendChild(evidenceRow("评测", `${percent(evaluation.role_accuracy)} 角色识别 · ${percent(evaluation.source_keyword_recall)} 来源召回`));
  evidenceSummary.appendChild(evidenceRow("压测", `${benchmark.concurrency || 0} 并发 · p95 ${formatNumber(benchmark.p95_ms)}ms · 成功 ${percent(benchmark.success_rate)}`));
  evidenceSummary.appendChild(evidenceRow("图谱", `${evaluation.graph_edges || 0} 条关系 · ${evaluation.cases || 0} 个离线用例`));
  evidenceSummary.appendChild(evidenceRow("角色演示", `${demo.role_passed || 0}/${demo.total || 0} 路由通过 · 来源 ${demo.with_sources || 0} · 护栏 ${demo.with_generation_guard || 0}`));
  evidenceSummary.appendChild(evidenceRow("简历对标", `${alignmentSummary.verified || 0}/${alignmentSummary.total || 0} 已验证 · ${alignmentSummary.pending_external_service || 0} 项待外部服务`));
  evidenceSummary.appendChild(evidenceRow("生产就绪", readiness.production_ready ? "外部服务已就绪" : "仍使用本地兜底/预留接口"));
  const details = document.createElement("details");
  details.className = "evidence-details";
  details.innerHTML = `
    <summary>${claims.supported_count || 0} 条可证明成果，${claims.unsupported_count || 0} 条待补证</summary>
    <ul>
      ${(claims.supported_claims || []).map((claim) => `<li>${escapeHtml(claim)}</li>`).join("")}
    </ul>
  `;
  evidenceSummary.appendChild(details);
  const links = document.createElement("div");
  links.className = "evidence-links";
  links.innerHTML = `
    <a href="/api/reports/artifacts/demo_report" target="_blank" rel="noreferrer">演示报告</a>
    <a href="/api/reports/artifacts/resume_summary" target="_blank" rel="noreferrer">简历摘要</a>
    <a href="/api/reports/artifacts/acceptance_audit" target="_blank" rel="noreferrer">验收审计</a>
    <a href="/api/reports/artifacts/doctor" target="_blank" rel="noreferrer">自检报告</a>
  `;
  evidenceSummary.appendChild(links);
  if (pendingItems.length) {
    const alignmentDetails = document.createElement("details");
    alignmentDetails.className = "evidence-details";
    alignmentDetails.innerHTML = `
      <summary>简历待补证项</summary>
      <ul>
        ${pendingItems.map((item) => `<li>${escapeHtml(item.label)}：${escapeHtml(statusLabel(item.status))}</li>`).join("")}
      </ul>
    `;
    evidenceSummary.appendChild(alignmentDetails);
  }
}

function evidenceRow(label, value) {
  const row = document.createElement("div");
  row.className = "evidence-row";
  row.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>`;
  return row;
}

function appendMessage({ author, meta, body, sources = [], route = null, user = false }) {
  const template = document.querySelector("#messageTemplate");
  const node = template.content.firstElementChild.cloneNode(true);
  if (user) node.classList.add("user");
  node.querySelector("strong").textContent = author;
  node.querySelector(".message-head span").textContent = meta || "";
  node.querySelector(".message-body").textContent = body;
  const sourceContainer = node.querySelector(".sources");
  if (route?.tools?.length) {
    const details = document.createElement("details");
    details.className = "source";
    details.innerHTML = `
      <summary>工具调用 · ${route.tools.length} 个</summary>
      <p>${escapeHtml(route.tools.map((tool) => `${tool.name}: ${tool.summary}`).join("；"))}</p>
    `;
    sourceContainer.appendChild(details);
  }
  sources.forEach((source) => {
    const details = document.createElement("details");
    details.className = "source";
    details.innerHTML = `
      <summary>[${source.index}] ${escapeHtml(source.title)} · ${Math.round(source.score * 100)}%</summary>
      <p>${escapeHtml(source.excerpt)}</p>
      <p>${escapeHtml(source.source)}</p>
    `;
    sourceContainer.appendChild(details);
  });
  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

async function readError(response) {
  try {
    const payload = await response.json();
    return payload.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}

function setFormBusy(form, busy) {
  form.querySelectorAll("input, select, button").forEach((item) => {
    item.disabled = busy;
  });
}

function setChatBusy(busy) {
  state.busy = busy;
  sendButton.disabled = busy;
  sendButton.textContent = busy ? "生成中" : "发送";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function percent(value) {
  if (typeof value !== "number") return "0%";
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value) {
  if (typeof value !== "number") return "0";
  return value.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
}

function statusLabel(status) {
  const labels = {
    verified: "已验证",
    implemented: "已实现",
    pending_external_service: "待接入外部服务",
    pending_evidence: "待补充证据",
  };
  return labels[status] || status || "未知";
}

appendMessage({
  author: "问答智能体",
  meta: "本地知识库已就绪",
  body: "请提出食品安全法律法规问题，或导入新的法规资料后查询。",
});

refreshAll().catch((error) => {
  healthText.textContent = "服务连接失败";
  appendMessage({ author: "系统", meta: "初始化失败", body: error.message });
});
