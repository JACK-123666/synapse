<h1 align="center">Synapse</h1>
<p align="center">多 Agent 知识检索平台</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

---

一个异步、自愈的生产级 Agent 框架。解决三个实际问题：

- **模糊短句懂你**：「那个怎么弄？」→ 三路融合识别，准确路由
- **对话越长越省钱**：自动摘要压缩，Token 不随轮数线性增长
- **Agent 挂了不用管**：Z-score 实时检测，自动摘除、自动恢复

---

## 快速开始

```bash
git clone https://github.com/JACK-123666/synapse.git && cd synapse
cp .env.example .env         # 填入 LLM_API_KEY
docker-compose up -d
```

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"什么是 RAG？"}'
```

> 启动后访问 API 文档 → http://localhost:8000/docs

---

## API

`POST /chat` — 核心接口。`session_id` 区分会话，`user_id` 可选（画像用）。

```json
{ "session_id":"demo","message":"帮我查一下向量数据库","user_id":"u1" }
```
```json
{ "reply":"向量数据库是...","intent":"knowledge_retrieval","agent_used":"retrieval_agent","confidence":0.85 }
```

`GET /health` — 各服务连通性 + Agent 实时健康分

`GET /metrics` — Prometheus 标准指标（QPS、延迟分布、健康分）

---

## 原理

一条请求走六个模块：

```
               ┌─ 意图识别 ─→ 路由调度 ─→ Agent 执行 ─┐
用户消息 ─→ 记忆召回                                   记忆更新 ─→ 回复
               └────────── 可观测性（横切监控）──────────┘
```

### 意图识别：三路投票

LLM 语义、向量相似度、关键词匹配同时跑（`asyncio.gather` 并行），加权融合。LLM 挂了？自动把权重让给另外两路。

| 路径 | 权重 | 特点 |
|---|---|---|
| LLM 语义 | 0.5 | 最准，最慢 |
| 向量相似度 | 0.3 | 中等 |
| 关键词匹配 | 0.2 | 最快 |

### 路由调度：权重 + 降级

每个意图绑定一条降级链：

```
knowledge_retrieval → RetrievalAgent → FallbackAgent
summarize          → SummarizeAgent → FallbackAgent
small_talk         → RetrievalAgent → FallbackAgent
```

按权重选 Agent，超时或异常自动降级到下一个。FallbackAgent 100% 兜底。

Z-score 异常检测在后台持续运行：最近 20 次耗时超过「均值+3σ」→ 权重要×0.5 → 健康分<0.3 → 摘除 → 恢复后自动加回。

### 记忆：三级 + 自动压缩

| 层 | 存哪 | 是什么 | 干啥 |
|---|---|---|---|
| 短期 | Redis | 最近 10 轮对话 | 上下文连贯 |
| 长期 | ChromaDB | 历史摘要向量 | 跨会话相似召回 |
| 画像 | Redis | 偏好标签、术语 | 个性化 prompt |

对话到第 8 轮自动触发压缩（后台异步，不卡回复）：LLM 生成摘要 → embedding 存 ChromaDB → 清 Redis。新会话按语义召回相关历史。

---

## 可配置项

全部通过 `.env` 覆盖，核心几个：

| 参数 | 默认 | 说明 |
|---|---|---|
| `LLM_PROVIDER` | openai | openai / claude |
| `LLM_API_KEY` | — | **必填** |
| `LLM_MODEL` | gpt-4o-mini | |
| `INTENT_LLM_WEIGHT` | 0.5 | 意图融合权重 |
| `SUMMARY_TRIGGER_ROUNDS` | 8 | 几轮触发压缩 |
| `ZSCORE_THRESHOLD` | 3.0 | 异常判定门槛 |
| `AGENT_HEALTH_THRESHOLD` | 0.3 | 低于此摘除 |

详见 [.env.example](.env.example)。

---

## 目录

```
synapse/
├── app/
│   ├── main.py              # 入口 + 生命周期
│   ├── config.py            # 全局配置
│   ├── storage.py           # Redis / ChromaDB 连接
│   ├── api/chat.py          # 端点
│   ├── llm/client.py        # LLM 封装 (OpenAI/Claude)
│   ├── intent/              # 意图识别 (LLM+向量+关键词)
│   ├── router/              # 路由调度 (注册+分发+降级)
│   ├── agents/              # Agent (检索/摘要/兜底)
│   ├── memory/              # 记忆 (短期/长期/画像/压缩)
│   └── observability/       # 监控 (Prometheus+Z-score)
├── docker-compose.yml       # 一键部署
├── Dockerfile
├── prometheus.yml
└── requirements.txt
```

放一个自定义 Agent → 继承 `BaseAgent`，在 `main.py` 注册路由即用。

---

MIT
