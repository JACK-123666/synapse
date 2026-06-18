# Synapse

多 Agent 知识检索，一条命令启动。

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python">
  <img src="https://img.shields.io/badge/license-MIT-blue">
</p>

---

## 干嘛的

用户说「那个怎么弄」，你得知道他要查文档还是想让你总结——这是意图识别。

查完文档聊了 20 轮，Token 不随轮数线性增长——这是 Token 膨胀。

某天 OpenAI 抽风，RetrievalAgent 全超时，用户看到 500——这是单点故障。

Synapse 就干这三件事。六个模块串一起，跑在 FastAPI 上，Docker 一行命令起来。

---

## 跑起来

```bash
git clone https://github.com/JACK-123666/synapse.git && cd synapse
cp .env.example .env       # 把 API_KEY 填上
docker-compose up -d        
```

试一下：

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","message":"什么是 RAG"}'
```

```json
{"reply":"RAG 是检索增强生成...","intent":"knowledge_retrieval","agent_used":"retrieval_agent","confidence":0.85}
```

**其他端点**

`GET /health` — Redis、ChromaDB、LLM 连通性，Agent 当前健康分

`GET /metrics` — Prometheus 吃的指标

浏览器打开 http://localhost:8000/docs 有 Swagger。

---

## 怎么工作的

一条请求穿过六个模块，顺序固定：

```
用户消息 → 意图识别 → 记忆召回 → 路由调度 → Agent 执行 → 记忆更新 → 返回
```

### 意图识别

三路融合识别，准确路由

| 方法 | 权重 | 一句话 |
|---|---|---|
| LLM 分类 | 0.5 | few-shot prompt，最准 |
| 向量匹配 | 0.3 | 拿 message embedding 去 ChromaDB 找最近的意图示例 |
| 关键词 | 0.2 | 扫一遍预置词典，命中就加分 |

### 路由

每个意图对应一串 Agent——主力的不行换备用的，备用的不行换兜底的。兜底的绝不挂。

```
knowledge_retrieval  →  RetrievalAgent  →  FallbackAgent
summarize            →  SummarizeAgent  →  FallbackAgent
small_talk           →  RetrievalAgent  →  FallbackAgent
```

选 Agent 看权重，权重看健康分。健康分怎么来的——每条请求的耗时记下来，最近 20 次的均值加三倍标准差是警戒线，超了就降权。降到 0.3 以下直接踢出候选列表，后台每 10 秒偷偷看一眼，恢复了就拉回来。

### 记忆

三层。短期塞 Redis，长期塞 ChromaDB。

对话超过 8 轮，后台起个任务把短期记忆丢给 LLM 总结成一段话，转成向量存进 ChromaDB，Redis 里的清掉。下次新会话进来，先拿消息去 ChromaDB 搜，看有没有以前聊过的相似内容，有就拼进 prompt 里。

---

## 配置

`.env.example`，改几个就行：

```
LLM_PROVIDER=openai      # openai / claude / deepseek
LLM_API_KEY=sk-xxx       # 必填
LLM_MODEL=gpt-4o-mini
DEEPSEEK_MODEL=deepseek-chat  # 用 deepseek 时才需要
INTENT_LLM_WEIGHT=0.5    # 觉得 LLM 太慢可以调低
SUMMARY_TRIGGER_ROUNDS=8 # 几轮开始压缩
ZSCORE_THRESHOLD=3.0     # 多敏感算异常
```

---

## 目录

```
app/
├── main.py          入口，启动时注册 Agent 和路由
├── config.py        所有可配参数
├── api/chat.py      /chat /health /metrics
├── llm/client.py    OpenAI / Claude 
├── intent/          意图识别
├── router/          注册表 + 分发 + 降级
├── agents/          Retrieval / Summarize / Fallback
├── memory/          短期 / 长期 / 画像 / 压缩
├── observability/   Prometheus 指标 + Z-score
docker-compose.yml
Dockerfile
```

自己加 Agent：继承 `agents/base.py` 里的 `BaseAgent`，`main.py` 启动时注册一下就行。

---

MIT
