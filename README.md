# Synapse

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

带意图识别和记忆管理的多 Agent 对话服务。FastAPI + Redis + ChromaDB，Docker 一键部署。

## 为什么写这个

LLM 对话上生产会碰到三个问题：

**意图模糊。** 用户说"那个怎么弄"，你得判断他在查文档还是闲聊。单靠 LLM 分类慢且贵，单靠关键词不准。三路融合——LLM 语义 + 向量相似度 + 关键词投票——某路挂了自动把权重分给其他路，保证意图识别不成为单点。

**Token 膨胀。** 聊 20 轮把全部历史塞进 prompt，又贵又慢。只保留最近 N 轮会丢掉跨会话记忆——三天前聊过"向量数据库"，今天问"它的写入性能"，系统应该知道"它"指什么。做法是 Redis 存最近 10 轮，超阈值后台异步压成摘要存 ChromaDB，新消息进来先召回相似历史拼入 prompt。

**Agent 会挂。** 调 API 遇到 429、超时、自己写的逻辑 bug——任何一个炸了用户看到 500。每个意图绑一串 Agent，主挂切备，备挂切兜底。同时用 Z-score 监控延迟——超 μ+3σ 自动降权摘除，恢复后自动拉回，半夜 OpenAI 抽了系统自己切到 DeepSeek，好了再切回来。

## 快速开始

```bash
git clone https://github.com/JACK-123666/synapse.git && cd synapse
cp .env.example .env      # 填入 LLM_API_KEY
docker compose up -d
```

发一条消息试试：

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"什么是向量数据库"}'
```

浏览器打开 `http://localhost:8000/chat` 有对话界面，`http://localhost:8000/docs` 有 Swagger。

## 架构

一条请求经过 6 个步骤：

```text
消息 → 意图识别 → 记忆召回 → 路由分发 → Agent 执行 → 记忆更新 → 返回
```

### 意图识别

三路融合，加权投票：

| 方法 | 权重 | 说明 |
|------|------|------|
| LLM 语义 | 0.5 | few-shot prompt，最准确 |
| 向量匹配 | 0.3 | embedding 与意图示例做相似度比对 |
| 关键词 | 0.2 | 预置词典命中计分 |

任一路失败时权重自动重分配，三路全挂则返回默认意图。

### 路由与降级

每个意图绑定一串 Agent，按健康分排序。主 Agent 失败自动降级到备用，备用失败走兜底。兜底 Agent 不调 LLM，返回预设文本，保证任何情况都有回复。

Agent 的健康分由 Z-score 滑动窗口动态计算——最近 20 次请求的延迟分布，超 μ+3σ 视为异常，权重衰减至 0.3 以下摘除，后台定时重检恢复。

### 记忆

- **短期** (Redis)：当前会话的最近 N 轮对话，TTL 24 小时
- **长期** (ChromaDB)：对话摘要的向量索引，新消息进入时召回相似历史，拼入 prompt
- **压缩**：会话超过 8 轮后台触发，LLM 将短期记忆压缩为摘要存入 ChromaDB，清空 Redis 释放 token 预算

### 联网搜索

知识库中未命中时，自动通过 DuckDuckGo 搜索并注入结果到 prompt。前端开关可控。

## 配置

关键环境变量，全部在 `.env` 中设置：

```text
LLM_PROVIDER=deepseek       # openai / claude / deepseek
LLM_API_KEY=sk-xxx          # 必填
LLM_MODEL=gpt-4o-mini       # provider=openai 时生效
DEEPSEEK_MODEL=deepseek-chat
LLM_BASE_URL=https://api.openai.com/v1

# 可选：embedding 专用密钥（DeepSeek 不支持 embedding，需单独配）
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=

# 内存与压缩
SHORT_TERM_MAX_ROUNDS=10
SUMMARY_TRIGGER_ROUNDS=8

# 意图权重
INTENT_LLM_WEIGHT=0.5
INTENT_VECTOR_WEIGHT=0.3
INTENT_KEYWORD_WEIGHT=0.2
```

更多参数见 `.env.example`。

## 目录

```
app/
├── main.py                 FastAPI 入口，启动时注册 Agent 和路由
├── config.py               所有可配参数
├── store.py                Redis / ChromaDB 连接单例
├── api/chat.py             /chat /health /models /metrics
├── llm/gateway.py          LLM 统一客户端（支持运行时切换）
├── intent/
│   ├── blend.py            三路融合
│   ├── semantic.py         LLM 语义分类
│   ├── keyword.py          关键词投票
│   └── vector.py           向量相似度
├── router/
│   ├── pool.py             Agent 注册表
│   └── route.py            分发与降级
├── agents/
│   ├── base.py             基类
│   ├── knowledge.py        知识检索
│   ├── summary.py          摘要
│   └── safety.py           兜底
├── memory/
│   ├── recent.py           短期记忆 (Redis)
│   ├── archive.py          长期记忆 (ChromaDB)
│   ├── compress.py         压缩调度
│   └── profile.py          用户画像
├── observability/
│   ├── health.py           异常检测与自愈
│   └── metrics.py          Prometheus 指标
├── tools/
│   └── search.py           联网搜索
└── _singleton.py           工具函数
```

## 扩展 Agent

继承 `BaseAgent`，实现 `execute()` 方法：

```python
from app.agents.base import BaseAgent, AgentContext, AgentResponse

class MyAgent(BaseAgent):
    agent_id = "my_agent"
    description = "自定义 Agent"

    async def execute(self, context: AgentContext) -> AgentResponse:
        # 你的逻辑
        return AgentResponse(reply="done", metadata={})
```

然后在 `main.py` 启动事件中注册并绑定路由即可。

## License

MIT
