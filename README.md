# Synapse — 多 Agent 知识检索平台

Synapse 是一个生产级的异步多 Agent 框架，解决三个核心问题：

- **模糊短句意图识别**：用户说「那个怎么弄？」也能准确路由到正确 Agent
- **长对话 Token 膨胀**：自动摘要压缩 + 按需召回，大幅降低 LLM 成本
- **Agent 故障自愈**：Z-score 实时检测 + 自动摘除恢复，无需人工干预

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/JACK-123666/synapse.git
cd synapse

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY

# 3. 启动（需要 Docker）
docker-compose up -d

# 4. 测试
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "什么是 RAG？"}'
```

启动后访问：
- API 文档：http://localhost:8000/docs
- Prometheus：http://localhost:9090
- 健康检查：http://localhost:8000/health

## API

### POST /chat

```json
// 请求
{
  "session_id": "demo-001",
  "message": "帮我查一下向量数据库的原理",
  "user_id": "user-123"
}

// 响应
{
  "reply": "向量数据库是一种专门用于存储和检索高维向量的数据库...",
  "intent": "knowledge_retrieval",
  "agent_used": "retrieval_agent",
  "confidence": 0.85
}
```

| 字段 | 说明 |
|------|------|
| `session_id` | 会话 ID，同一会话共享短期记忆 |
| `message` | 用户输入（支持模糊短句） |
| `user_id` | 可选，用于个性化画像 |

### GET /health

返回 Redis、ChromaDB、LLM 连通性及各 Agent 实时健康分。

### GET /metrics

Prometheus 标准格式指标（请求数、延迟分布、Agent 健康分）。

## 工作原理

整个请求链路分为六个解耦模块：

```
用户消息
  → 意图识别（三路融合：LLM + 向量 + 关键词投票）
    → 记忆召回（短期 Redis + 长期 ChromaDB + 用户画像）
      → 路由调度（按 Agent 权重分发，失败自动降级）
        → Agent 执行（检索 / 摘要 / 兜底，可插拔扩展）
          → 记忆更新（追加短期记忆，达到阈值自动压缩）
            → 指标记录（Prometheus + Z-score 异常检测）

         ┌──────────────────────────────┐
         │  可观测性（横切所有模块）         │
         │  · 请求计数 / 延迟直方图         │
         │  · Agent 健康分动态调节          │
         │  · Z-score 滑动窗口异常检测      │
         └──────────────────────────────┘
```

### 三路融合意图识别

每种识别方式各有优劣，Synapse 让三路同时投票然后加权融合：

| 识别路径 | 权重 | 特点 |
|---------|------|------|
| LLM 语义分类 | 0.5 | 最准，但慢且贵 |
| 向量相似度 | 0.3 | 中等，依赖 embedding 质量 |
| 关键词匹配 | 0.2 | 最快，但粗糙 |

任一路失败时，其权重自动重新分配给其他路，最终取最高分作为意图。

### Agent 路由与故障自愈

系统维护一个路由表，每个意图对应一组候选 Agent（主 → 备用 → 兜底）：

```
knowledge_retrieval → RetrievalAgent → FallbackAgent
summarize          → SummarizeAgent → FallbackAgent
small_talk         → RetrievalAgent → FallbackAgent
```

- 按**权重**概率选择 Agent（权重高的被选概率大）
- 调用失败或超时 → 自动降级到下一个候选
- 最终必定有 **FallbackAgent** 兜底，100% 返回合法回复

Z-score 异常检测后台持续运行：

- 对每个 Agent 维护最近 20 次请求耗时的滑动窗口
- 当前耗时超过「均值 + 3×标准差」→ 判定异常
- 异常 Agent **权重减半**，健康分持续衰减
- 健康分 < 0.3 → 从路由池**摘除**
- 后台每 10 秒检查，正常后**自动恢复**

### 三级分层记忆

| 层级 | 存储 | 内容 | 作用 |
|------|------|------|------|
| 短期记忆 | Redis List | 当前会话最近 10 轮对话 | 维持上下文连贯 |
| 长期记忆 | ChromaDB 向量 | 历史对话摘要 embedding | 跨会话相似情景召回 |
| 用户画像 | Redis JSON | 偏好标签、常用术语 | 个性化 prompt 注入 |

**自动摘要压缩**：当对话达到 8 轮（或 Token 估算超 4000），自动触发 LLM 生成摘要 → embedding 存入 ChromaDB → 清空 Redis 短期记忆。下次请求时按语义相似度召回相关的历史摘要，拼接进上下文。

## 配置

所有参数通过环境变量覆盖，核心参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | openai | openai / claude |
| `LLM_API_KEY` | — | **必须配置** |
| `LLM_MODEL` | gpt-4o-mini | 对话模型 |
| `EMBEDDING_MODEL` | text-embedding-3-small | 向量化模型 |
| `INTENT_LLM_WEIGHT` | 0.5 | LLM 意图权重 |
| `SHORT_TERM_MAX_ROUNDS` | 10 | 短期记忆轮数 |
| `SUMMARY_TRIGGER_ROUNDS` | 8 | 触发压缩的轮数 |
| `TOKEN_BUDGET` | 4000 | Token 预算上限 |
| `ZSCORE_WINDOW_SIZE` | 20 | 异常检测窗口 |
| `ZSCORE_THRESHOLD` | 3.0 | 异常判定阈值 |
| `AGENT_HEALTH_THRESHOLD` | 0.3 | 摘除路由池阈值 |

完整配置见 [.env.example](.env.example)。

## 项目结构

```
synapse/
├── app/
│   ├── main.py                  # FastAPI 入口 + 启动关闭事件
│   ├── config.py                # 全局配置（环境变量覆盖）
│   ├── storage.py               # Redis / ChromaDB 连接单例
│   ├── api/chat.py              # /chat /health /metrics 端点
│   ├── llm/client.py            # LLM 统一封装（OpenAI / Claude 切换）
│   ├── intent/                  # 意图识别：LLM + 向量 + 关键词 → 融合
│   ├── router/                  # 路由调度：注册 + 权重分发 + 降级
│   ├── agents/                  # Agent 执行器：检索 / 摘要 / 兜底
│   ├── memory/                  # 记忆管理：短期 / 长期 / 画像 / 压缩
│   └── observability/           # 可观测性：Prometheus + Z-score 异常检测
├── docker-compose.yml           # 一键部署（api + redis + chromadb + prometheus）
├── Dockerfile
├── prometheus.yml
├── requirements.txt
└── .env.example
```

## 扩展自定义 Agent

```python
# app/agents/my_agent.py
from app.agents.base import AgentContext, AgentResponse, BaseAgent

class MyAgent(BaseAgent):
    agent_id = "my_agent"
    description = "我的自定义 Agent"

    async def execute(self, context: AgentContext) -> AgentResponse:
        return AgentResponse(reply="Hello!", metadata={})
```

然后在 `app/main.py` 的 `startup()` 中注册即可。

## License

MIT
