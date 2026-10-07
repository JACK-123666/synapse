# Synapse

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![LangChain](https://img.shields.io/badge/LangChain-1.x-orange)

基于 LangChain 的多 Agent 全能智能助手：知识库问答（RAG）、记忆检索、网页抓取、代码仓库助手、定时任务、插件扩展（本地 Python 插件 + MCP）。个人用开箱即用，企业用支持多用户与角色权限。FastAPI + Redis + ChromaDB + SQLite/PostgreSQL，Docker 一键部署。（English version: [README.md](README.md)）

## 为什么？

LLM 对话上生产会碰到三个问题：

**意图模糊。** 用户说"那个怎么弄"，你得判断他在查文档、翻聊天记录、看代码仓库还是闲聊。单靠 LLM 分类慢且贵，单靠关键词不准。三路融合——LLM 语义 + 向量相似度 + 关键词投票——某路挂了自动把权重分给其他路，保证意图识别不成为单点。意图不是写死的：每个能力、每个插件都可以注册自己的意图，三路识别自动生效。

**Token 膨胀。** 聊 20 轮把全部历史塞进 prompt，又贵又慢。只保留最近 N 轮会丢掉跨会话记忆——三天前聊过"向量数据库"，今天问"它的写入性能"，系统应该知道"它"指什么。做法是 Redis 存最近 10 轮，超阈值后台异步压成摘要存 ChromaDB，新消息进来先召回相似历史拼入 prompt。

**Agent 会挂。** 调 API 遇到 429、超时、自己写的逻辑 bug——任何一个炸了用户看到 500。每个意图绑一串 Agent，主挂切备，备挂切兜底。同时用 Z-score 监控延迟——超 μ+3σ 自动降权摘除，恢复后自动拉回，故障 Agent 不会拖垮整条链路。

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

浏览器打开 `http://localhost:8000/chat` 有对话界面（流式输出），`http://localhost:8000/docs` 有 Swagger。

**企业部署**：在 `.env` 设置 `AUTH_ENABLED=true` 与 `ADMIN_PASSWORD`，用 `POST /auth/login` 换取 token（或 `POST /auth/api-keys` 生成 API Key），管理员用 `POST /users` 添加成员。需要 PostgreSQL 时：`docker compose --profile postgres up -d`，并设置 `DATABASE_URL`。

## 能力一览

| 能力 | 怎么用 | 背后 |
|------|--------|------|
| 知识问答（RAG） | `POST /knowledge-bases` 建库，上传 txt/md/pdf/docx/html/代码，直接提问 | 切片 → 向量化 → 按用户权限检索 → 带引用回答；知识库分私有 / 共享 |
| 记忆检索 | "我上次问过什么？" / `GET /memories/search` | 短期 Redis + 长期摘要 ChromaDB + 用户画像 |
| 网页抓取 | 消息里带上链接，"总结这个网页" | 正文提取（trafilatura）、SSRF 防护、可抓取入库 |
| 联网搜索 | 知识问答时模型自主决定是否搜索 | Function Calling，前端开关 `X-Web-Search` |
| 代码仓库助手 | `POST /repos` 连接 GitHub / GitLab / 本地 Git，"看看最近的提交""总结 PR 12" | 提交、Issue、PR/MR、读文件、搜代码、changelog、索引进知识库做代码问答；写操作需管理员开启且逐次确认 |
| 定时任务 | "每天早上 9 点总结仓库新提交，推送到飞书" | APScheduler + 数据库持久化；prompt / 网页监控 / 知识库同步；飞书 / 企业微信 / 钉钉 Webhook |
| 插件 | 放进 `plugins/` 或 `POST /plugins/mcp/servers` | 本地 Python 插件热重载；MCP 服务即配即用；按角色设置工具白名单 |

## 架构

一条请求经过 6 个步骤：

```text
消息 → 意图识别 → 记忆召回 → 路由分发 → Agent 执行（LangChain create_agent + 工具） → 记忆更新 → 返回
```

### 能力与插件

内置能力和外部插件走同一套 `Capability` 接口，各自注册四样东西：工具、意图（描述 / 关键词 / 示例）、专属 Agent、路由。意图融合负责挑出专属 Agent，专属 Agent 只拿到自己那一组工具——工具再多也不会降低选择准确率。专属 Agent 失败时降级到通用 Agent（可用全部有权限的工具），最后到兜底 Agent。

| 意图 | 主 Agent | 降级链 |
|------|----------|--------|
| knowledge_retrieval / small_talk | retrieval_agent | fallback_agent |
| summarize | summarize_agent | fallback_agent |
| memory_query / web_browse / repo_management / schedule_task | 对应专属 Agent | general_agent → fallback_agent |
| general_task 及插件意图 | general_agent / 插件 Agent | general_agent → fallback_agent |

### 意图识别

三路融合，加权投票：

| 方法 | 权重 | 说明 |
|------|------|------|
| LLM 语义 | 0.5 | few-shot prompt，最准确；意图目录有扩展时自动生成示例 |
| 向量匹配 | 0.3 | embedding 与意图示例做相似度比对；示例变化自动重建索引 |
| 关键词 | 0.2 | 预置词典命中计分 |

任一路失败时权重自动重分配，三路全挂则返回默认意图。

### 路由与降级

每个意图绑定一串 Agent，按健康分排序。主 Agent 失败自动降级到备用，备用失败走兜底。兜底 Agent 不调 LLM，返回预设文本，保证任何情况都有回复。流式输出时，只要还没输出第一个字就可以无感降级。

Agent 的健康分由 Z-score 滑动窗口动态计算——最近 20 次请求的延迟分布，超 μ+3σ 视为异常，权重衰减至 0.3 以下摘除，后台定时重检恢复。

### 记忆

- **短期** (Redis)：当前会话的最近 N 轮对话，TTL 24 小时
- **长期** (ChromaDB)：对话摘要的向量索引，新消息进入时召回相似历史，拼入 prompt
- **压缩**：会话超过 8 轮后台触发，LLM 将短期记忆压缩为摘要存入 ChromaDB，只移除已压缩的消息（压缩期间的新消息保留）
- **隔离**：开启鉴权后会话与记忆按用户隔离

### 模型

LLM 调用统一走 LangChain ChatModel（`ChatOpenAI` 兼容 OpenAI / DeepSeek，`ChatAnthropic` 对接 Claude）。管理员可以 `POST /models/switch` 运行时切换全局模型；`/chat` 的 `model` 参数只影响当前请求，不改全局配置。

## 配置

关键环境变量，全部在 `.env` 中设置：

```text
LLM_PROVIDER=deepseek       # openai / claude / deepseek
LLM_API_KEY=sk-xxx          # 必填
LLM_MODEL=gpt-4o-mini       # provider=openai 时生效
DEEPSEEK_MODEL=deepseek-chat
LLM_BASE_URL=https://api.openai.com/v1

# embedding（DeepSeek 不支持 embedding：配专用密钥，或用本地模型 EMBEDDING_PROVIDER=local）
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_PROVIDER=openai

# 鉴权（企业部署务必开启）
AUTH_ENABLED=false
ADMIN_PASSWORD=

# 数据库（留空用 data/ 下的 SQLite）
DATABASE_URL=

# 代码仓库写操作（默认关闭）
REPO_WRITE_ENABLED=false

# 记忆与压缩
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
├── main.py                     FastAPI 入口；lifespan 启动顺序（配置 → 数据库 → 能力 → 插件 → 意图索引 → 自愈循环）
├── config.py                   所有可配参数（pydantic-settings，环境变量 / .env）
├── store.py                    Redis / ChromaDB 连接单例（Chroma 连接失败有 30 秒冷却）
│
├── api/                        HTTP 层，每个资源组一个模块
│   ├── chat.py                 POST /chat、POST /chat/stream（SSE）、GET /chat 页面
│   ├── system.py               /health、/metrics、/models、/capabilities
│   ├── auth.py                 登录、JWT、API Key、/users
│   ├── knowledge.py            知识库、文档上传 / 检索 / 重建索引
│   ├── memory.py               记忆浏览与检索
│   ├── repos.py                仓库连接、提交、变更日志、索引、写操作确认
│   ├── schedules.py            定时任务增删改查、cron 解析、执行历史
│   └── plugins.py              插件列表 / 启停 / 重载、MCP 服务、工具白名单
│
├── core/                       平台底座，不含业务逻辑
│   ├── context.py              contextvars 里的 RequestContext（用户、会话、模型覆盖）
│   ├── db.py                   SQLAlchemy 2.0 异步引擎、session_scope
│   ├── security.py             bcrypt、JWT、API Key 哈希、Fernet 加密
│   ├── deps.py                 get_current_user / require_admin 两个鉴权依赖
│   └── tasks.py                后台任务托管：spawn() 持强引用，关闭时 drain()
│
├── services/
│   ├── chat.py                 对话编排主链路：prepare → dispatch → 更新记忆
│   ├── users.py                用户、登录、API Key
│   └── policies.py             角色 → 工具白名单（落库，启动时加载）
│
├── llm/
│   ├── config.py               运行时配置；运行时覆盖优先于 .env
│   ├── factory.py              按解析出的配置创建并缓存 ChatModel / Embeddings
│   ├── gateway.py              chat() / embed() 统一入口（单一 LangChain 后端）
│   └── messages.py             dict 与 LangChain 消息互转
│
├── intent/                     意图识别
│   ├── catalog.py              动态意图目录：配置默认值 + 各能力/插件注册的意图
│   ├── semantic.py             LLM 语义路（few-shot；意图变多后自动生成 prompt）
│   ├── vector.py               向量相似度路（Chroma；目录变化时自动重建索引）
│   ├── keyword.py              关键词投票路
│   └── blend.py                融合：便宜两路短路 → 加权投票 → 某路失败时重分配权重
│
├── router/
│   ├── pool.py                 Agent 注册表 + 意图 → Agent 路由表
│   └── route.py                分发、按权重排序、降级（含流式）
│
├── agents/
│   ├── base.py                 BaseAgent / AgentContext / AgentResponse
│   ├── langchain_agent.py      LangChainAgent：按标签+角色选工具、静态 system prompt、缓存 Agent 图
│   ├── knowledge.py            RetrievalAgent —— RAG 上下文、闲聊、摘要兜底
│   ├── summary.py              SummarizationAgent —— 不用工具，支持流式
│   └── safety.py               FallbackAgent —— 绝不抛异常、绝不调 LLM
│
├── capabilities/               每个能力一个包，各自注册 工具 + 意图 + Agent + 路由
│   ├── base.py                 Capability 接口 + CapabilityTool 声明
│   ├── manager.py              注册 / 热重载；未提供 Agent 时自动生成 <能力名>_agent
│   ├── core.py                 四个基础 Agent 及其路由
│   ├── knowledge/              loaders（txt/md/pdf/docx/html/代码）· service（切片 → 向量化 → 检索）· tools
│   ├── memory/                 记忆检索工具 + MemoryAgent
│   ├── web/                    fetch（防 SSRF 的正文提取）· tools（搜索 / 抓取 / 存入知识库）
│   ├── repo/                   providers（GitHub/GitLab/本地 Git）· service（连接、变更日志）· tools
│   └── scheduler/              cron 解析 · APScheduler 服务 · Webhook 投递
│
├── plugins/
│   ├── manager.py              本地 Python 插件：发现、清单、启停、热重载
│   └── mcp.py                  MCP 服务 → 工具命名为 mcp_<服务名>_<工具名>
│
├── memory/
│   ├── recent.py               短期记忆（Redis、TTL、裁剪）
│   ├── archive.py              长期摘要（Chroma）+ 兼容保留的全局知识库
│   ├── compress.py             阈值判断 → LLM 摘要 → 入库 → 只裁掉已压缩的部分
│   └── profile.py              用户画像，乐观锁更新
│
├── observability/
│   ├── health.py               Z-score 延迟滑窗、权重衰减 / 恢复、从路由池摘除
│   └── metrics.py              Prometheus 计数 / 直方图 / 仪表
│
├── tools/
│   ├── registry.py             工具注册表：来源、标签、写操作标记、角色白名单
│   └── search.py               联网搜索
│
├── models/__init__.py          11 张 ORM 表（users、api_keys、knowledge_bases、documents……）
└── static/index.html           单文件聊天页（原生 JS + SSE）

plugins/example_plugin/         示例能力插件（时间 + 计算器）
tests/                          78 个 pytest 用例 —— Redis / Chroma / 数据库 / LLM 全部换成本地替身
tools/                          离线评测脚本（意图准确率、记忆 Token 开销）
```

## 扩展

新增能力或插件：实现 `Capability`，注册工具、意图、Agent。插件放进 `plugins/` 后调用 `POST /plugins/reload` 即可加载，详见 [插件开发指南](docs/PLUGIN_DEV.md)。

```python
from langchain_core.tools import tool
from app.capabilities.base import Capability
from app.intent.catalog import IntentSpec

@tool
async def query_order(order_id: str) -> str:
    """查询订单状态。"""
    return f"订单 {order_id} 已发货"

class OrderPlugin(Capability):
    name = "order"
    def tools(self):
        return [query_order]
    def intents(self):
        return [IntentSpec(name="order_query", description="查询订单", keywords=["订单"])]
```

只想写一个 Agent：继承 `LangChainAgent`，设置 `tool_tags` 和 `build_system_prompt()`；或继承 `BaseAgent` 实现 `execute()`。

## 开发与测试

```bash
python -m venv .venv && .venv/Scripts/activate     # Python 3.10 / 3.11（chromadb 0.4.24 需要 numpy<2）
pip install -r requirements-dev.txt   # 核心 + 测试依赖（只跑应用用 requirements.txt）
pytest                                              # Redis / Chroma / LLM 全部使用本地替身
python tools/eval_intent.py --no-llm                # 意图识别评测
python tools/eval_memory.py                         # 记忆 Token 评测
```

## License

MIT
