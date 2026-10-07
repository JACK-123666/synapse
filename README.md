# Synapse

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![LangChain](https://img.shields.io/badge/LangChain-1.x-orange)

An all-in-one, multi-agent assistant built on LangChain: knowledge-base Q&A (RAG), memory recall, web fetching, a code-repository assistant, scheduled tasks, and plugins (local Python plugins + MCP). Works out of the box for personal use; supports multiple users and roles for teams. FastAPI + Redis + ChromaDB + SQLite/PostgreSQL, one-command Docker deployment. (中文版见 [README.zh.md](README.zh.md))

## Why?

Putting an LLM chat loop into production surfaces three problems:

**Ambiguous intent.** When a user says "how does that work?", you have to decide whether they're asking about documentation, their chat history, a code repository, or just chatting. Pure LLM classification is slow and expensive; pure keyword matching is inaccurate. So we fuse three signals — LLM semantics + vector similarity + keyword voting — and when one path fails its weight is automatically redistributed to the survivors. Intents aren't hard-coded: every capability and plugin can register its own, and all three recognizers pick them up automatically.

**Token bloat.** Stuffing all 20 turns of history into the prompt gets slow and expensive, but keeping only the last N turns loses cross-session memory: you discussed "vector databases" three days ago, and today you ask "how fast are writes?" — the system should know what "it" refers to. The approach: Redis holds the last 10 turns; past that threshold a background task asynchronously compresses them into a summary stored in ChromaDB; on each new message, similar history is recalled and spliced into the prompt.

**Agents fail.** A 429, a timeout, or a bug in your own logic — any of them turns into a 500 for the user. Each intent binds to a chain of agents: if the primary fails, switch to the backup; if the backup fails, fall through to a fallback. At the same time, Z-score latency monitoring flags anything beyond μ+3σ, auto-downweights it out of rotation, and restores it once it recovers, so one failing agent can't take down the whole chain.

## Quick Start

```bash
git clone https://github.com/JACK-123666/synapse.git && cd synapse
cp .env.example .env      # fill in LLM_API_KEY
docker compose up -d
```

Try a message:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"What is a vector database?"}'
```

Open `http://localhost:8000/chat` for the chat UI (streaming), or `http://localhost:8000/docs` for Swagger.

**Team deployment**: set `AUTH_ENABLED=true` and `ADMIN_PASSWORD` in `.env`, exchange credentials for a token with `POST /auth/login` (or create an API key with `POST /auth/api-keys`), and let admins add members with `POST /users`. For PostgreSQL: `docker compose --profile postgres up -d` and set `DATABASE_URL`.

## Capabilities

| Capability | How to use | Under the hood |
|------------|------------|----------------|
| Knowledge Q&A (RAG) | Create a base with `POST /knowledge-bases`, upload txt/md/pdf/docx/html/code, then just ask | Chunking → embedding → permission-aware retrieval → answers with citations; private or shared bases |
| Memory recall | "What did I ask last time?" / `GET /memories/search` | Short-term Redis + long-term summaries in ChromaDB + user profile |
| Web fetching | Paste a link and say "summarize this page" | Main-content extraction (trafilatura), SSRF protection, optional save-to-knowledge-base |
| Web search | The model decides whether to search during knowledge Q&A | Function Calling, toggled by the `X-Web-Search` header |
| Repository assistant | Connect GitHub / GitLab / local Git via `POST /repos`, then "show recent commits", "summarize PR 12" | Commits, issues, PRs/MRs, file reads, code search, changelogs, indexing into a knowledge base for code Q&A; write actions require admin opt-in and per-action confirmation |
| Scheduled tasks | "Every day at 9 am, summarize new commits and push to Feishu" | APScheduler + DB persistence; prompt / web-watch / knowledge-sync jobs; Feishu / WeCom / DingTalk webhooks |
| Plugins | Drop into `plugins/` or `POST /plugins/mcp/servers` | Hot-reloadable local Python plugins; plug-and-play MCP servers; per-role tool allowlists |

## Architecture

A request flows through 6 steps:

```text
message → intent recognition → memory recall → routing → agent execution (LangChain create_agent + tools) → memory update → reply
```

### Capabilities & Plugins

Built-in capabilities and external plugins share one `Capability` interface, each registering four things: tools, intents (description / keywords / examples), dedicated agents, and routes. Intent fusion picks the dedicated agent, and that agent only sees its own toolset — so adding more tools doesn't hurt tool-selection accuracy. If a dedicated agent fails, it degrades to the general agent (all tools the user may use), then to the fallback.

| Intent | Primary agent | Failover chain |
|--------|---------------|----------------|
| knowledge_retrieval / small_talk | retrieval_agent | fallback_agent |
| summarize | summarize_agent | fallback_agent |
| memory_query / web_browse / repo_management / schedule_task | matching dedicated agent | general_agent → fallback_agent |
| general_task and plugin intents | general_agent / plugin agent | general_agent → fallback_agent |

### Intent Recognition

Three-way fusion with weighted voting:

| Method            | Weight | How it works                                         |
|-------------------|--------|------------------------------------------------------|
| LLM semantics     | 0.5    | few-shot prompt, most accurate; examples are generated when the intent catalog grows |
| Vector match      | 0.3    | embedding similarity against intent examples; the index rebuilds when examples change |
| Keywords          | 0.2    | scoring from a preset dictionary                     |

If any path fails, its weight is redistributed automatically. If all three fail, a default intent is returned.

### Routing & Failover

Each intent binds a chain of agents ordered by health score. A primary-agent failure degrades to the backup, and a backup failure falls through to the fallback. The fallback agent never calls the LLM — it returns a preset reply, guaranteeing a response in every situation. With streaming, failover is seamless as long as no token has been sent yet.

An agent's health score comes from a Z-score sliding window over the last 20 requests' latency distribution: anything beyond μ+3σ is flagged anomalous, its weight decays below 0.3 and it's removed from rotation, and a background task periodically re-checks and restores it.

### Memory

- **Short-term** (Redis): the last N turns of the current session, 24h TTL
- **Long-term** (ChromaDB): a vector index of conversation summaries; new messages recall similar history and splice it into the prompt
- **Compression**: when a session exceeds 8 turns, a background task compresses short-term memory into a summary in ChromaDB and removes only the compressed messages (anything appended meanwhile is kept)
- **Isolation**: with auth enabled, sessions and memories are isolated per user

### Models

All LLM calls go through LangChain chat models (`ChatOpenAI` for OpenAI / DeepSeek, `ChatAnthropic` for Claude). Admins can switch the global model at runtime with `POST /models/switch`; the `model` field on `/chat` affects only that request.

## Configuration

Key environment variables, all set in `.env`:

```text
LLM_PROVIDER=deepseek       # openai / claude / deepseek
LLM_API_KEY=sk-xxx          # required
LLM_MODEL=gpt-4o-mini       # effective when provider=openai
DEEPSEEK_MODEL=deepseek-chat
LLM_BASE_URL=https://api.openai.com/v1

# Embedding (DeepSeek has no embedding API: use a dedicated key, or EMBEDDING_PROVIDER=local)
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_PROVIDER=openai

# Auth (always enable for team deployments)
AUTH_ENABLED=false
ADMIN_PASSWORD=

# Database (empty = SQLite under data/)
DATABASE_URL=

# Repository write actions (off by default)
REPO_WRITE_ENABLED=false

# Memory & compression
SHORT_TERM_MAX_ROUNDS=10
SUMMARY_TRIGGER_ROUNDS=8

# Intent weights
INTENT_LLM_WEIGHT=0.5
INTENT_VECTOR_WEIGHT=0.3
INTENT_KEYWORD_WEIGHT=0.2
```

See `.env.example` for the full list.

## Project Structure

```
app/
├── main.py                     FastAPI entry; lifespan boot order (config → DB → capabilities → plugins → intent index → recovery loop)
├── config.py                   Every tunable (pydantic-settings, env / .env)
├── store.py                    Redis + ChromaDB connection singletons (Chroma has a 30s failure cooldown)
│
├── api/                        HTTP layer, one module per resource group
│   ├── chat.py                 POST /chat, POST /chat/stream (SSE), GET /chat page
│   ├── system.py               /health, /metrics, /models, /capabilities
│   ├── auth.py                 Login, JWT, API keys, /users
│   ├── knowledge.py            Knowledge bases, document upload / search / reindex
│   ├── memory.py               Memory browsing and search
│   ├── repos.py                Repo connections, commits, changelog, indexing, write confirmation
│   ├── schedules.py            Scheduled-task CRUD, cron parsing, run history
│   └── plugins.py              Plugin list / enable / reload, MCP servers, tool policies
│
├── core/                       Platform substrate — no business logic
│   ├── context.py              RequestContext in contextvars (user, session, model override)
│   ├── db.py                   SQLAlchemy 2.0 async engine, session_scope
│   ├── security.py             bcrypt, JWT, API-key hashing, Fernet encryption
│   ├── deps.py                 get_current_user / require_admin FastAPI dependencies
│   └── tasks.py                Background-task supervision: spawn() keeps strong refs, drain() on shutdown
│
├── services/
│   ├── chat.py                 The orchestration pipeline: prepare → dispatch → update memory
│   ├── users.py                Users, login, API keys
│   └── policies.py             Role → tool allow-list (persisted, loaded at startup)
│
├── llm/
│   ├── config.py               Runtime config; runtime override beats .env
│   ├── factory.py              Builds and caches ChatModel / Embeddings per resolved spec
│   ├── gateway.py              chat() / embed() entry point (single LangChain backend)
│   └── messages.py             dict ↔ LangChain message conversion
│
├── intent/                     Intent recognition
│   ├── catalog.py              Dynamic catalog: config defaults + intents registered by capabilities/plugins
│   ├── semantic.py             LLM lane (few-shot; prompt auto-generated once the catalog grows)
│   ├── vector.py               Embedding-similarity lane (Chroma; rebuilds when the catalog changes)
│   ├── keyword.py              Keyword-voting lane
│   └── blend.py                Fusion: cheap-lane short-circuit → weighted vote → reweight on lane failure
│
├── router/
│   ├── pool.py                 Agent registry + intent → agent routes
│   └── route.py                Dispatch, weighted ordering, failover (incl. streaming)
│
├── agents/
│   ├── base.py                 BaseAgent / AgentContext / AgentResponse
│   ├── langchain_agent.py      LangChainAgent: tools by tag+role, static system prompt, cached agent graph
│   ├── knowledge.py            RetrievalAgent — RAG context, chat, summarize fallback
│   ├── summary.py              SummarizationAgent — no tools, streams
│   └── safety.py               FallbackAgent — never raises, never calls the LLM
│
├── capabilities/               One package per capability; each registers tools + intents + agents + routes
│   ├── base.py                 Capability interface + CapabilityTool declaration
│   ├── manager.py              Registration / hot reload; auto-generates <capability>_agent when none is given
│   ├── core.py                 The four base agents and their routes
│   ├── knowledge/              loaders (txt/md/pdf/docx/html/code) · service (chunk → embed → retrieve) · tools
│   ├── memory/                 Memory-search tools + MemoryAgent
│   ├── web/                    fetch (SSRF-guarded extraction) · tools (search / fetch / save-to-KB)
│   ├── repo/                   providers (GitHub/GitLab/local git) · service (connections, changelog) · tools
│   └── scheduler/              cron parsing · APScheduler service · webhook delivery
│
├── plugins/
│   ├── manager.py              Local Python plugins: discovery, manifest, enable/disable, hot reload
│   └── mcp.py                  MCP servers → tools named mcp_<server>_<tool>
│
├── memory/
│   ├── recent.py               Short-term (Redis, TTL, trim)
│   ├── archive.py              Long-term summaries (Chroma) + legacy global KB
│   ├── compress.py             Threshold check → LLM summary → store → trim only what was compressed
│   └── profile.py              User profile, optimistic-locked updates
│
├── observability/
│   ├── health.py               Z-score latency window, weight decay / recovery, removal from rotation
│   └── metrics.py              Prometheus counters, histogram, gauge
│
├── tools/
│   ├── registry.py             Tool registry: source, tags, write flag, role allow-list
│   └── search.py               Web search
│
├── models/__init__.py          11 ORM tables (users, api_keys, knowledge_bases, documents, ...)
└── static/index.html           Single-file chat UI (vanilla JS, SSE)

plugins/example_plugin/         Example capability plugin (time + calculator)
tests/                          78 pytest cases — Redis / Chroma / DB / LLM all swapped for local fakes
tools/                          Offline evaluation scripts (intent accuracy, memory token cost)
```

## Extending

To add a capability or plugin, implement `Capability` and register tools, intents, and agents. Drop a plugin into `plugins/` and call `POST /plugins/reload`; see the [plugin guide](docs/PLUGIN_DEV.md) (Chinese).

```python
from langchain_core.tools import tool
from app.capabilities.base import Capability
from app.intent.catalog import IntentSpec

@tool
async def query_order(order_id: str) -> str:
    """Look up an order's status."""
    return f"Order {order_id} has shipped"

class OrderPlugin(Capability):
    name = "order"
    def tools(self):
        return [query_order]
    def intents(self):
        return [IntentSpec(name="order_query", description="Order lookups", keywords=["order"])]
```

To write just an agent: subclass `LangChainAgent` and set `tool_tags` and `build_system_prompt()`, or subclass `BaseAgent` and implement `execute()`.

## Development & Tests

```bash
python -m venv .venv && .venv/Scripts/activate     # Python 3.10 / 3.11 (chromadb 0.4.24 needs numpy<2)
pip install -r requirements-dev.txt   # core + test deps (run the app: requirements.txt)
pytest                                              # Redis / Chroma / LLM all replaced by local fakes
python tools/eval_intent.py --no-llm                # intent-recognition evaluation
python tools/eval_memory.py                         # memory token evaluation
```

## License

MIT
