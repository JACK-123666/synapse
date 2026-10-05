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
├── main.py                 FastAPI entry (lifespan: DB, capabilities, plugins, scheduler)
├── config.py               All configurable parameters
├── store.py                Redis / ChromaDB connection singletons
├── api/                    chat (incl. /chat/stream), system, auth, knowledge, memory, repos, schedules, plugins
├── core/                   Request context, database, security (JWT / API keys / encryption), auth dependencies, background-task supervision
├── models/                 ORM: users, knowledge bases, repo connections, schedules, plugin state, ...
├── services/               Chat orchestration, users, tool policies
├── llm/
│   ├── config.py           Runtime LLM config (runtime override > .env)
│   ├── gateway.py          Unified LLM entry point (chat / embed, single LangChain backend)
│   └── factory.py          LangChain ChatModel / Embeddings factory
├── intent/                 Three-way fusion + dynamic intent catalog (catalog.py)
├── router/                 Agent registry, dispatch & failover (incl. streaming)
├── agents/                 BaseAgent, LangChainAgent, retrieval, summarization, general, fallback
├── capabilities/           Built-ins: core / knowledge / memory / web / repo / scheduler
├── plugins/                Local plugin manager, MCP integration
├── memory/                 Short-term / long-term / compression / profile
├── observability/          Anomaly detection & self-healing, Prometheus metrics
├── tools/                  Tool registry, web search
└── static/index.html       Chat UI
plugins/example_plugin/     Example plugin
tests/                      pytest (all external services replaced by local fakes)
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
