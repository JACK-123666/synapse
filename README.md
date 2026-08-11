# Synapse

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A multi-agent conversational service with intent recognition and memory management. FastAPI + Redis + ChromaDB, one-command Docker deployment. (中文版见 [README.zh.md](README.zh.md))

## Why?

Putting an LLM chat loop into production surfaces three problems:

**Ambiguous intent.** When a user says "how does that work?", you have to decide whether they're asking about documentation or making small talk. Pure LLM classification is slow and expensive; pure keyword matching is inaccurate. So we fuse three signals — LLM semantics + vector similarity + keyword voting — and when one path fails its weight is automatically redistributed to the survivors. Intent recognition is never a single point of failure.

**Token bloat.** Stuffing all 20 turns of history into the prompt gets slow and expensive, but keeping only the last N turns loses cross-session memory: you discussed "vector databases" three days ago, and today you ask "how fast are writes?" — the system should know what "it" refers to. The approach: Redis holds the last 10 turns; past that threshold a background task asynchronously compresses them into a summary stored in ChromaDB; on each new message, similar history is recalled and spliced into the prompt.

**Agents fail.** A 429, a timeout, or a bug in your own logic — any of them turns into a 500 for the user. Each intent binds to a chain of agents: if the primary fails, switch to the backup; if the backup fails, fall through to a fallback. At the same time, Z-score latency monitoring flags anything beyond μ+3σ, auto-downweights it out of rotation, and restores it once it recovers. When OpenAI hiccups at 3 a.m., the system switches itself to DeepSeek — and back again when it's healthy.

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

Open `http://localhost:8000/chat` for the chat UI, or `http://localhost:8000/docs` for Swagger.

## Architecture

A request flows through 6 steps:

```text
message → intent recognition → memory recall → routing → agent execution → memory update → reply
```

### Intent Recognition

Three-way fusion with weighted voting:

| Method            | Weight | How it works                                         |
|-------------------|--------|------------------------------------------------------|
| LLM semantics     | 0.5    | few-shot prompt, most accurate                       |
| Vector match      | 0.3    | embedding similarity against intent examples         |
| Keywords          | 0.2    | scoring from a preset dictionary                     |

If any path fails, its weight is redistributed automatically. If all three fail, a default intent is returned.

### Routing & Failover

Each intent binds a chain of agents ordered by health score. A primary-agent failure degrades to the backup, and a backup failure falls through to the fallback. The fallback agent never calls the LLM — it returns a preset reply, guaranteeing a response in every situation.

An agent's health score comes from a Z-score sliding window over the last 20 requests' latency distribution: anything beyond μ+3σ is flagged anomalous, its weight decays below 0.3 and it's removed from rotation, and a background task periodically re-checks and restores it.

### Memory

- **Short-term** (Redis): the last N turns of the current session, 24h TTL
- **Long-term** (ChromaDB): a vector index of conversation summaries; new messages recall similar history and splice it into the prompt
- **Compression**: when a session exceeds 8 turns, a background task compresses short-term memory into a summary in ChromaDB and clears Redis to free up the token budget

### Web Search

When the knowledge base misses, DuckDuckGo is searched automatically and the results are injected into the prompt. Toggleable from the frontend.

## Configuration

Key environment variables, all set in `.env`:

```text
LLM_PROVIDER=deepseek       # openai / claude / deepseek
LLM_API_KEY=sk-xxx          # required
LLM_MODEL=gpt-4o-mini       # effective when provider=openai
DEEPSEEK_MODEL=deepseek-chat
LLM_BASE_URL=https://api.openai.com/v1

# Optional: a dedicated embedding key (DeepSeek doesn't support embedding)
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=

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
├── main.py                 FastAPI entry, registers agents and routes on startup
├── config.py               All configurable parameters
├── store.py                Redis / ChromaDB connection singletons
├── api/chat.py             /chat /health /models /metrics
├── llm/gateway.py          Unified LLM client (runtime switching supported)
├── intent/
│   ├── blend.py            Three-way fusion
│   ├── semantic.py         LLM semantic classification
│   ├── keyword.py          Keyword voting
│   └── vector.py           Vector similarity
├── router/
│   ├── pool.py             Agent registry
│   └── route.py            Dispatch & failover
├── agents/
│   ├── base.py             Base class
│   ├── knowledge.py        Knowledge retrieval
│   ├── summary.py          Summarization
│   └── safety.py           Fallback
├── memory/
│   ├── recent.py           Short-term memory (Redis)
│   ├── archive.py          Long-term memory (ChromaDB)
│   ├── compress.py         Compression scheduler
│   └── profile.py          User profile
├── observability/
│   ├── health.py           Anomaly detection & self-healing
│   └── metrics.py          Prometheus metrics
├── tools/
│   └── search.py           Web search
└── _singleton.py           Utilities
```

## Extending with a New Agent

Subclass `BaseAgent` and implement `execute()`:

```python
from app.agents.base import BaseAgent, AgentContext, AgentResponse

class MyAgent(BaseAgent):
    agent_id = "my_agent"
    description = "A custom agent"

    async def execute(self, context: AgentContext) -> AgentResponse:
        # your logic
        return AgentResponse(reply="done", metadata={})
```

Then register and bind it to a route in the `main.py` startup event.

## License

MIT
