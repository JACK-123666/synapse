"""测试公共夹具。

外部依赖全部替换为本地替身：
- Redis    → fakeredis（异步）
- ChromaDB → EphemeralClient（内存）
- 数据库   → 临时目录下的 SQLite
- LLM      → 可编排回复的假 ChatModel（支持工具调用）
- Embedding→ 由文本哈希生成的确定性向量
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, List

# 必须在导入 app 之前设置环境变量（get_settings 带缓存）
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="synapse-test-"))
os.environ.setdefault("DATA_DIR", str(_TMP_ROOT / "data"))
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_API_KEY", "sk-test")
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("PLUGINS_DIR", str(_TMP_ROOT / "plugins"))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402


class FakeToolChatModel(GenericFakeChatModel):
    """可编排回复的假模型：bind_tools 直接返回自身，按顺序吐出预设消息。"""

    def bind_tools(self, tools: Any, **kwargs: Any):  # type: ignore[override]
        return self


def fake_model(*replies: Any) -> FakeToolChatModel:
    messages: List[AIMessage] = [
        r if isinstance(r, AIMessage) else AIMessage(content=str(r)) for r in replies
    ]
    return FakeToolChatModel(messages=iter(messages))


def fake_vector(text: str, dim: int = 64) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * (dim // len(digest) + 1))[:dim]
    return [b / 255.0 for b in raw]


@pytest.fixture
def patch_model(monkeypatch):
    """用法：patch_model("回复1", AIMessage(tool_calls=[...]), ...)。"""

    def _apply(*replies: Any) -> FakeToolChatModel:
        model = fake_model(*replies)
        factory = lambda *a, **k: model  # noqa: E731
        monkeypatch.setattr("app.llm.factory.get_chat_model", factory)
        monkeypatch.setattr("app.agents.langchain_agent.get_chat_model", factory)
        return model

    return _apply


@pytest.fixture
def fake_embeddings(monkeypatch):
    from app.llm.gateway import get_llm_client

    llm = get_llm_client()

    async def _embed(text: str) -> List[float]:
        return fake_vector(text)

    async def _embed_batch(texts: Iterable[str]) -> List[List[float]]:
        return [fake_vector(t) for t in texts]

    monkeypatch.setattr(llm, "embed", _embed)
    monkeypatch.setattr(llm, "embed_batch", _embed_batch)
    return llm


@pytest.fixture
def fake_redis(monkeypatch):
    import fakeredis

    from app import store

    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(store, "_redis", client)
    return client


@pytest.fixture
def chroma(monkeypatch):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from app import store

    client = chromadb.EphemeralClient(
        ChromaSettings(allow_reset=True, anonymized_telemetry=False)
    )
    client.reset()
    monkeypatch.setattr(store, "_chroma", client)
    return client


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """每个测试一个独立的 SQLite 数据库，并创建初始管理员。"""
    from app.config import get_settings
    from app.core import db as core_db
    from app.services.users import bootstrap_admin

    settings = get_settings()
    monkeypatch.setattr(
        settings, "database_url", f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}"
    )
    monkeypatch.setattr(settings, "admin_password", "admin-pass")
    core_db.reset_engine_for_tests()
    await core_db.init_db()
    await bootstrap_admin()
    yield
    await core_db.close_db()


@pytest.fixture
def auth_enabled(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "auth_enabled", True)
    yield


@pytest.fixture(autouse=True)
def _isolate_registries(monkeypatch):
    """每个测试使用全新的工具注册表 / 意图目录 / Agent 注册表 / 能力管理器。"""
    from app.agents import langchain_agent
    from app.capabilities import manager as cap_manager
    from app.capabilities.knowledge import service as kb_service
    from app.capabilities.repo import service as repo_service
    from app.capabilities.scheduler import service as scheduler_service
    from app.intent import blend, catalog
    from app.memory import archive, compress, profile, recent
    from app.observability import health
    from app.router import pool, route
    from app.services import chat
    from app.tools import registry

    for module in (archive, compress, profile, recent):
        monkeypatch.setattr(module, "_instance", None)
    for module in (kb_service, repo_service, scheduler_service, chat):
        monkeypatch.setattr(module, "_service", None)
    monkeypatch.setattr(registry, "_registry", None)
    monkeypatch.setattr(catalog, "_catalog", None)
    monkeypatch.setattr(health, "_detector", None)
    monkeypatch.setattr(pool, "_instance", None)
    monkeypatch.setattr(route, "_instance", None)
    monkeypatch.setattr(blend, "_instance", None)
    monkeypatch.setattr(cap_manager, "_manager", None)

    # Agent 图缓存按 (模型配置, 工具集, system_prompt) 复用，跨测试必须清空，
    # 否则后续用例会拿到前面用例注入的假模型编译出来的图
    langchain_agent.clear_agent_cache()
    yield
