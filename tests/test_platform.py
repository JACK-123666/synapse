"""阶段 2：平台底座（鉴权、用户、意图目录、能力注册、对话编排）。"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.intent.catalog import IntentSpec, get_intent_catalog


def _auth_app() -> FastAPI:
    from app.api.auth import router, users_router
    from app.api.system import router as system_router

    app = FastAPI()
    app.include_router(router)
    app.include_router(users_router)
    app.include_router(system_router)
    return app


async def test_auth_disabled_is_local_admin(db):
    client = TestClient(_auth_app())
    me = client.get("/auth/me").json()
    assert me["role"] == "admin"
    assert client.get("/auth/config").json() == {"auth_enabled": False}


async def test_auth_login_api_key_and_roles(db, auth_enabled):
    client = TestClient(_auth_app())
    assert client.get("/auth/me").status_code == 401
    assert client.post("/auth/login", json={"username": "admin", "password": "bad"}).status_code == 401

    token = client.post("/auth/login", json={"username": "admin", "password": "admin-pass"}).json()["access_token"]
    admin = {"Authorization": f"Bearer {token}"}
    assert client.get("/auth/me", headers=admin).json()["username"] == "admin"

    # 管理员创建普通用户
    created = client.post("/users", json={"username": "alice", "password": "alice-pass"}, headers=admin)
    assert created.status_code == 200, created.text

    alice_token = client.post("/auth/login", json={"username": "alice", "password": "alice-pass"}).json()["access_token"]
    alice = {"Authorization": f"Bearer {alice_token}"}
    assert client.get("/users", headers=alice).status_code == 403
    # 普通用户不能切换模型
    assert client.post("/models/switch", json={"model": "x"}, headers=alice).status_code == 403

    # API Key：创建 → 使用 → 吊销
    key = client.post("/auth/api-keys", json={"name": "ci"}, headers=alice).json()
    assert key["api_key"].startswith("syn-")
    assert client.get("/auth/me", headers={"X-API-Key": key["api_key"]}).json()["username"] == "alice"
    assert client.get("/auth/me", headers={"Authorization": f"Bearer {key['api_key']}"}).status_code == 200
    assert client.delete(f"/auth/api-keys/{key['id']}", headers=alice).status_code == 200
    assert client.get("/auth/me", headers={"X-API-Key": key["api_key"]}).status_code == 401


async def test_initial_admin_cannot_be_deleted(db):
    from app.core.context import LOCAL_ADMIN_ID
    from app.services import users

    with pytest.raises(users.UserError):
        await users.delete_user(LOCAL_ADMIN_ID)


# ---- 意图目录 ----


async def test_catalog_drives_keyword_recognizer():
    from app.intent.keyword import KeywordIntentRecognizer

    catalog = get_intent_catalog()
    recognizer = KeywordIntentRecognizer()
    assert await recognizer.recognize("帮我部署一下服务") is None

    catalog.register(IntentSpec(name="deploy", description="部署", keywords=["部署"], source="plugin:x"))
    scores = await recognizer.recognize("帮我部署一下服务")
    assert max(scores, key=scores.get) == "deploy"

    catalog.unregister_source("plugin:x")
    assert "deploy" not in catalog.names()


def test_catalog_version_changes_with_examples():
    catalog = get_intent_catalog()
    v1 = catalog.version()
    catalog.register(IntentSpec(name="x", description="x", examples=["示例"], source="plugin:x"))
    assert catalog.version() != v1


def test_semantic_prompt_default_and_dynamic():
    from app.intent.semantic import LLMIntentRecognizer

    recognizer = LLMIntentRecognizer()
    default_prompt = recognizer._build_system_prompt()
    assert "示例 3" in default_prompt and "repo_management" not in default_prompt

    get_intent_catalog().register(
        IntentSpec(name="repo_management", description="代码仓库", examples=["看看最近的提交"], source="builtin:repo")
    )
    dynamic = recognizer._build_system_prompt()
    assert "repo_management" in dynamic and "看看最近的提交" in dynamic


async def test_vector_index_rebuilds_when_catalog_changes(chroma):
    """使用 ChromaDB 内置本地模型：首次建索引 → 版本一致跳过 → 示例变化后重建。"""
    from app.config import get_settings
    from app.intent.vector import VectorIntentRecognizer

    name = get_settings().chroma_collection_intents
    catalog = get_intent_catalog()

    await VectorIntentRecognizer().initialize()
    collection = chroma.get_collection(name)
    first_count = collection.count()
    assert collection.metadata["catalog_version"] == catalog.version()

    # 版本一致：新实例初始化时不重建（记录数不变）
    await VectorIntentRecognizer().initialize()
    assert chroma.get_collection(name).count() == first_count

    # 新增意图示例后刷新：重建索引并包含新示例
    catalog.register(IntentSpec(
        name="repo_management", description="代码仓库",
        examples=["看看 synapse 仓库最近的提交记录"], source="builtin:repo",
    ))
    recognizer = VectorIntentRecognizer()
    await recognizer.refresh()
    rebuilt = chroma.get_collection(name)
    assert rebuilt.count() == first_count + 1
    assert rebuilt.metadata["catalog_version"] == catalog.version()
    scores = await recognizer.recognize("看看 synapse 仓库最近的提交记录")
    assert max(scores, key=scores.get) == "repo_management"


def test_intent_short_circuit_criteria():
    """短路判据：两路结论一致、关键词够集中、向量原始相似度够高，缺一不可。"""
    from app.intent.blend import IntentFusion

    fusion = IntentFusion()
    ok = fusion._try_short_circuit({"small_talk": 1.0}, {"small_talk": 0.5, "x": 0.5}, 0.9)
    assert ok is not None, "一致且证据充分时应当短路"
    assert ok[0] == "small_talk"

    # 两路结论不一致
    assert fusion._try_short_circuit({"small_talk": 1.0}, {"summarize": 1.0}, 0.9) is None
    # 关键词命中被打散（消息有歧义）
    assert fusion._try_short_circuit({"a": 0.5, "b": 0.5}, {"a": 1.0}, 0.9) is None
    # 向量原始相似度不足：注意此处归一化分布是 1.0 也仍然不能短路——
    # 这正是曾经让短路永远不触发的缺陷（拿归一化分布当高置信判据）
    assert fusion._try_short_circuit({"a": 1.0}, {"a": 1.0}, 0.3) is None
    # 关键回归：向量 top-1 与关键词一致，但归一化分布被摊薄到 0.45（< 0.8 阈值），
    # 而原始相似度高达 0.8。旧实现拿归一化分布当判据，这里会误判为"不够自信"，
    # 导致短路永远不触发；现在应当正常短路。
    assert fusion._try_short_circuit({"a": 1.0}, {"a": 0.45, "b": 0.35, "c": 0.2}, 0.8) is not None
    # 任一路缺失
    assert fusion._try_short_circuit(None, {"a": 1.0}, 0.9) is None
    assert fusion._try_short_circuit({"a": 1.0}, None, 0.9) is None


# ---- 能力注册 ----


async def test_capability_registration_and_auto_agent():
    from langchain_core.tools import tool

    from app.capabilities.base import Capability
    from app.capabilities.core import CoreCapability
    from app.capabilities.manager import get_capability_manager
    from app.router.pool import get_agent_registry
    from app.tools.registry import get_tool_registry

    @tool
    def now_time() -> str:
        """当前时间。"""
        return "12:00"

    class TimeCapability(Capability):
        name = "timecap"
        description = "时间工具"

        def tools(self):
            return [now_time]

        def intents(self):
            return [IntentSpec(name="ask_time", description="询问时间", keywords=["几点"])]

    manager = get_capability_manager()
    await manager.register(CoreCapability())
    await manager.register(TimeCapability(), source="plugin:timecap")

    routes = get_agent_registry().get_routes()
    assert routes["knowledge_retrieval"] == ["retrieval_agent", "fallback_agent"]
    assert routes["ask_time"] == ["timecap_agent", "general_agent", "fallback_agent"]
    assert get_tool_registry().entry("now_time").source == "plugin:timecap"

    await manager.unregister("timecap")
    assert "ask_time" not in get_agent_registry().get_routes()
    assert get_tool_registry().get("now_time") is None
    assert "ask_time" not in get_intent_catalog().names()


async def test_knowledge_capability_keeps_core_route():
    from app.capabilities.core import CoreCapability
    from app.capabilities.knowledge import KnowledgeCapability
    from app.capabilities.manager import get_capability_manager
    from app.router.pool import get_agent_registry

    manager = get_capability_manager()
    await manager.register(CoreCapability())
    await manager.register(KnowledgeCapability())
    assert get_agent_registry().get_routes()["knowledge_retrieval"] == ["retrieval_agent", "fallback_agent"]
    assert "知识库" in get_intent_catalog().get("knowledge_retrieval").keywords


# ---- 对话编排 ----


async def test_chat_service_isolates_sessions_and_model_override(
    db, fake_redis, auth_enabled, patch_model, monkeypatch
):
    from app.capabilities.core import CoreCapability
    from app.capabilities.manager import get_capability_manager
    from app.core.deps import CurrentUser
    from app.intent.blend import get_intent_fusion
    from app.llm.gateway import get_llm_client
    from app.memory.recent import get_short_term_memory
    from app.services.chat import ChatInput, ChatService

    await get_capability_manager().register(CoreCapability())

    async def fake_recognize(message):
        return "small_talk", 0.9

    monkeypatch.setattr(get_intent_fusion(), "recognize", fake_recognize)

    seen_models = []

    def model_factory(*args, model_override=None, **kwargs):
        from app.llm.factory import resolve_model_spec
        from tests.conftest import fake_model

        seen_models.append(resolve_model_spec(model_override).model)
        return fake_model("你好呀")

    monkeypatch.setattr("app.agents.langchain_agent.get_chat_model", model_factory)

    llm = get_llm_client()
    before = llm.get_config()["model"]
    user = CurrentUser(id="u-1", username="alice", role="user")
    result = await ChatService().chat(user, ChatInput(session_id="s1", message="你好", model="temp-model"))

    assert result.reply == "你好呀"
    assert result.agent_used == "retrieval_agent"
    assert seen_models == ["temp-model"]
    assert llm.get_config()["model"] == before

    # 开启鉴权时，短期记忆按用户隔离
    assert len(await get_short_term_memory().get_messages("u-1:s1")) == 2
    assert await get_short_term_memory().get_messages("s1") == []
