"""阶段 1：LangChain 化（模型工厂、LLMClient、LangChainAgent、调度器、工具注册表）。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.agents.base import AgentContext, AgentResponse, BaseAgent
from app.agents.langchain_agent import LangChainAgent
from app.core.context import RequestContext, request_context
from app.llm import factory
from app.llm.gateway import get_llm_client


@pytest.fixture(autouse=True)
def _reset_llm_runtime():
    llm = get_llm_client()
    llm.reset_runtime()
    factory.clear_model_cache()
    yield
    llm.reset_runtime()
    factory.clear_model_cache()


# ---- 模型工厂 ----


def test_model_spec_runtime_and_request_override():
    llm = get_llm_client()
    llm.switch_model(model="global-model")
    assert factory.resolve_model_spec().model == "global-model"

    with request_context(RequestContext(model_override="request-model")):
        assert factory.resolve_model_spec().model == "request-model"
        # 请求级覆盖不修改全局状态
        assert llm.get_config()["model"] == "global-model"

    assert factory.resolve_model_spec().model == "global-model"


def test_build_models_per_provider():
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    llm = get_llm_client()
    openai_model = factory.get_chat_model(temperature=0.2, max_tokens=100)
    assert isinstance(openai_model, ChatOpenAI)
    # 相同配置复用同一实例
    assert factory.get_chat_model(temperature=0.2, max_tokens=100) is openai_model

    llm.switch_model(provider="deepseek")
    deepseek = factory.get_chat_model()
    assert isinstance(deepseek, ChatOpenAI)
    assert "deepseek" in str(deepseek.openai_api_base)

    llm.switch_model(provider="claude")
    claude = factory.get_chat_model()
    assert isinstance(claude, ChatAnthropic)
    assert not str(claude.anthropic_api_url).rstrip("/").endswith("/v1")


async def test_gateway_chat_uses_langchain(patch_model):
    patch_model("你好，我是假模型")
    reply = await get_llm_client().chat([{"role": "user", "content": "hi"}], system="sys")
    assert reply == "你好，我是假模型"


# ---- LangChainAgent ----


@tool("echo_tool", response_format="content_and_artifact")
async def echo_tool(text: str) -> Tuple[str, Dict[str, Any]]:
    """原样返回文本。"""
    return f"echo:{text}", {"echoed": text}


class EchoAgent(LangChainAgent):
    agent_id = "echo_agent"
    description = "测试 Agent"
    tool_tags = ("test",)


def _register_echo_tool():
    from app.tools.registry import get_tool_registry

    get_tool_registry().register(echo_tool, tags=("test",))


def _tool_call_message() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "echo_tool", "args": {"text": "abc"}, "id": "call_1"}],
    )


async def test_langchain_agent_tool_calling(patch_model):
    _register_echo_tool()
    patch_model(_tool_call_message(), "最终回答")
    response = await EchoAgent().execute(AgentContext(session_id="s", message="测试"))
    assert response.reply == "最终回答"
    assert response.metadata["function_calling"] is True
    assert response.metadata["tool_calls"][0]["name"] == "echo_tool"
    assert response.metadata["_artifacts"]["echo_tool"] == [{"echoed": "abc"}]


async def test_langchain_agent_stream(patch_model):
    _register_echo_tool()
    patch_model(_tool_call_message(), "流式回答")
    events = [e async for e in EchoAgent().stream(AgentContext(session_id="s", message="测试"))]
    types = [e["type"] for e in events]
    assert "tool_start" in types and "tool_end" in types
    assert types[-1] == "final"
    assert "".join(e["content"] for e in events if e["type"] == "token") == "流式回答"
    assert events[-1]["response"].reply == "流式回答"


async def test_retrieval_agent_collects_web_sources(patch_model):
    from app.agents.knowledge import RetrievalAgent
    from app.tools.registry import get_tool_registry

    @tool("web_search", response_format="content_and_artifact")
    async def fake_web_search(query: str) -> Tuple[str, List[Dict[str, str]]]:
        """假联网搜索。"""
        return "结果", [{"title": "T", "url": "https://example.com", "snippet": "S"}]

    get_tool_registry().register(fake_web_search, tags=("web",))
    patch_model(
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "c1"}]),
        "综合回答",
    )
    ctx = AgentContext(session_id="s", message="最新的新闻", intent="knowledge_retrieval")
    response = await RetrievalAgent().execute(ctx)
    assert response.reply == "综合回答"
    assert response.metadata["mode"] == "knowledge_retrieval"
    assert response.metadata["web_sources"][0]["url"] == "https://example.com"


async def test_summarization_agent_empty_content():
    from app.agents.summary import SummarizationAgent

    response = await SummarizationAgent().execute(AgentContext(session_id="s", message="  "))
    assert "没有足够的内容" in response.reply


# ---- 调度器 ----


class BrokenAgent(BaseAgent):
    agent_id = "broken_agent"
    description = "总是失败"

    async def execute(self, context: AgentContext) -> AgentResponse:
        raise RuntimeError("boom")


def _error_count(agent_id: str) -> float:
    from app.observability.metrics import synapse_request_total

    return synapse_request_total.labels(agent_id=agent_id, status="error")._value.get()


async def test_dispatcher_fallback_counts_error_once():
    from app.agents.safety import FallbackAgent
    from app.router.pool import get_agent_registry
    from app.router.route import get_task_dispatcher

    registry = get_agent_registry()
    registry.register_agent(BrokenAgent())
    registry.register_agent(FallbackAgent())
    registry.register_route("small_talk", ["broken_agent", "fallback_agent"])

    before = _error_count("broken_agent")
    response = await get_task_dispatcher().dispatch(
        AgentContext(session_id="s", message="hi", intent="small_talk")
    )
    assert response.metadata["agent_id"] == "fallback_agent"
    assert _error_count("broken_agent") - before == 1


async def test_dispatch_stream_falls_back_before_first_token():
    from app.agents.safety import FallbackAgent
    from app.router.pool import get_agent_registry
    from app.router.route import get_task_dispatcher

    registry = get_agent_registry()
    registry.register_agent(BrokenAgent())
    registry.register_agent(FallbackAgent())
    registry.register_route("small_talk", ["broken_agent", "fallback_agent"])

    events = [
        e async for e in get_task_dispatcher().dispatch_stream(
            AgentContext(session_id="s", message="hi", intent="small_talk")
        )
    ]
    assert events[-1]["type"] == "final"
    assert events[-1]["agent_id"] == "fallback_agent"


# ---- 工具注册表 ----


def test_tool_registry_policy_and_write_filter():
    from app.tools.registry import get_tool_registry

    @tool
    def read_thing() -> str:
        """读取。"""
        return "r"

    @tool
    def write_thing() -> str:
        """写入。"""
        return "w"

    registry = get_tool_registry()
    registry.register(read_thing, tags=("x",))
    registry.register(write_thing, tags=("x",), write=True)

    names = lambda tools: sorted(t.name for t in tools)  # noqa: E731
    assert names(registry.tools_for("user", tags=["x"])) == ["read_thing", "write_thing"]
    assert names(registry.tools_for("user", tags=["x"], include_write=False)) == ["read_thing"]

    registry.set_policy("user", ["read_*"])
    assert names(registry.tools_for("user", tags=["x"])) == ["read_thing"]
    # admin 不受白名单限制
    assert names(registry.tools_for("admin", tags=["x"])) == ["read_thing", "write_thing"]
