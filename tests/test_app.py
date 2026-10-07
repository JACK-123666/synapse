"""阶段 7：完整应用启动 + 对话（含流式）端到端测试。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent

_INTENT_JSON = '{"knowledge_retrieval": 0.05, "summarize": 0.05, "small_talk": 0.90}'


def _parse_sse(text: str):
    events = []
    for block in text.strip().split("\n\n"):
        data = [line[5:].strip() for line in block.splitlines() if line.startswith("data:")]
        if data:
            events.append(json.loads("".join(data)))
    return events


async def test_full_app_startup_and_chat(db, fake_redis, chroma, fake_embeddings, patch_model, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "plugins_dir", str(ROOT / "plugins"))
    from app.plugins import manager as plugin_module
    from app.plugins import mcp as mcp_module

    monkeypatch.setattr(plugin_module, "_manager", None)
    monkeypatch.setattr(mcp_module, "_manager", None)
    # 本用例的假模型按调用顺序排队回复，而意图识别的短路会省掉一次 LLM 调用、
    # 导致队列错位。这里显式关掉短路保持确定性；短路行为本身由
    # test_platform.py::test_intent_short_circuit_criteria 专项覆盖。
    monkeypatch.setattr(get_settings(), "intent_short_circuit", False)
    patch_model(_INTENT_JSON, "你好！很高兴见到你", _INTENT_JSON, "流式 你好 呀")

    from app.main import app

    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "healthy", health
        assert health["modules"]["database"] == "connected"
        assert "retrieval_agent" in health["modules"]["agents"]

        caps = client.get("/capabilities").json()
        names = {c["name"] for c in caps["capabilities"]}
        assert {"core", "knowledge", "memory", "web", "repo", "scheduler", "example_plugin"} <= names
        routes = caps["routes"]
        assert routes["knowledge_retrieval"] == ["retrieval_agent", "fallback_agent"]
        assert routes["repo_management"][0] == "repo_agent"
        assert routes["utility_tools"][0] == "example_plugin_agent"

        # 一次性对话
        resp = client.post("/chat", json={"session_id": "e2e", "message": "你好"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["reply"] == "你好！很高兴见到你"
        assert data["intent"] == "small_talk" and data["agent_used"] == "retrieval_agent"

        # 流式对话
        stream = client.post("/chat/stream", json={"session_id": "e2e", "message": "你好"})
        assert stream.status_code == 200
        events = _parse_sse(stream.text)
        types = [e["type"] for e in events]
        assert types[0] == "meta" and types[-1] == "done" and "token" in types
        assert "".join(e["content"] for e in events if e["type"] == "token") == "流式 你好 呀"
        assert events[-1]["agent_used"] == "retrieval_agent"

        # 两轮对话都写入了短期记忆
        history = client.get("/memories/sessions/e2e").json()
        assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]

        # 首页与聊天页
        assert client.get("/").status_code == 200
        assert "Synapse" in client.get("/chat").text
        assert client.get("/plugins/tools").status_code == 200
