"""阶段 5：定时任务。"""

from __future__ import annotations

from datetime import datetime

import pytest
from langchain_core.messages import AIMessage

from app.capabilities.knowledge.service import Principal
from app.capabilities.scheduler.cron import (
    ScheduleError,
    build_trigger,
    parse_rule_based,
    to_cron,
    validate_cron,
)
from app.capabilities.scheduler.service import build_webhook_payload, get_scheduler_service
from app.core.context import LOCAL_ADMIN_ID

ADMIN = Principal(LOCAL_ADMIN_ID, is_admin=True)


@pytest.fixture(autouse=True)
def _fresh_scheduler_service(monkeypatch):
    from app.capabilities.scheduler import service

    monkeypatch.setattr(service, "_service", None)
    yield


# ---- cron ----


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("0 9 * * 1", "2026-10-12 Mon"),   # 标准 crontab：1 = 周一
        ("0 9 * * 0", "2026-10-11 Sun"),   # 0 = 周日
        ("0 9 * * 7", "2026-10-11 Sun"),   # 7 也是周日
        ("0 9 * * 1-5", "2026-10-06 Tue"),
        ("0 9 * * 6,0", "2026-10-10 Sat"),
    ],
)
def test_weekday_follows_standard_crontab(expr, expected):
    trigger = build_trigger(expr)
    now = datetime(2026, 10, 5, 12, 0, tzinfo=trigger.timezone)  # 2026-10-05 是周一
    assert trigger.get_next_fire_time(None, now).strftime("%Y-%m-%d %a") == expected


@pytest.mark.parametrize(
    "text, cron",
    [
        ("每天早上 9 点", "0 9 * * *"),
        ("每天晚上八点半", "30 20 * * *"),
        ("每周一上午10:30", "30 10 * * 1"),
        ("每周日下午3点", "0 15 * * 0"),
        ("每隔 15 分钟", "*/15 * * * *"),
        ("每 2 小时", "0 */2 * * *"),
        ("每小时", "0 * * * *"),
        ("工作日下午 6 点", "0 18 * * 1-5"),
        ("每月 1 号 8 点", "0 8 1 * *"),
        ("every day at 7:15", "15 7 * * *"),
    ],
)
def test_rule_based_parsing(text, cron):
    assert parse_rule_based(text) == cron


async def test_to_cron_passthrough_and_invalid():
    assert await to_cron("0  9 * *  1") == "0 9 * * 1"
    with pytest.raises(ScheduleError):
        validate_cron("61 9 * * *")


async def test_to_cron_llm_fallback(patch_model):
    patch_model(AIMessage(content="", tool_calls=[{
        "name": "CronSpec", "args": {"cron": "0 7 * * 6", "explanation": "每周六 7 点"}, "id": "c1",
    }]))
    assert await to_cron("每逢周六清晨七点") == "0 7 * * 6"


def test_webhook_payload_formats():
    assert build_webhook_payload("https://open.feishu.cn/open-apis/bot/v2/hook/x", "T", "C")["msg_type"] == "text"
    assert build_webhook_payload("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=x", "T", "C")["msgtype"] == "text"
    assert build_webhook_payload("https://oapi.dingtalk.com/robot/send?access_token=x", "T", "C")["text"]["content"].startswith("T")
    assert build_webhook_payload("https://example.com/hook", "T", "C") == {"title": "T", "content": "C"}


# ---- 服务 ----


async def test_crud_and_job_registration(db, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "scheduler_enabled", True)
    service = get_scheduler_service()
    await service.start()
    try:
        created = await service.create(
            ADMIN, name="日报", cron="0 9 * * *", payload={"prompt": "总结昨天的提交"}
        )
        assert created["id"] in service.job_ids()
        assert len(created["next_runs"]) == 3

        with pytest.raises(ScheduleError):
            await service.create(ADMIN, name="日报", cron="0 9 * * *", payload={"prompt": "x"})
        with pytest.raises(ScheduleError):
            await service.create(ADMIN, name="坏任务", cron="0 9 * * *", kind="prompt", payload={})

        await service.update(ADMIN, created["id"], enabled=False)
        assert created["id"] not in service.job_ids()
        await service.update(ADMIN, created["id"], enabled=True, cron="30 8 * * 1-5")
        assert created["id"] in service.job_ids()

        # 重启后从数据库重新加载
        await service.shutdown()
        await service.start()
        assert created["id"] in service.job_ids()

        await service.delete(ADMIN, created["id"])
        assert created["id"] not in service.job_ids()
    finally:
        await service.shutdown()


async def test_prompt_job_runs_as_owner_and_pushes_webhook(db, fake_redis, monkeypatch):
    from app.services import chat as chat_module
    from app.services.chat import ChatResult

    calls = {}

    class FakeChatService:
        async def chat(self, user, inp):
            calls["user"], calls["message"], calls["session"] = user.id, inp.message, inp.session_id
            return ChatResult(reply="昨天有 3 个提交", intent="repo_management", agent_used="repo_agent", confidence=0.9)

    monkeypatch.setattr(chat_module, "get_chat_service", lambda: FakeChatService())
    service = get_scheduler_service()
    pushed = []

    async def fake_push(url, title, content):
        pushed.append((url, title, content))

    monkeypatch.setattr(service, "_push_webhook", fake_push)
    created = await service.create(
        ADMIN, name="提交日报", cron="0 9 * * *", payload={"prompt": "总结昨天的提交"},
        webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/abc",
    )
    run = await service.run_now(ADMIN, created["id"])
    assert run["status"] == "success" and run["output"] == "昨天有 3 个提交"
    assert calls == {"user": LOCAL_ADMIN_ID, "message": "总结昨天的提交", "session": created["session_id"]}
    assert pushed and "提交日报" in pushed[0][1]
    detail = await service.get(ADMIN, created["id"])
    assert detail["last_status"] == "success"
    assert (await service.runs(ADMIN, created["id"]))[0]["status"] == "success"


async def test_lock_prevents_concurrent_runs(db, fake_redis):
    service = get_scheduler_service()
    created = await service.create(ADMIN, name="锁测试", cron="0 9 * * *", payload={"prompt": "x"})
    await fake_redis.set(f"synapse:schedule_lock:{created['id']}", "1")
    run = await service.execute(created["id"])
    assert run["status"] == "skipped"


async def test_web_watch_detects_changes(db, fake_redis, patch_model, monkeypatch):
    from app.capabilities.web import fetch
    from app.capabilities.web.fetch import FetchResult

    pages = iter(["版本一的内容", "版本一的内容", "版本二新增了价格信息"])

    async def fake_fetch_page(url, max_chars=None):
        text = next(pages)
        return FetchResult(url=url, final_url=url, status=200, content_type="text/html",
                           title="价格页", text=text, truncated=False)

    monkeypatch.setattr(fetch, "fetch_page", fake_fetch_page)
    patch_model("价格信息有更新：新增了价格表")
    service = get_scheduler_service()
    pushed = []

    async def fake_push(url, title, content):
        pushed.append(content)

    monkeypatch.setattr(service, "_push_webhook", fake_push)
    created = await service.create(
        ADMIN, name="监控", cron="0 * * * *", kind="web_watch",
        payload={"url": "https://example.com/price"}, webhook_url="https://example.com/hook",
    )
    first = await service.execute(created["id"])
    assert "初始版本" in first["output"]
    second = await service.execute(created["id"])
    assert second["output"].startswith("[无变化]")
    third = await service.execute(created["id"])
    assert "价格信息有更新" in third["output"]
    # 初始版本与有变化时推送，无变化时不推送
    assert len(pushed) == 2


async def test_schedule_tools_and_api(db, fake_redis):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.schedules import router
    from app.capabilities.scheduler import create_schedule_tool, list_schedules_tool
    from app.core.context import RequestContext, request_context

    with request_context(RequestContext()):
        out = await create_schedule_tool.ainvoke({"name": "周报", "when": "每周五下午5点", "task": "写周报"})
        assert "0 17 * * 5" in out
        assert "周报" in await list_schedules_tool.ainvoke({})

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.post("/schedules/parse", json={"text": "每天 9 点"}).json()["cron"] == "0 9 * * *"
    created = client.post("/schedules", json={"name": "api 任务", "when": "0 8 * * *", "payload": {"prompt": "hi"}})
    assert created.status_code == 200, created.text
    assert len(client.get("/schedules").json()) == 2
    assert client.delete(f"/schedules/{created.json()['id']}").status_code == 200
    assert client.get("/schedules/not-exist").status_code == 404
