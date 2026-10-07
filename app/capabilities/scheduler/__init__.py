"""定时任务能力。"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.tools import tool

from app.agents.base import AgentContext, BaseAgent
from app.agents.langchain_agent import LangChainAgent
from app.capabilities.base import Capability, CapabilityTool, ToolDecl
from app.capabilities.knowledge.service import current_principal
from app.capabilities.scheduler.cron import ScheduleError, next_runs, to_cron
from app.capabilities.scheduler.service import get_scheduler_service
from app.intent.catalog import IntentSpec


def _describe(s: Dict) -> str:
    nxt = "、".join(s.get("next_runs", [])[:2]) or "（已暂停）"
    return f"- {s['name']}（{s['kind']}，cron: {s['cron']}，{'启用' if s['enabled'] else '暂停'}）下次: {nxt}"


@tool("create_schedule")
async def create_schedule_tool(name: str, when: str, task: str, webhook_url: str = "") -> str:
    """创建定时任务：到点后助手会自动执行 task 描述的指令（可联网、查仓库、查知识库等），结果可推送到 Webhook。

    Args:
        name: 任务名称
        when: 执行时间，自然语言（如“每天早上 9 点”“每周一 10:30”“每隔 30 分钟”）或 cron 表达式
        task: 到点后要执行的指令，如“总结 synapse 仓库昨天的提交”
        webhook_url: 可选，结果推送地址（飞书 / 企业微信 / 钉钉机器人或通用 HTTP）
    """
    try:
        cron = await to_cron(when)
        s = await get_scheduler_service().create(
            current_principal(), name=name, cron=cron, kind="prompt",
            payload={"prompt": task}, webhook_url=webhook_url,
        )
    except ScheduleError as exc:
        return f"创建失败: {exc}"
    return f"已创建定时任务「{s['name']}」，cron: {s['cron']}，接下来执行时间: {'、'.join(next_runs(s['cron'], 3))}"


@tool("create_web_watch")
async def create_web_watch_tool(name: str, url: str, when: str = "每小时", webhook_url: str = "") -> str:
    """创建网页监控任务：定期抓取网页，正文有变化时生成变化摘要并推送。

    Args:
        name: 任务名称
        url: 要监控的网页地址
        when: 检查频率（自然语言或 cron），默认每小时
        webhook_url: 可选，变化通知推送地址
    """
    try:
        cron = await to_cron(when)
        s = await get_scheduler_service().create(
            current_principal(), name=name, cron=cron, kind="web_watch",
            payload={"url": url}, webhook_url=webhook_url,
        )
    except ScheduleError as exc:
        return f"创建失败: {exc}"
    return f"已创建网页监控「{s['name']}」，cron: {s['cron']}。"


@tool("list_schedules")
async def list_schedules_tool() -> str:
    """列出当前用户的定时任务及下次执行时间。"""
    items = await get_scheduler_service().list(current_principal())
    if not items:
        return "当前没有定时任务。"
    return "\n".join(_describe(s) for s in items)


@tool("delete_schedule")
async def delete_schedule_tool(name: str) -> str:
    """删除定时任务。

    Args:
        name: 任务名称
    """
    service = get_scheduler_service()
    try:
        schedule = await service.find(current_principal(), name)
        await service.delete(current_principal(), schedule.id)
    except ScheduleError as exc:
        return f"删除失败: {exc}"
    return f"已删除定时任务「{name}」。"


@tool("toggle_schedule")
async def toggle_schedule_tool(name: str, enabled: bool) -> str:
    """暂停或恢复定时任务。

    Args:
        name: 任务名称
        enabled: true 恢复 / false 暂停
    """
    service = get_scheduler_service()
    try:
        schedule = await service.find(current_principal(), name)
        await service.update(current_principal(), schedule.id, enabled=enabled)
    except ScheduleError as exc:
        return f"操作失败: {exc}"
    return f"已{'恢复' if enabled else '暂停'}定时任务「{name}」。"


@tool("run_schedule_now")
async def run_schedule_now_tool(name: str) -> str:
    """立即执行一次定时任务并返回结果。

    Args:
        name: 任务名称
    """
    service = get_scheduler_service()
    try:
        schedule = await service.find(current_principal(), name)
        run = await service.run_now(current_principal(), schedule.id)
    except ScheduleError as exc:
        return f"执行失败: {exc}"
    return f"执行状态: {run['status']}\n{run['output']}"


class SchedulerAgent(LangChainAgent):
    """定时任务助手。"""

    agent_id = "scheduler_agent"
    description = "定时任务助手"
    tool_tags = ("schedule",)

    # 静态提示词：记忆召回 / 用户画像由 build_context_block 注入消息序列
    system_prompt = (
        "你负责管理用户的定时任务：创建、查看、暂停 / 恢复、删除、立即执行。\n"
        "创建任务时把用户描述拆成：名称、执行时间（when，保留用户原话即可）、要执行的指令（task）。\n"
        "用户要求“网页有变化时通知我”时使用 create_web_watch。\n"
        "创建成功后告诉用户 cron 表达式和接下来的执行时间。"
    )


class SchedulerCapability(Capability):
    """定时任务能力：任务管理工具与 SchedulerAgent。"""
    name = "scheduler"
    description = "定时任务"

    def tools(self) -> List[ToolDecl]:
        return [
            CapabilityTool(t, tags=("schedule",))
            for t in (
                create_schedule_tool,
                create_web_watch_tool,
                list_schedules_tool,
                delete_schedule_tool,
                toggle_schedule_tool,
                run_schedule_now_tool,
            )
        ]

    def intents(self) -> List[IntentSpec]:
        return [
            IntentSpec(
                name="schedule_task",
                description="创建、查看、修改、删除定时任务或周期提醒，例如“每天早上 9 点……”“每周一提醒我……”“网页有更新时通知我”",
                keywords=["定时", "每天", "每周", "每月", "每小时", "每隔", "提醒我", "cron", "计划任务", "定期", "schedule", "remind"],
                examples=[
                    "每天早上 9 点帮我总结 synapse 仓库的新提交",
                    "每周一提醒我写周报",
                    "列出我所有的定时任务",
                    "这个网页有更新时通知我",
                    "Remind me every Friday to check the logs",
                ],
            )
        ]

    def agents(self) -> List[BaseAgent]:
        return [SchedulerAgent()]

    def routes(self) -> Dict[str, List[str]]:
        return {"schedule_task": ["scheduler_agent", "general_agent", "fallback_agent"]}

    async def startup(self) -> None:
        """能力注册完成后启动调度器，并装载数据库中已有的任务。"""
        await get_scheduler_service().start()

    async def shutdown(self) -> None:
        """停用该能力时停止调度器。"""
        await get_scheduler_service().shutdown()
