"""后台任务托管。

asyncio 的事件循环对 task 只持**弱引用**：asyncio.create_task 的返回值若无人引用，
任务可能在执行途中被垃圾回收，异常也永远不会被取出（只会打印
"Task exception was never retrieved"）。

所有"发射后不管"的后台任务都应通过 spawn() 创建，由本模块持有强引用，
并在任务失败时统一记录日志。应用关闭时调用 drain() 等待其结束，
避免丢失正在写入的数据（如记忆压缩摘要）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Optional, Set

logger = logging.getLogger(__name__)

#: 当前在跑的后台任务（强引用，防止被 GC 回收）
_tasks: Set[asyncio.Task] = set()


def spawn(coro: Awaitable, *, name: Optional[str] = None) -> asyncio.Task:
    """创建一个受托管的后台任务。

    Args:
        coro: 待执行的协程
        name: 任务名（出现在日志与 asyncio.all_tasks 中）

    Returns:
        已创建并登记强引用的 Task
    """
    task = asyncio.create_task(coro, name=name)  # type: ignore[arg-type]
    _tasks.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error("后台任务 %s 失败: %s", t.get_name(), exc, exc_info=exc)

    task.add_done_callback(_on_done)
    return task


def pending() -> int:
    """当前在跑的后台任务数。"""
    return len(_tasks)


async def drain(timeout: float = 10.0) -> None:
    """等待所有后台任务结束，超时则取消（应用关闭时调用）。"""
    if not _tasks:
        return
    snapshot = list(_tasks)
    logger.info("关闭: 等待 %d 个后台任务结束...", len(snapshot))
    _, still_running = await asyncio.wait(snapshot, timeout=timeout)
    if still_running:
        logger.warning("关闭: %d 个后台任务超时未结束，取消", len(still_running))
        for task in still_running:
            task.cancel()
