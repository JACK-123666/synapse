"""任务分发与降级调度。

根据意图找到可用 Agent 列表，按权重概率选择。
调用失败时自动降级到下一个候选 Agent，直至 FallbackAgent。
确保最终必须返回一个合法回复。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import List, Optional

from app.agents.base import AgentContext, AgentResponse
from app.config import Settings, get_settings
from app.observability import metrics
from app.observability.anomaly_detector import get_anomaly_detector
from app.router.registry import AgentRegistry, get_agent_registry

logger = logging.getLogger(__name__)


class TaskDispatcher:
    """任务分发器。

    工作流程：
    1. 根据意图从注册表中获取候选 Agent 列表。
    2. 按权重概率选择 Agent 执行。
    3. 若执行失败（异常或超时），降级到下一个候选。
    4. 所有候选失败后，降级到兜底 Agent（FallbackAgent）。
    5. 每次执行后记录延迟到异常检测器，用于 Z-score 监控。
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._registry: AgentRegistry = get_agent_registry()
        self._detector = get_anomaly_detector()

    async def dispatch(self, context: AgentContext) -> AgentResponse:
        """分发任务并返回结果。

        核心方法：从意图映射到 Agent，按权重选择，失败降级。

        Args:
            context: Agent 执行上下文

        Returns:
            Agent 执行响应（保证非空）
        """
        intent = context.intent

        # 1. 获取候选 Agent ID 列表（只含健康的）
        candidates = self._registry.get_healthy_agents_for_intent(intent)

        # 2. 如果该意图无可用 Agent，尝试使用 small_talk 路由
        if not candidates and intent != "small_talk":
            logger.warning(
                "分发: 意图 '%s' 无可用 Agent，回退到 small_talk", intent
            )
            candidates = self._registry.get_healthy_agents_for_intent("small_talk")

        # 3. 确保兜底 Agent 在最后（绝对可用）
        if not candidates:
            candidates = ["fallback_agent"]

        # 确保 fallback_agent 在候选列表末尾（如果不在的话）
        fallback_id = "fallback_agent"
        candidates = [c for c in candidates if c != fallback_id]
        candidates.append(fallback_id)

        logger.info(
            "分发: session=%s 意图=%s 候选=%s",
            context.session_id, intent, candidates,
        )

        # 4. 按权重排序后尝试
        sorted_candidates = self._sort_by_weight(candidates)

        last_error: Optional[Exception] = None
        used_agent_id: Optional[str] = None

        for agent_id in sorted_candidates:
            used_agent_id = agent_id
            try:
                result = await self._execute_agent(
                    agent_id=agent_id,
                    context=context,
                )
                # 注入 agent_id 到元数据，方便 API 层识别
                result.metadata["agent_id"] = agent_id
                # 执行成功
                logger.info(
                    "分发: session=%s 由 Agent '%s' 成功处理",
                    context.session_id, agent_id,
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "分发: Agent '%s' 执行失败 (session=%s)，降级: %s",
                    agent_id, context.session_id, exc,
                )
                # 记录失败指标
                metrics.record_request(agent_id, "error", 0.0)
                continue

        # 5. 绝对不应该走到这里（fallback_agent 绝不抛异常）
        #    但为极端安全，返回硬编码兜底回复
        logger.critical("分发: 所有 Agent 均失败，返回硬编码兜底 (session=%s)",
                        context.session_id)
        return AgentResponse(
            reply="系统暂时无法处理您的请求，请稍后重试。",
            metadata={"mode": "hard_fallback"},
        )

    async def _execute_agent(
        self,
        agent_id: str,
        context: AgentContext,
    ) -> AgentResponse:
        """执行单个 Agent，带超时控制。

        Args:
            agent_id: Agent ID
            context: 执行上下文

        Returns:
            Agent 响应

        Raises:
            TimeoutError: 执行超时
            Exception: Agent 执行异常
        """
        agent = self._registry.get_agent(agent_id)
        if agent is None:
            raise RuntimeError(f"Agent '{agent_id}' 未注册")

        timeout = self._settings.agent_timeout

        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                agent.execute(context),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            elapsed = time.monotonic() - t0
            self._detector.record_latency(agent_id, elapsed)
            metrics.record_request(agent_id, "error", elapsed)
            raise TimeoutError(
                f"Agent '{agent_id}' 超时 ({timeout}s)"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            self._detector.record_latency(agent_id, elapsed)
            metrics.record_request(agent_id, "error", elapsed)
            raise

        elapsed = time.monotonic() - t0
        # 记录正常请求耗时到异常检测器（用于 Z-score 计算）
        self._detector.record_latency(agent_id, elapsed)
        metrics.record_request(agent_id, "success", elapsed)

        return result

    def _sort_by_weight(self, candidates: List[str]) -> List[str]:
        """按权重对候选 Agent 排序（权重高的优先）。

        引入随机扰动以避免高权重 Agent 被过度使用：
        排序键 = weight + random(-0.1, 0.1)

        Args:
            candidates: Agent ID 列表

        Returns:
            按加权随机排序后的列表
        """
        weighted: List[tuple] = []
        for aid in candidates:
            weight = self._registry.get_agent_weight(aid)
            # 随机扰动 ±0.1，避免严格排序导致低权重 Agent 永远不被使用
            jittered = weight + random.uniform(-0.1, 0.1)
            weighted.append((jittered, aid))

        # 降序排列（权重高的在前）
        weighted.sort(key=lambda x: x[0], reverse=True)
        return [aid for _, aid in weighted]

    def record_fallback(self, agent_id: str) -> None:
        """记录一次降级兜底事件。"""
        metrics.record_request(agent_id, "fallback", 0.0)


# ============================================================
# 全局单例
# ============================================================

_instance: Optional[TaskDispatcher] = None


def get_task_dispatcher() -> TaskDispatcher:
    """获取任务分发器单例。"""
    global _instance
    if _instance is None:
        _instance = TaskDispatcher()
    return _instance
