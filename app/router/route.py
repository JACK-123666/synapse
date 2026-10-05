"""任务分发与降级调度。

根据意图找到可用 Agent 列表，按权重概率选择。
调用失败时自动降级到下一个候选 Agent，直至 FallbackAgent。
所有路径最终返回一个合法回复。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from app.agents.base import AgentContext, AgentResponse
from app.config import Settings, get_settings
from app.observability import metrics
from app.observability.health import get_anomaly_detector
from app.router.pool import AgentRegistry, get_agent_registry

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
        # 候选列表由 _candidates_for 统一计算（健康过滤 + small_talk 回退 + 兜底置尾），
        # 原先此处有一份等价的内联副本，已合并
        sorted_candidates = self._ordered_candidates(
            self._candidates_for(intent, context.session_id)
        )

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
                # 失败指标已在 _execute_agent 中记录（含真实耗时），这里不再重复计数
                continue

        # 绝对不应该走到这里（fallback_agent 绝不抛异常）        #    但为极端安全，返回硬编码兜底回复
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

    def _candidates_for(self, intent: str, session_id: str) -> List[str]:
        """获取意图的候选 Agent 列表（只含健康的，兜底 Agent 保证在末尾）。"""
        candidates = self._registry.get_healthy_agents_for_intent(intent)
        if not candidates and intent != "small_talk":
            logger.warning(
                "分发: 意图 '%s' 无可用 Agent，回退到 small_talk", intent
            )
            candidates = self._registry.get_healthy_agents_for_intent("small_talk")
        fallback_id = "fallback_agent"
        candidates = [c for c in candidates if c != fallback_id]
        candidates.append(fallback_id)
        logger.info(
            "分发: session=%s 意图=%s 候选=%s", session_id, intent, candidates,
        )
        return candidates

    def _ordered_candidates(self, candidates: List[str]) -> List[str]:
        """按权重排序（兜底 Agent 永远最后，不参与排序）。"""
        fallback_id = "fallback_agent"
        main_candidates = [c for c in candidates if c != fallback_id]
        sorted_candidates = self._sort_by_weight(main_candidates)
        sorted_candidates.append(fallback_id)
        return sorted_candidates

    async def dispatch_stream(
        self, context: AgentContext
    ) -> AsyncIterator[Dict[str, Any]]:
        """流式分发。

        与 dispatch 使用相同的候选与降级规则：在某个 Agent 输出第一个 token 之前失败，
        自动降级到下一个候选；已经开始输出后失败，则输出 error 事件并结束。

        Yields:
            Agent 的流式事件；最后一个事件为 {"type": "final", "response": ..., "agent_id": ...}
        """
        candidates = self._ordered_candidates(
            self._candidates_for(context.intent, context.session_id)
        )

        # 流式输出不套整体超时（跨 yield 的超时会误伤调用方），依赖 LLM 自身的请求超时
        for agent_id in candidates:
            agent = self._registry.get_agent(agent_id)
            if agent is None:
                continue
            started = False
            t0 = time.monotonic()
            try:
                async for event in agent.stream(context):
                    if event.get("type") == "final":
                        response: AgentResponse = event["response"]
                        response.metadata["agent_id"] = agent_id
                        elapsed = time.monotonic() - t0
                        self._detector.record_latency(agent_id, elapsed)
                        metrics.record_request(agent_id, "success", elapsed)
                        yield {**event, "agent_id": agent_id}
                        return
                    started = True
                    yield event
            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - t0
                self._detector.record_latency(agent_id, elapsed)
                metrics.record_request(agent_id, "error", elapsed)
                logger.warning(
                    "流式分发: Agent '%s' 执行失败 (session=%s, 已输出=%s): %s",
                    agent_id, context.session_id, started, exc,
                )
                if started:
                    yield {"type": "error", "message": f"生成中断: {exc}", "agent_id": agent_id}
                    return
                continue

        logger.critical("流式分发: 所有 Agent 均失败 (session=%s)", context.session_id)
        response = AgentResponse(
            reply="系统暂时无法处理您的请求，请稍后重试。",
            metadata={"mode": "hard_fallback", "agent_id": "hard_fallback"},
        )
        yield {"type": "token", "content": response.reply}
        yield {"type": "final", "response": response, "agent_id": "hard_fallback"}

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


# 全局单例

_instance: Optional[TaskDispatcher] = None


def get_task_dispatcher() -> TaskDispatcher:
    """获取任务分发器单例。"""
    global _instance
    if _instance is None:
        _instance = TaskDispatcher()
    return _instance
