"""Agent 注册表与权重管理。

维护所有 Agent 的注册信息、路由表和当前权重。
路由表定义了意图到 Agent 的映射关系（主 Agent + 备用 Agent）。
权重受异常检测器动态调节，影响 dispatch 时的选择概率。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.config import Settings, get_settings
from app.agents.base import BaseAgent
from app.observability.anomaly_detector import (
    AnomalyDetector,
    get_anomaly_detector,
)

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Agent 注册表。

    职责：
    - 维护 Agent ID 到 Agent 实例的映射
    - 维护意图到候选 Agent 列表的路由表
    - 查询 Agent 的当前权重与健康状态
    - 提供给定意图的可用 Agent 列表（按权重排序）
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        #: Agent ID -> BaseAgent 实例
        self._agents: Dict[str, BaseAgent] = {}
        #: 意图 -> 候选 Agent ID 列表（主 Agent 在前，备用在后）
        self._intent_routes: Dict[str, List[str]] = {}
        #: 异常检测器引用
        self._detector: AnomalyDetector = get_anomaly_detector()

    # ----------------------------------------------------------
    # 注册
    # ----------------------------------------------------------

    def register_agent(self, agent: BaseAgent) -> None:
        """注册一个 Agent。

        注册后自动在异常检测器中登记，用于 Z-score 监控。

        Args:
            agent: Agent 实例
        """
        self._agents[agent.agent_id] = agent
        self._detector.register_agent(agent.agent_id)
        logger.info(
            "注册表: 已注册 Agent '%s' (%s)", agent.agent_id, agent.description
        )

    def register_route(
        self,
        intent: str,
        agent_ids: List[str],
    ) -> None:
        """注册意图到 Agent 的路由规则。

        agent_ids 列表的第一个元素为主 Agent，后续为降级备用 Agent。
        降级时按列表顺序依次尝试。

        Args:
            intent: 意图标签
            agent_ids: 候选 Agent ID 列表（优先级从高到低）
        """
        # 验证所有 Agent ID 已注册
        for aid in agent_ids:
            if aid not in self._agents:
                logger.warning(
                    "注册表: 路由 '%s' -> '%s' 中的 Agent 尚未注册，跳过",
                    intent, aid,
                )
                return

        self._intent_routes[intent] = agent_ids
        logger.info(
            "注册表: 已注册路由 '%s' -> %s", intent, agent_ids
        )

    # ----------------------------------------------------------
    # 查询
    # ----------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """根据 ID 获取 Agent 实例。"""
        return self._agents.get(agent_id)

    def get_agents_for_intent(self, intent: str) -> List[str]:
        """获取指定意图的候选 Agent ID 列表。

        Args:
            intent: 意图标签

        Returns:
            Agent ID 列表（主 Agent 在前，备用在后）。
            若路由表中无该意图，返回空列表。
        """
        return self._intent_routes.get(intent, [])

    def get_healthy_agents_for_intent(self, intent: str) -> List[str]:
        """获取指定意图的可用（健康）Agent ID 列表。

        从路由表中过滤掉被异常检测器标记为不健康的 Agent。

        Args:
            intent: 意图标签

        Returns:
            可用 Agent ID 列表
        """
        candidates = self.get_agents_for_intent(intent)
        healthy = [
            aid for aid in candidates
            if self._detector.is_agent_healthy(aid)
        ]
        return healthy

    def get_agent_weight(self, agent_id: str) -> float:
        """获取 Agent 的当前路由权重。

        权重由异常检测器动态维护，反映 Agent 的实时可靠性。

        Args:
            agent_id: Agent ID

        Returns:
            权重值（0.01 ~ 1.0）
        """
        return self._detector.get_weight(agent_id)

    def get_agent_health(self, agent_id: str) -> float:
        """获取 Agent 的当前健康分。"""
        return self._detector.get_health_score(agent_id)

    def is_healthy(self, agent_id: str) -> bool:
        """检查 Agent 是否健康。"""
        return self._detector.is_agent_healthy(agent_id)

    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """获取所有已注册 Agent。"""
        return dict(self._agents)

    def get_all_health_status(self) -> Dict[str, Dict[str, object]]:
        """获取所有 Agent 的健康状态摘要。

        Returns:
            {agent_id: {"healthy": bool, "health_score": float, "weight": float}}
        """
        status: Dict[str, Dict[str, object]] = {}
        for aid in self._agents:
            status[aid] = {
                "healthy": self.is_healthy(aid),
                "health_score": self.get_agent_health(aid),
                "weight": self.get_agent_weight(aid),
            }
        return status


# ============================================================
# 全局单例
# ============================================================

_instance: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    """获取 Agent 注册表单例。"""
    global _instance
    if _instance is None:
        _instance = AgentRegistry()
    return _instance
