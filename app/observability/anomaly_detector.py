"""Z-score 异常检测与权重自动调节。

核心机制：
1. 对每个 Agent 维护一个固定大小的滑动窗口（最近 N 次请求耗时）。
2. 每次请求完成后，计算窗口内耗时的均值 μ 和标准差 σ。
3. 若当前耗时 > μ + Z*σ（可配置阈值），判定为异常。
4. 异常时：降低 agent_health_score 和路由权重（衰减因子），连续异常持续衰减。
5. 正常时：逐步回调健康分和权重至 1.0（恢复因子）。
6. 当 health_score < 阈值时，将该 Agent 标记为不健康，从路由池摘除。

这套机制让系统具备自愈能力：故障 Agent 自动降权/摘除，
恢复后自动回切，无需人工干预。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Deque, Dict, Optional

from app.config import Settings, get_settings
from app.observability import metrics

logger = logging.getLogger(__name__)


@dataclass
class AgentHealthState:
    """单个 Agent 的健康状态记录。

    Attributes:
        agent_id: Agent 唯一标识
        latency_window: 最近 N 次请求耗时的滑动窗口
        health_score: 当前健康分（0.0 ~ 1.0），1.0 表示完全健康
        weight: 当前路由权重，影响 task_router 的选择概率
        consecutive_anomalies: 连续异常次数，用于持续衰减
        is_healthy: 是否健康（health_score >= 阈值），不健康则从路由池摘除
    """

    agent_id: str
    latency_window: Deque[float] = field(default_factory=deque)
    health_score: float = 1.0
    weight: float = 1.0
    consecutive_anomalies: int = 0
    is_healthy: bool = True


class AnomalyDetector:
    """Z-score 异常检测器。

    线程安全设计，使用 threading.Lock 保护内部状态。
    所有 Agent 的健康状态统一管理，供 task_router 查询。
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._states: Dict[str, AgentHealthState] = {}
        self._lock = threading.Lock()
        # 后台恢复任务句柄
        self._recovery_task: Optional[asyncio.Task] = None

    # ----------------------------------------------------------
    # 注册 / 查询
    # ----------------------------------------------------------

    def register_agent(self, agent_id: str) -> None:
        """注册一个新 Agent 到异常检测器。"""
        with self._lock:
            if agent_id not in self._states:
                self._states[agent_id] = AgentHealthState(agent_id=agent_id)
                metrics.update_health_score(agent_id, 1.0)
                logger.info("异常检测器: 已注册 Agent '%s'", agent_id)

    def get_state(self, agent_id: str) -> Optional[AgentHealthState]:
        """获取某个 Agent 的健康状态。"""
        with self._lock:
            return self._states.get(agent_id)

    def get_health_score(self, agent_id: str) -> float:
        """获取健康分，未注册则返回 1.0。"""
        with self._lock:
            state = self._states.get(agent_id)
            return state.health_score if state else 1.0

    def get_weight(self, agent_id: str) -> float:
        """获取路由权重，未注册则返回 1.0。"""
        with self._lock:
            state = self._states.get(agent_id)
            return state.weight if state else 1.0

    def is_agent_healthy(self, agent_id: str) -> bool:
        """判断 Agent 是否健康（可参与路由）。"""
        with self._lock:
            state = self._states.get(agent_id)
            return state.is_healthy if state else True

    def get_all_states(self) -> Dict[str, AgentHealthState]:
        """获取所有 Agent 状态快照（浅拷贝）。"""
        with self._lock:
            return dict(self._states)

    # ----------------------------------------------------------
    # 核心检测逻辑
    # ----------------------------------------------------------

    def record_latency(self, agent_id: str, latency: float) -> bool:
        """记录一次请求耗时，执行 Z-score 检测并更新健康状态。

        Args:
            agent_id: Agent ID
            latency: 本次请求耗时（秒）

        Returns:
            True 表示检测到异常，False 表示正常
        """
        with self._lock:
            state = self._states.get(agent_id)
            if state is None:
                # 未注册则自动注册
                state = AgentHealthState(agent_id=agent_id)
                self._states[agent_id] = state

            window_size = self._settings.zscore_window_size
            state.latency_window.append(latency)
            # 保持滑动窗口固定大小
            while len(state.latency_window) > window_size:
                state.latency_window.popleft()

            is_anomaly = self._detect_anomaly(state, latency)
            if is_anomaly:
                self._handle_anomaly(state)
            else:
                self._handle_recovery(state)

            return is_anomaly

    def _detect_anomaly(self, state: AgentHealthState, latency: float) -> bool:
        """Z-score 异常检测。

        计算滑动窗口的均值 μ 和标准差 σ，
        若 latency > μ + Z*σ 则判定为异常。

        当窗口样本数不足（<3）时跳过检测，避免统计意义不足。

        Args:
            state: Agent 健康状态
            latency: 当前请求耗时

        Returns:
            True 表示异常
        """
        window = list(state.latency_window)
        # 样本数不足，无法做有意义的统计推断
        if len(window) < 3:
            return False

        mean_val = mean(window)
        try:
            std_val = stdev(window)
        except Exception:
            std_val = 0.0

        # 标准差为 0（所有耗时完全一致）时，仅当当前耗时明显偏离均值才判异常
        if std_val == 0.0:
            return latency > mean_val * 1.5

        z_score = (latency - mean_val) / std_val
        threshold = self._settings.zscore_threshold

        # 当前耗时超过 μ + Z*σ 即异常
        is_anomaly = latency > mean_val + threshold * std_val

        if is_anomaly:
            logger.warning(
                "异常检测: Agent '%s' 延迟 %.3fs 超过阈值 "
                "(μ=%.3f, σ=%.3f, z=%.2f, 阈值=%.1f)",
                state.agent_id, latency, mean_val, std_val, z_score, threshold,
            )

        return is_anomaly

    def _handle_anomaly(self, state: AgentHealthState) -> None:
        """处理异常：衰减健康分和权重。

        - 健康分：乘以衰减因子，连续异常持续衰减
        - 权重：同步衰减，影响路由选择概率
        - 连续异常计数 +1
        - 健康分低于阈值时标记为不健康，触发摘除
        """
        decay = self._settings.weight_decay_factor
        state.consecutive_anomalies += 1
        # 健康分衰减
        state.health_score *= decay
        # 权重同步衰减
        state.weight *= decay
        # 下限保护
        state.weight = max(0.01, state.weight)

        threshold = self._settings.agent_health_threshold
        if state.health_score < threshold and state.is_healthy:
            state.is_healthy = False
            logger.error(
                "故障降级: Agent '%s' 健康分 %.3f 低于阈值 %.2f，"
                "已从路由池摘除（连续异常 %d 次）",
                state.agent_id, state.health_score, threshold,
                state.consecutive_anomalies,
            )

        metrics.update_health_score(state.agent_id, state.health_score)

    def _handle_recovery(self, state: AgentHealthState) -> None:
        """处理正常：逐步恢复健康分和权重。

        - 重置连续异常计数
        - 健康分和权重按恢复因子逐步回调，上限 1.0
        - 若此前被摘除且健康分恢复到阈值以上，重新加入路由池
        """
        state.consecutive_anomalies = 0
        recovery = self._settings.weight_recovery_factor

        # 逐步回调
        state.health_score = min(1.0, state.health_score * recovery)
        state.weight = min(1.0, state.weight * recovery)

        threshold = self._settings.agent_health_threshold
        if not state.is_healthy and state.health_score >= threshold:
            state.is_healthy = True
            logger.info(
                "故障恢复: Agent '%s' 健康分回升至 %.3f，重新加入路由池",
                state.agent_id, state.health_score,
            )

        metrics.update_health_score(state.agent_id, state.health_score)

    # ----------------------------------------------------------
    # 后台恢复任务
    # ----------------------------------------------------------

    async def start_recovery_loop(self) -> None:
        """启动后台恢复检测任务。

        定期检查所有 Agent 状态：即使没有新请求进来，
        也会让异常 Agent 的健康分随时间逐步恢复，
        避免因短暂的延迟尖峰导致永久摘除。
        """
        self._recovery_task = asyncio.create_task(self._recovery_loop())
        logger.info("异常检测器: 后台恢复任务已启动")

    async def stop_recovery_loop(self) -> None:
        """停止后台恢复任务。"""
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass
            self._recovery_task = None
            logger.info("异常检测器: 后台恢复任务已停止")

    async def _recovery_loop(self) -> None:
        """后台恢复循环：定期对不健康 Agent 做无新请求的恢复。"""
        interval = self._settings.anomaly_check_interval
        while True:
            try:
                await asyncio.sleep(interval)
                self._passive_recovery()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.exception("后台恢复任务异常: %s", exc)

    def _passive_recovery(self) -> None:
        """被动恢复：无新请求时让异常 Agent 健康分缓慢回升。"""
        recovery = self._settings.weight_recovery_factor
        threshold = self._settings.agent_health_threshold
        with self._lock:
            for state in self._states.values():
                # 仅对未满分的 Agent 做被动恢复
                if state.health_score < 1.0:
                    # 被动恢复速度减半，避免过快回切
                    state.health_score = min(1.0, state.health_score * recovery)
                    state.weight = min(1.0, state.weight * recovery)
                    if not state.is_healthy and state.health_score >= threshold:
                        state.is_healthy = True
                        logger.info(
                            "被动恢复: Agent '%s' 健康分回升至 %.3f，重新加入路由池",
                            state.agent_id, state.health_score,
                        )
                    metrics.update_health_score(state.agent_id, state.health_score)


# ============================================================
# 全局单例
# ============================================================

_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """获取全局异常检测器单例。"""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector
