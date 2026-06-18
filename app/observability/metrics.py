"""Prometheus 指标定义与暴露。

定义三类核心指标：
- synapse_request_total：请求计数（Counter），按 agent_id / status 标签维度
- synapse_request_duration_seconds：请求延迟（Histogram），按 agent_id 标签维度
- agent_health_score：Agent 实时健康分（Gauge），动态调整

这些指标通过 /metrics 端点暴露给 Prometheus 拉取。
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ============================================================
# 指标定义
# ============================================================

#: 请求总数计数器，标签: agent_id（执行 Agent）、status（success/fallback/error）
synapse_request_total = Counter(
    name="synapse_request_total",
    documentation="Synapse 平台请求总数",
    labelnames=("agent_id", "status"),
)

#: 请求延迟直方图，标签: agent_id
synapse_request_duration_seconds = Histogram(
    name="synapse_request_duration_seconds",
    documentation="Synapse 请求处理延迟（秒）",
    labelnames=("agent_id",),
    # 桶值覆盖从 50ms 到 60s 的延迟范围
    buckets=(
        0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
    ),
)

#: Agent 健康分（0.0~1.0），动态调整，低于阈值时从路由池摘除
agent_health_score = Gauge(
    name="agent_health_score",
    documentation="Agent 实时健康分（0.0-1.0），低于阈值摘除路由池",
    labelnames=("agent_id",),
)

#: 意图识别置信度直方图，用于监控意图识别质量
intent_confidence = Histogram(
    name="intent_confidence",
    documentation="意图识别融合后的置信度分布",
    labelnames=("intent",),
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

#: 摘要压缩触发次数
memory_compression_total = Counter(
    name="memory_compression_total",
    documentation="短期记忆摘要压缩触发总次数",
)


def record_request(agent_id: str, status: str, duration: float) -> None:
    """记录一次请求的指标。

    Args:
        agent_id: 执行该请求的 Agent ID
        status: 请求状态，success / fallback / error
        duration: 请求耗时（秒）
    """
    synapse_request_total.labels(agent_id=agent_id, status=status).inc()
    synapse_request_duration_seconds.labels(agent_id=agent_id).observe(duration)


def update_health_score(agent_id: str, score: float) -> None:
    """更新某个 Agent 的健康分。

    Args:
        agent_id: Agent ID
        score: 健康分（0.0 ~ 1.0）
    """
    # 确保分数在 [0, 1] 范围内
    clamped = max(0.0, min(1.0, score))
    agent_health_score.labels(agent_id=agent_id).set(clamped)


def record_intent_confidence(intent: str, confidence: float) -> None:
    """记录意图识别置信度。"""
    intent_confidence.labels(intent=intent).observe(confidence)


def record_compression() -> None:
    """记录一次摘要压缩触发。"""
    memory_compression_total.inc()
