"""全局配置模块。

所有阈值、权重、LLM 参数、ChromaDB 集合名、Redis 连接信息等统一管理，
均可通过环境变量覆盖。使用 pydantic-settings 实现类型安全的配置加载。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 在 Settings 类定义之前，手动把 .env 注入 os.environ
# 优先读 Docker 容器内的绝对路径，其次读开发环境的相对路径
# override=False：不覆盖 OS 环境变量（docker-compose environment 优先级更高）
_env_file = Path("/app/.env")
if not _env_file.exists():
    _env_file = Path(".env")
if _env_file.exists():
    load_dotenv(_env_file, override=False)


class Settings(BaseSettings):
    """Synapse 全局配置。

    通过环境变量或 .env 文件加载，所有字段均带有默认值，
    确保 Docker Compose 一键启动时无需额外配置即可运行。
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    # Redis 配置
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis 连接 URL，用于短期记忆与用户画像缓存",
    )

    # ChromaDB 配置
    chroma_host: str = Field(default="chromadb", description="ChromaDB 服务主机")
    chroma_port: int = Field(default=8000, description="ChromaDB 服务端口")

    # ChromaDB 集合名
    chroma_collection_intents: str = Field(
        default="intent_examples", description="意图示例向量集合"
    )
    chroma_collection_summary: str = Field(
        default="session_summaries", description="会话摘要长期记忆集合"
    )
    chroma_collection_knowledge: str = Field(
        default="knowledge_base", description="知识库向量集合"
    )
    chroma_collection_profile: str = Field(
        default="user_profiles", description="用户画像向量集合"
    )

    # LLM 配置
    llm_provider: str = Field(
        default="openai", description="LLM 提供商: openai / claude"
    )
    llm_api_key: str = Field(default="", description="LLM API 密钥")
    llm_model: str = Field(default="gpt-4o-mini", description="OpenAI 模型名")
    llm_base_url: str = Field(
        default="https://api.openai.com/v1", description="OpenAI 兼容 base_url"
    )
    claude_model: str = Field(
        default="claude-3-5-sonnet-20241022", description="Claude 模型名"
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com/v1", description="Anthropic API base_url"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding 模型名"
    )
    embedding_api_key: str = Field(
        default="",
        description="Embedding 专用 API 密钥；留空回退 llm_api_key。"
        "DeepSeek 不支持 embedding，需指向支持 embedding 的服务（如 OpenAI）",
    )
    embedding_base_url: str = Field(
        default="",
        description="Embedding 专用 base_url；留空回退 llm_base_url",
    )
    llm_timeout: int = Field(default=60, description="LLM 请求超时（秒）")

    # DeepSeek 专用配置（兼容 OpenAI 协议，base_url 指向 DeepSeek）
    deepseek_model: str = Field(
        default="deepseek-chat", description="DeepSeek 模型名"
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API base_url（兼容 OpenAI 协议）",
    )

    # 应用层
    cors_origins: str = Field(
        default="*",
        description="CORS 允许来源，逗号分隔；* 表示全部（此时自动禁用凭证）",
    )
    docs_enabled: bool = Field(
        default=True,
        description="是否开启 /docs /redoc API 文档；生产环境建议设为 false",
    )

    # 意图识别三路融合权重
    intent_llm_weight: float = Field(default=0.5, description="LLM 语义理解权重")
    intent_vector_weight: float = Field(default=0.3, description="向量相似度权重")
    intent_keyword_weight: float = Field(default=0.2, description="关键词投票权重")

    # 记忆管理
    short_term_max_rounds: int = Field(
        default=10, description="短期记忆最大保留轮数"
    )
    short_term_ttl: int = Field(
        default=86400, description="短期记忆 TTL（秒），默认 24 小时"
    )
    summary_trigger_rounds: int = Field(
        default=8, description="触发摘要压缩的轮数阈值"
    )
    long_term_recall_k: int = Field(
        default=3, description="长期记忆召回 Top-K"
    )
    token_budget: int = Field(
        default=4000, description="单次 Token 估算上限，超出触发压缩"
    )

    # 路由与故障恢复
    agent_health_threshold: float = Field(
        default=0.3, description="健康分低于此阈值时从路由池摘除"
    )
    weight_decay_factor: float = Field(
        default=0.5, description="异常时权重衰减因子"
    )
    weight_recovery_factor: float = Field(
        default=1.1, description="恢复时权重回调因子"
    )
    agent_timeout: int = Field(
        default=30, description="Agent 请求超时（秒），超时触发降级"
    )

    # Z-score 异常检测
    zscore_window_size: int = Field(
        default=20, description="滑动窗口大小（最近 N 次请求）"
    )
    zscore_threshold: float = Field(
        default=3.0, description="Z-score 异常阈值，耗时 > μ + Z*σ 即异常"
    )
    anomaly_check_interval: int = Field(
        default=10, description="异常检测后台任务间隔（秒）"
    )

    # 预定义意图列表
    # 这些意图标签用于意图识别、路由分发和Agent选择
    known_intents: List[str] = Field(
        default_factory=lambda: [
            "knowledge_retrieval",
            "summarize",
            "small_talk",
        ]
    )

    # 意图描述
    # 新增意图时同步在此添加描述，LLM 意图识别器自动生效
    intent_descriptions: Dict[str, str] = Field(
        default_factory=lambda: {
            "knowledge_retrieval": "查询知识、检索文档、了解概念、求知探索",
            "summarize": "摘要、总结、压缩、概括文本或对话",
            "small_talk": "问候、闲聊、感谢、道别等非任务型对话",
        }
    )

    # 意图关键词字典
    # 每个意图对应一组关键词，命中后按匹配数加权
    intent_keywords: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "knowledge_retrieval": [
                "查询", "搜索", "找", "是什么", "什么是", "解释",
                "文档", "资料", "知识", "检索", "原理", "区别",
                "how", "what", "why", "explain", "search", "find",
            ],
            "summarize": [
                "总结", "摘要", "概括", "归纳", "压缩", "简述",
                "提炼", "重点", "summarize", "summary", "brief",
            ],
            "small_talk": [
                "你好", "嗨", "谢谢", "再见", "早上好", "晚上好",
                "hello", "hi", "thanks", "bye", "how are you",
            ],
        }
    )

    # 意图示例文本
    # 用于在 ChromaDB 中构建意图 embedding 索引
    intent_examples: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "knowledge_retrieval": [
                "帮我查一下这个技术方案的原理",
                "什么是向量数据库？",
                "请检索相关文档并解释这个概念",
                "How does authentication work in this system?",
            ],
            "summarize": [
                "请帮我总结一下刚才的对话",
                "把这段内容压缩成摘要",
                "归纳一下会议要点",
                "Summarize the key points of this article.",
            ],
            "small_talk": [
                "你好呀",
                "今天天气怎么样",
                "谢谢你",
                "Hi, how are you doing today?",
            ],
        }
    )


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例。

    使用 lru_cache 确保整个应用生命周期内只加载一次配置。
    """
    return Settings()
