"""记忆管理模块。

三级分层记忆架构：
- 短期记忆（Redis）：当前会话最近 N 轮对话
- 长期情景记忆（ChromaDB）：历史对话摘要向量，支持相似情景召回
- 用户画像：静态/动态用户特征

配合摘要压缩器，当短期记忆达到阈值时自动触发 LLM 摘要，
压缩后存入长期记忆并清空短期记忆，降低 Token 消耗。
"""

from app.memory.short_term import ShortTermMemory, get_short_term_memory
from app.memory.long_term import LongTermMemory, get_long_term_memory
from app.memory.user_profile import UserProfileManager, get_user_profile_manager
from app.memory.compressor import MemoryCompressor, get_memory_compressor

__all__ = [
    "ShortTermMemory",
    "get_short_term_memory",
    "LongTermMemory",
    "get_long_term_memory",
    "UserProfileManager",
    "get_user_profile_manager",
    "MemoryCompressor",
    "get_memory_compressor",
]
