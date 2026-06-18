"""意图识别模块 - 三路融合。

三路意图识别：
1. LLM 语义理解：调用 LLM 进行 few-shot 意图分类
2. 向量相似度：用户消息 embedding 与意图示例 embedding 的余弦相似度
3. 关键词加权投票：基于预定义关键词字典的匹配加权

融合策略：三路输出的置信度/得分归一化后加权求和，得分最高者为最终意图。
"""

from app.intent.llm_intent import LLMIntentRecognizer
from app.intent.vector_intent import VectorIntentRecognizer
from app.intent.keyword_intent import KeywordIntentRecognizer
from app.intent.fusion import IntentFusion, get_intent_fusion

__all__ = [
    "LLMIntentRecognizer",
    "VectorIntentRecognizer",
    "KeywordIntentRecognizer",
    "IntentFusion",
    "get_intent_fusion",
]
