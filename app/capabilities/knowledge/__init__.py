"""知识库（RAG）能力。

知识问答沿用 knowledge_retrieval 意图与 RetrievalAgent：RetrievalAgent 在回答前会检索
当前用户可访问的知识库，并可通过 knowledge_search 工具指定知识库检索。
"""

from __future__ import annotations

from typing import List

from app.capabilities.base import Capability, CapabilityTool, ToolDecl
from app.capabilities.knowledge.tools import knowledge_search_tool, list_knowledge_bases_tool
from app.intent.catalog import IntentSpec


class KnowledgeCapability(Capability):
    name = "knowledge"
    description = "RAG 知识库问答"

    def tools(self) -> List[ToolDecl]:
        return [
            CapabilityTool(knowledge_search_tool, tags=("knowledge",)),
            CapabilityTool(list_knowledge_bases_tool, tags=("knowledge",)),
        ]

    def intents(self) -> List[IntentSpec]:
        # 合并到默认意图 knowledge_retrieval（只追加关键词与示例，不改描述与路由）
        return [
            IntentSpec(
                name="knowledge_retrieval",
                description="",
                keywords=["知识库", "文档里", "资料库", "手册"],
                examples=["在知识库里查一下报销流程是怎样的", "根据产品手册回答这个问题"],
            )
        ]
