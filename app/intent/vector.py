"""向量相似度意图识别。

将用户消息 embedding 与预定义的意图示例 embedding 做余弦相似度匹配，
返回 Top-K 意图及其相似度分数。

预置的意图示例 embedding 在系统启动时通过 ChromaDB 构建索引。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings, get_settings
from app.intent.catalog import IntentCatalog, get_intent_catalog
from app.store import get_chroma

logger = logging.getLogger(__name__)


class VectorIntentRecognizer:
    """基于向量相似度的意图识别器。

    使用 ChromaDB 存储意图示例的 embedding。
    初始化时将配置中的 intent_examples 写入 ChromaDB（幂等），
    之后的识别只需查询相似度即可。
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        catalog: Optional[IntentCatalog] = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        # 意图示例从动态意图目录读取（默认值即 config.intent_examples）
        self._catalog: IntentCatalog = catalog or get_intent_catalog()
        self._initialized: bool = False
        #: 意图示例集合句柄（懒加载 + 缓存，避免每次识别都发起 HTTP 往返）
        self._collection: Optional[Any] = None

    async def refresh(self) -> None:
        """意图目录变化后重建索引（插件启停、热重载时调用）。"""
        self._initialized = False
        # 集合会被删除重建，旧句柄作废
        self._collection = None
        await self.initialize()

    async def initialize(self) -> None:
        """初始化意图向量索引。

        将配置中的意图示例文本 embedding 后存入 ChromaDB。
        若集合已存在则跳过（幂等）。
        """
        if self._initialized:
            return

        try:
            await asyncio.to_thread(self._init_sync)
            self._initialized = True
            logger.info("向量意图识别器: 意图示例已索引")
        except Exception as exc:  # noqa: BLE001
            logger.error("向量意图识别器: 初始化失败: %s", exc)

    def _init_sync(self) -> None:
        """同步执行 ChromaDB 初始化（在 asyncio.to_thread 中调用）。"""
        # 本方法可能删除并重建集合，缓存的旧句柄必须作废
        self._collection = None
        client = get_chroma()
        version = self._catalog.version()
        try:
            collection = client.get_collection(
                name=self._settings.chroma_collection_intents
            )
            # 集合已存在、非空且示例版本一致，跳过
            count = collection.count()
            stored_version = (collection.metadata or {}).get("catalog_version")
            if count > 0 and stored_version == version:
                logger.info(
                    "向量意图识别器: 集合 '%s' 已有 %d 条记录，跳过初始化",
                    self._settings.chroma_collection_intents, count,
                )
                return
            # 意图示例有变化（或旧版本集合没有版本号），删除后重建
            logger.info(
                "向量意图识别器: 意图示例已变化 (%s -> %s)，重建索引",
                stored_version, version,
            )
            client.delete_collection(name=self._settings.chroma_collection_intents)
        except Exception:
            # 集合不存在，创建
            pass

        collection = client.get_or_create_collection(
            name=self._settings.chroma_collection_intents,
            metadata={"hnsw:space": "cosine", "catalog_version": version},
        )

        # 收集所有意图示例
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        for intent, examples in self._catalog.examples().items():
            for i, example in enumerate(examples):
                ids.append(f"{intent}_{i}")
                documents.append(example)
                metadatas.append({"intent": intent})

        if not documents:
            logger.info("向量意图识别器: 无意图示例可索引")
            return

        # 需要 embedding，但 ChromaDB 的 add 在没有 embedding 时会用默认函数
        # 这里我们先添加文档，让 ChromaDB 用默认 embedding 函数
        # 注意：生产环境应使用自定义 embedding_function
        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                "向量意图识别器: 已索引 %d 条意图示例", len(documents)
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("向量意图识别器: 索引意图示例失败: %s", exc)

    def _collection_sync(self):
        """获取意图示例集合句柄（同步，必须放在 asyncio.to_thread 中调用）。

        get_collection 是一次真实的 HTTP 往返（Chroma 通常是独立服务），
        直接写在协程里会阻塞整个事件循环；句柄本身缓存复用。
        """
        if self._collection is None:
            self._collection = get_chroma().get_collection(
                name=self._settings.chroma_collection_intents
            )
        return self._collection

    async def recognize(self, message: str) -> Optional[Dict[str, float]]:
        """识别用户消息的意图（向量相似度）。

        Args:
            message: 用户输入消息

        Returns:
            {意图标签: 相似度分数} 字典，识别失败返回 None
        """
        scores, _ = await self.recognize_detailed(message)
        return scores

    async def recognize_detailed(
        self, message: str
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """返回 (归一化分数分布, 最近一条示例的原始余弦相似度)。

        为什么要同时返回原始相似度：归一化分布是"Top-K 里各意图各占多少"，
        Top-K 横跨多个意图时会被摊薄——即使语义上极其接近，也拿不到高分，
        因此它不适合当"高置信"的判据。原始相似度才反映"到底有多接近"，
        供融合器的短路判断使用。
        """
        if not message.strip():
            return None, 0.0

        if not self._initialized:
            await self.initialize()

        # 在 ChromaDB 中查询相似意图：用 collection 默认 embedding 函数 embed 查询文本，
        # 与初始化写入（collection.add(documents=)）使用同一 embedding 函数，保证维度一致。
        # 不依赖 LLM embedding API，DeepSeek 等不支持 embedding 的 provider 也能工作。
        try:
            collection = await asyncio.to_thread(self._collection_sync)
            results = await asyncio.to_thread(
                collection.query,
                query_texts=[message],
                n_results=3,
                include=["metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("向量意图识别: ChromaDB 查询失败: %s", exc)
            # 句柄可能已失效（集合被重建 / 服务重启），下次调用重新获取
            self._collection = None
            return None, 0.0

        distances = (results.get("distances") or [[]])[0]
        top_similarity = max(0.0, 1.0 - float(distances[0])) if distances else 0.0
        return self._parse_results(results), top_similarity

    def _parse_results(self, results: Dict) -> Optional[Dict[str, float]]:
        """解析 ChromaDB 查询结果，汇总各意图的相似度分数。

        对每个匹配到的文档，将其相似度累加到对应意图的分数上。
        使用倒数距离作为相似度指标：相似度 = 1.0 - distance（cosine distance）。
        """
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not metadatas:
            return None

        scores: Dict[str, float] = {}
        for meta, dist in zip(metadatas, distances):
            intent = meta.get("intent", "unknown")
            # cosine distance ∈ [0, 2]，越小越相似
            similarity = max(0.0, 1.0 - dist)
            scores[intent] = scores.get(intent, 0.0) + similarity

        # 归一化
        total = sum(scores.values())
        if total > 0:
            normalized = {k: v / total for k, v in scores.items()}
            logger.debug(
                "向量意图识别: -> %s", normalized
            )
            return normalized

        return None
