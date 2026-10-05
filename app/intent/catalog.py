"""动态意图目录。

意图目录 = config.py 中的默认意图（known_intents / intent_descriptions /
intent_keywords / intent_examples）+ 各能力 / 插件在运行时注册的意图。

三路识别器（LLM 语义、向量、关键词）都从这里读取意图元数据，
因此新增能力或插件时只需注册 IntentSpec，无需修改识别器代码。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class IntentSpec:
    """一个意图的元数据。

    Attributes:
        name: 意图标签（路由键）
        description: 意图描述（注入 LLM 分类 prompt）
        keywords: 关键词（关键词投票路）
        examples: 示例语句（向量相似度路 + few-shot）
        source: 来源，config / builtin:<能力名> / plugin:<插件名>
    """

    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    source: str = "config"


class IntentCatalog:
    """意图目录（线程安全）。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._specs: Dict[str, IntentSpec] = {}
        self._lock = threading.Lock()
        self._load_defaults()

    def _load_defaults(self) -> None:
        s = self._settings
        for name in s.known_intents:
            self._specs[name] = IntentSpec(
                name=name,
                description=s.intent_descriptions.get(name, "其他"),
                keywords=list(s.intent_keywords.get(name, [])),
                examples=list(s.intent_examples.get(name, [])),
                source="config",
            )

    # 注册 / 注销

    def register(self, spec: IntentSpec) -> None:
        with self._lock:
            existing = self._specs.get(spec.name)
            if existing is not None and existing.source == "config":
                # 默认意图只合并关键词和示例，不覆盖描述
                merged = IntentSpec(
                    name=existing.name,
                    description=existing.description,
                    keywords=list(dict.fromkeys(existing.keywords + spec.keywords)),
                    examples=list(dict.fromkeys(existing.examples + spec.examples)),
                    source="config",
                )
                self._specs[spec.name] = merged
            else:
                self._specs[spec.name] = spec
        logger.info("意图目录: 已注册意图 '%s' (来源=%s)", spec.name, spec.source)

    def unregister(self, name: str) -> None:
        with self._lock:
            spec = self._specs.get(name)
            if spec is not None and spec.source != "config":
                self._specs.pop(name, None)

    def unregister_source(self, source: str) -> List[str]:
        with self._lock:
            names = [n for n, s in self._specs.items() if s.source == source]
            for n in names:
                self._specs.pop(n, None)
        if names:
            logger.info("意图目录: 已注销来源 %s 的意图 %s", source, names)
        return names

    # 查询

    def names(self) -> List[str]:
        with self._lock:
            return list(self._specs.keys())

    def get(self, name: str) -> Optional[IntentSpec]:
        return self._specs.get(name)

    def specs(self) -> List[IntentSpec]:
        with self._lock:
            return list(self._specs.values())

    def descriptions(self) -> Dict[str, str]:
        with self._lock:
            return {n: s.description for n, s in self._specs.items()}

    def keywords(self) -> Dict[str, List[str]]:
        with self._lock:
            return {n: list(s.keywords) for n, s in self._specs.items()}

    def examples(self) -> Dict[str, List[str]]:
        with self._lock:
            return {n: list(s.examples) for n, s in self._specs.items()}

    def version(self) -> str:
        """意图示例内容的哈希，用于判断向量索引是否需要重建。"""
        payload = json.dumps(self.examples(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


_catalog: Optional[IntentCatalog] = None


def get_intent_catalog() -> IntentCatalog:
    """获取全局意图目录单例。"""
    global _catalog
    if _catalog is None:
        _catalog = IntentCatalog()
    return _catalog
