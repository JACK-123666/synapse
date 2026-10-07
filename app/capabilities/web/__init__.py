"""网页能力：联网搜索、网页抓取、抓取入库。"""

from __future__ import annotations

from typing import Dict, List

from app.agents.base import BaseAgent
from app.agents.langchain_agent import LangChainAgent
from app.capabilities.base import Capability, CapabilityTool, ToolDecl
from app.capabilities.web.quote import stock_quote_tool
from app.capabilities.web.tools import (
    fetch_url_tool,
    save_url_to_knowledge_tool,
    web_search_tool,
)
from app.intent.catalog import IntentSpec


class WebAgent(LangChainAgent):
    """网页助手：抓取 / 阅读 / 总结网页，联网搜索。"""

    agent_id = "web_agent"
    description = "网页抓取与联网搜索助手"
    tool_tags = ("web",)

    # 静态提示词：记忆召回 / 用户画像由 build_context_block 注入消息序列
    system_prompt = (
        "你是网页阅读助手。用户给出链接时，调用 fetch_url 抓取正文后再回答；"
        "没有链接但需要最新信息时，先调用 web_search 搜索，再按需抓取具体页面。\n"
        "查询股票行情、实时报价时调用 stock_quote，它比搜索更快也更准确。\n"
        "回答时注明信息来源链接；抓取失败要如实说明原因，不要编造网页内容。\n"
        "用户要求把网页保存到知识库时，调用 save_url_to_knowledge。"
    )


class WebCapability(Capability):
    name = "web"
    description = "网页抓取与联网搜索"

    def tools(self) -> List[ToolDecl]:
        return [
            CapabilityTool(web_search_tool, tags=("web",)),
            CapabilityTool(fetch_url_tool, tags=("web",)),
            CapabilityTool(stock_quote_tool, tags=("web",)),
            CapabilityTool(save_url_to_knowledge_tool, tags=("web", "knowledge")),
        ]

    def intents(self) -> List[IntentSpec]:
        return [
            IntentSpec(
                name="web_browse",
                description="抓取或阅读指定网页 / 链接的内容，提取或总结网页正文",
                keywords=[
                    "网页", "链接", "http://", "https://", "www.", "抓取", "网址", "url", "网站", "爬取",
                    "股价", "股票", "行情", "涨跌", "市值", "涨停", "跌停", "大盘", "指数",
                ],
                examples=[
                    "帮我看看这个网页讲了什么 https://example.com/post",
                    "抓取一下这个链接的正文",
                    "总结这篇网页文章的要点",
                    "太极实业现在股价多少",
                    "腾讯控股今天的行情怎么样",
                    "Fetch the content of this URL and summarize it",
                ],
            )
        ]

    def agents(self) -> List[BaseAgent]:
        return [WebAgent()]

    def routes(self) -> Dict[str, List[str]]:
        return {"web_browse": ["web_agent", "general_agent", "fallback_agent"]}
