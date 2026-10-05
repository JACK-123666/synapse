"""代码仓库助手能力（GitHub / GitLab / 本地 Git）。"""

from __future__ import annotations

from typing import Dict, List

from app.agents.base import AgentContext, BaseAgent
from app.agents.langchain_agent import LangChainAgent, format_recall
from app.capabilities.base import Capability, CapabilityTool, ToolDecl
from app.capabilities.repo.tools import READ_TOOLS, WRITE_TOOLS
from app.intent.catalog import IntentSpec


class RepoAgent(LangChainAgent):
    """代码仓库助手。"""

    agent_id = "repo_agent"
    description = "代码仓库助手"
    tool_tags = ("repo", "knowledge")
    max_tokens = 3000

    def build_system_prompt(self, context: AgentContext) -> str:
        parts: List[str] = [
            "你是代码仓库助手，可以查看提交记录、Issue、PR/MR、读取代码文件、搜索代码、生成变更日志。",
            "先用 repo_list 确认有哪些仓库；用户没指定仓库且只有一个时可以直接使用。",
            "总结 PR / 提交时先读取 diff 再归纳要点；引用代码时注明文件路径。",
            "创建 Issue、发表评论属于写操作，只会生成待确认操作，务必把操作 ID 告诉用户，由用户确认后执行。",
        ]
        recall_text = format_recall(context.long_term_recall)
        if recall_text:
            parts.append(f"\n【历史相关摘要】\n{recall_text}")
        if context.user_profile_context:
            parts.append(f"\n【用户画像】\n{context.user_profile_context}")
        return "\n".join(parts)


class RepoCapability(Capability):
    name = "repo"
    description = "代码仓库助手"

    def tools(self) -> List[ToolDecl]:
        return [CapabilityTool(t, tags=("repo",)) for t in READ_TOOLS] + [
            CapabilityTool(t, tags=("repo",), write=True) for t in WRITE_TOOLS
        ]

    def intents(self) -> List[IntentSpec]:
        return [
            IntentSpec(
                name="repo_management",
                description="代码仓库相关：查看提交记录、Issue、PR/MR、代码文件，生成变更日志，代码问答（GitHub / GitLab / 本地 Git）",
                keywords=[
                    "仓库", "提交记录", "commit", "issue", "pull request", "merge request", "合并请求",
                    "分支", "branch", "github", "gitlab", "changelog", "变更日志", "代码库",
                ],
                examples=[
                    "看一下 synapse 仓库最近的提交",
                    "列出这个项目还没关闭的 issue",
                    "帮我总结一下 PR 12 改了什么",
                    "生成最近一周的 changelog",
                    "Show me the latest commits on the main branch",
                ],
            )
        ]

    def agents(self) -> List[BaseAgent]:
        return [RepoAgent()]

    def routes(self) -> Dict[str, List[str]]:
        return {"repo_management": ["repo_agent", "general_agent", "fallback_agent"]}
