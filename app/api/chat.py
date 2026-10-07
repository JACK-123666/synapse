"""对话 API。

POST /chat
- 接收用户消息，执行完整处理链路：
  意图识别 → 路由调度 → Agent 执行 → 记忆管理 → 返回回复

POST /chat/stream
- 同上，以 SSE（text/event-stream）流式返回

系统类接口（/health、/models、/metrics）见 app/api/system.py
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, Header
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.deps import CurrentUser, get_current_user
from app.services.chat import ChatInput, get_chat_service

logger = logging.getLogger(__name__)

router = APIRouter()


# 请求 / 响应模型

class ChatRequest(BaseModel):
    """对话请求。

    填 session_id 和 message 就能用，user_id 选填。
    同一个 session_id 会共享短期记忆，实现多轮对话。
    model 可选：临时指定本次请求使用的模型（不影响全局配置）。
    """

    session_id: str = Field(
        ..., min_length=1,
        description="会话标识，随便起个名字就行（如 test-001）。同一会话多轮对话用同一个 ID",
    )
    message: str = Field(
        ..., min_length=1,
        description="你想说的话，支持模糊短句（如「那个怎么弄」「帮我查一下」）",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="可选。填了会记录你的偏好，下次回答更懂你（开启鉴权时自动使用登录用户）",
    )
    model: Optional[str] = Field(
        default=None,
        description="可选。本次请求使用的模型名，覆盖全局配置（如 gpt-4o、deepseek-chat）",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": "demo-001",
                    "message": "什么是向量数据库？",
                    "user_id": "u1",
                },
                {
                    "session_id": "demo-001",
                    "message": "它和 MySQL 有什么区别",
                },
            ]
        }
    }


class ChatResponse(BaseModel):
    """对话响应。"""

    reply: str = Field(description="Agent 生成的回复内容")
    intent: str = Field(description="识别出的意图：knowledge_retrieval 查知识 / summarize 做摘要 / small_talk 闲聊 / 以及各能力与插件注册的意图")
    agent_used: str = Field(description="实际干活的是哪个 Agent")
    confidence: float = Field(description="意图识别的把握有多大（0~1）")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="引用的知识库片段")
    web_sources: List[Dict[str, Any]] = Field(default_factory=list, description="引用的联网搜索结果")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="本次调用过的工具")


def _to_input(request: ChatRequest, x_web_search: str) -> ChatInput:
    return ChatInput(
        session_id=request.session_id,
        message=request.message,
        user_id=request.user_id,
        model=request.model,
        web_search=x_web_search == "1",
    )


# GET /chat — 返回聊天网页

@router.get("/chat", include_in_schema=False)
async def chat_page():
    """聊天页面。"""
    return FileResponse("app/static/index.html")


# POST /chat

@router.post("/chat", response_model=ChatResponse, tags=["对话"], summary="发送消息")
async def chat(
    request: ChatRequest,
    x_web_search: str = Header(default="1", alias="X-Web-Search"),
    user: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    """发一条消息给 Synapse，拿到回复。

    背后自动完成：意图识别 → 记忆召回 → Agent 执行 → 记忆更新。

    同一个 session_id 连续调用即可多轮对话，系统会记住上下文。
    对话超过 8 轮会自动压缩历史，不额外消耗 Token。

    Header X-Web-Search: 1/0 控制是否启用联网搜索。
    """
    result = await get_chat_service().chat(user, _to_input(request, x_web_search))
    return ChatResponse(**result.to_dict())


# POST /chat/stream

def _sse(event: Dict[str, Any]) -> str:
    event_type = event.get("type", "message")
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"


@router.post("/chat/stream", tags=["对话"], summary="发送消息（流式）")
async def chat_stream(
    request: ChatRequest,
    x_web_search: str = Header(default="1", alias="X-Web-Search"),
    user: CurrentUser = Depends(get_current_user),
) -> StreamingResponse:
    """与 POST /chat 相同，但以 SSE 流式返回。

    事件类型：
    - meta：意图与置信度
    - token：增量文本
    - tool_start / tool_end：工具调用开始 / 结束
    - done：完整结果（与 /chat 响应字段一致）
    - error：生成中断
    """
    inp = _to_input(request, x_web_search)

    async def event_source() -> AsyncIterator[str]:
        """把对话服务的事件流转成 SSE 文本；生成过程中的异常也转成 error 事件，避免连接被硬断。"""
        try:
            async for event in get_chat_service().stream(user, inp):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[API] 流式对话失败: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
