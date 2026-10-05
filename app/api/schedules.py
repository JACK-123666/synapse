"""定时任务 API。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.capabilities.knowledge.service import Principal
from app.capabilities.scheduler.cron import ScheduleError, next_runs, to_cron
from app.capabilities.scheduler.service import KINDS, get_scheduler_service
from app.core.deps import CurrentUser, get_current_user

router = APIRouter(prefix="/schedules", tags=["定时任务"])


class ScheduleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    when: str = Field(..., description="cron 表达式（分 时 日 月 周）或自然语言，如“每天早上 9 点”")
    kind: str = Field(default="prompt", description=f"任务类型：{' / '.join(KINDS)}")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description='prompt: {"prompt": "..."}；web_watch: {"url": "..."}；kb_sync: {"kb_id": "...", "urls": [...]}',
    )
    webhook_url: str = Field(default="", description="结果推送地址（飞书 / 企业微信 / 钉钉 / 通用 HTTP）")
    enabled: bool = True


class ScheduleUpdate(BaseModel):
    name: Optional[str] = None
    when: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    enabled: Optional[bool] = None


class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1)


def _who(user: CurrentUser) -> Principal:
    return Principal(user_id=user.id, is_admin=user.is_admin)


def _http_error(exc: ScheduleError) -> HTTPException:
    status = 404 if "不存在" in str(exc) or "找不到" in str(exc) else 400
    return HTTPException(status_code=status, detail=str(exc))


@router.post("/parse", summary="把时间描述解析为 cron，并给出接下来的执行时间")
async def parse(req: ParseRequest, _: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        cron = await to_cron(req.text)
    except ScheduleError as exc:
        raise _http_error(exc) from exc
    return {"cron": cron, "next_runs": next_runs(cron, 5)}


@router.get("", summary="定时任务列表")
async def list_schedules(user: CurrentUser = Depends(get_current_user)) -> List[Dict[str, Any]]:
    return await get_scheduler_service().list(_who(user))


@router.post("", summary="创建定时任务")
async def create_schedule(req: ScheduleCreate, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        cron = await to_cron(req.when)
        return await get_scheduler_service().create(
            _who(user), name=req.name, cron=cron, kind=req.kind,
            payload=req.payload, webhook_url=req.webhook_url, enabled=req.enabled,
        )
    except ScheduleError as exc:
        raise _http_error(exc) from exc


@router.get("/{schedule_id}", summary="定时任务详情")
async def get_schedule(schedule_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return await get_scheduler_service().get(_who(user), schedule_id)
    except ScheduleError as exc:
        raise _http_error(exc) from exc


@router.patch("/{schedule_id}", summary="修改定时任务")
async def update_schedule(
    schedule_id: str, req: ScheduleUpdate, user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        fields = req.model_dump(exclude_none=True)
        if "when" in fields:
            fields["cron"] = await to_cron(fields.pop("when"))
        return await get_scheduler_service().update(_who(user), schedule_id, **fields)
    except ScheduleError as exc:
        raise _http_error(exc) from exc


@router.delete("/{schedule_id}", summary="删除定时任务")
async def delete_schedule(schedule_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        await get_scheduler_service().delete(_who(user), schedule_id)
    except ScheduleError as exc:
        raise _http_error(exc) from exc
    return {"ok": True}


@router.post("/{schedule_id}/run", summary="立即执行一次")
async def run_schedule(schedule_id: str, user: CurrentUser = Depends(get_current_user)) -> Dict[str, Any]:
    try:
        return await get_scheduler_service().run_now(_who(user), schedule_id)
    except ScheduleError as exc:
        raise _http_error(exc) from exc


@router.get("/{schedule_id}/runs", summary="执行记录")
async def list_runs(
    schedule_id: str,
    limit: int = Query(20, ge=1, le=200),
    user: CurrentUser = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    try:
        return await get_scheduler_service().runs(_who(user), schedule_id, limit)
    except ScheduleError as exc:
        raise _http_error(exc) from exc
