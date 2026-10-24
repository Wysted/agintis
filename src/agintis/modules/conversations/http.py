"""HTTP surface of the conversations module: tenant-scoped thread CRUD and transcript.

Every route resolves the tenant first and every use case filters by it, so a thread of another
tenant answers 404 exactly like a missing one. Running a thread (streaming a message or a
resume) lives in the agents catalog (`POST /v1/threads/{id}/runs/stream`).
"""

from datetime import datetime
from http import HTTPStatus
from typing import Annotated, Literal, cast

from fastapi import APIRouter, Depends, Path, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from agintis.modules.conversations.application import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    ThreadsServices,
)
from agintis.modules.conversations.domain import TITLE_MAX_LENGTH, Thread, ThreadStatus
from agintis.platform.http.tenant import resolve_tenant
from agintis.shared_kernel.ids import TenantId, ThreadId

AGENT_ID_MAX_LENGTH = 64
AGENT_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
THREAD_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
CURSOR_MAX_LENGTH = 512


class CreateThreadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(max_length=AGENT_ID_MAX_LENGTH, pattern=AGENT_ID_PATTERN)
    # Length is checked by the domain (after trimming), so a padded title is not rejected early.
    title: str | None = Field(default=None, max_length=TITLE_MAX_LENGTH * 4)


class ThreadResponse(BaseModel):
    """`status` is the stored lease-level status: a `busy` thread whose run died stays `busy`
    until the lease expires (then it is acquirable again); see `Thread`."""

    id: str
    agent_id: str
    title: str | None
    status: ThreadStatus
    created_at: datetime
    updated_at: datetime


class PendingClarification(BaseModel):
    interrupt_id: str
    question: str
    options: list[str]


class ThreadDetailResponse(ThreadResponse):
    # The questions an interrupted thread waits for (answer them with `resume`/`resumes`).
    pending_clarifications: list[PendingClarification]


class SourceResponse(BaseModel):
    source: str
    title: str | None
    locator: str | None


class MessageResponse(BaseModel):
    """One chat turn; the `[n]` markers of an assistant `text` index its `sources` (1-based)."""

    role: Literal["user", "assistant"]
    text: str
    sources: list[SourceResponse]


class ThreadMessagesResponse(BaseModel):
    items: list[MessageResponse]


class ThreadListResponse(BaseModel):
    items: list[ThreadResponse]
    next_cursor: str | None


def to_response(thread: Thread) -> ThreadResponse:
    return ThreadResponse(
        id=thread.id,
        agent_id=thread.agent_id,
        title=thread.title,
        status=thread.status,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
    )


router = APIRouter(prefix="/v1", tags=["threads"])


async def get_thread_services(request: Request) -> ThreadsServices:
    """Typed services published on `app.state` by the app factory (tests override this)."""
    return cast(ThreadsServices, request.app.state.threads)


Services = Annotated[ThreadsServices, Depends(get_thread_services)]
Tenant = Annotated[TenantId, Depends(resolve_tenant)]
ThreadIdParam = Annotated[str, Path(pattern=THREAD_ID_PATTERN)]


@router.post("/threads", status_code=HTTPStatus.CREATED)
async def create_thread(
    body: CreateThreadRequest, services: Services, tenant_id: Tenant
) -> ThreadResponse:
    thread = await services.create.execute(tenant_id, body.agent_id, body.title)
    return to_response(thread)


@router.get("/threads")
async def list_threads(
    services: Services,
    tenant_id: Tenant,
    agent_id: Annotated[
        str | None, Query(max_length=AGENT_ID_MAX_LENGTH, pattern=AGENT_ID_PATTERN)
    ] = None,
    status: ThreadStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[str | None, Query(max_length=CURSOR_MAX_LENGTH)] = None,
) -> ThreadListResponse:
    page = await services.list.execute(
        tenant_id, agent_id=agent_id, status=status, limit=limit, cursor=cursor
    )
    return ThreadListResponse(
        items=[to_response(t) for t in page.items], next_cursor=page.next_cursor
    )


@router.get("/threads/{thread_id}")
async def get_thread(
    thread_id: ThreadIdParam, services: Services, tenant_id: Tenant
) -> ThreadDetailResponse:
    detail = await services.get.detail(tenant_id, ThreadId(thread_id))
    return ThreadDetailResponse(
        **to_response(detail.thread).model_dump(),
        pending_clarifications=[
            PendingClarification(
                interrupt_id=q.interrupt_id, question=q.question, options=list(q.options)
            )
            for q in detail.pending
        ],
    )


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: ThreadIdParam, services: Services, tenant_id: Tenant
) -> ThreadMessagesResponse:
    messages = await services.get.messages(tenant_id, ThreadId(thread_id))
    return ThreadMessagesResponse(
        items=[
            MessageResponse(
                role=m.role,
                text=m.text,
                sources=[
                    SourceResponse(source=s.source, title=s.title, locator=s.locator)
                    for s in m.sources
                ],
            )
            for m in messages
        ]
    )


@router.delete("/threads/{thread_id}", status_code=HTTPStatus.NO_CONTENT)
async def delete_thread(
    thread_id: ThreadIdParam, services: Services, tenant_id: Tenant
) -> Response:
    await services.delete.execute(tenant_id, ThreadId(thread_id))
    return Response(status_code=HTTPStatus.NO_CONTENT)
