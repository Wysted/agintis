"""HTTP surface of the agents catalog: list agents and stream a run over SSE.

Pre-stream failures (unknown agent, tenant, size, LLM not configured; for a thread run also
thread not found / busy / in the wrong state) are raised inside the `prepared_run` /
`prepared_thread_run` DEPENDENCY: FastAPI resolves dependencies before it starts the generator
endpoint, so those errors still become a normal status code + error body. A generator path
operation commits `200` on its first yield; nothing that can be checked earlier may live there.

The same dependency owns the cleanup: its teardown closes the run stream when the response
ends, whatever the reason (done, error, client disconnect), without depending on garbage
collection of the suspended endpoint generator.

Wire format: one SSE event per core event, `event` = type, `id` = `<run_id>:<sequence>`, `data` =
JSON of an explicit DTO (the core dataclasses are never serialised directly). Resume
(`Last-Event-ID`) is NOT supported: ids only keep the format resume-safe for later. Request
body size limits (bytes) belong to the gateway; the app caps the message length in characters.
"""

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Annotated, Any, Literal, Self, cast

from fastapi import APIRouter, Depends, Path, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agintis.modules.agents_catalog.application import (
    AGENT_ID_PATTERN,
    AgentsCatalogServices,
)
from agintis.modules.agents_catalog.domain import (
    Clarification,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunStarted,
    Scope,
    ScopeLevel,
    SourceRef,
    Sources,
    Step,
    Token,
    TokenUsage,
)
from agintis.platform.http.tenant import resolve_tenant
from agintis.shared_kernel.errors import InvalidRequestError
from agintis.shared_kernel.ids import TenantId, ThreadId

logger = logging.getLogger(__name__)

MAX_RESUME_DEPTH = 8
MAX_RESUME_ANSWERS = 16
MAX_INTERRUPT_ID_LENGTH = 64
THREAD_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
INTERNAL_ERROR_MESSAGE = "Internal error while streaming the run."


# --- request / response DTOs ------------------------------------------------------------------


class AgentSummary(BaseModel):
    id: str
    description: str


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(pattern=AGENT_ID_PATTERN)
    message: str = Field(min_length=1)


class ThreadRunRequest(BaseModel):
    """One of: a new user turn (`message`), the answer to the single pending question
    (`resume`, any JSON, not null) or the answers to several (`resumes`, `{interrupt_id: JSON}`
    naming exactly the pending interrupts)."""

    model_config = ConfigDict(extra="forbid")

    message: str | None = Field(default=None, min_length=1)
    resume: Any = None
    resumes: dict[str, Any] | None = Field(
        default=None, min_length=1, max_length=MAX_RESUME_ANSWERS
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        given = [v for v in (self.message, self.resume, self.resumes) if v is not None]
        if len(given) != 1:
            raise ValueError(
                "send exactly one of message, resume and resumes (resume must not be null)"
            )
        if self.resumes is not None:
            if any(v is None for v in self.resumes.values()):
                raise ValueError("resumes values must not be null")
            if any(len(key) > MAX_INTERRUPT_ID_LENGTH for key in self.resumes):
                raise ValueError(f"resumes keys are at most {MAX_INTERRUPT_ID_LENGTH} characters")
        return self


def _exceeds_depth(value: object, limit: int) -> bool:
    """Whether a JSON value nests containers deeper than `limit` (recursion bounded by it)."""
    if isinstance(value, dict):
        children: list[object] = list(value.values())
    elif isinstance(value, list):
        children = list(value)
    else:
        return False
    return limit == 0 or any(_exceeds_depth(child, limit - 1) for child in children)


def _check_answer_size(body: ThreadRunRequest, max_chars: int) -> None:
    """Resume payloads are bounded like messages: serialized length and nesting depth."""
    # Keys are interrupt ids (length-checked by the model): only the values count.
    answers = list(body.resumes.values()) if body.resumes is not None else body.resume
    if answers is None:
        return
    if len(json.dumps(answers, separators=(",", ":"), ensure_ascii=False)) > max_chars:
        raise InvalidRequestError(f"the answer exceeds {max_chars} characters")
    if _exceeds_depth(answers, MAX_RESUME_DEPTH):
        raise InvalidRequestError(f"the answer is nested deeper than {MAX_RESUME_DEPTH} levels")


# --- SSE data DTOs (wire shape, pinned by tests) ----------------------------------------------


class RunStartedData(BaseModel):
    run_id: str
    agent_id: str


class TokenData(BaseModel):
    text: str


class StepData(BaseModel):
    node: str
    status: Literal["started", "finished"]
    label: str | None = None


class SourceData(BaseModel):
    source: str
    title: str | None
    locator: str | None


class SourcesData(BaseModel):
    items: list[SourceData]


class ScopeData(BaseModel):
    level: ScopeLevel
    score: float | None


class ClarificationData(BaseModel):
    question: str
    options: list[str]
    interrupt_id: str


class ErrorData(BaseModel):
    code: str
    message: str
    retryable: bool


class UsageData(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int


class DoneData(BaseModel):
    finish_reason: str
    usage: UsageData | None


SCORE_DECIMALS = 4  # wire precision of `Scope.score` (the domain keeps the raw value)


def _round_score(score: float | None) -> float | None:
    return None if score is None else round(score, SCORE_DECIMALS)


def _source(ref: SourceRef) -> SourceData:
    return SourceData(source=ref.source, title=ref.title, locator=ref.locator)


def _usage(usage: TokenUsage | None) -> UsageData | None:
    if usage is None:
        return None
    return UsageData(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cached_input_tokens=usage.cached_input_tokens,
    )


def to_wire(event: object) -> tuple[str, BaseModel] | None:
    """Core event -> (SSE event name, DTO); None for an event this edge does not know."""
    match event:
        case RunStarted():
            return "run_started", RunStartedData(run_id=event.run_id, agent_id=event.agent_id)
        case Token():
            return "token", TokenData(text=event.text)
        case Step():
            return "step", StepData(node=event.node, status=event.status, label=event.label)
        case Sources():
            return "sources", SourcesData(items=[_source(ref) for ref in event.items])
        case Scope():
            return "scope", ScopeData(level=event.level, score=_round_score(event.score))
        case Clarification():
            return "clarification", ClarificationData(
                question=event.question,
                options=list(event.options),
                interrupt_id=event.interrupt_id,
            )
        case RunFailed():
            return "error", ErrorData(
                code=event.code, message=event.message, retryable=event.retryable
            )
        case RunCompleted():
            return "done", DoneData(finish_reason=event.finish_reason, usage=_usage(event.usage))
        case _:
            return None


# --- routes -----------------------------------------------------------------------------------

router = APIRouter(prefix="/v1", tags=["agents"])


async def get_services(request: Request) -> AgentsCatalogServices:
    """Typed services published on `app.state` by the app factory (tests override this)."""
    return cast(AgentsCatalogServices, request.app.state.agents_catalog)


Services = Annotated[AgentsCatalogServices, Depends(get_services)]


async def prepared_run(
    body: RunRequest,
    services: Services,
    tenant_id: Annotated[TenantId, Depends(resolve_tenant)],
) -> AsyncIterator[AsyncGenerator[RunEvent]]:
    """Run every pre-stream check, hand over the not-yet-started stream, close it afterwards."""
    if len(body.message) > services.max_message_chars:
        raise InvalidRequestError(f"message exceeds {services.max_message_chars} characters")
    events = services.run_agent.start(body.agent_id, tenant_id, body.message)
    try:
        yield events
    finally:
        await events.aclose()


@router.get("/agents")
async def list_agents(services: Services) -> list[AgentSummary]:
    return [
        AgentSummary(id=r.agent_id, description=r.description) for r in services.registry.list()
    ]


async def _sse(events: AsyncGenerator[RunEvent]) -> AsyncGenerator[ServerSentEvent]:
    run_id = "unknown"
    sequence = 0
    try:
        async for event in events:
            if isinstance(event, RunStarted):
                run_id = event.run_id
            wire = to_wire(event)
            if wire is None:
                logger.warning("Unknown run event %s skipped", type(event).__name__)
                continue
            name, data = wire
            yield ServerSentEvent(event=name, id=f"{run_id}:{sequence}", data=data)
            sequence += 1
    except Exception:
        # A raise here would silently truncate the stream: end it with an explicit error event.
        logger.exception("Streaming run %s failed", run_id)
        failure = ErrorData(code="internal", message=INTERNAL_ERROR_MESSAGE, retryable=False)
        yield ServerSentEvent(event="error", id=f"{run_id}:{sequence}", data=failure)


@router.post("/runs/stream", response_class=EventSourceResponse)
async def stream_run(
    events: Annotated[AsyncGenerator[RunEvent], Depends(prepared_run)],
) -> AsyncIterator[ServerSentEvent]:
    async with aclosing(_sse(events)) as frames:
        async for frame in frames:
            yield frame


async def prepared_thread_run(
    thread_id: Annotated[str, Path(pattern=THREAD_ID_PATTERN)],
    body: ThreadRunRequest,
    services: Services,
    tenant_id: Annotated[TenantId, Depends(resolve_tenant)],
) -> AsyncIterator[AsyncGenerator[RunEvent]]:
    """Pre-stream checks plus the thread lease, which is released when the response ends."""
    if body.message is not None and len(body.message) > services.max_message_chars:
        raise InvalidRequestError(f"message exceeds {services.max_message_chars} characters")
    _check_answer_size(body, services.max_message_chars)
    async with services.run_agent.start_thread(
        tenant_id,
        ThreadId(thread_id),
        message=body.message,
        resume=body.resume,
        resumes=body.resumes,
    ) as events:
        yield events


@router.post("/threads/{thread_id}/runs/stream", response_class=EventSourceResponse)
async def stream_thread_run(
    events: Annotated[AsyncGenerator[RunEvent], Depends(prepared_thread_run)],
) -> AsyncIterator[ServerSentEvent]:
    async with aclosing(_sse(events)) as frames:
        async for frame in frames:
            yield frame
