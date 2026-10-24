"""Ports of the agents catalog: `AgentRunner` and `ThreadAccess`.

Owned by this module (the consumer); agents and the composition root implement them and reach
them through `api.py`.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from enum import StrEnum

from agintis.modules.agents_catalog.domain import Clarification, RunnerEvent, TranscriptMessage
from agintis.shared_kernel.errors import AppError, ErrorKind
from agintis.shared_kernel.ids import RunId, TenantId, ThreadId


@dataclass(frozen=True, slots=True)
class RunInput:
    """One run. Stateless: `message` only. Thread run: `thread_id` plus EXACTLY one of `message`
    (a new user turn) or `resume`: the answers to the pending interrupts as an explicit
    `{interrupt_id: value}` map (never a bare value: a bare dict that looks like an id map is
    ambiguous for LangGraph, and several pending interrupts reject a bare value)."""

    tenant_id: TenantId
    message: str | None
    run_id: RunId
    thread_id: ThreadId | None = None
    resume: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if (self.message is None) == (self.resume is None):
            raise ValueError("A run needs exactly one of message and resume")
        if self.resume is not None and (self.thread_id is None or not self.resume):
            raise ValueError("Only a thread run can resume, with at least one answer")


class AgentUnavailableError(AppError):
    """The agent exists but cannot run right now (misconfigured or missing LLM)."""

    kind = ErrorKind.UNAVAILABLE
    code = "agent_unavailable"


class ThreadsUnavailableError(AppError):
    """Thread runs are disabled in this deployment (no checkpointer)."""

    kind = ErrorKind.UNAVAILABLE
    code = "threads_unavailable"

    def __init__(self) -> None:
        super().__init__("Conversation threads are not available")


class ThreadsNotSupportedError(AppError):
    """The thread's agent cannot keep conversation state."""

    kind = ErrorKind.INVALID
    code = "agent_threads_unsupported"

    def __init__(self, agent_id: str) -> None:
        super().__init__(f"Agent {agent_id!r} does not support threads")


class ThreadAwaitingAnswerError(AppError):
    """A message was sent to a thread whose checkpoint holds an unanswered question."""

    kind = ErrorKind.CONFLICT
    code = "thread_interrupted"

    def __init__(self) -> None:
        super().__init__("Thread is waiting for the answer to a pending question: resume it")


class NoPendingQuestionError(AppError):
    """A resume was sent to a thread whose checkpoint holds no unanswered question."""

    kind = ErrorKind.CONFLICT
    code = "thread_not_interrupted"

    def __init__(self) -> None:
        super().__init__("Thread is not waiting for an answer: send a message instead")


class AmbiguousResumeError(AppError):
    """`resume` (one bare answer) was sent while several questions are pending."""

    kind = ErrorKind.CONFLICT
    code = "multiple_interrupts_pending"

    def __init__(self) -> None:
        super().__init__(
            "Several questions are pending: answer them all with `resumes`, "
            "keyed by interrupt_id (GET the thread to list them)"
        )


class ResumeMismatchError(AppError):
    """`resumes` does not name exactly the pending interrupts."""

    kind = ErrorKind.CONFLICT
    code = "interrupt_mismatch"

    def __init__(self) -> None:
        super().__init__(
            "The answers do not match the pending questions: "
            "send exactly one answer per interrupt_id (GET the thread to list them)"
        )


class AgentRunner(ABC):
    """Runs one agent and streams its events.

    Contract of `astream` (an async generator, so the use case can close it on cancellation):
    yield content events (token, step, sources, clarification) and optionally a final
    `RunCompleted` carrying finish reason and usage. The use case enforces this: RunStarted and
    RunFailed (or anything else) coming from a runner are dropped. Unexpected failures are
    raised, not swallowed; `asyncio.CancelledError` must propagate.

    A runner that keeps state per thread (`supports_threads`) receives `RunInput.thread_id` and,
    when the run answers an interrupt, `RunInput.resume`. When it pauses on an interrupt it
    yields a `Clarification` and finishes with `RunCompleted(finish_reason="interrupted")`.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def supports_threads(self) -> bool:
        """Whether runs may carry a `thread_id` (state kept between runs)."""
        return False

    def ensure_available(self) -> None:
        """Raise `AgentUnavailableError` when a run would fail for a configuration reason.

        Called BEFORE the stream opens so the client gets a proper status, not an error event.
        """
        return None

    async def pending_interrupts(self, thread_id: ThreadId) -> tuple[Clarification, ...]:
        """The unanswered questions in the thread's state (each with its `interrupt_id`).

        The source of truth for the thread status: the checkpoint, not the `threads` row.
        Runners without thread support have none.
        """
        return ()

    async def thread_messages(self, thread_id: ThreadId) -> tuple[TranscriptMessage, ...]:
        """The chat turns kept in the thread's state, oldest first (user and assistant only),
        each assistant turn with the sources its citations index. Runners without thread
        support, or a thread without state, have none."""
        return ()

    @abstractmethod
    def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]: ...


class ThreadRunMode(StrEnum):
    MESSAGE = "message"
    RESUME = "resume"


class ThreadOutcome(StrEnum):
    """Where a thread goes when its run ends."""

    IDLE = "idle"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class ThreadRunTicket:
    """Proof that a run owns a thread; `lease_token` is opaque and needed to release it."""

    tenant_id: TenantId
    thread_id: ThreadId
    agent_id: str
    lease_token: str


class ThreadAccess(ABC):
    """What a run needs from the conversation threads (implemented in the composition root).

    `acquire` fails (client-safe `AppError`s: not found for the tenant, busy, wrong state)
    BEFORE any stream opens; `release` must be called exactly once per ticket.
    """

    @abstractmethod
    async def acquire(
        self, tenant_id: TenantId, thread_id: ThreadId, mode: ThreadRunMode
    ) -> ThreadRunTicket: ...

    @abstractmethod
    async def release(self, ticket: ThreadRunTicket, outcome: ThreadOutcome) -> None: ...
