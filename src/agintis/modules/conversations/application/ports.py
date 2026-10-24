"""Ports of the conversations module.

Tenant contract (same as knowledge): every repository operation is tenant-scoped and adapters
MUST inject the tenant into the query themselves, so a thread of another tenant is
indistinguishable from a missing one.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import ClassVar, Literal

from agintis.modules.conversations.domain import RunLease, RunMode, Thread, ThreadStatus
from agintis.shared_kernel.criteria import Criteria
from agintis.shared_kernel.ids import TenantId, ThreadId

THREAD_FILTERABLE_FIELDS = frozenset(
    {"id", "tenant_id", "agent_id", "title", "status", "lease_until", "created_at", "updated_at"}
)


class ThreadRepository(ABC):
    FILTERABLE_FIELDS: ClassVar[frozenset[str]] = THREAD_FILTERABLE_FIELDS

    @abstractmethod
    async def add(self, thread: Thread) -> None: ...

    @abstractmethod
    async def get(self, tenant_id: TenantId, thread_id: ThreadId) -> Thread | None: ...

    @abstractmethod
    async def search(self, tenant_id: TenantId, criteria: Criteria) -> list[Thread]:
        """Threads of `tenant_id` matching `criteria` (filter/order/limit); unknown fields raise
        `UnknownFieldError`."""

    @abstractmethod
    async def delete(self, tenant_id: TenantId, thread_id: ThreadId) -> bool:
        """Delete the thread unless a run holds a valid lease. False = nothing deleted (it does
        not exist for the tenant, or it is leased)."""

    @abstractmethod
    async def acquire_run_lease(
        self, tenant_id: TenantId, thread_id: ThreadId, mode: RunMode, ttl: timedelta
    ) -> RunLease:
        """ATOMICALLY move the thread to busy (see `Thread.check_can_start`).

        Pure mutual exclusion: idle, interrupted and expired-busy threads are acquirable for any
        `mode` (recorded in the lease only); whether the run makes sense is decided by the caller
        from the checkpoint. Raises `ThreadNotFoundError` or `ThreadBusyError`; of two concurrent
        callers exactly one wins.
        """

    @abstractmethod
    async def release_run_lease(
        self, tenant_id: TenantId, thread_id: ThreadId, token: str, outcome: ThreadStatus
    ) -> bool:
        """End the run that holds `token`: status becomes `outcome` (idle | interrupted).

        False when the lease is no longer this token's (expired and taken over, or the thread
        is gone): a late release must never clobber a newer run.
        """


class ThreadStateStore(ABC):
    """The conversation state attached to a thread (LangGraph checkpoints)."""

    @abstractmethod
    async def delete(self, thread_id: ThreadId) -> None: ...


class AgentDirectory(ABC):
    """Which agents exist; implemented in the composition root (modules stay independent)."""

    @abstractmethod
    def exists(self, agent_id: str) -> bool: ...

    @abstractmethod
    def supports_threads(self, agent_id: str) -> bool:
        """Whether the agent can keep conversation state (a thread makes no sense otherwise)."""


@dataclass(frozen=True, slots=True)
class PendingQuestion:
    interrupt_id: str
    question: str
    options: tuple[str, ...] = ()


class PendingQuestions(ABC):
    """The unanswered questions of a thread, read from its conversation state."""

    @abstractmethod
    async def for_thread(
        self, agent_id: str, thread_id: ThreadId
    ) -> tuple[PendingQuestion, ...]: ...


class NoPendingQuestions(PendingQuestions):
    async def for_thread(self, agent_id: str, thread_id: ThreadId) -> tuple[PendingQuestion, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class MessageSource:
    """A source an assistant message cites (its `[n]` markers index the message's sources)."""

    source: str
    title: str | None = None
    locator: str | None = None


@dataclass(frozen=True, slots=True)
class ThreadMessage:
    """One chat turn of a thread."""

    role: Literal["user", "assistant"]
    text: str
    sources: tuple[MessageSource, ...] = ()


class ThreadTranscript(ABC):
    """The chat turns of a thread, read from its conversation state, oldest first."""

    @abstractmethod
    async def for_thread(self, agent_id: str, thread_id: ThreadId) -> tuple[ThreadMessage, ...]: ...


class NoThreadTranscript(ThreadTranscript):
    async def for_thread(self, agent_id: str, thread_id: ThreadId) -> tuple[ThreadMessage, ...]:
        return ()


class NoThreadState(ThreadStateStore):
    """State store for deployments without a checkpointer: nothing is ever stored."""

    async def delete(self, thread_id: ThreadId) -> None:
        return None
