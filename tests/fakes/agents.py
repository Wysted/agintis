"""Fake AgentRunners for the agents catalog and HTTP tests."""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import cast

from agintis.modules.agents_catalog.api import (
    AgentRunner,
    AgentUnavailableError,
    Clarification,
    RunInput,
    RunnerEvent,
    ThreadAccess,
    ThreadOutcome,
    ThreadRunMode,
    ThreadRunTicket,
    Token,
    TranscriptMessage,
)
from agintis.shared_kernel.ids import TenantId, ThreadId


class FakeRunner(AgentRunner):
    """Yields scripted events (loosely typed to feed contract violations), then raises `error`."""

    def __init__(
        self,
        events: Sequence[object] = (),
        *,
        agent_id: str = "fake",
        description: str = "A fake agent",
        error: BaseException | None = None,
        unavailable: bool = False,
        supports_threads: bool = False,
        pending: Sequence[Clarification] = (),
        pending_error: Exception | None = None,
        messages: Sequence[TranscriptMessage] = (),
    ) -> None:
        self.messages = tuple(messages)  # what the "checkpoint" holds as chat turns
        self.pending = tuple(pending)  # what the "checkpoint" holds; tests may change it
        self._pending_error = pending_error
        self._supports_threads = supports_threads
        self._agent_id = agent_id
        self._description = description
        self._events = tuple(events)
        self._error = error
        self._unavailable = unavailable
        self.inputs: list[RunInput] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def description(self) -> str:
        return self._description

    @property
    def supports_threads(self) -> bool:
        return self._supports_threads

    async def pending_interrupts(self, thread_id: ThreadId) -> tuple[Clarification, ...]:
        if self._pending_error is not None:
            raise self._pending_error
        return self.pending

    async def thread_messages(self, thread_id: ThreadId) -> tuple[TranscriptMessage, ...]:
        return self.messages

    def ensure_available(self) -> None:
        if self._unavailable:
            raise AgentUnavailableError(
                "The language model is not configured", code="llm_not_configured"
            )

    async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
        self.inputs.append(run_input)
        for event in self._events:
            yield cast(RunnerEvent, event)
        if self._error is not None:
            raise self._error


class BlockingRunner(AgentRunner):
    """Yields one event, then waits forever; records whether its stream was cancelled/closed."""

    def __init__(self, first: RunnerEvent) -> None:
        self._first = first
        self.started = asyncio.Event()
        self.cancelled = False
        self.finalized = False
        self.finalized_event = asyncio.Event()

    @property
    def supports_threads(self) -> bool:
        return True

    @property
    def agent_id(self) -> str:
        return "blocking"

    @property
    def description(self) -> str:
        return "Blocks after the first event"

    async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
        try:
            yield self._first
            self.started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.finalized = True
            self.finalized_event.set()


class EndlessRunner(AgentRunner):
    """Yields tokens without ever awaiting anything else (always suspended at a yield)."""

    def __init__(self) -> None:
        self.finalized_event = asyncio.Event()

    @property
    def supports_threads(self) -> bool:
        return True

    @property
    def agent_id(self) -> str:
        return "endless"

    @property
    def description(self) -> str:
        return "Never ends"

    async def astream(self, run_input: RunInput) -> AsyncGenerator[RunnerEvent]:
        try:
            while True:
                yield Token("x")
        finally:
            self.finalized_event.set()


class FakeThreadAccess(ThreadAccess):
    """Records acquire/release; `acquire_error` simulates not found / busy / wrong state."""

    def __init__(
        self,
        agent_id: str = "fake",
        *,
        acquire_error: Exception | None = None,
        release_error: Exception | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._acquire_error = acquire_error
        self._release_error = release_error
        self.acquired: list[tuple[TenantId, ThreadId, ThreadRunMode]] = []
        self.released: list[tuple[ThreadRunTicket, ThreadOutcome]] = []
        self.released_event = asyncio.Event()

    async def acquire(
        self, tenant_id: TenantId, thread_id: ThreadId, mode: ThreadRunMode
    ) -> ThreadRunTicket:
        if self._acquire_error is not None:
            raise self._acquire_error
        self.acquired.append((tenant_id, thread_id, mode))
        return ThreadRunTicket(tenant_id, thread_id, self._agent_id, f"token-{len(self.acquired)}")

    async def release(self, ticket: ThreadRunTicket, outcome: ThreadOutcome) -> None:
        self.released.append((ticket, outcome))
        self.released_event.set()
        if self._release_error is not None:
            raise self._release_error
