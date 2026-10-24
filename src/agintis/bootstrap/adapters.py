"""Adapters between modules: the only place where two business modules meet.

Modules stay independent (import-linter): `agents_catalog` and `conversations` each own the port
they consume, and these classes implement them on top of the OTHER module's `api.py`.
"""

from typing import Protocol

from agintis.agent_kit.usage import RunUsage, UsageSink
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    ThreadAccess,
    ThreadOutcome,
    ThreadRunMode,
    ThreadRunTicket,
)
from agintis.modules.conversations.api import (
    AgentDirectory,
    MessageSource,
    PendingQuestion,
    PendingQuestions,
    RunMode,
    ThreadMessage,
    ThreadsServices,
    ThreadStateStore,
    ThreadStatus,
    ThreadTranscript,
)
from agintis.platform.checkpointer import Checkpointer
from agintis.platform.usage import Purpose, UsageRow
from agintis.platform.usage.repository import PgUsageRepository
from agintis.shared_kernel.errors import FailureInfo
from agintis.shared_kernel.ids import TenantId, ThreadId
from agintis.shared_kernel.ports import Clock, ErrorClassifier

_MODES = {ThreadRunMode.MESSAGE: RunMode.MESSAGE, ThreadRunMode.RESUME: RunMode.RESUME}
_OUTCOMES = {
    ThreadOutcome.IDLE: ThreadStatus.IDLE,
    ThreadOutcome.INTERRUPTED: ThreadStatus.INTERRUPTED,
}


class ConversationsThreadAccess(ThreadAccess):
    """`agents_catalog.ThreadAccess` over the conversations run lease."""

    def __init__(self, threads: ThreadsServices) -> None:
        self._threads = threads

    async def acquire(
        self, tenant_id: TenantId, thread_id: ThreadId, mode: ThreadRunMode
    ) -> ThreadRunTicket:
        lease = await self._threads.acquire_lease.execute(tenant_id, thread_id, _MODES[mode])
        return ThreadRunTicket(
            tenant_id=lease.tenant_id,
            thread_id=lease.thread_id,
            agent_id=lease.agent_id,
            lease_token=lease.token,
        )

    async def release(self, ticket: ThreadRunTicket, outcome: ThreadOutcome) -> None:
        await self._threads.release_lease.execute(
            ticket.tenant_id, ticket.thread_id, ticket.lease_token, _OUTCOMES[outcome]
        )


class RegistryAgentDirectory(AgentDirectory):
    """`conversations.AgentDirectory` over the agents registry."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def exists(self, agent_id: str) -> bool:
        return agent_id in self._registry

    def supports_threads(self, agent_id: str) -> bool:
        return agent_id in self._registry and self._registry.get(agent_id).supports_threads


class AgentsPendingQuestions(PendingQuestions):
    """`conversations.PendingQuestions` over the runner's view of its checkpoint."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    async def for_thread(self, agent_id: str, thread_id: ThreadId) -> tuple[PendingQuestion, ...]:
        if agent_id not in self._registry:
            return ()  # the agent was removed: there is nobody to ask
        pending = await self._registry.get(agent_id).pending_interrupts(thread_id)
        return tuple(PendingQuestion(c.interrupt_id, c.question, c.options) for c in pending)


class AgentsThreadTranscript(ThreadTranscript):
    """`conversations.ThreadTranscript` over the runner's view of its checkpoint."""

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    async def for_thread(self, agent_id: str, thread_id: ThreadId) -> tuple[ThreadMessage, ...]:
        if agent_id not in self._registry:
            return ()  # the agent was removed: its state is unreadable
        turns = await self._registry.get(agent_id).thread_messages(thread_id)
        return tuple(
            ThreadMessage(
                t.role,
                t.text,
                tuple(MessageSource(s.source, s.title, s.locator) for s in t.sources),
            )
            for t in turns
        )


class CheckpointerStateStore(ThreadStateStore):
    """`conversations.ThreadStateStore` over the LangGraph checkpointer."""

    def __init__(self, checkpointer: Checkpointer) -> None:
        self._checkpointer = checkpointer

    async def delete(self, thread_id: ThreadId) -> None:
        await self._checkpointer.delete_thread(thread_id)


class SpecificErrorClassifier(Protocol):
    """Recognises some failures (a `FailureInfo`) and returns None for the rest."""

    def classify(self, exc: BaseException) -> FailureInfo | None: ...


class ChainedErrorClassifier:
    """The run's `ErrorClassifier`: module-specific classifiers first, then the generic one.

    Each specific classifier returns None for what it does not recognise (retrieval outages
    become a retryable `retrieval_unavailable`); the last one (the provider classifier) always
    answers, `internal` included. Lives here so `platform` stays free of module imports.
    """

    def __init__(self, fallback: ErrorClassifier, *specific: SpecificErrorClassifier) -> None:
        self._specific = specific
        self._fallback = fallback

    def classify(self, exc: BaseException) -> FailureInfo:
        for classifier in self._specific:
            if (failure := classifier.classify(exc)) is not None:
                return failure
        return self._fallback.classify(exc)


class PgUsageSink(UsageSink):
    """`agent_kit.UsageSink` over the platform usage ledger: one `chat` row per model of the run."""

    def __init__(self, repository: PgUsageRepository, clock: Clock) -> None:
        self._repository = repository
        self._clock = clock

    async def record(self, run: RunUsage) -> None:
        now = self._clock.now()
        await self._repository.add_all(
            [
                UsageRow(
                    run_id=run.run_id,
                    model=model,
                    purpose=Purpose.CHAT,
                    created_at=now,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    calls=usage.calls,
                    tenant_id=run.tenant_id,
                    agent_id=run.agent_id,
                    thread_id=run.thread_id,
                )
                for model, usage in run.usage.items()
            ]
        )
