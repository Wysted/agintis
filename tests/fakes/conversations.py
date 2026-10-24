"""In-memory conversations ports (the contract suite pins them to the Postgres adapter)."""

from dataclasses import replace
from datetime import timedelta

from agintis.modules.conversations.api import (
    AgentDirectory,
    PendingQuestion,
    PendingQuestions,
    RunLease,
    RunMode,
    Thread,
    ThreadMessage,
    ThreadNotFoundError,
    ThreadRepository,
    ThreadStateStore,
    ThreadStatus,
    ThreadTranscript,
)
from agintis.shared_kernel.criteria import Criteria, ensure_fields, tenant_scoped
from agintis.shared_kernel.criteria.in_memory import apply
from agintis.shared_kernel.ids import TenantId, ThreadId, new_id
from agintis.shared_kernel.ports import Clock


def _record(thread: Thread) -> dict[str, object]:
    return {
        "id": thread.id,
        "tenant_id": thread.tenant_id,
        "agent_id": thread.agent_id,
        "title": thread.title,
        "status": thread.status.value,
        "lease_until": thread.lease_until,
        "created_at": thread.created_at,
        "updated_at": thread.updated_at,
    }


class InMemoryThreadRepository(ThreadRepository):
    """Same observable behaviour as `PgThreadRepository`, with the lease rules of the domain."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self.items: dict[ThreadId, Thread] = {}
        self._tokens: dict[ThreadId, str] = {}

    async def add(self, thread: Thread) -> None:
        self.items[thread.id] = thread

    async def get(self, tenant_id: TenantId, thread_id: ThreadId) -> Thread | None:
        found = self.items.get(thread_id)
        return found if found is not None and found.tenant_id == tenant_id else None

    async def search(self, tenant_id: TenantId, criteria: Criteria) -> list[Thread]:
        ensure_fields(criteria, self.FILTERABLE_FIELDS)
        scoped = replace(criteria, filter=tenant_scoped(tenant_id, criteria.filter))
        return apply(scoped, self.items.values(), as_record=_record)

    async def delete(self, tenant_id: TenantId, thread_id: ThreadId) -> bool:
        thread = await self.get(tenant_id, thread_id)
        if thread is None or thread.is_leased(self._clock.now()):
            return False
        del self.items[thread_id]
        self._tokens.pop(thread_id, None)
        return True

    async def acquire_run_lease(
        self, tenant_id: TenantId, thread_id: ThreadId, mode: RunMode, ttl: timedelta
    ) -> RunLease:
        thread = await self.get(tenant_id, thread_id)
        if thread is None:
            raise ThreadNotFoundError
        now = self._clock.now()
        started = thread.start_run(now, ttl)  # raises ThreadBusyError
        token = new_id()
        self.items[thread_id] = started
        self._tokens[thread_id] = token
        assert started.lease_until is not None
        return RunLease(thread_id, tenant_id, thread.agent_id, mode, token, started.lease_until)

    async def release_run_lease(
        self, tenant_id: TenantId, thread_id: ThreadId, token: str, outcome: ThreadStatus
    ) -> bool:
        thread = await self.get(tenant_id, thread_id)
        if thread is None or self._tokens.get(thread_id) != token:
            return False
        self.items[thread_id] = thread.end_run(outcome, self._clock.now())
        del self._tokens[thread_id]
        return True


class FakeAgentDirectory(AgentDirectory):
    """`agent_ids` exist; those also in `stateless` keep no state (no thread support)."""

    def __init__(self, *agent_ids: str, stateless: tuple[str, ...] = ()) -> None:
        self._agent_ids = set(agent_ids)
        self._stateless = set(stateless)

    def exists(self, agent_id: str) -> bool:
        return agent_id in self._agent_ids

    def supports_threads(self, agent_id: str) -> bool:
        return agent_id in self._agent_ids and agent_id not in self._stateless


class FakePendingQuestions(PendingQuestions):
    def __init__(self, *questions: PendingQuestion) -> None:
        self.questions = questions
        self.calls: list[tuple[str, ThreadId]] = []

    async def for_thread(self, agent_id: str, thread_id: ThreadId) -> tuple[PendingQuestion, ...]:
        self.calls.append((agent_id, thread_id))
        return self.questions


class FakeThreadTranscript(ThreadTranscript):
    def __init__(self, *messages: ThreadMessage) -> None:
        self.messages = messages
        self.calls: list[tuple[str, ThreadId]] = []

    async def for_thread(self, agent_id: str, thread_id: ThreadId) -> tuple[ThreadMessage, ...]:
        self.calls.append((agent_id, thread_id))
        return self.messages


class FakeThreadStateStore(ThreadStateStore):
    def __init__(self, error: Exception | None = None) -> None:
        self.deleted: list[ThreadId] = []
        self._error = error

    async def delete(self, thread_id: ThreadId) -> None:
        if self._error is not None:
            raise self._error
        self.deleted.append(thread_id)
