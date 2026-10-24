import logging
from datetime import timedelta

import pytest

from agintis.modules.conversations.api import (
    AcquireRunLease,
    CreateThread,
    DeleteThread,
    GetThread,
    InvalidCursorError,
    ListThreads,
    MessageSource,
    PendingQuestion,
    ReleaseRunLease,
    RunMode,
    ThreadAgentNotFoundError,
    ThreadAgentUnsupportedError,
    ThreadBusyError,
    ThreadMessage,
    ThreadNotFoundError,
    ThreadsDisabledError,
    ThreadStatus,
)
from agintis.shared_kernel.criteria import UnknownFieldError
from agintis.shared_kernel.criteria.criteria import Criteria
from agintis.shared_kernel.criteria.filters import Filter
from agintis.shared_kernel.ids import TenantId, ThreadId
from tests.fakes.clock import StepClock
from tests.fakes.conversations import (
    FakeAgentDirectory,
    FakePendingQuestions,
    FakeThreadStateStore,
    FakeThreadTranscript,
    InMemoryThreadRepository,
)

A = TenantId("tenant-a")
B = TenantId("tenant-b")
TTL = timedelta(minutes=5)


class World:
    def __init__(self) -> None:
        self.clock = StepClock()
        self.repo = InMemoryThreadRepository(self.clock)
        self.state = FakeThreadStateStore()
        self.counter = 0
        self.questions = FakePendingQuestions(PendingQuestion("i1", "Which one?", ("a",)))
        self.create = CreateThread(
            self.repo,
            FakeAgentDirectory("chat", "stateless", stateless=("stateless",)),
            self.clock,
            id_generator=self._next_id,
        )
        self.transcript = FakeThreadTranscript(
            ThreadMessage("user", "hi"),
            ThreadMessage("assistant", "hello [1]", (MessageSource("doc", "Doc", "p. 1"),)),
        )
        self.get = GetThread(self.repo, self.questions, self.transcript)
        self.list = ListThreads(self.repo, max_page_size=3)
        self.delete = DeleteThread(self.repo, self.state)
        self.acquire = AcquireRunLease(self.repo, TTL)
        self.release = ReleaseRunLease(self.repo)

    def _next_id(self) -> str:
        self.counter += 1
        return f"t{self.counter}"


@pytest.fixture
def world() -> World:
    return World()


async def test_create_validates_the_agent_and_persists_idle(world: World) -> None:
    thread = await world.create.execute(A, "chat", " Plan ")

    assert (thread.id, thread.title, thread.status) == ("t1", "Plan", ThreadStatus.IDLE)
    assert await world.get.execute(A, thread.id) == thread
    with pytest.raises(ThreadAgentNotFoundError):
        await world.create.execute(A, "nope")


async def test_create_refuses_an_agent_that_keeps_no_state(world: World) -> None:
    with pytest.raises(ThreadAgentUnsupportedError) as info:
        await world.create.execute(A, "stateless")

    assert info.value.code == "agent_threads_unsupported"
    assert (await world.list.execute(A)).items == ()


async def test_create_is_unavailable_when_threads_are_disabled(world: World) -> None:
    disabled = CreateThread(
        world.repo, FakeAgentDirectory("chat"), world.clock, threads_enabled=False
    )

    with pytest.raises(ThreadsDisabledError):
        await disabled.execute(A, "chat")
    with pytest.raises(ThreadAgentNotFoundError):  # an unknown agent is still a 404 first
        await disabled.execute(A, "nope")


async def test_detail_reads_the_pending_questions_only_for_an_interrupted_thread(
    world: World,
) -> None:
    thread = await world.create.execute(A, "chat")

    idle = await world.get.detail(A, thread.id)
    lease = await world.acquire.execute(A, thread.id, RunMode.MESSAGE)
    await world.release.execute(A, thread.id, lease.token, ThreadStatus.INTERRUPTED)
    interrupted = await world.get.detail(A, thread.id)

    assert idle.pending == ()
    assert interrupted.pending == (PendingQuestion("i1", "Which one?", ("a",)),)
    assert world.questions.calls == [("chat", thread.id)]  # one read, none for the idle one
    with pytest.raises(ThreadNotFoundError):
        await world.get.detail(B, thread.id)
    assert world.questions.calls == [("chat", thread.id)]  # the tenant check came first


async def test_messages_come_from_the_transcript_port_of_the_threads_agent(world: World) -> None:
    thread = await world.create.execute(A, "chat")

    assert await world.get.messages(A, thread.id) == world.transcript.messages
    assert world.transcript.calls == [("chat", thread.id)]


async def test_messages_check_the_tenant_before_reading_the_transcript(world: World) -> None:
    thread = await world.create.execute(A, "chat")

    with pytest.raises(ThreadNotFoundError):
        await world.get.messages(B, thread.id)
    with pytest.raises(ThreadNotFoundError):
        await world.get.messages(A, ThreadId("missing"))
    assert world.transcript.calls == []


async def test_get_is_tenant_scoped(world: World) -> None:
    thread = await world.create.execute(A, "chat")

    with pytest.raises(ThreadNotFoundError):
        await world.get.execute(B, thread.id)
    with pytest.raises(ThreadNotFoundError):
        await world.get.execute(A, ThreadId("missing"))


async def test_list_pages_newest_updated_first_without_overlaps(world: World) -> None:
    ids = []
    for _ in range(7):
        ids.append((await world.create.execute(A, "chat")).id)
        world.clock.advance(seconds=1)
    await world.create.execute(B, "chat")

    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = await world.list.execute(A, limit=50, cursor=cursor)  # capped to max_page_size=3
        seen.extend(t.id for t in page.items)
        pages += 1
        cursor = page.next_cursor
        if cursor is None:
            break

    assert pages == 3
    assert seen == list(reversed(ids))


async def test_list_orders_ties_by_id_and_keeps_the_cursor_stable(world: World) -> None:
    for _ in range(4):  # same instant: only the id tiebreak orders them
        await world.create.execute(A, "chat")

    first = await world.list.execute(A, limit=2)
    second = await world.list.execute(A, limit=2, cursor=first.next_cursor)

    assert [t.id for t in first.items] == ["t1", "t2"]
    assert [t.id for t in second.items] == ["t3", "t4"]
    assert second.next_cursor is None


async def test_list_filters_by_agent_and_status(world: World) -> None:
    quiet = await world.create.execute(A, "chat")
    busy = await world.create.execute(A, "chat")
    await world.acquire.execute(A, busy.id, RunMode.MESSAGE)

    by_status = await world.list.execute(A, status=ThreadStatus.BUSY)
    by_agent = await world.list.execute(A, agent_id="other")

    assert [t.id for t in by_status.items] == [busy.id]
    assert by_agent.items == ()
    assert quiet.id not in [t.id for t in by_status.items]


@pytest.mark.parametrize("cursor", ["!!!", "bm90LWpzb24", "e30", "eyJ1IjoiMjAyNiIsImkiOjF9"])
async def test_list_rejects_garbage_cursors(world: World, cursor: str) -> None:
    with pytest.raises(InvalidCursorError):
        await world.list.execute(A, cursor=cursor)


async def test_repository_rejects_unknown_filter_fields(world: World) -> None:
    with pytest.raises(UnknownFieldError):
        await world.repo.search(A, Criteria(filter=Filter.eq("secret", "x")))


async def test_delete_removes_the_row_then_the_state(world: World) -> None:
    thread = await world.create.execute(A, "chat")

    await world.delete.execute(A, thread.id)

    assert world.state.deleted == [thread.id]
    with pytest.raises(ThreadNotFoundError):
        await world.get.execute(A, thread.id)


async def test_delete_is_tenant_scoped_and_leaves_foreign_state_alone(world: World) -> None:
    thread = await world.create.execute(A, "chat")

    with pytest.raises(ThreadNotFoundError):
        await world.delete.execute(B, thread.id)

    assert world.state.deleted == []
    assert await world.get.execute(A, thread.id) == thread


async def test_delete_refuses_a_thread_with_a_running_run(world: World) -> None:
    thread = await world.create.execute(A, "chat")
    await world.acquire.execute(A, thread.id, RunMode.MESSAGE)

    with pytest.raises(ThreadBusyError):
        await world.delete.execute(A, thread.id)

    assert world.state.deleted == []
    world.clock.advance(minutes=6)  # the lease expired: the run is dead
    await world.delete.execute(A, thread.id)


async def test_delete_logs_a_failing_state_cleanup_but_the_thread_is_gone(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    thread = await world.create.execute(A, "chat")
    failing = DeleteThread(world.repo, FakeThreadStateStore(error=RuntimeError("db down")))

    with caplog.at_level(logging.ERROR):
        await failing.execute(A, thread.id)

    assert "could not be" in caplog.text
    with pytest.raises(ThreadNotFoundError):
        await world.get.execute(A, thread.id)


async def test_lease_lifecycle_is_mode_agnostic(world: World) -> None:
    thread = await world.create.execute(A, "chat")

    lease = await world.acquire.execute(A, thread.id, RunMode.MESSAGE)
    assert lease.agent_id == "chat"
    assert (await world.get.execute(A, thread.id)).status is ThreadStatus.BUSY
    assert await world.release.execute(A, thread.id, lease.token, ThreadStatus.INTERRUPTED)

    # Interrupted and idle threads accept either mode: the checkpoint decides, not the lease.
    resume = await world.acquire.execute(A, thread.id, RunMode.RESUME)
    assert await world.release.execute(A, thread.id, resume.token, ThreadStatus.IDLE)
    assert (await world.get.execute(A, thread.id)).status is ThreadStatus.IDLE
    again = await world.acquire.execute(A, thread.id, RunMode.RESUME)
    assert again.mode is RunMode.RESUME


async def test_second_acquire_conflicts_until_the_lease_expires(world: World) -> None:
    thread = await world.create.execute(A, "chat")
    first = await world.acquire.execute(A, thread.id, RunMode.MESSAGE)

    with pytest.raises(ThreadBusyError):
        await world.acquire.execute(A, thread.id, RunMode.MESSAGE)
    world.clock.advance(minutes=6)
    second = await world.acquire.execute(A, thread.id, RunMode.MESSAGE)

    assert second.token != first.token


async def test_a_resume_can_take_over_a_dead_resume_run(world: World) -> None:
    thread = await world.create.execute(A, "chat")
    first = await world.acquire.execute(A, thread.id, RunMode.MESSAGE)
    await world.release.execute(A, thread.id, first.token, ThreadStatus.INTERRUPTED)
    await world.acquire.execute(A, thread.id, RunMode.RESUME)  # this run crashes: never released

    with pytest.raises(ThreadBusyError):
        await world.acquire.execute(A, thread.id, RunMode.RESUME)
    world.clock.advance(minutes=6)
    retry = await world.acquire.execute(A, thread.id, RunMode.RESUME)  # the checkpoint decides

    assert retry.mode is RunMode.RESUME


async def test_a_late_release_cannot_clobber_a_newer_run(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    thread = await world.create.execute(A, "chat")
    old = await world.acquire.execute(A, thread.id, RunMode.MESSAGE)
    world.clock.advance(minutes=6)
    await world.acquire.execute(A, thread.id, RunMode.MESSAGE)

    with caplog.at_level(logging.WARNING):
        released = await world.release.execute(A, thread.id, old.token, ThreadStatus.IDLE)

    assert released is False
    assert "lost" in caplog.text
    assert (await world.get.execute(A, thread.id)).status is ThreadStatus.BUSY


async def test_acquire_is_tenant_scoped(world: World) -> None:
    thread = await world.create.execute(A, "chat")

    with pytest.raises(ThreadNotFoundError):
        await world.acquire.execute(B, thread.id, RunMode.MESSAGE)


async def test_a_run_cannot_end_busy(world: World) -> None:
    thread = await world.create.execute(A, "chat")
    lease = await world.acquire.execute(A, thread.id, RunMode.MESSAGE)

    with pytest.raises(ValueError, match="busy"):
        await world.release.execute(A, thread.id, lease.token, ThreadStatus.BUSY)
