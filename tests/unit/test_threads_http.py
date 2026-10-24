"""Threads over HTTP: CRUD, tenant isolation, and the interrupt -> resume run flow."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver

from agintis.agent_kit.runtime import LangGraphRunner
from agintis.bootstrap.adapters import (
    AgentsPendingQuestions,
    AgentsThreadTranscript,
    ConversationsThreadAccess,
    RegistryAgentDirectory,
)
from agintis.main import create_app
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    AgentRunner,
    AgentsCatalogServices,
    Clarification,
    RunAgent,
    RunCompleted,
    SourceRef,
    Token,
    TranscriptMessage,
)
from agintis.modules.agents_catalog.http import MAX_RESUME_DEPTH, get_services
from agintis.modules.conversations.api import (
    AcquireRunLease,
    CreateThread,
    DeleteThread,
    GetThread,
    ListThreads,
    ReleaseRunLease,
    RunMode,
    Thread,
    ThreadsServices,
    ThreadStatus,
)
from agintis.modules.conversations.http import get_thread_services
from agintis.platform.llm import ProviderErrorClassifier
from agintis.shared_kernel.ids import TenantId, ThreadId
from tests.fakes.agents import FakeRunner
from tests.fakes.clock import StepClock
from tests.fakes.conversations import FakeThreadStateStore, InMemoryThreadRepository
from tests.fakes.graphs import (
    Graph,
    ask_then_fail_graph,
    asking_graph,
    interrupt_and_error_graph,
    nested_in_node_graph,
    threaded,
    two_questions_graph,
)
from tests.support.settings import make_settings
from tests.unit.test_agents_http import _frames

Maker = Callable[[BaseCheckpointSaver[Any] | None], Graph]
ACME = {"X-Tenant-Id": "acme"}
OTHER = {"X-Tenant-Id": "other"}
MAX_CHARS = 20


@dataclass
class World:
    client: httpx.AsyncClient
    threads: ThreadsServices
    state: FakeThreadStateStore
    repo: InMemoryThreadRepository
    clock: StepClock


def _services(
    runners: tuple[AgentRunner, ...], *, with_threads: bool = True
) -> tuple[
    ThreadsServices,
    AgentsCatalogServices,
    FakeThreadStateStore,
    InMemoryThreadRepository,
    StepClock,
]:
    registry = AgentRegistry(runners)
    clock = StepClock()
    repo = InMemoryThreadRepository(clock)
    state = FakeThreadStateStore()
    threads = ThreadsServices(
        create=CreateThread(
            repo, RegistryAgentDirectory(registry), clock, threads_enabled=with_threads
        ),
        get=GetThread(repo, AgentsPendingQuestions(registry), AgentsThreadTranscript(registry)),
        list=ListThreads(repo, max_page_size=3),
        delete=DeleteThread(repo, state),
        acquire_lease=AcquireRunLease(repo, timedelta(minutes=5)),
        release_lease=ReleaseRunLease(repo),
    )
    catalog = AgentsCatalogServices(
        registry=registry,
        run_agent=RunAgent(
            registry,
            ProviderErrorClassifier(),
            threads=ConversationsThreadAccess(threads) if with_threads else None,
        ),
        max_message_chars=MAX_CHARS,
    )
    return threads, catalog, state, repo, clock


@asynccontextmanager
async def _world(*runners: AgentRunner, with_threads: bool = True) -> AsyncIterator[World]:
    threads, catalog, state, repo, clock = _services(runners, with_threads=with_threads)
    app = create_app(make_settings())
    app.dependency_overrides[get_thread_services] = lambda: threads
    app.dependency_overrides[get_services] = lambda: catalog
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
    ):
        yield World(c, threads, state, repo, clock)


def _asking_runner(make: Maker | None = None, agent_id: str = "ask") -> LangGraphRunner:
    return threaded(make or (lambda saver: asking_graph("Which one?", saver)), agent_id).runner


def _echo_runner(**kwargs: Any) -> FakeRunner:
    return FakeRunner(
        [Token("hello"), RunCompleted("stop")], agent_id="echo", supports_threads=True, **kwargs
    )


async def _create(w: World, agent_id: str = "echo", tenant: dict[str, str] = ACME) -> str:
    response = await w.client.post("/v1/threads", json={"agent_id": agent_id}, headers=tenant)
    assert response.status_code == HTTPStatus.CREATED
    return str(response.json()["id"])


async def _status(w: World, thread_id: str, tenant: dict[str, str] = ACME) -> str:
    response = await w.client.get(f"/v1/threads/{thread_id}", headers=tenant)
    return str(response.json()["status"])


async def _hold(w: World, thread_id: str) -> str:
    """Take the run lease like a running run would; returns its token."""
    lease = await w.threads.acquire_lease.execute(
        TenantId("acme"), ThreadId(thread_id), RunMode.MESSAGE
    )
    return lease.token


def _types(response: httpx.Response) -> list[str]:
    return [frame["event"] for frame in _frames(response.text)]


# --- CRUD -------------------------------------------------------------------------------------


async def test_create_returns_201_with_the_thread() -> None:
    async with _world(_echo_runner()) as w:
        response = await w.client.post(
            "/v1/threads", json={"agent_id": "echo", "title": " Plan "}, headers=ACME
        )

    assert response.status_code == HTTPStatus.CREATED
    body = response.json()
    assert set(body) == {"id", "agent_id", "title", "status", "created_at", "updated_at"}
    assert (body["agent_id"], body["title"], body["status"]) == ("echo", "Plan", "idle")


async def test_create_with_an_unknown_agent_is_404() -> None:
    async with _world(_echo_runner()) as w:
        response = await w.client.post("/v1/threads", json={"agent_id": "nope"}, headers=ACME)

    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["error"]["code"] == "agent_not_found"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"agent_id": ""},
        {"agent_id": "echo", "extra": 1},
        {"agent_id": "echo", "title": "x" * 900},
    ],
)
async def test_create_validates_the_body(body: dict[str, Any]) -> None:
    async with _world(_echo_runner()) as w:
        response = await w.client.post("/v1/threads", json=body, headers=ACME)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_over_long_title_after_trimming_is_422() -> None:
    async with _world(_echo_runner()) as w:
        response = await w.client.post(
            "/v1/threads", json={"agent_id": "echo", "title": "x" * 300}, headers=ACME
        )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_tenant_is_required() -> None:
    async with _world(_echo_runner()) as w:
        response = await w.client.get("/v1/threads")

    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["error"]["code"] == "tenant_required"


async def test_get_and_delete() -> None:
    async with _world(_echo_runner()) as w:
        thread_id = await _create(w)

        got = await w.client.get(f"/v1/threads/{thread_id}", headers=ACME)
        deleted = await w.client.delete(f"/v1/threads/{thread_id}", headers=ACME)
        again = await w.client.get(f"/v1/threads/{thread_id}", headers=ACME)

    assert got.status_code == HTTPStatus.OK
    assert got.json()["id"] == thread_id
    assert deleted.status_code == HTTPStatus.NO_CONTENT
    assert deleted.content == b""
    assert again.status_code == HTTPStatus.NOT_FOUND
    assert again.json()["error"] == {"code": "thread_not_found", "message": "Thread does not exist"}
    assert w.state.deleted == [thread_id]


async def test_other_tenants_see_404_everywhere_and_nothing_is_deleted() -> None:
    async with _world(_echo_runner()) as w:
        thread_id = await _create(w)

        get = await w.client.get(f"/v1/threads/{thread_id}", headers=OTHER)
        delete = await w.client.delete(f"/v1/threads/{thread_id}", headers=OTHER)
        run = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"message": "hi"}, headers=OTHER
        )
        listing = await w.client.get("/v1/threads", headers=OTHER)

    assert get.status_code == delete.status_code == run.status_code == HTTPStatus.NOT_FOUND
    assert listing.json() == {"items": [], "next_cursor": None}
    assert w.state.deleted == []
    assert (await w.threads.get.execute(TenantId("acme"), ThreadId(thread_id))).id == thread_id


async def test_messages_are_the_transcript_with_the_sources_of_each_answer() -> None:
    turns = (
        TranscriptMessage("user", "what?"),
        TranscriptMessage("assistant", "this [1]", (SourceRef("doc", "Doc", "p. 2"),)),
        TranscriptMessage("user", "and?"),
        TranscriptMessage("assistant", "that", (SourceRef("raw"),)),
    )
    async with _world(_echo_runner(messages=turns)) as w:
        thread_id = await _create(w)

        response = await w.client.get(f"/v1/threads/{thread_id}/messages", headers=ACME)

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "items": [
            {"role": "user", "text": "what?", "sources": []},
            {
                "role": "assistant",
                "text": "this [1]",
                "sources": [{"source": "doc", "title": "Doc", "locator": "p. 2"}],
            },
            {"role": "user", "text": "and?", "sources": []},
            {
                "role": "assistant",
                "text": "that",
                "sources": [{"source": "raw", "title": None, "locator": None}],
            },
        ]
    }


async def test_messages_of_a_new_thread_are_empty() -> None:
    async with _world(_echo_runner()) as w:
        thread_id = await _create(w)

        response = await w.client.get(f"/v1/threads/{thread_id}/messages", headers=ACME)

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"items": []}


async def test_messages_of_another_tenant_or_a_missing_thread_are_404() -> None:
    async with _world(_echo_runner(messages=(TranscriptMessage("user", "secret"),))) as w:
        thread_id = await _create(w)

        other = await w.client.get(f"/v1/threads/{thread_id}/messages", headers=OTHER)
        missing = await w.client.get("/v1/threads/nope/messages", headers=ACME)

    assert other.status_code == missing.status_code == HTTPStatus.NOT_FOUND
    assert other.json()["error"]["code"] == "thread_not_found"
    assert "secret" not in other.text


async def test_messages_do_not_shadow_the_run_stream_route() -> None:
    async with _world(_echo_runner()) as w:
        thread_id = await _create(w)

        run = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"message": "hi"}, headers=ACME
        )

    assert run.status_code == HTTPStatus.OK
    assert "done" in _types(run)


async def test_malformed_thread_id_is_422() -> None:
    async with _world(_echo_runner()) as w:
        response = await w.client.get("/v1/threads/bad%20id!", headers=ACME)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_list_paginates_with_an_opaque_cursor_and_filters() -> None:
    async with _world(_echo_runner(), FakeRunner(agent_id="fake", supports_threads=True)) as w:
        for _ in range(4):
            await _create(w)
        other_agent = await _create(w, "fake")

        first = (await w.client.get("/v1/threads?limit=3", headers=ACME)).json()
        second = (
            await w.client.get(f"/v1/threads?limit=3&cursor={first['next_cursor']}", headers=ACME)
        ).json()
        by_agent = (await w.client.get("/v1/threads?agent_id=fake", headers=ACME)).json()
        idle = await w.client.get("/v1/threads?status=idle&limit=100", headers=ACME)
        busy = (await w.client.get("/v1/threads?status=busy", headers=ACME)).json()

    assert len(first["items"]) == 3
    assert first["next_cursor"]
    assert len(second["items"]) == 2
    assert second["next_cursor"] is None
    ids = [t["id"] for t in first["items"] + second["items"]]
    assert len(set(ids)) == 5
    assert [t["id"] for t in by_agent["items"]] == [other_agent]
    assert idle.status_code == HTTPStatus.OK
    assert len(idle.json()["items"]) == 3  # capped by the server's max page size (3 here)
    assert idle.json()["next_cursor"]
    assert busy["items"] == []


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "status=weird", "cursor=" + "a" * 600])
async def test_list_validates_its_query(query: str) -> None:
    async with _world(_echo_runner()) as w:
        response = await w.client.get(f"/v1/threads?{query}", headers=ACME)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


async def test_list_with_a_garbage_cursor_is_422_invalid_cursor() -> None:
    async with _world(_echo_runner()) as w:
        response = await w.client.get("/v1/threads?cursor=garbage", headers=ACME)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["error"]["code"] == "invalid_cursor"


async def test_a_running_thread_cannot_be_deleted() -> None:
    async with _world(_echo_runner()) as w:
        thread_id = await _create(w)
        await w.threads.acquire_lease.execute(
            TenantId("acme"), ThreadId(thread_id), RunMode.MESSAGE
        )

        response = await w.client.delete(f"/v1/threads/{thread_id}", headers=ACME)

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.json()["error"]["code"] == "thread_busy"
    assert w.state.deleted == []


# --- thread runs ------------------------------------------------------------------------------


async def test_message_run_streams_the_usual_frames_and_the_thread_ends_idle() -> None:
    runner = _echo_runner()
    async with _world(runner) as w:
        thread_id = await _create(w)

        response = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"message": "hi"}, headers=ACME
        )
        status = await _status(w, thread_id)

    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _types(response) == ["run_started", "token", "done"]
    assert status == "idle"
    assert runner.inputs[0].thread_id == thread_id
    assert runner.inputs[0].message == "hi"


async def test_interrupt_then_resume_flow() -> None:
    async with _world(_asking_runner()) as w:
        thread_id = await _create(w, "ask")
        url = f"/v1/threads/{thread_id}/runs/stream"

        first = await w.client.post(url, json={"message": "start"}, headers=ACME)
        detail = (await w.client.get(f"/v1/threads/{thread_id}", headers=ACME)).json()
        message_while_interrupted = await w.client.post(url, json={"message": "x"}, headers=ACME)
        resumed = await w.client.post(url, json={"resume": "blue"}, headers=ACME)
        after_resume = (await w.client.get(f"/v1/threads/{thread_id}", headers=ACME)).json()
        resume_when_idle = await w.client.post(url, json={"resume": "again"}, headers=ACME)

    frames = _frames(first.text)
    assert [f["event"] for f in frames] == ["run_started", "step", "clarification", "done"]
    asked = frames[2]["data"]
    assert (asked["question"], asked["options"]) == ("Which one?", [])
    assert len(asked["interrupt_id"]) == 32
    assert frames[3]["data"]["finish_reason"] == "interrupted"
    # A client that lost the stream can recover the question and its id.
    assert detail["status"] == "interrupted"
    assert detail["pending_clarifications"] == [
        {"interrupt_id": asked["interrupt_id"], "question": "Which one?", "options": []}
    ]
    assert message_while_interrupted.status_code == HTTPStatus.CONFLICT
    assert message_while_interrupted.json()["error"]["code"] == "thread_interrupted"
    assert _frames(resumed.text)[-1]["data"]["finish_reason"] == "stop"
    assert "clarification" not in _types(resumed)
    assert (after_resume["status"], after_resume["pending_clarifications"]) == ("idle", [])
    assert resume_when_idle.status_code == HTTPStatus.CONFLICT
    assert resume_when_idle.json()["error"]["code"] == "thread_not_interrupted"


async def test_two_pending_questions_need_resumes_keyed_by_interrupt_id() -> None:
    async with _world(_asking_runner(two_questions_graph, "ask2")) as w:
        thread_id = await _create(w, "ask2")
        url = f"/v1/threads/{thread_id}/runs/stream"

        first = await w.client.post(url, json={"message": "start"}, headers=ACME)
        ids = {
            f["data"]["question"]: f["data"]["interrupt_id"]
            for f in _frames(first.text)
            if f["event"] == "clarification"
        }
        listed = (await w.client.get(f"/v1/threads/{thread_id}", headers=ACME)).json()
        bare = await w.client.post(url, json={"resume": "x"}, headers=ACME)
        partial = await w.client.post(url, json={"resumes": {ids["Question a"]: "x"}}, headers=ACME)
        unknown = await w.client.post(
            url, json={"resumes": {"nope": "x", "other": "y"}}, headers=ACME
        )
        status_after_rejections = await _status(w, thread_id)
        ok = await w.client.post(
            url,
            json={"resumes": {ids["Question a"]: "one", ids["Question b"]: {"n": 2}}},
            headers=ACME,
        )
        final = await _status(w, thread_id)

    assert set(ids) == {"Question a", "Question b"}
    assert {c["interrupt_id"] for c in listed["pending_clarifications"]} == set(ids.values())
    assert bare.status_code == HTTPStatus.CONFLICT
    assert bare.json()["error"]["code"] == "multiple_interrupts_pending"
    assert "resumes" in bare.json()["error"]["message"]
    for rejected in (partial, unknown):
        assert rejected.status_code == HTTPStatus.CONFLICT
        assert rejected.json()["error"]["code"] == "interrupt_mismatch"
    assert status_after_rejections == "interrupted"  # rejections never consume the questions
    assert _frames(ok.text)[-1]["data"]["finish_reason"] == "stop"
    assert final == "idle"


async def test_a_message_over_a_stale_idle_status_is_refused_and_corrected() -> None:
    async with _world(_asking_runner()) as w:
        thread_id = await _create(w, "ask")
        url = f"/v1/threads/{thread_id}/runs/stream"
        await w.client.post(url, json={"message": "start"}, headers=ACME)
        # The release of the `interrupted` status got lost: the row says idle, the checkpoint asks.
        w.repo.items[ThreadId(thread_id)] = replace(
            w.repo.items[ThreadId(thread_id)], status=ThreadStatus.IDLE
        )

        refused = await w.client.post(url, json={"message": "x"}, headers=ACME)
        status = await _status(w, thread_id)

    assert refused.status_code == HTTPStatus.CONFLICT
    assert refused.json()["error"]["code"] == "thread_interrupted"
    assert status == "interrupted"


async def test_a_resume_over_a_stale_interrupted_status_is_refused_and_the_status_corrected() -> (
    None
):
    async with _world(_echo_runner()) as w:
        thread_id = await _create(w)
        token = await _hold(w, thread_id)
        await w.threads.release_lease.execute(
            TenantId("acme"), ThreadId(thread_id), token, ThreadStatus.INTERRUPTED
        )

        refused = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"resume": "x"}, headers=ACME
        )
        status = await _status(w, thread_id)

    assert refused.status_code == HTTPStatus.CONFLICT
    assert refused.json()["error"]["code"] == "thread_not_interrupted"
    assert status == "idle"


async def test_a_resume_can_take_over_after_a_resume_run_crashed() -> None:
    async with _world(_asking_runner()) as w:
        thread_id = await _create(w, "ask")
        url = f"/v1/threads/{thread_id}/runs/stream"
        await w.client.post(url, json={"message": "start"}, headers=ACME)
        # A resume run took the lease and died with its process: busy, never released.
        await w.threads.acquire_lease.execute(TenantId("acme"), ThreadId(thread_id), RunMode.RESUME)
        busy = await w.client.post(url, json={"resume": "blue"}, headers=ACME)
        w.clock.advance(minutes=6)  # the lease expires

        resumed = await w.client.post(url, json={"resume": "blue"}, headers=ACME)
        status = await _status(w, thread_id)

    assert busy.json()["error"]["code"] == "thread_busy"
    assert _frames(resumed.text)[-1]["data"]["finish_reason"] == "stop"
    assert status == "idle"


async def test_nested_subgraph_questions_are_resumed_with_resumes_over_http() -> None:
    nested = _asking_runner(nested_in_node_graph, "nested")
    async with _world(nested) as w:
        thread_id = await _create(w, "nested")
        url = f"/v1/threads/{thread_id}/runs/stream"

        first = await w.client.post(url, json={"message": "start"}, headers=ACME)
        ids = {
            f["data"]["question"]: f["data"]["interrupt_id"]
            for f in _frames(first.text)
            if f["event"] == "clarification"
        }
        resumed = await w.client.post(
            url, json={"resumes": {i: q for q, i in ids.items()}}, headers=ACME
        )
        status = await _status(w, thread_id)

    assert set(ids) == {"deep a", "deep b"}
    assert _frames(resumed.text)[-1]["data"]["finish_reason"] == "stop"
    assert status == "idle"


async def test_a_crashed_run_that_left_the_question_pending_is_not_stranded() -> None:
    """Resume consumed, then a later node fails: the checkpoint has no question -> idle."""
    failing = _asking_runner(lambda saver: ask_then_fail_graph(saver), "failing")
    async with _world(failing) as w:
        thread_id = await _create(w, "failing")
        url = f"/v1/threads/{thread_id}/runs/stream"
        await w.client.post(url, json={"message": "start"}, headers=ACME)

        resumed = await w.client.post(url, json={"resume": "blue"}, headers=ACME)
        status = await _status(w, thread_id)

    assert _types(resumed)[-1] == "error"
    assert status == "idle"  # the answer is consumed: a message may follow, a resume may not


async def test_interrupt_and_error_in_one_superstep_follow_the_checkpoint() -> None:
    toy = threaded(interrupt_and_error_graph, "mixed")
    async with _world(toy.runner) as w:
        thread_id = await _create(w, "mixed")

        response = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"message": "go"}, headers=ACME
        )
        status = await _status(w, thread_id)
        pending = await toy.runner.pending_interrupts(ThreadId(thread_id))

    assert _types(response)[-1] in {"error", "done"}
    assert status == ("interrupted" if pending else "idle")  # never out of step with the state


async def test_a_busy_thread_is_409_before_the_stream_opens() -> None:
    async with _world(_echo_runner()) as w:
        thread_id = await _create(w)
        await w.threads.acquire_lease.execute(
            TenantId("acme"), ThreadId(thread_id), RunMode.MESSAGE
        )

        response = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"message": "hi"}, headers=ACME
        )

    assert response.status_code == HTTPStatus.CONFLICT
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "thread_busy"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": "hi", "resume": "x"},
        {"message": "hi", "resumes": {"i": "x"}},
        {"resume": "x", "resumes": {"i": "x"}},
        {"resume": None},
        {"resumes": {}},
        {"resumes": {"i": None}},
        {"message": ""},
        {"message": "hi", "x": 1},
    ],
)
async def test_a_thread_run_needs_exactly_one_of_message_and_resume(body: dict[str, Any]) -> None:
    async with _world(_echo_runner()) as w:
        thread_id = await _create(w)

        response = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json=body, headers=ACME
        )
        status = await _status(w, thread_id)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert status == "idle"  # nothing was acquired


async def test_an_over_long_message_is_rejected_without_taking_the_lease() -> None:
    async with _world(_echo_runner()) as w:
        thread_id = await _create(w)

        response = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream",
            json={"message": "x" * (MAX_CHARS + 1)},
            headers=ACME,
        )
        status = await _status(w, thread_id)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert status == "idle"


async def test_resume_accepts_any_json_value() -> None:
    runner = _echo_runner(pending=[Clarification("Which?", (), "i1")])
    async with _world(runner) as w:
        thread_id = await _create(w)
        token = await _hold(w, thread_id)
        await w.threads.release_lease.execute(
            TenantId("acme"), ThreadId(thread_id), token, ThreadStatus.INTERRUPTED
        )

        response = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream",
            json={"resume": {"pick": [1, 2]}},
            headers=ACME,
        )

    assert response.status_code == HTTPStatus.OK
    assert runner.inputs[0].resume == {"i1": {"pick": [1, 2]}}


async def test_an_oversized_or_too_deep_answer_is_422_and_nothing_is_acquired() -> None:
    deep: Any = "x"
    for _ in range(MAX_RESUME_DEPTH + 1):
        deep = {"k": deep}
    async with _world(_echo_runner(pending=[Clarification("Q", (), "i1")])) as w:
        thread_id = await _create(w)
        url = f"/v1/threads/{thread_id}/runs/stream"

        big = await w.client.post(url, json={"resume": "x" * (MAX_CHARS + 1)}, headers=ACME)
        big_map = await w.client.post(
            url, json={"resumes": {"i1": "x" * (MAX_CHARS + 1)}}, headers=ACME
        )
        nested = await w.client.post(url, json={"resume": deep}, headers=ACME)
        status = await _status(w, thread_id)

    for response in (big, big_map, nested):
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert status == "idle"


async def test_create_for_an_agent_without_thread_support_is_422() -> None:
    async with _world(FakeRunner(agent_id="stateless")) as w:
        response = await w.client.post("/v1/threads", json={"agent_id": "stateless"}, headers=ACME)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["error"]["code"] == "agent_threads_unsupported"


async def test_create_is_503_when_threads_are_disabled() -> None:
    async with _world(_echo_runner(), with_threads=False) as w:
        response = await w.client.post("/v1/threads", json={"agent_id": "echo"}, headers=ACME)

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["error"]["code"] == "threads_unavailable"


def _seed(w: World, agent_id: str) -> str:
    """A thread that already exists (created when the agent could still hold a conversation)."""
    thread = Thread.create(ThreadId("seeded"), TenantId("acme"), agent_id, None, w.clock.now())
    w.repo.items[thread.id] = thread
    return thread.id


async def test_agent_that_lost_thread_support_is_422_and_the_thread_stays_idle() -> None:
    async with _world(FakeRunner(agent_id="stateless")) as w:
        thread_id = _seed(w, "stateless")

        response = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"message": "hi"}, headers=ACME
        )
        status = await _status(w, thread_id)

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["error"]["code"] == "agent_threads_unsupported"
    assert status == "idle"


async def test_thread_runs_are_503_when_threads_are_disabled() -> None:
    async with _world(_echo_runner(), with_threads=False) as w:
        thread_id = _seed(w, "echo")

        response = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"message": "hi"}, headers=ACME
        )

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["error"]["code"] == "threads_unavailable"


async def test_a_failing_run_leaves_the_thread_usable() -> None:
    failing = FakeRunner(
        [Token("a")], agent_id="echo", supports_threads=True, error=RuntimeError("x")
    )
    async with _world(failing) as w:
        thread_id = await _create(w)

        response = await w.client.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"message": "hi"}, headers=ACME
        )
        status = await _status(w, thread_id)

    assert _types(response)[-1] == "error"
    assert status == "idle"


async def test_the_stateless_endpoint_keeps_working_next_to_threads() -> None:
    async with _world(_echo_runner(), FakeRunner([Token("x")], agent_id="plain")) as w:
        response = await w.client.post(
            "/v1/runs/stream", json={"agent_id": "plain", "message": "hi"}, headers=ACME
        )

    assert response.status_code == HTTPStatus.OK
    assert _types(response) == ["run_started", "token", "done"]
