"""HTTP -> Postgres threads + real checkpointer: interrupt, resume, delete end to end."""

from collections.abc import AsyncIterator
from datetime import timedelta
from http import HTTPStatus

import httpx
import pytest
from langchain_core.runnables import RunnableConfig
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.engine import URL

from agintis.bootstrap.adapters import (
    AgentsPendingQuestions,
    AgentsThreadTranscript,
    CheckpointerStateStore,
    ConversationsThreadAccess,
    RegistryAgentDirectory,
)
from agintis.main import create_app
from agintis.modules.agents_catalog.api import AgentRegistry, AgentsCatalogServices, RunAgent
from agintis.modules.agents_catalog.http import get_services
from agintis.modules.conversations.api import (
    AcquireRunLease,
    CreateThread,
    DeleteThread,
    GetThread,
    ListThreads,
    ReleaseRunLease,
    ThreadsServices,
)
from agintis.modules.conversations.http import get_thread_services
from agintis.modules.conversations.infrastructure.postgres import PgThreadRepository
from agintis.platform.checkpointer import Checkpointer
from agintis.platform.clock import SystemClock
from agintis.platform.db import create_session_factory
from agintis.platform.llm import ProviderErrorClassifier
from tests.fakes.graphs import asking_graph, cited_graph, threaded
from tests.support.postgres import libpq_dsn, postgres_engines
from tests.support.settings import make_settings
from tests.unit.test_agents_http import _frames

pytestmark = pytest.mark.integration

ACME = {"X-Tenant-Id": "acme"}
OTHER = {"X-Tenant-Id": "other"}


@pytest.fixture
async def client(pg_test_url: URL) -> AsyncIterator[tuple[httpx.AsyncClient, Checkpointer]]:
    checkpointer = Checkpointer(
        SecretStr(libpq_dsn(pg_test_url)),
        min_size=1,
        max_size=3,
        acquire_timeout_seconds=5,
        connect_timeout_seconds=3,
    )
    await checkpointer.open(wait=True)
    await checkpointer.setup()
    toy = threaded(lambda s: asking_graph("Which one?", s), "ask", checkpointer.saver)
    cite = threaded(cited_graph, "cite", checkpointer.saver)
    registry = AgentRegistry((toy.runner, cite.runner))
    async with postgres_engines(pg_test_url) as (engine, _lock_engine):
        factory = create_session_factory(engine)
        async with factory.begin() as session:
            await session.execute(text("TRUNCATE threads"))
        repo = PgThreadRepository(factory)
        threads = ThreadsServices(
            create=CreateThread(repo, RegistryAgentDirectory(registry), SystemClock()),
            get=GetThread(repo, AgentsPendingQuestions(registry), AgentsThreadTranscript(registry)),
            list=ListThreads(repo),
            delete=DeleteThread(repo, CheckpointerStateStore(checkpointer)),
            acquire_lease=AcquireRunLease(repo, timedelta(minutes=5)),
            release_lease=ReleaseRunLease(repo),
        )
        catalog = AgentsCatalogServices(
            registry,
            RunAgent(
                registry, ProviderErrorClassifier(), threads=ConversationsThreadAccess(threads)
            ),
            1000,
        )
        app = create_app(make_settings())
        app.dependency_overrides[get_thread_services] = lambda: threads
        app.dependency_overrides[get_services] = lambda: catalog
        try:
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as http_client,
            ):
                yield http_client, checkpointer
        finally:
            await checkpointer.aclose()


async def test_interrupt_resume_and_delete_over_postgres(
    client: tuple[httpx.AsyncClient, Checkpointer],
) -> None:
    http, checkpointer = client
    created = await http.post("/v1/threads", json={"agent_id": "ask", "title": "T"}, headers=ACME)
    assert created.status_code == HTTPStatus.CREATED
    thread_id = created.json()["id"]
    url = f"/v1/threads/{thread_id}/runs/stream"

    first = await http.post(url, json={"message": "start"}, headers=ACME)
    detail = (await http.get(f"/v1/threads/{thread_id}", headers=ACME)).json()
    interrupted = detail["status"]
    foreign = await http.post(url, json={"resume": "x"}, headers=OTHER)
    resumed = await http.post(url, json={"resume": "blue"}, headers=ACME)
    idle = (await http.get(f"/v1/threads/{thread_id}", headers=ACME)).json()["status"]

    assert [f["event"] for f in _frames(first.text)][-2:] == ["clarification", "done"]
    assert _frames(first.text)[-1]["data"]["finish_reason"] == "interrupted"
    assert interrupted == "interrupted"
    asked = next(f for f in _frames(first.text) if f["event"] == "clarification")["data"]
    assert detail["pending_clarifications"] == [
        {"interrupt_id": asked["interrupt_id"], "question": "Which one?", "options": []}
    ]
    assert foreign.status_code == HTTPStatus.NOT_FOUND
    assert _frames(resumed.text)[-1]["data"]["finish_reason"] == "stop"
    assert idle == "idle"

    messages = await http.get(f"/v1/threads/{thread_id}/messages", headers=ACME)
    assert messages.json() == {
        "items": [
            {"role": "user", "text": "start", "sources": []},
            {"role": "assistant", "text": "you chose blue", "sources": []},
        ]
    }  # read back from the persisted checkpoint
    foreign_messages = await http.get(f"/v1/threads/{thread_id}/messages", headers=OTHER)
    assert foreign_messages.status_code == HTTPStatus.NOT_FOUND

    key: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    assert await checkpointer.saver.aget_tuple(key) is not None

    deleted = await http.delete(f"/v1/threads/{thread_id}", headers=ACME)

    assert deleted.status_code == HTTPStatus.NO_CONTENT
    assert (await http.get(f"/v1/threads/{thread_id}", headers=ACME)).status_code == 404
    assert await checkpointer.saver.aget_tuple(key) is None


async def test_cited_sources_survive_the_postgres_checkpoint(
    client: tuple[httpx.AsyncClient, Checkpointer],
) -> None:
    http, _checkpointer = client
    created = await http.post("/v1/threads", json={"agent_id": "cite"}, headers=ACME)
    thread_id = created.json()["id"]
    for message in ("first", "second"):
        run = await http.post(
            f"/v1/threads/{thread_id}/runs/stream", json={"message": message}, headers=ACME
        )
        assert _frames(run.text)[-1]["data"]["finish_reason"] == "stop"

    messages = await http.get(f"/v1/threads/{thread_id}/messages", headers=ACME)

    assert messages.json() == {
        "items": [
            {"role": "user", "text": "first", "sources": []},
            {"role": "assistant", "text": "answer 1", "sources": []},
            {"role": "user", "text": "second", "sources": []},
            {
                "role": "assistant",
                "text": "answer 2",
                "sources": [{"source": "doc-2", "title": "Doc", "locator": None}],
            },
        ]
    }
