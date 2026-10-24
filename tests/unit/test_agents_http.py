import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

import httpx
import pytest

from agintis.bootstrap.container import Container
from agintis.main import create_app
from agintis.modules.agents_catalog.api import (
    AgentRegistry,
    AgentRunner,
    AgentsCatalogServices,
    Clarification,
    RunAgent,
    RunCompleted,
    RunFailed,
    Scope,
    SourceRef,
    Sources,
    Step,
    Token,
    TokenUsage,
)
from agintis.modules.agents_catalog.http import get_services
from agintis.platform.http.errors import STATUS_BY_KIND
from agintis.platform.llm import ProviderErrorClassifier
from agintis.shared_kernel.errors import ErrorKind
from tests.fakes.agents import FakeRunner
from tests.support.settings import make_settings

TENANT = {"X-Tenant-Id": "acme"}
BODY = {"agent_id": "fake", "message": "hello"}


def services_for(*runners: AgentRunner, max_chars: int = 50) -> AgentsCatalogServices:
    registry = AgentRegistry(tuple(runners))
    return AgentsCatalogServices(
        registry=registry,
        run_agent=RunAgent(registry, ProviderErrorClassifier(), id_generator=lambda: "run-1"),
        max_message_chars=max_chars,
    )


@asynccontextmanager
async def _client(
    *runners: AgentRunner, max_chars: int = 50, **settings: Any
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(make_settings(**settings))
    app.dependency_overrides[get_services] = lambda: services_for(*runners, max_chars=max_chars)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
    ):
        yield c


def _frames(text: str) -> list[dict[str, Any]]:
    frames = []
    for block in text.strip().split("\n\n"):
        fields: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            key, _, value = line.partition(": ")
            fields[key] = json.loads(value) if key == "data" else value
        if fields:
            frames.append(fields)
    return frames


async def _stream(runner: AgentRunner) -> list[dict[str, Any]]:
    async with _client(runner) as c:
        response = await c.post("/v1/runs/stream", json=BODY, headers=TENANT)
    assert response.status_code == HTTPStatus.OK
    return _frames(response.text)


async def test_list_agents() -> None:
    async with _client(FakeRunner(agent_id="a", description="Agent A")) as c:
        response = await c.get("/v1/agents")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == [{"id": "a", "description": "Agent A"}]


async def test_stream_frames_in_order() -> None:
    runner = FakeRunner(
        [
            Step("retrieve", "started"),
            Token("Hi"),
            Sources((SourceRef("doc-1", title="T"),)),
            RunCompleted("stop", TokenUsage(1, 2, 3)),
        ]
    )
    async with _client(runner) as c:
        response = await c.post("/v1/runs/stream", json=BODY, headers=TENANT)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    frames = _frames(response.text)
    assert [(f["event"], f["id"]) for f in frames] == [
        ("run_started", "run-1:0"),
        ("step", "run-1:1"),
        ("token", "run-1:2"),
        ("sources", "run-1:3"),
        ("done", "run-1:4"),
    ]
    assert runner.inputs[0].tenant_id == "acme"


# --- wire shape of every event (pinned) -------------------------------------------------------


async def test_wire_shape_of_every_event() -> None:
    runner = FakeRunner(
        [
            Step("n", "finished"),
            Step("topic/retrieve", "started", "ncg"),
            Scope("low", 0.5),
            Scope("out"),
            Token("Hi"),
            Sources((SourceRef("doc", title="T", locator="p1"), SourceRef("other"))),
            Clarification("Which one?", ("a", "b"), "i-1"),
            Clarification("Free text?"),
            RunCompleted("length", TokenUsage(1, 2, 3, cached_input_tokens=1)),
        ]
    )

    data = {f["event"] + str(i): f["data"] for i, f in enumerate(await _stream(runner))}

    assert data == {
        "run_started0": {"run_id": "run-1", "agent_id": "fake"},
        "step1": {"node": "n", "status": "finished", "label": None},
        "step2": {"node": "topic/retrieve", "status": "started", "label": "ncg"},
        "scope3": {"level": "low", "score": 0.5},
        "scope4": {"level": "out", "score": None},
        "token5": {"text": "Hi"},
        "sources6": {
            "items": [
                {"source": "doc", "title": "T", "locator": "p1"},
                {"source": "other", "title": None, "locator": None},
            ]
        },
        "clarification7": {"question": "Which one?", "options": ["a", "b"], "interrupt_id": "i-1"},
        "clarification8": {"question": "Free text?", "options": [], "interrupt_id": ""},
        "done9": {
            "finish_reason": "length",
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "cached_input_tokens": 1,
            },
        },
    }


async def test_scope_score_is_rounded_on_the_wire_only() -> None:
    frames = await _stream(FakeRunner([Scope("in", 0.6103449), Scope("out", 0.00004)]))

    scopes = [f["data"] for f in frames if f["event"] == "scope"]
    assert scopes == [{"level": "in", "score": 0.6103}, {"level": "out", "score": 0.0}]


async def test_wire_shape_of_default_done_without_usage() -> None:
    frames = await _stream(FakeRunner([Token("a")]))

    assert frames[-1]["data"] == {"finish_reason": "stop", "usage": None}


async def test_mid_stream_failure_wire_shape() -> None:
    frames = await _stream(FakeRunner([Token("a")], error=RuntimeError("secret")))

    assert [f["event"] for f in frames] == ["run_started", "token", "error"]
    assert frames[-1]["data"] == {
        "code": "internal",
        "message": "Internal error while running the agent.",
        "retryable": False,
    }


async def test_retryable_error_wire_shape() -> None:
    timeout = type("APITimeoutError", (Exception,), {"__module__": "openai"})()

    frames = await _stream(FakeRunner(error=timeout))

    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["code"] == "llm_timeout"
    assert frames[-1]["data"]["retryable"] is True


async def test_runner_cannot_inject_lifecycle_events() -> None:
    runner = FakeRunner([Token("a"), RunFailed("x", "y"), Token("b")])

    frames = await _stream(runner)

    assert [f["event"] for f in frames] == ["run_started", "token", "token", "done"]


async def test_unknown_event_is_skipped_by_the_edge() -> None:
    from agintis.modules.agents_catalog.http import to_wire

    assert to_wire(object()) is None


# --- pre-stream errors -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "headers", "status", "code"),
    [
        ({**BODY, "agent_id": "nope"}, TENANT, HTTPStatus.NOT_FOUND, "agent_not_found"),
        (
            {"agent_id": "fake", "message": ""},
            TENANT,
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "invalid_request",
        ),
        ({"agent_id": "fake"}, TENANT, HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request"),
        (
            {**BODY, "agent_id": "bad id<script>"},
            TENANT,
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "invalid_request",
        ),
        ({**BODY, "extra": 1}, TENANT, HTTPStatus.UNPROCESSABLE_ENTITY, "invalid_request"),
        (
            {**BODY, "message": "x" * 51},
            TENANT,
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "invalid_request",
        ),
        (BODY, {}, HTTPStatus.BAD_REQUEST, "tenant_required"),
        (BODY, {"X-Tenant-Id": "a b"}, HTTPStatus.BAD_REQUEST, "invalid_tenant"),
    ],
)
async def test_pre_stream_errors_use_status_and_error_body(
    body: dict[str, Any], headers: dict[str, str], status: HTTPStatus, code: str
) -> None:
    async with _client(FakeRunner(), app_env="prod") as c:
        response = await c.post("/v1/runs/stream", json=body, headers=headers)

    assert response.status_code == status
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == code
    assert set(response.json()["error"]) == {"code", "message"}


async def test_llm_not_configured_is_503_before_the_stream() -> None:
    async with _client(FakeRunner(unavailable=True)) as c:
        response = await c.post("/v1/runs/stream", json=BODY, headers=TENANT)

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert response.json()["error"]["code"] == "llm_not_configured"


def test_every_error_kind_has_a_status() -> None:
    assert set(STATUS_BY_KIND) == set(ErrorKind)


# --- default tenant (fail-open: needs an explicit APP_ENV=local) -------------------------------


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"app_env": "local"}, HTTPStatus.OK),
        ({}, HTTPStatus.BAD_REQUEST),  # app_env left at its default: not explicit
        ({"app_env": "prod"}, HTTPStatus.BAD_REQUEST),
        ({"app_env": "test"}, HTTPStatus.BAD_REQUEST),
    ],
)
async def test_default_tenant_needs_explicit_local(
    settings: dict[str, Any], expected: HTTPStatus, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app(make_settings(http={"default_tenant": "dev"}, **settings))
    app.dependency_overrides[get_services] = lambda: services_for(FakeRunner())
    with caplog.at_level("WARNING"):
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
        ):
            response = await c.post("/v1/runs/stream", json=BODY)

    assert response.status_code == expected
    warned = any("HTTP_DEFAULT_TENANT" in r.getMessage() for r in caplog.records)
    assert warned is (expected == HTTPStatus.OK)


async def test_real_container_registers_direct_chat_and_503_without_llm() -> None:
    app = create_app(make_settings())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c,
    ):
        assert isinstance(app.state.container, Container)
        listed = await c.get("/v1/agents")
        run = await c.post(
            "/v1/runs/stream",
            json={"agent_id": "direct-chat", "message": "hi"},
            headers=TENANT,
        )

    assert [a["id"] for a in listed.json()] == ["direct-chat"]
    assert run.status_code == HTTPStatus.SERVICE_UNAVAILABLE
    assert run.json()["error"]["code"] == "llm_not_configured"
