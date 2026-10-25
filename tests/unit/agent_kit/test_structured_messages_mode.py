"""Structured-output calls under LangGraph `messages` mode (the mode the runner always uses).

A real `ChatOpenAI` over a mock transport: if the call were streamed, langchain-openai would go
through the SDK's `beta.chat.completions.stream` and dump a `ContentDoneEvent` whose `parsed` is
typed `None`, raising a pydantic `UserWarning`. Warnings are errors here.
"""

import json
from typing import Any

import httpx2
import pytest
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from pydantic import BaseModel

from agintis.agent_kit.nodes.structured import bind_structured, invoke_structured
from agintis.agent_kit.runtime import LangGraphRunner
from agintis.modules.agents_catalog.api import RunCompleted, RunInput, TokenUsage
from agintis.shared_kernel.ids import RunId, TenantId
from tests.fakes.graphs import ChatState

pytestmark = pytest.mark.filterwarnings("error")

_CONTENT = json.dumps({"topic": "billing"})
_USAGE = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}


class _Hint(BaseModel):
    topic: str


class _Calls:
    def __init__(self) -> None:
        self.stream_flags: list[bool] = []


def _sse() -> bytes:
    base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {**base, "choices": [{"index": 0, "delta": {"content": _CONTENT}}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


def _chat_model(calls: _Calls) -> ChatOpenAI:
    def handler(request: httpx2.Request) -> httpx2.Response:
        stream = bool(json.loads(request.content).get("stream"))
        calls.stream_flags.append(stream)
        if stream:
            return httpx2.Response(
                200, content=_sse(), headers={"content-type": "text/event-stream"}
            )
        message = {"role": "assistant", "content": _CONTENT}
        body = {
            "id": "c1",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": _USAGE,
        }
        return httpx2.Response(200, json=body)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return ChatOpenAI(
        model="m",
        api_key="x",
        base_url="http://llm.test/v1",
        http_async_client=client,
    )


def _runner(model: ChatOpenAI, parsed: list[BaseModel | None]) -> LangGraphRunner:
    structured = bind_structured(model, _Hint)

    async def node(state: ChatState) -> dict[str, Any]:
        messages: list[BaseMessage] = state["messages"]
        parsed.append((await invoke_structured(structured, messages)).parsed)
        return {}

    graph = StateGraph(ChatState).add_node("hint", node).add_edge(START, "hint").compile()
    return LangGraphRunner("toy", "A toy agent", graph)


async def test_structured_call_in_messages_mode_does_not_warn_and_counts_usage() -> None:
    calls = _Calls()
    model = _chat_model(calls)
    parsed: list[BaseModel | None] = []
    run_input = RunInput(TenantId("acme"), "hi", RunId("0123456789abcdef0123456789abcdef"))

    events = [e async for e in _runner(model, parsed).astream(run_input)]

    assert parsed == [_Hint(topic="billing")]
    assert calls.stream_flags == [False]
    assert events[-1] == RunCompleted("stop", TokenUsage(7, 5, 12))


async def test_bind_structured_does_not_mutate_the_shared_model() -> None:
    model = _chat_model(_Calls())

    bind_structured(model, _Hint)

    assert model.disable_streaming is False
