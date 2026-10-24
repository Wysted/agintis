"""LangGraphRunner with threads: thread_id in the config, interrupts as questions, resumes."""

import logging
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from agintis.agent_kit.runtime import LangGraphRunner
from agintis.agent_kit.runtime.langgraph_runner import (
    FALLBACK_QUESTION,
    clarification_from_interrupt,
)
from agintis.modules.agents_catalog.api import (
    AgentUnavailableError,
    Clarification,
    RunCompleted,
    RunInput,
    SourceRef,
    Step,
    TranscriptMessage,
)
from agintis.shared_kernel.ids import RunId, TenantId, ThreadId
from tests.fakes.graphs import (
    Graph,
    StreamingModel,
    asking_graph,
    cited_graph,
    config_probe_graph,
    llm_graph,
    nested_in_node_graph,
    state_probe_graph,
    subgraph_as_node_graph,
    threaded,
    two_questions_graph,
)

THREAD = ThreadId("thread-1")
Maker = Callable[[BaseCheckpointSaver[Any] | None], Graph]


def _input(
    message: str | None = "hi",
    *,
    run: str = "0123456789abcdef0123456789abcdef",
    thread: ThreadId | None = THREAD,
    resume: dict[str, object] | None = None,
) -> RunInput:
    return RunInput(TenantId("acme"), message, RunId(run), thread, resume)


async def _events(runner: LangGraphRunner, run_input: RunInput) -> list[Any]:
    return [e async for e in runner.astream(run_input)]


def _asking(payload: object = "Which one?") -> Maker:
    return lambda saver: asking_graph(payload, saver)


async def test_thread_run_puts_only_the_thread_id_in_configurable() -> None:
    seen: list[RunnableConfig] = []
    runtimes: list[Any] = []
    toy = threaded(lambda saver: config_probe_graph(seen, runtimes, checkpointer=saver))

    await _events(toy.runner, _input())

    assert seen[0]["configurable"]["thread_id"] == THREAD
    assert runtimes[0].context.tenant_id == "acme"  # the tenant travels in context=
    # ... so it is never persisted with the checkpoints.
    key: RunnableConfig = {"configurable": {"thread_id": THREAD}}
    persisted = [str(item.metadata) async for item in toy.saver.alist(key)]
    assert persisted
    assert not any("acme" in entry for entry in persisted)


async def test_stateless_run_has_no_thread_id() -> None:
    seen: list[RunnableConfig] = []
    runner = LangGraphRunner("toy", "d", config_probe_graph(seen, []))

    await _events(runner, _input(thread=None))

    assert "thread_id" not in seen[0].get("configurable", {})


async def test_a_graph_compiled_with_a_checkpointer_cannot_be_the_plain_graph() -> None:
    saved = state_probe_graph([], checkpointer=InMemorySaver())

    with pytest.raises(ValueError, match="thread_graph"):
        LangGraphRunner("toy", "d", saved)


async def test_supports_threads_follows_the_thread_graph() -> None:
    plain = state_probe_graph([])

    assert LangGraphRunner("toy", "d", plain).supports_threads is False
    saved = state_probe_graph([], checkpointer=InMemorySaver())
    assert LangGraphRunner("toy", "d", plain, thread_graph=saved).supports_threads is True


async def test_thread_run_on_a_runner_without_checkpointer_is_unavailable() -> None:
    runner = LangGraphRunner("toy", "d", state_probe_graph([]))

    with pytest.raises(AgentUnavailableError) as info:
        await _events(runner, _input())

    assert info.value.code == "threads_unavailable"
    assert await runner.pending_interrupts(THREAD) == ()


async def test_stateless_and_thread_runs_use_their_own_graph() -> None:
    plain_seen: list[list[BaseMessage]] = []
    saved_seen: list[list[BaseMessage]] = []
    runner = LangGraphRunner(
        "toy",
        "d",
        state_probe_graph(plain_seen),
        thread_graph=state_probe_graph(saved_seen, checkpointer=InMemorySaver()),
    )

    await _events(runner, _input(thread=None))
    await _events(runner, _input())

    assert len(plain_seen) == 1
    assert len(saved_seen) == 1


async def test_the_thread_remembers_earlier_turns_and_threads_are_isolated() -> None:
    seen: list[list[BaseMessage]] = []
    toy = threaded(lambda saver: state_probe_graph(seen, checkpointer=saver))

    await _events(toy.runner, _input("first", run="a" * 32))
    await _events(toy.runner, _input("second", run="b" * 32))
    await _events(toy.runner, _input("other thread", run="c" * 32, thread=ThreadId("thread-2")))

    assert [[m.content for m in turn] for turn in seen] == [
        ["first"],
        ["first", "second"],
        ["other thread"],
    ]


async def test_llm_agent_gets_its_history_on_the_second_turn() -> None:
    model = StreamingModel(pieces=["ok"])
    toy = threaded(lambda saver: llm_graph(model, checkpointer=saver))

    await _events(toy.runner, _input("one", run="a" * 32))
    await _events(toy.runner, _input("two", run="b" * 32))

    state = await toy.graph.aget_state({"configurable": {"thread_id": THREAD}})
    assert [m.content for m in state.values["messages"]] == ["one", "ok", "two", "ok"]


async def test_interrupt_emits_the_question_with_its_interrupt_id() -> None:
    toy = threaded(_asking(Clarification("Which one?", ("a", "b"))))

    events = await _events(toy.runner, _input())

    questions = [e for e in events if isinstance(e, Clarification)]
    assert [(q.question, q.options) for q in questions] == [("Which one?", ("a", "b"))]
    assert len(questions[0].interrupt_id) == 32  # LangGraph's xxh3 hex id
    assert events[-1] == RunCompleted("interrupted")
    assert [e for e in events if isinstance(e, Step)] == [Step("ask", "started")]


async def test_pending_interrupts_read_the_checkpoint() -> None:
    toy = threaded(_asking("Which one?"))
    assert await toy.runner.pending_interrupts(THREAD) == ()  # nothing yet

    events = await _events(toy.runner, _input())
    asked = next(e for e in events if isinstance(e, Clarification))

    assert await toy.runner.pending_interrupts(THREAD) == (asked,)
    await _events(toy.runner, _input(None, run="b" * 32, resume={asked.interrupt_id: "x"}))
    assert await toy.runner.pending_interrupts(THREAD) == ()


async def test_resume_continues_the_graph_with_the_answer() -> None:
    toy = threaded(_asking("Which one?"))

    first = await _events(toy.runner, _input(run="a" * 32))
    asked = next(e for e in first if isinstance(e, Clarification))
    resumed = await _events(
        toy.runner, _input(None, run="b" * 32, resume={asked.interrupt_id: "blue"})
    )

    assert not [e for e in resumed if isinstance(e, Clarification)]
    assert resumed[-1] == RunCompleted("stop")
    state = await toy.graph.aget_state({"configurable": {"thread_id": THREAD}})
    assert [m.content for m in state.values["messages"]] == ["hi", "you chose blue"]
    assert isinstance(state.values["messages"][-1], AIMessage)


async def test_a_dict_answer_that_looks_like_an_id_map_is_still_just_the_value() -> None:
    """The explicit {interrupt_id: value} wrapper removes the 32-hex dict ambiguity."""
    toy = threaded(_asking("Which one?"))
    first = await _events(toy.runner, _input(run="a" * 32))
    asked = next(e for e in first if isinstance(e, Clarification))
    lookalike = {"a" * 32: "x"}

    await _events(toy.runner, _input(None, run="b" * 32, resume={asked.interrupt_id: lookalike}))

    state = await toy.graph.aget_state({"configurable": {"thread_id": THREAD}})
    assert state.values["messages"][-1].content == f"you chose {lookalike}"


async def test_two_pending_interrupts_are_both_asked_and_answered_by_id() -> None:
    toy = threaded(two_questions_graph)

    first = await _events(toy.runner, _input(run="a" * 32))
    questions = {q.question: q for q in first if isinstance(q, Clarification)}
    assert set(questions) == {"Question a", "Question b"}
    assert questions["Question a"].interrupt_id != questions["Question b"].interrupt_id
    assert first[-1] == RunCompleted("interrupted")
    assert {q.question for q in await toy.runner.pending_interrupts(THREAD)} == set(questions)

    resumed = await _events(
        toy.runner,
        _input(
            None,
            run="b" * 32,
            resume={
                questions["Question a"].interrupt_id: "one",
                questions["Question b"].interrupt_id: "two",
            },
        ),
    )

    assert resumed[-1] == RunCompleted("stop")
    state = await toy.graph.aget_state({"configurable": {"thread_id": THREAD}})
    assert sorted(m.content for m in state.values["messages"]) == ["a=one", "b=two", "hi"]
    assert await toy.runner.pending_interrupts(THREAD) == ()


async def test_interrupt_ids_from_a_subgraph_node_are_the_raw_ids_to_resume_with() -> None:
    toy = threaded(subgraph_as_node_graph)

    first = await _events(toy.runner, _input(run="a" * 32))
    asked = next(e for e in first if isinstance(e, Clarification))
    pending = await toy.runner.pending_interrupts(THREAD)

    assert asked.question == "Question in the subgraph"
    assert pending == (asked,)  # the id streamed is the id of the checkpoint, un-namespaced
    resumed = await _events(
        toy.runner, _input(None, run="b" * 32, resume={asked.interrupt_id: "x"})
    )
    assert resumed[-1] == RunCompleted("stop")
    state = await toy.graph.aget_state(_key())
    assert state.values["messages"][-1].content == "sub=x"
    assert await toy.runner.pending_interrupts(THREAD) == ()


async def test_interrupt_ids_from_nested_subgraphs_in_nodes_resume_via_the_map() -> None:
    toy = threaded(nested_in_node_graph)

    first = await _events(toy.runner, _input(run="a" * 32))
    streamed = {e.question: e.interrupt_id for e in first if isinstance(e, Clarification)}
    pending = {q.question: q.interrupt_id for q in await toy.runner.pending_interrupts(THREAD)}

    assert set(streamed) == {"deep a", "deep b"}
    assert streamed == pending
    resumed = await _events(
        toy.runner,
        _input(None, run="b" * 32, resume={i: f"ans-{q}" for q, i in streamed.items()}),
    )
    assert resumed[-1] == RunCompleted("stop")
    state = await toy.graph.aget_state(_key())
    assert sorted(m.content for m in state.values["messages"]) == [
        "a=ans-deep a",
        "b=ans-deep b",
        "hi",
    ]
    assert await toy.runner.pending_interrupts(THREAD) == ()


def _key() -> RunnableConfig:
    return {"configurable": {"thread_id": THREAD}}


async def test_a_new_message_run_after_done_starts_a_fresh_turn() -> None:
    toy = threaded(_asking("Which one?"))
    first = await _events(toy.runner, _input(run="a" * 32))
    asked = next(e for e in first if isinstance(e, Clarification))
    await _events(toy.runner, _input(None, run="b" * 32, resume={asked.interrupt_id: "x"}))

    again = await _events(toy.runner, _input("again", run="c" * 32))

    assert any(isinstance(e, Clarification) for e in again)
    assert again[-1] == RunCompleted("interrupted")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (Clarification("Q", ("x",)), Clarification("Q", ("x",), "id1")),
        ("Plain question", Clarification("Plain question", (), "id1")),
        (
            {"question": "Which?", "options": ["a", "b"]},
            Clarification("Which?", ("a", "b"), "id1"),
        ),
        ({"question": "Which?"}, Clarification("Which?", (), "id1")),
        ({"question": "Which?", "options": ("a",)}, Clarification("Which?", ("a",), "id1")),
    ],
)
def test_interrupt_payloads_that_are_questions(payload: object, expected: Clarification) -> None:
    assert clarification_from_interrupt(payload, "id1") == expected


@pytest.mark.parametrize(
    "payload",
    [
        None,
        42,
        "   ",
        {"options": ["a"]},
        {"question": ""},
        {"question": 3},
        {"question": "Q", "options": [1]},
        {"question": "Q", "options": "ab"},
        ["list"],
        object(),
    ],
)
def test_malformed_interrupt_payloads_get_a_generic_question_and_a_warning(
    payload: object, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        result = clarification_from_interrupt(payload, "id1")

    assert result == Clarification(FALLBACK_QUESTION, (), "id1")
    assert "not a question" in caplog.text


async def test_malformed_payload_is_never_echoed_to_the_client() -> None:
    toy = threaded(_asking({"internal": "SECRET-STATE"}))

    events = await _events(toy.runner, _input())

    assert any(isinstance(e, Clarification) and e.question == FALLBACK_QUESTION for e in events)
    assert "SECRET-STATE" not in repr(events)
    assert events[-1] == RunCompleted("interrupted")


async def test_thread_messages_are_the_chat_turns_in_order_with_their_cited_sources() -> None:
    toy = threaded(cited_graph)
    await _events(toy.runner, _input("first"))
    await _events(toy.runner, _input("second"))

    assert await toy.runner.thread_messages(THREAD) == (
        TranscriptMessage("user", "first"),
        TranscriptMessage("assistant", "answer 1"),  # no sources cited
        TranscriptMessage("user", "second"),
        TranscriptMessage("assistant", "answer 2", (SourceRef("doc-2", "Doc", None),)),
    )  # the system note of each turn is not a chat turn


async def test_thread_messages_of_a_thread_without_checkpoint_are_empty() -> None:
    toy = threaded(cited_graph)

    assert await toy.runner.thread_messages(THREAD) == ()


async def test_thread_messages_of_a_runner_without_checkpointer_are_empty() -> None:
    runner = LangGraphRunner("toy", "d", state_probe_graph([]))

    assert await runner.thread_messages(THREAD) == ()
