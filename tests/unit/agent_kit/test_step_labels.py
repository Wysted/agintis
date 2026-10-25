"""Step labels: a `step_label` strategy names Send branches; subgraph steps inherit it."""

import operator
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command, Send

from agintis.agent_kit.runtime import LangGraphRunner, StepLabeler
from agintis.modules.agents_catalog.api import RunInput, Step
from agintis.shared_kernel.ids import RunId, TenantId

SECRET_QUERY = "my national id is 12.345.678-9"


class Task(TypedDict):
    topic: str
    query: str


class Branch(TypedDict):
    topic: str
    query: str
    found: NotRequired[list[str]]


class BranchOut(TypedDict):
    found: list[str]


class Parent(TypedDict):
    query: str
    found: Annotated[list[str], operator.add]


def _graph(topics: tuple[str, ...] = ("rules", "handbook")) -> Pregel[Any, Any, Any, Any]:
    async def extract(state: Branch) -> dict[str, Any]:
        return {}

    async def retrieve(state: Branch) -> dict[str, Any]:
        return {"found": [state["topic"]]}

    branch = (
        StateGraph(Branch, input_schema=Task, output_schema=BranchOut)
        .add_node("extract_filters", extract)
        .add_node("retrieve", retrieve)
        .add_edge(START, "extract_filters")
        .add_edge("extract_filters", "retrieve")
        .compile(checkpointer=False)
    )

    async def analyze(state: Parent) -> dict[str, Any]:
        return {}

    async def route(state: Parent) -> Command[Any]:
        return Command(goto=[Send("topic", Task(topic=t, query=state["query"])) for t in topics])

    async def merge(state: Parent) -> dict[str, Any]:
        return {}

    return (
        StateGraph(Parent)
        .add_node("analyze", analyze)
        .add_node("route", route, destinations=("topic",))
        .add_node("topic", branch)
        .add_node("merge", merge)
        .add_edge(START, "analyze")
        .add_edge("analyze", "route")
        .add_edge("topic", "merge")
        .add_edge("merge", END)
        .compile()
    )


def _topic_label(node: str, task_input: Any) -> str | None:
    return task_input["topic"] if node == "topic" else None


def _run_input() -> RunInput:
    return RunInput(TenantId("acme"), SECRET_QUERY, RunId("0123456789abcdef0123456789abcdef"))


async def _steps(runner: LangGraphRunner) -> list[Step]:
    return [e async for e in runner.astream(_run_input()) if isinstance(e, Step)]


def _runner(step_label: StepLabeler | None = None) -> LangGraphRunner:
    graph = _graph()
    # The graph input is the Parent state: map the message to it.
    kwargs: dict[str, Any] = {"input_mapper": lambda run_input: {"query": run_input.message}}
    if step_label is not None:
        kwargs["step_label"] = step_label
    return LangGraphRunner("toy", "A toy agent", graph, **kwargs)


async def test_topic_branches_and_their_inner_steps_carry_the_topic() -> None:
    steps = await _steps(_runner(_topic_label))

    for status in ("started", "finished"):
        topics = [s.label for s in steps if s.node == "topic" and s.status == status]
        assert sorted(t or "" for t in topics) == ["handbook", "rules"]
    # Inner steps inherit the label of their parent task (one pair per branch).
    for inner in ("topic/extract_filters", "topic/retrieve"):
        for status in ("started", "finished"):
            labels = [s.label for s in steps if s.node == inner and s.status == status]
            assert sorted(t or "" for t in labels) == ["handbook", "rules"], (inner, status)


async def test_other_steps_have_no_label() -> None:
    steps = await _steps(_runner(_topic_label))

    assert {s.label for s in steps if s.node in {"analyze", "route", "merge"}} == {None}
    assert [s.status for s in steps if s.node == "analyze"] == ["started", "finished"]


async def test_finished_reuses_the_label_of_the_same_task() -> None:
    steps = await _steps(_runner(_topic_label))

    # Started and finished labels pair per task even though END carries no input.
    started = sorted(s.label or "" for s in steps if s.node == "topic" and s.status == "started")
    finished = sorted(s.label or "" for s in steps if s.node == "topic" and s.status == "finished")
    assert started == finished == ["handbook", "rules"]


async def test_default_strategy_labels_nothing() -> None:
    steps = await _steps(_runner())

    assert steps
    assert {s.label for s in steps} == {None}


async def test_the_raw_task_input_never_reaches_the_events() -> None:
    def leaky(node: str, task_input: Any) -> str | None:
        return "ok" if node == "topic" else None

    events = [e async for e in _runner(leaky).astream(_run_input())]

    assert SECRET_QUERY not in repr(events)


async def test_a_failing_or_odd_labeler_does_not_fail_the_run() -> None:
    def boom(node: str, task_input: Any) -> str | None:
        raise RuntimeError("bug")

    def odd(node: str, task_input: Any) -> Any:
        return 7 if node == "topic" else "   "

    assert {s.label for s in await _steps(_runner(boom))} == {None}
    assert {s.label for s in await _steps(_runner(odd))} == {None}


async def test_labels_are_bounded() -> None:
    def long(node: str, task_input: Any) -> str | None:
        return "x" * 1000 if node == "topic" else None

    labels = {s.label for s in await _steps(_runner(long)) if s.node == "topic"}

    assert labels == {"x" * 100}
