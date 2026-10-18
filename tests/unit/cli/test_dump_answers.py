import asyncio
import json
import stat
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agintis.agent_kit.judge import AnswerRecord, ScopeVerdict, record_from_json
from agintis.agent_kit.judge.render import faithfulness_messages
from agintis.agent_kit.nodes import SourceLimits
from agintis.agent_kit.runtime import AgentContext
from agintis.cli import dump_answers
from agintis.cli._common import EXIT_ABORTED, EXIT_USAGE, UsageError
from agintis.cli.dump_answers import (
    DumpWriter,
    Question,
    RepeatedFailureError,
    answer_question,
    load_questions,
    repeated_failure,
    write_dump,
)
from agintis.cli.dump_answers import (
    dump_answers as run_dump,
)
from agintis.cli.judge import load_dump
from agintis.modules.agents.rag_chat import (
    RagChatDeps,
    agent_deps,
    build_graph,
    parse_agent_spec,
    roles_used,
)
from agintis.modules.knowledge.api import MetadataField, RagProfile, RagProfileRegistry
from agintis.shared_kernel.ids import TenantId
from tests.fakes.llm import ScriptedChatModel, chat_models_for, rate_limit_error
from tests.fakes.retrieval import FakeRetrieval, make_hit
from tests.unit.agents.support import (
    PROFILES as CLARIFICATION_PROFILES,
)
from tests.unit.agents.support import (
    REFUSAL,
    Script,
    by_query,
    make_model,
    make_probe,
    make_retrieval,
    multi_topic_spec,
    scoped_spec,
)

TENANT = TenantId("acme")
PROFILES = RagProfileRegistry(
    [
        RagProfile(
            "docs",
            3,
            100,
            "simple",
            "structured",
            metadata_fields=(MetadataField("kind", "str", filterable=True),),
        )
    ]
)
SPEC = parse_agent_spec(
    {
        "id": "docs-rag",
        "description": "Answers from the docs.",
        "topics": [{"name": "docs", "description": "d", "profile": "docs"}],
    },
    origin="test",
)


def _write(path: Path, *rows: object) -> Path:
    path.write_text("".join(f"{json.dumps(r)}\n" for r in rows), encoding="utf-8")
    return path


# --- questions file -----------------------------------------------------------------------------


def test_questions_accept_label_or_expected_scope_and_a_default(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "q.jsonl",
        {"id": "a", "question": "Q1", "label": "out"},
        {"id": "b", "question": "Q2", "expected_scope": "in"},
        {"id": "c", "question": "Q3"},
    )
    assert [q.expected_scope for q in load_questions(path)] == ["out", "in", None]
    assert [q.expected_scope for q in load_questions(path, "in")] == ["out", "in", "in"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"id": "a"}], "non-empty strings"),
        ([{"id": "a", "question": "  "}], "non-empty strings"),
        ([{"id": "a", "question": "q", "label": "maybe"}], "scope label must be one of"),
        ([{"id": "a", "question": "q"}, {"id": "a", "question": "r"}], "already used"),
        ([], "no questions"),
        (["text"], "JSON object"),
    ],
)
def test_bad_questions_files_are_usage_errors(
    tmp_path: Path, rows: list[object], message: str
) -> None:
    with pytest.raises(UsageError, match=message):
        load_questions(_write(tmp_path / "q.jsonl", *rows))


def test_a_missing_questions_file_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="cannot read"):
        load_questions(tmp_path / "nope.jsonl")


# --- the dump over a real graph -----------------------------------------------------------------


def _scripted_graph() -> tuple[Any, ScriptedChatModel]:
    model = ScriptedChatModel(
        structured=[
            {
                "standalone_query": "deadline?",
                "is_clear": True,
                "confidence": 0.9,
                "request_kind": "domain",
            },
            {"kind": None},
        ],
        replies=["It is 24 months [1]."],
    )
    deps = RagChatDeps(
        chat_models_for(model),
        FakeRetrieval(
            (make_hit("The deadline is 24 months.", uri="file:///a.pdf", title="Doc A"),)
        ),
    )
    return build_graph(SPEC, PROFILES, deps), model


async def test_the_dump_records_answer_sources_and_models_of_a_real_graph_run() -> None:
    graph, _ = _scripted_graph()
    deps = RagChatDeps(chat_models_for(), FakeRetrieval(()))
    labels = {
        role.value: agent_deps(SPEC, deps).chat_models.describe(role).label
        for role in roles_used(SPEC)
    }

    records = await run_dump(
        graph,
        [Question("q1", "What is the deadline?", "in")],
        tenant=TENANT,
        agent=SPEC.id,
        limits=SPEC.limits,
        models=labels,
        meta={"git": "abc1234"},
        citations="inline",
    )

    record = records[0]
    assert (record.id, record.agent, record.expected_scope) == ("q1", "docs-rag", "in")
    assert record.answer == "It is 24 months [1]."
    assert record.error is None
    assert record.scope is None  # no clarification policy on this spec
    assert [(s.n, s.title, s.uri, s.text) for s in record.sources] == [
        (1, "Doc A", "file:///a.pdf", "The deadline is 24 months.")
    ]
    assert record.models["synthesis"] == "default (openai:synthesis)"
    assert record.meta == {"git": "abc1234"}


async def test_the_judge_renders_the_sources_exactly_like_the_synthesis_showed_them() -> None:
    graph, model = _scripted_graph()
    record = (
        await run_dump(
            graph,
            [Question("q1", "What is the deadline?")],
            tenant=TENANT,
            agent=SPEC.id,
            limits=SPEC.limits,
            models={},
            meta={},
            citations="inline",
        )
    )[0]

    sources_message = model.seen[0][-2]  # [system, sources (user role), question]
    assert record.sources[0].locator  # the locator the model saw travels in the dump
    assert record.sources[0].locator in sources_message.text
    assert record.rendered_sources in sources_message.text
    assert record.sources[0].render() in record.rendered_sources
    # and the judge's own prompt carries that very block
    judge_prompt = faithfulness_messages("sys", record)[1].text
    assert record.rendered_sources in judge_prompt


async def test_a_refusal_without_any_evidence_keeps_its_scope_with_a_null_score() -> None:
    model = make_model(Script([]), [])
    deps = RagChatDeps(chat_models_for(model), make_retrieval(), probe=make_probe(q=None))
    graph = build_graph(scoped_spec(), CLARIFICATION_PROFILES, deps)

    record = await answer_question(
        graph,
        Question("q", "q", "out"),
        tenant=TENANT,
        agent="scoped-rag",
        limits=SourceLimits(),
        models={},
        meta={},
        citations="inline",
    )

    assert record.scope == ScopeVerdict("out", None)
    assert record.answer == REFUSAL
    assert record.sources == ()
    assert record_from_json(json.loads(json.dumps(record.to_json()))) == record


async def test_the_rewrite_then_synthesize_path_dumps_the_sources_the_model_saw() -> None:
    # One source, `min_sources` 2: insufficient -> rewrite; the rewrite leaves the query unchanged
    # (degraded), so the graph goes straight to synthesis WITHOUT a second merge.
    script = Script([["alpha"]], rewrites=["question"])
    model = make_model(script, ["Only this is known [1]."])
    retrieval = FakeRetrieval(respond=by_query(question={"alpha": ("a-only",)}))
    deps = RagChatDeps(chat_models_for(model), retrieval)
    graph = build_graph(multi_topic_spec(min_sources=2), CLARIFICATION_PROFILES, deps)

    record = await answer_question(
        graph,
        Question("q", "question", "in"),
        tenant=TENANT,
        agent="multi-rag",
        limits=SourceLimits(),
        models={},
        meta={},
        citations="inline",
    )

    assert record.error is None
    assert record.answer == "Only this is known [1]."
    assert [s.uri for s in record.sources] == ["file:///a-only"]
    assert script.rewrites == []  # the rewrite really ran
    shown = model.seen[0][-2].text  # what the synthesis model was shown
    assert record.rendered_sources in shown


async def test_a_weak_score_keeps_its_value() -> None:
    model = make_model(Script([]), [])
    deps = RagChatDeps(chat_models_for(model), make_retrieval(), probe=make_probe(q=0.1))
    record = await answer_question(
        build_graph(scoped_spec(), CLARIFICATION_PROFILES, deps),
        Question("q", "q"),
        tenant=TENANT,
        agent="scoped-rag",
        limits=SourceLimits(),
        models={},
        meta={},
        citations="inline",
    )
    assert record.scope == ScopeVerdict("out", pytest.approx(0.1))  # type: ignore[arg-type]


async def test_the_dump_survives_the_jsonl_roundtrip_into_the_judge_loader(tmp_path: Path) -> None:
    graph, _ = _scripted_graph()
    records = await run_dump(
        graph,
        [Question("q1", "What is the deadline?")],
        tenant=TENANT,
        agent=SPEC.id,
        limits=SPEC.limits,
        models={},
        meta={},
        citations="inline",
    )
    path = tmp_path / "answers.jsonl"
    write_dump(path, records)

    assert load_dump(path) == records
    assert record_from_json(json.loads(path.read_text().splitlines()[0])) == records[0]


# --- fake graph: scope, limits, isolation, concurrency ------------------------------------------


class FakeGraph:
    def __init__(self, responder: Any, delay: float = 0.0) -> None:
        self._responder = responder
        self._delay = delay
        self.contexts: list[AgentContext] = []
        self.configs: list[RunnableConfig | None] = []

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, *, context: Any = None
    ) -> Any:
        self.contexts.append(context)
        self.configs.append(config)
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._responder(input["messages"][0].content)


def _state(**extra: Any) -> dict[str, Any]:
    return {"answer": "Respuesta [1].", "sources": [], "messages": [], **extra}


async def test_the_scope_event_and_the_sources_the_synthesis_saw_are_recorded() -> None:
    item = {"source": "file:///a.pdf", "title": None, "locator": None, "text": "x" * 50}
    graph = FakeGraph(lambda _q: _state(sources=[item], scope_level="low", scope_score=0.5234))
    record = await answer_question(
        graph,
        Question("q", "Q?"),
        tenant=TENANT,
        agent="a",
        limits=SourceLimits(max_source_chars=10),
        models={},
        meta={},
        citations="inline",
    )

    assert record.scope is not None
    assert (record.scope.level, record.scope.score) == ("low", 0.5234)
    assert record.sources[0].text == "x" * 10  # cut like the synthesis cut it
    assert record.sources[0].title is None


async def test_a_refused_turn_has_scope_out_and_no_sources() -> None:
    graph = FakeGraph(
        lambda _q: _state(answer="Fixed message.", scope_level="out", scope_score=0.2)
    )
    record = await answer_question(
        graph, Question("q", "Q?", "out"), tenant=TENANT, agent="a",
        limits=SourceLimits(), models={}, meta={}, citations="inline",
    )  # fmt: skip
    assert record.answer == "Fixed message."
    assert record.sources == ()
    assert record.scope is not None
    assert record.scope.level == "out"


async def test_the_refusal_reason_is_recorded_and_old_dumps_without_it_still_load() -> None:
    graph = FakeGraph(lambda _q: _state(answer="Hola.", refusal_reason="social"))
    record = await answer_question(
        graph, Question("q", "Hola"), tenant=TENANT, agent="a",
        limits=SourceLimits(), models={}, meta={}, citations="inline",
    )  # fmt: skip
    row = record.to_json()
    assert (record.refusal_reason, row["refusal_reason"]) == ("social", "social")
    assert record_from_json(row) == record

    del row["refusal_reason"]
    assert record_from_json(row).refusal_reason is None


async def test_the_answer_falls_back_to_the_last_ai_message() -> None:
    graph = FakeGraph(lambda _q: {"messages": [AIMessage("From history.")]})
    record = await answer_question(
        graph, Question("q", "Q?"), tenant=TENANT, agent="a",
        limits=SourceLimits(), models={}, meta={}, citations="inline",
    )  # fmt: skip
    assert record.answer == "From history."


async def test_a_failing_question_is_recorded_and_the_others_still_run() -> None:
    def respond(question: str) -> dict[str, Any]:
        if question == "boom":
            raise rate_limit_error()
        return _state()

    graph = FakeGraph(respond)
    records = await run_dump(
        graph,
        [Question("a", "ok"), Question("b", "boom"), Question("c", "ok")],
        tenant=TENANT,
        agent="a",
        limits=SourceLimits(),
        models={},
        meta={},
        citations="inline",
        concurrency=3,
    )

    assert [r.id for r in records] == ["a", "b", "c"]  # file order
    assert [r.error for r in records] == [None, "RateLimitError (status 429)", None]
    assert records[1].answer == ""


async def test_every_record_carries_the_agents_citation_style() -> None:
    def respond(question: str) -> dict[str, Any]:
        if question == "boom":
            raise rate_limit_error()
        return _state()

    records = await run_dump(
        FakeGraph(respond),
        [Question("a", "ok"), Question("b", "boom")],
        tenant=TENANT,
        agent="a",
        limits=SourceLimits(),
        models={},
        meta={},
        citations="none",
    )

    assert [r.citations for r in records] == ["none", "none"]  # the failed one too


async def test_each_run_gets_its_own_context_and_a_dump_tag() -> None:
    graph = FakeGraph(lambda _q: _state())
    await run_dump(
        graph,
        [Question("a", "x"), Question("b", "y")],
        tenant=TENANT,
        agent="a",
        limits=SourceLimits(),
        models={},
        meta={},
        citations="inline",
    )
    assert {c.tenant_id for c in graph.contexts} == {TENANT}
    assert len({c.run_id for c in graph.contexts}) == 2
    assert all("agintis:dump-answers" in (c or {}).get("tags", []) for c in graph.configs)


# --- command line -------------------------------------------------------------------------------


def test_main_reports_usage_errors_before_booting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    agents = tmp_path / "agents"
    agents.mkdir()
    questions = _write(tmp_path / "q.jsonl", {"id": "a", "question": "Q"})
    code = dump_answers.main(
        ["--profiles", str(tmp_path), "--agents", str(agents), "--agent", "nope",
         "--questions", str(questions), "--out", str(tmp_path / "out.jsonl")]
    )  # fmt: skip
    assert code == EXIT_USAGE
    assert "unknown agent 'nope'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--concurrency", "0"], "--concurrency"),
        (["--limit", "0"], "--limit"),
        (["--out", "{dir}"], "is a directory"),
        (["--out", "{dir}/missing/out.jsonl"], "does not exist"),
    ],
)
def test_bad_flags_are_usage_errors(
    tmp_path: Path, flags: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "a.yaml").write_text(
        "id: docs-rag\ndescription: d\ntopics:\n  - {name: docs, description: d, profile: docs}\n"
    )
    questions = _write(tmp_path / "q.jsonl", {"id": "a", "question": "Q"})
    base = ["--profiles", str(tmp_path), "--agents", str(agents), "--agent", "docs-rag",
            "--questions", str(questions), "--out", str(tmp_path / "out.jsonl")]  # fmt: skip
    flags = [f.format(dir=tmp_path) for f in flags]
    # later flags override the earlier ones (argparse keeps the last)
    assert dump_answers.main([*base, *flags]) == EXIT_USAGE
    assert message in capsys.readouterr().err


def test_an_unexpected_failure_aborts_with_exit_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def boom(_args: Any, _writer: Any) -> Any:
        raise ConnectionError("db down")

    monkeypatch.setattr(dump_answers, "run", boom)
    code = dump_answers.main(
        ["--profiles", str(tmp_path), "--agents", str(tmp_path), "--agent", "a",
         "--questions", str(tmp_path / "q"), "--out", str(tmp_path / "o")]
    )  # fmt: skip
    assert code == EXIT_ABORTED
    assert "ConnectionError" in capsys.readouterr().err


# --- incremental, private, fail-fast ------------------------------------------------------------


def _record(ident: str, error: str | None = None) -> AnswerRecord:
    return AnswerRecord(
        id=ident,
        question="q",
        answer="" if error else "a",
        agent="a",
        error=error,
        citations="inline",
    )


def test_the_writer_appends_flushed_lines_to_a_private_partial_and_renames_on_commit(
    tmp_path: Path,
) -> None:
    out = tmp_path / "answers.jsonl"
    writer = DumpWriter(out)

    writer.write(_record("a"))
    assert not out.exists()
    # flushed per record: readable before the run ends (what an abort keeps)
    assert [json.loads(x)["id"] for x in writer.partial.read_text().splitlines()] == ["a"]
    assert stat.S_IMODE(writer.partial.stat().st_mode) == 0o600

    writer.write(_record("b"))
    writer.commit()

    assert not writer.partial.exists()
    assert [json.loads(x)["id"] for x in out.read_text().splitlines()] == ["a", "b"]
    assert stat.S_IMODE(out.stat().st_mode) == 0o600


def test_an_uncommitted_writer_leaves_the_partial_behind(tmp_path: Path) -> None:
    writer = DumpWriter(tmp_path / "answers.jsonl")
    writer.write(_record("a"))
    writer.close()
    assert writer.partial.exists()
    assert not (tmp_path / "answers.jsonl").exists()


def test_committing_an_empty_dump_leaves_an_empty_private_file(tmp_path: Path) -> None:
    writer = DumpWriter(tmp_path / "answers.jsonl")
    writer.commit()
    assert (tmp_path / "answers.jsonl").read_text() == ""


def test_write_dump_is_private_too(tmp_path: Path) -> None:
    write_dump(tmp_path / "d.jsonl", [_record("a")])
    assert stat.S_IMODE((tmp_path / "d.jsonl").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("errors", "expected"),
    [
        ([], None),
        ([None, None, None, None], None),
        (["Op"], None),  # too early
        (["Op", "Op"], None),
        (["Op", "Op", "Op"], "Op"),  # the first records all fail the same way
        (["Op", "Op", None], None),  # three records, not all failing
        (["Op", "Op", None, "Op", "Op"], "Op"),  # > half of >= 5
        (["Op", "Op", None, None, "Op"], "Op"),  # 3 of 5 is more than half
        (["A", "B", "C", "A", "B"], None),  # mixed classes are not an infrastructure outage
    ],
)
def test_repeated_failure_rule(errors: list[str | None], expected: str | None) -> None:
    assert repeated_failure(errors) == expected


async def test_the_dump_aborts_when_the_same_failure_repeats_keeping_what_was_written() -> None:
    graph = FakeGraph(lambda _q: (_ for _ in ()).throw(rate_limit_error()), delay=0.01)
    sunk: list[AnswerRecord] = []

    with pytest.raises(RepeatedFailureError, match="RateLimitError"):
        await run_dump(
            graph,
            [Question(f"q{i}", "x") for i in range(20)],
            tenant=TENANT,
            agent="a",
            limits=SourceLimits(),
            models={},
            meta={},
            citations="inline",
            sink=sunk.append,
        )

    assert [r.id for r in sunk] == ["q0", "q1", "q2"]  # stopped as soon as it was clear
    assert len(graph.contexts) < 20  # the remaining questions were never asked


async def test_records_reach_the_sink_in_file_order_even_with_concurrency() -> None:
    sunk: list[str] = []
    graph = FakeGraph(lambda _q: _state())
    await run_dump(
        graph,
        [Question(f"q{i}", "x") for i in range(6)],
        tenant=TENANT,
        agent="a",
        limits=SourceLimits(),
        models={},
        meta={},
        citations="inline",
        concurrency=3,
        sink=lambda r: sunk.append(r.id),
    )
    assert sunk == [f"q{i}" for i in range(6)]


def test_main_keeps_the_partial_dump_when_the_run_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def partial_then_fail(_args: Any, writer: DumpWriter) -> Any:
        writer.write(_record("paid-for"))
        raise RepeatedFailureError("RateLimitError repeated over 3 questions")

    monkeypatch.setattr(dump_answers, "run", partial_then_fail)
    out = tmp_path / "o.jsonl"

    code = dump_answers.main(
        ["--profiles", str(tmp_path), "--agents", str(tmp_path), "--agent", "a",
         "--questions", str(tmp_path / "q"), "--out", str(out)]
    )  # fmt: skip

    assert code == EXIT_ABORTED
    assert not out.exists()
    assert "paid-for" in (tmp_path / "o.jsonl.partial").read_text()
    assert "o.jsonl.partial" in capsys.readouterr().err


def test_main_commits_and_returns_1_when_some_questions_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run_ok(_args: Any, writer: DumpWriter) -> list[AnswerRecord]:
        records = [_record("a"), _record("b", "RateLimitError"), _record("c")]
        for record in records:
            writer.write(record)
        return records

    monkeypatch.setattr(dump_answers, "run", run_ok)
    out = tmp_path / "o.jsonl"
    code = dump_answers.main(
        ["--profiles", str(tmp_path), "--agents", str(tmp_path), "--agent", "a",
         "--questions", str(tmp_path / "q"), "--out", str(out)]
    )  # fmt: skip
    assert code == 1
    assert len(out.read_text().splitlines()) == 3
    assert not (tmp_path / "o.jsonl.partial").exists()


def test_the_judge_dump_directory_is_gitignored() -> None:
    ignore = (Path(__file__).parents[3] / ".gitignore").read_text()
    assert "case/*/judge/" in ignore.splitlines()


def test_a_looser_existing_partial_is_tightened_to_0600(tmp_path: Path) -> None:
    partial = tmp_path / "answers.jsonl.partial"
    partial.write_text("old\n")
    partial.chmod(0o644)

    writer = DumpWriter(tmp_path / "answers.jsonl")
    writer.write(_record("a"))
    writer.close()

    assert stat.S_IMODE(partial.stat().st_mode) == 0o600
    assert "old" not in partial.read_text()
