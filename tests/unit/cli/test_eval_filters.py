"""`cli.eval --extract-filters`: extraction through the agent's own code, filtered runs, report."""

import argparse
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.agent_kit.nodes import FilterClause, clauses_to_filter
from agintis.bootstrap.container import Container
from agintis.cli import eval as eval_cli
from agintis.cli._common import EXIT_USAGE, UsageError
from agintis.cli.eval import (
    build_analyzers,
    build_filter_extractors,
    build_parser,
    check_usage,
    evaluate,
    extract_question_filters,
    report_for,
    topics_by_profile,
)
from agintis.cli.eval_metrics import (
    EvalQuestion,
    QuestionOutcome,
    changed_cases,
    format_comparison,
)
from agintis.modules.agents.rag_chat import AgentSpec, parse_agent_spec
from agintis.modules.knowledge.api import (
    DENSE_RETRIEVAL_MODE,
    IngestDocument,
    MetadataField,
    ProbeProfiles,
    RagProfile,
    RagProfileRegistry,
    RetrieveForTopic,
    SourceFile,
)
from agintis.modules.knowledge.application import ChunkDraft, Chunker, ParsedDocument
from agintis.modules.knowledge.infrastructure.registry import build_retrieval_modes
from agintis.platform.llm import ChatModels, ChatRole, LlmConfig, LlmProfile, ModelCatalog
from agintis.shared_kernel.criteria import Filter
from agintis.shared_kernel.ids import TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    FakeParser,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryIngestLock,
)
from tests.fakes.llm import ScriptedChatModel, chat_models_for
from tests.support.settings import make_settings

TENANT = TenantId("t")
TEXT = "\n\n".join(f"paragraph {i}" for i in range(10))
PROFILE = RagProfile(
    "p",
    3,
    100,
    "fake",
    "kinds",
    metadata_fields=(
        MetadataField("kind", "str", filterable=True),
        MetadataField("secret", "str"),
    ),
)


class _KindChunker(Chunker):
    """Paragraphs 0-4 are `memo`s, 5-9 `letter`s."""

    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
        return [
            ChunkDraft(
                text=s.text,
                metadata={"page": i + 1, "kind": "memo" if i < 5 else "letter"},
            )
            for i, s in enumerate(parsed.sections)
        ]


OTHER = RagProfile("other", 3, 100, "fake", "kinds", metadata_fields=PROFILE.metadata_fields)


async def _container(chat_models: ChatModels) -> Container:
    documents, chunks = InMemoryDocumentRepository(), InMemoryChunkRepository()
    embeddings = DeterministicFakeEmbedding(size=16)
    await IngestDocument(
        documents,
        chunks,
        {"fake": FakeParser()},
        {"kinds": _KindChunker()},
        embeddings,
        FixedClock(),
        InMemoryIngestLock(),
    ).execute(TENANT, PROFILE, SourceFile("file:///d.txt", "text/plain", TEXT.encode()))
    retrieve = RetrieveForTopic(
        documents,
        chunks,
        embeddings,
        retrieval_modes=build_retrieval_modes(branch_depth=20, rrf_k=60),
    )
    stub = SimpleNamespace(
        profiles=RagProfileRegistry([PROFILE, OTHER]),
        retrieve_for_topic=retrieve,
        probe_profiles=ProbeProfiles(chunks, embeddings),
        chat_models=chat_models,
        chunks=chunks,
    )
    return cast(Container, stub)


def _question(qid: str, text: str) -> EvalQuestion:
    return EvalQuestion(qid, text, "exact_reference", "p", text, {})


def _reply(kind: str | None) -> dict[str, Any]:
    return {"kind": kind}


def _spec(agent_id: str = "agent", profile: str = "p", **extra: Any) -> AgentSpec:
    return parse_agent_spec(
        {
            "id": agent_id,
            "description": "d",
            "topics": [
                {
                    "name": "t",
                    "description": "d",
                    "profile": profile,
                    "filters": {"kind": "HINT"},
                }
            ],
            **extra,
        },
        origin="t",
    )


def _clause(kind: str) -> FilterClause:
    return {"field": "kind", "op": "eq", "value": kind}


class TestEvaluateWithFilters:
    async def test_the_filter_narrows_the_search(self) -> None:
        container = await _container(chat_models_for())
        q = [_question("m", "paragraph 7")]

        async def run(kind: str) -> QuestionOutcome:
            (outcome,) = await evaluate(
                container,
                q,
                mode=DENSE_RETRIEVAL_MODE,
                ks=(10,),
                tenant=TENANT,
                rerank=False,
                filters={"m": [_clause(kind)]},
            )
            return outcome

        letter = await run("letter")  # paragraph 7 is a letter
        assert (letter.rank is not None, letter.filter_level, letter.filters_relaxed) == (
            True,
            "full",
            False,
        )
        memo = await run("memo")  # a wrong but non-empty filter excludes it: hints can hurt
        assert memo.rank is None
        assert memo.filter_level == "full"

    async def test_zero_hits_walk_down_the_ladder_and_report_the_level(self) -> None:
        container = await _container(chat_models_for())
        (outcome,) = await evaluate(
            container,
            [_question("m", "paragraph 7")],
            mode=DENSE_RETRIEVAL_MODE,
            ks=(10,),
            tenant=TENANT,
            rerank=False,
            filters={"m": [_clause("nonexistent")]},
        )
        assert outcome.filters_relaxed is True
        assert outcome.filter_level == "none"  # `kind` is free text: no strong step
        assert outcome.rank is not None  # recall first: the unfiltered search found it

    async def test_a_question_without_clauses_is_plain(self) -> None:
        container = await _container(chat_models_for())
        (outcome,) = await evaluate(
            container,
            [_question("m", "paragraph 7")],
            mode=DENSE_RETRIEVAL_MODE,
            ks=(10,),
            tenant=TENANT,
            rerank=False,
            filters={},
        )
        assert (outcome.filter_level, outcome.filters_relaxed) == ("none", False)

    async def test_without_filters_the_run_is_exactly_the_plain_one(self) -> None:
        container = await _container(chat_models_for())
        calls: list[tuple[Any, ...]] = []
        real = container.retrieve_for_topic.execute

        async def spy(*args: Any) -> Any:
            calls.append(args)
            return await real(*args)

        container.retrieve_for_topic.execute = spy  # type: ignore[method-assign,assignment]
        (outcome,) = await evaluate(
            container,
            [_question("m", "paragraph 7")],
            mode=DENSE_RETRIEVAL_MODE,
            ks=(10,),
            tenant=TENANT,
            rerank=False,
        )
        assert [len(c) for c in calls] == [3]  # (tenant, profile, query): no filter argument
        assert (outcome.filters_relaxed, outcome.filter_level) == (False, None)


class TestExtraction:
    async def test_runs_the_agents_extractor_once_per_question(self) -> None:
        model = ScriptedChatModel(structured=[_reply("letter"), _reply(None)])
        container = await _container(chat_models_for(model))
        extractors = build_filter_extractors(container, topics_by_profile([_spec()]), ["p"])

        clauses, inputs = await extract_question_filters(
            [_question("a", "letters please"), _question("b", "anything")], extractors
        )

        assert clauses == {"a": [_clause("letter")], "b": []}
        assert inputs == {}  # raw question: no analysis ran
        assert [m[1].content for m in model.structured_seen] == ["letters please", "anything"]
        # The allowlist and hint of the topic: `kind` only, `secret` is not even asked for.
        schema = extractors["p"].schema
        assert schema is not None
        assert list(schema.model_json_schema()["properties"]) == ["kind"]

    async def test_the_spec_configures_the_extractor_prompt_and_hints(self) -> None:
        spec = _spec(prompts={"extract_filters": "CUSTOM PROMPT"})
        model = ScriptedChatModel(structured=[_reply("memo")])
        container = await _container(chat_models_for(model))
        extractors = build_filter_extractors(container, topics_by_profile([spec]), ["p"])

        await extractors["p"].extract("q")

        assert model.structured_seen[0][0].content == "CUSTOM PROMPT"
        schema = extractors["p"].schema
        assert schema is not None
        assert schema.model_json_schema()["properties"]["kind"]["description"].startswith("HINT")

    async def test_profiles_without_a_topic_get_no_extractor_and_stay_plain(self) -> None:
        model = ScriptedChatModel(structured=[_reply("letter")])
        container = await _container(chat_models_for(model))
        extractors = build_filter_extractors(
            container, topics_by_profile([_spec()]), ["p", "other"]
        )
        questions = [_question("a", "x"), EvalQuestion("o", "y", "t", "other", "y", {})]

        clauses, _ = await extract_question_filters(questions, extractors)

        assert set(extractors) == {"p"}
        assert set(clauses) == {"a"}  # nothing extracted (and no LLM call) for `other`
        assert len(model.structured_seen) == 1

    async def test_analyze_extracts_from_the_standalone_query(self) -> None:
        model = ScriptedChatModel(
            structured=[
                {
                    "standalone_query": "REWRITTEN letters",
                    "is_clear": True,
                    "confidence": 0.9,
                    "request_kind": "domain",
                },
                _reply("letter"),
            ]
        )
        container = await _container(chat_models_for(model))
        topics = topics_by_profile([_spec(prompts={"analyze": "MY ANALYZE"})])
        extractors = build_filter_extractors(container, topics, ["p"])
        analyzers = build_analyzers(container, topics, ["p"])

        clauses, inputs = await extract_question_filters(
            [_question("a", "letters?")], extractors, analyzers
        )

        assert inputs == {"a": "REWRITTEN letters"}
        assert clauses == {"a": [_clause("letter")]}
        analyzer_call, extractor_call = model.structured_seen
        assert analyzer_call[0].content == "MY ANALYZE"
        assert extractor_call[1].content == "REWRITTEN letters"  # the node's own path, then extract

    def test_two_specs_on_one_profile_are_an_error(self) -> None:
        with pytest.raises(UsageError, match="both cover profile 'p'"):
            topics_by_profile([_spec("one"), _spec("two")])


class TestReport:
    def test_filtered_report_adds_filters_level_relaxed_and_extraction(self) -> None:
        outcomes = [
            QuestionOutcome("a", "p", "t", 1, 5, False, "full"),
            QuestionOutcome("b", "other", "t", 2, 5, True, "strong"),
        ]
        clauses: Mapping[str, Any] = {"a": [_clause("x")]}

        plain = report_for(outcomes, (1, 5))
        filtered = report_for(
            outcomes, (1, 5), clauses, {"p": "extracted", "other": "skipped (no topic)"}
        )

        assert "filters" not in plain["questions"][0]  # as before without the flag
        assert filtered["questions"][0]["filters"] == clauses["a"]
        assert filtered["questions"][1] == {
            "id": "b",
            "profile": "other",
            "type": "t",
            "rank": 2,
            "retrieved": 5,
            "filters": [],
            "filter_level": "strong",
            "filters_relaxed": True,
            "filter_extraction": "skipped (no topic)",
        }

    def test_comparison_shows_deltas_levels_and_the_questions_whose_rank_changed(self) -> None:
        plain = [
            QuestionOutcome("cmp-01", "p", "t", None, 10),
            QuestionOutcome("x", "p", "t", 1, 10),
        ]
        filtered = [
            QuestionOutcome("cmp-01", "p", "t", 1, 10, False, "full"),
            QuestionOutcome("x", "p", "t", 1, 10, True, "strong"),
        ]

        text = format_comparison(plain, filtered, (1, 10))

        assert changed_cases(plain, filtered) == [("cmp-01", None, 1)]
        assert "rank changed: cmp-01 miss->#1" in text
        assert "hit@1" in text
        assert "+0.500" in text
        assert "filters relaxed on zero hits (to level): x->strong" in text


class TestUsage:
    def _args(self, **changes: Any) -> argparse.Namespace:
        base = {"extract_filters": False, "agents": None, "analyze": False, "route": False}
        return argparse.Namespace(**{**base, **changes})

    def test_flags(self) -> None:
        args = build_parser().parse_args(
            [
                "--profiles",
                "p",
                "--questions",
                "q",
                "--extract-filters",
                "--agents",
                "a",
                "--analyze",
            ]
        )
        assert (args.extract_filters, args.agents, args.analyze) == (True, Path("a"), True)
        plain = build_parser().parse_args(["--profiles", "p", "--questions", "q"])
        assert (plain.extract_filters, plain.agents, plain.analyze) == (False, None, False)

    def test_analyze_needs_extract_filters(self) -> None:
        with pytest.raises(UsageError, match="--analyze needs --extract-filters"):
            check_usage(self._args(analyze=True))

    def test_extract_filters_needs_agents(self) -> None:
        with pytest.raises(UsageError, match="--agents"):
            check_usage(self._args(extract_filters=True))

    def test_agents_without_extract_filters_only_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        check_usage(self._args(agents=Path("a")))
        assert "--agents has no effect without --extract-filters" in caplog.text

    def test_main_reports_a_usage_error_with_exit_code_2(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = eval_cli.main(["--profiles", "p", "--questions", "q", "--analyze"])
        assert code == EXIT_USAGE
        assert "--analyze needs" in capsys.readouterr().err


class TestRun:
    @pytest.fixture
    def questions_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "q.jsonl"
        rows = [
            '{"id": "m", "question": "paragraph 7", "type": "x", "profile": "p", '
            '"expected_snippet": "paragraph 7", "locator": {}}',
            '{"id": "o", "question": "paragraph 7", "type": "x", "profile": "other", '
            '"expected_snippet": "paragraph 7", "locator": {}}',
        ]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    @pytest.fixture
    def agents_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "agents"
        directory.mkdir()
        (directory / "a.yaml").write_text(
            "id: agent\ndescription: d\n"
            "topics: [{name: t, description: d, profile: p, filters: {kind: HINT}}]\n",
            encoding="utf-8",
        )
        return directory

    def _patch(
        self, monkeypatch: pytest.MonkeyPatch, container: Container, *, llm: bool = True
    ) -> list[Any]:
        built: list[Any] = []

        class _Entered:
            async def __aenter__(self) -> Container:
                return container

            async def __aexit__(self, *exc: object) -> None:
                return None

        def build(settings: Any) -> _Entered:
            built.append(settings)
            return _Entered()

        settings = make_settings(llm={"provider": "openai"} if llm else {})
        monkeypatch.setattr(eval_cli, "get_settings", lambda: settings)
        monkeypatch.setattr(eval_cli, "Container", SimpleNamespace(build=build))
        return built

    def _args(self, questions: Path, **extra: Any) -> argparse.Namespace:
        base = {
            "profiles": questions.parent,
            "questions": questions,
            "k": (1, 5),
            "mode": DENSE_RETRIEVAL_MODE,
            "report": None,
            "tenant": "t",
            "rerank": False,
            "extract_filters": False,
            "agents": None,
            "analyze": False,
            "route": False,
        }
        return argparse.Namespace(**{**base, **extra})

    async def test_without_the_flag_no_llm_is_needed_and_the_report_is_unchanged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        questions_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._patch(monkeypatch, await _container(ChatModels(None)), llm=False)

        report = await eval_cli.run(self._args(questions_file))

        assert set(report) == {"tenant", "ks", "questions_file", "rerank", "modes", "provenance"}
        assert set(report["modes"][DENSE_RETRIEVAL_MODE]) == {"overall", "profiles", "questions"}
        assert "WITH extracted filters" not in capsys.readouterr().out

    async def test_the_llm_is_checked_before_the_app_boots(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        built = self._patch(monkeypatch, await _container(ChatModels(None)), llm=False)

        with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
            await eval_cli.run(self._args(questions_file, extract_filters=True, agents=agents_dir))
        assert built == []  # fails before any boot work

    async def test_both_runs_are_reported_side_by_side_and_uncovered_profiles_stay_plain(
        self,
        monkeypatch: pytest.MonkeyPatch,
        questions_file: Path,
        agents_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        model = ScriptedChatModel(structured=[_reply("letter")])
        built = self._patch(monkeypatch, await _container(chat_models_for(model)))

        report = await eval_cli.run(
            self._args(questions_file, extract_filters=True, agents=agents_dir)
        )

        mode = report["modes"][DENSE_RETRIEVAL_MODE]
        assert report["extract_filters"] is True
        assert report["filter_input"] == "question"
        assert report["filter_extraction"] == {"other": "skipped (no topic)", "p": "extracted"}
        assert report["filters"] == {"m": [_clause("letter")]}  # nothing for `other`
        assert mode["overall"]["questions"] == 2  # the plain run is still reported as before
        rows = {r["id"]: r for r in mode["with_filters"]["questions"]}
        assert (rows["m"]["filter_level"], rows["m"]["filter_extraction"]) == ("full", "extracted")
        assert rows["o"]["filter_extraction"] == "skipped (no topic)"
        assert rows["o"]["filters"] == []
        out = capsys.readouterr().out
        assert "plain vs filters" in out
        assert "other: skipped (no topic)" in out
        assert "m: kind eq 'letter'" in out
        assert clauses_to_filter(report["filters"]["m"]) == Filter.eq("kind", "letter")
        assert len(model.structured_seen) == 1  # extracted once, not once per mode
        assert built[0].agents.dir is None
        assert report["analyzer_model"] == {"agent": "default (openai:analyzer)"}

    async def test_the_agent_runs_and_reports_the_profile_its_spec_binds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        questions_file: Path,
        agents_dir: Path,
    ) -> None:
        (agents_dir / "a.yaml").write_text(
            "id: agent\ndescription: d\nmodels: {analyzer: fast}\n"
            "topics: [{name: t, description: d, profile: p, filters: {kind: HINT}}]\n",
            encoding="utf-8",
        )
        fast, default = ScriptedChatModel(structured=[_reply("letter")]), ScriptedChatModel()
        catalog = ModelCatalog(
            LlmConfig(provider="openai", models={role: "base" for role in ChatRole}),
            {"fast": LlmProfile("anthropic", "claude-fast", api_key="k")},
            builder=lambda model, **_kw: fast if model == "claude-fast" else default,
        )
        self._patch(monkeypatch, await _container(ChatModels(catalog)))
        # No LLM_PROVIDER at all: the profile bound to the analyzer is enough for the guard.
        profiles_file = agents_dir / "llm" / "profiles.yaml"
        profiles_file.parent.mkdir()
        profiles_file.write_text(
            "profiles:\n  fast: {provider: anthropic, model: claude-fast, api_key_env: FAST_KEY}\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("FAST_KEY", "k")
        settings = make_settings(llm={"profiles_file": profiles_file})
        monkeypatch.setattr(eval_cli, "get_settings", lambda: settings)

        report = await eval_cli.run(
            self._args(questions_file, extract_filters=True, agents=agents_dir)
        )

        assert report["analyzer_model"] == {"agent": "fast (anthropic:claude-fast)"}
        assert len(fast.structured_seen) == 1  # the extractor used the agent's own profile
        assert default.structured_seen == []

    async def test_analyze_records_the_filter_input_and_the_standalone_queries(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        model = ScriptedChatModel(
            structured=[
                {
                    "standalone_query": "letters about 7",
                    "is_clear": True,
                    "confidence": 1,
                    "request_kind": "domain",
                },
                _reply("letter"),
            ]
        )
        self._patch(monkeypatch, await _container(chat_models_for(model)))

        report = await eval_cli.run(
            self._args(questions_file, extract_filters=True, agents=agents_dir, analyze=True)
        )

        assert report["filter_input"] == "standalone_query"
        assert report["standalone_queries"] == {"m": "letters about 7"}
        assert model.structured_seen[1][1].content == "letters about 7"

    async def test_two_specs_on_one_profile_abort(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        (agents_dir / "b.yaml").write_text(
            "id: agent-two\ndescription: d\ntopics: [{name: t, description: d, profile: p}]\n",
            encoding="utf-8",
        )
        self._patch(monkeypatch, await _container(chat_models_for()))

        with pytest.raises(UsageError, match="both cover profile"):
            await eval_cli.run(self._args(questions_file, extract_filters=True, agents=agents_dir))
