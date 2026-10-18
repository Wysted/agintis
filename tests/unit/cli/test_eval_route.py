"""`cli.eval --route`: router accuracy and retrieval restricted to the routed topics."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agintis.agent_kit.nodes import RoutedTopic, RouteScope, TopicEvidence
from agintis.bootstrap.container import Container, StartupError
from agintis.cli import eval as eval_cli
from agintis.cli._common import UsageError
from agintis.cli.eval import (
    apply_router_override,
    build_parser,
    build_routers,
    check_usage,
    evaluate,
    evaluate_routed,
    evidence_summary,
    route_questions,
    routed_cut,
    router_cost,
    router_summary,
    topics_by_profile,
)
from agintis.cli.eval_metrics import EvalQuestion
from agintis.modules.agents.rag_chat import AgentSpec, parse_agent_spec
from agintis.modules.knowledge.api import (
    BM25_HYBRID_RETRIEVAL_MODE,
    DENSE_RETRIEVAL_MODE,
    HYBRID_RETRIEVAL_MODE,
    ScoredChunk,
    TextSearchConfigError,
)
from agintis.platform.llm import ChatModels
from agintis.shared_kernel.ids import TenantId
from tests.fakes.llm import ScriptedChatModel, chat_models_for
from tests.support.settings import make_settings
from tests.unit.cli.test_eval_filters import _container, _question

TENANT = TenantId("t")


def _spec(agent_id: str = "agent", **changes: Any) -> AgentSpec:
    """Two topics: `t_p` over the profile that holds the documents, `t_o` over an empty one."""
    return parse_agent_spec(
        {
            "id": agent_id,
            "description": "d",
            "topics": [
                {"name": "t_p", "description": "documents", "profile": "p"},
                {"name": "t_o", "description": "nothing here", "profile": "other"},
            ],
            **changes,
        },
        origin="t",
    )


def _routed(*pairs: tuple[str, float]) -> list[RoutedTopic]:
    return [RoutedTopic(topic=t, confidence=c) for t, c in pairs]


def _choice(topic: str, confidence: float) -> dict[str, Any]:
    return {"topic": topic, "confidence": confidence}


class _FixedRouter:
    def __init__(self, routes: dict[str, list[RoutedTopic]]) -> None:
        self._routes = routes

    async def route(
        self, query: str, scope: RouteScope | None = None, *, widen_to: int | None = None
    ) -> list[RoutedTopic]:
        return self._routes[query]


def _questions() -> list[EvalQuestion]:
    return [
        _question("m", "paragraph 7"),
        EvalQuestion("o", "orphan question", "x", "other", "orphan question", {}),
    ]


class TestUsage:
    def _args(self, **changes: Any) -> argparse.Namespace:
        base = {"extract_filters": False, "agents": None, "analyze": False, "route": False}
        return argparse.Namespace(**{**base, **changes})

    def test_the_flag_is_opt_in(self) -> None:
        parser = build_parser()
        assert parser.parse_args(["--profiles", "p", "--questions", "q"]).route is False
        assert parser.parse_args(["--profiles", "p", "--questions", "q", "--route"]).route is True

    def test_route_needs_agents(self) -> None:
        with pytest.raises(UsageError, match="--route needs --agents"):
            check_usage(self._args(route=True))

    def test_route_with_agents_is_valid_and_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        check_usage(self._args(route=True, agents=Path("a")))
        assert caplog.text == ""


class TestTopicBindings:
    def test_every_topic_of_every_spec_is_bound_by_profile(self) -> None:
        topics = topics_by_profile([_spec()])

        assert {name: b.topic.name for name, b in topics.items()} == {"p": "t_p", "other": "t_o"}
        assert {b.spec.id for b in topics.values()} == {"agent"}

    def test_two_specs_covering_a_profile_are_ambiguous(self) -> None:
        with pytest.raises(UsageError, match="both cover profile"):
            topics_by_profile([_spec("one"), _spec("two")])


class TestRouting:
    async def test_the_router_runs_once_per_covered_question_with_the_agents_own_router(
        self,
    ) -> None:
        model = ScriptedChatModel(
            structured=[
                {"topics": [_choice("t_o", 0.9), _choice("t_p", 0.8)]},
                {"topics": [_choice("t_o", 0.9)]},
            ]
        )
        container = await _container(chat_models_for(model))
        topics = topics_by_profile([_spec()])

        routers = build_routers(container, topics, ["p", "other"])
        routes = await route_questions(_questions(), topics, routers)

        assert list(routers) == ["agent"]
        assert routes == {
            "m": _routed(("t_o", 0.9), ("t_p", 0.8)),
            "o": _routed(("t_o", 0.9)),
        }
        assert [m[-1].content for m in model.structured_seen] == [
            "paragraph 7",
            "orphan question",
        ]

    async def test_uncovered_profiles_are_not_routed(self) -> None:
        topics = topics_by_profile([_spec()])
        questions = [EvalQuestion("x", "q", "t", "uncovered", "s", {})]

        routes = await route_questions(questions, topics, {"agent": _FixedRouter({})})

        assert routes == {}
        assert router_summary(questions, topics, routes)["questions"] == 0

    def test_router_accuracy_hit_at_1_and_any_selected(self) -> None:
        topics = topics_by_profile([_spec()])
        questions = [
            _question("m", "paragraph 7"),
            EvalQuestion("o", "q", "x", "other", "s", {}),
            EvalQuestion("o2", "q", "x", "other", "s", {}),
        ]
        routes = {
            "m": _routed(("t_p", 0.9)),
            "o": _routed(("t_p", 0.9), ("t_o", 0.7)),
            "o2": _routed(("t_p", 0.3)),
        }

        summary = router_summary(questions, topics, routes)

        assert summary["questions"] == 3
        assert summary["hit_at_1"] == pytest.approx(1 / 3)  # only m
        assert summary["any_selected"] == pytest.approx(2 / 3)  # m and o
        assert summary["mean_topics_selected"] == pytest.approx(4 / 3)
        assert [
            (r["id"], r["expected"], r["hit_at_1"], r["any_selected"]) for r in summary["rows"]
        ] == [
            ("m", "t_p", True, True),
            ("o", "t_o", False, True),
            ("o2", "t_o", False, False),
        ]


class TestRoutedRetrieval:
    async def _run(
        self, routes: dict[str, list[RoutedTopic]], spec: AgentSpec | None = None
    ) -> Any:
        container = await _container(chat_models_for())
        topics = topics_by_profile([spec or _spec()])
        return await evaluate_routed(
            container,
            _questions(),
            topics,
            routes,
            mode=DENSE_RETRIEVAL_MODE,
            ks=(1, 5),
            tenant=TENANT,
            rerank=False,
        )

    async def test_a_right_route_retrieves_like_the_per_profile_oracle(self) -> None:
        container = await _container(chat_models_for())
        oracle = await evaluate(
            container,
            _questions()[:1],
            mode=DENSE_RETRIEVAL_MODE,
            ks=(1, 5),
            tenant=TENANT,
            rerank=False,
        )

        (routed, _) = await self._run({"m": _routed(("t_p", 0.9)), "o": _routed(("t_o", 0.9))})

        assert (routed.rank, routed.retrieved) == (oracle[0].rank, oracle[0].retrieved)
        assert routed.rank is not None

    async def test_a_wrong_route_misses_even_though_the_oracle_finds_it(self) -> None:
        wrong, _ = await self._run({"m": _routed(("t_o", 0.9)), "o": _routed(("t_o", 0.9))})
        assert wrong.rank is None
        assert wrong.retrieved == 0

    async def test_routed_topics_are_merged_and_the_hit_keeps_a_rank(self) -> None:
        both, _ = await self._run(
            {"m": _routed(("t_o", 0.9), ("t_p", 0.8)), "o": _routed(("t_o", 0.9))}
        )
        assert both.rank is not None  # the empty topic does not hide the other's hits

    async def test_hits_are_capped_at_the_largest_k(self) -> None:
        both, _ = await self._run({"m": _routed(("t_p", 0.9)), "o": _routed(("t_o", 0.9))})
        assert both.retrieved <= 5

    async def test_a_question_that_was_not_routed_runs_on_its_own_profile(self) -> None:
        container = await _container(chat_models_for())
        topics = topics_by_profile([_spec()])

        (outcome,) = await evaluate_routed(
            container,
            _questions()[:1],
            topics,
            {},  # not routed
            mode=DENSE_RETRIEVAL_MODE,
            ks=(1, 5),
            tenant=TENANT,
            rerank=False,
        )

        assert outcome.rank is not None


class _RunHelpers:
    @pytest.fixture
    def questions_file(self, tmp_path: Path) -> Path:
        path = tmp_path / "q.jsonl"
        rows = [
            '{"id": "m", "question": "paragraph 7", "type": "x", "profile": "p", '
            '"expected_snippet": "paragraph 7", "locator": {}}',
            '{"id": "o", "question": "orphan question", "type": "x", "profile": "other", '
            '"expected_snippet": "orphan question", "locator": {}}',
        ]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

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

    @pytest.fixture
    def agents_dir(self, tmp_path: Path) -> Path:
        directory = tmp_path / "agents"
        directory.mkdir()
        (directory / "a.yaml").write_text(
            "id: agent\ndescription: d\ntopics:\n"
            "  - {name: t_p, description: documents, profile: p}\n"
            "  - {name: t_o, description: other documents, profile: other}\n",
            encoding="utf-8",
        )
        return directory


class TestModeSets(_RunHelpers):
    """`all` (the default) adds bm25_hybrid only where pg_textsearch and the tenant index exist;
    an explicit bm25_hybrid fails instead."""

    async def _container(self, *, bm25: bool) -> Container:
        container = await _container(ChatModels(None))

        async def require_pg_textsearch() -> None:
            if not bm25:
                raise StartupError("retrieval_mode bm25_hybrid needs the pg_textsearch extension")

        container.require_pg_textsearch = require_pg_textsearch  # type: ignore[method-assign]
        return container

    async def test_all_runs_every_mode_where_pg_textsearch_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path
    ) -> None:
        self._patch(monkeypatch, await self._container(bm25=True), llm=False)
        report = await eval_cli.run(self._args(questions_file, mode="all"))
        assert list(report["modes"]) == [
            DENSE_RETRIEVAL_MODE,
            HYBRID_RETRIEVAL_MODE,
            BM25_HYBRID_RETRIEVAL_MODE,
        ]
        assert "skipped_modes" not in report

    async def test_all_skips_bm25_hybrid_without_pg_textsearch_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path
    ) -> None:
        self._patch(monkeypatch, await self._container(bm25=False), llm=False)
        report = await eval_cli.run(self._args(questions_file, mode="all"))
        assert list(report["modes"]) == [DENSE_RETRIEVAL_MODE, HYBRID_RETRIEVAL_MODE]
        assert "pg_textsearch" in report["skipped_modes"][BM25_HYBRID_RETRIEVAL_MODE]

    async def test_all_skips_bm25_hybrid_when_the_tenant_has_no_bm25_index(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path
    ) -> None:
        """pg_textsearch is there, but the tenant was ingested while no profile used bm25_hybrid."""
        container = await self._container(bm25=True)

        async def unindexed(*args: object, **kwargs: object) -> list[ScoredChunk]:
            raise TextSearchConfigError("Tenant 'default' has chunks but no BM25 index")

        monkeypatch.setattr(container.chunks, "search_bm25", unindexed)
        self._patch(monkeypatch, container, llm=False)
        report = await eval_cli.run(self._args(questions_file, mode="all"))
        assert list(report["modes"]) == [DENSE_RETRIEVAL_MODE, HYBRID_RETRIEVAL_MODE]
        assert "no BM25 index" in report["skipped_modes"][BM25_HYBRID_RETRIEVAL_MODE]

    async def test_an_explicit_bm25_hybrid_fails_early_when_the_tenant_has_no_bm25_index(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path
    ) -> None:
        container = await self._container(bm25=True)
        bm25_calls: list[tuple[object, ...]] = []
        dense_calls: list[object] = []

        async def unindexed(*args: object, **kwargs: object) -> list[ScoredChunk]:
            bm25_calls.append(args)
            raise TextSearchConfigError("Tenant 't' has chunks but no BM25 index")

        async def dense(*args: object, **kwargs: object) -> list[ScoredChunk]:
            dense_calls.append(args)
            return []

        monkeypatch.setattr(container.chunks, "search_bm25", unindexed)
        monkeypatch.setattr(container.chunks, "search", dense)
        self._patch(monkeypatch, container, llm=False)
        with pytest.raises(TextSearchConfigError, match="no BM25 index"):
            await eval_cli.run(self._args(questions_file, mode=BM25_HYBRID_RETRIEVAL_MODE))
        assert bm25_calls == [(TenantId("t"), "probe", None, 1)]  # the probe, before any search
        assert dense_calls == []

    async def test_an_explicit_bm25_hybrid_fails_without_pg_textsearch(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path
    ) -> None:
        self._patch(monkeypatch, await self._container(bm25=False), llm=False)
        with pytest.raises(StartupError, match="pg_textsearch"):
            await eval_cli.run(self._args(questions_file, mode=BM25_HYBRID_RETRIEVAL_MODE))

    async def test_both_never_needs_pg_textsearch(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path
    ) -> None:
        self._patch(monkeypatch, await self._container(bm25=False), llm=False)
        report = await eval_cli.run(self._args(questions_file, mode="both"))
        assert list(report["modes"]) == [DENSE_RETRIEVAL_MODE, HYBRID_RETRIEVAL_MODE]


class TestRun(_RunHelpers):
    async def test_route_reports_accuracy_and_the_routed_retrieval_next_to_the_oracle(
        self,
        monkeypatch: pytest.MonkeyPatch,
        questions_file: Path,
        agents_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        model = ScriptedChatModel(
            structured=[
                {"topics": [_choice("t_o", 0.9), _choice("t_p", 0.8)]},
                {"topics": [_choice("t_o", 0.9)]},
            ]
        )
        built = self._patch(monkeypatch, await _container(chat_models_for(model)))

        report = await eval_cli.run(self._args(questions_file, route=True, agents=agents_dir))

        route = report["route"]
        assert route["questions"] == 2
        assert route["hit_at_1"] == pytest.approx(0.5)  # `m` routed t_o first
        assert route["any_selected"] == 1.0
        assert route["status"] == {"other": "routed", "p": "routed"}
        mode = report["modes"][DENSE_RETRIEVAL_MODE]
        assert set(mode) == {"overall", "profiles", "questions", "routed"}
        assert mode["routed"]["overall"]["questions"] == 2
        rows = {r["id"]: r for r in mode["routed"]["questions"]}
        assert rows["m"]["rank"] is not None
        assert len(model.structured_seen) == 2  # one router call per question, not per mode
        out = capsys.readouterr().out
        assert "ROUTED" in out
        assert "per-profile oracle vs routed" in out
        assert "routed" in out
        assert built[0].agents.dir is None
        assert "extract_filters" not in report

    async def test_without_the_flag_the_report_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        self._patch(monkeypatch, await _container(ChatModels(None)), llm=False)

        report = await eval_cli.run(self._args(questions_file))

        assert "route" not in report
        assert set(report["modes"][DENSE_RETRIEVAL_MODE]) == {"overall", "profiles", "questions"}

    async def test_the_llm_router_needs_the_llm_before_the_app_boots(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        built = self._patch(monkeypatch, await _container(ChatModels(None)), llm=False)

        with pytest.raises(RuntimeError, match=r"--route needs the LLM.*agent"):
            await eval_cli.run(self._args(questions_file, route=True, agents=agents_dir))
        assert built == []

    async def test_single_topic_agents_need_no_llm(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        (agents_dir / "a.yaml").write_text(
            "id: agent\ndescription: d\ntopics:\n  - {name: t_p, description: d, profile: p}\n",
            encoding="utf-8",
        )
        self._patch(monkeypatch, await _container(ChatModels(None)), llm=False)

        report = await eval_cli.run(self._args(questions_file, route=True, agents=agents_dir))

        assert report["route"]["questions"] == 1  # `other` is not covered
        assert report["route"]["hit_at_1"] == 1.0
        assert report["route"]["status"] == {"other": "skipped (no topic)", "p": "routed"}
        assert report["modes"][DENSE_RETRIEVAL_MODE]["routed"]["overall"]["questions"] == 2


class TestSummaryAndCut:
    def test_the_summary_reports_confidences_and_secondary_selections(self) -> None:
        topics = topics_by_profile([_spec()])
        questions = [_question("m", "x"), EvalQuestion("o", "y", "t", "other", "s", {})]
        routes = {
            "m": _routed(("t_p", 0.9), ("t_o", 0.7)),
            "o": _routed(("t_p", 0.5)),  # wrong, and unsure
        }

        summary = router_summary(questions, topics, routes)

        assert summary["with_secondary"] == 1
        assert summary["confidence_when_right"] == pytest.approx(0.9)
        assert summary["confidence_when_wrong"] == pytest.approx(0.5)
        rows = {r["id"]: r for r in summary["rows"]}
        assert rows["m"]["confidences"] == [0.9, 0.7]
        assert rows["m"]["secondary"] == ["t_o"]
        assert rows["o"]["secondary"] == []

    async def test_the_routed_list_is_cut_at_the_agents_max_sources(self) -> None:
        container = await _container(chat_models_for())
        spec = _spec(max_sources=3)
        topics = topics_by_profile([spec])

        (outcome, _) = await evaluate_routed(
            container,
            _questions(),
            topics,
            {"m": _routed(("t_p", 0.9)), "o": _routed(("t_o", 0.9))},
            mode=DENSE_RETRIEVAL_MODE,
            ks=(1, 5),
            tenant=TENANT,
            rerank=False,
        )

        assert routed_cut(spec, (1, 5)) == 3
        assert outcome.retrieved == 3  # five were retrieved, the agent would show three
        assert routed_cut(_spec(), (1, 5)) == 5  # never deeper than the largest k
        assert routed_cut(_spec(), (1, 20)) == 8  # default max_sources


class TestRunReuseAndAnalyze(_RunHelpers):
    async def test_the_per_profile_oracle_is_reused_not_repeated(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        questions, agents, helper = questions_file, agents_dir, self
        model = ScriptedChatModel(
            structured=[
                {"topics": [_choice("t_o", 0.9), _choice("t_p", 0.8)]},
                {"topics": [_choice("t_o", 0.9)]},
            ]
        )
        container = await _container(chat_models_for(model))
        calls: list[str] = []
        real = container.retrieve_for_topic.execute

        async def spy(tenant: Any, profile: Any, query: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append(profile.name)
            return await real(tenant, profile, query, *args, **kwargs)

        container.retrieve_for_topic.execute = spy  # type: ignore[method-assign,assignment]
        helper._patch(monkeypatch, container)

        report = await eval_cli.run(helper._args(questions, route=True, agents=agents))

        # 2 oracle searches (one per question) + 3 routed (m -> 2 topics, o -> 1): no 2nd oracle.
        assert len(calls) == 5
        assert report["route"]["retrieval_cut"] == {"agent": 5}
        assert report["route"]["route_input"] == "question"

    async def test_analyze_routes_on_the_standalone_query(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        questions, agents, helper = questions_file, agents_dir, self
        analysis = {"is_clear": True, "confidence": 1, "request_kind": "domain"}
        model = ScriptedChatModel(
            structured=[
                {"standalone_query": "standalone m", **analysis},
                {"standalone_query": "standalone o", **analysis},
                {"topics": [_choice("t_p", 0.9)]},
                {"topics": [_choice("t_o", 0.9)]},
            ]
        )
        helper._patch(monkeypatch, await _container(chat_models_for(model)))

        report = await eval_cli.run(
            helper._args(questions, route=True, analyze=True, agents=agents)
        )

        assert report["route"]["route_input"] == "standalone_query"
        assert report["route"]["hit_at_1"] == 1.0
        router_inputs = [seen[-1].content for seen in model.structured_seen[2:]]
        assert router_inputs == ["standalone m", "standalone o"]

    def test_analyze_is_valid_with_route_alone(self) -> None:
        args = argparse.Namespace(extract_filters=False, agents=Path("a"), analyze=True, route=True)
        check_usage(args)


def _evidence(**scores: float | None) -> list[TopicEvidence]:
    return [
        TopicEvidence(topic, () if score is None else (score,), score)
        for topic, score in scores.items()
    ]


class TestRouterOverride:
    def test_the_flag_defaults_to_the_specs_own_router(self) -> None:
        parser = build_parser()
        assert parser.parse_args(["--profiles", "p", "--questions", "q"]).router == "spec"
        args = ["--profiles", "p", "--questions", "q", "--route", "--router", "evidence"]
        assert parser.parse_args(args).router == "evidence"
        with pytest.raises(SystemExit):
            parser.parse_args(["--profiles", "p", "--questions", "q", "--router", "magic"])

    def test_it_needs_route(self) -> None:
        args = argparse.Namespace(
            extract_filters=False, agents=Path("a"), analyze=False, route=False, router="llm"
        )
        with pytest.raises(UsageError, match="--router needs --route"):
            check_usage(args)

    def test_it_forces_the_router_of_multi_topic_agents_only(self) -> None:
        single = parse_agent_spec(
            {"id": "one", "description": "d", "topics": [_spec().model_dump()["topics"][0]]},
            origin="t",
        )
        specs = [_spec(), single]

        assert [s.router for s in apply_router_override(specs, "spec")] == ["auto", "auto"]
        evidence = apply_router_override(specs, "evidence")
        assert [s.router for s in evidence] == ["evidence", "auto"]
        assert evidence[0].uses_evidence_router
        assert not evidence[0].uses_llm_router
        llm = apply_router_override([_spec(router="evidence")], "llm")
        assert llm[0].uses_llm_router
        assert not llm[0].uses_evidence_router


class TestEvidenceRouting:
    async def test_the_evidence_router_routes_without_a_model_and_keeps_every_score(self) -> None:
        container = await _container(ChatModels(None))
        topics = topics_by_profile(apply_router_override([_spec()], "evidence"))
        routers = build_routers(container, topics, ["p", "other"])
        scopes: dict[str, RouteScope] = {}
        evidence: dict[str, list[TopicEvidence]] = {}

        routes = await route_questions(
            _questions(), topics, routers, tenant=TENANT, scopes=scopes, evidence=evidence
        )

        assert routes["m"][0]["topic"] == "t_p"  # only `p` holds documents
        assert [r["topic"] for r in routes["m"]] == ["t_p"]
        assert [e.topic for e in evidence["m"]] == ["t_p", "t_o"]  # every topic, spec order
        assert evidence["m"][1].score is None  # `other` is empty
        assert scopes["m"].vectors.get("paragraph 7") is not None  # embedded once, kept

    async def test_the_summary_separates_right_from_wrong_and_sweeps_the_margin(self) -> None:
        topics = topics_by_profile([_spec(max_topics=2)])
        questions = [
            _question("r1", "x"),
            _question("r2", "y"),
            EvalQuestion("w", "z", "t", "other", "s", {}),  # expected t_o, evidence prefers t_p
            EvalQuestion("n", "n", "t", "other", "s", {}),  # expected t_o has no hit at all
        ]
        evidence = {
            "r1": _evidence(t_p=0.80, t_o=0.60),
            "r2": _evidence(t_p=0.70, t_o=0.69),
            "w": _evidence(t_p=0.75, t_o=0.72),
            "n": _evidence(t_p=0.50, t_o=None),
        }

        summary = evidence_summary(questions, topics, evidence, margins=(0.0, 0.03, 0.2))

        assert summary["best_score_when_right"]["n"] == 2
        assert summary["best_score_when_right"]["mean"] == pytest.approx(0.75)
        assert summary["best_score_when_wrong"]["n"] == 2
        assert summary["runner_up_gap_when_right"]["min"] == pytest.approx(0.01)
        assert summary["gap_to_best_when_wrong"]["n"] == 1  # `n` has no expected score
        assert summary["gap_to_best_when_wrong"]["max"] == pytest.approx(0.03)
        rows = {r["id"]: r for r in summary["rows"]}
        assert rows["w"]["expected_rank"] == 2
        assert rows["w"]["gap_to_best"] == pytest.approx(0.03)
        assert rows["n"]["expected_rank"] is None
        assert rows["n"]["gap_to_best"] is None
        assert rows["r1"]["scores"] == {"t_p": 0.80, "t_o": 0.60}
        sweep = {p["margin"]: p for p in summary["margin_sweep"]}
        # margin 0: only the primary; the expected topic is first for r1 and r2 only.
        assert sweep[0.0]["any_selected"] == pytest.approx(2 / 4)
        assert sweep[0.0]["mean_topics"] == pytest.approx(1.0)
        # margin 0.03 also rescues `w` (gap 0.03) and keeps r2's runner-up.
        assert sweep[0.03]["any_selected"] == pytest.approx(3 / 4)
        assert sweep[0.2]["mean_topics"] == pytest.approx((2 + 2 + 2 + 1) / 4)

    def test_the_cost_counts_llm_calls_probes_and_time(self) -> None:
        llm = topics_by_profile([_spec()])
        evidence = topics_by_profile(apply_router_override([_spec()], "evidence"))
        questions = _questions()
        routes = {"m": _routed(("t_p", 1.0)), "o": _routed(("t_p", 1.0))}

        assert router_cost(questions, llm, routes, 1.23456) == {
            "llm_calls": 2,
            "probe_embeddings": 0,
            "probe_searches": 0,
            "seconds": 1.235,
        }
        assert router_cost(questions, evidence, routes, 0.5) == {
            "llm_calls": 0,
            "probe_embeddings": 2,
            "probe_searches": 4,  # 2 questions x 2 topics
            "seconds": 0.5,
        }
        assert router_cost(questions, evidence, {}, 0.0)["probe_searches"] == 0  # not routed


class TestRunWithTheEvidenceRouter(_RunHelpers):
    async def test_it_needs_no_llm_and_reports_the_router_cost_and_calibration(
        self,
        monkeypatch: pytest.MonkeyPatch,
        questions_file: Path,
        agents_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._patch(monkeypatch, await _container(ChatModels(None)), llm=False)

        report = await eval_cli.run(
            self._args(questions_file, route=True, router="evidence", agents=agents_dir)
        )

        route = report["route"]
        assert route["router"] == "evidence"
        assert route["router_kinds"] == {"agent": "evidence"}
        assert route["cost"]["llm_calls"] == 0
        assert (route["cost"]["probe_embeddings"], route["cost"]["probe_searches"]) == (2, 4)
        assert route["questions"] == 2
        assert route["evidence"]["margin_sweep"]
        assert {r["id"] for r in route["evidence"]["rows"]} == {"m", "o"}
        assert route["hit_at_1"] == pytest.approx(0.5)  # `o`: its profile is empty, `p` wins
        out = capsys.readouterr().out
        assert "router cost: 0 LLM calls, 2 probe embeddings, 4 probe searches" in out
        assert "evidence_margin" in out
        assert "ROUTED" in out

    async def test_the_routed_retrieval_reuses_the_probe_embedding(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        container = await _container(ChatModels(None))
        vectors: list[Any] = []
        real = container.retrieve_for_topic.execute

        async def spy(tenant: Any, profile: Any, query: Any, *args: Any, **kwargs: Any) -> Any:
            vectors.append(kwargs.get("query_vector"))
            return await real(tenant, profile, query, *args, **kwargs)

        container.retrieve_for_topic.execute = spy  # type: ignore[method-assign,assignment]
        self._patch(monkeypatch, container, llm=False)

        await eval_cli.run(
            self._args(questions_file, route=True, router="evidence", agents=agents_dir)
        )

        # 2 oracle searches (raw, embed themselves) + 2 routed ones (m -> t_p, o -> t_p) with it.
        assert vectors[:2] == [None, None]
        assert all(v is not None for v in vectors[2:])
        assert len(vectors) == 4

    async def test_forcing_the_llm_router_needs_the_llm_even_if_the_spec_says_evidence(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        (agents_dir / "a.yaml").write_text(
            (agents_dir / "a.yaml").read_text(encoding="utf-8") + "router: evidence\n",
            encoding="utf-8",
        )
        built = self._patch(monkeypatch, await _container(ChatModels(None)), llm=False)

        # The spec alone would run without a model.
        await eval_cli.run(self._args(questions_file, route=True, agents=agents_dir))
        built.clear()
        with pytest.raises(RuntimeError, match=r"--route needs the LLM.*agent"):
            await eval_cli.run(
                self._args(questions_file, route=True, router="llm", agents=agents_dir)
            )
        assert built == []

    async def test_the_llm_router_is_costed_in_calls(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        model = ScriptedChatModel(
            structured=[{"topics": [_choice("t_p", 0.9)]}, {"topics": [_choice("t_o", 0.9)]}]
        )
        self._patch(monkeypatch, await _container(chat_models_for(model)))

        report = await eval_cli.run(
            self._args(questions_file, route=True, router="llm", agents=agents_dir)
        )

        assert report["route"]["router"] == "llm"
        assert report["route"]["cost"]["llm_calls"] == 2
        assert report["route"]["cost"]["probe_searches"] == 0
        assert "evidence" not in report["route"]

    async def test_without_the_flag_nothing_changes_for_the_default_router(
        self, monkeypatch: pytest.MonkeyPatch, questions_file: Path, agents_dir: Path
    ) -> None:
        model = ScriptedChatModel(
            structured=[{"topics": [_choice("t_p", 0.9)]}, {"topics": [_choice("t_o", 0.9)]}]
        )
        self._patch(monkeypatch, await _container(chat_models_for(model)))

        report = await eval_cli.run(self._args(questions_file, route=True, agents=agents_dir))

        assert report["route"]["router"] == "spec"
        assert report["route"]["router_kinds"] == {"agent": "llm"}
        assert report["route"]["hit_at_1"] == 1.0


class TestCalibrationAndLabels:
    def test_each_topic_reports_its_score_when_expected_and_in_the_background(self) -> None:
        topics = topics_by_profile([_spec(max_topics=2)])
        questions = [_question("a", "x"), _question("b", "y"), _question("c", "z")]
        evidence = {
            "a": _evidence(t_p=0.80, t_o=0.60),
            "b": _evidence(t_p=0.70, t_o=0.50),
            "c": _evidence(t_p=0.90, t_o=None),
        }

        scores = evidence_summary(questions, topics, evidence)["topic_scores"]

        assert scores["t_p"]["when_expected"]["n"] == 3
        assert scores["t_p"]["when_expected"]["median"] == pytest.approx(0.80)
        assert scores["t_p"]["background"]["n"] == 0
        assert scores["t_o"]["when_expected"]["n"] == 0
        assert scores["t_o"]["background"]["n"] == 2  # no hit = no sample
        assert scores["t_o"]["background"]["max"] == pytest.approx(0.60)

    def test_scores_in_the_background_are_raw_not_biased(self) -> None:
        topics = topics_by_profile([_spec()])
        found = [TopicEvidence("t_p", (0.7,), 0.7, bias=0.05), TopicEvidence("t_o", (), None)]

        scores = evidence_summary([_question("a", "x")], topics, {"a": found})["topic_scores"]

        assert scores["t_p"]["when_expected"]["median"] == pytest.approx(0.70)

    def test_the_sweep_counts_the_fallback_topics_like_the_router_selects_them(self) -> None:
        topics = topics_by_profile([_spec(max_topics=2)])
        questions = [_question("a", "x")]  # expected t_p
        nothing = {"a": _evidence(t_p=None, t_o=None)}

        (point,) = evidence_summary(questions, topics, nothing, margins=(0.0,))["margin_sweep"]

        # Router: no evidence -> fallback_topics (all topics up to max_topics).
        assert point["mean_topics"] == pytest.approx(2.0)
        assert point["any_selected"] == pytest.approx(1.0)

    def test_the_cost_counts_no_embedding_for_a_blank_query(self) -> None:
        evidence = topics_by_profile(apply_router_override([_spec()], "evidence"))
        questions = _questions()
        routes = {"m": _routed(("t_p", 1.0)), "o": _routed(("t_p", 1.0))}

        cost = router_cost(questions, evidence, routes, 0.0, {"o": "   "})

        assert (cost["probe_embeddings"], cost["probe_searches"]) == (1, 2)

    def test_the_override_is_a_validated_spec_and_does_not_fake_explicit_knobs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            (evidence,) = apply_router_override([_spec()], "evidence")
            (llm,) = apply_router_override([evidence], "llm")

        assert caplog.text == ""  # no knob was written, so none is "ignored"
        assert evidence.router == "evidence"
        assert llm.router == "llm"
        assert [t.name for t in llm.topics] == ["t_p", "t_o"]


class TestRunLabels(_RunHelpers):
    async def test_the_evidence_report_has_no_router_model_and_labels_cosine_scores(
        self,
        monkeypatch: pytest.MonkeyPatch,
        questions_file: Path,
        agents_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._patch(monkeypatch, await _container(ChatModels(None)), llm=False)

        report = await eval_cli.run(
            self._args(questions_file, route=True, router="evidence", agents=agents_dir)
        )

        assert "router_model" not in report["route"]
        assert report["route"]["confidence_kind"] == "cosine_score"
        out = capsys.readouterr().out
        assert "primary cosine score" in out
        assert "per topic score" in out
        assert "primary confidence" not in out

    async def test_the_llm_report_keeps_the_model_and_the_confidence_label(
        self,
        monkeypatch: pytest.MonkeyPatch,
        questions_file: Path,
        agents_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        model = ScriptedChatModel(
            structured=[{"topics": [_choice("t_p", 0.9)]}, {"topics": [_choice("t_o", 0.9)]}]
        )
        self._patch(monkeypatch, await _container(chat_models_for(model)))

        report = await eval_cli.run(self._args(questions_file, route=True, agents=agents_dir))

        assert report["route"]["router_model"] == {"agent": "default (openai:router)"}
        assert report["route"]["confidence_kind"] == "model_confidence"
        assert "primary confidence" in capsys.readouterr().out
