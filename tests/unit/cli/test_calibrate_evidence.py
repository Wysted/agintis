"""Evidence calibration: statistics, the pre-registered threshold rule, scoring through the
real EvidenceTopicRouter over a fake probe, file loading and the report (no network)."""

import json
import random
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.messages import AnyMessage

from agintis.agent_kit.nodes import EvidenceTopic, EvidenceTopicRouter
from agintis.bootstrap.container import Container
from agintis.cli import calibrate_evidence as cli
from agintis.cli._common import EXIT_ABORTED, EXIT_OK, EXIT_USAGE, UsageError
from agintis.cli.calibrate_evidence import (
    LabelledQuery,
    agent_chat_models,
    analyze_queries,
    build_report,
    check_analyzer_available,
    check_report_path,
    load_labelled,
    pick_spec,
    profiles_digest,
    score_queries,
)
from agintis.cli.calibrate_stats import (
    ScoredQuery,
    auroc,
    calibrate,
    category_of,
    clopper_pearson,
    distribution,
    quantile,
    quantile_interval,
)
from agintis.modules.agents.rag_chat import AgentSpec, parse_agent_spec
from agintis.modules.knowledge.api import RagProfile
from agintis.platform.llm import (
    ChatModels,
    ChatRole,
    LlmConfig,
    LlmError,
    LlmProfile,
    ModelCatalog,
)
from agintis.platform.usage import Purpose
from agintis.shared_kernel.ids import TenantId
from tests.fakes.llm import NamedModel, ScriptedChatModel
from tests.fakes.retrieval import FakeTopicProbe
from tests.support.settings import make_settings
from tests.unit.cli.conftest import FakeLedger

TENANT = TenantId("t")


class TestQuantile:
    def test_linear_interpolation_between_ranks(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert quantile(values, 0.0) == 1.0
        assert quantile(values, 1.0) == 4.0
        assert quantile(values, 0.5) == pytest.approx(2.5)
        assert quantile(values, 0.25) == pytest.approx(1.75)

    def test_a_single_value_is_every_quantile(self) -> None:
        assert quantile([0.3], 0.95) == 0.3

    def test_invalid_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            quantile([], 0.5)
        with pytest.raises(ValueError, match="within"):
            quantile([1.0], 1.5)

    def test_distribution_reports_the_standard_quantiles(self) -> None:
        stats = distribution([float(i) for i in range(101)])
        assert stats["n"] == 101
        assert (stats["min"], stats["max"], stats["mean"]) == (0.0, 100.0, 50.0)
        assert [stats[k] for k in ("p5", "p25", "p50", "p75", "p95", "p99")] == pytest.approx(
            [5.0, 25.0, 50.0, 75.0, 95.0, 99.0]
        )

    def test_matches_statistics_quantiles_inclusive(self) -> None:
        rng = random.Random(7)
        for size in (2, 5, 80, 101):
            values = sorted(rng.random() for _ in range(size))
            expected = statistics.quantiles(values, n=100, method="inclusive")
            for k in range(1, 100):
                assert quantile(values, k / 100) == pytest.approx(expected[k - 1])


class TestIntervals:
    def test_clopper_pearson_known_values(self) -> None:
        low, high = clopper_pearson(0, 10)
        assert low == 0.0
        assert high == pytest.approx(0.3085, abs=1e-3)
        low, high = clopper_pearson(10, 10)
        assert low == pytest.approx(0.6915, abs=1e-3)
        assert high == 1.0
        low, high = clopper_pearson(5, 10)
        assert low == pytest.approx(0.1871, abs=1e-3)
        assert high == pytest.approx(0.8129, abs=1e-3)
        low, high = clopper_pearson(4, 80)
        assert low == pytest.approx(0.0138, abs=1e-3)
        assert high == pytest.approx(0.1232, abs=1e-3)

    def test_clopper_pearson_rejects_impossible_counts(self) -> None:
        with pytest.raises(ValueError, match="successes"):
            clopper_pearson(3, 2)

    def test_p95_interval_of_80_values_is_bounded_by_the_order_statistics(self) -> None:
        values = [float(i) for i in range(1, 81)]  # value == rank
        interval = quantile_interval(values, 0.95)
        assert interval["upper_rank"] == 80
        assert interval["upper"] == 80.0
        assert interval["lower"] == float(interval["lower_rank"])
        assert 70 <= interval["lower_rank"] <= 76
        assert interval["coverage"] >= 0.95

    def test_the_median_interval_is_centred(self) -> None:
        values = [float(i) for i in range(1, 81)]
        interval = quantile_interval(values, 0.5)
        assert interval["lower_rank"] + interval["upper_rank"] == 81  # symmetric around n/2
        assert interval["coverage"] >= 0.95

    def test_a_side_the_sample_cannot_bound_is_none(self) -> None:
        interval = quantile_interval([0.1, 0.2, 0.3, 0.4, 0.5], 0.95)
        assert interval["upper"] is None  # even the max cannot bound a p95 with 5 values
        assert interval["lower"] == 0.4
        assert interval["coverage"] == pytest.approx(0.9774, abs=1e-3)  # one-sided

    def test_an_empty_sample_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            quantile_interval([], 0.5)

    def test_category_comes_from_the_id(self) -> None:
        assert category_of("out-pens-01") == "pens"
        assert category_of("in-ncg-20") == "ncg"
        assert category_of("solo") == "solo"


class TestAuroc:
    def test_perfect_and_inverted_separation(self) -> None:
        assert auroc([0.8, 0.9], [0.1, 0.2]) == 1.0
        assert auroc([0.1, 0.2], [0.8, 0.9]) == 0.0

    def test_ties_count_half(self) -> None:
        assert auroc([0.5], [0.5]) == 0.5
        assert auroc([0.5, 0.9], [0.5, 0.1]) == pytest.approx(0.875)

    def test_matches_the_pairwise_definition(self) -> None:
        pos, neg = [0.3, 0.6, 0.6, 0.9], [0.2, 0.6, 0.7]
        pairs = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
        assert auroc(pos, neg) == pytest.approx(sum(pairs) / len(pairs))

    def test_needs_both_classes(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            auroc([], [0.1])


def _q(qid: str, label: str, score: float) -> ScoredQuery:
    return ScoredQuery(qid, f"question {qid}", label, score, "topic")


class TestThresholdRule:
    def _queries(self) -> list[ScoredQuery]:
        outs = [_q(f"o{i}", "out", i / 100) for i in range(101)]  # 0.00 .. 1.00
        ins = [_q("i-low", "in", 0.50), _q("i-edge", "in", 0.95), _q("i-hi", "in", 0.99)]
        return outs + ins

    def test_threshold_is_p95_of_out_and_ignores_in(self) -> None:
        report = calibrate(self._queries())
        assert report["threshold"] == pytest.approx(0.95)
        shifted = calibrate([q for q in self._queries() if q.label == "out"] + [_q("i", "in", 5.0)])
        assert shifted["threshold"] == pytest.approx(0.95)

    def test_recall_and_false_in_rate_use_a_strict_comparison(self) -> None:
        report = calibrate(self._queries())
        # only i-hi is strictly above 0.95 (i-edge sits ON it)
        assert report["in_recall"] == pytest.approx(1 / 3)
        assert [r["id"] for r in report["in_below"]] == ["i-low", "i-edge"]
        assert [r["id"] for r in report["out_above"]] == ["o100", "o99", "o98", "o97", "o96"]
        assert report["out_false_in_rate"] == pytest.approx(5 / 101)

    def test_report_carries_distributions_auroc_and_rows(self) -> None:
        report = calibrate(self._queries())
        assert set(report["distributions"]) == {"in", "out"}
        assert report["distributions"]["out"]["n"] == 101
        assert 0.0 <= report["auroc"] <= 1.0
        assert report["out_above"][0] == {
            "id": "o100",
            "score": 1.0,
            "topic": "topic",
            "question": "question o100",
        }

    def test_out_of_scope_is_the_p50_of_out_and_lists_the_in_below_it(self) -> None:
        queries = [*self._queries(), _q("i-weak", "in", 0.30), _q("i-tie", "in", 0.50)]
        report = calibrate(queries)
        oos = report["out_of_scope"]
        assert oos["threshold"] == pytest.approx(0.50)
        # At or below, as the runtime refuses (`score <= out_of_scope_below`): i-low and i-tie
        # sit ON the threshold.
        assert [r["id"] for r in oos["in_below"]] == ["i-weak", "i-low", "i-tie"]
        assert oos["in_refused_rate"] == pytest.approx(3 / 5)
        assert oos["threshold_interval"]["n"] == 101
        assert oos["in_refused_rate_interval"]["k"] == 3

    def test_rates_carry_clopper_pearson_intervals(self) -> None:
        report = calibrate(self._queries())
        rate = report["out_false_in_rate_interval"]
        assert (rate["k"], rate["n"]) == (5, 101)
        assert rate["lower"] < rate["rate"] < rate["upper"]
        assert report["in_recall_interval"]["n"] == 3
        assert report["threshold_interval"]["upper"] is not None

    def test_strata_split_by_category_and_label(self) -> None:
        queries = [
            _q("out-pens-01", "out", 0.7),
            _q("out-pens-02", "out", 0.2),
            _q("out-gen-01", "out", 0.1),
            ScoredQuery("in-a", "q", "in", 0.9, "t", "custom"),
        ]
        report = calibrate(queries)
        strata = report["strata"]
        assert set(strata) == {"pens", "gen", "custom"}
        assert strata["pens"]["out"]["n"] == 2
        assert strata["pens"]["out"]["max"] == 0.7
        assert strata["pens"]["out"]["above_low_confidence"] == 1
        assert strata["gen"]["out"]["below_out_of_scope"] == 1
        assert strata["pens"]["out"]["below_out_of_scope"] == 1  # 0.2 is the median: ON it counts
        assert strata["custom"]["in"]["above_low_confidence"] == 1

    def test_needs_both_labels_and_valid_data(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            calibrate([_q("a", "out", 0.1)])
        with pytest.raises(ValueError, match="unknown labels"):
            calibrate([_q("a", "maybe", 0.1), _q("b", "out", 0.2)])
        with pytest.raises(ValueError, match="finite"):
            calibrate([_q("a", "in", float("nan")), _q("b", "out", 0.2)])


NAMES = ["ncg", "faq"]
PROFILES = {n: RagProfile(f"p_{n}", 3, 100, "simple", "structured") for n in NAMES}


def _router(probe: FakeTopicProbe, bias: float = 0.0, **kwargs: object) -> EvidenceTopicRouter:
    topics = [EvidenceTopic(n, PROFILES[n], bias if n == "faq" else 0.0) for n in NAMES]
    return EvidenceTopicRouter(probe, topics, **kwargs)  # type: ignore[arg-type]


class TestScoring:
    async def test_score_is_the_best_raw_topic_score_without_bias(self) -> None:
        probe = FakeTopicProbe(
            {
                "a": {"p_ncg": [0.5, 0.4], "p_faq": [0.7, 0.2]},
                "b": {"p_ncg": [0.6]},
            }
        )
        queries = [LabelledQuery("a", "a", "in"), LabelledQuery("b", "b", "out")]
        scored = await score_queries(_router(probe, bias=0.1), TENANT, queries)
        assert [(s.id, s.score, s.topic) for s in scored] == [("a", 0.7, "faq"), ("b", 0.6, "ncg")]
        assert probe.calls == 2  # one probe (one embedding) per query

    async def test_mean_aggregate_is_the_one_of_the_spec(self) -> None:
        probe = FakeTopicProbe({"a": {"p_ncg": [0.6, 0.2], "p_faq": [0.3]}})
        scored = await score_queries(
            _router(probe, aggregate="mean_top_k"), TENANT, [LabelledQuery("a", "a", "in")]
        )
        assert scored[0].score == pytest.approx(0.4)

    async def test_texts_replace_the_question_that_is_embedded(self) -> None:
        probe = FakeTopicProbe({"standalone": {"p_ncg": [0.8]}})
        scored = await score_queries(
            _router(probe),
            TENANT,
            [LabelledQuery("a", "the raw question", "in", "cat")],
            {"a": "standalone"},
        )
        assert probe.queries == ["standalone"]
        assert (scored[0].question, scored[0].score, scored[0].category) == (
            "the raw question",
            0.8,
            "cat",
        )

    async def test_analyze_queries_asks_the_analyzer_once_per_question(self) -> None:
        class Analyzer:
            def __init__(self) -> None:
                self.seen: list[str] = []

            async def analyze(self, messages: Sequence[AnyMessage]) -> dict[str, Any]:
                self.seen.append(str(messages[-1].content))
                return {"standalone_query": f"standalone of {messages[-1].content}"}

        analyzer = Analyzer()
        result = await analyze_queries(
            analyzer, [LabelledQuery("a", "uno", "in"), LabelledQuery("b", "dos", "out")]
        )
        assert result == {"a": "standalone of uno", "b": "standalone of dos"}
        assert analyzer.seen == ["uno", "dos"]

    async def test_no_evidence_at_all_fails_loudly(self) -> None:
        probe = FakeTopicProbe({"a": {}})
        with pytest.raises(RuntimeError, match="no topic has any evidence"):
            await score_queries(_router(probe), TENANT, [LabelledQuery("a", "a", "in")])


class TestFiles:
    def _write(self, tmp_path: Path, name: str, rows: list[object]) -> Path:
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
        return path

    def test_rows_of_several_files_are_loaded(self, tmp_path: Path) -> None:
        one = self._write(tmp_path, "a.jsonl", [{"id": "x", "question": "q1", "label": "in"}])
        two = self._write(tmp_path, "b.jsonl", [{"id": "y", "question": "q2", "label": "out"}])
        assert load_labelled([one, two]) == [
            LabelledQuery("x", "q1", "in"),
            LabelledQuery("y", "q2", "out"),
        ]

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ({"id": "x", "question": "q", "label": "maybe"}, "label must be one of"),
            ({"id": "x", "label": "in"}, "invalid row"),
            ({"id": "x", "question": "  ", "label": "in"}, "blank question"),
            ({"id": 7, "question": "q", "label": "in"}, "'id' must be a string"),
            ({"id": "x", "question": ["q"], "label": "in"}, "'question' must be a string"),
            ({"id": "x", "question": "q", "label": None}, "'label' must be a string"),
            ({"id": "x", "question": "q", "label": "in", "category": 3}, "'category' must be"),
        ],
    )
    def test_bad_rows_are_usage_errors_naming_the_line(
        self, tmp_path: Path, row: dict[str, object], message: str
    ) -> None:
        path = self._write(tmp_path, "a.jsonl", [row])
        with pytest.raises(UsageError, match=message) as info:
            load_labelled([path])
        assert "a.jsonl:1" in str(info.value)

    def test_a_row_that_is_not_an_object_is_refused(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "a.jsonl", [["x"]])
        with pytest.raises(UsageError, match="a JSON object"):
            load_labelled([path])

    def test_a_missing_file_is_a_usage_error(self, tmp_path: Path) -> None:
        with pytest.raises(UsageError, match="cannot read"):
            load_labelled([tmp_path / "nope.jsonl"])

    def test_the_category_is_optional(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path, "a.jsonl", [{"id": "x", "question": "q", "label": "in", "category": "c"}]
        )
        assert load_labelled([path]) == [LabelledQuery("x", "q", "in", "c")]

    def test_duplicate_ids_across_files_are_refused(self, tmp_path: Path) -> None:
        row = {"id": "x", "question": "q", "label": "in"}
        one = self._write(tmp_path, "a.jsonl", [row])
        two = self._write(tmp_path, "b.jsonl", [row])
        with pytest.raises(UsageError, match=r"already used in a\.jsonl:1"):
            load_labelled([one, two])


class TestReport:
    def test_report_has_meta_score_description_and_sorted_scores(self) -> None:
        scored = [_q("a", "in", 0.7), _q("b", "out", 0.2), _q("c", "out", 0.4)]
        report = build_report(scored, {"agent": "x"})
        assert report["meta"] == {"agent": "x"}
        assert "raw_score" in report["score"]
        assert "ALL" in report["score"]
        assert any("written looking at the documents" in c for c in report["caveats"])
        assert [r["id"] for r in report["scores"]] == ["a", "c", "b"]
        assert "standalone_queries" not in report
        json.dumps(report)  # serialisable
        assert "scores" not in build_report(scored, {}, with_scores=False)

    def test_report_carries_the_standalone_queries(self) -> None:
        scored = [_q("a", "in", 0.7), _q("b", "out", 0.2)]
        report = build_report(scored, {}, standalone={"a": "alpha", "b": "beta"})
        assert report["standalone_queries"] == {"a": "alpha", "b": "beta"}

    def test_pick_spec_requires_an_evidence_agent(self) -> None:
        specs = _specs()
        assert pick_spec(specs, "ev") is specs[0]
        with pytest.raises(UsageError, match="router: evidence"):
            pick_spec(specs, "llm")
        with pytest.raises(UsageError, match="unknown agent"):
            pick_spec(specs, "zzz")

    def test_profiles_digest_follows_the_yaml_files_only(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("x: 1\n", encoding="utf-8")
        (tmp_path / "b.yml").write_text("y: 2\n", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
        first = profiles_digest(tmp_path)
        assert set(first["files"]) == {"a.yaml", "b.yml"}
        assert profiles_digest(tmp_path) == first
        (tmp_path / "a.yaml").write_text("x: 2\n", encoding="utf-8")
        assert profiles_digest(tmp_path)["sha256"] != first["sha256"]

    def test_report_path_is_validated_before_anything_boots(self, tmp_path: Path) -> None:
        check_report_path(None)
        check_report_path(tmp_path / "report.json")
        with pytest.raises(UsageError, match="is a directory"):
            check_report_path(tmp_path)
        with pytest.raises(UsageError, match="does not exist"):
            check_report_path(tmp_path / "missing" / "report.json")


def _specs() -> list[AgentSpec]:
    class Spec:
        def __init__(self, id: str, evidence: bool) -> None:
            self.id, self.uses_evidence_router = id, evidence
            self.models: dict[str, str] = {}

    return cast(list[AgentSpec], [Spec("ev", True), Spec("llm", False)])


class TestProfileBoundAnalyzer:
    """The analyzer can come from a profile bound in the spec instead of LLM_PROVIDER."""

    def _spec(self) -> AgentSpec:
        return parse_agent_spec(
            {
                "id": "ev",
                "description": "d",
                "models": {"analyzer": "judge"},
                "topics": [
                    {"name": "a", "description": "d", "profile": "pa"},
                    {"name": "b", "description": "d", "profile": "pb"},
                ],
                "router": "evidence",
            },
            origin="test",
        )

    def _models(self, env: dict[str, str]) -> ChatModels:
        catalog = ModelCatalog(
            LlmConfig(provider="openai", models={role: "base" for role in ChatRole}),
            {"judge": LlmProfile("openai", "judge-model", api_key_env="JUDGE_KEY")},
            environ=env.get,
            builder=lambda *_a, **_k: ScriptedChatModel(),
        )
        return ChatModels(catalog)

    def test_without_analyze_an_unset_analyzer_key_does_not_matter(self) -> None:
        view = agent_chat_models(self._models({}), self._spec(), analyze=False)

        assert view.describe(ChatRole.SYNTHESIS).profile == "default"  # nothing was resolved

    def test_with_analyze_the_unset_key_fails_naming_profile_and_agent(self) -> None:
        with pytest.raises(LlmError, match=r"profile 'judge' of agent 'ev'.*JUDGE_KEY"):
            agent_chat_models(self._models({}), self._spec(), analyze=True)
        view = agent_chat_models(self._models({"JUDGE_KEY": "k"}), self._spec(), analyze=True)
        assert view.describe(ChatRole.ANALYZER).label == "judge (openai:judge-model)"

    def test_the_analyze_guard_accepts_a_bound_profile_without_llm_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JUDGE_KEY", "k")
        profiles = tmp_path / "llm_profiles.yaml"
        profiles.write_text(
            "profiles:\n  judge: {provider: openai, model: m, api_key_env: JUDGE_KEY}\n",
            encoding="utf-8",
        )
        settings = make_settings(llm={"profiles_file": profiles})

        check_analyzer_available(settings, self._spec())  # no LLM_PROVIDER, bound analyzer

        unbound = parse_agent_spec({**self._spec().model_dump(), "models": {}}, origin="test")
        with pytest.raises(UsageError, match="--analyze needs the LLM"):
            check_analyzer_available(settings, unbound)


class TestMain:
    """Every usage problem exits 2 BEFORE the services boot."""

    @pytest.fixture(autouse=True)
    def _no_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_: object, **__: object) -> None:
            raise AssertionError("the container must not boot")

        monkeypatch.setattr(Container, "build", boom)
        monkeypatch.setattr(cli, "load_agent_specs", lambda _path: _specs())
        monkeypatch.setattr(cli, "get_settings", make_settings)

    def _files(self, tmp_path: Path, *labels: str) -> list[str]:
        path = tmp_path / "q.jsonl"
        rows = [{"id": f"x-{n}", "question": "q", "label": label} for n, label in enumerate(labels)]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return [str(path)]

    def _argv(self, tmp_path: Path, files: list[str], *extra: str, agent: str = "ev") -> list[str]:
        return [
            *files,
            "--profiles",
            str(tmp_path),
            "--agents",
            str(tmp_path),
            "--agent",
            agent,
            *extra,
        ]

    def test_unknown_agent(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        argv = self._argv(tmp_path, self._files(tmp_path, "in", "out"), agent="zzz")
        assert cli.main(argv) == EXIT_USAGE
        assert "unknown agent" in capsys.readouterr().err

    def test_agent_without_the_evidence_router(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = self._argv(tmp_path, self._files(tmp_path, "in", "out"), agent="llm")
        assert cli.main(argv) == EXIT_USAGE
        assert "router: evidence" in capsys.readouterr().err

    @pytest.mark.parametrize("labels", [("in", "in"), ("out",)])
    def test_missing_labels(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], labels: tuple[str, ...]
    ) -> None:
        argv = self._argv(tmp_path, self._files(tmp_path, *labels))
        assert cli.main(argv) == EXIT_USAGE
        assert "at least one 'in' and one 'out'" in capsys.readouterr().err

    def test_missing_labelled_file(self, tmp_path: Path) -> None:
        assert cli.main(self._argv(tmp_path, [str(tmp_path / "nope.jsonl")])) == EXIT_USAGE

    def test_unwritable_report_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = self._argv(
            tmp_path,
            self._files(tmp_path, "in", "out"),
            "--report",
            str(tmp_path / "missing" / "r.json"),
        )
        assert cli.main(argv) == EXIT_USAGE
        assert "--report" in capsys.readouterr().err

    def test_analyze_needs_the_llm(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = self._argv(tmp_path, self._files(tmp_path, "in", "out"), "--analyze")
        assert cli.main(argv) == EXIT_USAGE
        assert "--analyze needs the LLM" in capsys.readouterr().err

    def test_a_boot_failure_after_valid_arguments_aborts_with_3(self, tmp_path: Path) -> None:
        # The no-boot guard raises AssertionError: valid arguments reach the boot -> abort.
        argv = self._argv(tmp_path, self._files(tmp_path, "in", "out"))
        assert cli.main(argv) == EXIT_ABORTED

    def test_a_report_write_failure_is_an_abort_not_a_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = _canned_report()

        async def fake_run(_args: object) -> dict[str, Any]:
            return report

        monkeypatch.setattr(cli, "run", fake_run)
        target = tmp_path / "r.json"
        argv = self._argv(tmp_path, self._files(tmp_path, "in", "out"), "--report", str(target))

        def refuse(*_a: object, **_k: object) -> int:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", refuse)
        assert cli.main(argv) == EXIT_ABORTED
        assert "cannot write the report" in capsys.readouterr().err

    def test_a_successful_run_prints_and_writes_the_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = _canned_report()

        async def fake_run(_args: object) -> dict[str, Any]:
            return report

        monkeypatch.setattr(cli, "run", fake_run)
        target = tmp_path / "r.json"
        argv = self._argv(tmp_path, self._files(tmp_path, "in", "out"), "--report", str(target))
        assert cli.main(argv) == EXIT_OK
        out = capsys.readouterr().out
        assert "low_confidence_below" in out
        assert "out_of_scope_below" in out
        assert "Clopper-Pearson" in out
        assert json.loads(target.read_text(encoding="utf-8"))["meta"]["agent"] == "ev"

    def test_the_llm_usage_of_the_run_reaches_the_ledger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ledger: FakeLedger
    ) -> None:
        async def fake_run(_args: object) -> dict[str, Any]:
            await NamedModel(replies=["x"], model_name="m-cal").ainvoke("hi")
            return _canned_report()

        monkeypatch.setattr(cli, "run", fake_run)
        argv = self._argv(tmp_path, self._files(tmp_path, "in", "out"))

        assert cli.main(argv) == EXIT_OK
        assert [(r.model, r.purpose, r.agent_id) for r in ledger.rows] == [
            ("m-cal", Purpose.CALIBRATE, "ev")
        ]


def _canned_report() -> dict[str, Any]:
    scored = [
        ScoredQuery("in-a-1", "qa", "in", 0.8, "t"),
        ScoredQuery("in-a-2", "qb", "in", 0.4, "t"),
        ScoredQuery("out-x-1", "qc", "out", 0.5, "t"),
        ScoredQuery("out-x-2", "qd", "out", 0.3, "t"),
        ScoredQuery("out-x-3", "qe", "out", 0.2, "t"),
    ]
    meta = {"agent": "ev", "query_input": "analyzed"}
    return build_report(scored, meta, standalone={s.id: s.question + "!" for s in scored})
