"""AgentSpec parsing, the directory loader and the validation against the profiles."""

from pathlib import Path
from typing import ClassVar

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.agent_kit.nodes import EvidenceTopicRouter, LlmTopicRouter, build_route_schema
from agintis.agent_kit.nodes.prompts import (
    DEFAULT_SYNTHESIZE_PROMPT,
    DEFAULT_UNCITED_SYNTHESIZE_PROMPT,
)
from agintis.modules.agents.rag_chat import (
    AgentSpec,
    InvalidAgentSpecError,
    RagChatDeps,
    build_router,
    load_agent_specs,
    parse_agent_spec,
    rag_chat_runners,
    validate_agent_specs,
)
from agintis.modules.knowledge.api import (
    MetadataField,
    ProbeProfiles,
    RagProfile,
    RagProfileRegistry,
    load_profiles_dir,
)
from tests.fakes.knowledge import InMemoryChunkRepository
from tests.fakes.llm import chat_models_for
from tests.fakes.retrieval import FakeRetrieval
from tests.support.postgres import PROJECT_ROOT

PROFILE = RagProfile(
    "docs",
    3,
    100,
    "simple",
    "structured",
    metadata_fields=(
        MetadataField("kind", "str", filterable=True),
        MetadataField("secret", "str"),
    ),
)
REGISTRY = RagProfileRegistry([PROFILE])

VALID = """\
id: docs-rag
description: Answers from the docs.
topics:
  - name: docs
    description: The documents.
    profile: docs
"""


def _write(directory: Path, name: str, text: str) -> None:
    (directory / name).write_text(text, encoding="utf-8")


def test_context_chars_is_a_bounded_knob_and_feeds_the_limits() -> None:
    raw: dict[str, object] = {"id": "a", "description": "d", "topics": [_topic()]}

    spec = parse_agent_spec({**raw, "max_context_chars": 3000, "max_sources": 4}, origin="a")

    assert (spec.limits.max_context_chars, spec.limits.max_sources) == (3000, 4)
    assert parse_agent_spec(raw, origin="a").limits.max_context_chars == 12000
    for bad in (10, 10_000_000):
        with pytest.raises(InvalidAgentSpecError, match="max_context_chars"):
            parse_agent_spec({**raw, "max_context_chars": bad}, origin="a")


def test_a_minimal_spec_gets_the_defaults() -> None:
    spec = parse_agent_spec({"id": "a", "description": "d", "topics": [_topic()]}, origin="a.yaml")

    assert spec.topics[0].filters is None  # every filterable field
    assert spec.prompts.to_prompt_set(citations="inline").synthesize.startswith("You answer")
    assert (spec.max_sources, spec.max_history_messages) == (8, 10)


def _topic(**changes: object) -> dict[str, object]:
    return {"name": "docs", "description": "d", "profile": "docs", **changes}


def test_prompt_overrides_replace_only_the_given_ones() -> None:
    spec = parse_agent_spec(
        {
            "id": "a",
            "description": "d",
            "topics": [_topic()],
            "prompts": {"synthesize": "Responde."},
        },
        origin="a.yaml",
    )
    prompts = spec.prompts.to_prompt_set(citations="inline")

    assert prompts.synthesize == "Responde."
    assert prompts.analyze.startswith("You prepare")


class TestCitations:
    RAW: ClassVar[dict[str, object]] = {"id": "a", "description": "d", "topics": [_topic()]}

    def test_inline_is_the_default_and_keeps_the_cited_prompt(self) -> None:
        spec = parse_agent_spec(self.RAW, origin="a.yaml")

        assert spec.citations == "inline"
        assert (
            spec.prompts.to_prompt_set(citations=spec.citations).synthesize
            == DEFAULT_SYNTHESIZE_PROMPT
        )

    def test_none_picks_the_uncited_default_prompt(self) -> None:
        spec = parse_agent_spec({**self.RAW, "citations": "none"}, origin="a.yaml")

        prompt = spec.prompts.to_prompt_set(citations=spec.citations).synthesize
        assert prompt == DEFAULT_UNCITED_SYNTHESIZE_PROMPT

    def test_a_synthesize_override_wins_in_both_styles(self) -> None:
        for style in ("inline", "none"):
            spec = parse_agent_spec(
                {**self.RAW, "citations": style, "prompts": {"synthesize": "Mine."}}, origin="a"
            )
            assert spec.prompts.to_prompt_set(citations=spec.citations).synthesize == "Mine."

    def test_an_unknown_style_is_rejected(self) -> None:
        with pytest.raises(InvalidAgentSpecError, match="citations"):
            parse_agent_spec({**self.RAW, "citations": "footnotes"}, origin="a.yaml")

    @pytest.mark.parametrize(
        ("synthesize", "warns"),
        [
            (None, False),
            ("Mine.", False),
            ("Answer from the sources [1].", True),
            ("Cite each source as [n].", True),
        ],
    )
    def test_none_warns_only_when_the_custom_prompt_carries_markers(
        self, caplog: pytest.LogCaptureFixture, synthesize: str | None, warns: bool
    ) -> None:
        prompts = {} if synthesize is None else {"synthesize": synthesize}
        with caplog.at_level("WARNING"):
            parse_agent_spec({**self.RAW, "citations": "none", "prompts": prompts}, origin="a")
        assert ("mentions [n] markers" in caplog.text) is warns


@pytest.mark.parametrize(
    "raw",
    [
        {"id": "a", "description": "d", "topics": [_topic()], "surprise": 1},  # extra key
        {"id": "a", "description": "d", "topics": [_topic(extra=1)]},  # extra key, nested
        {"id": "bad id!", "description": "d", "topics": [_topic()]},  # AGENT_ID_PATTERN
        {"id": "a", "description": "", "topics": [_topic()]},
        {"id": "a", "description": "d"},  # topics missing
        {"id": "a", "description": "d", "topics": [_topic()], "max_sources": 0},
        {"id": "a", "description": "d", "topics": [_topic()], "max_sources": 99},
        {"id": "a", "description": "d", "topics": [_topic(name="x y")]},
        {"id": "a", "description": "d", "topics": [_topic()], "prompts": {"analyze": ""}},
        {"id": "a", "description": "d", "topics": [_topic()], "prompts": {"nope": "x"}},
    ],
)
def test_invalid_specs_are_rejected_naming_the_file(raw: dict[str, object]) -> None:
    with pytest.raises(InvalidAgentSpecError, match=r"a\.yaml"):
        parse_agent_spec(raw, origin="a.yaml")


def test_a_spec_must_be_a_mapping() -> None:
    with pytest.raises(InvalidAgentSpecError, match="mapping"):
        parse_agent_spec(["x"], origin="a.yaml")


class TestLoader:
    def test_loads_every_yaml_file_sorted_by_name(self, tmp_path: Path) -> None:
        _write(tmp_path, "b.yaml", VALID.replace("docs-rag", "b-agent"))
        _write(tmp_path, "a.yml", VALID.replace("docs-rag", "a-agent"))
        _write(tmp_path, "notes.txt", "ignored")

        assert [s.id for s in load_agent_specs(tmp_path)] == ["a-agent", "b-agent"]

    def test_an_empty_directory_yields_no_agents(self, tmp_path: Path) -> None:
        assert load_agent_specs(tmp_path) == []

    def test_a_missing_directory_fails(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidAgentSpecError, match="does not exist"):
            load_agent_specs(tmp_path / "nope")

    def test_an_unknown_key_in_a_file_fails_naming_it(self, tmp_path: Path) -> None:
        _write(tmp_path, "bad.yaml", VALID + "colour: red\n")
        with pytest.raises(InvalidAgentSpecError, match=r"bad\.yaml.*colour"):
            load_agent_specs(tmp_path)

    def test_broken_yaml_and_duplicate_keys_and_aliases_fail(self, tmp_path: Path) -> None:
        for text in ("id: [unclosed", VALID + "id: again\n", "id: &a x\nref: *a\n"):
            directory = tmp_path / str(abs(hash(text)))
            directory.mkdir()
            _write(directory, "x.yaml", text)
            with pytest.raises(InvalidAgentSpecError, match=r"x\.yaml"):
                load_agent_specs(directory)

    def test_two_files_with_the_same_agent_id_fail(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.yaml", VALID)
        _write(tmp_path, "b.yaml", VALID)
        with pytest.raises(InvalidAgentSpecError, match=r"already defined in a\.yaml"):
            load_agent_specs(tmp_path)


class TestValidation:
    def _spec(self, **topic: object) -> AgentSpec:
        return parse_agent_spec(
            {"id": "a", "description": "d", "max_rewrites": 0, "topics": [_topic(**topic)]},
            origin="a.yaml",
        )

    def test_a_valid_spec_passes(self) -> None:
        validate_agent_specs([self._spec(filters={"kind": "hint"})], REGISTRY)

    def test_unknown_profile_fails_fast(self) -> None:
        with pytest.raises(InvalidAgentSpecError, match="unknown profile 'nope'"):
            validate_agent_specs([self._spec(profile="nope")], REGISTRY)

    def test_a_filter_that_is_not_filterable_fails_fast(self) -> None:
        with pytest.raises(InvalidAgentSpecError, match="secret"):
            validate_agent_specs([self._spec(filters={"secret": "x"})], REGISTRY)

    def test_a_topic_without_an_allowlist_logs_a_boot_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            validate_agent_specs([self._spec()], REGISTRY)
        assert "no `filters` allowlist" in caplog.text
        assert "ALL 1 filterable" in caplog.text

    def test_an_explicit_allowlist_or_no_filterable_fields_is_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bare = RagProfileRegistry([RagProfile("docs", 3, 100, "simple", "structured")])
        with caplog.at_level("WARNING"):
            validate_agent_specs([self._spec(filters={"kind": "hint"})], REGISTRY)
            validate_agent_specs([self._spec(filters={})], REGISTRY)
            validate_agent_specs([self._spec()], bare)
        assert caplog.text == ""

    def _rewrite_spec(self, **changes: object) -> AgentSpec:
        raw = {"id": "a", "description": "d", "topics": [_topic(filters={})], **changes}
        return parse_agent_spec(raw, origin="a.yaml")

    def test_relevance_rewrites_without_any_reranker_log_a_boot_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            validate_agent_specs([self._rewrite_spec(max_rewrites=1)], REGISTRY)
        assert "the rewrite loop never fires without a reranker" in caplog.text

    def test_the_rewrite_warning_is_silent_when_it_cannot_apply(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        reranked = RagProfileRegistry(
            [RagProfile("docs", 3, 100, "simple", "structured", reranker_key="tei")]
        )
        with caplog.at_level("WARNING"):
            validate_agent_specs([self._rewrite_spec(max_rewrites=1)], reranked)  # a reranker
            validate_agent_specs([self._rewrite_spec(max_rewrites=0)], REGISTRY)  # no rewrites
            validate_agent_specs(
                [self._rewrite_spec(max_rewrites=1, sufficiency="count")], REGISTRY
            )  # not relevance
        assert caplog.text == ""

    def test_field_names_that_clash_with_pydantic_pass_validation(self) -> None:
        registry = RagProfileRegistry(
            [
                RagProfile(
                    "docs",
                    3,
                    100,
                    "simple",
                    "structured",
                    metadata_fields=tuple(
                        MetadataField(n, "str", filterable=True)
                        for n in ("_x", "model_config", "model_dump")
                    ),
                )
            ]
        )
        validate_agent_specs([self._spec()], registry)

    def test_a_schema_that_cannot_be_built_fails_at_boot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(_plans: object) -> None:
            raise ValueError("cannot build")

        monkeypatch.setattr("agintis.modules.agents.rag_chat.spec.build_filter_schema", broken)
        with pytest.raises(InvalidAgentSpecError, match="cannot build"):
            validate_agent_specs([self._spec()], REGISTRY)

    def test_runners_are_built_only_after_validation(self) -> None:
        deps = RagChatDeps(chat_models_for(), FakeRetrieval())
        with pytest.raises(InvalidAgentSpecError):
            rag_chat_runners([self._spec(profile="nope")], REGISTRY, deps)

        (runner,) = rag_chat_runners([self._spec()], REGISTRY, deps)
        assert runner.agent_id == "a"
        assert runner.supports_threads is False


def _spec_raw(*topics: dict[str, object], **changes: object) -> dict[str, object]:
    return {"id": "a", "description": "d", "topics": list(topics) or [_topic()], **changes}


class TestMultiTopicSpec:
    def test_the_routing_knobs_have_bounded_named_defaults(self) -> None:
        spec = parse_agent_spec(_spec_raw(), origin="a.yaml")

        assert (spec.router, spec.max_topics, spec.max_rewrites, spec.min_sources) == (
            "auto",
            2,
            1,
            1,
        )

    @pytest.mark.parametrize(
        ("knob", "bad"),
        [
            ("max_topics", 0),
            ("max_topics", 9),
            ("max_rewrites", -1),
            ("max_rewrites", 4),
            ("min_sources", 0),
            ("min_sources", 21),
            ("router", "magic"),
        ],
    )
    def test_the_knobs_are_bounded(self, knob: str, bad: object) -> None:
        with pytest.raises(InvalidAgentSpecError, match=knob):
            parse_agent_spec(_spec_raw(**{knob: bad}), origin="a.yaml")

    def test_the_knob_boundaries_are_accepted(self) -> None:
        spec = parse_agent_spec(
            _spec_raw(max_topics=8, max_rewrites=0, min_sources=8), origin="a.yaml"
        )
        assert (spec.max_topics, spec.max_rewrites, spec.min_sources) == (8, 0, 8)

    def test_at_least_one_topic(self) -> None:
        with pytest.raises(InvalidAgentSpecError, match="topics"):
            parse_agent_spec({"id": "a", "description": "d", "topics": []}, origin="a.yaml")

    def test_the_old_single_topic_key_is_gone(self) -> None:
        with pytest.raises(InvalidAgentSpecError, match="topic"):
            parse_agent_spec({"id": "a", "description": "d", "topic": _topic()}, origin="a.yaml")

    def test_topic_names_must_be_unique(self) -> None:
        raw = _spec_raw(_topic(name="x", profile="p1"), _topic(name="x", profile="p2"))
        with pytest.raises(InvalidAgentSpecError, match=r"topic names must be unique.*'x'"):
            parse_agent_spec(raw, origin="a.yaml")

    def test_topic_profiles_must_be_unique(self) -> None:
        raw = _spec_raw(_topic(name="x", profile="p"), _topic(name="y", profile="p"))
        with pytest.raises(InvalidAgentSpecError, match=r"topic profiles must be unique.*'p'"):
            parse_agent_spec(raw, origin="a.yaml")

    def test_the_llm_router_needs_at_least_two_topics(self) -> None:
        with pytest.raises(InvalidAgentSpecError, match="at least two topics"):
            parse_agent_spec(_spec_raw(router="llm"), origin="a.yaml")

    def test_the_single_router_needs_exactly_one_topic(self) -> None:
        two = [_topic(name="x", profile="p1"), _topic(name="y", profile="p2")]
        with pytest.raises(InvalidAgentSpecError, match="exactly one topic"):
            parse_agent_spec(_spec_raw(*two, router="single"), origin="a.yaml")
        assert parse_agent_spec(_spec_raw(router="single"), origin="a.yaml").router == "single"

    def test_min_sources_cannot_exceed_max_sources(self) -> None:
        with pytest.raises(InvalidAgentSpecError, match="min_sources"):
            parse_agent_spec(_spec_raw(min_sources=5, max_sources=4), origin="a.yaml")

    def test_auto_calls_the_router_model_only_with_several_topics(self) -> None:
        one = parse_agent_spec(_spec_raw(), origin="a.yaml")
        two = parse_agent_spec(
            _spec_raw(_topic(name="x", profile="p1"), _topic(name="y", profile="p2")),
            origin="a.yaml",
        )
        forced = parse_agent_spec(
            _spec_raw(_topic(name="x", profile="p1"), _topic(name="y", profile="p2"), router="llm"),
            origin="a.yaml",
        )

        assert (one.uses_llm_router, two.uses_llm_router, forced.uses_llm_router) == (
            False,
            True,
            True,
        )

    def test_route_and_rewrite_prompts_are_overridable(self) -> None:
        spec = parse_agent_spec(_spec_raw(prompts={"route": "R", "rewrite": "W"}), origin="a.yaml")
        prompts = spec.prompts.to_prompt_set(citations="inline")
        assert (prompts.route, prompts.rewrite) == ("R", "W")
        assert (
            parse_agent_spec(_spec_raw(), origin="a")
            .prompts.to_prompt_set(citations="inline")
            .route.startswith("You route")
        )

    def test_every_topic_is_validated_against_its_own_profile(self) -> None:
        other = RagProfile(
            "other", 3, 100, "simple", "structured", metadata_fields=(MetadataField("n", "int"),)
        )
        registry = RagProfileRegistry([PROFILE, other])
        good = _topic(name="a", profile="docs", filters={"kind": "h"})
        bad = _topic(name="b", profile="other", filters={"kind": "h"})  # `n` is not filterable

        validate_agent_specs([parse_agent_spec(_spec_raw(good), origin="a")], registry)
        with pytest.raises(InvalidAgentSpecError, match=r"topic 'b'.*kind"):
            validate_agent_specs([parse_agent_spec(_spec_raw(good, bad), origin="a")], registry)

    def test_every_topic_problem_is_reported_at_once(self) -> None:
        raw = _spec_raw(_topic(name="a", profile="nope1"), _topic(name="b", profile="nope2"))
        with pytest.raises(InvalidAgentSpecError) as info:
            validate_agent_specs([parse_agent_spec(raw, origin="a")], REGISTRY)
        assert "nope1" in str(info.value)
        assert "nope2" in str(info.value)


class TestSpensionesAgent:
    """The real case config against the real case profiles (no network, no database)."""

    PROFILES_DIR = PROJECT_ROOT / "case" / "spensiones" / "profiles"
    AGENTS_DIR = PROJECT_ROOT / "case" / "spensiones" / "agents"
    FORBIDDEN: ClassVar[set[str]] = {"source_id", "doc_family", "cited_laws", "content_mode"}

    def _load(self) -> tuple[AgentSpec, RagProfileRegistry]:
        registry = RagProfileRegistry(load_profiles_dir(self.PROFILES_DIR))
        (spec,) = load_agent_specs(self.AGENTS_DIR)
        return spec, registry

    def test_it_loads_and_validates_without_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            spec, registry = self._load()
            validate_agent_specs([spec], registry)

        # Every topic has an explicit allowlist and no knob of the other router is set; the only
        # warning is the known one: no reranker deployed, so its single rewrite never fires.
        assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
            "agent 'spensiones-rag': sufficiency 'relevance' with max_rewrites=1 but no topic "
            "profile has a reranker: the rewrite loop never fires without a reranker"
        ]
        assert spec.id == "spensiones-rag"
        assert spec.uses_evidence_router
        assert not spec.uses_llm_router

    def test_the_evidence_knobs_are_the_pre_registered_defaults_with_no_bias(self) -> None:
        spec, _ = self._load()

        assert (spec.evidence_aggregate, spec.evidence_margin, spec.probe_k) == ("max", 0.02, 3)
        assert spec.max_topics == 2
        assert {"evidence_margin", "evidence_aggregate", "probe_k"} <= spec.model_fields_set
        assert [t.evidence_bias for t in spec.topics] == [0.0] * 7  # calibrated on a dev set only

    def test_one_topic_per_profile_with_a_spanish_description(self) -> None:
        spec, registry = self._load()

        assert {t.profile for t in spec.topics} == {p.name for p in registry.all()}
        assert len(spec.topics) == 7
        assert all(len(t.description) > 40 for t in spec.topics)

    def test_the_allowlists(self) -> None:
        spec, _ = self._load()
        filters = {t.name: set(t.filters or {}) for t in spec.topics}

        assert filters["compendio"] == {
            "system",
            "libro",
            "titulo",
            "letra",
            "capitulo",
            "anexo",
            "numeros",
            "amended_by_ncg",
        }
        assert filters["ncg"] == {"ncg_number", "issued_date"}
        assert filters["circular"] == {"circular_number", "series", "issued_date"}
        assert filters["resolucion"] == {"resolution_number", "issued_date"}
        assert filters["jurisprudencia_nota"] == {"nota_number", "issued_date"}
        assert filters["articulo_faq"] == {"aid"}
        assert filters["jurisprudencia_indice"] == set()
        assert not self.FORBIDDEN & set().union(*filters.values())

    def test_the_system_hint_only_asks_for_the_cesantia_value(self) -> None:
        spec, _ = self._load()
        compendio = next(t for t in spec.topics if t.name == "compendio")
        hint = (compendio.filters or {})["system"]

        assert "SOLO el valor SC" in hint
        assert "nunca uses SP" in hint

    def test_it_builds_the_runner_with_the_evidence_router_and_no_router_model(self) -> None:
        spec, registry = self._load()
        probe = ProbeProfiles(InMemoryChunkRepository(), DeterministicFakeEmbedding(size=4))
        deps = RagChatDeps(chat_models_for(), FakeRetrieval(), probe=probe)

        (runner,) = rag_chat_runners([spec], registry, deps)

        assert runner.agent_id == "spensiones-rag"
        assert isinstance(
            build_router(spec, deps.chat_models, probe=probe, profiles=registry),
            EvidenceTopicRouter,
        )

    def test_switching_back_to_the_llm_router_still_works(self) -> None:
        spec, registry = self._load()
        llm = AgentSpec.model_validate(
            {
                **{
                    k: v
                    for k, v in spec.model_dump(exclude_unset=True).items()
                    if k
                    not in {"evidence_margin", "evidence_aggregate", "probe_k", "clarification"}
                },
                "router": "auto",
            }
        )
        deps = RagChatDeps(chat_models_for(), FakeRetrieval())

        (runner,) = rag_chat_runners([llm], registry, deps)

        assert runner.agent_id == "spensiones-rag"
        router = build_router(llm, deps.chat_models)
        assert isinstance(router, LlmTopicRouter)
        items = build_route_schema([t.name for t in llm.topics]).model_json_schema()
        choice = items["$defs"]["TopicChoice"]
        assert choice["properties"]["topic"]["enum"] == [t.name for t in llm.topics]


class TestRoutingKnobsAndReachability:
    def test_the_new_knobs_have_bounded_defaults(self) -> None:
        spec = parse_agent_spec(_spec_raw(), origin="a.yaml")

        assert (
            spec.sufficiency,
            spec.secondary_min_confidence,
            spec.primary_weight,
            spec.min_rerank_score,
        ) == ("relevance", 0.6, 2, 0.1)

    @pytest.mark.parametrize(
        ("knob", "bad"),
        [
            ("secondary_min_confidence", -0.1),
            ("secondary_min_confidence", 1.1),
            ("primary_weight", 0),
            ("primary_weight", 6),
            ("min_rerank_score", 1.5),
            ("min_rerank_score", -1),
            ("sufficiency", "vibes"),
        ],
    )
    def test_the_new_knobs_are_bounded(self, knob: str, bad: object) -> None:
        with pytest.raises(InvalidAgentSpecError, match=knob):
            parse_agent_spec(_spec_raw(**{knob: bad}), origin="a.yaml")

    def _registry(self, *top_ks: int) -> tuple[RagProfileRegistry, list[dict[str, object]]]:
        profiles = [
            RagProfile(f"p{i}", k, 100, "simple", "structured") for i, k in enumerate(top_ks)
        ]
        topics = [_topic(name=f"t{i}", profile=p.name) for i, p in enumerate(profiles)]
        return RagProfileRegistry(profiles), topics

    def test_min_sources_must_be_reachable_by_one_attempt(self) -> None:
        registry, topics = self._registry(3, 2, 1)
        # max_topics 2: at most 3 + 2 hits per attempt.
        validate_agent_specs(
            [parse_agent_spec(_spec_raw(*topics, min_sources=5), origin="a")], registry
        )
        with pytest.raises(InvalidAgentSpecError, match=r"min_sources=6 can never be met.*5 hits"):
            validate_agent_specs(
                [parse_agent_spec(_spec_raw(*topics, min_sources=6), origin="a")], registry
            )

    def test_a_single_topic_agent_is_bounded_by_its_own_top_k(self) -> None:
        registry, topics = self._registry(3)
        with pytest.raises(InvalidAgentSpecError, match="min_sources=4"):
            validate_agent_specs(
                [parse_agent_spec(_spec_raw(*topics, min_sources=4), origin="a")], registry
            )

    def test_max_topics_one_counts_only_the_largest_profile(self) -> None:
        registry, topics = self._registry(3, 2)
        with pytest.raises(InvalidAgentSpecError, match="min_sources=4"):
            validate_agent_specs(
                [parse_agent_spec(_spec_raw(*topics, min_sources=4, max_topics=1), origin="a")],
                registry,
            )
