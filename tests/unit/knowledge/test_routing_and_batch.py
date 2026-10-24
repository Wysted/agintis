"""Profile routing by document metadata and the batch ingest use case."""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.modules.knowledge.api import (
    BatchOutcome,
    IngestBatch,
    IngestDocument,
    ProfileRouter,
    SourceFile,
    load_profiles_dir,
    parse_profile,
)
from agintis.modules.knowledge.domain import RagProfile, RoutingCondition, RoutingRule
from agintis.shared_kernel.ids import TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    FakeChunker,
    FakeParser,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryIngestLock,
)
from tests.support.postgres import PROJECT_ROOT

CASE_PROFILES = PROJECT_ROOT / "case" / "spensiones" / "profiles"


def _profile(name: str, *rules: RoutingRule, status: str = "active") -> RagProfile:
    return RagProfile(
        name,
        3,
        100,
        "fake",
        "fake",
        routing=rules,
        status=status,  # type: ignore[arg-type]
    )


def _rule(*, priority: int = 0, **match: Any) -> RoutingRule:
    return RoutingRule(
        tuple(
            RoutingCondition(field, any_of=tuple(map(str, values)))
            for field, values in match.items()
        ),
        priority=priority,
    )


class TestProfileRouter:
    def test_the_highest_priority_wins_whatever_the_order(self) -> None:
        narrow = _profile("narrow", _rule(doc_type=["ncg"], subtype=["NP"], priority=10))
        broad = _profile("broad", _rule(doc_type=["ncg"]))
        for profiles in ([narrow, broad], [broad, narrow]):
            router = ProfileRouter(profiles)
            assert router.select({"doc_type": "ncg", "subtype": "NP"}).profile is narrow
            assert router.select({"doc_type": "ncg", "subtype": "NC"}).profile is broad

    def test_an_overlap_at_the_same_priority_is_ambiguous_and_unrouted(self) -> None:
        a = _profile("a", _rule(doc_type=["x"]))
        b = _profile("b", _rule(doc_type=["x"], area=["p"]))
        selection = ProfileRouter([a, b]).select({"doc_type": "x", "area": "p"})
        assert selection.profile is None
        assert selection.reason == "ambiguous: a, b"
        assert ProfileRouter([a, b]).select({"doc_type": "x"}).profile is a  # b does not match

    def test_two_rules_of_one_profile_are_not_ambiguous(self) -> None:
        both = _profile("p", _rule(doc_type=["x"]), _rule(area=["p"]))
        assert ProfileRouter([both]).select({"doc_type": "x", "area": "p"}).profile is both

    def test_a_lower_priority_tie_does_not_matter(self) -> None:
        a = _profile("a", _rule(doc_type=["x"]))
        b = _profile("b", _rule(doc_type=["x"]))
        top = _profile("top", _rule(doc_type=["x"], priority=5))
        assert ProfileRouter([a, b, top]).select({"doc_type": "x"}).profile is top

    def test_a_profile_may_have_several_rules(self) -> None:
        profile = _profile("p", _rule(doc_type=["a"]), _rule(doc_type=["b"]))
        router = ProfileRouter([profile])
        assert router.select({"doc_type": "b"}).profile is profile
        assert router.select({"doc_type": "c"}).profile is None

    def test_every_match_condition_must_hold(self) -> None:
        router = ProfileRouter([_profile("p", _rule(doc_type=["a"], area=["x", "y"]))])
        assert router.select({"doc_type": "a", "area": "y"}).profile is not None
        assert router.select({"doc_type": "a", "area": "z"}).profile is None
        assert router.select({"doc_type": "a"}).profile is None  # a missing field never holds

    def test_regex_conditions_search_the_text(self) -> None:
        rule = RoutingRule((RoutingCondition("title", pattern=r"^\s*Resoluci[oó]n"),))
        router = ProfileRouter([_profile("p", rule)])
        assert router.select({"title": "  Resolución Exenta N° 1"}).profile is not None
        assert router.select({"title": "La Resolución"}).profile is None

    def test_exclusions_veto_a_rule(self) -> None:
        rule = RoutingRule(
            (RoutingCondition("doc_type", ("articulo_html",)),),
            (RoutingCondition("title", pattern="^Resolución"),),
        )
        router = ProfileRouter([_profile("faq", rule)])
        assert router.select({"doc_type": "articulo_html", "title": "¿Cómo?"}).profile is not None
        assert router.select({"doc_type": "articulo_html", "title": "Resolución X"}).profile is None

    def test_list_values_match_when_any_item_does(self) -> None:
        router = ProfileRouter([_profile("p", _rule(section=["Lobby"]))])
        assert router.select({"section": ("Otra", "Lobby")}).profile is not None
        assert router.select({"section": ("Otra",)}).profile is None
        assert router.select({"section": ()}).profile is None

    def test_numbers_compare_as_text(self) -> None:
        router = ProfileRouter([_profile("p", _rule(number=[2425]))])
        assert router.select({"number": 2425}).profile is not None
        assert router.select({"number": "2425"}).profile is not None

    def test_draft_profiles_are_never_selected(self) -> None:
        router = ProfileRouter([_profile("draft", _rule(doc_type=["a"]), status="draft")])
        selection = router.select({"doc_type": "a"})
        assert selection.profile is None
        assert "no active profile" in selection.reason

    def test_profiles_without_routing_are_never_selected(self) -> None:
        assert ProfileRouter([_profile("manual")]).select({"doc_type": "a"}).profile is None

    def test_the_reason_names_the_chosen_profile(self) -> None:
        router = ProfileRouter([_profile("legal", _rule(doc_type=["a"]))])
        assert "'legal'" in router.select({"doc_type": "a"}).reason


class TestYamlRouting:
    def test_the_yaml_shape_builds_rules(self, tmp_path: Path) -> None:
        profile = parse_profile(
            {
                "key": "p",
                "top_k": 3,
                "chunk_max_tokens": 100,
                "parser_key": "simple",
                "chunker_key": "structured",
                "status": "draft",
                "routing": [
                    {
                        "match": {"doc_type": ["a", "b"], "number": 7, "title": {"regex": "^X"}},
                        "exclude": {"area": "z"},
                        "priority": 4,
                    }
                ],
            },
            origin="p.yaml",
        )
        assert profile.status == "draft"
        (rule,) = profile.routing
        by_field = {c.field: c for c in rule.match}
        assert by_field["doc_type"].any_of == ("a", "b")
        assert by_field["number"].any_of == ("7",)
        assert by_field["title"].pattern == "^X"
        assert rule.exclude[0].any_of == ("z",)
        assert rule.priority == 4


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"doc_type": "compendio_libro"}, "spensiones_compendio"),
        ({"doc_type": "compendio_titulo"}, None),  # navigation hubs are not ingested
        ({"doc_type": "circular"}, "spensiones_circular"),
        ({"doc_type": "ncg"}, "spensiones_ncg"),
        (
            {"doc_type": "articulo_html", "title": "¿Cuánto me deben descontar?"},
            "spensiones_articulo_faq",
        ),
        (
            {"doc_type": "articulo_html", "title": "Resolución Exenta N° 5, del 3 de enero"},
            "spensiones_resolucion",
        ),
        ({"doc_type": "resolucion", "title": "Resolución Exenta N° 1266"}, "spensiones_resolucion"),
        (
            {"doc_type": "guia_faq", "title": '"Libro de Jurisprudencia de la Superintendencia"'},
            "spensiones_jurisprudencia_indice",
        ),
        ({"doc_type": "guia_faq", "title": "Nota Interna N° 1, del 2 de mayo"}, None),  # draft
        ({"doc_type": "oficio"}, None),
        # Lower-priority broad rules never beat a narrower one, in any file order.
        (
            {"doc_type": "articulo_html", "title": "Resolución N° 7"},
            "spensiones_resolucion",
        ),
    ],
)
def test_the_spensiones_index_rows_are_routed(row: dict[str, Any], expected: str | None) -> None:
    router = ProfileRouter(load_profiles_dir(CASE_PROFILES))
    selection = router.select(row)
    assert (selection.profile.name if selection.profile else None) == expected


FAKE_SOURCE = b"paragraph one\n\nparagraph two"


class _Selector:
    """Routes by the `kind` document field; anything else is unrouted."""

    def __init__(self, profile: RagProfile) -> None:
        self._profile = profile

    def select(self, document: Any) -> Any:
        from agintis.modules.knowledge.api import ProfileSelection

        if document.get("kind") == "ok":
            return ProfileSelection(self._profile, "routed")
        return ProfileSelection(None, "not for us")


class _Boom(FakeParser):
    async def parse(self, source: SourceFile) -> Any:
        if source.uri == "boom":
            raise RuntimeError("database is down")
        return await super().parse(source)


def _batch() -> tuple[IngestBatch, InMemoryChunkRepository]:
    documents = InMemoryDocumentRepository()
    chunks = InMemoryChunkRepository(documents)
    ingest = IngestDocument(
        documents,
        chunks,
        {"fake": _Boom()},
        {"fake": FakeChunker()},
        DeterministicFakeEmbedding(size=4),
        FixedClock(),
        InMemoryIngestLock(),
    )
    return IngestBatch(ingest, _Selector(_profile("legal"))), chunks  # type: ignore[arg-type]


def _source(uri: str, content: bytes = FAKE_SOURCE, **meta: Any) -> SourceFile:
    return SourceFile(uri, "text/plain", content, document_metadata=meta)


class TestIngestBatch:
    async def test_documents_are_routed_and_ingested(self) -> None:
        batch, chunks = _batch()
        outcomes = await batch.execute(
            TenantId("t"), [_source("a", kind="ok"), _source("b", kind="other")]
        )
        assert [(o.uri, o.status, o.profile, o.chunk_count) for o in outcomes] == [
            ("a", "indexed", "legal", 2),
            ("b", "unrouted", None, 0),
        ]
        assert outcomes[1].reason == "not for us"
        assert len(chunks.items) == 2

    async def test_reingesting_reports_skipped(self) -> None:
        batch, _ = _batch()
        await batch.execute(TenantId("t"), [_source("a", kind="ok")])
        (again,) = await batch.execute(TenantId("t"), [_source("a", kind="ok")])
        assert again.status == "skipped"

    async def test_a_document_problem_fails_that_document_only(self) -> None:
        batch, _ = _batch()
        outcomes = await batch.execute(
            TenantId("t"),
            [_source("empty", b"", kind="ok"), _source("fine", kind="ok")],
        )
        assert [o.status for o in outcomes] == ["failed", "indexed"]
        assert outcomes[0].reason is not None
        assert "EmptyDocumentError" in outcomes[0].reason

    async def test_an_infrastructure_error_aborts_the_batch(self) -> None:
        batch, _ = _batch()
        with pytest.raises(RuntimeError, match="database is down"):
            await batch.execute(
                TenantId("t"), [_source("boom", kind="ok"), _source("x", kind="ok")]
            )

    async def test_an_empty_batch_is_fine(self) -> None:
        batch, _ = _batch()
        assert await batch.execute(TenantId("t"), []) == []


class _ByTarget:
    """Routes by the `to` document field to one of several profiles."""

    def __init__(self, *profiles: RagProfile) -> None:
        self._profiles = {p.name: p for p in profiles}

    def select(self, document: Any) -> Any:
        from agintis.modules.knowledge.api import ProfileSelection

        profile = self._profiles.get(str(document.get("to")))
        return ProfileSelection(profile, "routed") if profile else ProfileSelection(None, "none")


def _rerouting_batch() -> tuple[IngestBatch, InMemoryDocumentRepository, InMemoryChunkRepository]:
    documents = InMemoryDocumentRepository()
    chunks = InMemoryChunkRepository(documents)
    ingest = IngestDocument(
        documents,
        chunks,
        {"fake": FakeParser()},
        {"fake": FakeChunker()},
        DeterministicFakeEmbedding(size=4),
        FixedClock(),
        InMemoryIngestLock(),
    )
    selector = _ByTarget(_profile("old"), _profile("new"))
    return IngestBatch(ingest, selector), documents, chunks  # type: ignore[arg-type]


class TestReroutedDocuments:
    async def test_a_source_moved_to_another_profile_reports_its_orphan(self) -> None:
        batch, documents, _ = _rerouting_batch()
        await batch.execute(TenantId("t"), [_source("a", to="old")])
        (moved,) = await batch.execute(TenantId("t"), [_source("a", to="new")])
        assert (moved.status, moved.profile) == ("indexed", "new")
        assert moved.rerouted_from == ("old",)
        assert moved.removed_rerouted is False
        assert {d.profile for d in documents.items.values()} == {"old", "new"}

    async def test_cleanup_deletes_the_orphan_and_its_chunks(self) -> None:
        batch, documents, chunks = _rerouting_batch()
        await batch.execute(TenantId("t"), [_source("a", to="old")])
        (moved,) = await batch.execute(
            TenantId("t"), [_source("a", to="new")], cleanup_rerouted=True
        )
        assert moved.rerouted_from == ("old",)
        assert moved.removed_rerouted is True
        assert {d.profile for d in documents.items.values()} == {"new"}
        assert {c.profile for c, _ in chunks.items} == {"new"}

    async def test_other_sources_and_tenants_are_left_alone(self) -> None:
        batch, documents, _ = _rerouting_batch()
        await batch.execute(TenantId("t"), [_source("b", to="old")])
        await batch.execute(TenantId("other"), [_source("a", to="old")])
        (first,) = await batch.execute(
            TenantId("t"), [_source("a", to="new")], cleanup_rerouted=True
        )
        assert first.rerouted_from == ()
        assert len(documents.items) == 3

    async def test_a_failed_ingest_never_removes_the_old_version(self) -> None:
        batch, documents, _ = _rerouting_batch()
        await batch.execute(TenantId("t"), [_source("a", to="old")])
        (failed,) = await batch.execute(
            TenantId("t"), [_source("a", b"", to="new")], cleanup_rerouted=True
        )
        assert failed.status == "failed"
        assert {d.profile for d in documents.items.values()} == {"old"}

    async def test_an_unchanged_source_reports_nothing(self) -> None:
        batch, _, _ = _rerouting_batch()
        await batch.execute(TenantId("t"), [_source("a", to="old")])
        (again,) = await batch.execute(TenantId("t"), [_source("a", to="old")])
        assert (again.status, again.rerouted_from) == ("skipped", ())


class TestPrune:
    async def _seeded(self) -> tuple[IngestBatch, InMemoryDocumentRepository]:
        batch, documents, _ = _rerouting_batch()
        await batch.execute(
            TenantId("t"), [_source("a", to="old"), _source("b", to="old"), _source("c", to="new")]
        )
        # "b" is stale for tenant t but is not in the batch: only tenant isolation keeps it.
        await batch.execute(TenantId("other"), [_source("a", to="old"), _source("b", to="old")])
        return batch, documents

    async def test_prunes_the_stale_uri_and_keeps_the_current_ones(self) -> None:
        batch, documents = await self._seeded()
        tenant = TenantId("t")
        outcomes = await batch.execute(tenant, [_source("a", to="old"), _source("c", to="new")])
        removed = await batch.prune(tenant, ["old", "new"], outcomes)
        assert removed is not None
        assert [(d.profile, d.source_uri) for d in removed] == [("old", "b")]
        left = {(d.tenant_id, d.profile, d.source_uri) for d in documents.items.values()}
        assert left == {
            ("t", "old", "a"),
            ("t", "new", "c"),
            ("other", "old", "a"),
            ("other", "old", "b"),
        }

    async def test_never_touches_profiles_outside_the_given_ones(self) -> None:
        batch, documents = await self._seeded()
        tenant = TenantId("t")
        outcomes = await batch.execute(tenant, [_source("a", to="old")])
        removed = await batch.prune(tenant, ["old"], outcomes)
        assert removed is not None
        assert [d.source_uri for d in removed] == ["b"]
        assert any(d.profile == "new" and d.tenant_id == tenant for d in documents.items.values())

    async def test_is_skipped_when_the_batch_had_failures(self) -> None:
        batch, documents = await self._seeded()
        before = len(documents.items)
        outcomes = await batch.execute(TenantId("t"), [_source("a", b"", to="old")])
        assert await batch.prune(TenantId("t"), ["old"], outcomes) is None
        assert len(documents.items) == before

    async def test_an_empty_or_nothing_ingested_batch_deletes_nothing(self) -> None:
        batch, documents = await self._seeded()
        before = len(documents.items)
        assert await batch.prune(TenantId("t"), ["old", "new"], []) is None
        unrouted = await batch.execute(TenantId("t"), [_source("z", to="nowhere")])
        assert await batch.prune(TenantId("t"), ["old", "new"], unrouted) is None
        assert len(documents.items) == before

    async def test_an_unrouted_source_is_protected(self) -> None:
        batch, documents = await self._seeded()
        tenant = TenantId("t")
        outcomes = await batch.execute(tenant, [_source("a", to="old"), _source("b", to="nowhere")])
        assert outcomes[1].status == "unrouted"
        removed = await batch.prune(tenant, ["old", "new"], outcomes)
        assert removed is not None
        assert [d.source_uri for d in removed] == ["c"]
        assert any(d.source_uri == "b" and d.tenant_id == tenant for d in documents.items.values())

    async def test_a_rerouted_source_keeps_its_old_version_for_cleanup_to_own(self) -> None:
        batch, documents = await self._seeded()
        tenant = TenantId("t")
        outcomes = await batch.execute(tenant, [_source("a", to="new"), _source("c", to="new")])
        assert outcomes[0].rerouted_from == ("old",)
        removed = await batch.prune(tenant, ["old", "new"], outcomes)
        assert removed is not None
        assert [(d.profile, d.source_uri) for d in removed] == [("old", "b")]
        assert any(
            d.profile == "old" and d.source_uri == "a" and d.tenant_id == tenant
            for d in documents.items.values()
        )


class TestRevert:
    async def _seeded(self) -> tuple[IngestBatch, InMemoryDocumentRepository]:
        batch, documents, _ = _rerouting_batch()
        await batch.execute(
            TenantId("t"), [_source("a", to="old"), _source("b", to="old"), _source("c", to="new")]
        )
        await batch.execute(TenantId("other"), [_source("a", to="old")])
        return batch, documents

    async def test_a_dry_run_lists_and_deletes_nothing(self) -> None:
        batch, documents = await self._seeded()
        before = len(documents.items)
        found = await batch.revert(TenantId("t"), ["a", "c"], apply=False)
        assert sorted(d.source_uri for d in found) == ["a", "c"]
        assert len(documents.items) == before

    async def test_apply_deletes_only_the_listed_sources_of_the_tenant(self) -> None:
        batch, documents = await self._seeded()
        removed = await batch.revert(TenantId("t"), ["a", "c", "zzz"], apply=True)
        assert sorted(d.source_uri for d in removed) == ["a", "c"]
        left = {(d.tenant_id, d.source_uri) for d in documents.items.values()}
        assert left == {("t", "b"), ("other", "a")}

    async def test_lookups_are_chunked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        batch, documents = await self._seeded()
        monkeypatch.setattr(
            "agintis.modules.knowledge.application.ingest_batch.REVERT_LOOKUP_SIZE", 1
        )
        removed = await batch.revert(TenantId("t"), ["a", "b", "c"], apply=True)
        assert len(removed) == 3
        assert {d.tenant_id for d in documents.items.values()} == {"other"}


class TestStreaming:
    async def test_outcomes_are_yielded_as_the_sources_are_consumed(self) -> None:
        batch, _ = _batch()
        produced: list[str] = []

        def sources() -> Any:
            for uri in ("a", "b", "c"):
                produced.append(uri)
                yield _source(uri, kind="ok")

        seen: list[tuple[str, list[str]]] = []
        async for outcome in batch.stream(TenantId("t"), sources()):
            seen.append((outcome.uri, list(produced)))
        assert seen == [("a", ["a"]), ("b", ["a", "b"]), ("c", ["a", "b", "c"])]

    async def test_ready_made_outcomes_pass_through_in_order(self) -> None:
        batch, _ = _batch()
        broken = BatchOutcome("row 2", "failed", reason="InvalidMetadataError: bad row")
        outcomes = await batch.execute(
            TenantId("t"), [_source("a", kind="ok"), broken, _source("c", kind="ok")]
        )
        assert [(o.uri, o.status) for o in outcomes] == [
            ("a", "indexed"),
            ("row 2", "failed"),
            ("c", "indexed"),
        ]
        assert outcomes[1] is broken
