"""Contract every knowledge repository backend must satisfy (tenant isolation first)."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from agintis.agent_kit.nodes import clauses_to_filter
from agintis.modules.knowledge.domain import (
    Chunk,
    Document,
    DocumentIdentityError,
    DocumentNotFoundError,
    DocumentStatus,
    DuplicateDocumentError,
    MetadataField,
    MetadataValue,
    TenantMismatchError,
    conform_metadata,
)
from agintis.shared_kernel.criteria import (
    AllOf,
    AnyOf,
    Criteria,
    Filter,
    FilterExpr,
    Order,
    OrderDirection,
    UnknownFieldError,
    tenant_scoped,
)
from agintis.shared_kernel.criteria.in_memory import matches
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId
from tests.contract.knowledge.conftest import Repos

A = TenantId("tenant-a")
B = TenantId("tenant-b")
VECTOR = [1.0, 0.0]


def _doc(
    tenant: TenantId, doc_id: str, *, content_hash: str | None = None, **kw: object
) -> Document:
    base: dict[str, object] = {
        "id": DocumentId(doc_id),
        "tenant_id": tenant,
        "profile": "p",
        "source_uri": f"file:///{doc_id}",
        "title": doc_id,
        "mime_type": "text/plain",
        "content_hash": content_hash or f"hash-{doc_id}",
        "profile_fingerprint": "fp",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "status": DocumentStatus.READY,
    }
    return Document(**{**base, **kw})  # type: ignore[arg-type]


def _chunk(
    tenant: TenantId,
    doc_id: str,
    ordinal: int,
    page: int | None = None,
    **extra: MetadataValue,
) -> Chunk:
    metadata: dict[str, MetadataValue] = dict(extra)
    if page is not None:
        metadata["page"] = page
    return Chunk(
        id=ChunkId(f"{tenant}-{doc_id}-{ordinal}"),
        document_id=DocumentId(doc_id),
        tenant_id=tenant,
        profile="p",
        ordinal=ordinal,
        text=f"text {ordinal}",
        metadata=metadata,
    )


async def _seed_chunks(repos: Repos) -> None:
    await repos.documents.save(_doc(A, "da"))
    await repos.documents.save(_doc(B, "db"))
    await repos.chunks.add_many(
        [_chunk(A, "da", 0, page=1), _chunk(A, "da", 1, page=2)], [VECTOR] * 2
    )
    await repos.chunks.add_many([_chunk(B, "db", 0, page=1), _chunk(B, "db", 1)], [VECTOR] * 2)


def _tenants(hits: list) -> set[str]:  # type: ignore[type-arg]
    return {h.chunk.tenant_id for h in hits}


class TestChunkRepository:
    async def test_search_only_returns_the_requesting_tenant(self, repos: Repos) -> None:
        await _seed_chunks(repos)
        hits = await repos.chunks.search(B, VECTOR, None, k=10)
        assert len(hits) == 2
        assert _tenants(hits) == {B}

    async def test_top_level_any_of_cannot_escape_the_tenant(self, repos: Repos) -> None:
        await _seed_chunks(repos)
        expr = AnyOf(Filter.eq("page", 1), Filter.eq("page", 2))
        hits = await repos.chunks.search(B, VECTOR, expr, k=10)
        assert _tenants(hits) == {B}
        assert len(hits) == 1

    async def test_filter_naming_another_tenant_matches_nothing(self, repos: Repos) -> None:
        await _seed_chunks(repos)
        assert await repos.chunks.search(B, VECTOR, Filter.eq("tenant_id", A), k=10) == []
        mixed = AnyOf(Filter.eq("tenant_id", A), Filter.eq("page", 1))
        assert _tenants(await repos.chunks.search(B, VECTOR, mixed, k=10)) == {B}
        assert await repos.chunks.search(B, VECTOR, Filter.in_("tenant_id", [A]), k=10) == []

    async def test_missing_metadata_matches_negative_operators(self, repos: Repos) -> None:
        await _seed_chunks(repos)
        neq = await repos.chunks.search(B, VECTOR, Filter.neq("page", 1), k=10)
        assert [h.chunk.ordinal for h in neq] == [1]  # chunk without page matches NEQ
        not_in = await repos.chunks.search(B, VECTOR, Filter.not_in("page", [1]), k=10)
        assert [h.chunk.ordinal for h in not_in] == [1]
        assert await repos.chunks.search(B, VECTOR, Filter.eq("page", 3), k=10) == []

    async def test_search_respects_k_and_orders_by_similarity(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da"))
        close, far = _chunk(A, "da", 0), _chunk(A, "da", 1)
        await repos.chunks.add_many([close, far], [[1.0, 0.0], [0.0, 1.0]])
        hits = await repos.chunks.search(A, [1.0, 0.1], None, k=1)
        assert [h.chunk.id for h in hits] == [close.id]

    async def test_unknown_field_is_rejected(self, repos: Repos) -> None:
        with pytest.raises(UnknownFieldError):
            await repos.chunks.search(A, VECTOR, Filter.eq("nope", 1), k=1)

    async def test_add_many_rejects_mixed_tenants(self, repos: Repos) -> None:
        with pytest.raises(ValueError, match="same tenant"):
            await repos.chunks.add_many([_chunk(A, "da", 0), _chunk(B, "db", 0)], [VECTOR] * 2)

    async def test_add_many_requires_a_document_of_the_chunks_tenant_and_profile(
        self, repos: Repos
    ) -> None:
        await repos.documents.save(_doc(A, "da"))
        # Missing document, document of ANOTHER tenant, and another profile: the same error,
        # so the caller cannot probe which ids exist for other tenants.
        wrong_profile = replace(_chunk(A, "da", 0), profile="other")
        for chunk in (_chunk(A, "ghost", 0), _chunk(B, "da", 0), wrong_profile):
            with pytest.raises(DocumentNotFoundError):
                await repos.chunks.add_many([chunk], [VECTOR])
        assert await repos.chunks.search(A, VECTOR, None, k=10) == []
        assert await repos.chunks.search(B, VECTOR, None, k=10) == []
        # The failed batch is atomic: a valid chunk next to an invalid one is not stored.
        with pytest.raises(DocumentNotFoundError):
            await repos.chunks.add_many([_chunk(A, "da", 1), _chunk(A, "ghost", 2)], [VECTOR] * 2)
        assert await repos.chunks.search(A, VECTOR, None, k=10) == []
        await repos.chunks.add_many([_chunk(A, "da", 0)], [VECTOR])  # the happy path still works
        assert len(await repos.chunks.search(A, VECTOR, None, k=10)) == 1

    async def test_delete_by_document_is_tenant_scoped(self, repos: Repos) -> None:
        await _seed_chunks(repos)
        await repos.chunks.delete_by_document(B, DocumentId("da"))  # wrong tenant: no-op
        assert len(await repos.chunks.search(A, VECTOR, None, k=10)) == 2
        await repos.chunks.delete_by_document(A, DocumentId("da"))
        assert await repos.chunks.search(A, VECTOR, None, k=10) == []
        assert len(await repos.chunks.search(B, VECTOR, None, k=10)) == 2


ARTICLE = MetadataField("article", "str", filterable=True)
CHAPTER = MetadataField("chapter", "str")  # declared but NOT filterable
EFFECTIVE = MetadataField("effective_date", "date", filterable=True, indexed=True)
WEIGHT = MetadataField("weight", "float", filterable=True)
LAWS = MetadataField("laws", "list[str]", filterable=True)
REFS = MetadataField("refs", "list[int]", filterable=True)
SCHEMA = (ARTICLE, CHAPTER, EFFECTIVE, WEIGHT, LAWS, REFS)


async def _seed_typed(repos: Repos) -> None:
    await repos.documents.save(_doc(A, "da"))
    await repos.documents.save(_doc(B, "db"))
    await repos.chunks.add_many(
        [
            _chunk(
                A,
                "da",
                0,
                page=1,
                article="12",
                effective_date="2020-01-01",
                weight=0.5,
                laws=("DL 3500", "Ley 21735"),
                refs=(31, 7),
            ),
            _chunk(
                A,
                "da",
                1,
                page=2,
                article="12 bis",
                effective_date="2024-06-30",
                weight=2,
                laws=("Ley 21735",),
                refs=(7,),
            ),
            _chunk(A, "da", 2, page=10, article="Z", chapter="I", laws=(), refs=()),
            _chunk(A, "da", 3),
        ],
        [VECTOR] * 4,
    )
    await repos.chunks.add_many([_chunk(B, "db", 0, page=1, article="12")], [VECTOR])


async def _ordinals(repos: Repos, expr: AnyOf | AllOf | Filter) -> list[int]:
    hits = await repos.chunks.search(A, VECTOR, expr, k=10, metadata_fields=SCHEMA)
    return sorted(h.chunk.ordinal for h in hits)


class TestServedVersions:
    async def test_search_serves_only_chunks_of_ready_documents(self, repos: Repos) -> None:
        for doc_id, status in (
            ("dr", DocumentStatus.READY),
            ("dp", DocumentStatus.PENDING),
            ("ds", DocumentStatus.SUPERSEDED),
        ):
            await repos.documents.save(_doc(A, doc_id, source_uri=doc_id, status=status))
            await repos.chunks.add_many([_chunk(A, doc_id, 0)], [VECTOR])
        hits = await repos.chunks.search(A, VECTOR, None, k=10)
        assert [h.chunk.document_id for h in hits] == ["dr"]

    async def test_chunks_follow_the_status_of_their_document(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "new", status=DocumentStatus.PENDING))
        await repos.chunks.add_many([_chunk(A, "new", 0)], [VECTOR])
        assert await repos.chunks.search(A, VECTOR, None, k=10) == []
        assert await repos.documents.mark_ready(A, DocumentId("new"), 1) is True
        assert len(await repos.chunks.search(A, VECTOR, None, k=10)) == 1


class TestProfileScopedMetadata:
    async def test_declared_filterable_fields_are_allowed(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.eq("article", "12")) == [0]
        assert await _ordinals(repos, Filter.eq("page", 10)) == [2]

    async def test_undeclared_or_non_filterable_fields_are_unknown(self, repos: Repos) -> None:
        await _seed_typed(repos)
        for field in ("chapter", "nope"):
            with pytest.raises(UnknownFieldError):
                await repos.chunks.search(
                    A, VECTOR, Filter.eq(field, "I"), k=1, metadata_fields=SCHEMA
                )
        # Without the profile schema `article` is not filterable at all.
        with pytest.raises(UnknownFieldError):
            await repos.chunks.search(A, VECTOR, Filter.eq("article", "12"), k=1)

    async def test_declared_fields_never_leak_across_tenants(self, repos: Repos) -> None:
        await _seed_typed(repos)
        hits = await repos.chunks.search(
            B, VECTOR, Filter.eq("article", "12"), k=10, metadata_fields=SCHEMA
        )
        assert _tenants(hits) == {B}
        assert len(hits) == 1


class TestMetadataFilterSemantics:
    async def test_equality_and_membership(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.in_("article", ["12", "Z"])) == [0, 2]
        # NEQ / NOT_IN match missing values (oracle semantics).
        assert await _ordinals(repos, Filter.neq("article", "12")) == [1, 2, 3]
        assert await _ordinals(repos, Filter.not_in("article", ["12", "12 bis"])) == [2, 3]

    async def test_types_are_strict(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.eq("page", True)) == []  # bool != int
        assert await _ordinals(repos, Filter.eq("page", "1")) == []  # str != int
        assert await _ordinals(repos, Filter.neq("page", "1")) == [0, 1, 2, 3]
        assert await _ordinals(repos, Filter.eq("page", 1.0)) == [0]  # int and float compare
        assert await _ordinals(repos, Filter.eq("article", 12)) == []
        assert await _ordinals(repos, Filter.gt("article", 1)) == []

    async def test_numeric_ordering_and_missing_values(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.gt("page", 1)) == [1, 2]  # 10 > 2 numerically
        assert await _ordinals(repos, Filter.lte("page", 2)) == [0, 1]
        assert await _ordinals(repos, Filter.gte("weight", 0.5)) == [0, 1]  # int 2 vs float
        assert await _ordinals(repos, Filter.lt("weight", 1)) == [0]

    async def test_text_ordering_is_by_code_point_not_locale(self, repos: Repos) -> None:
        await _seed_typed(repos)
        # "Z" < "a" in code-point order; many locales would put "Z" after "12 bis" anyway.
        assert await _ordinals(repos, Filter.gt("article", "12")) == [1, 2]
        assert await _ordinals(repos, Filter.lt("article", "a")) == [0, 1, 2]

    async def test_dates_are_iso_strings_compared_as_text(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.gte("effective_date", "2022-01-01")) == [1]
        assert await _ordinals(repos, Filter.lt("effective_date", "2022-01-01")) == [0]
        assert await _ordinals(repos, Filter.eq("effective_date", "2024-06-30")) == [1]

    async def test_composites(self, repos: Repos) -> None:
        await _seed_typed(repos)
        expr = AllOf(
            AnyOf(Filter.eq("article", "12"), Filter.eq("article", "Z")), Filter.gt("page", 1)
        )
        assert await _ordinals(repos, expr) == [2]
        assert await _ordinals(repos, AnyOf(Filter.eq("page", 1), Filter.eq("article", "Z"))) == [
            0,
            2,
        ]

    async def test_hostile_values_are_bound_not_interpolated(self, repos: Repos) -> None:
        await _seed_typed(repos)
        payload = "x'; DROP TABLE chunks; --"
        assert await _ordinals(repos, Filter.eq("article", payload)) == []
        assert len(await repos.chunks.search(A, VECTOR, None, k=10)) == 4

    async def test_core_columns_filter_and_text_ordering(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.gte("ordinal", 2)) == [2, 3]
        assert await _ordinals(repos, Filter.eq("profile", "p")) == [0, 1, 2, 3]
        assert await _ordinals(repos, Filter.eq("profile", "q")) == []
        assert await _ordinals(repos, Filter.not_in("text", ["text 0"])) == [1, 2, 3]


class TestListMetadataSemantics:
    """List fields are sets: EQ = contains, IN = overlaps; the negations include missing."""

    async def test_equality_is_containment(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.eq("laws", "Ley 21735")) == [0, 1]
        assert await _ordinals(repos, Filter.eq("laws", "DL 3500")) == [0]
        assert await _ordinals(repos, Filter.eq("refs", 7)) == [0, 1]
        assert await _ordinals(repos, Filter.eq("refs", 7.0)) == [0, 1]  # int and float compare
        assert await _ordinals(repos, Filter.eq("laws", "Ley")) == []  # whole items only

    async def test_membership_is_overlap(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.in_("laws", ["DL 3500", "nope"])) == [0]
        assert await _ordinals(repos, Filter.in_("refs", [31, 99])) == [0]
        assert await _ordinals(repos, Filter.in_("refs", [7, 31])) == [0, 1]

    async def test_negations_match_empty_and_missing(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.neq("laws", "Ley 21735")) == [2, 3]
        assert await _ordinals(repos, Filter.not_in("refs", [31, 99])) == [1, 2, 3]
        assert await _ordinals(repos, Filter.not_in("laws", ["DL 3500", "Ley 21735"])) == [2, 3]

    async def test_types_are_strict_and_ordering_never_matches(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.eq("refs", "7")) == []
        assert await _ordinals(repos, Filter.neq("refs", "7")) == [0, 1, 2, 3]
        assert await _ordinals(repos, Filter.eq("laws", 3500)) == []
        assert await _ordinals(repos, Filter.gt("refs", 0)) == []
        assert await _ordinals(repos, Filter.lte("laws", "ZZ")) == []

    async def test_hostile_values_are_bound(self, repos: Repos) -> None:
        await _seed_typed(repos)
        assert await _ordinals(repos, Filter.eq("laws", "x'}; DROP TABLE chunks; --")) == []
        assert len(await repos.chunks.search(A, VECTOR, None, k=10)) == 4

    async def test_tenants_stay_apart(self, repos: Repos) -> None:
        await _seed_typed(repos)
        hits = await repos.chunks.search(
            B, VECTOR, Filter.eq("laws", "Ley 21735"), k=10, metadata_fields=SCHEMA
        )
        assert hits == []


class TestDocumentRepository:
    async def test_the_metadata_hash_round_trips_and_may_be_absent(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "d1", metadata_hash="abc"))
        await repos.documents.save(_doc(A, "d2"))
        stored = await repos.documents.get(A, DocumentId("d1"))
        assert stored is not None
        assert stored.metadata_hash == "abc"
        legacy = await repos.documents.get(A, DocumentId("d2"))
        assert legacy is not None
        assert legacy.metadata_hash is None

    async def test_an_upsert_updates_the_metadata_hash(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "d1", status=DocumentStatus.PENDING))
        await repos.documents.save(
            _doc(A, "d1", status=DocumentStatus.PENDING, metadata_hash="new")
        )
        stored = await repos.documents.get(A, DocumentId("d1"))
        assert stored is not None
        assert stored.metadata_hash == "new"

    async def test_get_and_matching_are_tenant_scoped(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da"))
        await repos.documents.save(_doc(B, "db"))
        assert await repos.documents.get(B, DocumentId("da")) is None
        assert [d.id for d in await repos.documents.matching(B, Criteria())] == ["db"]

    async def test_matching_top_level_any_of_and_foreign_tenant_filters(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da"))
        await repos.documents.save(_doc(B, "db"))
        escape = Criteria(filter=AnyOf(Filter.eq("tenant_id", A), Filter.eq("id", "db")))
        assert [d.id for d in await repos.documents.matching(B, escape)] == ["db"]
        foreign = Criteria(filter=AllOf(Filter.eq("tenant_id", A)))
        assert await repos.documents.matching(B, foreign) == []

    async def test_matching_orders_and_paginates(self, repos: Repos) -> None:
        for name in ("d1", "d2", "d3"):
            await repos.documents.save(_doc(A, name))
        criteria = Criteria(order=(Order("id"),), offset=1, limit=1)
        assert [d.id for d in await repos.documents.matching(A, criteria)] == ["d2"]

    async def test_unknown_field_is_rejected(self, repos: Repos) -> None:
        with pytest.raises(UnknownFieldError):
            await repos.documents.matching(A, Criteria(filter=Filter.eq("nope", 1)))
        with pytest.raises(UnknownFieldError):
            await repos.documents.matching(A, Criteria(order=(Order("nope"),)))

    async def test_find_by_source_is_tenant_and_profile_scoped_and_returns_all_versions(
        self, repos: Repos
    ) -> None:
        old = _doc(A, "d1", source_uri="s")
        new = _doc(A, "d2", source_uri="s", status=DocumentStatus.PENDING)
        await repos.documents.save(old)
        await repos.documents.save(new)
        found = await repos.documents.find_by_source(A, "p", "s")
        assert {d.id for d in found} == {"d1", "d2"}
        assert await repos.documents.find_by_source(B, "p", "s") == []
        assert await repos.documents.find_by_source(A, "other", "s") == []
        assert await repos.documents.find_by_source(A, "p", "nope") == []

    async def test_save_is_an_upsert_by_id(self, repos: Repos) -> None:
        pending = _doc(A, "da", status=DocumentStatus.PENDING)
        await repos.documents.save(pending)
        await repos.documents.save(pending.mark_ready(2))
        stored = await repos.documents.get(A, DocumentId("da"))
        assert stored is not None
        assert (stored.status, stored.chunk_count) == (DocumentStatus.READY, 2)

    async def test_save_is_tenant_guarded(self, repos: Repos) -> None:
        original = _doc(A, "da", title="original")
        await repos.documents.save(original)
        with pytest.raises(TenantMismatchError):
            await repos.documents.save(_doc(B, "da", title="hijacked"))
        assert await repos.documents.get(A, DocumentId("da")) == original
        assert await repos.documents.get(B, DocumentId("da")) is None

    async def test_save_cannot_change_the_identity_of_a_document(self, repos: Repos) -> None:
        original = _doc(A, "da", source_uri="s", title="original")
        await repos.documents.save(original)
        for moved in (
            _doc(A, "da", source_uri="elsewhere"),
            _doc(A, "da", source_uri="s", profile="q"),
        ):
            with pytest.raises(DocumentIdentityError):
                await repos.documents.save(moved)
        assert await repos.documents.get(A, DocumentId("da")) == original
        await repos.documents.save(_doc(A, "da", source_uri="s", title="renamed"))  # still upserts
        stored = await repos.documents.get(A, DocumentId("da"))
        assert stored is not None
        assert stored.title == "renamed"

    async def test_partial_unique_on_ready_source(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "d1", source_uri="s"))
        with pytest.raises(DuplicateDocumentError):
            await repos.documents.save(_doc(A, "d2", source_uri="s"))
        # Same hash from another source, another profile or another tenant is fine.
        await repos.documents.save(_doc(A, "d3", source_uri="other", content_hash="hash-d1"))
        await repos.documents.save(_doc(A, "d4", source_uri="s", profile="q"))
        await repos.documents.save(_doc(B, "d5", source_uri="s"))

    async def test_pending_and_superseded_do_not_conflict_with_ready(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "d1", source_uri="s"))
        await repos.documents.save(_doc(A, "d2", source_uri="s", status=DocumentStatus.PENDING))
        await repos.documents.save(_doc(A, "d3", source_uri="s", status=DocumentStatus.PENDING))
        await repos.documents.save(_doc(A, "d4", source_uri="s", status=DocumentStatus.SUPERSEDED))
        assert len(await repos.documents.find_by_source(A, "p", "s")) == 4

    async def test_mark_ready_promotes_only_pending(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da", status=DocumentStatus.PENDING))
        assert await repos.documents.mark_ready(A, DocumentId("da"), 3) is True
        stored = await repos.documents.get(A, DocumentId("da"))
        assert stored is not None
        assert (stored.status, stored.chunk_count) == (DocumentStatus.READY, 3)
        # Conditional: a second call (already READY) does not apply and does not rewrite.
        assert await repos.documents.mark_ready(A, DocumentId("da"), 99) is False
        again = await repos.documents.get(A, DocumentId("da"))
        assert again is not None
        assert again.chunk_count == 3

    async def test_mark_ready_never_resurrects_missing_or_superseded(self, repos: Repos) -> None:
        assert await repos.documents.mark_ready(A, DocumentId("ghost"), 1) is False
        assert await repos.documents.get(A, DocumentId("ghost")) is None
        await repos.documents.save(_doc(A, "dold", status=DocumentStatus.SUPERSEDED))
        assert await repos.documents.mark_ready(A, DocumentId("dold"), 1) is False
        old = await repos.documents.get(A, DocumentId("dold"))
        assert old is not None
        assert old.status is DocumentStatus.SUPERSEDED

    async def test_mark_ready_is_tenant_scoped(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da", status=DocumentStatus.PENDING))
        assert await repos.documents.mark_ready(B, DocumentId("da"), 1) is False
        stored = await repos.documents.get(A, DocumentId("da"))
        assert stored is not None
        assert stored.status is DocumentStatus.PENDING

    async def test_update_title_changes_only_the_title_of_a_ready_document(
        self, repos: Repos
    ) -> None:
        original = _doc(A, "da", title="GetFile")
        await repos.documents.save(original)
        assert await repos.documents.update_title(A, DocumentId("da"), "REF 372") is True
        stored = await repos.documents.get(A, DocumentId("da"))
        assert stored == replace(original, title="REF 372")

    async def test_update_title_is_tenant_scoped_and_ready_only(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da", title="orig"))
        await repos.documents.save(_doc(A, "dp", source_uri="p", status=DocumentStatus.PENDING))
        assert await repos.documents.update_title(B, DocumentId("da"), "leak") is False
        assert await repos.documents.update_title(A, DocumentId("dp"), "x") is False
        assert await repos.documents.update_title(A, DocumentId("ghost"), "x") is False
        stored = await repos.documents.get(A, DocumentId("da"))
        assert stored is not None
        assert stored.title == "orig"

    async def test_mark_ready_atomically_supersedes_the_previous_ready_version(
        self, repos: Repos
    ) -> None:
        await repos.documents.save(_doc(A, "old", source_uri="s", chunk_count=2))
        await repos.documents.save(_doc(A, "new", source_uri="s", status=DocumentStatus.PENDING))
        await repos.documents.save(_doc(A, "elsewhere", source_uri="other"))

        assert await repos.documents.mark_ready(A, DocumentId("new"), 4) is True

        status = {d.id: d.status for d in await repos.documents.matching(A, Criteria())}
        assert status == {
            "old": DocumentStatus.SUPERSEDED,
            "new": DocumentStatus.READY,
            "elsewhere": DocumentStatus.READY,
        }

    async def test_delete_is_tenant_scoped_and_idempotent(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da"))
        await repos.documents.delete(B, DocumentId("da"))
        assert await repos.documents.get(A, DocumentId("da")) is not None
        await repos.documents.delete(A, DocumentId("da"))
        await repos.documents.delete(A, DocumentId("da"))
        assert await repos.documents.get(A, DocumentId("da")) is None

    async def test_matching_filters_every_core_type(self, repos: Repos) -> None:
        early = datetime(2026, 1, 1, tzinfo=UTC)
        late = datetime(2026, 6, 1, tzinfo=UTC)
        await repos.documents.save(_doc(A, "d1", created_at=early, chunk_count=1))
        await repos.documents.save(_doc(A, "d2", created_at=late, chunk_count=5))
        await repos.documents.save(
            _doc(A, "d3", created_at=late, status=DocumentStatus.PENDING, chunk_count=0)
        )

        async def ids(expr: Filter | AnyOf | AllOf) -> list[str]:
            return [d.id for d in await repos.documents.matching(A, Criteria(filter=expr))]

        assert await ids(Filter.gt("created_at", early)) == ["d2", "d3"]
        assert await ids(Filter.lte("created_at", early)) == ["d1"]
        assert await ids(Filter.eq("status", DocumentStatus.PENDING)) == ["d3"]
        assert await ids(Filter.gte("chunk_count", 1)) == ["d1", "d2"]
        assert await ids(Filter.in_("id", ["d1", "d3"])) == ["d1", "d3"]
        assert await ids(Filter.not_in("status", ["ready"])) == ["d3"]
        # Strict types: a naive datetime / a str never order against a timestamptz.
        assert await ids(Filter.gt("created_at", datetime(2000, 1, 1))) == []
        assert await ids(Filter.eq("chunk_count", "5")) == []
        assert await ids(Filter.neq("chunk_count", "5")) == ["d1", "d2", "d3"]

    async def test_matching_orders_desc_with_id_tiebreak(self, repos: Repos) -> None:
        same = datetime(2026, 3, 1, tzinfo=UTC)
        for name in ("d2", "d1", "d3"):
            await repos.documents.save(_doc(A, name, created_at=same, title="t"))
        criteria = Criteria(order=(Order("title", OrderDirection.DESC),))
        assert [d.id for d in await repos.documents.matching(A, criteria)] == ["d1", "d2", "d3"]


class TestConverterMatchesTheOracle:
    """For data conforming to the declared types, every backend equals the in-memory oracle."""

    SCHEMA = (
        MetadataField("article", "str", filterable=True),
        MetadataField("year", "int", filterable=True),
        MetadataField("weight", "float", filterable=True),
        MetadataField("effective", "date", filterable=True),
        MetadataField("laws", "list[str]", filterable=True),
        MetadataField("refs", "list[int]", filterable=True),
    )
    # Raw values as an extractor would produce them (regex groups are strings).
    RAW: tuple[dict[str, Any], ...] = (
        {
            "article": "12",
            "year": "2024",
            "weight": 0.5,
            "effective": "2020-01-01",
            "page": 1,
            "laws": ["DL 3500", "Ley 21735"],
            "refs": ["31", 7],
        },
        {
            "article": "12 bis",
            "year": 1999,
            "weight": 2,
            "effective": "2024-06-30",
            "page": "2",
            "laws": ["Ley 21735"],
            "refs": [7],
        },
        {
            "article": "Z",
            "year": "-5",
            "weight": "0.1",
            "effective": "2021-12-31",
            "page": 10,
            "laws": [],
            "refs": [],
        },
        {"article": "ñandú", "year": 2024, "weight": 1e20, "page": 3, "laws": ["DL 3500"]},
        {"article": "a%_b", "weight": -3.5, "effective": "1999-01-01"},
        {},
    )

    FILTERS: tuple[FilterExpr, ...] = (
        Filter.eq("article", "12"),
        Filter.neq("article", "12"),
        Filter.in_("article", ["12", "Z", "ñandú"]),
        Filter.not_in("article", ["12", "Z"]),
        Filter.gt("article", "12"),
        Filter.lte("article", "Z"),
        Filter.eq("article", "a%_b"),
        Filter.eq("article", 12),  # wrong kind: never equal
        Filter.neq("article", 12),
        Filter.eq("year", 2024),
        Filter.eq("year", 2024.0),  # int and float compare numerically
        Filter.gt("year", 1999),
        Filter.lte("year", -5),
        Filter.in_("year", [1999, 2024]),
        Filter.not_in("year", [2024]),
        Filter.eq("year", "2024"),
        Filter.neq("year", "2024"),
        Filter.eq("year", True),
        Filter.gte("weight", 0.1),
        Filter.lt("weight", 1),
        Filter.eq("weight", 0.1),
        Filter.eq("weight", 2),
        Filter.gt("weight", 1e19),
        Filter.in_("weight", [0.5, 0.1]),
        Filter.gte("effective", "2021-01-01"),
        Filter.lt("effective", "2021-01-01"),
        Filter.eq("effective", "2024-06-30"),
        Filter.neq("effective", "2024-06-30"),
        Filter.eq("page", 10),
        Filter.gt("page", 2),  # numeric, not lexicographic: 10 > 2
        Filter.not_in("page", [1, 3]),
        Filter.gte("ordinal", 2),
        Filter.gt("ordinal", 1.5),
        Filter.in_("ordinal", [0, 5]),
        Filter.eq("text", "text 3"),
        Filter.not_in("text", ["text 0", "text 1"]),
        AllOf(Filter.gt("year", 0), AnyOf(Filter.eq("article", "12"), Filter.eq("page", 10))),
        AnyOf(
            Filter.eq("year", 2024), Filter.lt("weight", 0), Filter.eq("effective", "1999-01-01")
        ),
        # List fields behave as sets: EQ = contains, IN = overlaps, NEQ / NOT_IN = negations
        # (a missing field and an empty list both match them), ordering never matches.
        Filter.eq("laws", "DL 3500"),
        Filter.neq("laws", "DL 3500"),
        Filter.eq("laws", "dl 3500"),
        Filter.in_("laws", ["Ley 21735", "DL 9"]),
        Filter.not_in("laws", ["Ley 21735", "DL 9"]),
        Filter.eq("laws", 3500),  # wrong kind: never contained
        Filter.neq("laws", 3500),
        Filter.in_("laws", [3500, 1]),
        Filter.not_in("laws", [3500]),
        Filter.gt("laws", "A"),
        Filter.lte("laws", "Z"),
        Filter.eq("laws", "x') OR 1=1 --"),
        Filter.eq("refs", 7),
        Filter.eq("refs", 7.0),
        Filter.eq("refs", 31),
        Filter.neq("refs", 31),
        Filter.in_("refs", [31, 99]),
        Filter.not_in("refs", [7]),
        Filter.eq("refs", "7"),
        Filter.neq("refs", "7"),
        Filter.lt("refs", 100),
        AnyOf(Filter.eq("laws", "DL 3500"), Filter.eq("refs", 7)),
        AllOf(Filter.neq("laws", "Ley 21735"), Filter.not_in("refs", [31])),
    )

    async def test_search_equals_the_oracle_for_declared_type_data(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da"))
        chunks = [
            replace(
                _chunk(A, "da", ordinal),
                metadata=conform_metadata(raw, self.SCHEMA),
            )
            for ordinal, raw in enumerate(self.RAW)
        ]
        await repos.chunks.add_many(chunks, [VECTOR] * len(chunks))

        def record(chunk: Chunk) -> dict[str, object]:
            core = {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "tenant_id": chunk.tenant_id,
                "profile": chunk.profile,
                "ordinal": chunk.ordinal,
                "text": chunk.text,
            }
            return {**chunk.metadata, **core}

        for expr in self.FILTERS:
            expected = sorted(
                c.ordinal for c in chunks if matches(tenant_scoped(A, expr), record(c))
            )
            hits = await repos.chunks.search(
                A, VECTOR, expr, k=len(chunks), metadata_fields=self.SCHEMA
            )
            assert sorted(h.chunk.ordinal for h in hits) == expected, expr


class TestExtractedFilters:
    """What the agent's filter extractor produces (`clauses_to_filter`) means the same on every
    backend: list fields (contains), date and int ranges, a point range normalized to `eq`."""

    async def _ordinals_for(self, repos: Repos, clauses: list[Any]) -> list[int]:
        expr = clauses_to_filter(clauses)
        assert expr is not None
        return await _ordinals(repos, expr)

    async def test_list_fields_match_by_containment(self, repos: Repos) -> None:
        await _seed_typed(repos)
        laws: Any = {"field": "laws", "op": "eq", "value": "Ley 21735"}
        refs: Any = {"field": "refs", "op": "eq", "value": 31}
        assert await self._ordinals_for(repos, [laws]) == [0, 1]
        assert await self._ordinals_for(repos, [refs]) == [0]

    async def test_a_date_range_is_inclusive(self, repos: Repos) -> None:
        await _seed_typed(repos)
        both: Any = [
            {"field": "effective_date", "op": "gte", "value": "2020-01-01"},
            {"field": "effective_date", "op": "lte", "value": "2021-01-01"},
        ]
        assert await self._ordinals_for(repos, both) == [0]
        assert await self._ordinals_for(repos, both[:1]) == [0, 1]

    async def test_an_int_range_and_the_point_range_normalized_to_eq(self, repos: Repos) -> None:
        await _seed_typed(repos)
        span: Any = [
            {"field": "page", "op": "gte", "value": 1},
            {"field": "page", "op": "lte", "value": 2},
        ]
        point: Any = [
            {"field": "page", "op": "gte", "value": 2},
            {"field": "page", "op": "lte", "value": 2},
        ]
        assert await self._ordinals_for(repos, span) == [0, 1]
        assert clauses_to_filter(point) == Filter.eq("page", 2)
        assert await self._ordinals_for(repos, point) == [1]

    async def test_clauses_combine_with_and(self, repos: Repos) -> None:
        await _seed_typed(repos)
        clauses: Any = [
            {"field": "laws", "op": "eq", "value": "Ley 21735"},
            {"field": "page", "op": "gte", "value": 2},
            {"field": "effective_date", "op": "lte", "value": "2030-01-01"},
        ]
        assert await self._ordinals_for(repos, clauses) == [1]
        wrong: Any = [{"field": "article", "op": "eq", "value": "nope"}]
        assert await self._ordinals_for(repos, wrong) == []  # zero hits: the fallback's trigger
