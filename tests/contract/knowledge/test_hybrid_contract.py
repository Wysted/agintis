"""Contract of the lexical and BM25 branches and of hybrid (dense + lexical, RRF) search.

Same guarantees as the dense search: tenant isolation, READY-only serving, filters applied in
BOTH branches, deterministic order. The fake implements simplified lexical scorers (AND / OR of
accent-free terms); the texts below only use words every backend tokenizes the same way. The
Postgres BM25 branch needs pg_textsearch and its index (the project's DB image).
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from agintis.modules.knowledge.domain import (
    Chunk,
    Document,
    DocumentStatus,
    MetadataField,
    MetadataValue,
)
from agintis.modules.knowledge.infrastructure.postgres import (
    PgChunkRepository,
    PgDocumentRepository,
)
from agintis.modules.knowledge.infrastructure.postgres.tables import BM25_INDEX
from agintis.platform.db import create_session_factory
from agintis.shared_kernel.criteria import Filter, UnknownFieldError
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId
from tests.conftest import skip_or_fail
from tests.contract.knowledge.conftest import Repos
from tests.fakes.knowledge import InMemoryChunkRepository, InMemoryDocumentRepository
from tests.support.postgres import truncate_all

A = TenantId("tenant-a")
B = TenantId("tenant-b")
QUERY_VECTOR = [1.0, 0.0]
NEAR = [1.0, 0.1]
FAR = [0.1, 1.0]
QUERY = "afiliación automática"
EXACT_TEXT = "El inicio de labores genera la afiliación automática al sistema"
OTHER_TEXT = "Las cotizaciones se pagan mensualmente"
SCHEMA = (MetadataField("area", "str", filterable=True),)


def _doc(tenant: TenantId, doc_id: str, status: DocumentStatus = DocumentStatus.READY) -> Document:
    return Document(
        id=DocumentId(doc_id),
        tenant_id=tenant,
        profile="p",
        source_uri=f"file:///{doc_id}",
        title=doc_id,
        mime_type="text/plain",
        content_hash=f"hash-{doc_id}",
        profile_fingerprint="fp",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        status=status,
    )


def _chunk(tenant: TenantId, doc_id: str, name: str, text: str, **metadata: MetadataValue) -> Chunk:
    return Chunk(
        id=ChunkId(f"{tenant}-{name}"),
        document_id=DocumentId(doc_id),
        tenant_id=tenant,
        profile="p",
        ordinal=int(name[-1]) if name[-1].isdigit() else 0,
        text=text,
        metadata=metadata,
    )


async def _hybrid(repos: Repos, tenant: TenantId = A, expr: Filter | None = None, k: int = 10):  # type: ignore[no-untyped-def]
    return await repos.chunks.search_hybrid(
        tenant, QUERY_VECTOR, QUERY, expr, k, metadata_fields=SCHEMA
    )


class TestLexicalBranch:
    async def test_matches_terms_regardless_of_accents_and_ranks_best_first(
        self, repos: Repos
    ) -> None:
        await repos.documents.save(_doc(A, "d"))
        short = _chunk(A, "d", "c1", "afiliación automática")
        long = _chunk(A, "d", "c2", EXACT_TEXT + " y otras obligaciones del empleador")
        other = _chunk(A, "d", "c3", OTHER_TEXT)
        await repos.chunks.add_many([long, other, short], [NEAR, NEAR, NEAR])
        hits = await repos.chunks.search_lexical(A, "afiliacion automatica", None, k=10)
        assert {h.chunk.id for h in hits} == {short.id, long.id}
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
        assert all(h.score > 0 for h in hits)

    async def test_terms_are_combined_with_and_and_blank_queries_match_nothing(
        self, repos: Repos
    ) -> None:
        await repos.documents.save(_doc(A, "d"))
        await repos.chunks.add_many([_chunk(A, "d", "c1", EXACT_TEXT)], [NEAR])
        assert await repos.chunks.search_lexical(A, "afiliación inexistente", None, k=10) == []
        assert await repos.chunks.search_lexical(A, "   ", None, k=10) == []
        assert await repos.chunks.search_lexical(A, "", None, k=10) == []

    async def test_search_syntax_in_the_query_is_plain_text(self, repos: Repos) -> None:
        """The query is user input: no OR, negation, phrase or tsquery operator gets through."""
        await repos.documents.save(_doc(A, "d"))
        match = _chunk(A, "d", "c1", EXACT_TEXT)
        await repos.chunks.add_many([match], [NEAR])

        async def ids(query: str) -> list[ChunkId]:
            return [h.chunk.id for h in await repos.chunks.search_lexical(A, query, None, k=10)]

        assert await ids("afiliación OR inexistente") == []  # OR does not widen: still AND
        assert await ids("-automática afiliación") == [match.id]  # '-' does not negate
        assert await ids('"automática afiliación"') == [match.id]  # quotes are not a phrase
        for hostile in ("afiliación & | ! ( ) :* <-> \\", "'", "(", "!!", "a:*b", "<->"):
            await ids(hostile)  # never a syntax error

    async def test_only_the_requesting_tenant_and_only_ready_documents(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da"))
        await repos.documents.save(_doc(B, "db"))
        await repos.documents.save(_doc(A, "old", DocumentStatus.SUPERSEDED))
        await repos.documents.save(_doc(A, "new", DocumentStatus.PENDING))
        mine = _chunk(A, "da", "c1", EXACT_TEXT)
        await repos.chunks.add_many([mine], [NEAR])
        await repos.chunks.add_many([_chunk(B, "db", "c1", EXACT_TEXT)], [NEAR])
        await repos.chunks.add_many([_chunk(A, "old", "c2", EXACT_TEXT)], [NEAR])
        await repos.chunks.add_many([_chunk(A, "new", "c3", EXACT_TEXT)], [NEAR])
        hits = await repos.chunks.search_lexical(A, QUERY, None, k=10)
        assert [h.chunk.id for h in hits] == [mine.id]
        foreign = Filter.eq("tenant_id", B)
        assert await repos.chunks.search_lexical(A, QUERY, foreign, k=10) == []

    async def test_filters_and_validation(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "d"))
        sales = _chunk(A, "d", "c1", EXACT_TEXT, area="sales")
        hr = _chunk(A, "d", "c2", EXACT_TEXT, area="hr")
        await repos.chunks.add_many([sales, hr], [NEAR, NEAR])
        got = await repos.chunks.search_lexical(
            A, QUERY, Filter.eq("area", "hr"), k=10, metadata_fields=SCHEMA
        )
        assert [h.chunk.id for h in got] == [hr.id]
        with pytest.raises(UnknownFieldError):
            await repos.chunks.search_lexical(A, QUERY, Filter.eq("area", "hr"), k=10)
        with pytest.raises(ValueError, match="k must be"):
            await repos.chunks.search_lexical(A, QUERY, None, k=0)

    async def test_ties_break_by_chunk_id_and_k_limits(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "d"))
        chunks = [_chunk(A, "d", f"c{n}", EXACT_TEXT) for n in (3, 1, 2)]
        await repos.chunks.add_many(chunks, [NEAR] * 3)
        hits = await repos.chunks.search_lexical(A, QUERY, None, k=10)
        assert [h.chunk.id for h in hits] == sorted(c.id for c in chunks)
        assert len(await repos.chunks.search_lexical(A, QUERY, None, k=2)) == 2


# On these tiny tables the planner would seq-scan and score every row, never exercising the
# BM25 index path (its filter-after-scoring scan, its cached scores).
INDEX_PATH = "-c enable_seqscan=off -c enable_sort=off"
# The opposite: every tenant row is scored outside the index, a non-matching one at 0.
NO_INDEX_PATH = "-c enable_indexscan=off -c enable_bitmapscan=off"


@asynccontextmanager
async def _planner_forced(url: URL, options: str) -> AsyncIterator[tuple[Repos, AsyncEngine]]:
    """Postgres repositories whose sessions run with the given planner `options`."""
    engine = create_async_engine(
        url.render_as_string(hide_password=False), connect_args={"options": options}
    )
    try:
        factory = create_session_factory(engine)
        await truncate_all(factory)
        repos = Repos(PgDocumentRepository(factory), PgChunkRepository(factory))
        try:
            await repos.chunks.search_bm25(A, QUERY, None, k=1)
        except DBAPIError as exc:  # no pg_textsearch / no index: an old pgvector-only server
            skip_or_fail(f"BM25 search unavailable: {exc.orig.__class__.__name__}")
        yield repos, engine
    finally:
        await engine.dispose()


@pytest.fixture(params=["in_memory", pytest.param("postgres", marks=pytest.mark.integration)])
async def bm25_repos(request: pytest.FixtureRequest) -> AsyncIterator[Repos]:
    if request.param == "in_memory":
        documents = InMemoryDocumentRepository()
        yield Repos(documents, InMemoryChunkRepository(documents))
        return
    async with _planner_forced(request.getfixturevalue("pg_test_url"), INDEX_PATH) as (repos, _):
        yield repos


@pytest.fixture
async def pg_bm25(pg_test_url: URL) -> AsyncIterator[tuple[Repos, AsyncEngine]]:
    async with _planner_forced(pg_test_url, INDEX_PATH) as pair:
        yield pair


@pytest.fixture
async def pg_bm25_no_index(pg_test_url: URL) -> AsyncIterator[tuple[Repos, AsyncEngine]]:
    async with _planner_forced(pg_test_url, NO_INDEX_PATH) as pair:
        yield pair


def _plan_nodes(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield node
    for child in node.get("Plans", []):
        yield from _plan_nodes(child)


async def _bm25_plan(repos: Repos, engine: AsyncEngine) -> list[dict[str, Any]]:
    """Plan nodes of the statement `search_bm25` sends, explained on the same engine."""
    sent: list[tuple[str, Any]] = []

    def record(conn: Any, cursor: Any, statement: str, parameters: Any, *args: Any) -> None:
        if "to_bm25query" in statement:
            sent.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        await repos.chunks.search_bm25(A, QUERY, None, k=5)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)
    ((statement, parameters),) = sent
    async with engine.connect() as conn:
        result = await conn.exec_driver_sql(f"EXPLAIN (FORMAT JSON) {statement}", parameters)
        plan = result.scalar_one()
    return list(_plan_nodes(plan[0]["Plan"]))


class TestBm25Branch:
    async def test_any_term_matches_accents_aside_and_more_terms_rank_first(
        self, bm25_repos: Repos
    ) -> None:
        repos = bm25_repos
        await repos.documents.save(_doc(A, "d"))
        both = _chunk(A, "d", "c1", EXACT_TEXT)
        one = _chunk(A, "d", "c2", "La afiliacion es obligatoria para todos")
        other = _chunk(A, "d", "c3", OTHER_TEXT)
        await repos.chunks.add_many([other, one, both], [NEAR, NEAR, NEAR])
        hits = await repos.chunks.search_bm25(A, "afiliacion automatica", None, k=10)
        assert [h.chunk.id for h in hits] == [both.id, one.id]  # OR, unlike search_lexical
        assert hits[0].score > hits[1].score > 0

    async def test_queries_without_a_matching_term_return_nothing(self, bm25_repos: Repos) -> None:
        repos = bm25_repos
        await repos.documents.save(_doc(A, "d"))
        await repos.chunks.add_many([_chunk(A, "d", "c1", EXACT_TEXT)], [NEAR])
        assert await repos.chunks.search_bm25(A, "inexistente", None, k=10) == []
        assert await repos.chunks.search_bm25(A, "   ", None, k=10) == []
        assert await repos.chunks.search_bm25(A, "", None, k=10) == []
        for hostile in ("afiliación & | ! ( ) :* <-> \\", "'", "(", "a:*b", "<@>"):
            await repos.chunks.search_bm25(A, hostile, None, k=10)  # never a syntax error

    async def test_only_the_requesting_tenant_and_only_ready_documents(
        self, bm25_repos: Repos
    ) -> None:
        repos = bm25_repos
        await repos.documents.save(_doc(A, "da"))
        await repos.documents.save(_doc(B, "db"))
        await repos.documents.save(_doc(A, "old", DocumentStatus.SUPERSEDED))
        await repos.documents.save(_doc(A, "new", DocumentStatus.PENDING))
        mine = _chunk(A, "da", "c1", EXACT_TEXT)
        await repos.chunks.add_many([mine], [NEAR])
        await repos.chunks.add_many([_chunk(B, "db", "c1", EXACT_TEXT)], [NEAR])
        await repos.chunks.add_many([_chunk(A, "old", "c2", EXACT_TEXT)], [NEAR])
        await repos.chunks.add_many([_chunk(A, "new", "c3", EXACT_TEXT)], [NEAR])
        hits = await repos.chunks.search_bm25(A, QUERY, None, k=10)
        assert [h.chunk.id for h in hits] == [mine.id]
        foreign = Filter.eq("tenant_id", B)
        assert await repos.chunks.search_bm25(A, QUERY, foreign, k=10) == []

    async def test_filters_and_validation(self, bm25_repos: Repos) -> None:
        repos = bm25_repos
        await repos.documents.save(_doc(A, "d"))
        sales = _chunk(A, "d", "c1", EXACT_TEXT, area="sales")
        hr = _chunk(A, "d", "c2", EXACT_TEXT, area="hr")
        await repos.chunks.add_many([sales, hr], [NEAR, NEAR])
        got = await repos.chunks.search_bm25(
            A, QUERY, Filter.eq("area", "hr"), k=10, metadata_fields=SCHEMA
        )
        assert [h.chunk.id for h in got] == [hr.id]
        with pytest.raises(UnknownFieldError):
            await repos.chunks.search_bm25(A, QUERY, Filter.eq("area", "hr"), k=10)
        with pytest.raises(ValueError, match="k must be"):
            await repos.chunks.search_bm25(A, QUERY, None, k=0)

    async def test_ties_break_by_chunk_id_and_k_limits(self, bm25_repos: Repos) -> None:
        repos = bm25_repos
        await repos.documents.save(_doc(A, "d"))
        chunks = [_chunk(A, "d", f"c{n}", EXACT_TEXT) for n in (3, 1, 2)]
        await repos.chunks.add_many(chunks, [NEAR] * 3)
        hits = await repos.chunks.search_bm25(A, QUERY, None, k=10)
        assert [h.chunk.id for h in hits] == sorted(c.id for c in chunks)
        assert len(await repos.chunks.search_bm25(A, QUERY, None, k=2)) == 2


@pytest.mark.integration
class TestBm25WithoutIndex:
    async def test_chunks_scored_outside_the_index_never_match(
        self, pg_bm25_no_index: tuple[Repos, AsyncEngine]
    ) -> None:
        """Without the index Postgres scores every tenant row, a non-matching one at 0: the
        repository must drop it."""
        repos, engine = pg_bm25_no_index
        await repos.documents.save(_doc(A, "d"))
        match = _chunk(A, "d", "c1", EXACT_TEXT)
        await repos.chunks.add_many([match, _chunk(A, "d", "c2", OTHER_TEXT)], [NEAR, NEAR])
        assert BM25_INDEX not in {n.get("Index Name") for n in await _bm25_plan(repos, engine)}
        hits = await repos.chunks.search_bm25(A, QUERY, None, k=10)
        assert [h.chunk.id for h in hits] == [match.id]


@pytest.mark.integration
class TestBm25IndexPath:
    """Postgres only: the BM25 index serves the search and filters never cut a tenant short."""

    async def test_a_small_tenant_is_served_past_a_big_tenants_better_matches(
        self, pg_bm25: tuple[Repos, AsyncEngine]
    ) -> None:
        repos, _ = pg_bm25
        await repos.documents.save(_doc(B, "db"))
        await repos.documents.save(_doc(A, "da"))
        big = [
            Chunk(ChunkId(f"b{n:02d}"), DocumentId("db"), B, "p", n, f"{EXACT_TEXT} afiliación")
            for n in range(40)
        ]
        await repos.chunks.add_many(big, [NEAR] * len(big))
        mine = _chunk(A, "da", "c1", "La afiliación es obligatoria para todos los trabajadores")
        await repos.chunks.add_many([mine], [NEAR])
        hits = await repos.chunks.search_bm25(A, QUERY, None, k=1)
        assert [h.chunk.id for h in hits] == [mine.id]

    async def test_the_plan_uses_the_bm25_index_without_rescoring_in_a_filter(
        self, pg_bm25: tuple[Repos, AsyncEngine]
    ) -> None:
        repos, engine = pg_bm25
        await repos.documents.save(_doc(A, "d"))
        await repos.chunks.add_many([_chunk(A, "d", "c1", EXACT_TEXT)], [NEAR])
        nodes = await _bm25_plan(repos, engine)
        assert BM25_INDEX in {n.get("Index Name") for n in nodes}
        # A `<@>` condition in a Filter re-scores every row the index scan returns.
        assert not [n["Filter"] for n in nodes if "<@>" in n.get("Filter", "")]


class TestHybridSearch:
    async def test_a_chunk_found_by_both_branches_outranks_single_branch_chunks(
        self, repos: Repos
    ) -> None:
        await repos.documents.save(_doc(A, "d"))
        both = _chunk(A, "d", "c1", EXACT_TEXT)
        dense_only = _chunk(A, "d", "c2", OTHER_TEXT)
        lexical_only = _chunk(A, "d", "c3", EXACT_TEXT + " y otras obligaciones del empleador")
        await repos.chunks.add_many(
            [both, dense_only, lexical_only], [[1.0, 0.05], [1.0, 0.0], FAR]
        )
        hits = await _hybrid(repos)
        by_id = {h.chunk.id: h for h in hits}
        assert hits[0].chunk.id == both.id
        assert set(by_id) == {both.id, dense_only.id, lexical_only.id}
        assert by_id[both.id].dense_rank == 2
        assert by_id[both.id].lexical_rank is not None
        assert by_id[dense_only.id].lexical_rank is None
        assert by_id[dense_only.id].dense_rank == 1
        assert by_id[lexical_only.id].dense_similarity is not None  # dense depth covers all rows
        assert [h.fused_score for h in hits] == sorted((h.fused_score for h in hits), reverse=True)
        assert len(await _hybrid(repos, k=2)) == 2

    async def test_a_lexical_match_is_found_even_when_the_dense_branch_is_too_shallow(
        self, repos: Repos
    ) -> None:
        await repos.documents.save(_doc(A, "d"))
        noise = [_chunk(A, "d", f"c{n}", OTHER_TEXT) for n in range(1, 4)]
        needle = _chunk(A, "d", "c4", EXACT_TEXT)
        await repos.chunks.add_many([*noise, needle], [NEAR] * 3 + [FAR])
        hits = await repos.chunks.search_hybrid(
            A, QUERY_VECTOR, QUERY, None, 1, metadata_fields=SCHEMA, branch_depth=1
        )
        # k=1 raises each branch to depth 1: the dense branch only sees a noise chunk, yet the
        # lexical branch contributes the needle, and RRF ties are broken by id.
        assert len(hits) == 1
        full = await repos.chunks.search_hybrid(
            A, QUERY_VECTOR, QUERY, None, 4, metadata_fields=SCHEMA, branch_depth=1
        )
        assert needle.id in {h.chunk.id for h in full}

    async def test_tenants_never_mix(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "da"))
        await repos.documents.save(_doc(B, "db"))
        mine = _chunk(A, "da", "c1", OTHER_TEXT)
        await repos.chunks.add_many([mine], [FAR])
        await repos.chunks.add_many([_chunk(B, "db", "c1", EXACT_TEXT)], [QUERY_VECTOR])
        hits = await _hybrid(repos, A)
        assert {h.chunk.tenant_id for h in hits} == {A}
        assert {h.chunk.tenant_id for h in await _hybrid(repos, B)} == {B}

    async def test_filters_apply_inside_both_branches(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "d"))
        allowed_lexical = _chunk(A, "d", "c1", EXACT_TEXT, area="hr")
        blocked_lexical = _chunk(A, "d", "c2", EXACT_TEXT, area="sales")
        allowed_dense = _chunk(A, "d", "c3", OTHER_TEXT, area="hr")
        blocked_dense = _chunk(A, "d", "c4", OTHER_TEXT, area="sales")
        await repos.chunks.add_many(
            [allowed_lexical, blocked_lexical, allowed_dense, blocked_dense],
            [FAR, QUERY_VECTOR, NEAR, QUERY_VECTOR],
        )
        hits = await _hybrid(repos, expr=Filter.eq("area", "hr"))
        assert {h.chunk.id for h in hits} == {allowed_lexical.id, allowed_dense.id}

    async def test_only_ready_documents_are_served(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "live"))
        await repos.documents.save(_doc(A, "old", DocumentStatus.SUPERSEDED))
        live = _chunk(A, "live", "c1", EXACT_TEXT)
        await repos.chunks.add_many([live], [FAR])
        await repos.chunks.add_many([_chunk(A, "old", "c2", EXACT_TEXT)], [QUERY_VECTOR])
        assert [h.chunk.id for h in await _hybrid(repos)] == [live.id]

    async def test_equal_scores_are_ordered_by_id_and_repeatable(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "d"))
        chunks = [_chunk(A, "d", f"c{n}", EXACT_TEXT) for n in (3, 1, 2)]
        await repos.chunks.add_many(chunks, [NEAR] * 3)
        first = [h.chunk.id for h in await _hybrid(repos)]
        assert first == sorted(c.id for c in chunks)
        assert [h.chunk.id for h in await _hybrid(repos)] == first

    async def test_validation(self, repos: Repos) -> None:
        with pytest.raises(UnknownFieldError):
            await repos.chunks.search_hybrid(A, QUERY_VECTOR, QUERY, Filter.eq("nope", 1), 3)
        with pytest.raises(ValueError, match="k must be"):
            await repos.chunks.search_hybrid(A, QUERY_VECTOR, QUERY, None, 0)

    async def test_blank_text_degrades_to_the_dense_ranking(self, repos: Repos) -> None:
        await repos.documents.save(_doc(A, "d"))
        close, far = _chunk(A, "d", "c1", OTHER_TEXT), _chunk(A, "d", "c2", EXACT_TEXT)
        await repos.chunks.add_many([far, close], [FAR, NEAR])
        hits = await repos.chunks.search_hybrid(A, QUERY_VECTOR, " ", None, 5)
        assert [h.chunk.id for h in hits] == [close.id, far.id]
        assert all(h.lexical_rank is None for h in hits)
