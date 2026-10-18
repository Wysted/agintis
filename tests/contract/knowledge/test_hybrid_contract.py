"""Contract of the lexical and BM25 branches and of hybrid (dense + lexical, RRF) search.

Same guarantees as the dense search: tenant isolation, READY-only serving, filters applied in
BOTH branches, deterministic order. The fake implements simplified lexical scorers (AND / OR of
accent-free terms); the texts below only use words every backend tokenizes the same way. The
Postgres BM25 branch needs pg_textsearch (the project's DB image); each tenant gets its own BM25
index at its first ingest.
"""

import asyncio
import json
import logging
import re
import secrets
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from psycopg import errors as pg_errors
from sqlalchemy import event, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from agintis.cli import ingest as ingest_cli
from agintis.modules.knowledge.api import BatchOutcome
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
    TextSearchConfigError,
)
from agintis.modules.knowledge.infrastructure.postgres.tables import bm25_index_name
from agintis.platform.db import create_session_factory
from agintis.shared_kernel.criteria import Filter, UnknownFieldError
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId
from tests.conftest import skip_or_fail
from tests.contract.knowledge.conftest import Repos
from tests.fakes.knowledge import InMemoryChunkRepository, InMemoryDocumentRepository
from tests.support.postgres import BM25_INDEXES_SQL, truncate_all
from tests.support.settings import FakeSettings

A = TenantId("tenant-a")
B = TenantId("tenant-b")
QUERY_VECTOR = [1.0, 0.0]
NEAR = [1.0, 0.1]
FAR = [0.1, 1.0]
QUERY = "afiliación automática"
EXACT_TEXT = "El inicio de labores genera la afiliación automática al sistema"
OTHER_TEXT = "Las cotizaciones se pagan mensualmente"
SCHEMA = (MetadataField("area", "str", filterable=True),)
WRITER_ROLE = "agintis_test_bm25_writer"


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
        repos = Repos(PgDocumentRepository(factory), PgChunkRepository(factory, bm25_indexes=True))
        try:
            await repos.chunks.search_bm25(A, QUERY, None, k=1)
        except DBAPIError as exc:  # no pg_textsearch: an old pgvector-only server
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


_PYFORMAT = re.compile(r"%\((\w+)\)s")


def _generic(statement: str) -> str:
    """`statement` with psycopg's `%(name)s` placeholders as `$n`, for EXPLAIN (GENERIC_PLAN)."""
    names: list[str] = []

    def number(match: re.Match[str]) -> str:
        if match.group(1) not in names:
            names.append(match.group(1))
        return f"${names.index(match.group(1)) + 1}"

    return _PYFORMAT.sub(number, statement).replace("%%", "%")


async def _bm25_plan(
    repos: Repos, engine: AsyncEngine, *, generic: bool = False
) -> list[dict[str, Any]]:
    """Plan nodes of the statement `search_bm25` sends, explained on the same engine.

    `generic` explains the plan Postgres switches to for a statement prepared and run often
    (psycopg prepares after a few runs): it never sees the bound values.
    """
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
        if generic:
            explain = f"EXPLAIN (GENERIC_PLAN, FORMAT JSON) {_generic(statement)}"
            result = await conn.exec_driver_sql(explain)
        else:
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

    async def test_a_tenant_without_chunks_finds_nothing(self, bm25_repos: Repos) -> None:
        repos = bm25_repos
        await repos.documents.save(_doc(A, "d"))
        await repos.chunks.add_many([_chunk(A, "d", "c1", EXACT_TEXT)], [NEAR])
        assert await repos.chunks.search_bm25(TenantId(f"t-{uuid4().hex}"), QUERY, None, 5) == []

    async def test_a_hostile_tenant_id_is_data(self, bm25_repos: Repos) -> None:
        repos = bm25_repos
        hostile = TenantId(f"x'; DROP TABLE chunks; -- \\ %s {uuid4().hex}")
        await repos.documents.save(_doc(hostile, "dh"))
        await repos.documents.save(_doc(A, "da"))
        mine = _chunk(hostile, "dh", "c1", EXACT_TEXT)
        await repos.chunks.add_many([mine], [NEAR])
        await repos.chunks.add_many([_chunk(A, "da", "c1", EXACT_TEXT)], [NEAR])
        assert [h.chunk.id for h in await repos.chunks.search_bm25(hostile, QUERY, None, 5)] == [
            mine.id
        ]
        assert {h.chunk.tenant_id for h in await repos.chunks.search_bm25(A, QUERY, None, 5)} == {A}

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
        plan = await _bm25_plan(repos, engine)
        assert bm25_index_name(A) not in {n.get("Index Name") for n in plan}
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
        assert bm25_index_name(A) in {n.get("Index Name") for n in nodes}
        # A `<@>` condition in a Filter re-scores every row the index scan returns.
        assert not [n["Filter"] for n in nodes if "<@>" in n.get("Filter", "")]

    async def test_a_generic_plan_can_still_use_the_partial_index(
        self, pg_bm25: tuple[Repos, AsyncEngine]
    ) -> None:
        """The tenant is a literal: from a bound `$n` the planner could not prove the index's
        `tenant_id = '<tenant>'` predicate."""
        repos, engine = pg_bm25
        await repos.documents.save(_doc(A, "d"))
        await repos.chunks.add_many([_chunk(A, "d", "c1", EXACT_TEXT)], [NEAR])
        nodes = await _bm25_plan(repos, engine, generic=True)
        assert bm25_index_name(A) in {n.get("Index Name") for n in nodes}


async def _bm25_indexes(engine: AsyncEngine) -> dict[str, str]:
    """{name: predicate} of the bm25 indexes on chunks."""
    async with engine.connect() as conn:
        rows = await conn.execute(text(BM25_INDEXES_SQL))
        return {str(name): str(predicate) for name, predicate in rows}


@pytest.mark.integration
class TestBm25TenantIndexes:
    """Postgres only: one partial BM25 index per tenant, created by its first ingest."""

    async def test_the_first_ingest_creates_the_tenants_index_and_later_ones_reuse_it(
        self, pg_bm25: tuple[Repos, AsyncEngine]
    ) -> None:
        repos, engine = pg_bm25
        tenant = TenantId(f"t-{uuid4().hex}")
        assert bm25_index_name(tenant) not in await _bm25_indexes(engine)
        for doc in ("d1", "d2", "d3"):
            await repos.documents.save(_doc(tenant, f"{tenant}-{doc}"))
        # Two first ingests at once must not fail on the catalog race.
        await asyncio.gather(
            *(
                repos.chunks.add_many(
                    [_chunk(tenant, f"{tenant}-d{n}", f"c{n}", EXACT_TEXT)], [NEAR]
                )
                for n in (1, 2)
            )
        )
        await repos.chunks.add_many([_chunk(tenant, f"{tenant}-d3", "c3", EXACT_TEXT)], [NEAR])
        indexes = await _bm25_indexes(engine)
        assert indexes[bm25_index_name(tenant)] == f"(tenant_id = '{tenant}'::text)"
        assert len(await repos.chunks.search_bm25(tenant, QUERY, None, 10)) == 3

    async def test_term_weights_never_mix_tenants(self, pg_bm25: tuple[Repos, AsyncEngine]) -> None:
        """A term frequent in tenant A keeps its full weight in tenant B."""
        repos, _ = pg_bm25
        a, b = TenantId(f"a-{uuid4().hex}"), TenantId(f"b-{uuid4().hex}")
        await repos.documents.save(_doc(b, f"{b}-d"))
        target = _chunk(b, f"{b}-d", "c0", "La afiliación es obligatoria")
        fillers = [_chunk(b, f"{b}-d", f"c{n}", OTHER_TEXT) for n in range(1, 4)]
        await repos.chunks.add_many([target, *fillers], [NEAR] * 4)
        (before,) = await repos.chunks.search_bm25(b, "afiliación", None, 10)

        await repos.documents.save(_doc(a, f"{a}-d"))
        common = [
            Chunk(ChunkId(f"{a}-{n}"), DocumentId(f"{a}-d"), a, "p", n, f"afiliación {OTHER_TEXT}")
            for n in range(30)
        ]
        await repos.chunks.add_many(common, [NEAR] * len(common))
        (after,) = await repos.chunks.search_bm25(b, "afiliación", None, 10)
        assert after.chunk.id == target.id
        assert after.score == pytest.approx(before.score)

        # Control: the same term made common in b ITSELF does lower its weight there.
        await repos.documents.save(_doc(b, f"{b}-d2"))
        own = [
            Chunk(ChunkId(f"{b}-{n}"), DocumentId(f"{b}-d2"), b, "p", n, f"afiliación {OTHER_TEXT}")
            for n in range(30)
        ]
        await repos.chunks.add_many(own, [NEAR] * len(own))
        hits = await repos.chunks.search_bm25(b, "afiliación", None, 50)
        (diluted,) = [h for h in hits if h.chunk.id == target.id]
        assert diluted.score < before.score

    async def test_a_tenant_with_chunks_but_no_index_says_how_to_fix_it(
        self, pg_bm25: tuple[Repos, AsyncEngine]
    ) -> None:
        repos, engine = pg_bm25
        tenant = TenantId(f"o'b-{uuid4().hex}")
        await repos.documents.save(_doc(tenant, f"{tenant}-d"))
        await repos.chunks.add_many([_chunk(tenant, f"{tenant}-d", "c1", EXACT_TEXT)], [NEAR])
        async with engine.begin() as conn:  # e.g. ingested before pg_textsearch was loaded
            await conn.execute(text(f'DROP INDEX "{bm25_index_name(tenant)}"'))
        with pytest.raises(TextSearchConfigError, match="no BM25 index") as info:
            await repos.chunks.search_bm25(tenant, QUERY, None, 10)
        assert "downgrade" not in str(info.value)  # never an outage as the remedy
        assert "re-run the ingest of this tenant as the owner of chunks" in str(info.value)
        # The statement in the message is the fix, as written.
        (statement,) = re.findall(r"`(CREATE INDEX[^`]+)`", str(info.value))
        async with engine.begin() as conn:
            await conn.exec_driver_sql(statement.replace("%", "%%"))
        assert len(await repos.chunks.search_bm25(tenant, QUERY, None, 10)) == 1

    async def test_an_ingest_run_of_unchanged_documents_builds_a_missing_index(
        self,
        pg_bm25: tuple[Repos, AsyncEngine],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The remedy the error names: re-running the ingest, even with nothing to re-ingest."""
        repos, engine = pg_bm25
        tenant = TenantId(f"t-{uuid4().hex}")
        await repos.documents.save(_doc(tenant, f"{tenant}-d"))
        await repos.chunks.add_many([_chunk(tenant, f"{tenant}-d", "c1", EXACT_TEXT)], [NEAR])
        async with engine.begin() as conn:  # e.g. ingested while no profile used bm25_hybrid
            await conn.execute(text(f'DROP INDEX "{bm25_index_name(tenant)}"'))

        seen: list[object] = []

        class EveryDocumentUnchanged:
            async def stream(self, tenant: object, items: Any, **kw: object) -> Any:
                for item in items:
                    seen.append(item)
                    yield BatchOutcome(item.uri, "skipped")  # unchanged: add_many never runs

        class IngestContainer:
            ingest_batch = EveryDocumentUnchanged()
            chunks = repos.chunks

            @classmethod
            def build(cls, settings: object, *, for_ingest: bool) -> "IngestContainer":
                return cls()

            async def __aenter__(self) -> "IngestContainer":
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

        index = tmp_path / "index" / "documents.jsonl"
        index.parent.mkdir()
        index.write_text('{"id": "d", "title": "D"}\n', encoding="utf-8")
        samples = tmp_path / "samples"
        (samples / "docs").mkdir(parents=True)
        (samples / "docs" / "d.txt").write_text(EXACT_TEXT, encoding="utf-8")
        manifest = [
            {
                "id": "d",
                "family": "docs",
                "path": "docs/d.txt",
                "url": "https://example.test/d",
                "content_type": "text/plain",
            }
        ]
        (samples / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        monkeypatch.setattr(ingest_cli, "get_settings", FakeSettings)
        monkeypatch.setattr(ingest_cli, "Container", IngestContainer)

        async def ingest(run_tenant: str) -> None:
            argv = ["--profiles", "p", "--index", str(index), "--samples", str(samples)]
            args = ingest_cli.build_parser().parse_args([*argv, "--tenant", run_tenant])
            await ingest_cli.run(args, lambda outcome: None)

        await ingest(tenant)
        assert seen  # the documents arrived, all unchanged
        assert not any(isinstance(item, BatchOutcome) for item in seen)
        assert bm25_index_name(tenant) in await _bm25_indexes(engine)
        assert len(await repos.chunks.search_bm25(tenant, QUERY, None, 10)) == 1

        # A tenant without chunks (a typo, an empty run) gets no full-table build, no orphan.
        empty = TenantId(f"t-{uuid4().hex}")
        await ingest(empty)
        assert bm25_index_name(empty) not in await _bm25_indexes(engine)

    async def test_the_index_build_waits_for_the_lock_on_chunks_only_so_long(
        self, pg_bm25: tuple[Repos, AsyncEngine]
    ) -> None:
        """A writer holding chunks must fail the first ingest, not queue every later writer."""
        repos, engine = pg_bm25
        quick = PgChunkRepository(
            create_session_factory(engine), bm25_indexes=True, bm25_lock_timeout_seconds=0.2
        )
        tenant = TenantId(f"t-{uuid4().hex}")
        await repos.documents.save(_doc(tenant, f"{tenant}-d"))
        async with engine.connect() as holder:
            await holder.execute(text("LOCK chunks IN ROW EXCLUSIVE MODE"))
            with pytest.raises(OperationalError, match="lock timeout"):
                await asyncio.wait_for(
                    quick.add_many([_chunk(tenant, f"{tenant}-d", "c1", EXACT_TEXT)], [NEAR]), 10
                )
            await holder.rollback()
        assert bm25_index_name(tenant) not in await _bm25_indexes(engine)

    async def test_a_search_racing_the_first_ingest_retries_once(
        self, pg_bm25: tuple[Repos, AsyncEngine], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The index was missing at plan time but exists by the follow-up check."""
        repos, _ = pg_bm25
        tenant = TenantId(f"t-{uuid4().hex}")
        await repos.documents.save(_doc(tenant, f"{tenant}-d"))
        await repos.chunks.add_many([_chunk(tenant, f"{tenant}-d", "c1", EXACT_TEXT)], [NEAR])
        chunks = repos.chunks
        assert isinstance(chunks, PgChunkRepository)
        real = chunks._bm25_rows
        calls: list[object] = []

        async def first_misses(stmt: Any) -> Any:
            calls.append(stmt)
            if len(calls) == 1:
                raise DBAPIError("stmt", None, pg_errors.UndefinedObject("index does not exist"))
            return await real(stmt)

        monkeypatch.setattr(chunks, "_bm25_rows", first_misses)
        assert len(await chunks.search_bm25(tenant, QUERY, None, 10)) == 1
        assert len(calls) == 2

    async def test_ingest_without_a_bm25_profile_creates_no_index(
        self, pg_bm25: tuple[Repos, AsyncEngine]
    ) -> None:
        repos, engine = pg_bm25
        plain = PgChunkRepository(create_session_factory(engine))  # no profile uses bm25_hybrid
        tenant = TenantId(f"t-{uuid4().hex}")
        await repos.documents.save(_doc(tenant, f"{tenant}-d"))
        await plain.add_many([_chunk(tenant, f"{tenant}-d", "c1", EXACT_TEXT)], [NEAR])
        assert bm25_index_name(tenant) not in await _bm25_indexes(engine)

    async def test_a_role_that_does_not_own_chunks_ingests_and_says_why_there_is_no_index(
        self,
        pg_bm25: tuple[Repos, AsyncEngine],
        pg_test_url: URL,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        repos, engine = pg_bm25
        password = secrets.token_hex(8)
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP ROLE IF EXISTS {WRITER_ROLE}"))
            await conn.execute(text(f"CREATE ROLE {WRITER_ROLE} LOGIN PASSWORD '{password}'"))
            await conn.execute(text(f"GRANT SELECT, INSERT ON chunks, documents TO {WRITER_ROLE}"))
        writer_url = pg_test_url.set(username=WRITER_ROLE, password=password)
        writer = create_async_engine(writer_url.render_as_string(hide_password=False))
        try:
            chunks = PgChunkRepository(create_session_factory(writer), bm25_indexes=True)
            tenant = TenantId(f"t-{uuid4().hex}")
            await repos.documents.save(_doc(tenant, f"{tenant}-d"))
            with caplog.at_level(logging.WARNING):
                for n in (1, 2):  # two documents' worth of chunks, one warning
                    await chunks.add_many(
                        [_chunk(tenant, f"{tenant}-d", f"c{n}", EXACT_TEXT)], [NEAR]
                    )
            assert caplog.text.count(f"role {WRITER_ROLE} does not own chunks") == 1
            assert bm25_index_name(tenant) not in await _bm25_indexes(engine)
        finally:
            await writer.dispose()
            async with engine.begin() as conn:
                await conn.execute(text(f"DROP OWNED BY {WRITER_ROLE}"))
                await conn.execute(text(f"DROP ROLE {WRITER_ROLE}"))


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
