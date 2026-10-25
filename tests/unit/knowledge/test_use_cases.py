import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from agintis.modules.knowledge.api import (
    ConcurrentIngestError,
    DocumentStatus,
    EmptyDocumentError,
    IngestDocument,
    RagProfile,
    RetrieveForTopic,
    SourceFile,
    UnknownStrategyError,
)
from agintis.modules.knowledge.application import (
    ChunkDraft,
    Chunker,
    ParsedDocument,
    metadata_hash,
)
from agintis.modules.knowledge.domain import MAX_TITLE_CHARS, Chunk, CleanerSpec
from agintis.modules.knowledge.infrastructure.parsers import SimpleParser
from agintis.shared_kernel.criteria import AnyOf, Filter, UnknownFieldError
from agintis.shared_kernel.ids import DocumentId, TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    FakeChunker,
    FakeCleaner,
    FakeParser,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryIngestLock,
    NullIngestLock,
)
from tests.support.documents import make_pdf

TENANT_A = TenantId("tenant-a")
TENANT_B = TenantId("tenant-b")
PROFILE = RagProfile(
    name="p",
    top_k=2,
    chunk_max_tokens=100,
    parser_key="fake",
    chunker_key="fake",
)
TEXT = "alpha paragraph\n\nbeta paragraph\n\ngamma paragraph"


def _source(
    text: str = TEXT, uri: str = "file:///doc.txt", document_metadata: dict[str, Any] | None = None
) -> SourceFile:
    return SourceFile(
        uri=uri,
        mime_type="text/plain",
        content=text.encode(),
        document_metadata=document_metadata or {},
    )


class ExplodingChunkRepository(InMemoryChunkRepository):
    """add_many stores a partial batch, then fails (simulates a crash mid-write)."""

    def __init__(self) -> None:
        super().__init__()
        self.failures_left = 0

    async def add_many(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if self.failures_left > 0:
            self.failures_left -= 1
            await super().add_many(chunks[:1], vectors[:1])
            raise RuntimeError("boom")
        await super().add_many(chunks, vectors)


class EmptyChunker(Chunker):
    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
        return []


class CountingEmbeddings(Embeddings):
    """Deterministic embeddings that record the size of every document batch."""

    def __init__(self) -> None:
        self._inner = DeterministicFakeEmbedding(size=16)
        self.batches: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


@dataclass
class Env:
    documents: InMemoryDocumentRepository
    chunks: ExplodingChunkRepository
    parser: FakeParser
    embeddings: CountingEmbeddings
    ingest: IngestDocument
    retrieve: RetrieveForTopic


@pytest.fixture
def env() -> Env:
    documents = InMemoryDocumentRepository()
    chunks = ExplodingChunkRepository()
    parser = FakeParser()
    embeddings = CountingEmbeddings()
    return Env(
        documents,
        chunks,
        parser,
        embeddings,
        IngestDocument(
            documents,
            chunks,
            {"fake": parser},
            {"fake": FakeChunker()},
            embeddings,
            FixedClock(),
            InMemoryIngestLock(),
        ),
        RetrieveForTopic(documents, chunks, embeddings),
    )


class TestIngestDocument:
    async def test_happy_path(self, env: Env) -> None:
        result = await env.ingest.execute(TENANT_A, PROFILE, _source())

        assert (result.chunk_count, result.status) == (3, "indexed")
        saved = await env.documents.get(TENANT_A, result.document_id)
        assert saved is not None
        assert saved.title == "title of file:///doc.txt"
        assert saved.created_at == FixedClock().now()
        assert saved.status is DocumentStatus.READY
        assert saved.chunk_count == 3
        assert saved.profile_fingerprint == PROFILE.fingerprint
        stored = [c for c, _ in env.chunks.items]
        assert [c.ordinal for c in stored] == [0, 1, 2]
        assert {c.document_id for c in stored} == {result.document_id}
        assert stored[1].metadata == {"page": 2}

    async def test_chunk_ids_are_deterministic_and_unique(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source())
        ids = [c.id for c, _ in env.chunks.items]
        assert len(set(ids)) == 3

    async def test_same_content_is_skipped_with_existing_chunk_count(self, env: Env) -> None:
        first = await env.ingest.execute(TENANT_A, PROFILE, _source())
        second = await env.ingest.execute(TENANT_A, PROFILE, _source())

        assert (second.status, second.chunk_count) == ("skipped", 3)
        assert second.document_id == first.document_id
        assert env.parser.calls == 1
        assert len(env.chunks.items) == 3

    async def test_same_content_other_tenant_is_not_skipped(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source())
        other = await env.ingest.execute(TENANT_B, PROFILE, _source())
        assert other.status == "indexed"

    async def test_changed_content_same_source_replaces_previous_document(self, env: Env) -> None:
        first = await env.ingest.execute(TENANT_A, PROFILE, _source())
        changed = await env.ingest.execute(TENANT_A, PROFILE, _source(TEXT + "\n\ndelta"))

        assert (changed.status, changed.chunk_count) == ("replaced", 4)
        assert list(env.documents.items) == [changed.document_id]
        assert changed.document_id != first.document_id
        assert {c.document_id for c, _ in env.chunks.items} == {changed.document_id}
        assert len(env.chunks.items) == 4

    async def test_changed_content_other_source_keeps_both_documents(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source(uri="file:///a.txt"))
        other = await env.ingest.execute(TENANT_A, PROFILE, _source("zeta", uri="file:///b.txt"))

        assert other.status == "indexed"
        assert len(env.documents.items) == 2

    async def test_different_fingerprint_reprocesses_same_content(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source())
        retuned = replace(PROFILE, chunk_max_tokens=PROFILE.chunk_max_tokens + 1)

        result = await env.ingest.execute(TENANT_A, retuned, _source())

        assert result.status == "replaced"
        assert env.parser.calls == 2
        assert len(env.documents.items) == 1
        (document,) = env.documents.items.values()
        assert document.profile_fingerprint == retuned.fingerprint
        assert len(env.chunks.items) == 3

    async def test_baseline_strategies_leave_the_stored_fingerprint_unchanged(
        self, env: Env
    ) -> None:
        result = await env.ingest.execute(TENANT_A, PROFILE, _source())
        saved = await env.documents.get(TENANT_A, result.document_id)
        assert saved is not None
        assert saved.profile_fingerprint == PROFILE.fingerprint

    async def test_bumping_a_strategy_version_reprocesses_same_content(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source())

        class BumpedChunker(FakeChunker):
            version = 2

        bumped = IngestDocument(
            env.documents,
            env.chunks,
            {"fake": env.parser},
            {"fake": BumpedChunker()},
            env.embeddings,
            FixedClock(),
            InMemoryIngestLock(),
        )
        result = await bumped.execute(TENANT_A, PROFILE, _source())

        assert result.status == "replaced"
        assert env.parser.calls == 2
        (document,) = env.documents.items.values()
        assert document.profile_fingerprint == PROFILE.fingerprint_for({"chunker:fake": 2})
        again = await bumped.execute(TENANT_A, PROFILE, _source())
        assert again.status == "skipped"

    async def test_a_delegate_version_bump_reprocesses_the_delegating_chunker(
        self, env: Env
    ) -> None:
        def build(delegate_version: int) -> IngestDocument:
            class Delegating(FakeChunker):
                def dependency_versions(self) -> dict[str, int]:
                    return {"chunker:delegate": delegate_version}

            return IngestDocument(
                env.documents,
                env.chunks,
                {"fake": env.parser},
                {"fake": Delegating()},
                env.embeddings,
                FixedClock(),
                InMemoryIngestLock(),
            )

        assert (await build(1).execute(TENANT_A, PROFILE, _source())).status == "indexed"
        assert (await build(1).execute(TENANT_A, PROFILE, _source())).status == "skipped"
        assert (await build(2).execute(TENANT_A, PROFILE, _source())).status == "replaced"

    async def test_the_metadata_hash_is_stored_with_the_document(self, env: Env) -> None:
        source = _source(document_metadata={"number": 7, "tags": ["a", "b"]})
        result = await env.ingest.execute(TENANT_A, PROFILE, source)
        saved = await env.documents.get(TENANT_A, result.document_id)
        assert saved is not None
        assert saved.metadata_hash == metadata_hash(source.document_metadata)
        assert saved.metadata_hash != metadata_hash({})

    async def test_the_caller_title_wins_over_the_parser_title(self, env: Env) -> None:
        source = _source(document_metadata={"title": "  REF 372\t\n sobre   inversiones "})

        result = await env.ingest.execute(TENANT_A, PROFILE, source)

        saved = await env.documents.get(TENANT_A, result.document_id)
        assert saved is not None
        assert saved.title == "REF 372 sobre inversiones"

    @pytest.mark.parametrize("title", ["", "   \n", 7, None, ["a"]])
    async def test_an_unusable_caller_title_keeps_the_parser_title(
        self, env: Env, title: object
    ) -> None:
        source = _source(document_metadata={"title": title})

        result = await env.ingest.execute(TENANT_A, PROFILE, source)

        saved = await env.documents.get(TENANT_A, result.document_id)
        assert saved is not None
        assert saved.title == "title of file:///doc.txt"

    async def test_the_caller_title_is_bounded(self, env: Env) -> None:
        source = _source(document_metadata={"title": "x" * (MAX_TITLE_CHARS * 2)})

        result = await env.ingest.execute(TENANT_A, PROFILE, source)

        saved = await env.documents.get(TENANT_A, result.document_id)
        assert saved is not None
        assert saved.title == "x" * MAX_TITLE_CHARS

    async def test_a_pdf_without_title_takes_the_caller_title(self, env: Env) -> None:
        uri = "https://host/apps/GetFile.php?id=003&namefile=REF-X/NP0000372.pdf"
        ingest = IngestDocument(
            env.documents,
            env.chunks,
            {"fake": SimpleParser()},
            {"fake": FakeChunker()},
            env.embeddings,
            FixedClock(),
            InMemoryIngestLock(),
        )
        pdf = SourceFile(
            uri=uri,
            mime_type="application/pdf",
            content=make_pdf(["Some text of the norm."]),
            document_metadata={"title": "REF N° 372"},
        )

        result = await ingest.execute(TENANT_A, PROFILE, pdf)

        saved = await env.documents.get(TENANT_A, result.document_id)
        assert saved is not None
        assert saved.title == "REF N° 372"
        assert [c.text for c, _ in env.chunks.items] == ["Some text of the norm."]

    async def test_the_title_does_not_change_chunks_nor_the_fingerprint(self, env: Env) -> None:
        plain = await env.ingest.execute(TENANT_A, PROFILE, _source(uri="file:///a.txt"))
        titled = await env.ingest.execute(
            TENANT_A, PROFILE, _source(uri="file:///b.txt", document_metadata={"title": "T"})
        )

        a = await env.documents.get(TENANT_A, plain.document_id)
        b = await env.documents.get(TENANT_A, titled.document_id)
        assert a is not None
        assert b is not None
        assert a.profile_fingerprint == b.profile_fingerprint
        assert [c.text for c, _ in env.chunks.items][:3] == [c.text for c, _ in env.chunks.items][
            3:
        ]

    async def test_a_skipped_document_gets_its_title_refreshed(
        self, env: Env, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = await env.ingest.execute(TENANT_A, PROFILE, _source())
        stored = await env.documents.get(TENANT_A, first.document_id)
        assert stored is not None
        assert stored.title == "title of file:///doc.txt"
        calls, embedded = env.parser.calls, list(env.embeddings.batches)

        # The metadata hash covers the title, so a NEW title is a reprocess (replaced) while the
        # same hash with a stale stored title (rows ingested before this rule) is only refreshed.
        source = _source(document_metadata={"title": "Real title"})
        env.documents.items[first.document_id] = replace(
            stored, metadata_hash=metadata_hash(source.document_metadata)
        )
        with caplog.at_level(logging.INFO):
            again = await env.ingest.execute(TENANT_A, PROFILE, source)

        assert again.status == "skipped"
        assert again.title_refreshed
        assert str(first.document_id) in caplog.text
        assert PROFILE.name in caplog.text
        assert again.document_id == first.document_id
        refreshed = await env.documents.get(TENANT_A, first.document_id)
        assert refreshed is not None
        assert refreshed.title == "Real title"
        assert (env.parser.calls, env.embeddings.batches) == (calls, embedded)

        third = await env.ingest.execute(TENANT_A, PROFILE, source)
        assert third.status == "skipped"
        assert not third.title_refreshed

    async def test_the_refresh_log_truncates_long_titles(
        self, env: Env, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = await env.ingest.execute(TENANT_A, PROFILE, _source())
        stored = await env.documents.get(TENANT_A, first.document_id)
        assert stored is not None
        source = _source(document_metadata={"title": "z" * MAX_TITLE_CHARS})
        env.documents.items[first.document_id] = replace(
            stored, metadata_hash=metadata_hash(source.document_metadata)
        )

        with caplog.at_level(logging.INFO):
            await env.ingest.execute(TENANT_A, PROFILE, source)

        assert "z" * 61 not in caplog.text

    async def test_the_parser_title_is_normalized_like_the_caller_title(self, env: Env) -> None:
        env.parser.title = "  \u202eevil\u200b\ttitle\n" + "y" * (MAX_TITLE_CHARS * 2)

        result = await env.ingest.execute(TENANT_A, PROFILE, _source())

        saved = await env.documents.get(TENANT_A, result.document_id)
        assert saved is not None
        assert saved.title == ("evil title " + "y" * MAX_TITLE_CHARS)[:MAX_TITLE_CHARS]

    async def test_a_skipped_document_without_a_caller_title_keeps_its_title(
        self, env: Env
    ) -> None:
        first = await env.ingest.execute(TENANT_A, PROFILE, _source())

        again = await env.ingest.execute(TENANT_A, PROFILE, _source())

        assert again.status == "skipped"
        kept = await env.documents.get(TENANT_A, first.document_id)
        assert kept is not None
        assert kept.title == "title of file:///doc.txt"

    async def test_update_title_is_tenant_scoped_and_ready_only(self, env: Env) -> None:
        first = await env.ingest.execute(TENANT_A, PROFILE, _source())

        assert not await env.documents.update_title(TENANT_B, first.document_id, "leak")
        assert await env.documents.update_title(TENANT_A, first.document_id, "ok")
        stored = await env.documents.get(TENANT_A, first.document_id)
        assert stored is not None
        assert stored.title == "ok"

    async def test_a_cleaner_version_bump_reprocesses_only_profiles_that_use_it(
        self, env: Env
    ) -> None:
        class BumpedCleaner(FakeCleaner):
            version = 3

        def build(cleaner: FakeCleaner) -> IngestDocument:
            return IngestDocument(
                env.documents,
                env.chunks,
                {"fake": env.parser},
                {"fake": FakeChunker()},
                env.embeddings,
                FixedClock(),
                InMemoryIngestLock(),
                cleaners={"drop": cleaner},
            )

        with_cleaner = replace(PROFILE, cleaners=(CleanerSpec("drop", {"drop": "zzz"}),))
        await build(FakeCleaner()).execute(TENANT_A, with_cleaner, _source())
        assert (await build(BumpedCleaner()).execute(TENANT_A, with_cleaner, _source())).status == (
            "replaced"
        )
        # A profile without that cleaner is unaffected by the bump.
        await build(FakeCleaner()).execute(TENANT_A, PROFILE, _source(uri="file:///other.txt"))
        skipped = await build(BumpedCleaner()).execute(
            TENANT_A, PROFILE, _source(uri="file:///other.txt")
        )
        assert skipped.status == "skipped"

    async def test_retry_after_add_many_failure_succeeds(self, env: Env) -> None:
        env.chunks.failures_left = 1
        with pytest.raises(RuntimeError, match="boom"):
            await env.ingest.execute(TENANT_A, PROFILE, _source())

        result = await env.ingest.execute(TENANT_A, PROFILE, _source())

        assert (result.status, result.chunk_count) == ("indexed", 3)
        assert len(env.chunks.items) == 3
        assert {c.document_id for c, _ in env.chunks.items} == {result.document_id}

    async def test_failure_leaves_no_orphan_document_or_chunks(self, env: Env) -> None:
        env.chunks.failures_left = 1
        with pytest.raises(RuntimeError, match="boom"):
            await env.ingest.execute(TENANT_A, PROFILE, _source())

        assert env.documents.items == {}
        assert env.chunks.items == []

    async def test_leftover_pending_document_is_cleaned_and_reprocessed(self, env: Env) -> None:
        # Simulates a crash where compensation never ran.
        await env.ingest.execute(TENANT_A, PROFILE, _source())
        (ready,) = env.documents.items.values()
        env.documents.items[ready.id] = replace(ready, status=DocumentStatus.PENDING)

        result = await env.ingest.execute(TENANT_A, PROFILE, _source())

        assert result.status == "indexed"
        assert list(env.documents.items) == [result.document_id]
        assert len(env.chunks.items) == 3

    async def test_same_content_from_two_sources_keeps_both_and_is_independent(
        self, env: Env
    ) -> None:
        a = await env.ingest.execute(TENANT_A, PROFILE, _source(uri="file:///a.txt"))
        b = await env.ingest.execute(TENANT_A, PROFILE, _source(uri="file:///b.txt"))

        assert (a.status, b.status) == ("indexed", "indexed")
        assert a.document_id != b.document_id
        assert len(env.documents.items) == 2

        changed = await env.ingest.execute(TENANT_A, PROFILE, _source("new", uri="file:///a.txt"))

        assert changed.status == "replaced"
        assert set(env.documents.items) == {b.document_id, changed.document_id}
        assert {c.document_id for c, _ in env.chunks.items} == {b.document_id, changed.document_id}
        assert env.documents.items[b.document_id].status is DocumentStatus.READY

    async def test_previous_version_is_served_until_the_new_one_is_ready(self, env: Env) -> None:
        old = await env.ingest.execute(TENANT_A, PROFILE, _source("old text"))
        inserted, resume = asyncio.Event(), asyncio.Event()
        original = env.chunks.add_many

        async def gated(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
            await original(chunks, vectors)
            inserted.set()
            await resume.wait()

        env.chunks.add_many = gated  # type: ignore[method-assign]
        task = asyncio.create_task(env.ingest.execute(TENANT_A, PROFILE, _source("new text")))
        await inserted.wait()

        # New chunks are stored but its document is still PENDING: only the old text is served.
        during = await env.retrieve.execute(TENANT_A, PROFILE, "old text")
        assert {h.citation.document_id for h in during.hits} == {old.document_id}

        resume.set()
        new = await task
        after = await env.retrieve.execute(TENANT_A, PROFILE, "new text")
        assert new.status == "replaced"
        assert {h.citation.document_id for h in after.hits} == {new.document_id}
        assert set(env.documents.items) == {new.document_id}

    async def test_failed_replacement_keeps_the_previous_version(self, env: Env) -> None:
        old = await env.ingest.execute(TENANT_A, PROFILE, _source("old text"))
        env.chunks.failures_left = 1
        with pytest.raises(RuntimeError, match="boom"):
            await env.ingest.execute(TENANT_A, PROFILE, _source("new text"))

        assert set(env.documents.items) == {old.document_id}
        assert env.documents.items[old.document_id].status is DocumentStatus.READY
        assert {c.document_id for c, _ in env.chunks.items} == {old.document_id}

    async def test_superseded_leftovers_are_cleaned_by_the_next_ingest(self, env: Env) -> None:
        first = await env.ingest.execute(TENANT_A, PROFILE, _source("one"))
        env.documents.items[first.document_id] = replace(
            env.documents.items[first.document_id], status=DocumentStatus.SUPERSEDED
        )

        result = await env.ingest.execute(TENANT_A, PROFILE, _source("one"))

        assert result.status == "indexed"
        assert set(env.documents.items) == {result.document_id}
        assert {c.document_id for c, _ in env.chunks.items} == {result.document_id}

    async def test_concurrent_ingests_of_the_same_source_leave_one_consistent_ready_doc(
        self, env: Env
    ) -> None:
        inserted, resume = asyncio.Event(), asyncio.Event()
        original = env.chunks.add_many

        async def gated(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
            await original(chunks, vectors)
            inserted.set()
            await resume.wait()

        env.chunks.add_many = gated  # type: ignore[method-assign]
        first = asyncio.create_task(env.ingest.execute(TENANT_A, PROFILE, _source()))
        await inserted.wait()
        second = asyncio.create_task(env.ingest.execute(TENANT_A, PROFILE, _source()))
        for _ in range(10):  # let the second one run until it blocks on the lock
            await asyncio.sleep(0)
        assert not second.done()
        env.chunks.add_many = original  # type: ignore[method-assign]
        resume.set()

        r1, r2 = await asyncio.gather(first, second)

        assert (r1.status, r2.status) == ("indexed", "skipped")
        assert r1.document_id == r2.document_id
        (doc,) = env.documents.items.values()
        assert doc.status is DocumentStatus.READY
        assert doc.chunk_count == len(env.chunks.items) == 3

    async def test_without_a_lock_a_lost_transition_is_compensated_never_resurrected(
        self, env: Env
    ) -> None:
        """Repository-level protection: the reviewer's N1 interleaving with NO serialization."""
        unlocked = IngestDocument(
            env.documents,
            env.chunks,
            {"fake": env.parser},
            {"fake": FakeChunker()},
            env.embeddings,
            FixedClock(),
            NullIngestLock(),
        )
        inserted, resume = asyncio.Event(), asyncio.Event()
        original = env.chunks.add_many

        async def gated(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
            await original(chunks, vectors)
            inserted.set()
            await resume.wait()

        env.chunks.add_many = gated  # type: ignore[method-assign]
        a = asyncio.create_task(unlocked.execute(TENANT_A, PROFILE, _source()))
        await inserted.wait()
        env.chunks.add_many = original  # type: ignore[method-assign]
        # B finds A's PENDING row, treats it as a crashed attempt, cleans it and finishes.
        b_result = await unlocked.execute(TENANT_A, PROFILE, _source())
        resume.set()

        with pytest.raises(ConcurrentIngestError):
            await a

        assert b_result.status == "indexed"
        (doc,) = env.documents.items.values()
        assert (doc.id, doc.status, doc.chunk_count) == (
            b_result.document_id,
            DocumentStatus.READY,
            3,
        )
        assert {c.document_id for c, _ in env.chunks.items} == {b_result.document_id}
        assert len(env.chunks.items) == 3

    async def test_failed_transition_never_leaves_a_ready_document_without_chunks(
        self, env: Env
    ) -> None:
        async def refuse(tenant_id: TenantId, document_id: DocumentId, chunk_count: int) -> bool:
            return False

        env.documents.mark_ready = refuse  # type: ignore[method-assign]
        with pytest.raises(ConcurrentIngestError):
            await env.ingest.execute(TENANT_A, PROFILE, _source())
        assert env.documents.items == {}
        assert env.chunks.items == []

    async def test_compensation_failure_does_not_mask_the_original_error(
        self, env: Env, caplog: pytest.LogCaptureFixture
    ) -> None:
        env.chunks.failures_left = 1

        async def broken_delete(tenant_id: TenantId, document_id: DocumentId) -> None:
            raise OSError("cleanup down")

        env.chunks.delete_by_document = broken_delete  # type: ignore[method-assign]
        with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="boom"):
            await env.ingest.execute(TENANT_A, PROFILE, _source())

        assert "Compensation failed" in caplog.text
        assert "cleanup down" in caplog.text

    async def test_failure_deleting_the_superseded_version_does_not_fail_the_ingest(
        self, env: Env, caplog: pytest.LogCaptureFixture
    ) -> None:
        old = await env.ingest.execute(TENANT_A, PROFILE, _source("old"))
        original = env.documents.delete

        async def flaky(tenant_id: TenantId, document_id: DocumentId) -> None:
            if document_id == old.document_id:
                raise OSError("db hiccup")
            await original(tenant_id, document_id)

        env.documents.delete = flaky  # type: ignore[method-assign]
        with caplog.at_level(logging.ERROR):
            new = await env.ingest.execute(TENANT_A, PROFILE, _source("new"))

        assert new.status == "replaced"
        assert env.documents.items[old.document_id].status is DocumentStatus.SUPERSEDED
        assert env.documents.items[new.document_id].status is DocumentStatus.READY
        assert "superseded" in caplog.text

    async def test_empty_document_raises_without_side_effects(self, env: Env) -> None:
        ingest = IngestDocument(
            env.documents,
            env.chunks,
            {"fake": env.parser},
            {"fake": EmptyChunker()},
            env.embeddings,
            FixedClock(),
            InMemoryIngestLock(),
        )
        with pytest.raises(EmptyDocumentError):
            await ingest.execute(TENANT_A, PROFILE, _source())
        assert env.documents.items == {}
        assert env.chunks.items == []
        assert env.embeddings.batches == []

    async def test_embeds_in_batches(self, env: Env) -> None:
        ingest = IngestDocument(
            env.documents,
            env.chunks,
            {"fake": env.parser},
            {"fake": FakeChunker()},
            env.embeddings,
            FixedClock(),
            InMemoryIngestLock(),
            embed_batch_size=2,
        )
        await ingest.execute(TENANT_A, PROFILE, _source())
        assert env.embeddings.batches == [2, 1]

    def test_embed_batch_size_must_be_positive(self, env: Env) -> None:
        with pytest.raises(ValueError, match="embed_batch_size"):
            IngestDocument(
                env.documents,
                env.chunks,
                {},
                {},
                env.embeddings,
                FixedClock(),
                InMemoryIngestLock(),
                embed_batch_size=0,
            )

    @pytest.mark.parametrize(
        ("parser_key", "chunker_key", "kind"),
        [("nope", "fake", "parser"), ("fake", "nope", "chunker")],
    )
    async def test_unknown_strategy_key(
        self, env: Env, parser_key: str, chunker_key: str, kind: str
    ) -> None:
        profile = RagProfile("p", 2, 100, parser_key, chunker_key)
        with pytest.raises(UnknownStrategyError, match=kind):
            await env.ingest.execute(TENANT_A, profile, _source())
        assert env.documents.items == {}


class TestRetrieveForTopic:
    async def test_returns_top_k_ordered_with_exact_match_first(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source())

        result = await env.retrieve.execute(TENANT_A, PROFILE, "beta paragraph")

        assert len(result.hits) == PROFILE.top_k
        scores = [h.chunk.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)
        assert result.hits[0].chunk.chunk.text == "beta paragraph"
        assert result.hits[0].chunk.score == pytest.approx(1.0)

    async def test_citations_carry_source_and_page(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source(uri="file:///manual.pdf"))

        result = await env.retrieve.execute(TENANT_A, PROFILE, "gamma paragraph")

        top = result.hits[0].citation
        assert top.source_uri == "file:///manual.pdf"
        assert top.title == "title of file:///manual.pdf"
        assert top.page == 3
        assert top.snippet == "gamma paragraph"
        assert top.chunk_id == result.hits[0].chunk.chunk.id
        assert all(h.citation.chunk_id == h.chunk.chunk.id for h in result.hits)

    async def test_never_returns_other_tenant_chunks(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source("secret of A"))
        await env.ingest.execute(TENANT_B, PROFILE, _source("data of B"))

        result = await env.retrieve.execute(TENANT_B, PROFILE, "secret of A")

        assert result.hits
        assert {h.chunk.chunk.tenant_id for h in result.hits} == {TENANT_B}
        assert all(h.chunk.chunk.text != "secret of A" for h in result.hits)
        assert all(h.citation.source_uri for h in result.hits)

    async def test_tenant_without_data_gets_empty_result(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source())
        result = await env.retrieve.execute(TENANT_B, PROFILE, "alpha paragraph")
        assert result.hits == ()

    async def test_profile_default_filter_is_applied(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source())
        filtered = RagProfile("p", 5, 100, "fake", "fake", default_filter=Filter.eq("page", 2))

        result = await env.retrieve.execute(TENANT_A, filtered, "alpha paragraph")

        assert [h.chunk.chunk.metadata["page"] for h in result.hits] == [2]

    async def test_caller_filter_combines_with_default_filter(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source())
        profile = RagProfile("p", 5, 100, "fake", "fake", default_filter=Filter.gte("page", 2))

        result = await env.retrieve.execute(
            TENANT_A, profile, "alpha paragraph", extra_filter=Filter.lte("page", 2)
        )

        assert [h.chunk.chunk.metadata["page"] for h in result.hits] == [2]

    async def test_only_chunks_of_the_requested_profile(self, env: Env) -> None:
        other = RagProfile("other", 5, 100, "fake", "fake")
        await env.ingest.execute(TENANT_A, other, _source())

        result = await env.retrieve.execute(TENANT_A, PROFILE, "alpha paragraph")

        assert result.hits == ()

    async def test_extra_filter_naming_another_tenant_returns_nothing(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source("secret of A"))
        await env.ingest.execute(TENANT_B, PROFILE, _source("data of B"))

        result = await env.retrieve.execute(
            TENANT_B, PROFILE, "secret of A", extra_filter=Filter.eq("tenant_id", TENANT_A)
        )

        assert result.hits == ()

    async def test_top_level_any_of_extra_filter_cannot_escape_the_tenant(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source("secret of A"))
        await env.ingest.execute(TENANT_B, PROFILE, _source("data of B"))

        result = await env.retrieve.execute(
            TENANT_B,
            PROFILE,
            "secret of A",
            extra_filter=AnyOf(Filter.eq("page", 1), Filter.eq("tenant_id", TENANT_A)),
        )

        assert {h.chunk.chunk.tenant_id for h in result.hits} == {TENANT_B}

    async def test_unknown_filter_field_is_rejected(self, env: Env) -> None:
        with pytest.raises(UnknownFieldError):
            await env.retrieve.execute(TENANT_A, PROFILE, "x", extra_filter=Filter.eq("nope", 1))

    async def test_dangling_chunks_are_dropped(self, env: Env) -> None:
        result = await env.ingest.execute(TENANT_A, PROFILE, _source())
        await env.documents.delete(TENANT_A, result.document_id)

        assert (await env.retrieve.execute(TENANT_A, PROFILE, "alpha paragraph")).hits == ()

    async def test_chunks_of_pending_documents_are_dropped(self, env: Env) -> None:
        result = await env.ingest.execute(TENANT_A, PROFILE, _source())
        ready = env.documents.items[result.document_id]
        env.documents.items[ready.id] = replace(ready, status=DocumentStatus.PENDING)

        assert (await env.retrieve.execute(TENANT_A, PROFILE, "alpha paragraph")).hits == ()

    async def test_overfetches_so_dropped_chunks_do_not_shrink_the_result(self, env: Env) -> None:
        # The two best chunks belong to a SUPERSEDED document; top_k=2 must still be filled.
        stale = await env.ingest.execute(TENANT_A, PROFILE, _source("a\n\nb", uri="file:///x"))
        env.documents.items[stale.document_id] = replace(
            env.documents.items[stale.document_id], status=DocumentStatus.SUPERSEDED
        )
        await env.ingest.execute(TENANT_A, PROFILE, _source("c\n\nd", uri="file:///y"))

        result = await env.retrieve.execute(TENANT_A, PROFILE, "a")

        assert len(result.hits) == PROFILE.top_k
        assert all(h.citation.source_uri == "file:///y" for h in result.hits)

    async def test_result_is_trimmed_to_top_k(self, env: Env) -> None:
        await env.ingest.execute(TENANT_A, PROFILE, _source("a\n\nb\n\nc\n\nd"))
        result = await env.retrieve.execute(TENANT_A, PROFILE, "a")
        assert len(result.hits) == PROFILE.top_k

    def test_overfetch_factor_must_be_at_least_one(self, env: Env) -> None:
        with pytest.raises(ValueError, match="overfetch_factor"):
            RetrieveForTopic(env.documents, env.chunks, env.embeddings, overfetch_factor=0)


class TestCleanersInIngest:
    @staticmethod
    def _ingest(cleaner: FakeCleaner) -> tuple[IngestDocument, InMemoryChunkRepository]:
        chunks = InMemoryChunkRepository()
        ingest = IngestDocument(
            InMemoryDocumentRepository(),
            chunks,
            {"fake": FakeParser()},
            {"fake": FakeChunker()},
            DeterministicFakeEmbedding(size=16),
            FixedClock(),
            InMemoryIngestLock(),
            cleaners={"drop": cleaner},
        )
        return ingest, chunks

    async def test_cleaners_run_between_parsing_and_chunking_with_their_params(self) -> None:
        cleaner = FakeCleaner()
        ingest, chunks = self._ingest(cleaner)
        profile = replace(PROFILE, cleaners=(CleanerSpec("drop", {"drop": "beta"}),))

        result = await ingest.execute(TENANT_A, profile, _source())

        assert result.chunk_count == 2
        assert [c.text for c, _ in chunks.items] == ["alpha paragraph", "gamma paragraph"]
        assert [dict(p) for p in cleaner.calls] == [{"drop": "beta"}]

    async def test_unregistered_cleaner_fails_fast(self) -> None:
        ingest, chunks = self._ingest(FakeCleaner())
        profile = replace(PROFILE, cleaners=(CleanerSpec("nope"),))
        with pytest.raises(UnknownStrategyError, match=r"cleaner.*nope"):
            await ingest.execute(TENANT_A, profile, _source())
        assert chunks.items == []

    async def test_cleaners_that_remove_everything_raise_empty_document(self) -> None:
        ingest, chunks = self._ingest(FakeCleaner())
        profile = replace(PROFILE, cleaners=(CleanerSpec("drop", {"drop": "paragraph"}),))
        with pytest.raises(EmptyDocumentError):
            await ingest.execute(TENANT_A, profile, _source())
        assert chunks.items == []

    async def test_changing_cleaner_params_reindexes(self) -> None:
        ingest, _ = self._ingest(FakeCleaner())
        first = replace(PROFILE, cleaners=(CleanerSpec("drop", {"drop": "beta"}),))
        second = replace(PROFILE, cleaners=(CleanerSpec("drop", {"drop": "gamma"}),))
        assert (await ingest.execute(TENANT_A, first, _source())).status == "indexed"
        assert (await ingest.execute(TENANT_A, first, _source())).status == "skipped"
        assert (await ingest.execute(TENANT_A, second, _source())).status == "replaced"
