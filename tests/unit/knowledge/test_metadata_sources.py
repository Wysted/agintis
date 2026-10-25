"""Metadata sources: group / document / parsed / const / extractor, transforms and ingest."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from agintis.modules.knowledge.api import (
    ChunkDraft,
    Chunker,
    ExtractionContext,
    Extractor,
    IngestDocument,
    InvalidProfileError,
    MetadataResolver,
    ParsedDocument,
    ParsedSection,
    SourceFile,
    UnknownStrategyError,
    ValueTransform,
)
from agintis.modules.knowledge.application import RawValue
from agintis.modules.knowledge.domain import (
    ExtractorSpec,
    InvalidMetadataError,
    MetadataField,
    ParamValue,
    RagProfile,
    TransformSpec,
)
from agintis.modules.knowledge.infrastructure.registry import build_extractors, build_transforms
from agintis.shared_kernel.ids import TenantId
from tests.fakes.clock import FixedClock
from tests.fakes.knowledge import (
    FakeParser,
    InMemoryChunkRepository,
    InMemoryDocumentRepository,
    InMemoryIngestLock,
)

TODAY = datetime(2026, 10, 15, tzinfo=UTC)


def _profile(*fields: MetadataField, **kwargs: Any) -> RagProfile:
    return RagProfile("p", 3, 100, "fake", "fake", metadata_fields=fields, **kwargs)


def _resolver() -> MetadataResolver:
    return MetadataResolver(build_extractors(), build_transforms(), FixedClock(TODAY))


def _parsed(**kwargs: Any) -> ParsedDocument:
    return ParsedDocument("Title", (ParsedSection("body text"),), **kwargs)


def _draft(**groups: tuple[str, ...]) -> ChunkDraft:
    return ChunkDraft("body text", metadata={"page": 1}, groups=groups)


class TestSources:
    def test_group_scalar_takes_the_first_value_and_list_all_of_them(self) -> None:
        profile = _profile(
            MetadataField("numero", source="group:numero"),
            MetadataField("numeros", "list[str]", source="group:numero"),
        )
        (out,) = _resolver().resolve(profile, _parsed(), [_draft(numero=("1", "2"))])
        assert out.metadata["numero"] == "1"
        assert out.metadata["numeros"] == ("1", "2")
        assert out.metadata["page"] == 1  # chunker metadata is kept

    def test_a_missing_group_leaves_the_field_out(self) -> None:
        profile = _profile(MetadataField("letra", source="group:letra"))
        (out,) = _resolver().resolve(profile, _parsed(), [_draft(numero=("1",))])
        assert "letra" not in out.metadata

    def test_document_fields_come_from_the_callers_metadata(self) -> None:
        profile = _profile(
            MetadataField("number", "int", source="document:number"),
            MetadataField("sections", "list[str]", source="document:section"),
            MetadataField("gone", source="document:absent"),
        )
        parsed = _parsed(document_metadata={"number": "2425", "section": ("a", "b"), "other": 1})
        (out,) = _resolver().resolve(profile, parsed, [_draft()])
        assert out.metadata["number"] == "2425"  # typed later, by conform_metadata
        assert out.metadata["sections"] == ("a", "b")
        assert "gone" not in out.metadata

    def test_parsed_values_come_from_the_cleaners(self) -> None:
        profile = _profile(
            MetadataField("keywords", "list[str]", source="parsed:keywords"),
            MetadataField("mode", source="parsed:mode"),
        )
        parsed = _parsed(collected={"keywords": ("a", "b"), "mode": ("index_card",)})
        (out,) = _resolver().resolve(profile, parsed, [_draft()])
        assert out.metadata["keywords"] == ("a", "b")
        assert out.metadata["mode"] == "index_card"  # a scalar field takes the first value

    def test_const_values(self) -> None:
        profile = _profile(MetadataField("family", source="const:circular"))
        (out,) = _resolver().resolve(profile, _parsed(), [_draft()])
        assert out.metadata["family"] == "circular"

    def test_empty_values_are_dropped(self) -> None:
        profile = _profile(
            MetadataField("a", source="document:a"),
            MetadataField("b", "list[str]", source="document:b"),
            MetadataField("c", source="document:c"),
        )
        parsed = _parsed(document_metadata={"a": "", "b": (), "c": None})
        (out,) = _resolver().resolve(profile, parsed, [_draft()])
        assert dict(out.metadata) == {"page": 1}

    def test_fields_without_a_source_are_left_to_the_chunker(self) -> None:
        profile = _profile(MetadataField("page_note", "str"))
        (out,) = _resolver().resolve(profile, _parsed(), [_draft()])
        assert dict(out.metadata) == {"page": 1}

    def test_no_declared_sources_returns_the_drafts_untouched(self) -> None:
        drafts = [_draft()]
        assert _resolver().resolve(_profile(), _parsed(), drafts) == drafts


class TestTransforms:
    def test_applied_in_order_after_shaping(self) -> None:
        profile = _profile(
            MetadataField(
                "system",
                source="document:area",
                transforms=(TransformSpec("value_map", {"map": {"billing": "BL"}}),),
            ),
            MetadataField(
                "date",
                "date",
                source="document:date",
                transforms=(TransformSpec("date_floor"), TransformSpec("reject_after")),
            ),
            MetadataField(
                "audiences",
                "list[str]",
                source="document:section",
                transforms=(TransformSpec("path_segment"),),
            ),
        )
        parsed = _parsed(
            document_metadata={"area": "billing", "date": "2021", "section": ("A > B", "C > B")}
        )
        (out,) = _resolver().resolve(profile, parsed, [_draft()])
        assert out.metadata["system"] == "BL"
        assert out.metadata["date"] == "2021-01-01"
        assert out.metadata["audiences"] == ("B",)

    def test_a_transform_can_drop_the_field(self) -> None:
        profile = _profile(
            MetadataField(
                "date", "date", source="document:date", transforms=(TransformSpec("reject_after"),)
            )
        )
        (out,) = _resolver().resolve(
            profile, _parsed(document_metadata={"date": "2030-01-01"}), [_draft()]
        )
        assert "date" not in out.metadata

    def test_later_transforms_are_skipped_once_the_value_is_gone(self) -> None:
        calls: list[RawValue] = []

        class Spy(ValueTransform):
            key = "spy"

            def apply(
                self, value: RawValue, params: Mapping[str, ParamValue], today: Any
            ) -> RawValue:
                calls.append(value)
                return value

        resolver = MetadataResolver({}, {"drop": _Drop(), "spy": Spy()}, FixedClock(TODAY))
        profile = _profile(
            MetadataField(
                "a", source="const:x", transforms=(TransformSpec("drop"), TransformSpec("spy"))
            )
        )
        resolver.resolve(profile, _parsed(), [_draft()])
        assert calls == []

    def test_unknown_transform_key_fails_loudly(self) -> None:
        profile = _profile(
            MetadataField("a", source="const:x", transforms=(TransformSpec("nope"),))
        )
        with pytest.raises(UnknownStrategyError, match="nope"):
            _resolver().resolve(profile, _parsed(), [_draft()])


class _Drop(ValueTransform):
    key = "drop"

    def apply(self, value: RawValue, params: Mapping[str, ParamValue], today: Any) -> RawValue:
        return None


class _Counting(Extractor):
    key = "counting"
    version = 3
    seen: ClassVar[list[ExtractionContext]] = []

    def extract(self, context: ExtractionContext, params: Mapping[str, ParamValue]) -> RawValue:
        self.seen.append(context)
        return f"{params.get('prefix', '')}{context.selector or 'none'}"


class TestExtractors:
    def test_registry_extractors_receive_the_field_params(self) -> None:
        profile = _profile(
            MetadataField(
                "refs", "list[str]", source="extractor:regex_all", params={"pattern": r"\w+ \w+"}
            )
        )
        (out,) = _resolver().resolve(profile, _parsed(), [_draft()])
        assert out.metadata["refs"] == ("body text",)

    def test_named_extractors_are_shared_by_several_fields(self) -> None:
        resolver = MetadataResolver({"counting": _Counting()}, {}, FixedClock(TODAY))
        profile = _profile(
            MetadataField("a", source="extractor:shared.x"),
            MetadataField("b", source="extractor:shared.y"),
            extractors={"shared": ExtractorSpec("counting", {"prefix": ">"})},
        )
        (out,) = resolver.resolve(profile, _parsed(), [_draft()])
        assert (out.metadata["a"], out.metadata["b"]) == (">x", ">y")

    def test_an_extractor_sees_the_fields_declared_before_it(self) -> None:
        _Counting.seen.clear()
        resolver = MetadataResolver({"counting": _Counting()}, {}, FixedClock(TODAY))
        profile = _profile(
            MetadataField("first", source="const:one"),
            MetadataField("second", source="extractor:counting"),
        )
        resolver.resolve(profile, _parsed(), [_draft()])
        assert dict(_Counting.seen[0].resolved) == {"first": "one"}
        assert _Counting.seen[0].today == TODAY.date()

    def test_the_memo_is_shared_by_the_chunks_of_one_document_only(self) -> None:
        _Counting.seen.clear()
        resolver = MetadataResolver({"counting": _Counting()}, {}, FixedClock(TODAY))
        profile = _profile(MetadataField("a", source="extractor:counting"))
        resolver.resolve(profile, _parsed(), [_draft(), _draft()])
        resolver.resolve(profile, _parsed(), [_draft()])
        first, second, third = _Counting.seen
        assert first.memo is second.memo
        assert first.memo is not third.memo

    def test_unknown_extractor_fails_loudly(self) -> None:
        profile = _profile(MetadataField("a", source="extractor:nope"))
        with pytest.raises(UnknownStrategyError, match="nope"):
            _resolver().resolve(profile, _parsed(), [_draft()])

    def test_versions_cover_extractors_and_transforms_in_use(self) -> None:
        resolver = MetadataResolver({"counting": _Counting()}, {"drop": _Drop()}, FixedClock(TODAY))
        profile = _profile(
            MetadataField("a", source="extractor:alias.x", transforms=(TransformSpec("drop"),)),
            MetadataField("b", source="const:x"),
            extractors={"alias": ExtractorSpec("counting")},
        )
        assert resolver.versions(profile) == {"extractor:counting": 3, "transform:drop": 1}


class _GroupChunker(Chunker):
    """One chunk per paragraph; the paragraph's first word is its `kind` group."""

    async def chunk(self, parsed: ParsedDocument, profile: RagProfile) -> list[ChunkDraft]:
        return [
            ChunkDraft(s.text, groups={"kind": (s.text.split()[0],), "n": ("12",)})
            for s in parsed.sections
        ]


def _ingest(
    resolver: MetadataResolver | None,
) -> tuple[IngestDocument, InMemoryChunkRepository, InMemoryDocumentRepository]:
    documents = InMemoryDocumentRepository()
    chunks = InMemoryChunkRepository(documents)
    ingest = IngestDocument(
        documents,
        chunks,
        {"fake": FakeParser()},
        {"fake": _GroupChunker()},
        DeterministicFakeEmbedding(size=4),
        FixedClock(TODAY),
        InMemoryIngestLock(),
        metadata=resolver,
    )
    return ingest, chunks, documents


FIELDS = (
    MetadataField("kind", source="group:kind", filterable=True),
    MetadataField("n", "int", source="group:n"),
    MetadataField("number", "int", source="document:number", filterable=True),
    MetadataField("laws", "list[str]", source="document:laws"),
)
SOURCE_BODY = b"alpha one\n\nbeta two"


class TestIngestIntegration:
    async def test_sources_are_resolved_and_typed_before_storage(self) -> None:
        ingest, chunks, _ = _ingest(_resolver())
        source = SourceFile(
            "file:///a.txt",
            "text/plain",
            SOURCE_BODY,
            document_metadata={"number": "2425", "laws": ("DL 3500",)},
        )
        await ingest.execute(TenantId("t"), _profile(*FIELDS), source)
        stored = [dict(c.metadata) for c, _ in chunks.items]
        assert stored == [
            {"kind": "alpha", "n": 12, "number": 2425, "laws": ("DL 3500",)},
            {"kind": "beta", "n": 12, "number": 2425, "laws": ("DL 3500",)},
        ]

    async def test_changed_document_metadata_reprocesses_the_same_content(self) -> None:
        ingest, chunks, _ = _ingest(_resolver())
        profile = _profile(*FIELDS)
        one = SourceFile("u", "text/plain", SOURCE_BODY, document_metadata={"number": 1})
        two = SourceFile("u", "text/plain", SOURCE_BODY, document_metadata={"number": 2})
        assert (await ingest.execute(TenantId("t"), profile, one)).status == "indexed"
        assert (await ingest.execute(TenantId("t"), profile, one)).status == "skipped"
        assert (await ingest.execute(TenantId("t"), profile, two)).status == "replaced"
        assert (await ingest.execute(TenantId("t"), profile, two)).status == "skipped"
        assert {c.metadata["number"] for c, _ in chunks.items} == {2}

    async def test_metadata_key_order_does_not_matter(self) -> None:
        ingest, _, _ = _ingest(_resolver())
        profile = _profile(*FIELDS)
        one = SourceFile("u", "text/plain", SOURCE_BODY, document_metadata={"a": 1, "b": (1, 2)})
        two = SourceFile("u", "text/plain", SOURCE_BODY, document_metadata={"b": [1, 2], "a": 1})  # type: ignore[dict-item]
        assert (await ingest.execute(TenantId("t"), profile, one)).status == "indexed"
        assert (await ingest.execute(TenantId("t"), profile, two)).status == "skipped"

    async def test_a_document_stored_without_a_metadata_hash_is_reprocessed_once(self) -> None:
        ingest, _, documents = _ingest(_resolver())
        profile = _profile(*FIELDS)
        source = SourceFile("u", "text/plain", SOURCE_BODY, document_metadata={"number": 1})
        assert (await ingest.execute(TenantId("t"), profile, source)).status == "indexed"
        (stored,) = documents.items.values()
        documents.items[stored.id] = replace(stored, metadata_hash=None)  # a pre-0002 row
        assert (await ingest.execute(TenantId("t"), profile, source)).status == "replaced"
        assert (await ingest.execute(TenantId("t"), profile, source)).status == "skipped"

    async def test_a_value_that_does_not_fit_its_type_fails_before_storage(self) -> None:
        ingest, chunks, documents = _ingest(_resolver())
        source = SourceFile("u", "text/plain", SOURCE_BODY, document_metadata={"number": "twenty"})
        with pytest.raises(InvalidMetadataError, match="number"):
            await ingest.execute(TenantId("t"), _profile(*FIELDS), source)
        assert chunks.items == []
        assert documents.items == {}

    async def test_declared_sources_without_a_resolver_fail_fast(self) -> None:
        ingest, chunks, documents = _ingest(None)
        with pytest.raises(InvalidProfileError, match="no MetadataResolver"):
            await ingest.execute(
                TenantId("t"), _profile(*FIELDS), SourceFile("u", "text/plain", SOURCE_BODY)
            )
        assert chunks.items == []
        assert documents.items == {}

    async def test_a_profile_without_sources_needs_no_resolver(self) -> None:
        ingest, chunks, _ = _ingest(None)
        await ingest.execute(
            TenantId("t"), _profile(MetadataField("f", "str")), SourceFile("u", "text/plain", b"x")
        )
        assert len(chunks.items) == 1

    async def test_an_extractor_version_bump_reindexes(self) -> None:
        profile = _profile(MetadataField("a", source="extractor:counting"))
        source = SourceFile("u", "text/plain", SOURCE_BODY)
        documents = InMemoryDocumentRepository()
        chunks = InMemoryChunkRepository(documents)

        def build(version: int) -> IngestDocument:
            class Versioned(_Counting):
                pass

            Versioned.version = version
            return IngestDocument(
                documents,
                chunks,
                {"fake": FakeParser()},
                {"fake": _GroupChunker()},
                DeterministicFakeEmbedding(size=4),
                FixedClock(TODAY),
                InMemoryIngestLock(),
                metadata=MetadataResolver({"counting": Versioned()}, {}, FixedClock(TODAY)),
            )

        assert (await build(1).execute(TenantId("t"), profile, source)).status == "indexed"
        assert (await build(1).execute(TenantId("t"), profile, source)).status == "skipped"
        assert (await build(2).execute(TenantId("t"), profile, source)).status == "replaced"
