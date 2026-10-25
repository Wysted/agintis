from collections.abc import Iterator, Mapping

import pytest

from agintis.modules.knowledge.api import InvalidMetadataError
from agintis.modules.knowledge.application import ChunkDraft
from agintis.modules.knowledge.domain import (
    MAX_TITLE_CHARS,
    RESERVED_CHUNK_FIELDS,
    Chunk,
    ChunkRepository,
    MetadataValue,
    chunk_id_for,
    display_title,
    ensure_single_tenant,
    freeze_metadata,
)
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId


def _chunk(tenant: str = "a", metadata: Mapping[str, MetadataValue] | None = None) -> Chunk:
    return Chunk(
        ChunkId("c"), DocumentId("d"), TenantId(tenant), "p", 0, "text", metadata=metadata or {}
    )


@pytest.mark.parametrize("key", sorted(RESERVED_CHUNK_FIELDS))
def test_chunk_rejects_reserved_metadata_keys(key: str) -> None:
    with pytest.raises(InvalidMetadataError, match="reserved"):
        _chunk(metadata={key: "B"})


@pytest.mark.parametrize("key", sorted(RESERVED_CHUNK_FIELDS))
def test_chunk_draft_rejects_reserved_metadata_keys(key: str) -> None:
    with pytest.raises(InvalidMetadataError, match="reserved"):
        ChunkDraft(text="x", metadata={key: "B"})


def test_metadata_keys_must_be_valid_field_names() -> None:
    with pytest.raises(InvalidMetadataError, match="valid field name"):
        _chunk(metadata={"bad-key": 1})


def test_metadata_is_read_only_and_detached_from_the_input() -> None:
    source = {"page": 1}
    chunk = Chunk(ChunkId("c"), DocumentId("d"), TenantId("a"), "p", 0, "t", metadata=source)
    source["page"] = 99
    assert chunk.metadata["page"] == 1
    with pytest.raises(TypeError):
        chunk.metadata["page"] = 2  # type: ignore[index]


def test_chunk_ids_are_deterministic_per_document_and_ordinal() -> None:
    assert chunk_id_for(DocumentId("d"), 0) == chunk_id_for(DocumentId("d"), 0)
    assert chunk_id_for(DocumentId("d"), 0) != chunk_id_for(DocumentId("d"), 1)
    assert chunk_id_for(DocumentId("d"), 0) != chunk_id_for(DocumentId("e"), 0)


def test_ensure_single_tenant() -> None:
    assert ensure_single_tenant([]) is None
    assert ensure_single_tenant([_chunk("a"), _chunk("a")]) == "a"
    with pytest.raises(ValueError, match="same tenant"):
        ensure_single_tenant([_chunk("a"), _chunk("b")])


def test_chunk_filterable_fields_include_core_and_pages() -> None:
    assert RESERVED_CHUNK_FIELDS | {"page", "page_end"} == ChunkRepository.FILTERABLE_FIELDS


class _MutatingMapping(Mapping[str, MetadataValue]):
    """Yields clean keys on the first iteration and a reserved key afterwards."""

    def __init__(self) -> None:
        self.iterations = 0

    def __getitem__(self, key: str) -> MetadataValue:
        return 1

    def __iter__(self) -> Iterator[str]:
        self.iterations += 1
        return iter(["page"] if self.iterations == 1 else ["tenant_id"])

    def __len__(self) -> int:
        return 1


def test_freeze_metadata_validates_the_copy_not_the_input() -> None:
    frozen = freeze_metadata(_MutatingMapping())
    assert dict(frozen) == {"page": 1}


def test_metadata_rejects_bool_values() -> None:
    with pytest.raises(InvalidMetadataError, match="bool"):
        _chunk(metadata={"page": True})


class TestDisplayTitle:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  REF 372\t\n sobre   inversiones ", "REF 372 sobre inversiones"),
            ("evil" + "\u202e" + "title" + "\u200b" + "\x07", "eviltitle"),
            ("a\x00b\ud800c", "abc"),
            ("a" + "\u00a0" + "b", "a b"),
            ("\u202e \x07 ", ""),
            ("", ""),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert display_title(raw) == expected

    def test_is_bounded_without_a_trailing_space(self) -> None:
        raw = "x" * (MAX_TITLE_CHARS - 1) + " tail"
        assert display_title(raw) == "x" * (MAX_TITLE_CHARS - 1)
        assert len(display_title("y" * (MAX_TITLE_CHARS * 2))) == MAX_TITLE_CHARS
