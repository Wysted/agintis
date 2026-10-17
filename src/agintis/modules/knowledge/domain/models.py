"""Knowledge entities and value objects."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, cast
from uuid import UUID, uuid5

from agintis.modules.knowledge.domain.errors import InvalidMetadataError
from agintis.shared_kernel.criteria import FilterExpr
from agintis.shared_kernel.criteria.filters import is_valid_field_name
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId

# A metadata value is a scalar or a homogeneous list (tuple once frozen) of strings or integers.
type MetadataScalar = str | int | float
type MetadataValue = MetadataScalar | tuple[str, ...] | tuple[int, ...]
# What callers may hand in (JSON decoded from storage yields lists); frozen into `MetadataValue`.
type MetadataInput = MetadataScalar | list[str] | list[int] | tuple[str, ...] | tuple[int, ...]
# Nested, immutable strategy parameters (chunker params, cleaner params): scalars, tuples and
# read-only mappings. `freeze_params` builds them from plain JSON-like data.
type ParamValue = (
    str | int | float | bool | tuple[ParamValue, ...] | Mapping[str, ParamValue] | None
)

SNIPPET_MAX_CHARS = 300

# Core chunk fields. Metadata must never use these names: they would shadow identity or
# tenant columns in filters and in backends that flatten metadata next to core fields.
RESERVED_CHUNK_FIELDS = frozenset({"id", "document_id", "tenant_id", "profile", "ordinal", "text"})

# No `bool`: strict-typed filters keep bool distinct from int, and no extractor needs it.
# `list[...]` fields are stored as JSONB arrays; filters treat them as sets of their elements.
type MetadataFieldType = Literal["str", "int", "float", "date", "list[str]", "list[int]"]
METADATA_FIELD_TYPES: frozenset[str] = frozenset(
    {"str", "int", "float", "date", "list[str]", "list[int]"}
)
# Element type of each list field type.
LIST_ELEMENT_TYPES: Mapping[str, Literal["str", "int"]] = MappingProxyType(
    {"list[str]": "str", "list[int]": "int"}
)
MAX_METADATA_LIST_ITEMS = 256

# Accepted prefixes of `MetadataField.source` (resolved when chunks are built, not by storage):
# a regex named group of the matched level, a document-level value supplied at ingest, a value
# collected by a cleaner, a constant, or a registered extractor.
METADATA_SOURCE_PREFIXES = ("group:", "document:", "parsed:", "const:", "extractor:")

# Implementation version every strategy starts at; only higher versions enter the fingerprint.
BASELINE_STRATEGY_VERSION = 1

# Retrieval strategy keys of a RagProfile (`retrieval_mode`); more may be registered.
DENSE_RETRIEVAL_MODE = "dense"
HYBRID_RETRIEVAL_MODE = "hybrid"

_CHUNK_ID_NAMESPACE = UUID("6f1c1c1e-5a0b-4d0e-9a53-2f7a2d9b7c11")


_UNSTORABLE = re.compile("[\x00\ud800-\udfff]")
_TITLE_DROPPED = frozenset({"Cc", "Cf"})


def check_storable_text(label: str, value: str) -> None:
    """Postgres text/JSONB cannot hold NUL, and lone surrogates cannot be encoded as UTF-8."""
    if "\x00" in value:
        raise InvalidMetadataError(f"{label} must not contain NUL characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise InvalidMetadataError(f"{label} is not valid UTF-8 text (lone surrogate)") from None


def strip_unstorable(text: str) -> str:
    """`text` without NUL and lone surrogates (what `check_storable_text` rejects)."""
    return _UNSTORABLE.sub("", text)


MAX_TITLE_CHARS = 300


def display_title(text: str) -> str:
    """`text` as a one-line display title: unstorable characters and Unicode control/format
    characters (Cc/Cf: bidi overrides, zero-width marks) removed, whitespace collapsed and
    the result cut to `MAX_TITLE_CHARS`. Control whitespace (tab, newline) separates words
    before the removal, so it is never glued together."""
    words = strip_unstorable(text).split()
    cleaned = " ".join(
        "".join(ch for ch in word if unicodedata.category(ch) not in _TITLE_DROPPED)
        for word in words
    )
    return " ".join(cleaned.split())[:MAX_TITLE_CHARS].rstrip()


def _freeze_list(key: str, items: Sequence[object]) -> tuple[str, ...] | tuple[int, ...]:
    """Validate a list value and return it as a tuple: homogeneous str or int, bounded."""
    if len(items) > MAX_METADATA_LIST_ITEMS:
        raise InvalidMetadataError(
            f"Metadata value for {key!r} has {len(items)} items (limit {MAX_METADATA_LIST_ITEMS})"
        )
    strings = [item for item in items if isinstance(item, str)]
    if len(strings) == len(items):
        for item in strings:
            check_storable_text(f"Metadata value for {key!r}", item)
        return tuple(strings)
    integers = [item for item in items if isinstance(item, int) and not isinstance(item, bool)]
    if len(integers) == len(items):
        return tuple(integers)
    raise InvalidMetadataError(
        f"Metadata value for {key!r} must be a list of only strings or only integers "
        "(no bool, no float, no mixed kinds)"
    )


def freeze_metadata(metadata: Mapping[str, MetadataInput]) -> Mapping[str, MetadataValue]:
    """Copy `metadata`, validate the COPY and return it read-only.

    Validating the copy (not the caller's mapping) closes the window where the input is
    mutated between validation and storage. Rejected: booleans (they are `int` in Python but
    strict-typed filters treat them as another kind, so they would never match), non-finite
    floats (not representable in JSON/JSONB), strings with NUL or lone surrogates (Postgres
    cannot store them), reserved or malformed keys. Lists (and tuples) become tuples; they must
    hold only strings or only integers and at most `MAX_METADATA_LIST_ITEMS` items.
    """
    copied: dict[str, MetadataInput] = dict(metadata)
    for key, value in copied.items():
        if isinstance(value, bool):
            raise InvalidMetadataError(f"Metadata value for {key!r} must not be a bool")
        if key in RESERVED_CHUNK_FIELDS:
            raise InvalidMetadataError(f"Metadata key {key!r} is reserved")
        if not is_valid_field_name(key):
            raise InvalidMetadataError(f"Metadata key {key!r} is not a valid field name")
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidMetadataError(f"Metadata value for {key!r} must be finite, got {value!r}")
        if isinstance(value, str):
            check_storable_text(f"Metadata value for {key!r}", value)
        elif isinstance(value, list | tuple):
            copied[key] = _freeze_list(key, value)
    return MappingProxyType(cast(dict[str, MetadataValue], copied))


def chunk_id_for(document_id: DocumentId, ordinal: int) -> ChunkId:
    """Deterministic chunk id: the same document and position always yield the same id."""
    return ChunkId(uuid5(_CHUNK_ID_NAMESPACE, f"{document_id}:{ordinal}").hex)


class DocumentStatus(StrEnum):
    PENDING = "pending"  # saved, chunks may be missing or partial
    READY = "ready"  # all chunks indexed; the only status retrieval serves
    SUPERSEDED = "superseded"  # replaced by a newer READY version of the same source; garbage


@dataclass(frozen=True)
class Document:
    id: DocumentId
    tenant_id: TenantId
    profile: str
    source_uri: str
    title: str
    mime_type: str
    content_hash: str
    profile_fingerprint: str
    created_at: datetime
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    # sha256 of the caller's document metadata at ingest (None = ingested before it was tracked,
    # which makes the next ingest reprocess the document once).
    metadata_hash: str | None = None

    def mark_ready(self, chunk_count: int) -> Document:
        return replace(self, status=DocumentStatus.READY, chunk_count=chunk_count)


@dataclass(frozen=True)
class Chunk:
    id: ChunkId
    document_id: DocumentId
    tenant_id: TenantId
    profile: str
    ordinal: int
    text: str
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    document_id: DocumentId
    chunk_id: ChunkId
    source_uri: str
    title: str
    page: int | None
    snippet: str
    heading: str | None = None

    @classmethod
    def from_scored(cls, scored: ScoredChunk, document: Document) -> Citation:
        chunk = scored.chunk
        page = chunk.metadata.get("page")
        heading = chunk.metadata.get("heading")
        return cls(
            document_id=document.id,
            chunk_id=chunk.id,
            source_uri=document.source_uri,
            title=document.title,
            page=page if isinstance(page, int) else None,
            snippet=chunk.text[:SNIPPET_MAX_CHARS],
            heading=heading if isinstance(heading, str) and heading else None,
        )


def freeze_params(value: object, *, where: str = "params") -> ParamValue:
    """Deep-copy JSON-like data into an immutable `ParamValue`.

    Mappings become read-only, lists become tuples.

    Accepts str, int, bool, finite float, None, lists/tuples and mappings with non-empty string
    keys; anything else raises `ValueError`. The result is hashable-by-content through
    `canonical_params`, which is what keeps profile fingerprints stable.
    """
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} must not contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, ParamValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{where} keys must be non-empty strings, got {key!r}")
            frozen[key] = freeze_params(item, where=f"{where}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(freeze_params(item, where=where) for item in value)
    raise ValueError(f"{where} holds an unsupported value of type {type(value).__name__}")


def thaw_params(value: ParamValue) -> Any:
    """Plain mutable JSON-like copy of a `ParamValue` (mappings -> dict, tuples -> list)."""
    if isinstance(value, Mapping):
        return {key: thaw_params(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_params(item) for item in value]
    return value


def canonical_params(value: ParamValue) -> Any:
    """Like `thaw_params` but with mapping keys sorted: the JSON of it is order-independent."""
    if isinstance(value, Mapping):
        return {key: canonical_params(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [canonical_params(item) for item in value]
    return value


@dataclass(frozen=True)
class TransformSpec:
    """One value transform of a metadata field: a registry key plus its (validated) params.

    Transforms run in declaration order on the raw value a field's `source` produced, before
    the value is checked against the field type. They are part of the profile fingerprint.
    """

    key: str
    params: Mapping[str, ParamValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("TransformSpec key must not be empty")
        frozen = freeze_params(self.params, where=f"transform {self.key!r} params")
        if not isinstance(frozen, Mapping):
            raise ValueError(f"transform {self.key!r} params must be a mapping")
        object.__setattr__(self, "params", frozen)


@dataclass(frozen=True)
class ExtractorSpec:
    """A named, configured extractor of a profile: registry `key` plus its (validated) params.

    Declared once under the profile's `extractors` and referenced by several metadata fields
    as `extractor:<name>[.<selector>]` (one lookup, many outputs), instead of repeating the
    params in every field.
    """

    key: str
    params: Mapping[str, ParamValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("ExtractorSpec key must not be empty")
        frozen = freeze_params(self.params, where=f"extractor {self.key!r} params")
        if not isinstance(frozen, Mapping):
            raise ValueError(f"extractor {self.key!r} params must be a mapping")
        object.__setattr__(self, "params", frozen)


@dataclass(frozen=True)
class MetadataField:
    """One declared chunk metadata field of a profile.

    `type` drives how a backend stores/compares the value: `date` values are ISO-8601 strings
    (`YYYY-MM-DD`, compared as text) so they stay immutable-indexable and need no casts;
    `list[str]` / `list[int]` are stored as JSONB arrays and filters treat them as sets (EQ =
    contains the value, IN = overlaps).
    `filterable` exposes the field to retrieval filters; `indexed` declares that an index on it
    is expected (only meaningful when filterable). Declaring it creates NOTHING: the index is
    schema, so it needs an Alembic migration (the Postgres adapter names it
    `ix_chunks_meta_<name>`), and bootstrap logs a warning for an indexed field without one.
    `source` tells the ingest where the value comes from, resolved when chunks are built (storage
    ignores it): `group:<name>` (named regex group of the matched level), `document:<field>`
    (document-level metadata the caller supplied at ingest), `parsed:<key>` (value a cleaner
    collected), `const:<value>` or `extractor:<key>[.<selector>]` (a registered extractor and
    its `params`, or a named instance of the profile's `extractors`). `transforms` post-process
    the raw value (registered keys).

    Values are checked at ingest against the declared `type` (see `conform_metadata`).
    """

    name: str
    type: MetadataFieldType = "str"
    filterable: bool = False
    indexed: bool = False
    source: str | None = None
    params: Mapping[str, ParamValue] = field(default_factory=dict)
    transforms: tuple[TransformSpec, ...] = ()

    def __post_init__(self) -> None:
        if not is_valid_field_name(self.name):
            raise ValueError(f"Metadata field name {self.name!r} is not a valid field name")
        if self.name in RESERVED_CHUNK_FIELDS:
            raise ValueError(f"Metadata field name {self.name!r} is reserved")
        if self.type not in METADATA_FIELD_TYPES:
            raise ValueError(f"Metadata field {self.name!r} has unsupported type {self.type!r}")
        if self.indexed and not self.filterable:
            raise ValueError(f"Metadata field {self.name!r} is indexed but not filterable")
        if self.source is not None and not any(
            self.source.startswith(prefix) and len(self.source) > len(prefix)
            for prefix in METADATA_SOURCE_PREFIXES
        ):
            raise ValueError(
                f"Metadata field {self.name!r} source must start with one of "
                f"{', '.join(METADATA_SOURCE_PREFIXES)} followed by a name or value"
            )
        frozen = freeze_params(self.params, where=f"metadata field {self.name!r} params")
        if not isinstance(frozen, Mapping):
            raise ValueError(f"Metadata field {self.name!r} params must be a mapping")
        if frozen and not (self.source or "").startswith("extractor:"):
            raise ValueError(f"Metadata field {self.name!r} has params but no extractor source")
        object.__setattr__(self, "params", frozen)
        object.__setattr__(self, "transforms", tuple(self.transforms))
        if self.transforms and self.source is None:
            raise ValueError(f"Metadata field {self.name!r} has transforms but no source")

    @property
    def is_list(self) -> bool:
        return self.type in LIST_ELEMENT_TYPES


# Fields every chunk may carry regardless of the profile: the parser provides them. They are
# filterable for every profile and a profile may not redeclare them.
# `page` = first page of the chunk; `page_end` = last page (equal to `page` for single-page chunks).
BUILTIN_METADATA_FIELDS: tuple[MetadataField, ...] = (
    MetadataField("page", "int", filterable=True),
    MetadataField("page_end", "int", filterable=True),
)
BUILTIN_METADATA_NAMES = frozenset(f.name for f in BUILTIN_METADATA_FIELDS)


def chunk_metadata_schema(declared: Iterable[MetadataField]) -> dict[str, MetadataField]:
    """Built-in metadata fields overlaid with the profile-declared ones, by name."""
    return {f.name: f for f in (*BUILTIN_METADATA_FIELDS, *declared)}


# ASCII only: `int("١٢")` and `int("1_000")` would also parse, which is NOT unambiguous.
_INT_PATTERN = re.compile(r"[+-]?[0-9]+")
_FLOAT_PATTERN = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")
_DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _fail(name: str, expected: str, value: object) -> InvalidMetadataError:
    return InvalidMetadataError(f"Metadata field {name!r} must be {expected}, got {value!r}")


def _conform_int(name: str, value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    if isinstance(value, str) and _INT_PATTERN.fullmatch(stripped := value.strip()):
        return int(stripped)
    raise _fail(name, "an integer (or a decimal integer string)", value)


def _conform_float(name: str, value: object) -> int | float:
    if isinstance(value, int | float):
        return value
    if isinstance(value, str) and _FLOAT_PATTERN.fullmatch(stripped := value.strip()):
        number = float(stripped)
        if math.isfinite(number):
            return number
    raise _fail(name, "a finite number (or a decimal string)", value)


def _conform_date(name: str, value: object) -> str:
    if isinstance(value, str) and _DATE_PATTERN.fullmatch(value):
        try:
            date.fromisoformat(value)
        except ValueError:
            pass
        else:
            return value
    raise _fail(name, "an ISO date YYYY-MM-DD", value)


def _conform_str(name: str, value: object) -> str:
    if isinstance(value, str):
        return value
    raise _fail(name, "a string", value)


_CONFORMERS: dict[str, Callable[[str, object], MetadataScalar]] = {
    "str": _conform_str,
    "int": _conform_int,
    "float": _conform_float,
    "date": _conform_date,
}


def _conform_list(name: str, value: object, element_type: str) -> list[MetadataScalar]:
    if not isinstance(value, list | tuple):
        raise _fail(name, f"a list of {element_type}", value)
    if len(value) > MAX_METADATA_LIST_ITEMS:
        raise _fail(name, f"a list of at most {MAX_METADATA_LIST_ITEMS} items", value)
    conformer = _CONFORMERS[element_type]
    return [conformer(f"{name}[{index}]", item) for index, item in enumerate(value)]


def _conform_value(field_: MetadataField, value: object) -> MetadataInput:
    element_type = LIST_ELEMENT_TYPES.get(field_.type)
    if element_type is None:
        return _CONFORMERS[field_.type](field_.name, value)
    items = _conform_list(field_.name, value, element_type)
    # Homogeneity is guaranteed by the element conformer; `freeze_metadata` re-checks the rest
    # (size, NUL, lone surrogates).
    return cast(list[str], items) if element_type == "str" else cast(list[int], items)


def conform_metadata(
    metadata: Mapping[str, MetadataInput], declared: Iterable[MetadataField]
) -> Mapping[str, MetadataValue]:
    """Check chunk metadata against the profile schema and return the canonical copy.

    Declared fields (and built-ins such as `page`) are coerced to their type so what is stored
    is exactly what typed filters compare against: `int`/`float` accept numbers and, for regex
    group values, unambiguous decimal strings ("2024" -> 2024; no thousands separators, no
    unicode digits); `date` must be an ISO `YYYY-MM-DD` string; `str` must already be a str.
    `list[str]` / `list[int]` need a list or tuple whose items are conformed one by one (so a
    list of group strings becomes a list of ints); the result is homogeneous, finite, free of
    NUL and bounded by `MAX_METADATA_LIST_ITEMS`. Anything else raises `InvalidMetadataError`
    (the whole ingest fails before storage).

    Undeclared keys are kept as-is (valid but NOT filterable, e.g. `heading` for citations):
    only `freeze_metadata`'s rules apply to them.
    """
    schema = chunk_metadata_schema(declared)
    conformed: dict[str, MetadataInput] = {}
    for key, value in metadata.items():
        field_ = schema.get(key)
        conformed[key] = value if field_ is None else _conform_value(field_, value)
    return freeze_metadata(conformed)


@dataclass(frozen=True)
class CleanerSpec:
    """One step of a profile's cleaner pipeline: a registry key plus its (validated) params.

    Cleaners run in declaration order after parsing and before chunking; they are part of the
    profile fingerprint (see `RagProfile.fingerprint`).
    """

    key: str
    params: Mapping[str, ParamValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("CleanerSpec key must not be empty")
        frozen = freeze_params(self.params, where=f"cleaner {self.key!r} params")
        if not isinstance(frozen, Mapping):
            raise ValueError(f"cleaner {self.key!r} params must be a mapping")
        object.__setattr__(self, "params", frozen)


@dataclass(frozen=True)
class RoutingCondition:
    """A test on one document-metadata field: exact membership OR a regex search.

    `any_of`: the field (or, for a list-valued field, one of its items) equals one of the
    values, compared as text. `pattern`: the regex is found in the field's text (a list is
    tested item by item). A document lacking the field never satisfies the condition. The
    pattern is only stored here; the profile loader checks it and the router compiles it.
    """

    field: str
    any_of: tuple[str, ...] = ()
    pattern: str | None = None

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("RoutingCondition field must not be empty")
        object.__setattr__(self, "any_of", tuple(self.any_of))
        if bool(self.any_of) == (self.pattern is not None):
            raise ValueError(
                f"RoutingCondition on {self.field!r} needs exactly one of values or a pattern"
            )


@dataclass(frozen=True)
class RoutingRule:
    """Selects documents for a profile: ALL `match` conditions hold and NO `exclude` one does.

    `priority` ranks the rules that match the same document: the highest wins (a narrow rule
    gets a higher number than the broad one it overlaps). Two profiles matching at the same top
    priority are ambiguous and the document is left unrouted; the default 0 therefore keeps
    non-overlapping rules working unchanged and turns an accidental overlap into a visible error
    instead of a silent file-name-order decision.
    """

    match: tuple[RoutingCondition, ...]
    exclude: tuple[RoutingCondition, ...] = ()
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "match", tuple(self.match))
        object.__setattr__(self, "exclude", tuple(self.exclude))
        if not self.match:
            raise ValueError("RoutingRule needs at least one match condition")


type ProfileStatus = Literal["active", "draft"]


@dataclass(frozen=True)
class RagProfile:
    """Retrieval/ingestion policy. Parser and chunker are strategy KEYS, not objects.

    `chunker_params` carries strategy-specific knobs (nested mappings/tuples allowed, stored
    immutable); `cleaners` is the ordered cleaner pipeline applied between parsing and chunking.
    Both are part of the profile fingerprint. Their per-key schemas are validated outside the
    domain (`validate_profiles`). `reranker_key` is a retrieval-time
    strategy ("none" = keep the vector order); it does not change what is indexed, so it is
    NOT part of the fingerprint. Neither are `routing` (which documents a profile is meant for,
    used by the profile router) and `status` (`draft` profiles load and validate but are never
    selected by the router).
    """

    name: str
    top_k: int
    chunk_max_tokens: int
    parser_key: str
    chunker_key: str
    chunker_params: Mapping[str, ParamValue] = field(default_factory=dict)
    default_filter: FilterExpr | None = None
    reranker_key: str = "none"
    retrieval_mode: str = DENSE_RETRIEVAL_MODE
    metadata_fields: tuple[MetadataField, ...] = ()
    cleaners: tuple[CleanerSpec, ...] = ()
    routing: tuple[RoutingRule, ...] = ()
    status: ProfileStatus = "active"
    extractors: Mapping[str, ExtractorSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RagProfile name must not be empty")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.chunk_max_tokens <= 0:
            raise ValueError("chunk_max_tokens must be greater than 0")
        if not self.parser_key or not self.chunker_key or not self.reranker_key:
            raise ValueError("parser_key, chunker_key and reranker_key must not be empty")
        if not self.retrieval_mode:
            raise ValueError("retrieval_mode must not be empty")
        frozen_params = freeze_params(self.chunker_params, where="chunker_params")
        if not isinstance(frozen_params, Mapping):
            raise ValueError("chunker_params must be a mapping")
        object.__setattr__(self, "chunker_params", frozen_params)
        object.__setattr__(self, "cleaners", tuple(self.cleaners))
        object.__setattr__(self, "metadata_fields", tuple(self.metadata_fields))
        object.__setattr__(self, "routing", tuple(self.routing))
        for name in self.extractors:
            if not is_valid_field_name(name):
                raise ValueError(f"Extractor name {name!r} is not a valid name")
        object.__setattr__(self, "extractors", MappingProxyType(dict(self.extractors)))
        names = [f.name for f in self.metadata_fields]
        if len(set(names)) != len(names):
            raise ValueError("metadata_fields names must be unique")
        if clash := BUILTIN_METADATA_NAMES.intersection(names):
            raise ValueError(f"metadata_fields redeclare built-in fields: {sorted(clash)}")

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the profile alone (every strategy at its baseline version)."""
        return self.fingerprint_for()

    def fingerprint_for(self, strategy_versions: Mapping[str, int | str] | None = None) -> str:
        """Stable hash of everything that changes how a document is parsed and chunked.

        Declared metadata fields count by (name, type, source): renaming, retyping or changing
        the extractor changes what is stored, so documents must be re-ingested. Their extractor
        `params`, `transforms` and the profile's named `extractors` count too (in a tagged part
        that exists only when one is used, so older profiles keep their hash). `filterable`
        and `indexed` only affect querying/indexing, and `routing` / `status` only choose which
        documents a profile is meant for: none of them is part of it.

        `strategy_versions` maps `"<kind>:<key>"` (`parser:simple`, `chunker:structured`,
        `cleaner:strip_lines`, `extractor:regex_all`, `transform:value_map`) to the
        implementation version of the strategy that will run;
        the ingest use case reads them from the resolved strategies. Only versions above
        `BASELINE_STRATEGY_VERSION` are hashed, in a tagged `{"versions": ...}` part. Legacy
        compatibility: a profile whose strategies are all at the baseline (and profiles
        without declared fields or cleaners) hash exactly as they did before each of those
        parts existed, so nothing is re-ingested until a strategy bumps its version.
        """
        parts: list[object] = [
            self.parser_key,
            self.chunker_key,
            self.chunk_max_tokens,
            [
                [key, canonical_params(self.chunker_params[key])]
                for key in sorted(self.chunker_params)
            ],
        ]
        if self.metadata_fields:
            parts.append(sorted((f.name, f.type, f.source) for f in self.metadata_fields))
            if self.extractors:
                parts.append(
                    {
                        "extractors": [
                            [name, spec.key, canonical_params(spec.params)]
                            for name, spec in sorted(self.extractors.items())
                        ]
                    }
                )
            if options := [
                [
                    f.name,
                    canonical_params(f.params),
                    [[t.key, canonical_params(t.params)] for t in f.transforms],
                ]
                for f in sorted(self.metadata_fields, key=lambda f: f.name)
                if f.params or f.transforms
            ]:
                # Tagged (a dict) so it can never collide with the other parts.
                parts.append({"field_options": options})
        if self.cleaners:
            # Tagged (a dict, not a list) so it can never collide with the metadata part.
            parts.append({"cleaners": [[c.key, canonical_params(c.params)] for c in self.cleaners]})
        if strategy_versions:
            pinned = {
                name: version
                for name, version in sorted(strategy_versions.items())
                if version != BASELINE_STRATEGY_VERSION
            }
            if pinned:
                parts.append({"versions": pinned})
        payload = json.dumps(parts, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
