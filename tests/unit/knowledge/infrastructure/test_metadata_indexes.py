"""Expected metadata indexes: derived from the converter, stable names, pure diff."""

import hashlib

import pytest
import sqlalchemy as sa
from pydantic import SecretStr
from sqlalchemy.dialects import postgresql

from agintis.modules.knowledge.domain import MetadataField
from agintis.modules.knowledge.infrastructure.postgres import (
    CatalogIndex,
    ConflictingIndexError,
    IndexStatus,
    MetadataIndexSpec,
    MetadataIndexSync,
    PgChunkRepository,
    diff_indexes,
    expected_metadata_indexes,
    metadata_index_name,
)
from agintis.modules.knowledge.infrastructure.postgres.chunks import MAX_EF_SEARCH
from agintis.modules.knowledge.infrastructure.postgres.fields import chunk_fields
from agintis.modules.knowledge.infrastructure.postgres.indexes import MAX_IDENTIFIER_LENGTH
from agintis.modules.knowledge.infrastructure.postgres.tables import bm25_index_name
from agintis.platform.db import compile_filter, create_engine, create_unpooled_engine
from agintis.shared_kernel.criteria import Filter


def _field(name: str, type_: str = "str", **kw: bool) -> MetadataField:
    return MetadataField(name, type_, filterable=True, indexed=True, **kw)  # type: ignore[arg-type]


def _sql(expr: object) -> str:
    return str(
        expr.compile(  # type: ignore[attr-defined]
            dialect=postgresql.dialect(),  # type: ignore[no-untyped-call]
            compile_kwargs={"literal_binds": True},
        )
    )


class TestExpectedIndexes:
    def test_only_indexed_scalar_fields_get_a_per_field_index(self) -> None:
        fields = [
            _field("article"),
            MetadataField("chapter", "str", filterable=True),  # not indexed
            _field("laws", "list[str]"),  # lists: served by the table-wide GIN
        ]
        assert [s.name for s in expected_metadata_indexes(fields)] == ["ix_chunks_meta_article"]

    def test_the_expression_is_the_one_range_predicates_compare(self) -> None:
        """The index only helps when it textually matches what the converter emits."""
        for field in (_field("article"), _field("weight", "int"), _field("day", "date")):
            (spec,) = expected_metadata_indexes([field])
            where = compile_filter(
                Filter.gte(field.name, "a" if field.type != "int" else 1), chunk_fields([field])
            )
            assert spec.expression in _sql(where)

    def test_text_uses_the_c_collation_and_numbers_the_numeric_cast(self) -> None:
        text_spec, number_spec = (
            expected_metadata_indexes([f])[0] for f in (_field("a"), _field("n", "float"))
        )
        assert 'COLLATE "C"' in text_spec.expression
        assert "AS NUMERIC" in number_spec.expression
        assert text_spec.create_sql.startswith(
            'CREATE INDEX CONCURRENTLY IF NOT EXISTS "ix_chunks_meta_a" ON chunks (('
        )
        assert " WHERE " in text_spec.create_sql

    def test_declaring_a_field_in_several_profiles_collapses_when_compatible(self) -> None:
        compatible = [_field("n", "int"), _field("n", "float"), _field("d", "date"), _field("d")]
        assert [s.name for s in expected_metadata_indexes(compatible)] == [
            "ix_chunks_meta_d",
            "ix_chunks_meta_n",
        ]

    def test_incompatible_types_under_one_name_are_a_conflict(self) -> None:
        with pytest.raises(ConflictingIndexError, match="incompatible"):
            expected_metadata_indexes([_field("n", "int"), _field("n", "str")])

    def test_output_is_sorted_by_name(self) -> None:
        names = [s.name for s in expected_metadata_indexes([_field("b"), _field("a")])]
        assert names == sorted(names)


class TestIndexNames:
    def test_short_names_are_the_conventional_prefix_plus_the_field(self) -> None:
        assert metadata_index_name(_field("article")) == "ix_chunks_meta_article"

    def test_long_names_are_truncated_stable_and_collision_free(self) -> None:
        a, b = _field("x" * 80), _field("x" * 79 + "y")
        name_a, name_b = metadata_index_name(a), metadata_index_name(b)
        assert len(name_a) == len(name_b) == MAX_IDENTIFIER_LENGTH
        assert name_a != name_b
        assert metadata_index_name(a) == name_a  # stable across calls
        assert name_a.startswith("ix_chunks_meta_xxx")

    def test_a_name_at_the_limit_is_kept_as_is(self) -> None:
        field = _field("y" * (MAX_IDENTIFIER_LENGTH - len("ix_chunks_meta_")))
        assert metadata_index_name(field) == f"ix_chunks_meta_{field.name}"


class TestBm25IndexName:
    def test_names_are_the_prefix_plus_a_short_sha256_of_the_tenant(self) -> None:
        tenant = "x'; DROP TABLE chunks; --" + "t" * 100
        digest = hashlib.sha256(tenant.encode()).hexdigest()[:16]
        assert bm25_index_name(tenant) == f"ix_chunks_bm25_{digest}"
        assert len(bm25_index_name(tenant)) <= MAX_IDENTIFIER_LENGTH
        assert bm25_index_name("a") != bm25_index_name("b")


def _row(spec: MetadataIndexSpec, **overrides: object) -> CatalogIndex:
    """The catalog row of an index this tool created for `spec`."""
    indexdef = f"CREATE INDEX {spec.name} ON public.chunks USING btree (...)"
    fields: dict[str, object] = {
        "name": spec.name,
        "valid": True,
        "comment": spec.comment_for(indexdef),
        "indexdef": indexdef,
    }
    return CatalogIndex(**{**fields, **overrides})  # type: ignore[arg-type]


class TestDiff:
    def _specs(self) -> list[MetadataIndexSpec]:
        return expected_metadata_indexes([_field("a"), _field("b"), _field("c"), _field("d")])

    def test_statuses(self) -> None:
        a, b, c, d = self._specs()
        existing = [
            _row(a),  # present
            _row(b, comment="agintis:expr:0000deadbeef:def:0000deadbeef"),  # another definition
            _row(c, valid=False),  # failed concurrent build
            CatalogIndex("ix_chunks_meta_gone", True, None),  # nobody declares it
        ]
        plan = diff_indexes([a, b, c, d], existing)
        assert {e.name: e.status for e in plan.entries} == {
            a.name: IndexStatus.PRESENT,
            b.name: IndexStatus.STALE,
            c.name: IndexStatus.INVALID,
            d.name: IndexStatus.MISSING,
            "ix_chunks_meta_gone": IndexStatus.ORPHAN,
        }
        assert not plan.in_sync

    @pytest.mark.parametrize("comment", [None, "someone else's note", ""])
    def test_an_index_without_our_fingerprint_is_unmanaged(self, comment: str | None) -> None:
        (a, *_) = self._specs()
        plan = diff_indexes([a], [_row(a, comment=comment)])
        assert plan.entries[0].status is IndexStatus.UNMANAGED
        assert not plan.in_sync

    def test_a_legacy_or_malformed_comment_of_ours_is_stale(self) -> None:
        (a, *_) = self._specs()
        for comment in ("agintis:expr:0123456789ab", "agintis:expr:garbage"):
            plan = diff_indexes([a], [_row(a, comment=comment)])
            assert plan.entries[0].status is IndexStatus.STALE

    def test_a_definition_changed_behind_our_back_is_unmanaged(self) -> None:
        """Right fingerprint comment, but the stored definition is not the one we created."""
        (a, *_) = self._specs()
        plan = diff_indexes([a], [_row(a, indexdef="CREATE INDEX x ON chunks (other)")])
        assert plan.entries[0].status is IndexStatus.UNMANAGED

    def test_an_unreadable_definition_is_not_compared(self) -> None:
        (a, *_) = self._specs()
        assert diff_indexes([a], [_row(a, indexdef=None)]).in_sync

    def test_an_index_being_built_is_building_not_invalid(self) -> None:
        a, b, *_ = self._specs()
        plan = diff_indexes(
            [a], [_row(a, valid=False, building=True), _row(b, valid=False, building=True)]
        )
        assert {e.name: e.status for e in plan.entries} == {
            a.name: IndexStatus.BUILDING,
            b.name: IndexStatus.BUILDING,  # undeclared but still building: not an orphan to drop
        }

    def test_everything_present_is_in_sync_and_empty_is_in_sync(self) -> None:
        specs = self._specs()
        assert diff_indexes(specs, [_row(s) for s in specs]).in_sync
        assert diff_indexes([], []).in_sync

    def test_the_fingerprint_changes_with_the_definition(self) -> None:
        (text_spec,) = expected_metadata_indexes([_field("a")])
        (number_spec,) = expected_metadata_indexes([_field("a", "int")])
        assert text_spec.fingerprint != number_spec.fingerprint

    def test_the_index_is_partial_on_the_expression_not_being_null(self) -> None:
        (spec,) = expected_metadata_indexes([_field("a")])
        assert spec.create_sql.endswith(f"WHERE ({spec.expression}) IS NOT NULL")
        assert spec.fingerprint != hashlib.sha256(spec.expression.encode()).hexdigest()[:12]

    def test_the_comment_records_fingerprint_and_definition_hash(self) -> None:
        (spec,) = expected_metadata_indexes([_field("a")])
        comment = spec.comment_for("CREATE INDEX ...")
        assert comment.startswith(f"agintis:expr:{spec.fingerprint}:def:")
        assert comment != spec.comment_for("CREATE INDEX ... different")


class TestSyncConstruction:
    def test_the_ddl_engine_must_be_unpooled(self) -> None:
        pooled = create_engine(
            SecretStr("postgresql+psycopg://u:p@localhost/db"),
            pool_size=1,
            max_overflow=0,
            pool_timeout_seconds=1,
            pool_recycle_seconds=60,
        )
        unpooled = create_unpooled_engine(SecretStr("postgresql+psycopg://u:p@localhost/db"))
        with pytest.raises(ValueError, match="unpooled"):
            MetadataIndexSync(pooled, pooled)
        with pytest.raises(ValueError, match="lock_timeout"):
            MetadataIndexSync(pooled, unpooled, lock_timeout_seconds=0)
        MetadataIndexSync(pooled, unpooled)


class TestSearchSettings:
    def _repo(self, **kwargs: int) -> PgChunkRepository:
        return PgChunkRepository(sa.null(), **kwargs)  # type: ignore[arg-type]

    def test_ef_search_is_raised_to_k_and_capped_at_pgvectors_limit(self) -> None:
        repo = self._repo(ef_search=40)
        assert repo._search_settings(10)["ef_search"] == "40"
        assert repo._search_settings(200)["ef_search"] == "200"
        assert repo._search_settings(5000)["ef_search"] == str(MAX_EF_SEARCH)

    def test_every_search_carries_the_statement_timeout(self) -> None:
        assert self._repo()._search_settings(1)["statement_timeout"] == "10000"
        assert self._repo(search_timeout_ms=250)._search_settings(1)["statement_timeout"] == "250"

    def test_the_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="search_timeout_ms"):
            self._repo(search_timeout_ms=0)


class TestTextSearchConfigSetting:
    @pytest.mark.parametrize("name", ["public.es_unaccent", "simple", "spanish"])
    def test_valid_names(self, name: str) -> None:
        PgChunkRepository(sa.null(), text_search_config=name)  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["", "a b", "x;drop table chunks", "a.b.c", "es-unaccent"])
    def test_invalid_names_are_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="text search configuration"):
            PgChunkRepository(sa.null(), text_search_config=name)  # type: ignore[arg-type]
