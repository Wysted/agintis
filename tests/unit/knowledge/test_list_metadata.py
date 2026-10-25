"""List-valued chunk metadata (`list[str]`, `list[int]`) and the new field options."""

from typing import Any

import pytest

from agintis.modules.knowledge.domain import (
    MAX_METADATA_LIST_ITEMS,
    Chunk,
    CleanerSpec,
    ExtractorSpec,
    InvalidMetadataError,
    MetadataField,
    RagProfile,
    RoutingCondition,
    RoutingRule,
    TransformSpec,
    conform_metadata,
    freeze_metadata,
)
from agintis.shared_kernel.ids import ChunkId, DocumentId, TenantId

LAWS = MetadataField("laws", "list[str]", filterable=True)
REFS = MetadataField("refs", "list[int]", filterable=True)


def _raw(**values: Any) -> dict[str, Any]:
    return values


class TestFreeze:
    def test_lists_become_tuples(self) -> None:
        frozen = freeze_metadata(_raw(a=["x", "y"], b=(1, 2), c=[]))
        assert frozen["a"] == ("x", "y")
        assert frozen["b"] == (1, 2)
        assert frozen["c"] == ()

    @pytest.mark.parametrize(
        "value",
        [["a", 1], [1.5], [True], [None], ["a", ["b"]], ["a\x00b"], ["lone \ud800"]],
    )
    def test_malformed_lists_are_rejected(self, value: list[Any]) -> None:
        with pytest.raises(InvalidMetadataError):
            freeze_metadata(_raw(a=value))

    def test_size_is_capped(self) -> None:
        freeze_metadata(_raw(a=["x"] * MAX_METADATA_LIST_ITEMS))
        with pytest.raises(InvalidMetadataError, match="items"):
            freeze_metadata(_raw(a=["x"] * (MAX_METADATA_LIST_ITEMS + 1)))

    def test_the_input_list_is_copied(self) -> None:
        source = ["a"]
        frozen = freeze_metadata(_raw(a=source))
        source.append("b")
        assert frozen["a"] == ("a",)

    def test_a_chunk_freezes_lists_read_back_from_storage(self) -> None:
        chunk = Chunk(
            ChunkId("c"),
            DocumentId("d"),
            TenantId("t"),
            "p",
            0,
            "text",
            _raw(laws=["DL 3500"]),
        )
        assert chunk.metadata["laws"] == ("DL 3500",)


class TestConform:
    def test_str_lists(self) -> None:
        assert conform_metadata(_raw(laws=["DL 3500", "Ley 21735"]), [LAWS])["laws"] == (
            "DL 3500",
            "Ley 21735",
        )

    def test_int_lists_coerce_decimal_strings_item_by_item(self) -> None:
        assert conform_metadata(_raw(refs=["31", 7, " 8 "]), [REFS])["refs"] == (31, 7, 8)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            (LAWS, "DL 3500"),  # a scalar is not a list
            (LAWS, ["ok", 3]),
            (LAWS, [None]),
            (REFS, ["31", "abc"]),
            (REFS, [1.5]),
            (REFS, ["١٢"]),  # non-ASCII digits are ambiguous
            (REFS, 7),
        ],
    )
    def test_wrong_items_are_rejected(self, field: MetadataField, value: Any) -> None:
        with pytest.raises(InvalidMetadataError):
            conform_metadata(_raw(**{field.name: value}), [field])

    def test_the_error_names_the_offending_item(self) -> None:
        with pytest.raises(InvalidMetadataError, match=r"refs\[1\]"):
            conform_metadata(_raw(refs=[1, "x"]), [REFS])

    def test_size_cap(self) -> None:
        with pytest.raises(InvalidMetadataError, match="at most"):
            conform_metadata(_raw(refs=list(range(MAX_METADATA_LIST_ITEMS + 1))), [REFS])

    def test_non_finite_and_nul_are_rejected(self) -> None:
        with pytest.raises(InvalidMetadataError):
            conform_metadata(_raw(laws=["a\x00"]), [LAWS])
        with pytest.raises(InvalidMetadataError):
            conform_metadata(_raw(refs=[float("inf")]), [REFS])


class TestFieldDeclaration:
    def test_is_list(self) -> None:
        assert LAWS.is_list
        assert REFS.is_list
        assert not MetadataField("a", "str").is_list

    def test_unknown_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported type"):
            MetadataField("a", "list[float]")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "source", ["group:a", "document:a", "parsed:a", "const:a", "extractor:a"]
    )
    def test_every_source_kind_is_accepted(self, source: str) -> None:
        assert MetadataField("a", source=source).source == source

    @pytest.mark.parametrize("source", ["", "group:", "nope:a", "a"])
    def test_other_sources_are_rejected(self, source: str) -> None:
        with pytest.raises(ValueError, match="source must start with"):
            MetadataField("a", source=source)

    def test_params_need_an_extractor(self) -> None:
        MetadataField("a", source="extractor:regex_all", params={"pattern": "x"})
        with pytest.raises(ValueError, match="no extractor source"):
            MetadataField("a", source="group:a", params={"pattern": "x"})

    def test_transforms_need_a_source(self) -> None:
        with pytest.raises(ValueError, match="no source"):
            MetadataField("a", transforms=(TransformSpec("value_map"),))

    def test_params_are_frozen(self) -> None:
        raw: dict[str, Any] = {"nested": {"k": [1, 2]}}
        field = MetadataField("a", source="extractor:x", params=raw)
        nested: Any = field.params["nested"]
        assert nested["k"] == (1, 2)
        with pytest.raises(TypeError):
            nested["k"] = ()


def _profile(**overrides: Any) -> RagProfile:
    base: dict[str, Any] = {
        "name": "p",
        "top_k": 3,
        "chunk_max_tokens": 100,
        "parser_key": "simple",
        "chunker_key": "structured",
    }
    return RagProfile(**{**base, **overrides})


class TestFingerprint:
    def test_profiles_without_the_new_options_keep_their_hash(self) -> None:
        plain = _profile(metadata_fields=(MetadataField("a", source="group:a"),))
        assert (
            plain.fingerprint
            == _profile(
                metadata_fields=(MetadataField("a", source="group:a", params={}, transforms=()),)
            ).fingerprint
        )

    def test_field_params_and_transforms_change_the_hash(self) -> None:
        def with_field(**kwargs: Any) -> str:
            return _profile(
                metadata_fields=(MetadataField("a", source="extractor:regex_all", **kwargs),)
            ).fingerprint

        base = with_field(params={"pattern": "x"})
        assert with_field(params={"pattern": "y"}) != base
        assert (
            with_field(params={"pattern": "x"}, transforms=(TransformSpec("date_floor"),)) != base
        )
        assert with_field(params={"pattern": "x", "group": 1}) != base

    def test_param_key_order_does_not_matter_but_transform_order_does(self) -> None:
        one = TransformSpec("value_map", {"map": {"a": "b"}})
        two = TransformSpec("date_floor")

        def with_transforms(*specs: TransformSpec) -> str:
            return _profile(
                metadata_fields=(MetadataField("a", source="const:x", transforms=specs),)
            ).fingerprint

        assert with_transforms(one, two) != with_transforms(two, one)
        assert (
            _profile(
                metadata_fields=(MetadataField("a", source="extractor:r", params={"x": 1, "y": 2}),)
            ).fingerprint
            == _profile(
                metadata_fields=(MetadataField("a", source="extractor:r", params={"y": 2, "x": 1}),)
            ).fingerprint
        )

    def test_named_extractors_change_the_hash(self) -> None:
        fields = (MetadataField("a", source="extractor:t.libro"),)
        base = _profile(metadata_fields=fields)
        named = _profile(
            metadata_fields=fields, extractors={"t": ExtractorSpec("layered_regex", {"keys": {}})}
        )
        other = _profile(
            metadata_fields=fields,
            extractors={"t": ExtractorSpec("layered_regex", {"keys": {"x": 1}})},
        )
        assert len({base.fingerprint, named.fingerprint, other.fingerprint}) == 3

    def test_routing_and_status_are_not_part_of_the_hash(self) -> None:
        rule = RoutingRule((RoutingCondition("doc_type", ("report",)),))
        assert _profile(routing=(rule,), status="draft").fingerprint == _profile().fingerprint

    def test_cleaners_still_count(self) -> None:
        patterns: dict[str, Any] = {"patterns": ["x"]}
        cleaned = _profile(cleaners=(CleanerSpec("strip_lines", patterns),))
        assert cleaned.fingerprint != _profile().fingerprint


class TestRoutingDeclaration:
    def test_a_condition_needs_exactly_one_test(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            RoutingCondition("title")
        with pytest.raises(ValueError, match="exactly one"):
            RoutingCondition("title", ("a",), "b")
        assert RoutingCondition("title", pattern="x").pattern == "x"

    def test_a_rule_needs_a_match(self) -> None:
        with pytest.raises(ValueError, match="at least one match"):
            RoutingRule(())

    def test_extractor_names_must_be_plain_names(self) -> None:
        with pytest.raises(ValueError, match="not a valid name"):
            _profile(extractors={"not valid": ExtractorSpec("x")})
        with pytest.raises(ValueError, match="key must not be empty"):
            ExtractorSpec("")
