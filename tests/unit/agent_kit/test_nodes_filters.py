"""Dynamic filter schema, value conformance, FilterExpr building and the extractor."""

from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agintis.agent_kit.nodes import (
    ExtractFiltersNode,
    FilterClause,
    FilterExtractor,
    build_filter_schema,
    clauses_from_values,
    clauses_to_filter,
    plan_filter_fields,
)
from agintis.agent_kit.nodes.filters import field_kind
from agintis.modules.knowledge.api import MetadataField, RagProfile
from agintis.shared_kernel.criteria import AllOf, Filter
from tests.fakes.llm import RaisingModel, ScriptedChatModel, chat_models_for, rate_limit_error
from tests.unit.agent_kit.support import runtime_for


def _profile(*fields: MetadataField) -> RagProfile:
    return RagProfile("p", 3, 100, "simple", "structured", metadata_fields=fields)


FIELDS = (
    MetadataField("kind", "str", filterable=True),
    MetadataField("year", "int", filterable=True, indexed=True),
    MetadataField("issued", "date", filterable=True),
    MetadataField("tags", "list[str]", filterable=True),
    MetadataField("refs", "list[int]", filterable=True),
    MetadataField("weight", "float", filterable=True),
    MetadataField("secret", "str"),
)
PROFILE = _profile(*FIELDS)


def _empty_reply(profile: RagProfile) -> dict[str, Any]:
    """Every wire property of the profile's filter schema, all null."""
    schema = build_filter_schema(plan_filter_fields(profile))
    return {name: None for name in schema.model_json_schema()["properties"]}


def _open_objects(node: Any) -> list[str]:
    """Strict-mode violations in a JSON schema: open objects and non-required properties."""
    problems: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                problems.append("object without additionalProperties: false")
            if set(node.get("required", [])) != set(node.get("properties", {})):
                problems.append("properties not all required")
        for key, value in node.items():
            if key == "additionalProperties" and value not in (False, None):
                problems.append("open additionalProperties")
            problems += _open_objects(value)
    elif isinstance(node, list):
        for item in node:
            problems += _open_objects(item)
    return problems


class TestSchema:
    def test_only_filterable_fields_are_planned_in_profile_order(self) -> None:
        plans = plan_filter_fields(PROFILE)
        assert [p.name for p in plans] == ["kind", "year", "issued", "tags", "refs", "weight"]

    def test_kinds_eq_for_scalars_and_lists_range_for_dates_and_indexed_numbers(self) -> None:
        kinds = {p.name: p.kind for p in plan_filter_fields(PROFILE)}
        assert kinds == {
            "kind": "eq",
            "year": "range",  # int + indexed
            "issued": "range",  # date
            "tags": "eq",  # a list field: eq = contains
            "refs": "eq",
            "weight": "eq",  # float but not indexed
        }
        assert field_kind(MetadataField("n", "int", filterable=True)) == "eq"

    def test_schema_has_typed_nullable_required_properties_only(self) -> None:
        schema = build_filter_schema(plan_filter_fields(PROFILE)).model_json_schema()
        props = schema["properties"]
        assert set(props) == {
            "kind",
            "year_gte",
            "year_lte",
            "issued_gte",
            "issued_lte",
            "tags",
            "refs",
            "weight",
        }
        assert set(schema["required"]) == set(props)  # required...
        for name, spec in props.items():  # ...and nullable
            assert {"type": "null"} in spec["anyOf"], name
        assert {"type": "integer"} in props["refs"]["anyOf"]  # list[int]: its element type
        assert {"type": "string"} in props["tags"]["anyOf"]
        assert {"type": "number"} in props["weight"]["anyOf"]
        assert _open_objects(schema) == []  # strict json_schema compatible: no open objects

    def test_hints_are_part_of_the_property_descriptions(self) -> None:
        plans = plan_filter_fields(PROFILE, {"kind": "The document kind, e.g. memo."})
        props = build_filter_schema(plans).model_json_schema()["properties"]
        assert list(props) == ["kind"]
        assert props["kind"]["description"].startswith("The document kind, e.g. memo.")

    def test_unknown_or_not_filterable_hint_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="secret"):
            plan_filter_fields(PROFILE, {"secret": "x"})
        with pytest.raises(ValueError, match="nope"):
            plan_filter_fields(PROFILE, {"nope": "x"})

    def test_a_range_bound_name_cannot_collide_with_another_field(self) -> None:
        clash = _profile(
            MetadataField("year", "int", filterable=True, indexed=True),
            MetadataField("year_gte", "int", filterable=True),
        )
        with pytest.raises(ValueError, match="collide"):
            plan_filter_fields(clash)

    def test_no_filterable_fields_means_no_plans(self) -> None:
        assert plan_filter_fields(_profile(MetadataField("secret", "str"))) == ()
        assert plan_filter_fields(PROFILE, {}) == ()


class TestClauses:
    def test_values_are_typed_exactly_as_at_ingest(self) -> None:
        plans = plan_filter_fields(PROFILE)
        clauses = clauses_from_values(
            plans,
            {
                "kind": "  memo ",
                "year_gte": 2020,
                "issued_lte": "2024-12-31",
                "tags": "x",
                "refs": 31,
                "weight": 2,
            },
        )
        assert clauses == [
            {"field": "kind", "op": "eq", "value": "memo"},
            {"field": "year", "op": "gte", "value": 2020},
            {"field": "issued", "op": "lte", "value": "2024-12-31"},
            {"field": "tags", "op": "eq", "value": "x"},
            {"field": "refs", "op": "eq", "value": 31},
            {"field": "weight", "op": "eq", "value": 2},
        ]

    def test_null_blank_and_invalid_values_are_dropped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        plans = plan_filter_fields(PROFILE)
        values = {
            "kind": "   ",  # blank
            "year_gte": "soon",  # not an integer
            "issued_gte": "5 May 2024",  # not an ISO date
            "issued_lte": None,
            "tags": 7,  # wrong kind for list[str]
            "refs": True,  # a bool is not an int
            "weight": "heavy",
        }
        with caplog.at_level("DEBUG", logger="agintis.agent_kit.nodes.filters"):
            assert clauses_from_values(plans, values) == []
        assert "Dropped filter value" in caplog.text

    def test_an_inverted_range_is_dropped(self) -> None:
        plans = plan_filter_fields(PROFILE)
        assert clauses_from_values(plans, {"year_gte": 2025, "year_lte": 2020}) == []
        both = clauses_from_values(plans, {"year_gte": 2020, "year_lte": 2025})
        assert [c["op"] for c in both] == ["gte", "lte"]

    def test_one_bad_value_does_not_drop_the_others(self) -> None:
        plans = plan_filter_fields(PROFILE)
        clauses = clauses_from_values(plans, {"kind": "memo", "year_gte": "nope"})
        assert clauses == [{"field": "kind", "op": "eq", "value": "memo"}]

    def test_filter_expression_is_an_and_of_the_clauses(self) -> None:
        assert clauses_to_filter([]) is None
        one: list[FilterClause] = [{"field": "kind", "op": "eq", "value": "memo"}]
        assert clauses_to_filter(one) == Filter.eq("kind", "memo")
        many: list[FilterClause] = [
            {"field": "kind", "op": "eq", "value": "memo"},
            {"field": "year", "op": "gte", "value": 2020},
            {"field": "year", "op": "lte", "value": 2025},
        ]
        assert clauses_to_filter(many) == AllOf(
            Filter.eq("kind", "memo"), Filter.gte("year", 2020), Filter.lte("year", 2025)
        )


class TestExtractor:
    async def test_asks_the_analyzer_with_the_dynamic_schema_and_returns_typed_clauses(
        self,
    ) -> None:
        model = ScriptedChatModel(
            structured=[
                {
                    "kind": "memo",
                    "year_gte": 2021,
                    "year_lte": None,
                    "issued_gte": None,
                    "issued_lte": None,
                    "tags": None,
                    "refs": 31,
                    "weight": None,
                }
            ]
        )
        extractor = FilterExtractor(chat_models_for(model), PROFILE, prompt="EXTRACT")

        clauses = await extractor.extract("memos since 2021 amended by 31")

        assert clauses == [
            {"field": "kind", "op": "eq", "value": "memo"},
            {"field": "year", "op": "gte", "value": 2021},
            {"field": "refs", "op": "eq", "value": 31},
        ]
        assert model.schemas == [extractor.schema]
        system, human = model.structured_seen[0]
        assert (system, human) == (
            SystemMessage("EXTRACT"),
            HumanMessage("memos since 2021 amended by 31"),
        )

    async def test_no_filterable_fields_never_calls_the_model(self) -> None:
        model = ScriptedChatModel()
        extractor = FilterExtractor(chat_models_for(model), _profile(MetadataField("s", "str")))

        assert not extractor.enabled
        assert await extractor.extract("anything") == []
        assert model.schemas == []

    async def test_a_blank_query_never_calls_the_model(self) -> None:
        model = ScriptedChatModel()
        assert await FilterExtractor(chat_models_for(model), PROFILE).extract("  ") == []
        assert model.schemas == []

    async def test_unusable_model_output_yields_no_filters_and_logs_no_user_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        model = ScriptedChatModel(structured=["this is not json SECRET-USER-TEXT"])
        extractor = FilterExtractor(chat_models_for(model), PROFILE)

        with caplog.at_level("DEBUG"):
            assert await extractor.extract("q") == []
        assert "SECRET-USER-TEXT" not in caplog.text

    async def test_a_provider_error_propagates(self) -> None:
        error = rate_limit_error()
        extractor = FilterExtractor(chat_models_for(RaisingModel(error=error)), PROFILE)

        with pytest.raises(type(error)):
            await extractor.extract("q")

    async def test_one_bad_field_drops_only_that_field(self) -> None:
        # `ref` is an int: "REF 123" (and True) fail the strict schema as a whole; the raw JSON
        # is then validated field by field, so `kind` survives.
        profile = _profile(
            MetadataField("kind", "str", filterable=True),
            MetadataField("ref", "int", filterable=True),
        )
        replies = [
            {"kind": "memo", "ref": "REF 123"},
            {"kind": "memo", "ref": True},
            {"kind": "memo", "ref": 123},
            {"kind": 7, "ref": 123},
        ]
        model = ScriptedChatModel(structured=replies)
        extractor = FilterExtractor(chat_models_for(model), profile)

        results = [await extractor.extract("q") for _ in replies]

        kind = {"field": "kind", "op": "eq", "value": "memo"}
        assert results == [
            [kind],
            [kind],
            [kind, {"field": "ref", "op": "eq", "value": 123}],
            [{"field": "ref", "op": "eq", "value": 123}],
        ]

    async def test_a_bool_is_never_coerced_into_a_number_field(self) -> None:
        profile = _profile(
            MetadataField("n", "int", filterable=True), MetadataField("w", "float", filterable=True)
        )
        model = ScriptedChatModel(structured=[{"n": True, "w": True}, {"n": False, "w": 1}])
        extractor = FilterExtractor(chat_models_for(model), profile)

        assert await extractor.extract("q") == []
        assert await extractor.extract("q") == [{"field": "w", "op": "eq", "value": 1}]

    async def test_the_structured_binding_is_built_once(self) -> None:
        model = ScriptedChatModel(structured=[_empty_reply(PROFILE), _empty_reply(PROFILE)])
        extractor = FilterExtractor(chat_models_for(model), PROFILE)

        await extractor.extract("a")
        await extractor.extract("b")

        assert len(model.schemas) == 1


class TestUnsafeFieldNames:
    """Metadata names that would clash with pydantic as attributes work through aliases."""

    NAMES = ("_x", "model_config", "model_dump", "model_fields", "schema", "copy")
    PROFILE = _profile(*(MetadataField(n, "str", filterable=True) for n in NAMES))

    def test_the_schema_builds_and_exposes_the_exact_names(self) -> None:
        schema = build_filter_schema(plan_filter_fields(self.PROFILE))

        assert tuple(schema.model_json_schema()["properties"]) == self.NAMES
        assert all(name.startswith("f") for name in schema.model_fields)  # internal names
        parsed = schema.model_validate({n: f"v-{n}" for n in self.NAMES})
        assert parsed.model_dump(by_alias=True) == {n: f"v-{n}" for n in self.NAMES}

    async def test_extraction_round_trips_every_name(self) -> None:
        reply = {n: (f"v-{n}" if n in ("_x", "model_config") else None) for n in self.NAMES}
        model = ScriptedChatModel(structured=[reply])
        extractor = FilterExtractor(chat_models_for(model), self.PROFILE)

        assert await extractor.extract("q") == [
            {"field": "_x", "op": "eq", "value": "v-_x"},
            {"field": "model_config", "op": "eq", "value": "v-model_config"},
        ]

    def test_a_range_field_with_an_unsafe_name_uses_aliased_bounds(self) -> None:
        profile = _profile(MetadataField("model_dump", "date", filterable=True))
        schema = build_filter_schema(plan_filter_fields(profile))
        assert tuple(schema.model_json_schema()["properties"]) == (
            "model_dump_gte",
            "model_dump_lte",
        )

    async def test_the_node_filters_the_standalone_query_and_replaces_stale_filters(self) -> None:
        reply = _empty_reply(PROFILE)
        model = ScriptedChatModel(structured=[{**reply, "kind": "memo"}])
        node = ExtractFiltersNode({"t": FilterExtractor(chat_models_for(model), PROFILE)})
        state: Any = {"topic": "t", "query": "standalone", "filters": [{"x": 1}]}

        result = await node(state, runtime_for())

        assert result == {"filters": [{"field": "kind", "op": "eq", "value": "memo"}]}
        assert model.structured_seen[0][1] == HumanMessage("standalone")

    async def test_the_node_without_fields_resets_the_previous_filters(self) -> None:
        extractor = FilterExtractor(chat_models_for(), _profile())
        state: Any = {"topic": "t", "query": "q", "filters": [{"stale": True}]}

        assert await ExtractFiltersNode({"t": extractor})(state, runtime_for()) == {"filters": []}

    async def test_each_topic_uses_its_own_extractor(self) -> None:
        reply = _empty_reply(PROFILE)
        model = ScriptedChatModel(structured=[{**reply, "kind": "memo"}])
        node = ExtractFiltersNode(
            {
                "a": FilterExtractor(chat_models_for(model), PROFILE),
                "b": FilterExtractor(chat_models_for(), _profile()),
            }
        )

        with_fields = await node({"topic": "a", "query": "q"}, runtime_for())
        without = await node({"topic": "b", "query": "q"}, runtime_for())

        assert with_fields == {"filters": [{"field": "kind", "op": "eq", "value": "memo"}]}
        assert without == {"filters": []}


def test_schema_instances_forbid_extra_fields() -> None:
    schema: type[BaseModel] = build_filter_schema(plan_filter_fields(PROFILE))
    with pytest.raises(ValueError, match="extra"):
        schema.model_validate({**_empty_reply(PROFILE), "bogus": 1})
