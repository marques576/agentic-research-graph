"""Tests for core/schema.py — ObjectType, LinkType, and validation logic."""

import pytest

from core.schema import (
    ObjectType,
    LinkType,
    PropertyDef,
    validate_data_type,
    validate_cardinality,
    validate_property_value,
    validate_object_properties,
)


class TestPropertyDef:
    def test_basic(self):
        pd = PropertyDef(name="age", data_type="number", required=True)
        assert pd.name == "age"
        assert pd.data_type == "number"
        assert pd.required is True

    def test_to_dict(self):
        pd = PropertyDef(name="name", data_type="string", required=True)
        d = pd.to_dict()
        assert d == {"name": "name", "data_type": "string", "required": True}

    def test_from_dict(self):
        data = {"name": "score", "data_type": "number", "required": False}
        pd = PropertyDef.from_dict(data)
        assert pd.name == "score"
        assert pd.data_type == "number"
        assert pd.required is False

    def test_from_dict_defaults_required(self):
        pd = PropertyDef.from_dict({"name": "x", "data_type": "string"})
        assert pd.required is False


class TestObjectType:
    def test_basic(self):
        ot = ObjectType(
            name="Person",
            properties=[
                PropertyDef("name", "string", True),
                PropertyDef("age", "number", False),
            ],
            description="A human being",
        )
        assert ot.name == "Person"
        assert len(ot.properties) == 2
        assert ot.description == "A human being"

    def test_to_dict_and_back(self):
        ot = ObjectType(
            name="Document",
            properties=[PropertyDef("title", "string", True)],
            description="A written record",
        )
        d = ot.to_dict()
        ot2 = ObjectType.from_dict(d)
        assert ot2.name == ot.name
        assert len(ot2.properties) == 1
        assert ot2.properties[0].name == "title"
        assert ot2.description == ot.description

    def test_strips_name(self):
        ot = ObjectType(name="  Person  ", properties=[], description="")
        assert ot.name == "Person"


class TestLinkType:
    def test_basic(self):
        lt = LinkType(
            name="EMPLOYED_BY",
            source_type="Person",
            target_type="Organization",
            cardinality="many_to_one",
            description="Employment relationship",
        )
        assert lt.name == "EMPLOYED_BY"
        assert lt.source_type == "Person"
        assert lt.target_type == "Organization"
        assert lt.cardinality == "many_to_one"

    def test_defaults(self):
        lt = LinkType(name="KNOWS", source_type="Person", target_type="Person")
        assert lt.cardinality == "many_to_many"
        assert lt.description == ""

    def test_to_dict_and_back(self):
        lt = LinkType(
            name="AUTHORED",
            source_type="Person",
            target_type="Document",
            cardinality="one_to_many",
        )
        d = lt.to_dict()
        lt2 = LinkType.from_dict(d)
        assert lt2.name == lt.name
        assert lt2.cardinality == "one_to_many"


class TestValidateDataType:
    def test_valid_types(self):
        for dt in ("string", "number", "boolean", "datetime", "reference"):
            assert validate_data_type(dt) == dt

    def test_case_insensitive(self):
        assert validate_data_type("STRING") == "string"
        assert validate_data_type("Boolean") == "boolean"

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid data type"):
            validate_data_type("foobar")


class TestValidateCardinality:
    def test_valid(self):
        for c in ("one_to_one", "one_to_many", "many_to_many"):
            assert validate_cardinality(c) == c

    def test_invalid(self):
        with pytest.raises(ValueError, match="Invalid cardinality"):
            validate_cardinality("one_to_infinity")


class TestValidatePropertyValue:
    def test_string_valid(self):
        validate_property_value(PropertyDef("name", "string", True), "Alice")

    def test_string_invalid(self):
        with pytest.raises(ValueError, match="expected string"):
            validate_property_value(PropertyDef("name", "string", True), 42)

    def test_number_valid_int(self):
        validate_property_value(PropertyDef("age", "number", True), 42)

    def test_number_valid_float(self):
        validate_property_value(PropertyDef("score", "number", False), 3.14)

    def test_number_invalid(self):
        with pytest.raises(ValueError, match="expected number"):
            validate_property_value(PropertyDef("age", "number", True), "42")

    def test_number_not_bool(self):
        with pytest.raises(ValueError, match="expected number"):
            validate_property_value(PropertyDef("flag", "number", True), True)

    def test_boolean_valid(self):
        validate_property_value(PropertyDef("active", "boolean", True), True)
        validate_property_value(PropertyDef("active", "boolean", True), False)

    def test_boolean_invalid(self):
        with pytest.raises(ValueError, match="expected boolean"):
            validate_property_value(PropertyDef("active", "boolean", True), 1)

    def test_datetime_valid(self):
        validate_property_value(
            PropertyDef("created", "datetime", True), "2024-01-15T10:30:00+00:00"
        )

    def test_datetime_invalid(self):
        with pytest.raises(ValueError, match="not a valid ISO 8601"):
            validate_property_value(PropertyDef("created", "datetime", True), "not-a-date")

    def test_datetime_wrong_type(self):
        with pytest.raises(ValueError, match="expected datetime string"):
            validate_property_value(PropertyDef("created", "datetime", True), 42)

    def test_reference_valid(self):
        validate_property_value(
            PropertyDef("doc_id", "reference", True),
            "550e8400-e29b-41d4-a716-446655440000",
        )

    def test_reference_invalid(self):
        with pytest.raises(ValueError, match="expected reference"):
            validate_property_value(PropertyDef("doc_id", "reference", True), 123)

    def test_optional_null_ok(self):
        validate_property_value(PropertyDef("nickname", "string", False), None)

    def test_required_null_fails(self):
        with pytest.raises(ValueError, match="required but value is None"):
            validate_property_value(PropertyDef("name", "string", True), None)


class TestValidateObjectProperties:
    def test_valid(self):
        ot = ObjectType("Person", [
            PropertyDef("name", "string", True),
            PropertyDef("age", "number", False),
        ])
        validate_object_properties(ot, {"name": "Alice", "age": 30})

    def test_missing_required(self):
        ot = ObjectType("Person", [
            PropertyDef("name", "string", True),
            PropertyDef("age", "number", True),
        ])
        with pytest.raises(ValueError, match="Missing required property 'age'"):
            validate_object_properties(ot, {"name": "Alice"})

    def test_unknown_property(self):
        ot = ObjectType("Person", [PropertyDef("name", "string", True)])
        with pytest.raises(ValueError, match="Unknown property 'extra'"):
            validate_object_properties(ot, {"name": "Alice", "extra": "bad"})

    def test_wrong_type(self):
        ot = ObjectType("Person", [PropertyDef("name", "string", True)])
        with pytest.raises(ValueError, match="expected string"):
            validate_object_properties(ot, {"name": 123})

    def test_empty_properties(self):
        ot = ObjectType("Empty", [])
        validate_object_properties(ot, {})
