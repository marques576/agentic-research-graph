"""
Object Type / Link Type / Property definitions and validation logic.

Data types are limited to: string | number | boolean | datetime | reference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


DataType = Literal["string", "number", "boolean", "datetime", "reference"]
Cardinality = Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]

VALID_DATA_TYPES: set[str] = {"string", "number", "boolean", "datetime", "reference"}
VALID_CARDINALITIES: set[str] = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}


@dataclass
class PropertyDef:
    """Definition of a single property on an Object Type."""
    name: str
    data_type: DataType
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PropertyDef:
        return cls(
            name=data["name"],
            data_type=data["data_type"],
            required=data.get("required", False),
        )


@dataclass
class ObjectType:
    """A declared type of entity (e.g. Person, Organization, Document)."""
    name: str
    properties: list[PropertyDef] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        self.name = self.name.strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "properties": [p.to_dict() for p in self.properties],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectType:
        props = [PropertyDef.from_dict(p) for p in data.get("properties", [])]
        return cls(
            name=data["name"],
            properties=props,
            description=data.get("description", ""),
        )


@dataclass
class LinkType:
    """A declared relationship between two Object Types."""
    name: str
    source_type: str
    target_type: str
    cardinality: Cardinality = "many_to_many"
    description: str = ""

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.source_type = self.source_type.strip()
        self.target_type = self.target_type.strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "cardinality": self.cardinality,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinkType:
        return cls(
            name=data["name"],
            source_type=data["source_type"],
            target_type=data["target_type"],
            cardinality=data.get("cardinality", "many_to_many"),
            description=data.get("description", ""),
        )


def validate_data_type(data_type: str) -> DataType:
    """Validate and normalise a data type string."""
    dt = data_type.strip().lower()
    if dt not in VALID_DATA_TYPES:
        raise ValueError(
            f"Invalid data type '{data_type}'. Must be one of: "
            f"{', '.join(sorted(VALID_DATA_TYPES))}"
        )
    return dt  # type: ignore[return-value]


def validate_cardinality(cardinality: str) -> Cardinality:
    """Validate and normalise a cardinality string."""
    c = cardinality.strip().lower()
    if c not in VALID_CARDINALITIES:
        raise ValueError(
            f"Invalid cardinality '{cardinality}'. Must be one of: "
            f"{', '.join(sorted(VALID_CARDINALITIES))}"
        )
    return c  # type: ignore[return-value]


def validate_property_value(prop_def: PropertyDef, value: Any) -> None:
    """Validate that a property value matches its declared data type.

    Raises ValueError with a descriptive message on failure.
    """
    if value is None:
        if prop_def.required:
            raise ValueError(f"Property '{prop_def.name}' is required but value is None")
        return

    dt = prop_def.data_type

    if dt == "string":
        if not isinstance(value, str):
            raise ValueError(
                f"Property '{prop_def.name}' expected string, got {type(value).__name__}"
            )
    elif dt == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"Property '{prop_def.name}' expected number, got {type(value).__name__}"
            )
    elif dt == "boolean":
        if not isinstance(value, bool):
            raise ValueError(
                f"Property '{prop_def.name}' expected boolean, got {type(value).__name__}"
            )
    elif dt == "datetime":
        if not isinstance(value, str):
            raise ValueError(
                f"Property '{prop_def.name}' expected datetime string, got {type(value).__name__}"
            )
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError):
            raise ValueError(
                f"Property '{prop_def.name}' is not a valid ISO 8601 datetime: {value}"
            )
    elif dt == "reference":
        if not isinstance(value, str):
            raise ValueError(
                f"Property '{prop_def.name}' expected reference (UUID string), got {type(value).__name__}"
            )


def validate_object_properties(
    object_type: ObjectType,
    properties: dict[str, Any],
) -> None:
    """Validate a dict of property values against an ObjectType's property definitions.

    Raises ValueError with a descriptive message on failure.
    """
    prop_map = {p.name: p for p in object_type.properties}

    for prop_def in object_type.properties:
        value = properties.get(prop_def.name)
        if prop_def.required and (value is None or prop_def.name not in properties):
            raise ValueError(
                f"Missing required property '{prop_def.name}' for type '{object_type.name}'"
            )
        if value is not None:
            validate_property_value(prop_def, value)

    for key in properties:
        if key not in prop_map:
            raise ValueError(
                f"Unknown property '{key}' for type '{object_type.name}'. "
                f"Defined properties: {list(prop_map.keys())}"
            )
