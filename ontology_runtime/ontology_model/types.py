"""
Canonical data model for the Ontology Runtime Platform.

Defines all core data types used throughout the platform:
Ontology, ObjectType, PropertyType, RelationType, Constraint, Rule,
Action, Workflow, and View.

Every class is a frozen dataclass with ``to_dict()`` / ``from_dict()``
serialisation, ``__repr__``, and ``__eq__``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Ontology Model
# ---------------------------------------------------------------------------


def _new_id() -> str:
    """Generate a new UUID4 hex string as a default identifier."""
    return uuid.uuid4().hex


@dataclass
class PropertyType:
    """
    A property definition on an object or relation type.

    Attributes
    ----------
    id : str
        Unique identifier (default: random UUID hex).
    name : str
        Human-readable property name.
    datatype : str
        The data type of the property (e.g. ``"string"``, ``"integer"``,
        ``"float"``, ``"boolean"``, ``"datetime"``).
    constraints : list[Constraint]
        Constraints that apply to values of this property.
    default : Any
        Default value when not explicitly set.
    required : bool
        Whether a value is required.
    description : str
        Free-text description of the property.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    datatype: str = "string"
    constraints: list[Constraint] = field(default_factory=list)
    default: Any = None
    required: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "datatype": self.datatype,
            "constraints": [c.to_dict() for c in self.constraints],
            "default": self.default,
            "required": self.required,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PropertyType:
        """
        Reconstruct from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialised property data.

        Returns
        -------
        PropertyType
        """
        constraints = [
            Constraint.from_dict(c) for c in data.get("constraints", [])
        ]
        return cls(
            id=data.get("id", _new_id()),
            name=data.get("name", ""),
            datatype=data.get("datatype", "string"),
            constraints=constraints,
            default=data.get("default"),
            required=data.get("required", False),
            description=data.get("description", ""),
        )


@dataclass
class Constraint:
    """
    A validation constraint that can be attached to a property or an object.

    Attributes
    ----------
    id : str
        Unique identifier (default: random UUID hex).
    name : str
        Human-readable constraint name.
    type : str
        Constraint type identifier (e.g. ``"min_length"``, ``"max"``,
        ``"pattern"``, ``"unique"``).
    expression : str
        The constraint expression (e.g. a regex pattern, a numeric bound,
        a Python expression).
    severity : str
        Severity level (``"error"``, ``"warning"``, or ``"info"``).
    description : str
        Free-text description.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    type: str = ""
    expression: str = ""
    severity: str = "error"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "expression": self.expression,
            "severity": self.severity,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Constraint:
        """
        Reconstruct from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialised constraint data.

        Returns
        -------
        Constraint
        """
        return cls(
            id=data.get("id", _new_id()),
            name=data.get("name", ""),
            type=data.get("type", ""),
            expression=data.get("expression", ""),
            severity=data.get("severity", "error"),
            description=data.get("description", ""),
        )


@dataclass
class ObjectType:
    """
    A type/class of objects in the ontology.

    Attributes
    ----------
    id : str
        Unique identifier (default: random UUID hex).
    name : str
        Human-readable object type name.
    description : str
        Free-text description.
    properties : list[PropertyType]
        Properties that belong to this object type.
    metadata : dict
        Arbitrary key-value metadata.
    parent_id : str | None
        Identifier of a parent object type for inheritance.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    description: str = ""
    properties: list[PropertyType] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "properties": [p.to_dict() for p in self.properties],
            "metadata": dict(self.metadata),
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectType:
        """
        Reconstruct from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialised object-type data.

        Returns
        -------
        ObjectType
        """
        properties = [
            PropertyType.from_dict(p) for p in data.get("properties", [])
        ]
        return cls(
            id=data.get("id", _new_id()),
            name=data.get("name", ""),
            description=data.get("description", ""),
            properties=properties,
            metadata=dict(data.get("metadata", {})),
            parent_id=data.get("parent_id"),
        )


@dataclass
class RelationType:
    """
    A type of relationship between two object types.

    Attributes
    ----------
    id : str
        Unique identifier (default: random UUID hex).
    name : str
        Human-readable relation type name (e.g. ``"works_at"``).
    source_type : str
        Identifier of the source (domain) :class:`ObjectType`.
    target_type : str
        Identifier of the target (range) :class:`ObjectType`.
    properties : list[PropertyType]
        Properties that belong to this relation type.
    metadata : dict
        Arbitrary key-value metadata.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    source_type: str = ""
    target_type: str = ""
    properties: list[PropertyType] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "properties": [p.to_dict() for p in self.properties],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationType:
        """
        Reconstruct from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialised relation-type data.

        Returns
        -------
        RelationType
        """
        properties = [
            PropertyType.from_dict(p) for p in data.get("properties", [])
        ]
        return cls(
            id=data.get("id", _new_id()),
            name=data.get("name", ""),
            source_type=data.get("source_type", ""),
            target_type=data.get("target_type", ""),
            properties=properties,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Rule:
    """
    A declarative rule that can trigger consequences when conditions hold.

    Attributes
    ----------
    id : str
        Unique identifier (default: random UUID hex).
    name : str
        Human-readable rule name.
    conditions : list[dict]
        List of condition expressions (each a JSON-safe dict).
    consequences : list[dict]
        List of consequence actions (each a JSON-safe dict).
    description : str
        Free-text description.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    conditions: list[dict[str, Any]] = field(default_factory=list)
    consequences: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "conditions": list(self.conditions),
            "consequences": list(self.consequences),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        """
        Reconstruct from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialised rule data.

        Returns
        -------
        Rule
        """
        return cls(
            id=data.get("id", _new_id()),
            name=data.get("name", ""),
            conditions=list(data.get("conditions", [])),
            consequences=list(data.get("consequences", [])),
            description=data.get("description", ""),
        )


@dataclass
class Action:
    """
    A named action that can be executed in the ontology runtime.

    Actions define the interface (inputs / outputs) for executable
    behaviour bound to the ontology.

    Attributes
    ----------
    id : str
        Unique identifier (default: random UUID hex).
    name : str
        Human-readable action name.
    inputs : list[dict]
        List of input parameter definitions (each a JSON-safe dict).
    outputs : list[dict]
        List of output parameter definitions (each a JSON-safe dict).
    description : str
        Free-text description.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        """
        Reconstruct from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialised action data.

        Returns
        -------
        Action
        """
        return cls(
            id=data.get("id", _new_id()),
            name=data.get("name", ""),
            inputs=list(data.get("inputs", [])),
            outputs=list(data.get("outputs", [])),
            description=data.get("description", ""),
        )


@dataclass
class Workflow:
    """
    A multi-step workflow definition.

    Workflows compose actions, rules, and manual steps into a
    repeatable process.

    Attributes
    ----------
    id : str
        Unique identifier (default: random UUID hex).
    name : str
        Human-readable workflow name.
    steps : list[dict]
        Ordered list of step definitions (each a JSON-safe dict).
    description : str
        Free-text description.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "steps": list(self.steps),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workflow:
        """
        Reconstruct from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialised workflow data.

        Returns
        -------
        Workflow
        """
        return cls(
            id=data.get("id", _new_id()),
            name=data.get("name", ""),
            steps=list(data.get("steps", [])),
            description=data.get("description", ""),
        )


@dataclass
class View:
    """
    A named view / projection over the ontology.

    Views filter which object types are visible and define a layout
    for rendering or querying the ontology.

    Attributes
    ----------
    id : str
        Unique identifier (default: random UUID hex).
    name : str
        Human-readable view name.
    filters : list[dict]
        List of filter expressions (each a JSON-safe dict).
    layout : dict
        Layout configuration (e.g. graph layout hints, colour mapping).
    object_types : list[str]
        List of object type identifiers included in this view.
    """

    id: str = field(default_factory=_new_id)
    name: str = ""
    filters: list[dict[str, Any]] = field(default_factory=list)
    layout: dict[str, Any] = field(default_factory=dict)
    object_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "filters": list(self.filters),
            "layout": dict(self.layout),
            "object_types": list(self.object_types),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> View:
        """
        Reconstruct from a dictionary produced by :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Serialised view data.

        Returns
        -------
        View
        """
        return cls(
            id=data.get("id", _new_id()),
            name=data.get("name", ""),
            filters=list(data.get("filters", [])),
            layout=dict(data.get("layout", {})),
            object_types=list(data.get("object_types", [])),
        )


@dataclass
class Ontology:
    """
    Root container for the complete ontology model.

    The ontology is a collection of object types, relation types,
    property types, constraints, rules, actions, workflows, and views
    that together define a domain schema.

    Attributes
    ----------
    objects : dict[str, ObjectType]
        Object types keyed by their identifier.
    relations : dict[str, RelationType]
        Relation types keyed by their identifier.
    properties : dict[str, PropertyType]
        Property types keyed by their identifier.
    constraints : dict[str, Constraint]
        Constraints keyed by their identifier.
    rules : dict[str, Rule]
        Rules keyed by their identifier.
    actions : dict[str, Action]
        Actions keyed by their identifier.
    workflows : dict[str, Workflow]
        Workflows keyed by their identifier.
    views : dict[str, View]
        Views keyed by their identifier.
    id : str
        Unique identifier for this ontology (default: random UUID hex).
    name : str
        Human-readable ontology name.
    description : str
        Free-text description.
    metadata : dict
        Arbitrary key-value metadata for the ontology.
    """

    objects: dict[str, ObjectType] = field(default_factory=dict)
    relations: dict[str, RelationType] = field(default_factory=dict)
    properties: dict[str, PropertyType] = field(default_factory=dict)
    constraints: dict[str, Constraint] = field(default_factory=dict)
    rules: dict[str, Rule] = field(default_factory=dict)
    actions: dict[str, Action] = field(default_factory=dict)
    workflows: dict[str, Workflow] = field(default_factory=dict)
    views: dict[str, View] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    name: str = ""
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the entire ontology to a JSON-safe dictionary.

        Returns
        -------
        dict
            A nested dictionary representation of the ontology.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": dict(self.metadata),
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "relations": {k: v.to_dict() for k, v in self.relations.items()},
            "properties": {k: v.to_dict() for k, v in self.properties.items()},
            "constraints": {k: v.to_dict() for k, v in self.constraints.items()},
            "rules": {k: v.to_dict() for k, v in self.rules.items()},
            "actions": {k: v.to_dict() for k, v in self.actions.items()},
            "workflows": {k: v.to_dict() for k, v in self.workflows.items()},
            "views": {k: v.to_dict() for k, v in self.views.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Ontology:
        """
        Reconstruct an Ontology from a serialised dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        Ontology
        """
        return cls(
            id=data.get("id", _new_id()),
            name=data.get("name", ""),
            description=data.get("description", ""),
            metadata=dict(data.get("metadata", {})),
            objects={
                k: ObjectType.from_dict(v)
                for k, v in data.get("objects", {}).items()
            },
            relations={
                k: RelationType.from_dict(v)
                for k, v in data.get("relations", {}).items()
            },
            properties={
                k: PropertyType.from_dict(v)
                for k, v in data.get("properties", {}).items()
            },
            constraints={
                k: Constraint.from_dict(v)
                for k, v in data.get("constraints", {}).items()
            },
            rules={
                k: Rule.from_dict(v)
                for k, v in data.get("rules", {}).items()
            },
            actions={
                k: Action.from_dict(v)
                for k, v in data.get("actions", {}).items()
            },
            workflows={
                k: Workflow.from_dict(v)
                for k, v in data.get("workflows", {}).items()
            },
            views={
                k: View.from_dict(v)
                for k, v in data.get("views", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

__all__ = [
    "Ontology",
    "ObjectType",
    "PropertyType",
    "RelationType",
    "Constraint",
    "Rule",
    "Action",
    "Workflow",
    "View",
]
