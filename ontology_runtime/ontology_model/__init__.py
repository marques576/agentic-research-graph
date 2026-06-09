"""
ontology_model — Canonical data model for the Ontology Runtime Platform.

Provides all core data types: Ontology, ObjectType, PropertyType,
RelationType, Constraint, Rule, Action, Workflow, and View.
All classes support ``to_dict()`` / ``from_dict()`` serialisation.
"""

from __future__ import annotations

from .types import (
    Action,
    Constraint,
    ObjectType,
    Ontology,
    PropertyType,
    RelationType,
    Rule,
    View,
    Workflow,
)

__all__ = [
    "Action",
    "Constraint",
    "ObjectType",
    "Ontology",
    "PropertyType",
    "RelationType",
    "Rule",
    "View",
    "Workflow",
]
