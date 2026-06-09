"""Versioning support for ontology schemas."""

from __future__ import annotations

from .versioning import (
    SemanticChange,
    SemanticChangeType,
    SemanticDiff,
    OntologyVersion,
    VersionManager,
)

__all__ = [
    "SemanticChange",
    "SemanticChangeType",
    "SemanticDiff",
    "OntologyVersion",
    "VersionManager",
]
