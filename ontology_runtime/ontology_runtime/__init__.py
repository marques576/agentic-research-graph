"""Ontology Runtime API – core runtime, OQL query engine, and supporting types."""

from __future__ import annotations

from .runtime import OntologyRuntime
from .query import OntologyQuery, QueryResult

__all__ = [
    "OntologyRuntime",
    "Runtime",
    "OntologyQuery",
    "QueryResult",
]

# Convenience alias
Runtime = OntologyRuntime
