"""
ontology_operations — Atomic operations and transactions for the Ontology Runtime.

Defines :class:`OperationType`, :class:`OntologyOperation`,
:class:`OntologyTransaction`, and the :func:`apply_operation` dispatcher
for modifying an :class:`~ontology_runtime.ontology_model.Ontology`.
"""

from __future__ import annotations

from .operations import (
    OperationType,
    OntologyOperation,
    OntologyTransaction,
    apply_operation,
)

__all__ = [
    "OperationType",
    "OntologyOperation",
    "OntologyTransaction",
    "apply_operation",
]
