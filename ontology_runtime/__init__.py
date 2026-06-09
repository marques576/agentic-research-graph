"""
Ontology Runtime Platform — core packages.

This package provides the foundational building blocks for working with
ontologies in a structured, composable way:

* :mod:`ontology_model`     — Canonical data model (Ontology, ObjectType, etc.)
* :mod:`ontology_operations` — Atomic operations and transactions
* :mod:`ontology_protocol`  — Protocol layer (request/response, adapter)
* :mod:`ontology_events`    — Event system (EventBus, EventType, OntologyEvent)
"""

from __future__ import annotations

from .ontology_model import (
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
from .ontology_operations import (
    OperationType,
    OntologyOperation,
    OntologyTransaction,
    apply_operation,
)
from .ontology_protocol import (
    OntologyProtocol,
    ProtocolAdapter,
    ProtocolRequest,
    ProtocolResponse,
)
from .ontology_events import (
    EventBus,
    EventType,
    OntologyEvent,
)

__all__ = [
    # ontology_model
    "Action",
    "Constraint",
    "ObjectType",
    "Ontology",
    "PropertyType",
    "RelationType",
    "Rule",
    "View",
    "Workflow",
    # ontology_operations
    "OperationType",
    "OntologyOperation",
    "OntologyTransaction",
    "apply_operation",
    # ontology_protocol
    "OntologyProtocol",
    "ProtocolAdapter",
    "ProtocolRequest",
    "ProtocolResponse",
    # ontology_events
    "EventBus",
    "EventType",
    "OntologyEvent",
]
