"""
ontology_protocol — Protocol layer for the Ontology Runtime Platform.

Defines :class:`ProtocolRequest`, :class:`ProtocolResponse`,
the :class:`OntologyProtocol` ABC, and :class:`ProtocolAdapter` that
bridges protocol requests to the runtime implementation.
"""

from __future__ import annotations

from .protocol import (
    OntologyProtocol,
    ProtocolAdapter,
    ProtocolRequest,
    ProtocolResponse,
)

__all__ = [
    "OntologyProtocol",
    "ProtocolAdapter",
    "ProtocolRequest",
    "ProtocolResponse",
]
