"""
ontology_events — Event system for the Ontology Runtime Platform.

Defines :class:`EventType`, :class:`OntologyEvent`, and the
:class:`EventBus` pub-sub mechanism.
"""

from __future__ import annotations

from .events import EventBus, EventType, OntologyEvent

__all__ = [
    "EventBus",
    "EventType",
    "OntologyEvent",
]
