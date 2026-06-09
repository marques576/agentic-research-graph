"""
Event system for the Ontology Runtime Platform.

Provides :class:`EventType` (enum of all domain events),
:class:`OntologyEvent` (an event instance with metadata), and
:class:`EventBus` (pub-sub mechanism for subscribing and emitting
events).
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

# ---------------------------------------------------------------------------
# EventType
# ---------------------------------------------------------------------------


class EventType(enum.Enum):
    """
    Enumeration of all ontology domain events.

    Each value represents a distinct kind of event that can be emitted
    by the runtime.
    """

    # Object lifecycle
    OBJECT_ADDED = "object_added"
    OBJECT_REMOVED = "object_removed"
    OBJECT_UPDATED = "object_updated"

    # Relation lifecycle
    RELATION_ADDED = "relation_added"
    RELATION_REMOVED = "relation_removed"

    # Property lifecycle
    PROPERTY_ADDED = "property_added"
    PROPERTY_REMOVED = "property_removed"

    # Proposal lifecycle (collaborative ontology editing)
    PROPOSAL_CREATED = "proposal_created"
    PROPOSAL_APPROVED = "proposal_approved"
    PROPOSAL_REJECTED = "proposal_rejected"

    # Validation
    VALIDATION_FAILED = "validation_failed"

    # Workspace lifecycle
    WORKSPACE_CREATED = "workspace_created"
    WORKSPACE_SWITCHED = "workspace_switched"
    WORKSPACE_MERGED = "workspace_merged"


# ---------------------------------------------------------------------------
# OntologyEvent
# ---------------------------------------------------------------------------


@dataclass
class OntologyEvent:
    """
    A single event in the ontology runtime.

    Attributes
    ----------
    id : str
        Unique event identifier (default: random UUID hex).
    type : EventType
        The kind of event.
    data : dict
        Event-specific payload (e.g. the affected entity's identifier,
        the new state, etc.).
    timestamp : datetime
        When the event occurred (default: UTC now).
    source : str
        Identifier of the component that emitted the event
        (e.g. ``"ontology_operations"``, ``"protocol_adapter"``).
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    type: EventType = EventType.OBJECT_ADDED
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""

    def __repr__(self) -> str:
        return (
            f"OntologyEvent(id={self.id!r}, "
            f"type={self.type.value}, "
            f"source={self.source!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OntologyEvent):
            return NotImplemented
        return self.id == other.id


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

_Callback = Callable[[OntologyEvent], None]
_SubscriptionId = str


@dataclass
class _Subscription:
    """Internal subscription record."""

    id: str
    event_type: EventType | None  # None = all events
    callback: _Callback


class EventBus:
    """
    Publish-subscribe event bus for ontology events.

    Subscribers register callbacks for specific :class:`EventType` values
    (or all events).  When :meth:`emit` is called, all matching callbacks
    are invoked synchronously in subscription order.

    Attributes
    ----------
    subscriptions : list[tuple[str, EventType | None, Callable]]
        List of (subscription_id, event_type, callback) triples.
        Exposed for introspection; use :meth:`subscribe` and
        :meth:`unsubscribe` to modify.
    """

    def __init__(self) -> None:
        """Initialise an empty event bus."""
        self._subscriptions: list[_Subscription] = []

    @property
    def subscriptions(
        self,
    ) -> list[tuple[str, EventType | None, _Callback]]:
        """
        Return a read-only view of current subscriptions.

        Returns
        -------
        list of (subscription_id, event_type, callback) tuples.
        """
        return [(s.id, s.event_type, s.callback) for s in self._subscriptions]

    def subscribe(
        self,
        event_type: EventType,
        callback: _Callback,
    ) -> str:
        """
        Register a callback for the given event type.

        Parameters
        ----------
        event_type : EventType
            The event type to subscribe to.
        callback : Callable[[OntologyEvent], None]
            Callable to invoke when an event of the given type is emitted.

        Returns
        -------
        str
            A subscription identifier that can be passed to
            :meth:`unsubscribe`.
        """
        sub_id = uuid.uuid4().hex
        self._subscriptions.append(
            _Subscription(id=sub_id, event_type=event_type, callback=callback)
        )
        return sub_id

    def subscribe_all(self, callback: _Callback) -> str:
        """
        Register a callback for *all* event types.

        Parameters
        ----------
        callback : Callable[[OntologyEvent], None]
            Callable to invoke for every emitted event.

        Returns
        -------
        str
            A subscription identifier that can be passed to
            :meth:`unsubscribe`.
        """
        sub_id = uuid.uuid4().hex
        self._subscriptions.append(
            _Subscription(id=sub_id, event_type=None, callback=callback)
        )
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove a subscription by its identifier.

        Parameters
        ----------
        subscription_id : str
            The subscription identifier returned by :meth:`subscribe` or
            :meth:`subscribe_all`.

        Returns
        -------
        bool
            ``True`` if a subscription was removed, ``False`` if the
            identifier was not found.
        """
        for i, sub in enumerate(self._subscriptions):
            if sub.id == subscription_id:
                self._subscriptions.pop(i)
                return True
        return False

    def emit(self, event: OntologyEvent) -> None:
        """
        Emit an event to all matching subscribers.

        Callbacks are invoked synchronously.  If a callback raises an
        exception, it is propagated immediately; subsequent callbacks
        for the same event will **not** be called.

        Parameters
        ----------
        event : OntologyEvent
            The event to emit.
        """
        for sub in self._subscriptions:
            if sub.event_type is None or sub.event_type == event.type:
                sub.callback(event)

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._subscriptions.clear()

    def __repr__(self) -> str:
        return f"EventBus(subscriptions={len(self._subscriptions)})"


__all__ = [
    "EventBus",
    "EventType",
    "OntologyEvent",
]
