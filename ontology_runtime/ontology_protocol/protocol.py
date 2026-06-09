"""
Protocol layer for the Ontology Runtime Platform.

Provides :class:`ProtocolRequest` and :class:`ProtocolResponse` as the
standard message envelope, the :class:`OntologyProtocol` abstract base
class for implementing protocol handlers, and :class:`ProtocolAdapter`
that translates protocol requests into runtime method calls.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ontology_runtime.ontology_model import Ontology
from ontology_runtime.ontology_operations import (
    OntologyOperation,
    OntologyTransaction,
    apply_operation,
)


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


@dataclass
class ProtocolRequest:
    """
    A structured request in the ontology protocol.

    Attributes
    ----------
    action : str
        The action to perform (e.g. ``"get_object_type"``,
        ``"add_relation_type"``, ``"commit_transaction"``).
    params : dict
        Parameters for the action.
    request_id : str
        Unique identifier for this request (default: random UUID hex).
    timestamp : datetime
        When the request was created (default: UTC now).
    """

    action: str
    params: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return (
            f"ProtocolRequest(action={self.action!r}, "
            f"request_id={self.request_id!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProtocolRequest):
            return NotImplemented
        return self.request_id == other.request_id


@dataclass
class ProtocolResponse:
    """
    A structured response in the ontology protocol.

    Attributes
    ----------
    success : bool
        Whether the request was handled successfully.
    data : any
        Response payload (e.g. the result of a query, the modified
        ontology, or an error detail).
    error : str | None
        Human-readable error message if ``success`` is ``False``.
    request_id : str
        Matches the :attr:`ProtocolRequest.request_id` this is a response to.
    """

    success: bool = False
    data: Any = None
    error: str | None = None
    request_id: str = ""

    def __repr__(self) -> str:
        status = "success" if self.success else "error"
        return (
            f"ProtocolResponse({status}, "
            f"request_id={self.request_id!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProtocolResponse):
            return NotImplemented
        return (
            self.success == other.success
            and self.request_id == other.request_id
        )


# ---------------------------------------------------------------------------
# OntologyProtocol ABC
# ---------------------------------------------------------------------------


class OntologyProtocol(ABC):
    """
    Abstract base class for ontology protocol handlers.

    Subclasses implement :meth:`handle` to process incoming
    :class:`ProtocolRequest` instances and return
    :class:`ProtocolResponse` instances.

    The protocol separates the transport layer from the business logic —
    concrete implementations can be backed by HTTP, WebSocket, in-process
    calls, message queues, etc.
    """

    @abstractmethod
    def handle(self, request: ProtocolRequest) -> ProtocolResponse:
        """
        Process a single protocol request and return a response.

        Parameters
        ----------
        request : ProtocolRequest
            The incoming request to handle.

        Returns
        -------
        ProtocolResponse
            The response to send back.
        """
        ...


# ---------------------------------------------------------------------------
# ProtocolAdapter
# ---------------------------------------------------------------------------


class ProtocolAdapter(OntologyProtocol):
    """
    Adapter that translates :class:`ProtocolRequest` objects into
    :class:`~ontology_runtime.ontology_model.Ontology` method calls and
    operation dispatches.

    Provides a standard set of actions covering CRUD operations on all
    ontology entity types, transaction support, and ontology-level queries.

    Attributes
    ----------
    ontology : Ontology
        The ontology instance this adapter operates on.
    """

    # ── Action dispatch table ──────────────────────────────────────────
    _ACTION_HANDLERS: dict[str, str] | None = None

    def __init__(self, ontology: Ontology) -> None:
        """
        Initialise the adapter with an ontology instance.

        Parameters
        ----------
        ontology : Ontology
            The ontology to operate on.
        """
        self.ontology = ontology

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def handle(self, request: ProtocolRequest) -> ProtocolResponse:
        """
        Dispatch a request to the appropriate action handler.

        Parameters
        ----------
        request : ProtocolRequest
            The incoming request.

        Returns
        -------
        ProtocolResponse
            The result of handling the request.
        """
        handler_name = f"_action__{request.action}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            return ProtocolResponse(
                success=False,
                error=f"Unknown action: {request.action}",
                request_id=request.request_id,
            )

        try:
            result = handler(**request.params)
            return ProtocolResponse(
                success=True,
                data=result,
                request_id=request.request_id,
            )
        except Exception as exc:
            return ProtocolResponse(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                request_id=request.request_id,
            )

    # ── Query actions --------------------------------------------------

    def _action__get_object_type(self, object_type_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single object type by its identifier.

        Parameters
        ----------
        object_type_id : str
            The object-type identifier.

        Returns
        -------
        dict | None
            Serialised object type, or ``None`` if not found.
        """
        obj = self.ontology.objects.get(object_type_id)
        return obj.to_dict() if obj else None

    def _action__list_object_types(self) -> list[dict[str, Any]]:
        """Return all object types as a list of serialised dicts."""
        return [o.to_dict() for o in self.ontology.objects.values()]

    def _action__get_relation_type(self, relation_type_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single relation type by its identifier.

        Parameters
        ----------
        relation_type_id : str
            The relation-type identifier.

        Returns
        -------
        dict | None
            Serialised relation type, or ``None`` if not found.
        """
        rel = self.ontology.relations.get(relation_type_id)
        return rel.to_dict() if rel else None

    def _action__list_relation_types(self) -> list[dict[str, Any]]:
        """Return all relation types as a list of serialised dicts."""
        return [r.to_dict() for r in self.ontology.relations.values()]

    def _action__get_property(self, property_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single property by its identifier.

        Parameters
        ----------
        property_id : str
            The property identifier.

        Returns
        -------
        dict | None
            Serialised property, or ``None`` if not found.
        """
        prop = self.ontology.properties.get(property_id)
        return prop.to_dict() if prop else None

    def _action__list_properties(self) -> list[dict[str, Any]]:
        """Return all properties as a list of serialised dicts."""
        return [p.to_dict() for p in self.ontology.properties.values()]

    def _action__get_constraint(self, constraint_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single constraint by its identifier.

        Parameters
        ----------
        constraint_id : str
            The constraint identifier.

        Returns
        -------
        dict | None
            Serialised constraint, or ``None`` if not found.
        """
        c = self.ontology.constraints.get(constraint_id)
        return c.to_dict() if c else None

    def _action__list_constraints(self) -> list[dict[str, Any]]:
        """Return all constraints as a list of serialised dicts."""
        return [c.to_dict() for c in self.ontology.constraints.values()]

    def _action__get_rule(self, rule_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single rule by its identifier.

        Parameters
        ----------
        rule_id : str
            The rule identifier.

        Returns
        -------
        dict | None
            Serialised rule, or ``None`` if not found.
        """
        r = self.ontology.rules.get(rule_id)
        return r.to_dict() if r else None

    def _action__list_rules(self) -> list[dict[str, Any]]:
        """Return all rules as a list of serialised dicts."""
        return [r.to_dict() for r in self.ontology.rules.values()]

    def _action__get_action(self, action_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single action by its identifier.

        Parameters
        ----------
        action_id : str
            The action identifier.

        Returns
        -------
        dict | None
            Serialised action, or ``None`` if not found.
        """
        a = self.ontology.actions.get(action_id)
        return a.to_dict() if a else None

    def _action__list_actions(self) -> list[dict[str, Any]]:
        """Return all actions as a list of serialised dicts."""
        return [a.to_dict() for a in self.ontology.actions.values()]

    def _action__get_workflow(self, workflow_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single workflow by its identifier.

        Parameters
        ----------
        workflow_id : str
            The workflow identifier.

        Returns
        -------
        dict | None
            Serialised workflow, or ``None`` if not found.
        """
        wf = self.ontology.workflows.get(workflow_id)
        return wf.to_dict() if wf else None

    def _action__list_workflows(self) -> list[dict[str, Any]]:
        """Return all workflows as a list of serialised dicts."""
        return [w.to_dict() for w in self.ontology.workflows.values()]

    def _action__get_view(self, view_id: str) -> dict[str, Any] | None:
        """
        Retrieve a single view by its identifier.

        Parameters
        ----------
        view_id : str
            The view identifier.

        Returns
        -------
        dict | None
            Serialised view, or ``None`` if not found.
        """
        v = self.ontology.views.get(view_id)
        return v.to_dict() if v else None

    def _action__list_views(self) -> list[dict[str, Any]]:
        """Return all views as a list of serialised dicts."""
        return [v.to_dict() for v in self.ontology.views.values()]

    def _action__get_ontology(self) -> dict[str, Any]:
        """
        Return the full ontology as a serialised dictionary.

        Returns
        -------
        dict
            The complete ontology serialisation.
        """
        return self.ontology.to_dict()

    # ── Mutation actions via operations ───────────────────────────────

    def _action__apply_operation(self, **payload: Any) -> dict[str, Any]:
        """
        Apply a single :class:`OntologyOperation` to the ontology.

        Parameters
        ----------
        **payload
            Must include ``type`` (str), ``target_id`` (str), and
            optionally ``payload`` (dict).

        Returns
        -------
        dict
            Status of the operation.
        """
        from ontology_runtime.ontology_operations import OperationType

        op_type_str = payload.pop("type", "")
        try:
            op_type = OperationType(op_type_str)
        except ValueError:
            raise ValueError(f"Invalid operation type: {op_type_str}") from None

        operation = OntologyOperation(
            type=op_type,
            target_id=payload.pop("target_id", ""),
            payload=payload.pop("payload", {}),
            description=payload.pop("description", ""),
        )
        apply_operation(self.ontology, operation)
        return {"status": "applied", "target_id": operation.target_id}

    def _action__commit_transaction(self, **payload: Any) -> dict[str, Any]:
        """
        Build and commit an :class:`OntologyTransaction`.

        Parameters
        ----------
        **payload
            Must include ``operations`` (list of dicts), each with
            ``type``, ``target_id``, and optionally ``payload``.

        Returns
        -------
        dict
            Summary of the committed transaction.
        """
        from ontology_runtime.ontology_operations import OperationType

        tx = OntologyTransaction()
        for op_data in payload.get("operations", []):
            op_type = OperationType(op_data["type"])
            operation = OntologyOperation(
                type=op_type,
                target_id=op_data.get("target_id", ""),
                payload=op_data.get("payload", {}),
                description=op_data.get("description", ""),
            )
            tx.add_operation(operation)

        applied = tx.commit(self.ontology)
        return {
            "transaction_id": tx.id,
            "operations_applied": len(applied),
            "applied_operation_ids": [
                op.target_id for op in applied
            ],
        }


__all__ = [
    "OntologyProtocol",
    "ProtocolAdapter",
    "ProtocolRequest",
    "ProtocolResponse",
]
