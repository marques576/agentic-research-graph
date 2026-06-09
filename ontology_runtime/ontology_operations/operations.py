"""
Atomic operations and transactions for modifying an Ontology.

Provides :class:`OperationType` (enum of all mutation kinds),
:class:`OntologyOperation` (a single atomic change descriptor),
:class:`OntologyTransaction` (groups operations with commit/rollback),
and :func:`apply_operation` which dispatches operations to type-specific
handlers.
"""

from __future__ import annotations

import copy
import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ontology_runtime.ontology_model import (
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


# ---------------------------------------------------------------------------
# OperationType
# ---------------------------------------------------------------------------

class OperationType(enum.Enum):
    """
    Enumeration of all supported ontology operation types.

    Each value represents a distinct mutation that can be applied to
    an :class:`~ontology_runtime.ontology_model.Ontology`.
    """

    # Object type operations
    ADD_OBJECT_TYPE = "add_object_type"
    REMOVE_OBJECT_TYPE = "remove_object_type"
    RENAME_OBJECT_TYPE = "rename_object_type"
    MERGE_OBJECT_TYPES = "merge_object_types"
    SPLIT_OBJECT_TYPE = "split_object_type"

    # Relation type operations
    ADD_RELATION_TYPE = "add_relation_type"
    REMOVE_RELATION_TYPE = "remove_relation_type"
    RENAME_RELATION_TYPE = "rename_relation_type"
    MERGE_RELATION_TYPES = "merge_relation_types"
    SPLIT_RELATION_TYPE = "split_relation_type"

    # Property operations
    ADD_PROPERTY = "add_property"
    REMOVE_PROPERTY = "remove_property"
    RENAME_PROPERTY = "rename_property"
    MERGE_PROPERTIES = "merge_properties"
    SPLIT_PROPERTY = "split_property"

    # Constraint operations
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"

    # Rule operations
    ADD_RULE = "add_rule"
    REMOVE_RULE = "remove_rule"

    # Action operations
    ADD_ACTION = "add_action"
    REMOVE_ACTION = "remove_action"

    # Workflow operations
    ADD_WORKFLOW = "add_workflow"
    REMOVE_WORKFLOW = "remove_workflow"

    # View operations
    ADD_VIEW = "add_view"
    REMOVE_VIEW = "remove_view"

    # Ontology-level operations
    RENAME_ONTOLOGY = "rename_ontology"
    UPDATE_METADATA = "update_metadata"


# ---------------------------------------------------------------------------
# OntologyOperation
# ---------------------------------------------------------------------------

@dataclass
class OntologyOperation:
    """
    A single atomic operation on an ontology.

    Attributes
    ----------
    type : OperationType
        The kind of operation.
    target_id : str
        Identifier of the target entity being acted upon.
    payload : dict
        Operation-specific data (e.g. new attribute values, creation data).
    timestamp : datetime
        When the operation was created (default: UTC now).
    description : str
        Optional human-readable description of the operation.
    """

    type: OperationType
    target_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"OntologyOperation(type={self.type.value}, "
            f"target_id={self.target_id!r}, "
            f"timestamp={self.timestamp.isoformat()})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OntologyOperation):
            return NotImplemented
        return (
            self.type == other.type
            and self.target_id == other.target_id
            and self.payload == other.payload
            and self.timestamp == other.timestamp
        )


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

@dataclass
class OntologyTransaction:
    """
    A group of operations that can be committed or rolled back as a unit.

    Tracks which operations have been applied so that :meth:`rollback`
    can revert them in reverse order.

    Attributes
    ----------
    id : str
        Unique transaction identifier (default: random UUID hex).
    operations : list[OntologyOperation]
        The ordered list of operations in this transaction.
    applied_indices : list[int]
        Indices of operations that have been successfully applied
        during the last :meth:`commit` call.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    operations: list[OntologyOperation] = field(default_factory=list)
    applied_indices: list[int] = field(default_factory=list)

    def add_operation(self, operation: OntologyOperation) -> None:
        """
        Append an operation to the transaction.

        Parameters
        ----------
        operation : OntologyOperation
            The operation to add.
        """
        self.operations.append(operation)

    def commit(self, ontology: Ontology) -> list[OntologyOperation]:
        """
        Apply all operations to the given ontology in order.

        Each operation is applied via :func:`apply_operation`.  If an
        operation fails, previously applied operations are *not*
        automatically rolled back (call :meth:`rollback` to revert).

        Parameters
        ----------
        ontology : Ontology
            The ontology to mutate in-place.

        Returns
        -------
        list[OntologyOperation]
            The list of successfully applied operations.
        """
        self.applied_indices.clear()
        for i, op in enumerate(self.operations):
            apply_operation(ontology, op)
            self.applied_indices.append(i)
        return [self.operations[i] for i in self.applied_indices]

    def rollback(self, ontology: Ontology) -> None:
        """
        Revert all applied operations in reverse order.

        Builds inverse operations for each applied operation and
        applies them.  Clears the applied-indices tracker afterwards.

        Parameters
        ----------
        ontology : Ontology
            The ontology to mutate in-place.
        """
        # Walk applied operations in reverse
        for i in reversed(self.applied_indices):
            op = self.operations[i]
            inverse = _invert_operation(ontology, op)
            if inverse is not None:
                apply_operation(ontology, inverse)
        self.applied_indices.clear()

    def __repr__(self) -> str:
        return (
            f"OntologyTransaction(id={self.id!r}, "
            f"operations={len(self.operations)}, "
            f"applied={len(self.applied_indices)})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OntologyTransaction):
            return NotImplemented
        return (
            self.id == other.id
            and self.operations == other.operations
        )


# ---------------------------------------------------------------------------
# Helpers  (used internally by rollback)
# ---------------------------------------------------------------------------

def _snapshot_key(ontology: Ontology, target_id: str) -> Any | None:
    """Return the object snapshot for an id across all ontology dicts."""
    for collection in (
        ontology.objects,
        ontology.relations,
        ontology.properties,
        ontology.constraints,
        ontology.rules,
        ontology.actions,
        ontology.workflows,
        ontology.views,
    ):
        if target_id in collection:
            return copy.deepcopy(collection[target_id])
    return None


def _invert_operation(
    ontology: Ontology, op: OntologyOperation
) -> OntologyOperation | None:
    """
    Build the inverse of an operation for rollback purposes.

    Parameters
    ----------
    ontology : Ontology
        Current ontology state (used to capture the entity being removed).
    op : OntologyOperation
        The operation to invert.

    Returns
    -------
    OntologyOperation | None
        The inverse operation, or ``None`` if no inversion is needed
        (e.g. for idempotent operations).
    """
    # ── Inverses for ADD operations → REMOVE ──────────────────────────
    if op.type == OperationType.ADD_OBJECT_TYPE:
        return OntologyOperation(
            type=OperationType.REMOVE_OBJECT_TYPE,
            target_id=op.target_id,
            description=f"Rollback: remove object type {op.target_id}",
        )
    if op.type == OperationType.ADD_RELATION_TYPE:
        return OntologyOperation(
            type=OperationType.REMOVE_RELATION_TYPE,
            target_id=op.target_id,
            description=f"Rollback: remove relation type {op.target_id}",
        )
    if op.type == OperationType.ADD_PROPERTY:
        return OntologyOperation(
            type=OperationType.REMOVE_PROPERTY,
            target_id=op.target_id,
            description=f"Rollback: remove property {op.target_id}",
        )
    if op.type == OperationType.ADD_CONSTRAINT:
        return OntologyOperation(
            type=OperationType.REMOVE_CONSTRAINT,
            target_id=op.target_id,
            description=f"Rollback: remove constraint {op.target_id}",
        )
    if op.type == OperationType.ADD_RULE:
        return OntologyOperation(
            type=OperationType.REMOVE_RULE,
            target_id=op.target_id,
            description=f"Rollback: remove rule {op.target_id}",
        )
    if op.type == OperationType.ADD_ACTION:
        return OntologyOperation(
            type=OperationType.REMOVE_ACTION,
            target_id=op.target_id,
            description=f"Rollback: remove action {op.target_id}",
        )
    if op.type == OperationType.ADD_WORKFLOW:
        return OntologyOperation(
            type=OperationType.REMOVE_WORKFLOW,
            target_id=op.target_id,
            description=f"Rollback: remove workflow {op.target_id}",
        )
    if op.type == OperationType.ADD_VIEW:
        return OntologyOperation(
            type=OperationType.REMOVE_VIEW,
            target_id=op.target_id,
            description=f"Rollback: remove view {op.target_id}",
        )

    # ── Inverses for REMOVE operations → ADD (needs payload from snapshot) ──
    if op.type == OperationType.REMOVE_OBJECT_TYPE:
        snapshot = _snapshot_key(ontology, op.target_id)
        return OntologyOperation(
            type=OperationType.ADD_OBJECT_TYPE,
            target_id=op.target_id,
            payload={"object_type": snapshot} if snapshot else {},
            description=f"Rollback: restore object type {op.target_id}",
        )
    if op.type == OperationType.REMOVE_RELATION_TYPE:
        snapshot = _snapshot_key(ontology, op.target_id)
        return OntologyOperation(
            type=OperationType.ADD_RELATION_TYPE,
            target_id=op.target_id,
            payload={"relation_type": snapshot} if snapshot else {},
            description=f"Rollback: restore relation type {op.target_id}",
        )
    if op.type == OperationType.REMOVE_PROPERTY:
        snapshot = _snapshot_key(ontology, op.target_id)
        return OntologyOperation(
            type=OperationType.ADD_PROPERTY,
            target_id=op.target_id,
            payload={"property": snapshot} if snapshot else {},
            description=f"Rollback: restore property {op.target_id}",
        )
    if op.type == OperationType.REMOVE_CONSTRAINT:
        snapshot = _snapshot_key(ontology, op.target_id)
        return OntologyOperation(
            type=OperationType.ADD_CONSTRAINT,
            target_id=op.target_id,
            payload={"constraint": snapshot} if snapshot else {},
            description=f"Rollback: restore constraint {op.target_id}",
        )
    if op.type == OperationType.REMOVE_RULE:
        snapshot = _snapshot_key(ontology, op.target_id)
        return OntologyOperation(
            type=OperationType.ADD_RULE,
            target_id=op.target_id,
            payload={"rule": snapshot} if snapshot else {},
            description=f"Rollback: restore rule {op.target_id}",
        )
    if op.type == OperationType.REMOVE_ACTION:
        snapshot = _snapshot_key(ontology, op.target_id)
        return OntologyOperation(
            type=OperationType.ADD_ACTION,
            target_id=op.target_id,
            payload={"action": snapshot} if snapshot else {},
            description=f"Rollback: restore action {op.target_id}",
        )
    if op.type == OperationType.REMOVE_WORKFLOW:
        snapshot = _snapshot_key(ontology, op.target_id)
        return OntologyOperation(
            type=OperationType.ADD_WORKFLOW,
            target_id=op.target_id,
            payload={"workflow": snapshot} if snapshot else {},
            description=f"Rollback: restore workflow {op.target_id}",
        )
    if op.type == OperationType.REMOVE_VIEW:
        snapshot = _snapshot_key(ontology, op.target_id)
        return OntologyOperation(
            type=OperationType.ADD_VIEW,
            target_id=op.target_id,
            payload={"view": snapshot} if snapshot else {},
            description=f"Rollback: restore view {op.target_id}",
        )

    # ── RENAME / MERGE / SPLIT / UPDATE → capture old value from payload ──
    if op.type == OperationType.RENAME_OBJECT_TYPE:
        old_name = op.payload.get("old_name", "")
        return OntologyOperation(
            type=OperationType.RENAME_OBJECT_TYPE,
            target_id=op.target_id,
            payload={"name": old_name, "old_name": op.payload.get("name", "")},
            description=f"Rollback: rename object type back to {old_name}",
        )
    if op.type == OperationType.RENAME_RELATION_TYPE:
        old_name = op.payload.get("old_name", "")
        return OntologyOperation(
            type=OperationType.RENAME_RELATION_TYPE,
            target_id=op.target_id,
            payload={"name": old_name, "old_name": op.payload.get("name", "")},
            description=f"Rollback: rename relation type back to {old_name}",
        )
    if op.type == OperationType.RENAME_PROPERTY:
        old_name = op.payload.get("old_name", "")
        return OntologyOperation(
            type=OperationType.RENAME_PROPERTY,
            target_id=op.target_id,
            payload={"name": old_name, "old_name": op.payload.get("name", "")},
            description=f"Rollback: rename property back to {old_name}",
        )
    if op.type == OperationType.MERGE_OBJECT_TYPES:
        # Inverting a merge is not perfectly reconstructible — no-op for safety
        return None
    if op.type == OperationType.SPLIT_OBJECT_TYPE:
        return None
    if op.type == OperationType.MERGE_RELATION_TYPES:
        return None
    if op.type == OperationType.SPLIT_RELATION_TYPE:
        return None
    if op.type == OperationType.MERGE_PROPERTIES:
        return None
    if op.type == OperationType.SPLIT_PROPERTY:
        return None
    if op.type == OperationType.UPDATE_METADATA:
        old_metadata = op.payload.get("old_metadata", {})
        return OntologyOperation(
            type=OperationType.UPDATE_METADATA,
            target_id=op.target_id,
            payload={"metadata": old_metadata, "old_metadata": {}},
            description="Rollback: restore previous metadata",
        )
    if op.type == OperationType.RENAME_ONTOLOGY:
        old_name = op.payload.get("old_name", "")
        return OntologyOperation(
            type=OperationType.RENAME_ONTOLOGY,
            target_id=op.target_id,
            payload={"name": old_name, "old_name": op.payload.get("name", "")},
            description=f"Rollback: rename ontology back to {old_name}",
        )

    return None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_HANDLERS: dict[OperationType, callable] = {}


def _register(op_type: OperationType) -> callable:
    """Decorator to register a handler function for an OperationType."""

    def wrapper(func: callable) -> callable:
        _HANDLERS[op_type] = func
        return func

    return wrapper


def apply_operation(ontology: Ontology, operation: OntologyOperation) -> None:
    """
    Apply a single operation to an ontology in-place.

    Dispatches to the registered handler for ``operation.type``.

    Parameters
    ----------
    ontology : Ontology
        The ontology to mutate.
    operation : OntologyOperation
        The operation to apply.

    Raises
    ------
    ValueError
        If the operation type is unknown or no handler is registered.
    KeyError
        If the target entity does not exist (for remove / rename / etc.).
    TypeError
        If the payload is malformed for the operation type.
    """
    handler = _HANDLERS.get(operation.type)
    if handler is None:
        raise ValueError(f"No handler registered for operation type: {operation.type}")
    handler(ontology, operation)


# ===================================================================
# Handlers
# ===================================================================


@_register(OperationType.ADD_OBJECT_TYPE)
def _handle_add_object_type(ontology: Ontology, op: OntologyOperation) -> None:
    """Add a new object type from ``op.payload``."""
    data = op.payload.get("object_type", op.payload)
    if isinstance(data, ObjectType):
        obj_type = data
    else:
        obj_type = ObjectType.from_dict(data)
    if op.target_id:
        obj_type.id = op.target_id
    ontology.objects[obj_type.id] = obj_type


@_register(OperationType.REMOVE_OBJECT_TYPE)
def _handle_remove_object_type(ontology: Ontology, op: OntologyOperation) -> None:
    """Remove an object type by ``op.target_id``."""
    if op.target_id not in ontology.objects:
        raise KeyError(f"Object type not found: {op.target_id}")
    del ontology.objects[op.target_id]


@_register(OperationType.RENAME_OBJECT_TYPE)
def _handle_rename_object_type(ontology: Ontology, op: OntologyOperation) -> None:
    """Rename an object type's ``name`` field."""
    obj = ontology.objects.get(op.target_id)
    if obj is None:
        raise KeyError(f"Object type not found: {op.target_id}")
    new_name = op.payload.get("name")
    if new_name is None:
        raise TypeError("RENAME_OBJECT_TYPE requires payload 'name'")
    op.payload["old_name"] = obj.name
    obj.name = new_name


@_register(OperationType.MERGE_OBJECT_TYPES)
def _handle_merge_object_types(ontology: Ontology, op: OntologyOperation) -> None:
    """Merge source object types into a target type, then remove the sources."""
    target_id = op.payload.get("target_id", op.target_id)
    source_ids = op.payload.get("source_ids", [])
    keep_props = op.payload.get("keep_properties", True)

    target = ontology.objects.get(target_id)
    if target is None:
        raise KeyError(f"Target object type not found: {target_id}")

    for sid in source_ids:
        source = ontology.objects.pop(sid, None)
        if source is None:
            continue
        if keep_props:
            existing_names = {p.name for p in target.properties}
            for prop in source.properties:
                if prop.name not in existing_names:
                    target.properties.append(prop)
                    existing_names.add(prop.name)


@_register(OperationType.SPLIT_OBJECT_TYPE)
def _handle_split_object_type(ontology: Ontology, op: OntologyOperation) -> None:
    """Split an object type into multiple new types."""
    source = ontology.objects.get(op.target_id)
    if source is None:
        raise KeyError(f"Object type not found: {op.target_id}")
    new_type_configs = op.payload.get("new_types", [])
    for cfg in new_type_configs:
        new_obj = ObjectType(
            name=cfg.get("name", ""),
            description=cfg.get("description", source.description),
            properties=[
                p for p in source.properties if p.name in cfg.get("property_names", [])
            ],
            metadata=dict(cfg.get("metadata", {})),
            parent_id=cfg.get("parent_id", source.id),
        )
        ontology.objects[new_obj.id] = new_obj
    # Optionally remove the original
    if op.payload.get("remove_original", True):
        del ontology.objects[op.target_id]


# ── Relation type handlers ──────────────────────────────────────────────


@_register(OperationType.ADD_RELATION_TYPE)
def _handle_add_relation_type(ontology: Ontology, op: OntologyOperation) -> None:
    """Add a new relation type from ``op.payload``."""
    data = op.payload.get("relation_type", op.payload)
    if isinstance(data, RelationType):
        rel_type = data
    else:
        rel_type = RelationType.from_dict(data)
    if op.target_id:
        rel_type.id = op.target_id
    ontology.relations[rel_type.id] = rel_type


@_register(OperationType.REMOVE_RELATION_TYPE)
def _handle_remove_relation_type(ontology: Ontology, op: OntologyOperation) -> None:
    """Remove a relation type by ``op.target_id``."""
    if op.target_id not in ontology.relations:
        raise KeyError(f"Relation type not found: {op.target_id}")
    del ontology.relations[op.target_id]


@_register(OperationType.RENAME_RELATION_TYPE)
def _handle_rename_relation_type(ontology: Ontology, op: OntologyOperation) -> None:
    """Rename a relation type's ``name`` field."""
    rel = ontology.relations.get(op.target_id)
    if rel is None:
        raise KeyError(f"Relation type not found: {op.target_id}")
    new_name = op.payload.get("name")
    if new_name is None:
        raise TypeError("RENAME_RELATION_TYPE requires payload 'name'")
    op.payload["old_name"] = rel.name
    rel.name = new_name


@_register(OperationType.MERGE_RELATION_TYPES)
def _handle_merge_relation_types(ontology: Ontology, op: OntologyOperation) -> None:
    """Merge multiple relation types into one."""
    target_id = op.payload.get("target_id", op.target_id)
    source_ids = op.payload.get("source_ids", [])
    for sid in source_ids:
        ontology.relations.pop(sid, None)


@_register(OperationType.SPLIT_RELATION_TYPE)
def _handle_split_relation_type(ontology: Ontology, op: OntologyOperation) -> None:
    """Split a relation type into multiple new types."""
    source = ontology.relations.get(op.target_id)
    if source is None:
        raise KeyError(f"Relation type not found: {op.target_id}")
    new_type_configs = op.payload.get("new_types", [])
    for cfg in new_type_configs:
        new_rel = RelationType(
            name=cfg.get("name", ""),
            source_type=cfg.get("source_type", source.source_type),
            target_type=cfg.get("target_type", source.target_type),
            properties=list(source.properties),
            metadata=dict(cfg.get("metadata", {})),
        )
        ontology.relations[new_rel.id] = new_rel
    if op.payload.get("remove_original", True):
        del ontology.relations[op.target_id]


# ── Property handlers ───────────────────────────────────────────────────


@_register(OperationType.ADD_PROPERTY)
def _handle_add_property(ontology: Ontology, op: OntologyOperation) -> None:
    """Add a new property type from ``op.payload``."""
    data = op.payload.get("property", op.payload)
    if isinstance(data, PropertyType):
        prop = data
    else:
        prop = PropertyType.from_dict(data)
    if op.target_id:
        prop.id = op.target_id
    ontology.properties[prop.id] = prop


@_register(OperationType.REMOVE_PROPERTY)
def _handle_remove_property(ontology: Ontology, op: OntologyOperation) -> None:
    """Remove a property type by ``op.target_id``."""
    if op.target_id not in ontology.properties:
        raise KeyError(f"Property not found: {op.target_id}")
    del ontology.properties[op.target_id]


@_register(OperationType.RENAME_PROPERTY)
def _handle_rename_property(ontology: Ontology, op: OntologyOperation) -> None:
    """Rename a property's ``name`` field."""
    prop = ontology.properties.get(op.target_id)
    if prop is None:
        raise KeyError(f"Property not found: {op.target_id}")
    new_name = op.payload.get("name")
    if new_name is None:
        raise TypeError("RENAME_PROPERTY requires payload 'name'")
    op.payload["old_name"] = prop.name
    prop.name = new_name


@_register(OperationType.MERGE_PROPERTIES)
def _handle_merge_properties(ontology: Ontology, op: OntologyOperation) -> None:
    """Merge multiple properties.  Keeps the first and removes the rest."""
    target_id = op.payload.get("target_id", op.target_id)
    source_ids = op.payload.get("source_ids", [])
    target_prop = ontology.properties.get(target_id)
    if target_prop is None:
        raise KeyError(f"Target property not found: {target_id}")
    for sid in source_ids:
        source = ontology.properties.pop(sid, None)
        if source is not None and source.default is not None:
            target_prop.default = source.default


@_register(OperationType.SPLIT_PROPERTY)
def _handle_split_property(ontology: Ontology, op: OntologyOperation) -> None:
    """Split a property into multiple new properties."""
    source = ontology.properties.get(op.target_id)
    if source is None:
        raise KeyError(f"Property not found: {op.target_id}")
    new_configs = op.payload.get("new_properties", [])
    for cfg in new_configs:
        new_prop = PropertyType(
            name=cfg.get("name", source.name),
            datatype=cfg.get("datatype", source.datatype),
            constraints=list(source.constraints),
            default=cfg.get("default", source.default),
            required=cfg.get("required", source.required),
            description=cfg.get("description", source.description),
        )
        ontology.properties[new_prop.id] = new_prop
    if op.payload.get("remove_original", True):
        del ontology.properties[op.target_id]


# ── Constraint handlers ─────────────────────────────────────────────────


@_register(OperationType.ADD_CONSTRAINT)
def _handle_add_constraint(ontology: Ontology, op: OntologyOperation) -> None:
    """Add a new constraint from ``op.payload``."""
    data = op.payload.get("constraint", op.payload)
    if isinstance(data, Constraint):
        constraint = data
    else:
        constraint = Constraint.from_dict(data)
    if op.target_id:
        constraint.id = op.target_id
    ontology.constraints[constraint.id] = constraint


@_register(OperationType.REMOVE_CONSTRAINT)
def _handle_remove_constraint(ontology: Ontology, op: OntologyOperation) -> None:
    """Remove a constraint by ``op.target_id``."""
    if op.target_id not in ontology.constraints:
        raise KeyError(f"Constraint not found: {op.target_id}")
    del ontology.constraints[op.target_id]


# ── Rule handlers ───────────────────────────────────────────────────────


@_register(OperationType.ADD_RULE)
def _handle_add_rule(ontology: Ontology, op: OntologyOperation) -> None:
    """Add a new rule from ``op.payload``."""
    data = op.payload.get("rule", op.payload)
    if isinstance(data, Rule):
        rule = data
    else:
        rule = Rule.from_dict(data)
    if op.target_id:
        rule.id = op.target_id
    ontology.rules[rule.id] = rule


@_register(OperationType.REMOVE_RULE)
def _handle_remove_rule(ontology: Ontology, op: OntologyOperation) -> None:
    """Remove a rule by ``op.target_id``."""
    if op.target_id not in ontology.rules:
        raise KeyError(f"Rule not found: {op.target_id}")
    del ontology.rules[op.target_id]


# ── Action handlers ─────────────────────────────────────────────────────


@_register(OperationType.ADD_ACTION)
def _handle_add_action(ontology: Ontology, op: OntologyOperation) -> None:
    """Add a new action from ``op.payload``."""
    data = op.payload.get("action", op.payload)
    if isinstance(data, Action):
        action = data
    else:
        action = Action.from_dict(data)
    if op.target_id:
        action.id = op.target_id
    ontology.actions[action.id] = action


@_register(OperationType.REMOVE_ACTION)
def _handle_remove_action(ontology: Ontology, op: OntologyOperation) -> None:
    """Remove an action by ``op.target_id``."""
    if op.target_id not in ontology.actions:
        raise KeyError(f"Action not found: {op.target_id}")
    del ontology.actions[op.target_id]


# ── Workflow handlers ───────────────────────────────────────────────────


@_register(OperationType.ADD_WORKFLOW)
def _handle_add_workflow(ontology: Ontology, op: OntologyOperation) -> None:
    """Add a new workflow from ``op.payload``."""
    data = op.payload.get("workflow", op.payload)
    if isinstance(data, Workflow):
        wf = data
    else:
        wf = Workflow.from_dict(data)
    if op.target_id:
        wf.id = op.target_id
    ontology.workflows[wf.id] = wf


@_register(OperationType.REMOVE_WORKFLOW)
def _handle_remove_workflow(ontology: Ontology, op: OntologyOperation) -> None:
    """Remove a workflow by ``op.target_id``."""
    if op.target_id not in ontology.workflows:
        raise KeyError(f"Workflow not found: {op.target_id}")
    del ontology.workflows[op.target_id]


# ── View handlers ───────────────────────────────────────────────────────


@_register(OperationType.ADD_VIEW)
def _handle_add_view(ontology: Ontology, op: OntologyOperation) -> None:
    """Add a new view from ``op.payload``."""
    data = op.payload.get("view", op.payload)
    if isinstance(data, View):
        view = data
    else:
        view = View.from_dict(data)
    if op.target_id:
        view.id = op.target_id
    ontology.views[view.id] = view


@_register(OperationType.REMOVE_VIEW)
def _handle_remove_view(ontology: Ontology, op: OntologyOperation) -> None:
    """Remove a view by ``op.target_id``."""
    if op.target_id not in ontology.views:
        raise KeyError(f"View not found: {op.target_id}")
    del ontology.views[op.target_id]


# ── Ontology-level handlers ─────────────────────────────────────────────


@_register(OperationType.RENAME_ONTOLOGY)
def _handle_rename_ontology(ontology: Ontology, op: OntologyOperation) -> None:
    """Rename the ontology's ``name`` field."""
    new_name = op.payload.get("name")
    if new_name is None:
        raise TypeError("RENAME_ONTOLOGY requires payload 'name'")
    op.payload["old_name"] = ontology.name
    ontology.name = new_name


@_register(OperationType.UPDATE_METADATA)
def _handle_update_metadata(ontology: Ontology, op: OntologyOperation) -> None:
    """Update the ontology's metadata with key-value pairs."""
    meta = op.payload.get("metadata", {})
    op.payload["old_metadata"] = dict(ontology.metadata)
    ontology.metadata.update(meta)


__all__ = [
    "OperationType",
    "OntologyOperation",
    "OntologyTransaction",
    "apply_operation",
]
