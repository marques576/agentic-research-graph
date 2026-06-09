"""OntologyRuntime – the central facade for interacting with the ontology platform.

All mutations are proposal-driven (they create ``OntologyProposal`` instances
rather than directly modifying the ontology).  Queries, validation, reasoning,
and analysis are delegates to their respective sub-engines from the existing
ontology_runtime ecosystem.
"""

from __future__ import annotations

import time
from typing import Any

from ontology_runtime.ontology_model import Ontology, ObjectType, RelationType
from ontology_runtime.ontology_events import EventBus, EventType, OntologyEvent
from ontology_runtime.ontology_operations import (
    OperationType,
    OntologyOperation,
)
from ontology_runtime.ontology_validation import (
    ValidationEngine,
    ValidationReport,
)
from ontology_runtime.ontology_reasoning import (
    InferenceReport,
    ReasoningFramework,
    RuleBasedReasoner,
    GraphTraversalReasoner,
)
from ontology_runtime.ontology_refactoring import (
    RefactoringEngine,
    RefactoringReport,
)
from ontology_runtime.ontology_runtime.query import OntologyQuery, QueryResult
from ontology_runtime.ontology_versioning.versioning import (
    SemanticChange,
    SemanticChangeType,
    SemanticDiff,
    OntologyVersion,
    VersionManager,
)
from ontology_runtime.ontology_workspaces.workspaces import (
    MergeResult,
    Workspace,
    WorkspaceManager,
)
from ontology_runtime.ontology_proposals.proposals import (
    OntologyProposal,
    ProposalManager,
    ProposalStatus,
)


# ---------------------------------------------------------------------------
# OntologyRuntime
# ---------------------------------------------------------------------------


class OntologyRuntime:
    """Central facade that agents and users interact with.

    The runtime wraps an ``Ontology`` schema, an instance store (optional),
    and the versioning / workspace / proposal subsystems into a single API.

    **All mutations are proposal-driven.**  ``create_object``, ``update_object``,
    etc. return an ``OntologyProposal`` rather than directly modifying the
    ontology.  The caller must then shepherd the proposal through the
    lifecycle (validate → approve → apply) for changes to take effect.

    Parameters
    ----------
    ontology : Ontology
        The ontology schema instance.
    event_bus : EventBus | None
        Optional event bus for decoupled notifications.  Created if not given.
    version_manager : VersionManager | None
        Optional version manager.  Created if not given.
    workspace_manager : WorkspaceManager | None
        Optional workspace manager.  Created if not given.
    proposal_manager : ProposalManager | None
        Optional proposal manager.  Created if not given.
    instance_store : Any
        Optional instance/graph store (e.g. ``KnowledgeGraph``) for
        entity-level queries.
    """

    def __init__(
        self,
        ontology: Ontology,
        event_bus: EventBus | None = None,
        version_manager: VersionManager | None = None,
        workspace_manager: WorkspaceManager | None = None,
        proposal_manager: ProposalManager | None = None,
        instance_store: Any = None,
    ) -> None:
        self._ontology = ontology
        self._event_bus = event_bus or EventBus()
        self._version_manager = version_manager or VersionManager()
        self._workspace_manager = workspace_manager or WorkspaceManager(ontology)
        self._proposal_manager = proposal_manager or ProposalManager()
        self._instance_store = instance_store

        # Sub-engines
        self._query_engine = OntologyQuery(schema=ontology, instance_store=instance_store)
        self._validation_engine = ValidationEngine()
        self._refactoring_engine = RefactoringEngine()

        # Reasoning framework with default reasoners
        self._reasoning_framework = ReasoningFramework(
            reasoners=[
                RuleBasedReasoner(name="default_rules"),
                GraphTraversalReasoner(name="graph_traversal"),
            ],
        )

        # Record initial version
        if not self._version_manager.get_history():
            self._version_manager.create_version(
                ontology=ontology,
                changes=[],
                author="system",
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ontology(self) -> Ontology:
        """The active ontology schema (read from current workspace's copy)."""
        return self._workspace_manager.get_ontology()

    @ontology.setter
    def ontology(self, value: Ontology) -> None:
        self._workspace_manager.set_ontology(value)
        self._ontology = value
        self._query_engine._schema = value

    @property
    def event_bus(self) -> EventBus:
        """The runtime's event bus."""
        return self._event_bus

    # ------------------------------------------------------------------
    # Helpers — emit typed events
    # ------------------------------------------------------------------

    def _emit(self, event_type: EventType, **data: Any) -> None:
        self._event_bus.emit(OntologyEvent(
            type=event_type,
            data=data or {},
            source="OntologyRuntime",
        ))

    # ------------------------------------------------------------------
    # Object lifecycle (proposal-driven mutations)
    # ------------------------------------------------------------------

    def create_object(
        self,
        name: str,
        description: str = "",
        properties: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OntologyProposal:
        """Propose creation of a new ontology object type.

        Parameters
        ----------
        name : str
            Canonical name for the new object type.
        description : str
            Optional description.
        properties : list[dict] | None
            Optional list of property definitions.
        metadata : dict | None
            Optional metadata dict.

        Returns
        -------
        OntologyProposal — in DRAFT status.  Caller must validate/approve/apply.
        """
        operation = OntologyOperation(
            type=OperationType.ADD_OBJECT_TYPE,
            target_id=name,
            payload={
                "object_type": ObjectType(
                    id=name,
                    name=name,
                    description=description,
                    metadata=metadata or {},
                ).to_dict() if hasattr(ObjectType, 'to_dict') else {
                    "id": name, "name": name, "description": description,
                },
            },
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Create object type '{name}'",
            changes=[operation],
            author="user",
            description=description,
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="create_object")
        return proposal

    def update_object(
        self,
        object_id: str,
        changes: dict[str, Any],
    ) -> OntologyProposal:
        """Propose updating an existing ontology object type.

        Parameters
        ----------
        object_id : str
            The object type name to update.
        changes : dict
            Key-value pairs of attributes to change.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.RENAME_OBJECT_TYPE,
            target_id=object_id,
            payload=changes,
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Update object type '{object_id}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="update_object")
        return proposal

    def delete_object(self, object_id: str) -> OntologyProposal:
        """Propose deletion of an ontology object type.

        Parameters
        ----------
        object_id : str
            The object type name to delete.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.REMOVE_OBJECT_TYPE,
            target_id=object_id,
            payload={},
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Delete object type '{object_id}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="delete_object")
        return proposal

    # ------------------------------------------------------------------
    # Relation lifecycle (proposal-driven)
    # ------------------------------------------------------------------

    def create_relation(
        self,
        name: str,
        source_type: str,
        target_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> OntologyProposal:
        """Propose creation of a new relation type.

        Parameters
        ----------
        name : str
            Relation type label.
        source_type : str
            Domain (source) entity type identifier.
        target_type : str
            Range (target) entity type identifier.
        metadata : dict | None
            Optional metadata.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.ADD_RELATION_TYPE,
            target_id=name,
            payload={
                "relation_type": RelationType(
                    id=name,
                    name=name,
                    source_type=source_type,
                    target_type=target_type,
                    metadata=metadata or {},
                ).to_dict(),
            },
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Create relation '{name}' ({source_type} → {target_type})",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="create_relation")
        return proposal

    def update_relation(
        self,
        relation_id: str,
        changes: dict[str, Any],
    ) -> OntologyProposal:
        """Propose updating an existing relation type.

        Parameters
        ----------
        relation_id : str
            The relation type identifier.
        changes : dict
            Key-value pairs to update.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.RENAME_RELATION_TYPE,
            target_id=relation_id,
            payload=changes,
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Update relation '{relation_id}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="update_relation")
        return proposal

    def delete_relation(self, relation_id: str) -> OntologyProposal:
        """Propose deletion of a relation type.

        Parameters
        ----------
        relation_id : str
            The relation type identifier to delete.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.REMOVE_RELATION_TYPE,
            target_id=relation_id,
            payload={},
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Delete relation '{relation_id}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="delete_relation")
        return proposal

    # ------------------------------------------------------------------
    # Property lifecycle (proposal-driven)
    # ------------------------------------------------------------------

    def create_property(
        self,
        property_name: str,
        object_type: str = "",
        data_type: str = "string",
        metadata: dict[str, Any] | None = None,
    ) -> OntologyProposal:
        """Propose adding a property to the ontology.

        Parameters
        ----------
        property_name : str
            The property name.
        object_type : str
            The object type this property belongs to (if applicable).
        data_type : str
            The property data type (``"string"``, ``"integer"``, …).
        metadata : dict | None
            Optional metadata.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.ADD_PROPERTY,
            target_id=f"{object_type}.{property_name}" if object_type else property_name,
            payload={
                "property_name": property_name,
                "object_type": object_type,
                "data_type": data_type,
                "metadata": metadata or {},
            },
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Add property '{property_name}' to '{object_type or 'global'}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="create_property")
        return proposal

    def delete_property(
        self,
        property_name: str,
        object_type: str = "",
    ) -> OntologyProposal:
        """Propose removing a property.

        Parameters
        ----------
        property_name : str
            The property name.
        object_type : str
            The object type this property belongs to.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.REMOVE_PROPERTY,
            target_id=f"{object_type}.{property_name}" if object_type else property_name,
            payload={"property_name": property_name, "object_type": object_type},
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Remove property '{property_name}' from '{object_type or 'global'}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="delete_property")
        return proposal

    # ------------------------------------------------------------------
    # Constraint lifecycle (proposal-driven)
    # ------------------------------------------------------------------

    def create_constraint(
        self,
        constraint_name: str,
        rules: dict[str, Any],
        target: str = "",
    ) -> OntologyProposal:
        """Propose adding a constraint to the ontology.

        Parameters
        ----------
        constraint_name : str
            Name for the constraint.
        rules : dict
            Constraint definition (e.g. ``{"unique": True}``).
        target : str
            Optional target element (object type or property path).

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.ADD_CONSTRAINT,
            target_id=constraint_name,
            payload={"rules": rules, "target": target},
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Add constraint '{constraint_name}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="create_constraint")
        return proposal

    def delete_constraint(self, constraint_name: str) -> OntologyProposal:
        """Propose removing a constraint.

        Parameters
        ----------
        constraint_name : str
            Name of the constraint to remove.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.REMOVE_CONSTRAINT,
            target_id=constraint_name,
            payload={},
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Remove constraint '{constraint_name}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="delete_constraint")
        return proposal

    # ------------------------------------------------------------------
    # Rule lifecycle (proposal-driven)
    # ------------------------------------------------------------------

    def create_rule(
        self,
        rule_name: str,
        condition: str,
        action: str,
        metadata: dict[str, Any] | None = None,
    ) -> OntologyProposal:
        """Propose adding an inference rule.

        Parameters
        ----------
        rule_name : str
            Name for the rule.
        condition : str
            Rule condition expression.
        action : str
            Rule action expression.
        metadata : dict | None
            Optional metadata.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.ADD_RULE,
            target_id=rule_name,
            payload={"condition": condition, "action": action, "metadata": metadata or {}},
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Add rule '{rule_name}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="create_rule")
        return proposal

    def delete_rule(self, rule_name: str) -> OntologyProposal:
        """Propose removing an inference rule.

        Parameters
        ----------
        rule_name : str
            Name of the rule to remove.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.REMOVE_RULE,
            target_id=rule_name,
            payload={},
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Remove rule '{rule_name}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="delete_rule")
        return proposal

    # ------------------------------------------------------------------
    # Workflow lifecycle (proposal-driven)
    # ------------------------------------------------------------------

    def create_workflow(
        self,
        workflow_name: str,
        steps: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> OntologyProposal:
        """Propose adding a workflow definition.

        Parameters
        ----------
        workflow_name : str
            Name for the workflow.
        steps : list[dict]
            Ordered list of workflow step definitions.
        metadata : dict | None
            Optional metadata.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.ADD_WORKFLOW,
            target_id=workflow_name,
            payload={"steps": steps, "metadata": metadata or {}},
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Add workflow '{workflow_name}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="create_workflow")
        return proposal

    def delete_workflow(self, workflow_name: str) -> OntologyProposal:
        """Propose removing a workflow definition.

        Parameters
        ----------
        workflow_name : str
            Name of the workflow to remove.

        Returns
        -------
        OntologyProposal
        """
        operation = OntologyOperation(
            type=OperationType.REMOVE_WORKFLOW,
            target_id=workflow_name,
            payload={},
        )
        proposal = self._proposal_manager.create_proposal(
            title=f"Remove workflow '{workflow_name}'",
            changes=[operation],
            author="user",
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="delete_workflow")
        return proposal

    # ------------------------------------------------------------------
    # Query methods (OQL delegates)
    # ------------------------------------------------------------------

    def query(
        self,
        object_type: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Query ontology objects by type and optional filters.

        Delegates to OQL (``OntologyQuery``).

        Parameters
        ----------
        object_type : str | None
            Canonical object type name to filter by.
        filters : dict | None
            Key-value attribute filters.

        Returns
        -------
        QueryResult
        """
        return self._query_engine.query(object_type=object_type, filters=filters)

    def search(self, text: str) -> QueryResult:
        """Full-text fuzzy search across ontology types, relations, and instances.

        Parameters
        ----------
        text : str
            Search query string.

        Returns
        -------
        QueryResult
        """
        return self._query_engine.search(text=text)

    def find_related(
        self,
        object_id: str,
        relation_type: str | None = None,
        max_depth: int = 1,
    ) -> QueryResult:
        """Graph traversal to find entities related to *object_id*.

        Parameters
        ----------
        object_id : str
            Starting entity or type.
        relation_type : str | None
            Optional relation type filter.
        max_depth : int
            Maximum traversal depth.

        Returns
        -------
        QueryResult
        """
        return self._query_engine.find_related(
            object_id=object_id,
            relation_type=relation_type,
            max_depth=max_depth,
        )

    # ------------------------------------------------------------------
    # Schema inspection
    # ------------------------------------------------------------------

    def get_object(self, object_id: str) -> dict[str, Any] | None:
        """Return metadata for a single ontology object type.

        Parameters
        ----------
        object_id : str
            The canonical object type identifier or name.

        Returns
        -------
        dict | None
        """
        return self._query_engine.get_object(object_id)

    def get_relation(self, relation_id: str) -> dict[str, Any] | None:
        """Return metadata for a single relation type.

        Parameters
        ----------
        relation_id : str
            The relation type identifier.

        Returns
        -------
        dict | None
        """
        for rel in self._query_engine.get_all_relations():
            if rel.get("id") == relation_id or rel.get("name") == relation_id:
                return rel
        return None

    def get_property(self, property_id: str) -> dict[str, Any] | None:
        """Return metadata for a single property.

        Parameters
        ----------
        property_id : str
            The property identifier.

        Returns
        -------
        dict | None
        """
        for prop in self._query_engine.get_all_properties():
            if prop.get("id") == property_id or prop.get("name") == property_id:
                return prop
        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """Validate the current ontology for internal consistency.

        Delegates to ``ontology_validation.ValidationEngine``.

        Returns
        -------
        ValidationReport
        """
        return self._validation_engine.validate(self.ontology)

    # ------------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------------

    def reason(self, target: str) -> InferenceReport:
        """Perform deductive reasoning about *target*.

        Delegates to ``ontology_reasoning.ReasoningFramework``.

        Parameters
        ----------
        target : str
            The entity or type to reason about.

        Returns
        -------
        InferenceReport
        """
        return self._reasoning_framework.reason(self.ontology, target=target)

    # ------------------------------------------------------------------
    # Analysis / Refactoring
    # ------------------------------------------------------------------

    def analyze(self) -> RefactoringReport:
        """Analyse the ontology for structural issues and improvements.

        Delegates to ``ontology_refactoring.RefactoringEngine``.

        Returns
        -------
        RefactoringReport
        """
        return self._refactoring_engine.analyze(self.ontology)

    def suggest_refactors(self) -> RefactoringReport:
        """Return a focused set of actionable refactoring suggestions.

        Delegates to ``ontology_refactoring.RefactoringEngine``.

        Returns
        -------
        RefactoringReport
        """
        return self._refactoring_engine.analyze(self.ontology)

    # ------------------------------------------------------------------
    # Version history
    # ------------------------------------------------------------------

    def history(self) -> list[OntologyVersion]:
        """Return the full version history of the ontology.

        Returns
        -------
        list[OntologyVersion]
        """
        return self._version_manager.get_history()

    # ------------------------------------------------------------------
    # Workspace management
    # ------------------------------------------------------------------

    def create_workspace(self, name: str, parent: str | None = None) -> Workspace:
        """Create a new workspace forked from *parent* (or the current one).

        Parameters
        ----------
        name : str
            Human-readable name.
        parent : str | None
            Workspace to fork from.

        Returns
        -------
        Workspace
        """
        ws = self._workspace_manager.create_workspace(name=name, parent=parent)
        self._emit(EventType.WORKSPACE_CREATED, name=name, parent=parent, workspace_id=ws.workspace_id)
        return ws

    def switch_workspace(self, workspace_id: str) -> None:
        """Switch the active workspace.

        Parameters
        ----------
        workspace_id : str
            Workspace identifier.
        """
        self._workspace_manager.switch_workspace(workspace_id)
        self._ontology = self._workspace_manager.get_ontology()
        self._query_engine._schema = self._ontology
        self._emit(EventType.WORKSPACE_SWITCHED, workspace_id=workspace_id)

    def merge_workspace(self, source: str, target: str) -> MergeResult:
        """Merge changes from *source* workspace into *target* workspace.

        Parameters
        ----------
        source : str
            Workspace to merge from.
        target : str
            Workspace to merge into.

        Returns
        -------
        MergeResult
        """
        result = self._workspace_manager.merge_workspace(source, target)
        self._emit(EventType.WORKSPACE_MERGED, source=source, target=target)
        return result

    def list_workspaces(self) -> list[Workspace]:
        """List all registered workspaces.

        Returns
        -------
        list[Workspace]
        """
        return self._workspace_manager.list_workspaces()

    def get_current_workspace(self) -> str:
        """Return the active workspace identifier.

        Returns
        -------
        str
        """
        return self._workspace_manager.get_current()

    # ------------------------------------------------------------------
    # Proposal management
    # ------------------------------------------------------------------

    def propose(
        self,
        title: str,
        changes: list[OntologyOperation],
        author: str = "user",
        description: str = "",
    ) -> OntologyProposal:
        """Create a new proposal with arbitrary changes.

        This is the low-level proposal entry point; convenience methods
        like ``create_object`` use it internally.

        Parameters
        ----------
        title : str
            Proposal title.
        changes : list[OntologyOperation]
            The operations the proposal bundles.
        author : str
            Creator label.
        description : str
            Optional description.

        Returns
        -------
        OntologyProposal
        """
        proposal = self._proposal_manager.create_proposal(
            title=title,
            changes=changes,
            author=author,
            description=description,
        )
        self._emit(EventType.PROPOSAL_CREATED, proposal_id=proposal.id, action="propose")
        return proposal

    def validate_proposal(self, proposal_id: str) -> tuple[bool, str]:
        """Validate a proposal (runs ``ValidationEngine`` on it).

        Parameters
        ----------
        proposal_id : str
            The proposal identifier.

        Returns
        -------
        (is_valid: bool, notes: str)
        """
        return self._proposal_manager.validate_proposal(proposal_id)

    def approve_proposal(self, proposal_id: str, notes: str = "") -> OntologyProposal:
        """Approve a validated proposal.

        Parameters
        ----------
        proposal_id : str
            The proposal identifier.
        notes : str
            Optional reviewer notes.

        Returns
        -------
        OntologyProposal
        """
        proposal = self._proposal_manager.approve_proposal(proposal_id, notes=notes)
        self._emit(EventType.PROPOSAL_APPROVED, proposal_id=proposal_id)
        return proposal

    def reject_proposal(self, proposal_id: str, reason: str = "") -> OntologyProposal:
        """Reject a proposal.

        Parameters
        ----------
        proposal_id : str
            The proposal identifier.
        reason : str
            Reason for rejection.

        Returns
        -------
        OntologyProposal
        """
        proposal = self._proposal_manager.reject_proposal(proposal_id, reason=reason)
        self._emit(EventType.PROPOSAL_REJECTED, proposal_id=proposal_id)
        return proposal

    def apply_proposal(self, proposal_id: str) -> list[OntologyOperation]:
        """Apply an approved proposal to the current workspace ontology.

        After application, a new version snapshot is created.

        Parameters
        ----------
        proposal_id : str
            The proposal identifier.

        Returns
        -------
        list[OntologyOperation]
            The operations that were applied.
        """
        # Apply to workspace ontology
        ws_ontology = self._workspace_manager.get_ontology()
        applied = self._proposal_manager.apply_proposal(proposal_id, ws_ontology)

        # Record a new version
        proposal = self._proposal_manager.get_proposal(proposal_id)
        author = proposal.author if proposal else "system"
        self._version_manager.create_version(
            ontology=ws_ontology,
            changes=[
                SemanticChange(
                    change_type=SemanticChangeType.OBJECT_CREATED,
                    target_id=op.target_id,
                    details=op.payload,
                )
                for op in applied
            ],
            author=author,
        )

        self._emit(EventType.PROPOSAL_APPROVED, proposal_id=proposal_id, action="applied")
        return applied

    def list_proposals(
        self,
        status_filter: ProposalStatus | None = None,
    ) -> list[OntologyProposal]:
        """List proposals, optionally filtered by status.

        Parameters
        ----------
        status_filter : ProposalStatus | None
            Optional status filter.

        Returns
        -------
        list[OntologyProposal]
        """
        return self._proposal_manager.list_proposals(status_filter=status_filter)

    def get_proposal(self, proposal_id: str) -> OntologyProposal | None:
        """Retrieve a single proposal.

        Parameters
        ----------
        proposal_id : str
            The proposal identifier.

        Returns
        -------
        OntologyProposal | None
        """
        return self._proposal_manager.get_proposal(proposal_id)
