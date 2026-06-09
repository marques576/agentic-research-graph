"""
Ontology agents — capability-gated agent classes for interacting with ontologies.

Provides an abstract OntologyAgent that checks capabilities before acting,
and concrete agents (Architect, Critic, Validator, Reasoning, Refactoring)
each with a specific set of allowed capabilities.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any

from ..ontology_model.types import Ontology


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class Capability(str, enum.Enum):
    """Capabilities that an ontology agent may possess."""

    CREATE_OBJECT = "create_object"
    READ_OBJECT = "read_object"
    UPDATE_OBJECT = "update_object"
    DELETE_OBJECT = "delete_object"
    CREATE_RELATION = "create_relation"
    READ_RELATION = "read_relation"
    VALIDATE = "validate"
    REASON = "reason"
    ANALYZE = "analyze"
    PROPOSE = "propose"


@dataclass
class AgentCapabilities:
    """Capability profile for an agent.

    Attributes
    ----------
    capabilities : list[Capability]
        The capabilities this agent is allowed to exercise.
    agent_id : str
        Unique identifier for the agent.
    """

    capabilities: list[Capability] = field(default_factory=list)
    agent_id: str = ""


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class OntologyAgent(abc.ABC):
    """Abstract base for all ontology agents.

    Each agent is initialised with an *ontology* (or a runtime reference)
    and a set of *capabilities*.  All public action methods check the
    capability set before proceeding.

    Parameters
    ----------
    ontology : Ontology
        The ontology the agent operates on.
    capabilities : AgentCapabilities | None
        If None, the agent is given an empty capability set (no actions
        permitted).
    name : str
        Optional human-readable name.
    """

    def __init__(
        self,
        ontology: Ontology,
        capabilities: AgentCapabilities | None = None,
        name: str = "",
    ) -> None:
        self.ontology = ontology
        self._capabilities = capabilities or AgentCapabilities(
            capabilities=[],
            agent_id=name or self.__class__.__name__,
        )
        self.name = name or self.__class__.__name__

    # ------------------------------------------------------------------
    # Capability checks
    # ------------------------------------------------------------------

    def _check_capability(self, cap: Capability) -> None:
        """Raise PermissionError if *cap* is not in the agent's capability set."""
        if cap not in self._capabilities.capabilities:
            raise PermissionError(
                f"Agent '{self.name}' does not have capability '{cap.value}'"
            )

    # ------------------------------------------------------------------
    # Action methods
    # ------------------------------------------------------------------

    def inspect(self) -> dict[str, Any]:
        """Return a summary of what this agent sees in the ontology.

        Requires READ_OBJECT capability.
        """
        self._check_capability(Capability.READ_OBJECT)
        return {
            "agent": self.name,
            "capabilities": [c.value for c in self._capabilities.capabilities],
            "object_count": len(self.ontology.objects),
            "relation_count": len(self.ontology.relations),
            "object_ids": sorted(self.ontology.objects.keys()),
        }

    def query(self, obj_id: str) -> dict[str, Any] | None:
        """Retrieve an object by id.

        Requires READ_OBJECT capability.
        """
        self._check_capability(Capability.READ_OBJECT)
        obj = self.ontology.objects.get(obj_id)
        if obj is None:
            return None
        return {
            "id": obj.id,
            "name": obj.name,
            "type": obj.type,
            "properties": dict(obj.properties),
            "parent": obj.parent,
            "relations": [
                {
                    "id": r.id,
                    "relation_type": r.relation_type,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                }
                for r in self.ontology.get_relations_for(obj_id)
            ],
        }

    def propose(
        self,
        suggestion: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a proposal for a change to the ontology.

        Requires PROPOSE capability.

        Parameters
        ----------
        suggestion : str
            Description of the proposed change.
        details : dict | None
            Extra structured information.

        Returns
        -------
        dict with keys ``agent``, ``suggestion``, ``details``.
        """
        self._check_capability(Capability.PROPOSE)
        return {
            "agent": self.name,
            "suggestion": suggestion,
            "details": details or {},
        }

    def validate(self) -> dict[str, Any]:
        """Run validation on the ontology.

        Requires VALIDATE capability.

        Returns
        -------
        dict with validation report data.
        """
        self._check_capability(Capability.VALIDATE)
        from ..ontology_validation.validation import ValidationEngine

        engine = ValidationEngine()
        report = engine.validate(self.ontology)
        return {
            "agent": self.name,
            "passed": report.passed,
            "issue_count": len(report.issues),
            "issues": [
                {
                    "severity": iss.severity.value,
                    "category": iss.category,
                    "message": iss.message,
                    "target_id": iss.target_id,
                }
                for iss in report.issues
            ],
            "summary": report.summary,
        }

    def reason(
        self,
        target: str | None = None,
    ) -> dict[str, Any]:
        """Run reasoning on the ontology.

        Requires REASON capability.

        Parameters
        ----------
        target : str | None
            Optional object id to focus reasoning on.

        Returns
        -------
        dict with reasoning report data.
        """
        self._check_capability(Capability.REASON)
        from ..ontology_reasoning.reasoning import (
            GraphTraversalReasoner,
            ReasoningFramework,
            RuleBasedReasoner,
        )

        framework = ReasoningFramework()
        framework.register(GraphTraversalReasoner(max_depth=3))
        report = framework.reason(self.ontology, target=target)
        return {
            "agent": self.name,
            "reasoner_type": report.reasoner_type,
            "reasoning_time_ms": report.reasoning_time_ms,
            "result_count": len(report.results),
            "results": [
                {
                    "inference_count": len(r.inferences),
                    "explanation": r.explanation,
                    "confidence": r.confidence,
                    "sample_inferences": r.inferences[:5],
                }
                for r in report.results
            ],
        }

    def analyze(self) -> dict[str, Any]:
        """Run refactoring analysis on the ontology.

        Requires ANALYZE capability.

        Returns
        -------
        dict with refactoring report data.
        """
        self._check_capability(Capability.ANALYZE)
        from ..ontology_refactoring.refactoring import RefactoringEngine

        engine = RefactoringEngine()
        report = engine.analyze(self.ontology)
        return {
            "agent": self.name,
            "ontology_id": report.ontology_id,
            "analysis_time_ms": report.analysis_time_ms,
            "suggestion_count": len(report.suggestions),
            "suggestions": [
                {
                    "type": s.suggestion_type,
                    "confidence": s.confidence,
                    "description": s.description,
                    "rationale": s.rationale,
                    "target_ids": s.target_ids,
                }
                for s in report.suggestions
            ],
        }


# ---------------------------------------------------------------------------
# Concrete agents
# ---------------------------------------------------------------------------


class ArchitectAgent(OntologyAgent):
    """An agent that can create and modify ontology objects and relations.

    Capabilities: CREATE_OBJECT, CREATE_RELATION, CREATE_PROPERTY (via
    UPDATE_OBJECT), PROPOSE.

    Parameters
    ----------
    ontology : Ontology
    name : str
    """

    def __init__(
        self,
        ontology: Ontology,
        name: str = "",
    ) -> None:
        caps = AgentCapabilities(
            capabilities=[
                Capability.CREATE_OBJECT,
                Capability.READ_OBJECT,
                Capability.UPDATE_OBJECT,
                Capability.CREATE_RELATION,
                Capability.READ_RELATION,
                Capability.PROPOSE,
            ],
            agent_id=name or "ArchitectAgent",
        )
        super().__init__(ontology, capabilities=caps, name=name or "ArchitectAgent")

    def create_object(
        self,
        obj_id: str,
        name: str = "",
        obj_type: str = "unknown",
        properties: dict[str, Any] | None = None,
        parent: str | None = None,
    ) -> dict[str, Any]:
        """Create a new ontology object.

        Requires CREATE_OBJECT capability.
        """
        self._check_capability(Capability.CREATE_OBJECT)
        from ..ontology_model.types import OntologyObject

        if obj_id in self.ontology.objects:
            return {"success": False, "error": f"Object '{obj_id}' already exists"}

        new_obj = OntologyObject(
            id=obj_id,
            name=name or obj_id,
            type=obj_type,
            properties=properties or {},
            parent=parent,
        )
        self.ontology.add_object(new_obj)
        return {
            "success": True,
            "object_id": obj_id,
            "action": "created",
        }

    def create_relation(
        self,
        rel_id: str,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new relation between two objects.

        Requires CREATE_RELATION capability.
        """
        self._check_capability(Capability.CREATE_RELATION)
        from ..ontology_model.types import OntologyRelation

        if source_id not in self.ontology.objects:
            return {"success": False, "error": f"Source object '{source_id}' not found"}
        if target_id not in self.ontology.objects:
            return {"success": False, "error": f"Target object '{target_id}' not found"}

        new_rel = OntologyRelation(
            id=rel_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
        )
        self.ontology.add_relation(new_rel)
        return {
            "success": True,
            "relation_id": rel_id,
            "action": "created",
        }


class CriticAgent(OntologyAgent):
    """An agent that only analyses / critiques the ontology.

    Capabilities: READ_OBJECT, READ_RELATION, ANALYZE.
    """

    def __init__(
        self,
        ontology: Ontology,
        name: str = "",
    ) -> None:
        caps = AgentCapabilities(
            capabilities=[
                Capability.READ_OBJECT,
                Capability.READ_RELATION,
                Capability.ANALYZE,
            ],
            agent_id=name or "CriticAgent",
        )
        super().__init__(ontology, capabilities=caps, name=name or "CriticAgent")


class ValidatorAgent(OntologyAgent):
    """An agent that only validates ontologies.

    Capabilities: READ_OBJECT, READ_RELATION, VALIDATE.
    """

    def __init__(
        self,
        ontology: Ontology,
        name: str = "",
    ) -> None:
        caps = AgentCapabilities(
            capabilities=[
                Capability.READ_OBJECT,
                Capability.READ_RELATION,
                Capability.VALIDATE,
            ],
            agent_id=name or "ValidatorAgent",
        )
        super().__init__(ontology, capabilities=caps, name=name or "ValidatorAgent")


class ReasoningAgent(OntologyAgent):
    """An agent that only performs reasoning.

    Capabilities: READ_OBJECT, READ_RELATION, REASON.
    """

    def __init__(
        self,
        ontology: Ontology,
        name: str = "",
    ) -> None:
        caps = AgentCapabilities(
            capabilities=[
                Capability.READ_OBJECT,
                Capability.READ_RELATION,
                Capability.REASON,
            ],
            agent_id=name or "ReasoningAgent",
        )
        super().__init__(ontology, capabilities=caps, name=name or "ReasoningAgent")


class RefactoringAgent(OntologyAgent):
    """An agent that only performs refactoring analysis.

    Capabilities: READ_OBJECT, READ_RELATION, ANALYZE.
    """

    def __init__(
        self,
        ontology: Ontology,
        name: str = "",
    ) -> None:
        caps = AgentCapabilities(
            capabilities=[
                Capability.READ_OBJECT,
                Capability.READ_RELATION,
                Capability.ANALYZE,
            ],
            agent_id=name or "RefactoringAgent",
        )
        super().__init__(ontology, capabilities=caps, name=name or "RefactoringAgent")
