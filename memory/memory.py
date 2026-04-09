"""
Shared memory system for the knowledge discovery agent.

The memory object is the single source of truth passed between all agents.
Every tool call result is written back here so the entire agent loop
has a consistent, growing view of what has been discovered.

When a ``DomainOntology`` is attached (via ``AgentMemory.set_ontology``),
entity types are normalised on write and relationship edges are validated
against the ontology's allowed-triple schema before being committed.
Invalid edges are logged to ``ontology.rejection_log`` but not stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graph.knowledge_graph import KnowledgeGraph


@dataclass
class Entity:
    """A named entity extracted from a document."""
    name: str
    entity_type: str          # e.g. "company", "person", "institution"
    source: str               # where this was found
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.entity_type,
            "source": self.source,
            "attributes": self.attributes,
        }


@dataclass
class Relationship:
    """A directed relationship between two entities."""
    source: str
    target: str
    relation_type: str        # e.g. "funded_by", "employs", "collaborates_with"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class Hypothesis:
    """A candidate hypothesis about a hidden relationship in the corpus."""
    statement: str
    entities_involved: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supporting_evidence: list[str] = field(default_factory=list)
    refuting_evidence: list[str] = field(default_factory=list)
    validated: bool = False
    hypothesis_type: str = "structural_relationship"

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "entities_involved": self.entities_involved,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "refuting_evidence": self.refuting_evidence,
            "validated": self.validated,
            "hypothesis_type": self.hypothesis_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hypothesis:
        return cls(
            statement=data.get("statement", ""),
            entities_involved=data.get("entities_involved", []),
            confidence=float(data.get("confidence", 0.0)),
            supporting_evidence=data.get("supporting_evidence", []),
            refuting_evidence=data.get("refuting_evidence", []),
            validated=bool(data.get("validated", False)),
            hypothesis_type=data.get("hypothesis_type", "structural_relationship"),
        )


class AgentMemory:
    """
    Central shared memory for the knowledge graph agent loop.

    All agents read from and write to this object.  It stores:
      - entities      : named things discovered so far
      - relationships : confirmed edges between entities
      - documents     : raw / summarised document contents
      - evidence      : raw text snippets used to support reasoning
      - reasoning_log : step-by-step trace of the loop for observability
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}          # keyed by name (lowercase)
        self.relationships: list[Relationship] = []
        self.hypotheses: list[Hypothesis] = []
        self.documents: dict[str, str] = {}            # doc_id -> text
        self.evidence: list[str] = []
        self.reasoning_log: list[dict[str, Any]] = []
        self.iteration: int = 0
        self.graph: KnowledgeGraph = KnowledgeGraph()
        # Optional ontology — set via set_ontology() after construction
        self._ontology: Any | None = None  # DomainOntology | None

    # ------------------------------------------------------------------
    # Ontology attachment
    # ------------------------------------------------------------------

    def set_ontology(self, ontology: Any) -> None:
        """
        Attach a DomainOntology to this memory instance.

        Once attached, entity types are normalised and relationships are
        validated on every write.  The ontology can be replaced at any time
        (e.g. after the OntologyLearnerAgent extends it).

        Parameters
        ----------
        ontology : DomainOntology
            The ontology to use for normalisation and validation.
        """
        self._ontology = ontology

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        """Add or merge an entity.  Existing attributes are NOT overwritten.

        If an ontology is attached the entity type is normalised to its
        canonical form before storage (e.g. ``"person"`` → ``"researcher"``).
        """
        if self._ontology is not None:
            entity = Entity(
                name=entity.name,
                entity_type=self._ontology.normalise_type(entity.entity_type),
                source=entity.source,
                attributes=entity.attributes,
            )
        key = entity.name.lower()
        if key not in self.entities:
            self.entities[key] = entity
        else:
            existing = self.entities[key]
            for k, v in entity.attributes.items():
                if k not in existing.attributes:
                    existing.attributes[k] = v
        self.graph.add_entity(
            name=entity.name,
            entity_type=entity.entity_type,
            attributes=entity.attributes,
        )

    def get_entity(self, name: str) -> Entity | None:
        return self.entities.get(name.lower())

    def all_entity_names(self) -> list[str]:
        return [e.name for e in self.entities.values()]

    # ------------------------------------------------------------------
    # Relationship helpers
    # ------------------------------------------------------------------

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship; skip exact duplicates.

        If an ontology is attached the (source_type, relation_type, target_type)
        triple is validated before storage.  Invalid edges are logged to
        ``self._ontology.rejection_log`` and silently dropped.
        """
        if self._ontology is not None:
            src_entity = self.entities.get(rel.source.lower())
            tgt_entity = self.entities.get(rel.target.lower())
            src_type = src_entity.entity_type if src_entity else "unknown"
            tgt_type = tgt_entity.entity_type if tgt_entity else "unknown"
            valid, reason = self._ontology.validate_relationship(
                src_type, rel.relation_type, tgt_type
            )
            if not valid:
                self._ontology.log_rejection(
                    source=rel.source,
                    relation_type=rel.relation_type,
                    target=rel.target,
                    source_type=src_type,
                    target_type=tgt_type,
                    reason=reason,
                )
                self.reasoning_log.append({
                    "iteration": self.iteration,
                    "step": "ontology_rejection",
                    "content": {
                        "edge": f"{rel.source} --[{rel.relation_type}]--> {rel.target}",
                        "reason": reason,
                    },
                })
                return

        for existing in self.relationships:
            if (
                existing.source == rel.source
                and existing.target == rel.target
                and existing.relation_type == rel.relation_type
            ):
                if rel.confidence > existing.confidence:
                    existing.confidence = rel.confidence
                existing.evidence.extend(rel.evidence)
                self.graph.add_relationship(
                    source=rel.source,
                    target=rel.target,
                    relation_type=rel.relation_type,
                    confidence=existing.confidence,
                    evidence=rel.evidence[0] if rel.evidence else "",
                )
                return
        self.relationships.append(rel)
        self.graph.add_relationship(
            source=rel.source,
            target=rel.target,
            relation_type=rel.relation_type,
            confidence=rel.confidence,
            evidence=rel.evidence[0] if rel.evidence else "",
        )

    # ------------------------------------------------------------------
    # Graph convenience methods
    # ------------------------------------------------------------------

    def add_graph_entity(
        self,
        name: str,
        entity_type: str,
        attributes: dict = {},
    ) -> None:
        """
        Add entity to both the entity list and the knowledge graph.

        Parameters
        ----------
        name : str
            Entity name.
        entity_type : str
            Category label (e.g. "company", "person").
        attributes : dict
            Optional extra metadata.
        """
        entity = Entity(name=name, entity_type=entity_type, source="graph", attributes=dict(attributes))
        self.add_entity(entity)

    def add_graph_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        confidence: float = 1.0,
        evidence: str = "",
    ) -> None:
        """
        Add relationship to both the relationships list and the knowledge graph.

        Parameters
        ----------
        source : str
            Source entity name.
        target : str
            Target entity name.
        relation_type : str
            Relationship label.
        confidence : float
            Confidence score in [0, 1].
        evidence : str
            Text snippet supporting this relationship.
        """
        rel = Relationship(
            source=source,
            target=target,
            relation_type=relation_type,
            confidence=confidence,
            evidence=[evidence] if evidence else [],
        )
        self.add_relationship(rel)

    def get_graph_neighbors(self, node: str) -> list[str]:
        """
        Return graph neighbors of a node.

        Parameters
        ----------
        node : str
            Entity name.

        Returns
        -------
        list of neighbour node names.
        """
        return self.graph.neighbors(node)

    # ------------------------------------------------------------------
    # Document helpers
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, text: str) -> None:
        self.documents[doc_id] = text

    # ------------------------------------------------------------------
    # Evidence helpers
    # ------------------------------------------------------------------

    def add_evidence(self, snippet: str) -> None:
        if snippet not in self.evidence:
            self.evidence.append(snippet)

    # ------------------------------------------------------------------
    # Reasoning log
    # ------------------------------------------------------------------

    def log_step(self, step_type: str, content: Any) -> None:
        """Append a reasoning step to the trace log."""
        self.reasoning_log.append(
            {"iteration": self.iteration, "step": step_type, "content": content}
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relationships": [r.to_dict() for r in self.relationships],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "documents": list(self.documents.keys()),
            "evidence_count": len(self.evidence),
            "iterations": self.iteration,
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMemory:
        """
        Reconstruct an ``AgentMemory`` instance from a serialised dict.

        Parameters
        ----------
        data : dict
            A dict previously produced by ``AgentMemory.to_dict()``, or
            the top-level report JSON (which embeds the full memory snapshot
            under the ``"memory"`` key).

        Returns
        -------
        AgentMemory
            A freshly instantiated object with all fields populated.
        """
        source = data.get("memory", data)

        mem = cls()
        mem.iteration = source.get("iterations", data.get("iterations", 0))

        for e_dict in source.get("entities", []):
            ent = Entity(
                name=e_dict.get("name", ""),
                entity_type=e_dict.get("type", "unknown"),
                source=e_dict.get("source", "restored"),
                attributes=e_dict.get("attributes", {}),
            )
            mem.entities[ent.name.lower()] = ent

        for r_dict in source.get("relationships", []):
            rel = Relationship(
                source=r_dict.get("source", ""),
                target=r_dict.get("target", ""),
                relation_type=r_dict.get("type", r_dict.get("relation_type", "")),
                confidence=float(r_dict.get("confidence", 0.0)),
                evidence=r_dict.get("evidence", []),
            )
            mem.relationships.append(rel)

        for h_dict in source.get("hypotheses", []):
            try:
                mem.hypotheses.append(Hypothesis.from_dict(h_dict))
            except Exception:
                pass

        graph_data = source.get("graph")
        if graph_data:
            try:
                mem.graph.from_dict(graph_data)
            except Exception:
                pass

        return mem

    def summary(self) -> str:
        best_conf = max((h.confidence for h in self.hypotheses), default=0.0)
        return (
            f"Memory snapshot — iteration {self.iteration}: "
            f"{len(self.entities)} entities, "
            f"{len(self.relationships)} relationships, "
            f"{len(self.hypotheses)} hypotheses (best conf: {best_conf:.2f}), "
            f"{len(self.documents)} documents, "
            f"{len(self.evidence)} evidence snippets."
        )
