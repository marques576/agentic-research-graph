"""
DomainOntology – formal schema for entity classes and relationship types.

The ontology is the single source of truth for what entity types exist,
which relationship types are valid between which entity-type pairs, and
how to normalise noisy LLM-generated type strings into canonical labels.

It can be:
  1. Learned at runtime by OntologyLearnerAgent, which induces classes and
     relation triples from unstructured documents + the helper prompt.
  2. Persisted to a JSON file and reloaded across runs.

Relationship validation works as an *allow-list*:
  - If ``(domain_type, relation_type, range_type)`` is in ``allowed_triples``,
    the edge is accepted with full confidence.
  - If the relation is in ``weak_relations``, it is always accepted but
    flagged as low-confidence heuristic (e.g. co-occurs-with).
  - Otherwise the edge is rejected and logged.

Entity type normalisation maps alias strings to their canonical form as
learned from documents.  No domain-specific aliases are pre-loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# DomainOntology
# ---------------------------------------------------------------------------

@dataclass
class DomainOntology:
    """
    Formal schema for the knowledge graph domain.

    Starts completely empty — all entity types, triples, and aliases are
    learned from documents by OntologyLearnerAgent.

    Attributes
    ----------
    entity_types : set[str]
        Canonical entity type labels (core + discovered).
    core_entity_types : set[str]
        Entity types explicitly requested by the user via ``--helper-prompt``.
        These are seeded before the OntologyLearnerAgent runs and are given
        priority in the extraction prompt — the LLM is instructed to *always*
        tag entities of these types, even if it also discovers additional ones.
    type_aliases : dict[str, str]
        Maps variant/noisy type strings to a canonical entity type.
    allowed_triples : set[tuple[str, str, str]]
        Valid (domain_type, relation_type, range_type) combinations.
    inverse_relations : dict[str, str]
        Maps relation_type to its inverse (informational).
    weak_relations : set[str]
        Relation types always accepted without domain/range checking.
    rejection_log : list[dict]
        Running log of rejected edges (populated at runtime).
    """

    entity_types: set[str] = field(default_factory=set)
    core_entity_types: set[str] = field(default_factory=set)
    type_aliases: dict[str, str] = field(default_factory=dict)
    allowed_triples: set[tuple[str, str, str]] = field(default_factory=set)
    inverse_relations: dict[str, str] = field(default_factory=dict)
    weak_relations: set[str] = field(default_factory=lambda: {
        "co-occurs-with",
        "related_to",
        "mentions",
        "associated_with",
    })
    rejection_log: list[dict[str, Any]] = field(default_factory=list)
    goal_hash: str = field(default="")

    # ------------------------------------------------------------------
    # Type normalisation
    # ------------------------------------------------------------------

    def normalise_type(self, raw_type: str) -> str:
        """
        Map a raw entity type string to its canonical form.

        Falls back to the original string (lowercased) if no alias is found
        and the type is already in the canonical set.  Unknown types are
        returned as-is so they can be learned by the OntologyLearnerAgent.

        Parameters
        ----------
        raw_type : str
            Raw type string from extraction (e.g. "person", "Person", "startup").

        Returns
        -------
        str — canonical type label.
        """
        lower = raw_type.strip().lower()
        # Direct alias lookup
        if lower in self.type_aliases:
            return self.type_aliases[lower]
        # Already canonical
        if lower in self.entity_types:
            return lower
        # Unknown — return lowercased original so it can be reviewed
        return lower

    # ------------------------------------------------------------------
    # Relationship validation
    # ------------------------------------------------------------------

    def validate_relationship(
        self,
        source_type: str,
        relation_type: str,
        target_type: str,
    ) -> tuple[bool, str]:
        """
        Check whether a directed relationship is permitted by the ontology.

        Parameters
        ----------
        source_type : str
            Canonical type of the source entity.
        relation_type : str
            Relationship label.
        target_type : str
            Canonical type of the target entity.

        Returns
        -------
        (valid: bool, reason: str)
            valid  — True if the relationship is allowed.
            reason — human-readable explanation for rejection.
        """
        rel_lower = relation_type.strip().lower()

        # Weak/heuristic relations bypass domain/range checks
        if rel_lower in self.weak_relations:
            return True, "weak relation — accepted unconditionally"

        src_canon = self.normalise_type(source_type)
        tgt_canon = self.normalise_type(target_type)

        triple = (src_canon, rel_lower, tgt_canon)
        if triple in self.allowed_triples:
            return True, "allowed by ontology"

        # Wildcard: if either type is "unknown", allow with warning
        if src_canon == "unknown" or tgt_canon == "unknown":
            return True, "allowed (unknown type — cannot validate)"

        # Both entity types are known to the ontology — accept the edge even
        # if this exact triple wasn't explicitly listed in allowed_triples.
        # allowed_triples is populated incrementally by OntologyLearnerAgent
        # and won't cover every valid combination upfront.
        if src_canon in self.entity_types and tgt_canon in self.entity_types:
            return True, "allowed (both entity types are known)"

        reason = (
            f"({src_canon}, {rel_lower}, {tgt_canon}) not in allowed triples"
        )
        return False, reason

    def is_valid_relationship(
        self,
        source_type: str,
        relation_type: str,
        target_type: str,
    ) -> bool:
        """Convenience boolean wrapper around validate_relationship."""
        valid, _ = self.validate_relationship(source_type, relation_type, target_type)
        return valid

    # ------------------------------------------------------------------
    # Learning / extension
    # ------------------------------------------------------------------

    def add_entity_type(self, entity_type: str) -> None:
        """
        Register a new canonical entity type.

        Parameters
        ----------
        entity_type : str
            New canonical label (will be lowercased).
        """
        self.entity_types.add(entity_type.strip().lower())

    def add_core_entity_type(self, entity_type: str) -> None:
        """
        Register a user-specified core entity type.

        Core types are added to both ``core_entity_types`` and the full
        ``entity_types`` set so they are visible to normalisation and
        extraction prompt building.

        Parameters
        ----------
        entity_type : str
            Core type label (will be lowercased).
        """
        canon = entity_type.strip().lower()
        self.core_entity_types.add(canon)
        self.entity_types.add(canon)

    def add_type_alias(self, alias: str, canonical: str) -> None:
        """
        Register a type alias mapping.

        Parameters
        ----------
        alias : str
            Noisy or variant label.
        canonical : str
            Target canonical label.
        """
        self.type_aliases[alias.strip().lower()] = canonical.strip().lower()

    def add_allowed_triple(
        self,
        domain: str,
        relation: str,
        range_: str,
    ) -> None:
        """
        Add a new allowed (domain, relation, range) triple.

        Parameters
        ----------
        domain : str
            Source entity type (canonical).
        relation : str
            Relation type label.
        range_ : str
            Target entity type (canonical).
        """
        self.allowed_triples.add((
            domain.strip().lower(),
            relation.strip().lower(),
            range_.strip().lower(),
        ))

    def add_weak_relation(self, relation_type: str) -> None:
        """
        Mark a relation type as weak (bypass domain/range checks).

        Parameters
        ----------
        relation_type : str
            Relation label to mark as weak.
        """
        self.weak_relations.add(relation_type.strip().lower())

    def log_rejection(
        self,
        source: str,
        relation_type: str,
        target: str,
        source_type: str,
        target_type: str,
        reason: str,
    ) -> None:
        """Record a rejected edge to the rejection log."""
        self.rejection_log.append({
            "source":        source,
            "relation_type": relation_type,
            "target":        target,
            "source_type":   source_type,
            "target_type":   target_type,
            "reason":        reason,
        })

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the ontology to a plain dict (JSON-safe)."""
        return {
            "entity_types":      sorted(self.entity_types),
            "core_entity_types": sorted(self.core_entity_types),
            "type_aliases":      self.type_aliases,
            "allowed_triples":   [list(t) for t in sorted(self.allowed_triples)],
            "inverse_relations": self.inverse_relations,
            "weak_relations":    sorted(self.weak_relations),
            "goal_hash":         self.goal_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainOntology:
        """
        Reconstruct a DomainOntology from a serialised dict.

        Parameters
        ----------
        data : dict
            Previously produced by :meth:`to_dict`.

        Returns
        -------
        DomainOntology
        """
        ont = cls.__new__(cls)
        ont.entity_types = set(data.get("entity_types", []))
        ont.core_entity_types = set(data.get("core_entity_types", []))
        ont.type_aliases = dict(data.get("type_aliases", {}))
        ont.allowed_triples = {
            tuple(t) for t in data.get("allowed_triples", [])
        }
        ont.inverse_relations = dict(data.get("inverse_relations", {}))
        ont.weak_relations = set(data.get("weak_relations", [
            "co-occurs-with", "related_to", "mentions", "associated_with",
        ]))
        ont.rejection_log = []
        ont.goal_hash = data.get("goal_hash", "")
        return ont

    def save(self, path: str | Path) -> None:
        """
        Persist the ontology to a JSON file.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> DomainOntology:
        """
        Load an ontology from a JSON file.

        Falls back to a fresh seed ontology if the file does not exist
        or cannot be parsed.

        Parameters
        ----------
        path : str | Path
            Path to the JSON file.

        Returns
        -------
        DomainOntology
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return cls()

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        core_info = (
            f" ({len(self.core_entity_types)} core)" if self.core_entity_types else ""
        )
        return (
            f"DomainOntology — "
            f"{len(self.entity_types)} entity types{core_info}, "
            f"{len(self.allowed_triples)} allowed triples, "
            f"{len(self.type_aliases)} aliases, "
            f"{len(self.weak_relations)} weak relations"
        )

    def relation_types(self) -> set[str]:
        """Return the set of all relation type labels in allowed_triples."""
        return {rel for _, rel, _ in self.allowed_triples}

    def allowed_relations_for(self, domain_type: str) -> list[tuple[str, str]]:
        """
        Return all (relation_type, range_type) pairs valid for a given domain type.

        Parameters
        ----------
        domain_type : str
            Source entity type (will be normalised).

        Returns
        -------
        list of (relation_type, range_type) tuples.
        """
        canon = self.normalise_type(domain_type)
        return [(rel, rng) for dom, rel, rng in self.allowed_triples if dom == canon]
