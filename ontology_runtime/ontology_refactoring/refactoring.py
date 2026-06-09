"""
Ontology refactoring — analysis and suggestion engine for improving ontologies.

The RefactoringEngine analyses an Ontology and produces suggestions for
improvements.  It never auto-modifies; all output is advisory.

Checks implemented:
1. detect_duplicate_concepts — name similarity (lowercase match, substring)
2. detect_duplicate_relations — similar relation names
3. detect_overloaded_concepts — objects with 5+ properties
4. detect_dead_concepts — objects not referenced by any relation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..ontology_model.types import Ontology


@dataclass
class RefactorSuggestion:
    """A single suggestion produced by the refactoring engine.

    Attributes
    ----------
    suggestion_type : str
        Category label, e.g. "duplicate_concept", "overloaded_concept".
    confidence : float
        Confidence score in [0, 1].
    description : str
        Human-readable description of the suggestion.
    rationale : str
        Explanation of why this change is beneficial.
    changes : list[dict]
        Concrete change proposals, each a dict with at least a ``type``
        key (e.g. "merge", "rename", "split", "remove").
    target_ids : list[str]
        Object or relation ids that the suggestion pertains to.
    """

    suggestion_type: str = ""
    confidence: float = 0.0
    description: str = ""
    rationale: str = ""
    changes: list[dict[str, Any]] = field(default_factory=list)
    target_ids: list[str] = field(default_factory=list)


@dataclass
class RefactoringReport:
    """Report produced by running a refactoring analysis.

    Attributes
    ----------
    suggestions : list[RefactorSuggestion]
        All suggestions found during analysis.
    ontology_id : str
        Identifier for the analysed ontology (metadata id or empty).
    analysis_time_ms : float
        Wall-clock time for the analysis in milliseconds.
    """

    suggestions: list[RefactorSuggestion] = field(default_factory=list)
    ontology_id: str = ""
    analysis_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Refactoring engine
# ---------------------------------------------------------------------------


class RefactoringEngine:
    """Analyses an Ontology and produces improvement suggestions.

    Parameters
    ----------
    detect_duplicate_concepts : bool
    detect_duplicate_relations : bool
    detect_overloaded_concepts : bool
    detect_dead_concepts : bool
        Per-check toggles, all default True.
    min_overloaded_properties : int
        Threshold for "overloaded" (default 5).
    """

    def __init__(
        self,
        detect_duplicate_concepts: bool = True,
        detect_duplicate_relations: bool = True,
        detect_overloaded_concepts: bool = True,
        detect_dead_concepts: bool = True,
        min_overloaded_properties: int = 5,
    ) -> None:
        self._detect_duplicate_concepts = detect_duplicate_concepts
        self._detect_duplicate_relations = detect_duplicate_relations
        self._detect_overloaded_concepts = detect_overloaded_concepts
        self._detect_dead_concepts = detect_dead_concepts
        self._min_overloaded_properties = min_overloaded_properties

    def analyze(self, ontology: Ontology) -> RefactoringReport:
        """Run all enabled analyses against *ontology*.

        Parameters
        ----------
        ontology : Ontology

        Returns
        -------
        RefactoringReport
        """
        start = time.perf_counter()
        suggestions: list[RefactorSuggestion] = []

        if self._detect_duplicate_concepts:
            suggestions.extend(self._detect_duplicate_concepts_impl(ontology))
        if self._detect_duplicate_relations:
            suggestions.extend(self._detect_duplicate_relations_impl(ontology))
        if self._detect_overloaded_concepts:
            suggestions.extend(
                self._detect_overloaded_concepts_impl(ontology),
            )
        if self._detect_dead_concepts:
            suggestions.extend(self._detect_dead_concepts_impl(ontology))

        elapsed = (time.perf_counter() - start) * 1000
        return RefactoringReport(
            suggestions=suggestions,
            ontology_id=ontology.id,
            analysis_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_duplicate_concepts_impl(
        ontology: Ontology,
    ) -> list[RefactorSuggestion]:
        """Check 1: objects with similar or identical names."""
        suggestions: list[RefactorSuggestion] = []
        items = list(ontology.objects.items())

        for i, (oid_a, obj_a) in enumerate(items):
            name_a = obj_a.name.lower().strip()
            name_bare_a = name_a.replace("_", " ").replace("-", " ").strip()
            for _, obj_b in items[i + 1:]:
                name_b = obj_b.name.lower().strip()
                name_bare_b = name_b.replace("_", " ").replace("-", " ").strip()

                if name_a == name_b:
                    suggestions.append(RefactorSuggestion(
                        suggestion_type="duplicate_concept",
                        confidence=0.95,
                        description=(
                            f"'{obj_a.id}' and '{obj_b.id}' have "
                            f"identical names ('{obj_a.name}')"
                        ),
                        rationale=(
                            "Identical object names indicate possible "
                            "duplicate concepts that should be merged."
                        ),
                        changes=[
                            {
                                "type": "merge",
                                "keep_id": oid_a,
                                "remove_id": obj_b.id,
                            },
                        ],
                        target_ids=[oid_a, obj_b.id],
                    ))
                    continue

                if (name_a in name_b or name_b in name_a
                        or name_bare_a in name_bare_b
                        or name_bare_b in name_bare_a):
                    suggestions.append(RefactorSuggestion(
                        suggestion_type="duplicate_concept",
                        confidence=0.7,
                        description=(
                            f"'{obj_a.id}' ('{obj_a.name}') and "
                            f"'{obj_b.id}' ('{obj_b.name}') have "
                            f"overlapping names"
                        ),
                        rationale=(
                            "Overlapping names may indicate duplicate "
                            "or closely related concepts worth reviewing."
                        ),
                        changes=[
                            {
                                "type": "review",
                                "object_a": obj_a.id,
                                "object_b": obj_b.id,
                                "suggestion": "Consider merging or renaming",
                            },
                        ],
                        target_ids=[oid_a, obj_b.id],
                    ))

        return suggestions

    @staticmethod
    def _detect_duplicate_relations_impl(
        ontology: Ontology,
    ) -> list[RefactorSuggestion]:
        """Check 2: relations with similar names (lowercase match)."""
        suggestions: list[RefactorSuggestion] = []
        seen: dict[str, list[str]] = {}

        for rel_id, rel in ontology.relations.items():
            key = rel.name.lower().strip()
            if key not in seen:
                seen[key] = []
            seen[key].append(rel_id)

        for key, rel_ids in seen.items():
            if len(rel_ids) > 1:
                suggestions.append(RefactorSuggestion(
                    suggestion_type="duplicate_relation",
                    confidence=0.8,
                    description=(
                        f"Relation type '{key}' is used by "
                        f"{len(rel_ids)} relation(s): {rel_ids}"
                    ),
                    rationale=(
                        "Multiple relations with the same name may "
                        "indicate merged or duplicated relation definitions."
                    ),
                    changes=[
                        {
                            "type": "consolidate",
                            "relation_name": key,
                            "relation_ids": rel_ids,
                        },
                    ],
                    target_ids=rel_ids,
                ))

        return suggestions

    @staticmethod
    def _detect_overloaded_concepts_impl(
        ontology: Ontology,
    ) -> list[RefactorSuggestion]:
        """Check 3: objects with many properties."""
        suggestions: list[RefactorSuggestion] = []
        for obj in ontology.objects.values():
            if len(obj.properties) >= 5:
                suggestions.append(RefactorSuggestion(
                    suggestion_type="overloaded_concept",
                    confidence=0.6,
                    description=(
                        f"Object '{obj.id}' has "
                        f"{len(obj.properties)} properties"
                    ),
                    rationale=(
                        "Objects with many properties may be overloaded. "
                        "Consider splitting into sub-concepts or using "
                        "composition."
                    ),
                    changes=[
                        {
                            "type": "split",
                            "object_id": obj.id,
                            "property_count": len(obj.properties),
                            "suggestion": (
                                "Group related properties into child "
                                "objects"
                            ),
                        },
                    ],
                    target_ids=[obj.id],
                ))
        return suggestions

    @staticmethod
    def _detect_dead_concepts_impl(
        ontology: Ontology,
    ) -> list[RefactorSuggestion]:
        """Check 4: objects not referenced by any relation."""
        suggestions: list[RefactorSuggestion] = []
        related_ids: set[str] = set()
        for rel in ontology.relations.values():
            related_ids.add(rel.source_type)
            related_ids.add(rel.target_type)

        for obj in ontology.objects.values():
            if obj.id not in related_ids:
                suggestions.append(RefactorSuggestion(
                    suggestion_type="dead_concept",
                    confidence=0.9,
                    description=(
                        f"Object '{obj.id}' ('{obj.name}') is not "
                        f"referenced by any relation"
                    ),
                    rationale=(
                        "Objects with no relations may be unused or "
                        "orphaned. Consider removing or connecting them."
                    ),
                    changes=[
                        {
                            "type": "remove",
                            "object_id": obj.id,
                            "suggestion": (
                                "Verify usefulness, then remove or add "
                                "relations"
                            ),
                        },
                    ],
                    target_ids=[obj.id],
                ))
        return suggestions
