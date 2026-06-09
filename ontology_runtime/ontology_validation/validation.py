"""
Ontology validation — checks an Ontology for structural issues.

Provides ValidationEngine, which runs several built-in checks:

1. Missing endpoints — relation source/target types not in ontology.objects
2. Invalid references — dangling references (parent_id points to non-existent object)
3. Hierarchy cycles — DFS cycle detection via parent_id
4. Orphans — objects with no relations at all
5. Duplicate concepts — objects with similar names (case-insensitive match,
   Levenshtein distance ≤ 2)
6. Missing properties — required PropertyType entries where no value is set
   (for properties where PropertyType.required is True)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from ..ontology_model.types import Ontology


class Severity(str, enum.Enum):
    """Severity level for a validation issue."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """A single issue found during ontology validation.

    Attributes
    ----------
    severity : Severity
        How serious the issue is (ERROR, WARNING, INFO).
    category : str
        Category label, e.g. "missing_endpoint", "orphan".
    message : str
        Human-readable description of the issue.
    target_id : str | None
        The ontology object or relation id the issue pertains to,
        or None if not applicable.
    details : dict
        Extra structured information about the issue.
    """

    severity: Severity = Severity.WARNING
    category: str = ""
    message: str = ""
    target_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Report produced by running a validation pass.

    Attributes
    ----------
    passed : bool
        True if there are zero ERROR-severity issues.
    issues : list[ValidationIssue]
        All issues found during validation.
    summary : dict
        Aggregate counts keyed by category.
    """

    passed: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)

    def build_summary(self) -> None:
        """Populate *summary* with issue counts per category."""
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.category] = counts.get(issue.category, 0) + 1
        self.summary = counts
        self.passed = not any(
            iss.severity == Severity.ERROR for iss in self.issues
        )


# ---------------------------------------------------------------------------
# Levenshtein distance (no external deps)
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # insert
                prev[j] + 1,          # delete
                prev[j - 1] + cost,   # substitute
            )
        prev, curr = curr, prev
    return prev[n]


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------


class ValidationEngine:
    """Runs a suite of structural checks against an Ontology.

    Parameters
    ----------
    check_missing_endpoints : bool
    check_invalid_references : bool
    check_hierarchy_cycles : bool
    check_orphans : bool
    check_duplicate_concepts : bool
    check_missing_properties : bool
        Per-check toggles, all default True.
    """

    def __init__(
        self,
        check_missing_endpoints: bool = True,
        check_invalid_references: bool = True,
        check_hierarchy_cycles: bool = True,
        check_orphans: bool = True,
        check_duplicate_concepts: bool = True,
        check_missing_properties: bool = True,
    ) -> None:
        self._check_missing_endpoints = check_missing_endpoints
        self._check_invalid_references = check_invalid_references
        self._check_hierarchy_cycles = check_hierarchy_cycles
        self._check_orphans = check_orphans
        self._check_duplicate_concepts = check_duplicate_concepts
        self._check_missing_properties = check_missing_properties

    def validate(self, ontology: Ontology) -> ValidationReport:
        """Run all enabled checks against *ontology*.

        Parameters
        ----------
        ontology : Ontology
            The ontology to validate.

        Returns
        -------
        ValidationReport
        """
        issues: list[ValidationIssue] = []

        if self._check_missing_endpoints:
            issues.extend(self._check_missing_endpoints_impl(ontology))
        if self._check_invalid_references:
            issues.extend(self._check_invalid_references_impl(ontology))
        if self._check_hierarchy_cycles:
            issues.extend(self._check_hierarchy_cycles_impl(ontology))
        if self._check_orphans:
            issues.extend(self._check_orphans_impl(ontology))
        if self._check_duplicate_concepts:
            issues.extend(self._check_duplicate_concepts_impl(ontology))
        if self._check_missing_properties:
            issues.extend(self._check_missing_properties_impl(ontology))

        report = ValidationReport(issues=issues)
        report.build_summary()
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_missing_endpoints_impl(ontology: Ontology) -> list[ValidationIssue]:
        """Check 1: relation source/target object types not in ontology.objects."""
        issues: list[ValidationIssue] = []
        obj_ids = set(ontology.objects.keys())
        for rel_id, rel in ontology.relations.items():
            if rel.source_type not in obj_ids:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="missing_endpoint",
                    message=(
                        f"Relation '{rel_id}' references source type "
                        f"'{rel.source_type}' which is not in ontology.objects"
                    ),
                    target_id=rel_id,
                    details={
                        "relation_id": rel_id,
                        "missing_id": rel.source_type,
                        "role": "source",
                    },
                ))
            if rel.target_type not in obj_ids:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="missing_endpoint",
                    message=(
                        f"Relation '{rel_id}' references target type "
                        f"'{rel.target_type}' which is not in ontology.objects"
                    ),
                    target_id=rel_id,
                    details={
                        "relation_id": rel_id,
                        "missing_id": rel.target_type,
                        "role": "target",
                    },
                ))
        return issues

    @staticmethod
    def _check_invalid_references_impl(ontology: Ontology) -> list[ValidationIssue]:
        """Check 2: dangling references — parent_id points to non-existent object."""
        issues: list[ValidationIssue] = []
        obj_ids = set(ontology.objects.keys())
        for obj_id, obj in ontology.objects.items():
            if obj.parent_id is not None and obj.parent_id not in obj_ids:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="invalid_reference",
                    message=(
                        f"Object '{obj_id}' has parent_id '{obj.parent_id}' "
                        f"which does not exist"
                    ),
                    target_id=obj_id,
                    details={
                        "object_id": obj_id,
                        "dangling_ref": obj.parent_id,
                        "ref_type": "parent_id",
                    },
                ))
        return issues

    @staticmethod
    def _check_hierarchy_cycles_impl(ontology: Ontology) -> list[ValidationIssue]:
        """Check 3: hierarchy cycles via DFS cycle detection.

        Uses the ``parent_id`` field on each ObjectType to build a
        directed parent-child graph and looks for cycles.
        """
        issues: list[ValidationIssue] = []
        obj_ids = set(ontology.objects.keys())

        parent_map: dict[str, str | None] = {}
        for obj_id, obj in ontology.objects.items():
            parent_map[obj_id] = obj.parent_id

        WHITE, GRAY, BLACK = 0, 1, 2
        colour: dict[str, int] = {oid: WHITE for oid in obj_ids}
        path: list[str] = []

        def _dfs(node: str) -> bool:
            """Returns True if a cycle is found."""
            colour[node] = GRAY
            path.append(node)
            parent = parent_map.get(node)
            if parent is not None and parent in obj_ids:
                if colour.get(parent) == GRAY:
                    # Cycle detected
                    cycle_start = path.index(parent)
                    cycle = path[cycle_start:] + [parent]
                    issues.append(ValidationIssue(
                        severity=Severity.ERROR,
                        category="hierarchy_cycle",
                        message=(
                            f"Hierarchy cycle detected: "
                            f"{' -> '.join(cycle)}"
                        ),
                        target_id=node,
                        details={"cycle": cycle},
                    ))
                    path.pop()
                    colour[node] = BLACK
                    return True
                if colour.get(parent) == WHITE:
                    if _dfs(parent):
                        path.pop()
                        colour[node] = BLACK
                        return True
            path.pop()
            colour[node] = BLACK
            return False

        for oid in obj_ids:
            if colour[oid] == WHITE:
                _dfs(oid)

        return issues

    @staticmethod
    def _check_orphans_impl(ontology: Ontology) -> list[ValidationIssue]:
        """Check 4: objects with no relations at all."""
        issues: list[ValidationIssue] = []
        related_ids: set[str] = set()
        for rel in ontology.relations.values():
            related_ids.add(rel.source_type)
            related_ids.add(rel.target_type)

        for oid in sorted(ontology.objects.keys()):
            if oid not in related_ids:
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="orphan",
                    message=(
                        f"Object '{oid}' has no relations "
                        f"(incoming or outgoing)"
                    ),
                    target_id=oid,
                    details={"object_id": oid},
                ))
        return issues

    @staticmethod
    def _check_duplicate_concepts_impl(ontology: Ontology) -> list[ValidationIssue]:
        """Check 5: objects with similar names.

        Uses case-insensitive name matching and Levenshtein distance ≤ 2.
        """
        issues: list[ValidationIssue] = []
        items = list(ontology.objects.items())

        for i, (oid_a, obj_a) in enumerate(items):
            name_a = obj_a.name.lower().strip()
            for _, obj_b in items[i + 1:]:
                name_b = obj_b.name.lower().strip()
                if name_a == name_b:
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        category="duplicate_concept",
                        message=(
                            f"Objects '{obj_a.id}' and '{obj_b.id}' "
                            f"have identical names ('{obj_a.name}')"
                        ),
                        target_id=obj_a.id,
                        details={
                            "object_a": obj_a.id,
                            "object_b": obj_b.id,
                            "name_a": obj_a.name,
                            "name_b": obj_b.name,
                            "similarity": "exact",
                        },
                    ))
                elif _levenshtein(name_a, name_b) <= 2:
                    issues.append(ValidationIssue(
                        severity=Severity.INFO,
                        category="duplicate_concept",
                        message=(
                            f"Objects '{obj_a.id}' and '{obj_b.id}' "
                            f"have similar names ('{obj_a.name}' vs "
                            f"'{obj_b.name}', Levenshtein ≤ 2)"
                        ),
                        target_id=obj_a.id,
                        details={
                            "object_a": obj_a.id,
                            "object_b": obj_b.id,
                            "name_a": obj_a.name,
                            "name_b": obj_b.name,
                            "similarity": "levenshtein",
                            "distance": _levenshtein(name_a, name_b),
                        },
                    ))
        return issues

    @staticmethod
    def _check_missing_properties_impl(ontology: Ontology) -> list[ValidationIssue]:
        """Check 6: required properties on objects that aren't satisfied.

        A property is considered "missing" if it has ``required=True``
        but we cannot verify it's set.  Since the current model stores
        property *definitions* on the object, we flag properties that
        have no ``default`` value and are marked required — they may
        leave instances incomplete.
        """
        issues: list[ValidationIssue] = []
        for obj_id, obj in ontology.objects.items():
            missing = []
            for prop in obj.properties:
                if prop.required and prop.default is None:
                    missing.append(prop.name)
            if missing:
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="missing_properties",
                    message=(
                        f"Object '{obj_id}' has required properties "
                        f"with no default value: {sorted(missing)}"
                    ),
                    target_id=obj_id,
                    details={
                        "object_id": obj_id,
                        "missing_properties": sorted(missing),
                    },
                ))
        return issues
