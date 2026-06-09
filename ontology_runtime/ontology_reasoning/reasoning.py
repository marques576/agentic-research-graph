"""
Ontology reasoning — inference engines for deriving new knowledge.

Provides:
- RuleBasedReasoner: matches rule conditions against ontology objects and
  relations, then asserts consequences.
- GraphTraversalReasoner: walks relations to find indirect paths (transitive
  closure).
- ReasoningFramework: orchestrates multiple reasoners and aggregates results.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any

from ..ontology_model.types import Ontology

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """A single inference produced by a reasoner.

    Attributes
    ----------
    inferences : list[dict]
        Each dict is a derived fact, e.g.
        ``{"source": "A", "relation": "related_to", "target": "C"}``.
    explanation : str
        Human-readable description of how the inference was reached.
    confidence : float
        Confidence score in [0, 1].
    """

    inferences: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 1.0


@dataclass
class InferenceReport:
    """Aggregated output from running one or more reasoners.

    Attributes
    ----------
    results : list[InferenceResult]
        Individual results from each reasoner.
    reasoning_time_ms : float
        Total wall-clock time spent reasoning, in milliseconds.
    reasoner_type : str
        A label identifying the type of reasoner that produced this report.
    """

    results: list[InferenceResult] = field(default_factory=list)
    reasoning_time_ms: float = 0.0
    reasoner_type: str = ""


# ---------------------------------------------------------------------------
# Reasoner interface
# ---------------------------------------------------------------------------


class ReasonerInterface(abc.ABC):
    """Abstract base for all ontology reasoners.

    Subclasses must implement :meth:`reason`.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__

    @abc.abstractmethod
    def reason(
        self,
        ontology: Ontology,
        target: str | None = None,
    ) -> InferenceReport:
        """Run reasoning on *ontology*, optionally focused on *target*.

        Parameters
        ----------
        ontology : Ontology
            The ontology to reason over.
        target : str | None
            Optional object id to focus reasoning on.

        Returns
        -------
        InferenceReport
        """
        ...


# ---------------------------------------------------------------------------
# Rule-based reasoner
# ---------------------------------------------------------------------------

RuleCondition = dict[str, Any]
"""A condition dict for rule matching.

Supported forms:
    ``{"type": "has_object", "id": "some_id"}``
    ``{"type": "has_relation", "source": ..., "relation": ..., "target": ...}``
    ``{"type": "has_property", "object_id": ..., "key": ..., "value": ...}``
"""


@dataclass
class Rule:
    """A single inference rule.

    Attributes
    ----------
    name : str
        Human-readable rule name.
    conditions : list[RuleCondition]
        All conditions must match for the rule to fire.
    consequences : list[RuleCondition]
        Facts to assert when the rule fires.
    confidence : float
        Confidence to assign to inferences from this rule.
    """

    name: str = ""
    conditions: list[RuleCondition] = field(default_factory=list)
    consequences: list[RuleCondition] = field(default_factory=list)
    confidence: float = 0.8


class RuleBasedReasoner(ReasonerInterface):
    """Reasoner that applies a set of user-defined rules.

    Each rule's conditions are matched against the ontology objects and
    relations.  When all conditions of a rule are satisfied, its consequences
    are recorded as inferences.

    Parameters
    ----------
    rules : list[Rule]
        Rules to apply during reasoning.
    name : str
        Optional label for this reasoner instance.
    """

    def __init__(
        self,
        rules: list[Rule] | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name or "RuleBasedReasoner")
        self._rules: list[Rule] = list(rules) if rules else []

    def add_rule(self, rule: Rule) -> None:
        """Register an additional rule."""
        self._rules.append(rule)

    def reason(
        self,
        ontology: Ontology,
        target: str | None = None,
    ) -> InferenceReport:
        """Match rules against *ontology* and return inferred facts.

        Parameters
        ----------
        ontology : Ontology
        target : str | None
            If given, only consider objects/relations involving *target*.

        Returns
        -------
        InferenceReport
        """
        start = time.perf_counter()
        results: list[InferenceResult] = []

        for rule in self._rules:
            bindings = self._match_conditions(rule.conditions, ontology, target)
            if bindings:
                inferred_facts = self._apply_consequences(
                    rule.consequences, bindings,
                )
                if inferred_facts:
                    results.append(InferenceResult(
                        inferences=inferred_facts,
                        explanation=(
                            f"Rule '{rule.name}' fired — "
                            f"matched {len(bindings)} condition set(s)"
                        ),
                        confidence=rule.confidence,
                    ))

        elapsed = (time.perf_counter() - start) * 1000
        return InferenceReport(
            results=results,
            reasoning_time_ms=elapsed,
            reasoner_type=self.name,
        )

    def _match_conditions(
        self,
        conditions: list[RuleCondition],
        ontology: Ontology,
        target: str | None,
    ) -> list[dict[str, Any]]:
        """Return a list of binding dicts that satisfy *conditions*."""
        if not conditions:
            return [{}]

        binding_sets: list[dict[str, Any]] = [{}]

        for cond in conditions:
            new_sets: list[dict[str, Any]] = []
            cond_type = cond.get("type", "")

            for bindings in binding_sets:
                matches = self._match_single_condition(
                    cond_type, cond, ontology, target, bindings,
                )
                new_sets.extend(matches)

            if not new_sets:
                return []
            binding_sets = new_sets

        return binding_sets

    def _match_single_condition(
        self,
        cond_type: str,
        cond: dict[str, Any],
        ontology: Ontology,
        target: str | None,
        bindings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if cond_type == "has_object":
            return self._match_has_object(cond, ontology, bindings)
        elif cond_type == "has_relation":
            return self._match_has_relation(cond, ontology, target, bindings)
        elif cond_type == "has_property":
            return self._match_has_property(cond, ontology, bindings)
        return []

    def _match_has_object(
        self,
        cond: dict[str, Any],
        ontology: Ontology,
        bindings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        obj_id = cond.get("id", "")
        if obj_id:
            if obj_id in ontology.objects:
                return [{**bindings}]
            return []
        results = []
        for oid in ontology.objects:
            results.append({**bindings, "object_id": oid})
        return results

    def _match_has_relation(
        self,
        cond: dict[str, Any],
        ontology: Ontology,
        target: str | None,
        bindings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        source = cond.get("source", "*")
        rel_name = cond.get("relation", "*")
        target_cond = cond.get("target", "*")

        results = []
        for rel_id, rel in ontology.relations.items():
            if target is not None and rel.source_type != target and rel.target_type != target:
                continue
            if source != "*" and rel.source_type != source:
                continue
            if rel_name != "*" and rel.name != rel_name:
                continue
            if target_cond != "*" and rel.target_type != target_cond:
                continue
            results.append({**bindings,
                            "relation_id": rel_id,
                            "source_id": rel.source_type,
                            "relation_name": rel.name,
                            "target_id": rel.target_type})
        return results

    def _match_has_property(
        self,
        cond: dict[str, Any],
        ontology: Ontology,
        bindings: dict[str, Any],
    ) -> list[dict[str, Any]]:
        obj_id = cond.get("object_id", "")
        key = cond.get("key", "")
        value = cond.get("value")

        results = []
        for oid, obj in ontology.objects.items():
            if obj_id and oid != obj_id:
                continue
            for prop in obj.properties:
                if key and prop.name != key:
                    continue
                if value is not None and prop.default != value:
                    continue
                results.append({**bindings,
                                "object_id": oid,
                                "property_name": prop.name,
                                "property_value": prop.default})
        return results

    @staticmethod
    def _apply_consequences(
        consequences: list[RuleCondition],
        binding_sets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Materialise consequences for each binding set."""
        facts: list[dict[str, Any]] = []
        for bindings in binding_sets:
            for cons in consequences:
                fact = {}
                for k, v in cons.items():
                    if isinstance(v, str) and v.startswith("$"):
                        var_key = v[1:]
                        fact[k] = bindings.get(var_key, v)
                    else:
                        fact[k] = v
                facts.append(fact)
        return facts


# ---------------------------------------------------------------------------
# Graph traversal reasoner
# ---------------------------------------------------------------------------


class GraphTraversalReasoner(ReasonerInterface):
    """Reasoner that computes transitive closures over relations.

    Walks the ontology's relation graph to find indirect paths between
    objects.  Supports configurable depth limits and relation-name filters.

    Parameters
    ----------
    max_depth : int
        Maximum number of hops when traversing (default 5).
    relation_filter : set[str] | None
        If given, only traverse relations whose name is in this set.
    name : str
        Optional label.
    """

    def __init__(
        self,
        max_depth: int = 5,
        relation_filter: set[str] | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name or "GraphTraversalReasoner")
        self.max_depth = max_depth
        self.relation_filter = relation_filter

    def reason(
        self,
        ontology: Ontology,
        target: str | None = None,
    ) -> InferenceReport:
        """Compute transitive closure from *target* (or all objects).

        Parameters
        ----------
        ontology : Ontology
        target : str | None
            Starting object id.  If None, runs from every object.

        Returns
        -------
        InferenceReport
        """
        start = time.perf_counter()
        inferences: list[dict[str, Any]] = []

        if target is not None:
            sources = [target] if target in ontology.objects else []
        else:
            sources = list(ontology.objects.keys())

        for src in sources:
            paths = self._transitive_closure(src, ontology)
            for path in paths:
                if len(path) >= 2:
                    inferences.append({
                        "source": path[0],
                        "target": path[-1],
                        "path": path,
                        "hops": len(path) - 1,
                        "relation_type": "transitive_closure",
                    })

        elapsed = (time.perf_counter() - start) * 1000
        result = InferenceResult(
            inferences=inferences,
            explanation=(
                f"Transitive closure from {len(sources)} source(s) "
                f"with max_depth={self.max_depth}"
            ),
            confidence=0.7,
        )
        return InferenceReport(
            results=[result],
            reasoning_time_ms=elapsed,
            reasoner_type=self.name,
        )

    def _transitive_closure(
        self,
        start: str,
        ontology: Ontology,
    ) -> list[list[str]]:
        """BFS-based exploration returning all paths up to max_depth."""
        paths: list[list[str]] = []
        queue: list[list[str]] = [[start]]

        while queue:
            path = queue.pop(0)
            if len(path) > self.max_depth:
                continue
            paths.append(path)
            current = path[-1]
            for rel in ontology.relations.values():
                if rel.source_type != current:
                    continue
                if (self.relation_filter is not None
                        and rel.name not in self.relation_filter):
                    continue
                if rel.target_type not in path:
                    queue.append(path + [rel.target_type])
        return paths


# ---------------------------------------------------------------------------
# Reasoning framework
# ---------------------------------------------------------------------------


class ReasoningFramework:
    """Orchestrates multiple reasoners over an ontology.

    Parameters
    ----------
    reasoners : list[ReasonerInterface] | None
        Initial set of reasoners to manage.
    """

    def __init__(
        self,
        reasoners: list[ReasonerInterface] | None = None,
    ) -> None:
        self._reasoners: list[ReasonerInterface] = list(reasoners) if reasoners else []

    def register(self, reasoner: ReasonerInterface) -> None:
        """Add a reasoner to the framework."""
        self._reasoners.append(reasoner)

    def unregister(self, name: str) -> bool:
        """Remove a reasoner by name.  Returns True if found and removed."""
        for i, r in enumerate(self._reasoners):
            if r.name == name:
                self._reasoners.pop(i)
                return True
        return False

    def reason(
        self,
        ontology: Ontology,
        target: str | None = None,
    ) -> InferenceReport:
        """Run all registered reasoners and aggregate their results.

        Parameters
        ----------
        ontology : Ontology
        target : str | None

        Returns
        -------
        InferenceReport
            Contains one InferenceResult per reasoner that produced output.
        """
        start = time.perf_counter()
        all_results: list[InferenceResult] = []

        for reasoner in self._reasoners:
            report = reasoner.reason(ontology, target=target)
            all_results.extend(report.results)

        elapsed = (time.perf_counter() - start) * 1000
        return InferenceReport(
            results=all_results,
            reasoning_time_ms=elapsed,
            reasoner_type="ReasoningFramework",
        )
