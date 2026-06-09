"""
Ontology testing — unit-test framework for ontology correctness.

Provides:
- OntologyTestCase: a base class with assertion helpers for ontologies.
- OntologyTestRunner: runs test functions and collects results into a TestSuite.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from ..ontology_model.types import Ontology


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TestResult:
    """Outcome of a single test.

    Attributes
    ----------
    name : str
        Test name.
    passed : bool
        True if the test succeeded.
    message : str
        Detailed message (pass description or failure reason).
    duration_ms : float
        Execution time in milliseconds.
    """

    name: str = ""
    passed: bool = True
    message: str = ""
    duration_ms: float = 0.0


@dataclass
class TestSuite:
    """Collection of test results for a given ontology.

    Attributes
    ----------
    name : str
        Label for this suite.
    results : list[TestResult]
        Individual test outcomes.
    total : int
        Number of tests run.
    passed : int
        Number of passing tests.
    failed : int
        Number of failing tests.
    """

    name: str = ""
    results: list[TestResult] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0

    def build_counts(self) -> None:
        """Recompute *total*, *passed*, and *failed* from *results*."""
        self.total = len(self.results)
        self.passed = sum(1 for r in self.results if r.passed)
        self.failed = self.total - self.passed


# ---------------------------------------------------------------------------
# Base test case
# ---------------------------------------------------------------------------


class OntologyTestCase:
    """Base class for ontology tests.

    Subclass this and add methods with names starting with ``test_``.
    Each such method will be discovered and run by OntologyTestRunner.

    All assertion methods raise ``AssertionError`` with a descriptive
    message on failure.
    """

    # The ontology under test — set by OntologyTestRunner before running
    ontology: Ontology

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    @staticmethod
    def assert_has_object(ontology: Ontology, obj_id: str, msg: str = "") -> None:
        """Assert that *obj_id* exists in *ontology.objects*."""
        if obj_id not in ontology.objects:
            raise AssertionError(
                msg or f"Expected object '{obj_id}' not found in ontology"
            )

    @staticmethod
    def assert_has_relation(ontology: Ontology, rel_id: str, msg: str = "") -> None:
        """Assert that a relation with *rel_id* exists."""
        for rel in ontology.relations:
            if rel.id == rel_id:
                return
        raise AssertionError(
            msg or f"Expected relation '{rel_id}' not found in ontology"
        )

    @staticmethod
    def assert_has_property(ontology: Ontology, obj_id: str, key: str, msg: str = "") -> None:
        """Assert that *obj_id* has a property with the given *key*."""
        obj = ontology.objects.get(obj_id)
        if obj is None:
            raise AssertionError(
                msg or f"Object '{obj_id}' not found — cannot check property '{key}'"
            )
        if key not in obj.properties:
            raise AssertionError(
                msg or f"Object '{obj_id}' does not have property '{key}'"
            )

    @staticmethod
    def assert_is_subclass(
        ontology: Ontology,
        child_id: str,
        parent_id: str,
        msg: str = "",
    ) -> None:
        """Assert that *child_id*'s parent chain includes *parent_id*."""
        current = ontology.objects.get(child_id)
        if current is None:
            raise AssertionError(
                msg or f"Child object '{child_id}' not found"
            )
        visited: set[str] = set()
        while current is not None:
            if current.id == parent_id:
                return
            if current.id in visited:
                raise AssertionError(
                    msg or f"Cycle detected while checking subclass for '{child_id}'"
                )
            visited.add(current.id)
            if current.parent:
                current = ontology.objects.get(current.parent)
            else:
                current = None
        raise AssertionError(
            msg or f"Object '{child_id}' is not a subclass (direct or indirect) of '{parent_id}'"
        )

    @staticmethod
    def assert_no_cycles(ontology: Ontology, msg: str = "") -> None:
        """Assert that there are no cycles in the parent hierarchy."""
        obj_ids = ontology.object_ids()
        parent_map: dict[str, str | None] = {
            oid: ontology.objects[oid].parent for oid in obj_ids
        }
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: dict[str, int] = {oid: WHITE for oid in obj_ids}

        def _dfs(node: str) -> bool:
            colour[node] = GRAY
            parent = parent_map.get(node)
            if parent is not None and parent in obj_ids:
                if colour.get(parent) == GRAY:
                    return True  # cycle
                if colour.get(parent) == WHITE and _dfs(parent):
                    return True
            colour[node] = BLACK
            return False

        for oid in obj_ids:
            if colour[oid] == WHITE:
                if _dfs(oid):
                    raise AssertionError(
                        msg or f"Cycle detected in ontology hierarchy involving '{oid}'"
                    )

    @staticmethod
    def assert_valid(ontology: Ontology, msg: str = "") -> None:
        """Assert that the ontology passes basic structural checks.

        Runs the ValidationEngine checks for missing endpoints and
        hierarchy cycles.
        """
        # Local import to avoid circular dependency at module level
        from ..ontology_validation.validation import (
            Severity,
            ValidationEngine,
        )

        engine = ValidationEngine(
            check_missing_endpoints=True,
            check_invalid_references=True,
            check_hierarchy_cycles=True,
            check_orphans=False,
            check_duplicate_concepts=False,
            check_missing_properties=False,
        )
        report = engine.validate(ontology)
        errors = [iss for iss in report.issues if iss.severity == Severity.ERROR]
        if errors:
            detail = "; ".join(e.message for e in errors[:5])
            raise AssertionError(
                msg or f"Ontology validation failed: {detail}"
            )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


class OntologyTestRunner:
    """Discovers and runs ontology tests.

    Parameters
    ----------
    test_functions : list[Callable] | None
        Optional pre-defined list of test callables.  If None, tests
        are discovered from an OntologyTestCase subclass.
    """

    def __init__(
        self,
        test_functions: list[Callable[[Ontology], None]] | None = None,
    ) -> None:
        self._test_functions = test_functions or []

    def add_test(self, func: Callable[[Ontology], None]) -> None:
        """Register an additional test function."""
        self._test_functions.append(func)

    def run_tests(
        self,
        ontology: Ontology,
        test_functions: list[Callable[[Ontology], None]] | None = None,
    ) -> TestSuite:
        """Run all registered (or provided) test functions against *ontology*.

        Parameters
        ----------
        ontology : Ontology
            The ontology to test.
        test_functions : list[Callable] | None
            Override for the registered test functions for this run.

        Returns
        -------
        TestSuite
        """
        funcs = test_functions if test_functions is not None else self._test_functions
        results: list[TestResult] = []

        for func in funcs:
            name = getattr(func, "__name__", str(func))
            start = time.perf_counter()
            try:
                func(ontology)
                elapsed = (time.perf_counter() - start) * 1000
                results.append(TestResult(
                    name=name,
                    passed=True,
                    message=f"Test '{name}' passed",
                    duration_ms=elapsed,
                ))
            except AssertionError as e:
                elapsed = (time.perf_counter() - start) * 1000
                results.append(TestResult(
                    name=name,
                    passed=False,
                    message=str(e),
                    duration_ms=elapsed,
                ))
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                tb = traceback.format_exc()
                results.append(TestResult(
                    name=name,
                    passed=False,
                    message=f"Unexpected error: {e}\n{tb}",
                    duration_ms=elapsed,
                ))

        suite = TestSuite(
            name="OntologyTestSuite",
            results=results,
        )
        suite.build_counts()
        return suite

    @staticmethod
    def discover_from_test_case(
        test_case_class: type[OntologyTestCase],
    ) -> list[Callable[[Ontology], None]]:
        """Discover test methods from an ``OntologyTestCase`` subclass.

        Finds all callable attributes whose name starts with ``test_``.
        Each returned callable expects a single *ontology* argument.

        Parameters
        ----------
        test_case_class : type[OntologyTestCase]
            A subclass of OntologyTestCase.

        Returns
        -------
        list of bound methods that accept an Ontology.
        """
        funcs: list[Callable[[Ontology], None]] = []
        for attr_name in dir(test_case_class):
            if attr_name.startswith("test_"):
                method = getattr(test_case_class, attr_name)
                if callable(method):
                    funcs.append(method)
        return funcs
