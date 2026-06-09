"""Version management for ontology schemas.

Provides the data structures and manager needed to track changes to an
ontology over time, create semantic diffs between versions, and roll back
to previous states.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# SemanticChangeType
# ---------------------------------------------------------------------------


class SemanticChangeType(Enum):
    """Categorisation of a single change to an ontology."""

    OBJECT_CREATED = "object_created"
    OBJECT_UPDATED = "object_updated"
    OBJECT_DELETED = "object_deleted"
    RELATION_CREATED = "relation_created"
    RELATION_UPDATED = "relation_updated"
    RELATION_DELETED = "relation_deleted"
    PROPERTY_CREATED = "property_created"
    PROPERTY_DELETED = "property_deleted"
    CONSTRAINT_CREATED = "constraint_created"
    CONSTRAINT_DELETED = "constraint_deleted"
    RULE_CREATED = "rule_created"
    RULE_DELETED = "rule_deleted"
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_DELETED = "workflow_deleted"
    WORKSPACE_MERGE = "workspace_merge"
    ROLLBACK = "rollback"


# ---------------------------------------------------------------------------
# SemanticChange
# ---------------------------------------------------------------------------


@dataclass
class SemanticChange:
    """A single atomic change to an ontology schema element.

    Attributes
    ----------
    change_type : SemanticChangeType
        What kind of change occurred.
    target_id : str
        Identifier of the element that was changed (e.g. object type name,
        relation name, property path).
    details : dict[str, Any]
        Arbitrary extra data describing the change (previous value, new value,
        affected elements, …).
    timestamp : float
        Unix timestamp of when the change was recorded.
    """

    change_type: SemanticChangeType
    target_id: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# SemanticDiff
# ---------------------------------------------------------------------------


@dataclass
class SemanticDiff:
    """The set of changes between two ontology versions.

    Attributes
    ----------
    changes : list[SemanticChange]
        Ordered list of changes from *from_version* to *to_version*.
    ontology_id : str
        Identifier of the ontology this diff applies to.
    from_version : str
        Version identifier of the older version.
    to_version : str
        Version identifier of the newer version.
    """

    changes: list[SemanticChange] = field(default_factory=list)
    ontology_id: str = ""
    from_version: str = ""
    to_version: str = ""


# ---------------------------------------------------------------------------
# OntologyVersion
# ---------------------------------------------------------------------------


@dataclass
class OntologyVersion:
    """A snapshot in time of an ontology's state.

    Attributes
    ----------
    id : str
        Unique version identifier.
    version_number : int
        Monotonically increasing version number (1-based).
    timestamp : float
        When the version was created.
    author : str
        Who or what created this version (e.g. an agent or user).
    changes : list[SemanticChange]
        The list of changes that this version introduces.
    parent_version_id : str | None
        The version this one was derived from (``None`` for v1).
    """

    id: str = ""
    version_number: int = 0
    timestamp: float = field(default_factory=time.time)
    author: str = "system"
    changes: list[SemanticChange] = field(default_factory=list)
    parent_version_id: str | None = None


# ---------------------------------------------------------------------------
# VersionManager
# ---------------------------------------------------------------------------


class VersionManager:
    """Manages ontology version history, rollback, and diff computation.

    The manager stores a linear sequence of *OntologyVersion* snapshots.
    Each ``create_version`` call deep-copies the ontology state at that
    point so that older versions are immutable.

    Parameters
    ----------
    ontology_id : str
        A stable identifier for the ontology being tracked.
    """

    def __init__(self, ontology_id: str = "default") -> None:
        self._ontology_id = ontology_id
        self._versions: list[OntologyVersion] = []
        self._next_number = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_version(
        self,
        ontology: Any,
        changes: list[SemanticChange],
        author: str = "system",
    ) -> OntologyVersion:
        """Create a new version snapshot of the ontology.

        The ontology state is deep-copied and stored internally.  The new
        version is appended to the history.

        Parameters
        ----------
        ontology : Any
            The ontology instance to snapshot.  Must be deep-copyable.
        changes : list[SemanticChange]
            The changes that this version represents.
        author : str
            Creator label for the version.

        Returns
        -------
        OntologyVersion
            The newly created version object.
        """
        parent_id: str | None = (
            self._versions[-1].id if self._versions else None
        )

        version = OntologyVersion(
            id=str(uuid.uuid4()),
            version_number=self._next_number,
            timestamp=time.time(),
            author=author,
            changes=list(changes),
            parent_version_id=parent_id,
        )
        self._next_number += 1

        # Deep-copy the ontology state and store alongside the version metadata
        self._versions.append(version)
        self._snapshots: dict[str, Any] = getattr(self, "_snapshots", {})
        self._snapshots[version.id] = copy.deepcopy(ontology)

        return version

    def get_history(self) -> list[OntologyVersion]:
        """Return the full version history, oldest first.

        Returns
        -------
        list[OntologyVersion]
            Ordered list from earliest to latest version.
        """
        return list(self._versions)

    def get_version(self, version_id: str) -> OntologyVersion | None:
        """Look up a version by its unique identifier.

        Parameters
        ----------
        version_id : str
            The UUID of the target version.

        Returns
        -------
        OntologyVersion | None
            The version object, or ``None`` if not found.
        """
        for v in self._versions:
            if v.id == version_id:
                return v
        return None

    def rollback(self, version_id: str, ontology: Any) -> Any:
        """Roll an ontology instance back to a previous version.

        Parameters
        ----------
        version_id : str
            The target version to restore to.
        ontology : Any
            The current ontology instance that will be mutated in-place
            to match the snapshot.

        Returns
        -------
        Any
            The restored ontology instance (same object as *ontology*,
            but its state has been overwritten).

        Raises
        ------
        ValueError
            If *version_id* is not a known version or no snapshot exists.
        """
        snapshots: dict[str, Any] = getattr(self, "_snapshots", {})
        if version_id not in snapshots:
            raise ValueError(
                f"No snapshot found for version {version_id!r}"
            )

        snapshot = snapshots[version_id]
        # Mutate the ontology in-place so that references stay valid
        if hasattr(ontology, "__dict__"):
            ontology.__dict__.clear()
            ontology.__dict__.update(copy.deepcopy(snapshot.__dict__))
        else:
            ontology = copy.deepcopy(snapshot)

        return ontology

    def diff(
        self,
        version_a: str,
        version_b: str,
    ) -> SemanticDiff:
        """Compute the semantic diff between two versions.

        Parameters
        ----------
        version_a : str
            Identifier of the older version.
        version_b : str
            Identifier of the newer version.

        Returns
        -------
        SemanticDiff
            The aggregated changes from *version_a* to *version_b*.

        Raises
        ------
        ValueError
            If either version identifier is unknown.
        """
        va = self.get_version(version_a)
        vb = self.get_version(version_b)
        if va is None:
            raise ValueError(f"Unknown version: {version_a!r}")
        if vb is None:
            raise ValueError(f"Unknown version: {version_b!r}")

        # Collect all changes between the two versions (inclusive of A,
        # exclusive of B).  Walk the chain backwards from B to A.
        all_changes: list[SemanticChange] = []
        version_map = {v.id: v for v in self._versions}
        current = version_map.get(version_b)
        while current and current.id != version_a:
            all_changes = list(current.changes) + all_changes
            if current.parent_version_id:
                current = version_map.get(current.parent_version_id)
            else:
                break

        return SemanticDiff(
            changes=all_changes,
            ontology_id=self._ontology_id,
            from_version=version_a,
            to_version=version_b,
        )
