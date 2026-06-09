"""Workspace management for ontology schemas.

Workspaces provide isolated copies of an ontology that can be edited
independently and later merged back.  This enables parallel experimentation
by multiple agents or users without interfering with each other.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


@dataclass
class Workspace:
    """An isolated copy of an ontology with its own change history.

    Attributes
    ----------
    workspace_id : str
        Unique identifier for this workspace.
    name : str
        Human-readable workspace label.
    parent : str | None
        Identifier of the workspace (or main branch) this was forked from.
    created_at : float
        Unix timestamp of creation.
    """

    workspace_id: str = ""
    name: str = ""
    parent: str | None = None
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# MergeConflict & MergeResult
# ---------------------------------------------------------------------------


@dataclass
class MergeConflict:
    """Describes a single conflict encountered during a merge.

    Attributes
    ----------
    path : str
        The ontology element path where the conflict occurred (e.g. an
        object type name or property name).
    ours : Any
        The value in the *source* (ours) workspace.
    theirs : Any
        The value in the *target* (theirs) workspace.
    resolution : str | None
        How the conflict was resolved; ``None`` until resolved.
    """

    path: str = ""
    ours: Any = None
    theirs: Any = None
    resolution: str | None = None


@dataclass
class MergeResult:
    """Outcome of a workspace merge operation.

    Attributes
    ----------
    success : bool
        ``True`` if the merge completed (with or without auto-resolved
        conflicts).
    diff : list[dict[str, Any]]
        Summary of the changes that were applied.
    conflicts : list[MergeConflict]
        Conflicts that the merge could not auto-resolve.
    """

    success: bool = False
    diff: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[MergeConflict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------


class WorkspaceManager:
    """Manages a collection of ontology workspaces and their merge lifecycle.

    There is always a **main** workspace (``"main"``) that acts as the
    canonical ontology.  All other workspaces are forks of *main* or of
    another workspace.

    Parameters
    ----------
    main_ontology : Any
        The initial ontology instance that populates the main workspace.
        Must support ``copy.deepcopy``.
    """

    def __init__(self, main_ontology: Any) -> None:
        self._workspaces: dict[str, Workspace] = {}
        self._ontologies: dict[str, Any] = {}
        self._current: str | None = None

        # Seed the main workspace
        main_ws = Workspace(
            workspace_id="main",
            name="main",
            parent=None,
            created_at=time.time(),
        )
        self._workspaces["main"] = main_ws
        self._ontologies["main"] = copy.deepcopy(main_ontology)
        self._current = "main"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_workspace(self, name: str, parent: str | None = None) -> Workspace:
        """Create a new workspace forked from *parent* (or the current one).

        Parameters
        ----------
        name : str
            Human-readable name for the new workspace.
        parent : str | None
            Workspace identifier to fork from.  Defaults to the current
            workspace if not given.

        Returns
        -------
        Workspace
            The newly created workspace.

        Raises
        ------
        ValueError
            If *parent* is specified but does not exist.
        """
        parent_id = parent or self._current
        if parent_id is None:
            raise ValueError("No parent workspace specified and no current workspace set.")
        if parent_id not in self._workspaces:
            raise ValueError(f"Parent workspace {parent_id!r} does not exist.")

        ws_id = str(uuid.uuid4())
        ws = Workspace(
            workspace_id=ws_id,
            name=name,
            parent=parent_id,
            created_at=time.time(),
        )
        self._workspaces[ws_id] = ws
        self._ontologies[ws_id] = copy.deepcopy(self._ontologies[parent_id])
        return ws

    def switch_workspace(self, workspace_id: str) -> None:
        """Switch the *current* workspace pointer.

        Parameters
        ----------
        workspace_id : str
            Identifier of the workspace to switch to.

        Raises
        ------
        ValueError
            If the workspace does not exist.
        """
        if workspace_id not in self._workspaces:
            raise ValueError(
                f"Workspace {workspace_id!r} does not exist."
            )
        self._current = workspace_id

    def merge_workspace(
        self,
        source: str,
        target: str,
        *,
        auto_resolve: bool = True,
    ) -> MergeResult:
        """Merge changes from *source* workspace into *target* workspace.

        A naive diff-based merge is performed: the ontology of *source* is
        compared to the ontology of *target*, and new / changed elements
        from *source* are copied over.  Conflicting changes (both sides
        modified the same element) are reported.

        Parameters
        ----------
        source : str
            Workspace identifier of the branch to merge **from**.
        target : str
            Workspace identifier of the branch to merge **into**.
        auto_resolve : bool
            If ``True``, conflicts where both sides added the same element
            are silently kept (source wins).  If ``False``, every conflict
            is returned without applying.

        Returns
        -------
        MergeResult
            Outcome of the merge.

        Raises
        ------
        ValueError
            If either workspace does not exist.
        """
        if source not in self._workspaces:
            raise ValueError(f"Source workspace {source!r} does not exist.")
        if target not in self._workspaces:
            raise ValueError(f"Target workspace {target!r} does not exist.")

        source_ont = self._ontologies[source]
        target_ont = self._ontologies[target]

        diff: list[dict[str, Any]] = []
        conflicts: list[MergeConflict] = []

        # Simple field-level merge for ontology attributes
        if hasattr(source_ont, "__dict__") and hasattr(target_ont, "__dict__"):
            source_dict = source_ont.__dict__
            target_dict = target_ont.__dict__

            for key, s_val in source_dict.items():
                if key.startswith("_"):
                    continue
                if key not in target_dict:
                    # New attribute in source → add to target
                    target_dict[key] = copy.deepcopy(s_val)
                    diff.append({
                        "action": "added",
                        "path": key,
                        "value": repr(s_val)[:200],
                    })
                else:
                    t_val = target_dict[key]
                    if s_val != t_val:
                        # Both sides differ — treat as conflict
                        conflict = MergeConflict(
                            path=key,
                            ours=s_val,
                            theirs=t_val,
                        )
                        if auto_resolve:
                            # Source wins
                            target_dict[key] = copy.deepcopy(s_val)
                            conflict.resolution = "source_wins"
                            diff.append({
                                "action": "resolved_conflict",
                                "path": key,
                                "resolution": "source_wins",
                            })
                        else:
                            conflict.resolution = "unresolved"
                        conflicts.append(conflict)

        success = not any(c.resolution == "unresolved" for c in conflicts)
        return MergeResult(success=success, diff=diff, conflicts=conflicts)

    def delete_workspace(self, workspace_id: str) -> None:
        """Delete a workspace and its ontology copy.

        The ``"main"`` workspace cannot be deleted.

        Parameters
        ----------
        workspace_id : str
            Identifier of the workspace to remove.

        Raises
        ------
        ValueError
            If the workspace is ``"main"`` or does not exist.
        """
        if workspace_id == "main":
            raise ValueError("Cannot delete the main workspace.")
        if workspace_id not in self._workspaces:
            raise ValueError(
                f"Workspace {workspace_id!r} does not exist."
            )
        del self._workspaces[workspace_id]
        self._ontologies.pop(workspace_id, None)
        if self._current == workspace_id:
            self._current = "main"

    def list_workspaces(self) -> list[Workspace]:
        """Return all registered workspaces.

        Returns
        -------
        list[Workspace]
            All workspaces in registration order.
        """
        return list(self._workspaces.values())

    def get_current(self) -> str:
        """Return the identifier of the active workspace.

        Returns
        -------
        str
            Workspace ID (defaults to ``"main"``).
        """
        return self._current or "main"

    def get_ontology(self, workspace_id: str | None = None) -> Any:
        """Return the ontology instance for a given workspace.

        Parameters
        ----------
        workspace_id : str | None
            Target workspace.  Defaults to the current workspace.

        Returns
        -------
        Any
            The ontology instance.
        """
        ws_id = workspace_id or self._current or "main"
        return self._ontologies.get(ws_id)

    def set_ontology(self, ontology: Any, workspace_id: str | None = None) -> None:
        """Replace the ontology in a workspace with a new instance.

        Parameters
        ----------
        ontology : Any
            The new ontology instance.
        workspace_id : str | None
            Target workspace.  Defaults to the current workspace.
        """
        ws_id = workspace_id or self._current or "main"
        self._ontologies[ws_id] = ontology
