"""Workspace management for ontology schemas."""

from __future__ import annotations

from .workspaces import (
    Workspace,
    MergeConflict,
    MergeResult,
    WorkspaceManager,
)

__all__ = [
    "Workspace",
    "MergeConflict",
    "MergeResult",
    "WorkspaceManager",
]
