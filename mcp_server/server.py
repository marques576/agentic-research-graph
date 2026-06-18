"""
MCP server exposing the ontology harness as tools.

Run with:
    uv run python -m mcp_server.server --db ontology.db

Connect via Claude Desktop, Claude Code, or any MCP client by adding to
the MCP config:
    {
      "mcpServers": {
        "ontology-harness": {
          "command": "uv",
          "args": ["run", "python", "-m", "mcp_server.server", "--db", "ontology.db"]
        }
      }
    }

Tool surface (see individual tool docstrings for details):
  Schema tools:
    - define_object_type
    - define_link_type
    - get_schema

  Write tools:
    - upsert_object
    - upsert_link
    - delete_object
    - delete_link

  Read tools:
    - get_object
    - find_objects
    - get_neighbors
    - get_provenance
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on the path
_src_root = Path(__file__).resolve().parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from mcp.server.fastmcp import FastMCP

from core.store import OntologyStore
from core.actions import ActionRegistry, ActionError
from core.query import QueryAPI


_DB_PATH: str | None = None
_store: OntologyStore | None = None
_actions: ActionRegistry | None = None
_query: QueryAPI | None = None


def _get_db_path() -> str:
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = os.environ.get("ONTOLOGY_DB", "ontology.db")
    return _DB_PATH


def _get_store() -> OntologyStore:
    global _store
    if _store is None:
        _store = OntologyStore(_get_db_path())
    return _store


def _get_actions() -> ActionRegistry:
    global _actions
    if _actions is None:
        _actions = ActionRegistry(_get_store())
    return _actions


def _get_query() -> QueryAPI:
    global _query
    if _query is None:
        _query = QueryAPI(_get_store())
    return _query


mcp = FastMCP("ontology-harness")


# ==========================================================================
# Schema tools
# ==========================================================================

@mcp.tool()
def define_object_type(
    name: str,
    properties: list[dict[str, Any]],
    description: str = "",
) -> dict[str, Any]:
    """
    Create a new Object Type (e.g. Person, Organization, Document).

    Properties is a list of dicts with keys:
      - name (str): property name
      - data_type (str): one of string, number, boolean, datetime, reference
      - required (bool, default False)

    This is how you define the schema.  Call get_schema first to see what
    types already exist before creating duplicates.

    Returns the created Object Type definition.
    """
    actions = _get_actions()
    try:
        return actions.create_object_type(
            name=name,
            properties=properties,
            description=description,
            source="mcp_client",
            agent="mcp_client",
        )
    except ActionError as e:
        raise ValueError(str(e))


@mcp.tool()
def define_link_type(
    name: str,
    source_type: str,
    target_type: str,
    cardinality: str = "many_to_many",
    description: str = "",
) -> dict[str, Any]:
    """
    Create a new Link Type (relationship) between two Object Types.

    Cardinality must be one of: one_to_one, one_to_many, many_to_one, many_to_many.

    Both source_type and target_type must already exist as Object Types.
    Call get_schema first to see what types are available.

    Returns the created Link Type definition.
    """
    actions = _get_actions()
    try:
        return actions.create_link_type(
            name=name,
            source_type=source_type,
            target_type=target_type,
            cardinality=cardinality,
            description=description,
            source="mcp_client",
            agent="mcp_client",
        )
    except ActionError as e:
        raise ValueError(str(e))


@mcp.tool()
def get_schema() -> dict[str, Any]:
    """
    Return all Object Types and Link Types currently defined in the ontology.

    Call this first to understand what types exist before creating objects
    or links.  Returns a dict with 'object_types' and 'link_types' lists.
    """
    return _get_query().get_schema()


# ==========================================================================
# Write tools
# ==========================================================================

@mcp.tool()
def upsert_object(
    object_type: str,
    properties: dict[str, Any],
    source: str,
    agent: str = "mcp_client",
    confidence: float | None = None,
    id: str | None = None,
) -> dict[str, Any]:
    """
    Create or update an object instance.

    If 'id' is provided and matches an existing object, its properties
    are updated.  Otherwise a new object is created with a generated UUID.

    Properties are validated against the Object Type's declared property
    schema — wrong types, missing required fields, or unknown fields
    will cause an error.

    Parameters:
      - object_type: must match an existing Object Type name
      - properties: dict of property values matching the type's schema
      - source: where this data came from (file path, URL, document id)
      - agent: who/what made this write (default "mcp_client")
      - confidence: optional 0-1 confidence score
      - id: optional UUID for updating an existing object

    Returns the object's id, type, and whether it was created or updated.
    """
    actions = _get_actions()
    try:
        return actions.upsert_object(
            object_type=object_type,
            properties=properties,
            source=source,
            agent=agent,
            confidence=confidence,
            id=id,
        )
    except ActionError as e:
        raise ValueError(str(e))


@mcp.tool()
def upsert_link(
    link_type: str,
    source_object_id: str,
    target_object_id: str,
    source: str,
    agent: str = "mcp_client",
    properties: dict[str, Any] | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """
    Create a link between two objects.

    Validates that:
      - The link type exists
      - Both objects exist and are not deleted
      - The source object's type matches the link type's source_type
      - The target object's type matches the link type's target_type
      - Cardinality constraints are respected

    Parameters:
      - link_type: must match an existing Link Type name
      - source_object_id: UUID of the source object
      - target_object_id: UUID of the target object
      - source: where this relationship was found (document id, etc.)
      - agent: who/what made this write
      - properties: optional extra properties on the link (e.g. date range)
      - confidence: optional 0-1 confidence score

    Returns the link's id, type, and confirmation.
    """
    actions = _get_actions()
    try:
        return actions.upsert_link(
            link_type=link_type,
            source_object_id=source_object_id,
            target_object_id=target_object_id,
            source=source,
            agent=agent,
            properties=properties,
            confidence=confidence,
        )
    except ActionError as e:
        raise ValueError(str(e))


@mcp.tool()
def delete_object(
    id: str,
    agent: str = "mcp_client",
    reason: str = "",
) -> dict[str, Any]:
    """
    Soft-delete an object by its UUID.

    The object is marked as deleted but its data and provenance history
    are preserved.  Deleted objects are hidden from normal queries.

    Parameters:
      - id: UUID of the object to delete
      - agent: who is performing the deletion
      - reason: why the object is being deleted (recorded in provenance)

    Returns confirmation of deletion.
    """
    actions = _get_actions()
    try:
        return actions.delete_object(id=id, agent=agent, reason=reason)
    except ActionError as e:
        raise ValueError(str(e))


@mcp.tool()
def delete_link(
    id: str,
    agent: str = "mcp_client",
    reason: str = "",
) -> dict[str, Any]:
    """
    Soft-delete a link by its UUID.

    The link is marked as deleted but its data and provenance history
    are preserved.  Deleted links are hidden from normal queries.

    Parameters:
      - id: UUID of the link to delete
      - agent: who is performing the deletion
      - reason: why the link is being deleted (recorded in provenance)

    Returns confirmation of deletion.
    """
    actions = _get_actions()
    try:
        return actions.delete_link(id=id, agent=agent, reason=reason)
    except ActionError as e:
        raise ValueError(str(e))


# ==========================================================================
# Read tools
# ==========================================================================

@mcp.tool()
def get_object(id: str) -> dict[str, Any] | None:
    """
    Retrieve an object by its UUID, including properties and full provenance
    history (who wrote it, when, from what source, via which action).

    Returns None if the object does not exist or has been deleted.
    """
    return _get_query().get_object(id)


@mcp.tool()
def find_objects(
    object_type: str | None = None,
    property_filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Find objects matching the given type and/or property filters.

    Parameters:
      - object_type: filter by Object Type name (e.g. "Person")
      - property_filters: dict of property_name -> value to match exactly

    Returns a list of matching objects.  Each object includes id,
    object_type, properties, and timestamps (but not provenance — use
    get_object for that).
    """
    return _get_query().find_objects(
        object_type=object_type,
        property_filters=property_filters,
    )


@mcp.tool()
def get_neighbors(
    object_id: str,
    link_type: str | None = None,
    direction: str = "both",
) -> list[dict[str, Any]]:
    """
    Return objects directly linked to the given object, one hop out.

    Parameters:
      - object_id: UUID of the source object
      - link_type: optional filter by link type name
      - direction: "out" (only outgoing), "in" (only incoming), or "both" (default)

    Returns a list of dicts, each containing the neighbor object, the link
    type, link id, and direction ("in" or "out").
    """
    return _get_query().get_neighbors(
        object_id=object_id,
        link_type=link_type,
        direction=direction,
    )


@mcp.tool()
def get_provenance(id: str) -> list[dict[str, Any]]:
    """
    Return the full write history for an object or link.

    Each record includes: source, agent, action_type, timestamp, and
    optionally confidence and previous_properties (for updates).

    Use this to trace where any piece of data came from.
    """
    return _get_query().get_provenance(id)


# ==========================================================================
# Entry point
# ==========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ontology Harness MCP server",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite database file (default: ONTOLOGY_DB env or ontology.db)",
    )
    args = parser.parse_args()

    if args.db:
        global _DB_PATH
        _DB_PATH = args.db

    db_path = _get_db_path()
    print(f"Ontology Harness MCP server starting with database: {db_path}", file=sys.stderr)

    mcp.run()


if __name__ == "__main__":
    main()
