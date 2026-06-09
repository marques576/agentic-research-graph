"""OQL — Ontology Query Language and data-access layer.

Provides a query engine on top of an ``Ontology`` schema and (optionally)
an instance store (*KnowledgeGraph*).  Supports type-based queries, full-text
fuzzy search, graph traversal, and schema inspection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ontology_runtime.ontology_model import Ontology


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """The result of an ontology query.

    Attributes
    ----------
    results : list[dict[str, Any]]
        The matching items.
    total_count : int
        Total number of matches (may exceed ``len(results)`` if paging
        is used, but in the simple implementation they are equal).
    query_time_ms : float
        Wall-clock execution time in milliseconds.
    """

    results: list[dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    query_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# OntologyQuery (OQL)
# ---------------------------------------------------------------------------


class OntologyQuery:
    """OQL query engine for ontology schemas and instance data.

    The query engine works against two data sources:

    * **Schema** — an ``Ontology`` instance from ``ontology_model``.
    * **Instance store** — an optional ``KnowledgeGraph``-like container
      that holds entity instances and their relationships.

    Parameters
    ----------
    schema : Ontology
        The ontology schema (``ontology_model.Ontology`` instance).
    instance_store : Any
        Optional graph / instance storage (e.g. ``KnowledgeGraph``
        instance).  Must support ``all_entities()``,
        ``get_entity(name)``, ``all_relationships()``,
        ``get_relationships(node)``, and ``neighbors(node)`` if used.
    """

    def __init__(self, schema: Ontology, instance_store: Any = None) -> None:
        self._schema = schema
        self._store = instance_store

    # ------------------------------------------------------------------
    # Schema queries
    # ------------------------------------------------------------------

    def query(
        self,
        object_type: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Query ontology objects (object types) matching *object_type*.

        If *object_type* is provided, only that type is returned.  Filters
        are applied as simple key-value matches on the object metadata.

        Parameters
        ----------
        object_type : str | None
            Canonical object type name to filter by.
        filters : dict[str, Any] | None
            Optional key-value pairs that must match object attributes.

        Returns
        -------
        QueryResult
        """
        start = time.perf_counter()
        results: list[dict[str, Any]] = []

        for obj_id, obj in self._schema.objects.items():
            if object_type and obj_id != object_type and obj.name != object_type:
                continue
            obj_info = obj.to_dict()
            obj_info["type"] = "object"
            obj_info.setdefault("description", "")

            # Apply filters
            if filters:
                matches = all(
                    str(obj_info.get(k, "")).lower() == str(v).lower()
                    for k, v in filters.items()
                )
                if not matches:
                    continue
            results.append(obj_info)

        elapsed = (time.perf_counter() - start) * 1000.0
        return QueryResult(
            results=results,
            total_count=len(results),
            query_time_ms=elapsed,
        )

    def search(self, text: str) -> QueryResult:
        """Fuzzy full-text search across ontology objects and relations.

        Matches are found by case-insensitive substring comparison against
        object type names, relation labels, property names, and aliases.

        Parameters
        ----------
        text : str
            Search query string.

        Returns
        -------
        QueryResult
        """
        start = time.perf_counter()
        results: list[dict[str, Any]] = []
        query_lower = text.strip().lower()
        if not query_lower:
            elapsed = (time.perf_counter() - start) * 1000.0
            return QueryResult(results=[], total_count=0, query_time_ms=elapsed)

        seen: set[str] = set()

        # Search object types
        for obj_id, obj in self._schema.objects.items():
            if query_lower in obj_id.lower() or query_lower in obj.name.lower():
                results.append({
                    "id": obj_id,
                    "type": "object",
                    "name": obj.name or obj_id,
                    "match_field": "object_type",
                    "relevance": 1.0,
                })
                seen.add(obj_id)

        # Search relation types
        for rel_id, rel in self._schema.relations.items():
            if query_lower in rel_id.lower() or query_lower in rel.name.lower():
                results.append({
                    "id": rel_id,
                    "type": "relation",
                    "name": rel.name or rel_id,
                    "source_type": rel.source_type,
                    "target_type": rel.target_type,
                    "match_field": "relation_type",
                    "relevance": 0.9,
                })

        # Search property names
        for prop_id, prop in self._schema.properties.items():
            if query_lower in prop_id.lower() or query_lower in prop.name.lower():
                results.append({
                    "id": prop_id,
                    "type": "property",
                    "name": prop.name or prop_id,
                    "datatype": prop.datatype,
                    "match_field": "property",
                    "relevance": 0.8,
                })
                seen.add(prop_id)

        # Search instances (if store available)
        if self._store is not None:
            try:
                entities = self._store.all_entities()
                for ent in entities:
                    name = ent.get("name", "")
                    if query_lower in name.lower() and name not in seen:
                        results.append({
                            "id": name,
                            "type": "entity",
                            "name": name,
                            "entity_type": ent.get("type", ""),
                            "match_field": "entity_name",
                            "relevance": 0.7,
                        })
                        seen.add(name)
            except Exception:
                pass

        elapsed = (time.perf_counter() - start) * 1000.0
        return QueryResult(
            results=results,
            total_count=len(results),
            query_time_ms=elapsed,
        )

    def find_related(
        self,
        object_id: str,
        relation_type: str | None = None,
        max_depth: int = 1,
    ) -> QueryResult:
        """Graph traversal to find entities related to *object_id*.

        Uses the instance store (if available) to perform a BFS traversal
        up to *max_depth* hops.  If no instance store is configured,
        schema-level relation lookups are returned instead.

        Parameters
        ----------
        object_id : str
            The entity or type name to start from.
        relation_type : str | None
            Optional relation type to filter by.
        max_depth : int
            Maximum traversal depth (default: 1, direct neighbours only).

        Returns
        -------
        QueryResult
        """
        start = time.perf_counter()
        results: list[dict[str, Any]] = []

        if self._store is not None:
            # BFS traversal on the instance graph
            visited: set[str] = {object_id}
            current_level = {object_id}
            depth = 0

            while current_level and depth < max_depth:
                next_level: set[str] = set()
                for node in current_level:
                    try:
                        neighbours = self._store.neighbors(node)
                    except Exception:
                        neighbours = []
                    for nb in neighbours:
                        if nb not in visited:
                            if relation_type:
                                try:
                                    rels = self._store.get_relationships(nb)
                                except Exception:
                                    rels = []
                                matching = any(
                                    r.get("relation_type") == relation_type
                                    for r in rels
                                    if r.get("source") == node
                                    or r.get("target") == node
                                )
                                if not matching:
                                    continue
                            visited.add(nb)
                            next_level.add(nb)
                            results.append({
                                "id": nb,
                                "type": "entity",
                                "name": nb,
                                "depth": depth + 1,
                                "relation": relation_type or "any",
                            })
                current_level = next_level
                depth += 1
        else:
            # Schema-level: return relation types for this object type
            for rel_id, rel in self._schema.relations.items():
                if relation_type and rel_id != relation_type and rel.name != relation_type:
                    continue
                if rel.source_type == object_id or rel.target_type == object_id:
                    results.append({
                        "id": rel_id,
                        "type": "relation",
                        "name": rel.name or rel_id,
                        "source_type": rel.source_type,
                        "target_type": rel.target_type,
                        "depth": 0,
                    })

        elapsed = (time.perf_counter() - start) * 1000.0
        return QueryResult(
            results=results,
            total_count=len(results),
            query_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Schema inspection
    # ------------------------------------------------------------------

    def get_object(self, object_id: str) -> dict[str, Any] | None:
        """Return metadata for a single ontology object type.

        Parameters
        ----------
        object_id : str
            The canonical object type identifier or name.

        Returns
        -------
        dict | None
            Object metadata dict, or ``None`` if not found.
        """
        obj = self._schema.objects.get(object_id)
        if obj is None:
            # Try by name
            for o in self._schema.objects.values():
                if o.name == object_id:
                    obj = o
                    break
        return obj.to_dict() if obj else None

    def get_all_objects(self) -> list[dict[str, Any]]:
        """Return all registered ontology object types.

        Returns
        -------
        list[dict]
            Each dict is the serialised form of an ``ObjectType``.
        """
        return [obj.to_dict() for obj in self._schema.objects.values()]

    def get_all_relations(self) -> list[dict[str, Any]]:
        """Return all relation types defined in the ontology.

        Returns
        -------
        list[dict]
            Each dict is the serialised form of a ``RelationType``.
        """
        return [rel.to_dict() for rel in self._schema.relations.values()]

    def get_all_properties(self) -> list[dict[str, Any]]:
        """Return all property types defined in the ontology.

        Returns
        -------
        list[dict]
            Each dict is the serialised form of a ``PropertyType``.
        """
        return [prop.to_dict() for prop in self._schema.properties.values()]
