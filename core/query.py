"""
Read API for the ontology: get object, neighbors, traverse, filter by type.

Built on top of the store's low-level query methods.  This is the primary
read interface that agents and tools will use.
"""

from __future__ import annotations

from typing import Any

from .store import OntologyStore


class QueryAPI:
    """
    Read-side interface for the ontology store.

    Wraps the store with convenience methods and multi-hop traversal.
    All methods return plain dicts/lists suitable for serialisation.
    """

    def __init__(self, store: OntologyStore) -> None:
        self.store = store

    # ------------------------------------------------------------------
    # Object queries
    # ------------------------------------------------------------------

    def get_object(self, object_id: str) -> dict[str, Any] | None:
        """
        Retrieve an object by its UUID, including properties and provenance.

        Returns None if the object does not exist or has been deleted.
        """
        obj = self.store.get_object(object_id)
        if obj is None:
            return None
        obj["provenance"] = self.store.get_provenance(object_id, target_type="object")
        return obj

    def find_objects(
        self,
        object_type: str | None = None,
        property_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find objects matching the given type and/or property filters.

        Returns a list of object dicts (without provenance for efficiency;
        use get_object() per-object to fetch provenance).
        """
        return self.store.find_objects(
            object_type=object_type,
            property_filters=property_filters,
        )

    # ------------------------------------------------------------------
    # Link queries
    # ------------------------------------------------------------------

    def get_link(self, link_id: str) -> dict[str, Any] | None:
        """
        Retrieve a link by its UUID, including provenance.

        Returns None if the link does not exist or has been deleted.
        """
        link = self.store.get_link(link_id)
        if link is None:
            return None
        link["provenance"] = self.store.get_provenance(link_id, target_type="link")
        return link

    # ------------------------------------------------------------------
    # Neighbor queries
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        object_id: str,
        link_type: str | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """
        Return objects directly linked to the given object, one hop out.

        Parameters
        ----------
        object_id : str
            UUID of the source object.
        link_type : str, optional
            Filter by link type name.
        direction : str
            "out" (only outgoing links), "in" (only incoming),
            or "both" (default).

        Returns
        -------
        List of dicts with keys: neighbor (object dict), link_type,
        link_id, direction ("in"|"out").
        """
        obj = self.store.get_object(object_id)
        if obj is None:
            return []

        results: list[dict[str, Any]] = []

        if direction in ("out", "both"):
            outgoing = self.store.find_links(
                source_object_id=object_id, link_type=link_type
            )
            for link in outgoing:
                neighbor = self.store.get_object(link["target_object_id"])
                if neighbor:
                    results.append({
                        "neighbor": neighbor,
                        "link_type": link["link_type"],
                        "link_id": link["id"],
                        "direction": "out",
                    })

        if direction in ("in", "both"):
            incoming = self.store.find_links(
                target_object_id=object_id, link_type=link_type
            )
            for link in incoming:
                neighbor = self.store.get_object(link["source_object_id"])
                if neighbor:
                    results.append({
                        "neighbor": neighbor,
                        "link_type": link["link_type"],
                        "link_id": link["id"],
                        "direction": "in",
                    })

        return results

    # ------------------------------------------------------------------
    # Multi-hop traversal
    # ------------------------------------------------------------------

    def traverse(
        self,
        start_object_id: str,
        max_depth: int = 3,
        link_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Breadth-first traversal from a starting object, up to max_depth hops.

        Parameters
        ----------
        start_object_id : str
            UUID of the starting object.
        max_depth : int
            Maximum number of hops (1–3).  Default 3.
        link_type : str, optional
            Filter by link type name at every hop.

        Returns
        -------
        Dict with:
            start : the starting object dict
            paths : list of path dicts, each with:
                - path : list of object dicts (start to end)
                - edges : list of link dicts connecting them
                - depth : number of hops
        """
        if max_depth < 1:
            max_depth = 1
        if max_depth > 3:
            max_depth = 3

        start_obj = self.store.get_object(start_object_id)
        if start_obj is None:
            return {"start": None, "paths": []}

        paths: list[dict[str, Any]] = []

        current_layer = [(start_obj, [], [])]

        for depth in range(1, max_depth + 1):
            next_layer: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = []

            for obj, obj_path, edge_path in current_layer:
                neighbors = self.get_neighbors(
                    obj["id"], link_type=link_type, direction="both"
                )

                for n in neighbors:
                    neighbor_obj = n["neighbor"]
                    nid = neighbor_obj["id"]

                    path_ids = {o["id"] for o in obj_path} | {obj["id"]}
                    if nid in path_ids:
                        continue

                    new_obj_path = list(obj_path) + [obj]
                    new_edge_path = list(edge_path) + [{
                        "link_type": n["link_type"],
                        "link_id": n["link_id"],
                        "direction": n["direction"],
                    }]

                    paths.append({
                        "path": new_obj_path + [neighbor_obj],
                        "edges": new_edge_path,
                        "depth": depth,
                    })

                    if depth < max_depth:
                        next_layer.append((neighbor_obj, new_obj_path, new_edge_path))

            current_layer = next_layer

        return {
            "start": start_obj,
            "paths": paths,
        }

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def get_provenance(self, target_id: str) -> list[dict[str, Any]]:
        """
        Return the full write history for an object or link.

        Returns a list of provenance records, ordered by timestamp ascending.
        Each record includes source, agent, action_type, timestamp, and
        optional confidence and previous_properties.
        """
        return self.store.get_provenance(target_id)

    # ------------------------------------------------------------------
    # Schema queries
    # ------------------------------------------------------------------

    def get_schema(self) -> dict[str, Any]:
        """
        Return all Object Types and Link Types currently defined.

        Agents should call this first to understand what exists.
        """
        obj_types = []
        for ot in self.store.get_all_object_types():
            obj_types.append({
                "name": ot.name,
                "properties": [p.to_dict() for p in ot.properties],
                "description": ot.description,
            })

        link_types = []
        for lt in self.store.get_all_link_types():
            link_types.append(lt.to_dict())

        return {
            "object_types": obj_types,
            "link_types": link_types,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of what's in the store."""
        return {
            "object_count": self.store.count_objects(),
            "link_count": self.store.count_links(),
            "object_type_count": len(self.store.get_all_object_types()),
            "link_type_count": len(self.store.get_all_link_types()),
        }
