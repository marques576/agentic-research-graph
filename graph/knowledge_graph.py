"""
KnowledgeGraph – NetworkX-backed directed multigraph for entity-relationship storage.

Provides entity (node) and relationship (edge) management with graph traversal
utilities and serialisation support.

Uses ``MultiDiGraph`` so that multiple distinct relationships between the same
pair of entities are preserved rather than silently overwritten.
"""

from __future__ import annotations

import json
from typing import Any

try:
    import networkx as nx  # type: ignore[import]
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    nx = None  # type: ignore[assignment]


class KnowledgeGraph:
    """
    Directed knowledge multigraph backed by NetworkX MultiDiGraph.

    Nodes represent entities; edges represent directed relationships between them.
    Multiple distinct relationship types between the same pair of nodes are
    stored as parallel edges rather than overwriting each other.

    All public methods gracefully handle missing nodes or unavailable networkx.
    """

    def __init__(self) -> None:
        if not _NX_AVAILABLE:
            raise RuntimeError(
                "networkx is required: uv add networkx  (or: pip install networkx)"
            )
        self._graph: Any = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_entity(self, name: str, entity_type: str, attributes: dict = {}) -> None:
        """
        Add or update an entity node.

        Parameters
        ----------
        name : str
            Unique entity name (used as node key).
        entity_type : str
            Category label, e.g. "company", "person", "grant".
        attributes : dict
            Arbitrary extra metadata stored on the node.
        """
        if self._graph.has_node(name):
            # Merge attributes without overwriting existing keys
            existing = self._graph.nodes[name].get("attributes", {})
            for k, v in attributes.items():
                if k not in existing:
                    existing[k] = v
            self._graph.nodes[name]["attributes"] = existing
        else:
            self._graph.add_node(name, type=entity_type, attributes=dict(attributes))

    def get_entity(self, name: str) -> dict | None:
        """
        Retrieve a node's attributes by name.

        Parameters
        ----------
        name : str
            Entity name to look up.

        Returns
        -------
        dict with keys ``name``, ``type``, ``attributes``, or ``None`` if not found.
        """
        if not self._graph.has_node(name):
            return None
        data = self._graph.nodes[name]
        return {
            "name": name,
            "type": data.get("type", "unknown"),
            "attributes": data.get("attributes", {}),
        }

    def all_entities(self) -> list[dict]:
        """
        Return all entity nodes as a list of dicts.

        Returns
        -------
        list of dicts with keys: name, type, attributes
        """
        result = []
        for node, data in self._graph.nodes(data=True):
            result.append({
                "name": node,
                "type": data.get("type", "unknown"),
                "attributes": data.get("attributes", {}),
            })
        return result

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        confidence: float = 1.0,
        evidence: str = "",
    ) -> None:
        """
        Add or update a directed relationship edge.

        If an edge with the **same** ``relation_type`` already exists between
        ``source`` and ``target``, its confidence is upgraded if the new value
        is higher and evidence is appended.

        If an edge with a **different** ``relation_type`` exists (or no edge
        exists at all), a new parallel edge is added — preserving both
        relationships rather than overwriting one with the other.

        Parameters
        ----------
        source : str
            Source entity name (node will be auto-created if missing).
        target : str
            Target entity name (node will be auto-created if missing).
        relation_type : str
            Label for the relationship, e.g. "employs", "funded_by".
        confidence : float
            Strength of the relationship in [0, 1].
        evidence : str
            Text snippet supporting this relationship.
        """
        # Auto-create nodes if they don't exist
        if not self._graph.has_node(source):
            self._graph.add_node(source, type="unknown", attributes={})
        if not self._graph.has_node(target):
            self._graph.add_node(target, type="unknown", attributes={})

        # Search existing parallel edges for the same relation_type
        existing_key: int | None = None
        if self._graph.has_edge(source, target):
            for key, edge_data in self._graph[source][target].items():
                if edge_data.get("relation_type") == relation_type:
                    existing_key = key
                    break

        if existing_key is not None:
            # Update the existing same-type edge
            edge_data = self._graph[source][target][existing_key]
            if confidence > edge_data.get("confidence", 0.0):
                edge_data["confidence"] = confidence
            if evidence:
                existing_ev = edge_data.get("evidence", "")
                edge_data["evidence"] = (
                    (existing_ev + " | " + evidence) if existing_ev else evidence
                )
        else:
            # Add a new parallel edge (different relation_type or first edge)
            self._graph.add_edge(
                source,
                target,
                relation_type=relation_type,
                confidence=confidence,
                evidence=evidence,
            )

    def get_relationships(self, node: str) -> list[dict]:
        """
        Return all edges that touch a given node (both in- and out-edges).

        Parameters
        ----------
        node : str
            Entity name to look up edges for.

        Returns
        -------
        list of dicts with keys: source, target, relation_type, confidence, evidence
        """
        if not self._graph.has_node(node):
            return []

        result = []
        for src, tgt, data in self._graph.out_edges(node, data=True):
            result.append({
                "source": src,
                "target": tgt,
                "relation_type": data.get("relation_type", ""),
                "confidence": data.get("confidence", 0.0),
                "evidence": data.get("evidence", ""),
            })
        for src, tgt, data in self._graph.in_edges(node, data=True):
            result.append({
                "source": src,
                "target": tgt,
                "relation_type": data.get("relation_type", ""),
                "confidence": data.get("confidence", 0.0),
                "evidence": data.get("evidence", ""),
            })
        return result

    def all_relationships(self) -> list[dict]:
        """
        Return every edge in the graph.

        Returns
        -------
        list of dicts with keys: source, target, relation_type, confidence, evidence
        """
        result = []
        for src, tgt, data in self._graph.edges(data=True):
            result.append({
                "source": src,
                "target": tgt,
                "relation_type": data.get("relation_type", ""),
                "confidence": data.get("confidence", 0.0),
                "evidence": data.get("evidence", ""),
            })
        return result

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def neighbors(self, node: str) -> list[str]:
        """
        Return names of all nodes directly connected to the given node.

        Includes both successors (out-edges) and predecessors (in-edges).

        Parameters
        ----------
        node : str
            Entity name.

        Returns
        -------
        Deduplicated list of neighbour node names.
        """
        if not self._graph.has_node(node):
            return []
        successors = list(self._graph.successors(node))
        predecessors = list(self._graph.predecessors(node))
        seen: set[str] = set()
        result = []
        for n in successors + predecessors:
            if n not in seen and n != node:
                seen.add(n)
                result.append(n)
        return result

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        """
        Find the shortest directed path between two nodes.

        Parameters
        ----------
        source : str
            Start entity name.
        target : str
            End entity name.

        Returns
        -------
        Ordered list of node names (inclusive of source and target),
        or ``None`` if no path exists or either node is absent.
        """
        if not self._graph.has_node(source) or not self._graph.has_node(target):
            return None
        try:
            path = nx.shortest_path(self._graph, source=source, target=target)
            return list(path)
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the graph to a plain dict.

        Returns
        -------
        dict with keys ``nodes`` (list of dicts) and ``edges`` (list of dicts).
        """
        nodes = []
        for node, data in self._graph.nodes(data=True):
            nodes.append({
                "name": node,
                "type": data.get("type", "unknown"),
                "attributes": data.get("attributes", {}),
            })

        edges = []
        for src, tgt, data in self._graph.edges(data=True):
            edges.append({
                "source": src,
                "target": tgt,
                "relation_type": data.get("relation_type", ""),
                "confidence": data.get("confidence", 0.0),
                "evidence": data.get("evidence", ""),
            })

        return {"nodes": nodes, "edges": edges}

    def from_dict(self, data: dict) -> None:
        """
        Populate (or extend) the graph from a serialised dict.

        Existing nodes / edges are merged, not replaced.

        Parameters
        ----------
        data : dict
            Must have keys ``nodes`` and ``edges`` matching the format
            produced by :meth:`to_dict`.
        """
        for node_data in data.get("nodes", []):
            name = node_data.get("name", "")
            if name:
                self.add_entity(
                    name=name,
                    entity_type=node_data.get("type", "unknown"),
                    attributes=node_data.get("attributes", {}),
                )

        for edge_data in data.get("edges", []):
            source = edge_data.get("source", "")
            target = edge_data.get("target", "")
            if source and target:
                self.add_relationship(
                    source=source,
                    target=target,
                    relation_type=edge_data.get("relation_type", edge_data.get("relation", "")),
                    confidence=float(edge_data.get("confidence", 1.0)),
                    evidence=edge_data.get("evidence", ""),
                )

    def export_graphml(self, path: str) -> None:
        """
        Export the graph to a GraphML file.

        Because GraphML does not support multi-edges natively, parallel edges
        between the same node pair are merged into a single edge with a
        combined ``relation_type`` label (``"type1 | type2"``).

        Parameters
        ----------
        path : str
            Destination file path (e.g. ``"output.graphml"``).
        """
        export_graph = nx.DiGraph()
        for node, data in self._graph.nodes(data=True):
            attrs = {k: str(v) for k, v in data.get("attributes", {}).items()}
            export_graph.add_node(node, type=data.get("type", "unknown"), **attrs)

        # For parallel edges between the same pair, merge relation types
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for src, tgt, data in self._graph.edges(data=True):
            key = (src, tgt)
            if key not in merged:
                merged[key] = {
                    "relation_type": data.get("relation_type", ""),
                    "confidence": data.get("confidence", 0.0),
                    "evidence": str(data.get("evidence", ""))[:500],
                }
            else:
                existing = merged[key]
                new_rel = data.get("relation_type", "")
                if new_rel and new_rel not in existing["relation_type"]:
                    existing["relation_type"] += f" | {new_rel}"
                new_conf = data.get("confidence", 0.0)
                if new_conf > existing["confidence"]:
                    existing["confidence"] = new_conf

        for (src, tgt), edata in merged.items():
            export_graph.add_edge(
                src,
                tgt,
                relation_type=edata["relation_type"],
                confidence=str(edata["confidence"]),
                evidence=edata["evidence"],
            )
        nx.write_graphml(export_graph, path)

    def load_from_seed(self, path: str) -> None:
        """
        Seed the graph from a JSON file in the ``graph_seed.json`` format.

        Parameters
        ----------
        path : str
            Path to a JSON file with keys ``nodes`` and ``edges``.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return self._graph.number_of_edges()

    def degree(self, node: str) -> int:
        """
        Return the total degree (in + out) of a node.

        Parameters
        ----------
        node : str
            Entity name.

        Returns
        -------
        int — 0 if node is not in the graph.
        """
        if not self._graph.has_node(node):
            return 0
        return self._graph.in_degree(node) + self._graph.out_degree(node)

    def top_nodes_by_degree(self, n: int = 10) -> list[tuple[str, int]]:
        """
        Return the top-n nodes sorted by total degree.

        Parameters
        ----------
        n : int
            Number of nodes to return.

        Returns
        -------
        list of (node_name, degree) tuples, sorted descending by degree.
        """
        degrees = [
            (node, self._graph.in_degree(node) + self._graph.out_degree(node))
            for node in self._graph.nodes()
        ]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:n]
