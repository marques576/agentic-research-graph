"""
NetworkX compiler — converts an Ontology into a NetworkX DiGraph.

NetworkX is an optional dependency.  If unavailable, the compiler raises
a RuntimeError at compile time.
"""

from __future__ import annotations

import time

try:
    import networkx as nx

    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    nx = None  # type: ignore[assignment]

from .compiler import CompilationResult, Compiler
from ..ontology_model.types import Ontology


class NetworkXCompiler(Compiler):
    """Compiles an Ontology into a ``networkx.DiGraph``.

    Each ObjectType becomes a node (keyed by id) with attributes
    ``name``, ``description``, ``properties``, and ``parent_id``.

    Each RelationType becomes a directed edge with attributes
    ``name``, ``id``, and ``properties``.

    Parameters
    ----------
    name : str
        Optional label for this compiler instance.
    """

    def __init__(self, name: str = "") -> None:
        super().__init__(name=name or "NetworkXCompiler")

    def compile(self, ontology: Ontology) -> CompilationResult:
        """Convert *ontology* to a ``networkx.DiGraph``.

        Parameters
        ----------
        ontology : Ontology

        Returns
        -------
        CompilationResult
            ``.output`` is a ``nx.DiGraph``.

        Raises
        ------
        RuntimeError
            If networkx is not installed.
        """
        if not _NX_AVAILABLE:
            raise RuntimeError(
                "networkx is required for NetworkXCompiler. "
                "Install it with: pip install networkx"
            )

        start = time.perf_counter()
        graph = nx.DiGraph()

        for obj_id, obj in ontology.objects.items():
            graph.add_node(
                obj_id,
                name=obj.name,
                description=obj.description,
                parent_id=obj.parent_id,
                properties={p.name: p.default for p in obj.properties},
            )

        for rel_id, rel in ontology.relations.items():
            graph.add_edge(
                rel.source_type,
                rel.target_type,
                name=rel.name,
                id=rel_id,
                properties={p.name: p.default for p in rel.properties},
            )

        elapsed = (time.perf_counter() - start) * 1000
        return CompilationResult(
            output=graph,
            format_name="networkx",
            compilation_time_ms=elapsed,
            metadata={
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
            },
        )
