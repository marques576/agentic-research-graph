"""
GraphExplorerAgent – traverses the knowledge graph to discover multi-hop
relationships that document retrieval alone cannot find.

Uses ``graph_neighbors`` and ``graph_shortest_path`` tools to find structural
connections between entities, then prompts the LLM to interpret indirect paths
and writes inferred edges directly back to the graph.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


class GraphExplorerAgent(BaseAgent):
    """
    Explores the knowledge graph to discover multi-hop hidden relationships.

    Uses graph_neighbors and graph_shortest_path tools to find structural
    connections between entities that may not be obvious from documents alone.
    Inferred edges are written directly to the graph with relation_type
    ``"inferred_connection"``.
    """

    name = "graph_explorer"

    def run(self, goal: str = "", **kwargs: Any) -> dict[str, Any]:
        """
        Explore the graph and return discovered paths and structural insights.

        Parameters
        ----------
        goal : str
            The original research goal, used to focus exploration.

        Returns
        -------
        dict with keys:
            explored_nodes   : list of node names visited
            discovered_paths : list of {"source", "target", "path", "interpretation"} dicts
        """
        self.memory.log_step("graph_explorer_start", {"goal": goal})

        explored_nodes: list[str] = []
        discovered_paths: list[dict[str, Any]] = []

        # --- 1. Get all entity names ---
        all_entities = list(self.memory.entities.values())
        entity_names = list({e.name for e in all_entities})

        if len(entity_names) < 2:
            self.memory.log_step("graph_explorer_skip", "fewer than 2 entities — skipping")
            return {
                "explored_nodes": explored_nodes,
                "discovered_paths": discovered_paths,
            }

        # --- 2. Limit to top-10 most-connected nodes ---
        top_nodes = self.memory.graph.top_nodes_by_degree(n=10)
        focus_names = [n for n, _ in top_nodes] if top_nodes else entity_names[:10]

        # --- 3. Explore pairs ---
        max_pairs = 20
        pairs_checked = 0

        for i, source in enumerate(focus_names):
            for target in focus_names[i + 1:]:
                if pairs_checked >= max_pairs:
                    break
                pairs_checked += 1

                path_result = self._use_tool(
                    "graph_shortest_path",
                    {"source": source, "target": target},
                )

                path = path_result.get("path", [])
                length = path_result.get("length", -1)

                # Only interested in indirect connections (length 2–4)
                if 2 <= length <= 4:
                    explored_nodes.extend(path)

                    interpretation = self._interpret_path(path, goal)
                    discovered_paths.append({
                        "source": source,
                        "target": target,
                        "path": path,
                        "interpretation": interpretation,
                    })

                    # Write the inferred connection directly to the graph so it
                    # persists across iterations and appears in the exported graph.
                    self.memory.add_graph_relationship(
                        source=path[0],
                        target=path[-1],
                        relation_type="inferred_connection",
                        confidence=0.45,
                        evidence=f"Multi-hop path: {' -> '.join(path)}",
                    )

                    self.memory.log_step(
                        "graph_exploration",
                        {
                            "path": path,
                            "interpretation": interpretation,
                            "length": length,
                        },
                    )

            if pairs_checked >= max_pairs:
                break

        # Deduplicate explored_nodes
        seen: set[str] = set()
        explored_nodes = [n for n in explored_nodes if not (n in seen or seen.add(n))]  # type: ignore[func-returns-value]

        result = {
            "explored_nodes": explored_nodes,
            "discovered_paths": discovered_paths,
        }
        self.memory.log_step("graph_explorer_done", result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _interpret_path(self, path: list[str], goal: str) -> str:
        """
        Ask the LLM to interpret what a graph path implies.

        Parameters
        ----------
        path : list of str
            Ordered sequence of node names (e.g. [A, B, C]).
        goal : str
            Original research goal for context.

        Returns
        -------
        str — a one-sentence interpretation of the hidden relationship,
        or a plain description if the LLM is unavailable.
        """
        if self.llm is None or not path:
            return ""

        path_str = " -> ".join(path)
        prompt = (
            f"You are a knowledge graph analyst.\n\n"
            f"Research goal: {goal}\n\n"
            f"Given this entity path in the knowledge graph: {path_str}\n\n"
            f"What hidden relationship might exist between "
            f"'{path[0]}' and '{path[-1]}'? "
            f"Answer in one concise sentence."
        )
        try:
            interpretation = self.llm.generate(prompt)
            return interpretation.strip().split("\n")[0][:300]
        except Exception:
            return f"Indirect connection found between '{path[0]}' and '{path[-1]}' via {path_str}."
