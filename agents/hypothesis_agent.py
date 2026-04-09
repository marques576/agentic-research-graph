"""
HypothesisAgent – generates candidate relationship hypotheses from accumulated evidence.

Responsibilities
----------------
1. Summarise the entities, relationships, and evidence discovered so far.
2. Ask the LLM to propose hidden-relationship hypotheses that are NOT already
   present as known edges in the graph.
3. Parse the LLM output into Hypothesis objects.
4. Deduplicate against existing hypotheses and store new ones in shared memory.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.base_agent import BaseAgent
from memory.memory import Hypothesis


class HypothesisAgent(BaseAgent):
    """
    Generates candidate hypotheses about hidden relationships in the knowledge graph.

    After a run() call, ``memory.hypotheses`` may be extended with new
    :class:`~memory.memory.Hypothesis` objects whose confidence scores reflect
    the LLM's initial assessment.

    Parameters
    ----------
    memory : AgentMemory
    tool_registry : ToolRegistry | None
    llm : LLM | None
    max_hypotheses : int
        Maximum number of hypotheses to generate per run.  Default 3.
    min_entities : int
        Minimum entity count required before generation is attempted.  Default 2.
    """

    name = "hypothesis"

    def __init__(
        self,
        memory: Any,
        tool_registry: Any = None,
        llm: Any = None,
        *,
        max_hypotheses: int = 3,
        min_entities: int = 2,
    ) -> None:
        super().__init__(memory=memory, tool_registry=tool_registry, llm=llm)
        self.max_hypotheses = max_hypotheses
        self.min_entities = min_entities

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, goal: str = "", **kwargs: Any) -> dict[str, Any]:
        """
        Generate hypotheses from the current memory state.

        Parameters
        ----------
        goal : str
            The research goal to focus hypotheses on.

        Returns
        -------
        dict with keys:
          ``hypotheses_generated`` (int) and ``hypotheses`` (list of dicts).
        """
        self.memory.log_step("hypothesis_start", {"goal": goal})

        entity_names = self.memory.all_entity_names()
        if len(entity_names) < self.min_entities:
            self.memory.log_step(
                "hypothesis_skipped",
                {"reason": f"Only {len(entity_names)} entities — need ≥ {self.min_entities}"},
            )
            return {"hypotheses_generated": 0, "hypotheses": []}

        context = self._build_context(goal)
        new_hypotheses = self._generate_hypotheses(context, goal)

        # Deduplicate against existing hypotheses and store
        stored: list[Hypothesis] = []
        existing_statements = {h.statement.lower() for h in self.memory.hypotheses}
        for hyp in new_hypotheses:
            if hyp.statement.lower() not in existing_statements:
                self.memory.hypotheses.append(hyp)
                existing_statements.add(hyp.statement.lower())
                stored.append(hyp)
                self.memory.log_step(
                    "hypothesis_generated",
                    {
                        "statement": hyp.statement,
                        "entities": hyp.entities_involved,
                        "confidence": hyp.confidence,
                    },
                )

        result = {
            "hypotheses_generated": len(stored),
            "hypotheses": [h.to_dict() for h in stored],
        }
        self.memory.log_step("hypothesis_done", result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(self, goal: str) -> str:
        """Build a compact, LLM-readable summary of the current memory state."""
        entity_names = self.memory.all_entity_names()[:50]

        _TRIVIAL_REL_TYPES = frozenset({
            "co-occurs-with", "co_occurs_with", "co-occurrence",
            "related_to", "related-to", "mentions", "mentioned_in",
        })

        all_rels = sorted(
            self.memory.relationships, key=lambda r: r.confidence, reverse=True
        )

        # Prefer specific typed relationships; fall back to trivial only if nothing else
        typed_rels = [r for r in all_rels if r.relation_type not in _TRIVIAL_REL_TYPES]
        fallback_rels = [r for r in all_rels if r.relation_type in _TRIVIAL_REL_TYPES]
        display_rels = typed_rels[:40] if typed_rels else fallback_rels[:20]

        rel_lines = [
            f"  {r.source} --[{r.relation_type}]--> {r.target} (conf: {r.confidence:.2f})"
            + (f"  [evidence: {r.evidence[:120]}]" if r.evidence else "")
            for r in display_rels
        ]

        # Inferred edges from graph explorer
        graph_rels = self.memory.graph.all_relationships()
        inferred_lines = [
            f"  {r['source']} --[inferred]--> {r['target']}"
            for r in graph_rels
            if r.get("relation_type") == "inferred_connection"
        ][:10]

        # All evidence snippets (not capped — give the LLM everything)
        evidence_parts = [
            f"  [{i+1}] {ev[:300]}"
            for i, ev in enumerate(self.memory.evidence)
        ]

        parts: list[str] = []
        if goal:
            parts.append(f"RESEARCH GOAL: {goal}")
        parts.append(
            f"\nKNOWN ENTITIES ({len(entity_names)} shown):\n  "
            + ", ".join(entity_names)
        )
        if rel_lines:
            parts.append("\nKNOWN RELATIONSHIPS (typed, highest confidence first):\n" + "\n".join(rel_lines))
        if inferred_lines:
            parts.append("\nINFERRED GRAPH CONNECTIONS:\n" + "\n".join(inferred_lines))
        if evidence_parts:
            parts.append(f"\nEVIDENCE SNIPPETS (all {len(evidence_parts)}):\n" + "\n".join(evidence_parts))

        return "\n".join(parts)

    def _generate_hypotheses(self, context: str, goal: str) -> list[Hypothesis]:
        """Call the LLM and parse its response into Hypothesis objects."""
        if self.llm is None:
            return []

        prompt = (
            f"{context}\n\n"
            f"Based on the entities, relationships, and evidence above, propose up to "
            f"{self.max_hypotheses} specific, testable hypotheses about HIDDEN or "
            f"NON-OBVIOUS relationships in this knowledge graph.\n"
            f"Focus on indirect connections, surprising patterns, and latent structures "
            f"that are NOT already listed as known relationships above.\n\n"
            f"For each hypothesis return a JSON object with:\n"
            f'  "statement"  : one clear sentence describing the proposed relationship\n'
            f'  "entities"   : list of entity names directly involved\n'
            f'  "confidence" : initial confidence score 0.0–1.0\n'
            f'  "type"       : e.g. "structural", "causal", "temporal", "hierarchical"\n\n'
            f"Return either a single JSON object or a JSON array of objects.\n"
            f"JSON:"
        )

        try:
            raw = self.llm.generate(prompt)
            raw = re.sub(r"```(?:json)?\n?", "", raw)
            raw = re.sub(r"```", "", raw).strip()
            return self._parse_hypotheses(raw)
        except Exception as exc:
            self.memory.log_step("hypothesis_llm_error", str(exc))
            return []

    def _parse_hypotheses(self, raw: str) -> list[Hypothesis]:
        """
        Parse LLM output into Hypothesis objects.

        Handles single-object, array, and noisy-text responses via multiple
        JSON extraction strategies.
        """
        parsed = self._extract_json(raw)
        if parsed is None:
            return []

        items: list[Any] = [parsed] if isinstance(parsed, dict) else (
            parsed if isinstance(parsed, list) else []
        )

        hypotheses: list[Hypothesis] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", "")).strip()
            if not statement:
                continue
            hyp = Hypothesis(
                statement=statement,
                entities_involved=list(
                    item.get("entities", item.get("entities_involved", []))
                ),
                confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                hypothesis_type=str(
                    item.get("type", item.get("hypothesis_type", "structural_relationship"))
                ),
            )
            hypotheses.append(hyp)

        return hypotheses[: self.max_hypotheses]

    def _extract_json(self, text: str) -> Any:
        """
        Robustly extract a JSON value from potentially noisy LLM text.

        Tries four strategies in order:
        1. Direct parse (LLM returned clean JSON)
        2. Find outermost array  [ … ]
        3. Find outermost object { … }
        4. Find a named array inside an object  {"hypotheses": [ … ]}
        """
        # Strategy 1: direct
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: outermost array
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: outermost object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 4: named nested array
        match = re.search(
            r'"(?:hypotheses|items|results|data)"\s*:\s*(\[.*?\])', text, re.DOTALL
        )
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        return None
