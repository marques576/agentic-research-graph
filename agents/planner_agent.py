"""
PlannerAgent – breaks a high-level research goal into an ordered task list.

The planner uses the LLM to decompose the goal into concrete steps that
the controller can execute sequentially.  Each step specifies which tool
to invoke and with what query.

The output is a list of task dicts:
    [
        {"step": 1, "action": "vector_search", "query": "..."},
        {"step": 2, "action": "read_document",  "query": "..."},
        ...
    ]
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.base_agent import BaseAgent


# Default task plan used when LLM output cannot be parsed.
# Uses only vector_search / extract_entities / summarize — no hardcoded doc IDs.
_DEFAULT_PLAN = [
    {"step": 1, "action": "vector_search",    "query": "{goal}"},
    {"step": 2, "action": "extract_entities", "query": "from_document"},
    {"step": 3, "action": "vector_search",    "query": "key entities and their relationships"},
    {"step": 4, "action": "extract_entities", "query": "from_last_document"},
    {"step": 5, "action": "vector_search",    "query": "connections patterns structure"},
    {"step": 6, "action": "extract_entities", "query": "from_last_document"},
    {"step": 7, "action": "summarize",        "query": "from_last_document"},
]


class PlannerAgent(BaseAgent):
    """
    Decomposes the user goal into an ordered list of tool-call tasks.

    Strategy
    --------
    1. Ask the LLM to generate a structured JSON plan.
    2. Parse the JSON; fall back to the hard-coded default plan if parsing fails.
    3. Substitute the goal text into any placeholder queries.
    4. Store the plan in memory and return it.
    """

    name = "planner"

    def run(self, goal: str = "", **kwargs: Any) -> dict[str, Any]:
        """
        Parameters
        ----------
        goal : str
            The high-level research question.

        Returns
        -------
        dict with key "plan" (list of task dicts) and "goal".
        """
        self.memory.log_step("planner_start", {"goal": goal})

        prompt = (
            f"You are a research planning agent.\n"
            f"Goal: {goal}\n\n"
            f"Break this goal into 8–12 sequential research steps.\n"
            f"Each step must specify:\n"
            f"  - step (int)\n"
            f"  - action (one of: vector_search, read_document, extract_entities,\n"
            f"    summarize)\n"
            f"  - query (string)\n\n"
            f"Return ONLY a JSON array of step objects.  "
            f"Focus on discovering relationships between entities in the documents.\n"
            f"PLAN:"
        )

        raw = self.llm.generate(prompt) if self.llm else ""
        plan = self._parse_plan(raw, goal)

        # Substitute goal placeholder
        for task in plan:
            if isinstance(task.get("query"), str):
                task["query"] = task["query"].replace("{goal}", goal)

        self.memory.log_step("plan_created", {"steps": len(plan), "plan": plan})
        return {"goal": goal, "plan": plan}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_plan(self, raw: str, goal: str) -> list[dict[str, Any]]:
        """Try to extract a JSON list from the LLM output; use default on failure."""
        # strip markdown code fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        # attempt to locate a JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                plan = json.loads(match.group())
                if isinstance(plan, list) and len(plan) > 0:
                    return plan
            except json.JSONDecodeError:
                pass

        # fall back to the hard-coded default
        default = [dict(step) for step in _DEFAULT_PLAN]
        for task in default:
            if isinstance(task.get("query"), str):
                task["query"] = task["query"].replace("{goal}", goal)
        return default
