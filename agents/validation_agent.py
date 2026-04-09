"""
ValidationAgent – validates hypotheses against accumulated evidence and updates confidence.

Responsibilities
----------------
1. Select the highest-priority unvalidated hypotheses from memory.
2. Retrieve evidence snippets and graph relationships relevant to each hypothesis.
3. Ask the LLM to assess how strongly the evidence supports or refutes the hypothesis.
4. Apply the LLM's confidence delta and store new supporting/refuting evidence.
5. Signal to the controller whether the loop should terminate
   (best hypothesis confidence >= threshold).
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.base_agent import BaseAgent
from memory.memory import Hypothesis


class ValidationAgent(BaseAgent):
    """
    Validates hypotheses by scoring them against accumulated evidence.

    After a run() call each processed hypothesis has:
    - an updated ``confidence`` score
    - ``supporting_evidence`` / ``refuting_evidence`` lists extended
    - ``validated`` flag set to True

    The returned dict's ``should_stop`` key tells the controller whether
    the confidence threshold has been reached and the loop may exit.

    Parameters
    ----------
    memory : AgentMemory
    tool_registry : ToolRegistry | None
    llm : LLM | None
    confidence_threshold : float
        Confidence level at which a hypothesis is considered confirmed.
        Default 0.75.
    max_per_run : int
        Maximum number of hypotheses to validate in a single run().  Default 2.
    """

    name = "validator"

    def __init__(
        self,
        memory: Any,
        tool_registry: Any = None,
        llm: Any = None,
        *,
        confidence_threshold: float = 0.75,
        max_per_run: int = 2,
    ) -> None:
        super().__init__(memory=memory, tool_registry=tool_registry, llm=llm)
        self.confidence_threshold = confidence_threshold
        self.max_per_run = max_per_run

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Validate pending hypotheses and update their confidence scores.

        Returns
        -------
        dict with keys:
          ``validated_count``  : int   — how many hypotheses were processed
          ``should_stop``      : bool  — True if best confidence >= threshold
          ``best_confidence``  : float — highest confidence across all hypotheses
          ``best_hypothesis``  : dict | None — the top-ranked hypothesis
          ``verdicts``         : list[dict] — per-hypothesis validation details
        """
        self.memory.log_step("validation_start", {})

        if not self.memory.hypotheses:
            return {
                "validated_count": 0,
                "should_stop": False,
                "best_confidence": 0.0,
                "best_hypothesis": None,
                "verdicts": [],
            }

        # Prioritise unvalidated hypotheses; within those, lowest confidence
        # first (most uncertain — benefit most from validation).
        candidates = sorted(
            [h for h in self.memory.hypotheses if not h.validated],
            key=lambda h: h.confidence,
        )

        verdicts: list[dict[str, Any]] = []
        for hyp in candidates[: self.max_per_run]:
            verdict = self._validate_hypothesis(hyp)
            verdicts.append(verdict)
            self.memory.log_step("validation_verdict", verdict)

        # Compute best confidence across ALL hypotheses (including previously validated)
        best_confidence = max(
            (h.confidence for h in self.memory.hypotheses), default=0.0
        )
        best_hyp: Hypothesis | None = max(
            self.memory.hypotheses,
            key=lambda h: h.confidence,
            default=None,  # type: ignore[call-overload]
        )
        should_stop = best_confidence >= self.confidence_threshold

        result: dict[str, Any] = {
            "validated_count": len(verdicts),
            "should_stop": should_stop,
            "best_confidence": best_confidence,
            "best_hypothesis": best_hyp.to_dict() if best_hyp else None,
            "verdicts": verdicts,
        }
        self.memory.log_step("validation_done", result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_hypothesis(self, hyp: Hypothesis) -> dict[str, Any]:
        """
        Run LLM validation on a single hypothesis.

        Updates the hypothesis in-place: adjusts confidence, extends
        evidence lists, and sets ``validated = True``.
        Returns a verdict dict for logging / reporting.
        """
        evidence_text = self._collect_evidence(hyp)
        relationships_text = self._relevant_relationships(hyp)
        confidence_before = hyp.confidence

        if self.llm is None:
            hyp.validated = True
            return {
                "statement": hyp.statement,
                "verdict": "UNKNOWN",
                "confidence_before": confidence_before,
                "confidence_after": hyp.confidence,
                "reasoning": "No LLM available for validation.",
            }

        prompt = (
            "You are a research analyst validating a hypothesis against evidence.\n\n"
            f"HYPOTHESIS:\n{hyp.statement}\n\n"
            f"ENTITIES INVOLVED: {', '.join(hyp.entities_involved)}\n\n"
            f"AVAILABLE EVIDENCE:\n{evidence_text}\n\n"
            f"RELEVANT GRAPH RELATIONSHIPS:\n{relationships_text}\n\n"
            "Assess whether the evidence supports or refutes this hypothesis.\n"
            "Return a JSON object with:\n"
            '  "verdict"          : "SUPPORTED", "REFUTED", or "INCONCLUSIVE"\n'
            '  "confidence_delta" : float in [-0.3, +0.3] — adjustment to current confidence\n'
            '  "reasoning"        : one paragraph explaining your assessment\n'
            '  "new_evidence"     : list of 0–3 specific text snippets most relevant to verdict\n'
            "JSON:"
        )

        try:
            raw = self.llm.generate(prompt)
            raw = re.sub(r"```(?:json)?\n?", "", raw)
            raw = re.sub(r"```", "", raw).strip()
            parsed = self._extract_json(raw)

            if parsed and isinstance(parsed, dict):
                delta = float(parsed.get("confidence_delta", 0.0))
                # Clamp delta to valid range
                delta = max(-0.3, min(0.3, delta))
                verdict_str = str(parsed.get("verdict", "INCONCLUSIVE")).upper()
                reasoning = str(parsed.get("reasoning", ""))
                new_evidence = parsed.get("new_evidence", [])

                # Apply update
                hyp.confidence = max(0.0, min(1.0, hyp.confidence + delta))
                hyp.validated = True

                if isinstance(new_evidence, list):
                    evidence_list = [str(e) for e in new_evidence if e]
                    if verdict_str == "SUPPORTED":
                        hyp.supporting_evidence.extend(evidence_list)
                    elif verdict_str == "REFUTED":
                        hyp.refuting_evidence.extend(evidence_list)

                return {
                    "statement": hyp.statement,
                    "verdict": verdict_str,
                    "confidence_before": confidence_before,
                    "confidence_after": hyp.confidence,
                    "reasoning": reasoning,
                }

        except Exception as exc:
            self.memory.log_step("validation_llm_error", str(exc))

        # Fallback if LLM call or parse failed
        hyp.validated = True
        return {
            "statement": hyp.statement,
            "verdict": "INCONCLUSIVE",
            "confidence_before": confidence_before,
            "confidence_after": hyp.confidence,
            "reasoning": "Validation failed — LLM error or unparseable response.",
        }

    def _collect_evidence(self, hyp: Hypothesis) -> str:
        """
        Build a focused evidence string for the hypothesis.

        Prefers snippets that mention one or more of the hypothesis entities;
        falls back to a general sample when no entity-specific snippets exist.
        """
        entity_set = {e.lower() for e in hyp.entities_involved}
        relevant: list[str] = []
        general: list[str] = []

        for snippet in self.memory.evidence:
            lower = snippet.lower()
            if any(ent in lower for ent in entity_set):
                relevant.append(snippet[:400])
            else:
                general.append(snippet[:400])
            if len(relevant) >= 5:
                break

        chosen = relevant if relevant else general[:3]
        if not chosen:
            return "(No evidence snippets available)"

        return "\n---\n".join(f"[{i+1}] {e}" for i, e in enumerate(chosen))

    def _relevant_relationships(self, hyp: Hypothesis) -> str:
        """Return memory relationships that touch any hypothesis entity."""
        entity_set = {e.lower() for e in hyp.entities_involved}
        lines = [
            f"  {r.source} --[{r.relation_type}]--> {r.target} (conf: {r.confidence:.2f})"
            for r in self.memory.relationships
            if r.source.lower() in entity_set or r.target.lower() in entity_set
        ]
        if not lines:
            return "  (No directly relevant relationships found)"
        return "\n".join(lines[:10])

    def _extract_json(self, text: str) -> Any:
        """Robustly extract a JSON object from potentially noisy LLM text."""
        # Strategy 1: direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: find outermost object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

        return None
