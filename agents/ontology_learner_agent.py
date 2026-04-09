"""
OntologyLearnerAgent – induces a domain ontology from unstructured documents.

This agent runs **once** before the main research loop.  It reads a sample
of the available documents, asks the LLM to propose:

  1. Entity classes that appear in the corpus.
  2. Valid relationship triples (domain_type, relation_type, range_type).
  3. Type aliases that should be collapsed to a canonical label.

The proposed schema is merged with the seed ontology already encoded in
``DomainOntology``.  The result is persisted to ``ontology.json`` in the
data directory so subsequent runs can reload it without re-running NLP.

When a ``helper_prompt`` is supplied (a one-paragraph domain hint), it is
prepended to every LLM call.  This dramatically improves vocabulary
consistency without hand-wiring the ontology.

If the LLM is unavailable or parsing fails, the agent falls back silently
to the seed ontology — the system always has a working schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from ontology.ontology import DomainOntology


_ENTITY_PROPOSAL_PROMPT = """\
{domain_hint}
You are a knowledge engineering assistant.  Below are text excerpts from a \
document corpus.  Your task is to identify the main entity classes that appear.

TEXT EXCERPTS:
{excerpts}

Return a JSON object with this exact schema:
{{
  "entity_types": ["class1", "class2", ...],
  "type_aliases": {{"variant_label": "canonical_label", ...}},
  "relation_triples": [
    ["domain_type", "relation_type", "range_type"],
    ...
  ],
  "weak_relations": ["heuristic_relation1", ...]
}}

Rules:
- entity_types: lowercase, singular nouns only (e.g. "researcher", not "Researchers")
- type_aliases: map synonyms/variants to the most precise canonical label
- relation_triples: only domain-specific relations (not "related_to" or "mentions")
- weak_relations: relations that are NLP heuristics, not domain knowledge
- Return valid JSON only — no markdown, no explanation.

JSON:"""


class OntologyLearnerAgent(BaseAgent):
    """
    Induces and extends the domain ontology from document evidence.

    Parameters
    ----------
    memory : AgentMemory
        Shared memory (read-only in this agent — it does not write entities).
    tool_registry : ToolRegistry | None
        Must include ``"read_document"`` and ``"vector_search"`` tools.
    llm : LLM
        LLM backend.
    ontology : DomainOntology
        The ontology object to extend in-place.
    helper_prompt : str
        Optional one-paragraph domain hint prepended to LLM prompts.
    ontology_save_path : Path | None
        If provided, the learned ontology is persisted here after learning.
    max_doc_samples : int
        Maximum number of document excerpts to include in the LLM prompt.
    """

    name = "ontology_learner"

    def __init__(
        self,
        memory: Any,
        tool_registry: Any = None,
        llm: Any = None,
        *,
        ontology: DomainOntology | None = None,
        helper_prompt: str = "",
        ontology_save_path: Path | None = None,
        max_doc_samples: int = 6,
    ) -> None:
        super().__init__(memory=memory, tool_registry=tool_registry, llm=llm)
        self.ontology: DomainOntology = ontology if ontology is not None else DomainOntology()
        self.helper_prompt = helper_prompt.strip()
        self.ontology_save_path = ontology_save_path
        self.max_doc_samples = max_doc_samples

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Induce ontology extensions from the current document corpus.

        Returns
        -------
        dict with keys:
          new_entity_types   : list[str]
          new_aliases        : dict[str, str]
          new_triples        : list[tuple]
          new_weak_relations : list[str]
          ontology_summary   : str
        """
        self.memory.log_step("ontology_learner_start", {
            "helper_prompt": self.helper_prompt[:80] if self.helper_prompt else "(none)",
        })

        # Collect document excerpts for the LLM
        excerpts = self._collect_excerpts()
        if not excerpts:
            self.memory.log_step("ontology_learner_skip", "no documents available")
            return self._empty_result()

        # Ask the LLM to propose schema extensions
        proposal = self._query_llm(excerpts)

        # Merge proposals into the ontology
        new_entity_types = self._merge_entity_types(proposal.get("entity_types", []))
        new_aliases = self._merge_aliases(proposal.get("type_aliases", {}))
        new_triples = self._merge_triples(proposal.get("relation_triples", []))
        new_weak = self._merge_weak(proposal.get("weak_relations", []))

        # Persist
        if self.ontology_save_path:
            try:
                self.ontology.save(self.ontology_save_path)
            except Exception as exc:
                self.memory.log_step("ontology_learner_save_error", str(exc))

        summary = self.ontology.summary()
        self.memory.log_step("ontology_learner_done", {
            "new_entity_types":   new_entity_types,
            "new_triples":        len(new_triples),
            "ontology_summary":   summary,
        })

        return {
            "new_entity_types":   new_entity_types,
            "new_aliases":        new_aliases,
            "new_triples":        new_triples,
            "new_weak_relations": new_weak,
            "ontology_summary":   summary,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_excerpts(self) -> str:
        """
        Pull document excerpts from memory.  Up to ``max_doc_samples``
        documents; first 800 characters each.
        """
        docs = list(self.memory.documents.values())
        if not docs:
            return ""
        sampled = docs[: self.max_doc_samples]
        parts = []
        for i, text in enumerate(sampled, 1):
            snippet = text[:800].replace("\n", " ").strip()
            parts.append(f"[Doc {i}] {snippet}")
        return "\n\n".join(parts)

    def _query_llm(self, excerpts: str) -> dict[str, Any]:
        """
        Call the LLM with the entity-proposal prompt and parse the response.

        Falls back to an empty dict on any error so the rest of the system
        continues with the seed ontology.
        """
        if self.llm is None:
            return {}

        domain_hint = (
            f"DOMAIN CONTEXT:\n{self.helper_prompt}\n\n"
            if self.helper_prompt
            else ""
        )

        prompt = _ENTITY_PROPOSAL_PROMPT.format(
            domain_hint=domain_hint,
            excerpts=excerpts,
        )

        try:
            raw = self.llm.generate(prompt)
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict):
                    return parsed
        except Exception as exc:
            self.memory.log_step("ontology_learner_llm_error", str(exc))

        return {}

    def _merge_entity_types(self, proposed: list[str]) -> list[str]:
        """Add new canonical entity types; return list of genuinely new ones."""
        added = []
        for raw in proposed:
            if not isinstance(raw, str):
                continue
            canon = raw.strip().lower()
            if canon and canon not in self.ontology.entity_types:
                self.ontology.add_entity_type(canon)
                added.append(canon)
        return added

    def _merge_aliases(self, proposed: dict[str, str]) -> dict[str, str]:
        """Add new type aliases; return the genuinely new mappings."""
        added: dict[str, str] = {}
        if not isinstance(proposed, dict):
            return added
        for alias, canonical in proposed.items():
            if not isinstance(alias, str) or not isinstance(canonical, str):
                continue
            a, c = alias.strip().lower(), canonical.strip().lower()
            if a and c and a not in self.ontology.type_aliases:
                self.ontology.add_type_alias(a, c)
                added[a] = c
        return added

    def _merge_triples(
        self, proposed: list[list[str]]
    ) -> list[tuple[str, str, str]]:
        """Add new allowed triples; return list of genuinely new ones."""
        added: list[tuple[str, str, str]] = []
        if not isinstance(proposed, list):
            return added
        for item in proposed:
            if not (isinstance(item, (list, tuple)) and len(item) == 3):
                continue
            if not all(isinstance(x, str) for x in item):
                continue
            triple = (
                item[0].strip().lower(),
                item[1].strip().lower(),
                item[2].strip().lower(),
            )
            if all(triple) and triple not in self.ontology.allowed_triples:
                self.ontology.add_allowed_triple(*triple)
                added.append(triple)
        return added

    def _merge_weak(self, proposed: list[str]) -> list[str]:
        """Add new weak relation types; return list of genuinely new ones."""
        added = []
        if not isinstance(proposed, list):
            return added
        for raw in proposed:
            if not isinstance(raw, str):
                continue
            rel = raw.strip().lower()
            if rel and rel not in self.ontology.weak_relations:
                self.ontology.add_weak_relation(rel)
                added.append(rel)
        return added

    def _empty_result(self) -> dict[str, Any]:
        return {
            "new_entity_types":   [],
            "new_aliases":        {},
            "new_triples":        [],
            "new_weak_relations": [],
            "ontology_summary":   self.ontology.summary(),
        }
