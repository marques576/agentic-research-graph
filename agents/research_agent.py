"""
ResearchAgent – retrieves and analyses documents from the vector store.

Responsibilities
----------------
1. Run vector_search to find relevant documents.
2. Read full document texts via read_document.
3. Extract entities from document text via extract_entities.
4. Extract entity relationships from text (LLM-assisted; co-occurrence
   fallback is opt-in via ``use_cooccurrence=True``).
5. Summarise documents via summarize.
6. Persist all findings to shared memory.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.base_agent import BaseAgent
from memory.memory import Entity, Relationship


class ResearchAgent(BaseAgent):
    """
    Searches the document store and extracts structured knowledge.

    After a run() call the following memory fields may be updated:
      - memory.documents     : new raw texts added
      - memory.entities      : entities extracted from those texts
      - memory.relationships : relationships extracted from those texts
      - memory.evidence      : text snippets for later validation

    Parameters
    ----------
    memory : AgentMemory
    tool_registry : ToolRegistry | None
    llm : LLM | None
    helper_prompt : str
        Optional one-paragraph domain hint injected into the relationship
        extraction LLM prompt to improve relation-type vocabulary.
    ontology : DomainOntology | None
        Live ontology — used to suggest valid relation types in the prompt.
    use_cooccurrence : bool
        When True, fall back to proximity-based co-occurrence edges when the
        LLM extraction returns nothing.  Default ``False`` — co-occurrence
        produces noisy ``co-occurs-with`` edges that are rarely meaningful.
    """

    name = "researcher"

    def __init__(
        self,
        memory: Any,
        tool_registry: Any = None,
        llm: Any = None,
        *,
        helper_prompt: str = "",
        ontology: Any = None,
        use_cooccurrence: bool = False,
    ) -> None:
        super().__init__(memory=memory, tool_registry=tool_registry, llm=llm)
        self.helper_prompt: str = helper_prompt.strip()
        self._ontology: Any = ontology
        self.use_cooccurrence: bool = use_cooccurrence

    def run(
        self,
        query: str = "",
        doc_id: str | None = None,
        max_docs: int = 3,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        query   : free-text search query
        doc_id  : if provided, read a specific document directly
        max_docs: maximum number of documents to process

        Returns
        -------
        dict with keys: documents_found, entities_found, summaries, relationships_found
        """
        self.memory.log_step("researcher_start", {"query": query, "doc_id": doc_id})

        docs_processed: list[str] = []
        entities_found: list[dict[str, str]] = []
        summaries: list[str] = []
        relationships_found: list[tuple[str, str, str]] = []

        # --- 1. determine which documents / chunks to work with ---
        # chunk_texts maps doc_id -> chunk text to process (may be a sub-section of the doc)
        chunk_texts: dict[str, str] = {}

        if doc_id:
            # direct lookup — fetch full document text
            result = self._use_tool("read_document", doc_id)
            if result["found"]:
                self.memory.add_document(doc_id, result["text"])
                chunk_texts[doc_id] = result["text"]
            else:
                self.memory.log_step("researcher_doc_not_found", {"doc_id": doc_id})
                return {
                    "documents_found": [],
                    "entities_found": [],
                    "summaries": [],
                    "relationships_found": [],
                }
        else:
            # semantic search — use the retrieved chunk text directly (not the full doc)
            search_results = self._use_tool("vector_search", query)
            seen_docs: set[str] = set()
            for r in search_results:
                did = r["doc_id"]
                if did in seen_docs or len(seen_docs) >= max_docs:
                    continue
                seen_docs.add(did)
                # Prefer the specific chunk text returned by the search (most relevant section)
                ct = r.get("chunk_text", "").strip()
                if not ct:
                    # Fallback: read full doc if chunk_text absent (cold-start or keyword path)
                    read_result = self._use_tool("read_document", did)
                    ct = read_result.get("text", "") if read_result.get("found") else ""
                if ct:
                    chunk_texts[did] = ct
                # Also register the full document in memory for later reference
                if did not in self.memory.documents:
                    full_result = self._use_tool("read_document", did)
                    if full_result.get("found"):
                        self.memory.add_document(did, full_result["text"])

        # --- 2. extract, summarise each chunk ---
        for did, text in chunk_texts.items():
            docs_processed.append(did)

            # store evidence snippet
            self.memory.add_evidence(text[:500])

            # extract entities
            entities = self._use_tool("extract_entities", text)
            for ent in entities:
                entity_obj = Entity(
                    name=ent["name"],
                    entity_type=ent["type"],
                    source=f"document:{did}",
                )
                self.memory.add_entity(entity_obj)
                entities_found.append(ent)

            # extract relationships from entities + text
            triples = self._extract_relationships(text, entities)
            for src, rel_type, tgt in triples:
                try:
                    self.memory.add_graph_relationship(
                        source=src,
                        target=tgt,
                        relation_type=rel_type,
                        confidence=0.5,
                        evidence=text[:200],
                    )
                    relationships_found.append((src, rel_type, tgt))
                    self.memory.log_step(
                        "researcher_relationship",
                        {"source": src, "relation": rel_type, "target": tgt},
                    )
                except Exception as exc:
                    self.memory.log_step("researcher_relationship_error", str(exc))

            # summarise
            summary = self._use_tool("summarize", text)
            summaries.append(summary)
            self.memory.add_evidence(summary)

        result = {
            "documents_found": docs_processed,
            "entities_found": entities_found,
            "summaries": summaries,
            "relationships_found": [
                {"source": s, "relation": r, "target": t}
                for s, r, t in relationships_found
            ],
        }
        self.memory.log_step("researcher_done", result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_relationships(
        self,
        text: str,
        entities: list[dict[str, str]],
    ) -> list[tuple[str, str, str]]:
        """
        Extract relationship triples from text.

        Uses an LLM-based approach guided by the domain helper prompt and the
        ontology's known relation types.  Falls back to the co-occurrence
        heuristic only if ``self.use_cooccurrence`` is True.

        Parameters
        ----------
        text : str
            Document text to analyse.
        entities : list of dicts
            Entities already extracted from the text (each dict has 'name', 'type').

        Returns
        -------
        list of (source, relation_type, target) tuples.
        """
        entity_names = [e["name"] for e in entities if e.get("name")]
        if len(entity_names) < 2:
            return []

        # --- LLM approach ---
        triples: list[tuple[str, str, str]] = []
        if self.llm is not None:
            try:
                # Build a domain-context hint from helper_prompt + ontology relation types
                domain_hint = ""
                if self.helper_prompt:
                    domain_hint += f"DOMAIN CONTEXT:\n{self.helper_prompt}\n\n"

                relation_hint = ""
                if self._ontology is not None:
                    rel_types = self._ontology.relation_types()
                    if rel_types:
                        relation_hint = (
                            f"Preferred relation types (use these labels when applicable): "
                            f"{', '.join(sorted(rel_types))}.\n"
                            f"You may invent a precise relation label if none fit.\n"
                        )

                prompt = (
                    f"{domain_hint}"
                    f"Given this text and the extracted entities, identify direct relationships.\n"
                    f"{relation_hint}"
                    f"Return a JSON array of triples: "
                    f"[[\"EntityA\", \"relation_type\", \"EntityB\"], ...]\n"
                    f"Only include relationships explicitly stated in the text.\n"
                    f"Use specific, descriptive relation labels (not 'related_to' or 'co-occurs-with').\n"
                    f"Entities: {entity_names}\n"
                    f"Text: {text[:8000]}\n"
                    f"JSON:"
                )
                raw = self.llm.generate(prompt)
                raw = re.sub(r"```(?:json)?", "", raw).strip()
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    if isinstance(parsed, list):
                        for item in parsed:
                            if (
                                isinstance(item, list)
                                and len(item) == 3
                                and all(isinstance(x, str) for x in item)
                            ):
                                triples.append((item[0], item[1], item[2]))
            except Exception as exc:
                self.memory.log_step("researcher_rel_extraction_error", str(exc))
                triples = []

        # --- Co-occurrence fallback (opt-in only) ---
        if not triples and self.use_cooccurrence and len(entity_names) >= 2:
            triples = self._cooccurrence_relationships(text, entity_names)

        return triples

    def _cooccurrence_relationships(
        self,
        text: str,
        entity_names: list[str],
    ) -> list[tuple[str, str, str]]:
        """
        Heuristic: two entities that appear within 200 characters of each other
        are linked with a ``co-occurs-with`` relationship (confidence 0.3).

        Parameters
        ----------
        text : str
            Document text.
        entity_names : list of str
            Entity names to check for co-occurrence.

        Returns
        -------
        list of (source, "co-occurs-with", target) tuples.
        """
        # Find character positions of each entity
        positions: dict[str, list[int]] = {}
        lower_text = text.lower()
        for name in entity_names:
            pos_list: list[int] = []
            search_from = 0
            lower_name = name.lower()
            while True:
                idx = lower_text.find(lower_name, search_from)
                if idx == -1:
                    break
                pos_list.append(idx)
                search_from = idx + 1
            if pos_list:
                positions[name] = pos_list

        triples: list[tuple[str, str, str]] = []
        found_names = list(positions.keys())
        seen: set[tuple[str, str]] = set()

        for i, name_a in enumerate(found_names):
            for name_b in found_names[i + 1:]:
                # Check if any position of a is within 200 chars of any position of b
                close = False
                for pos_a in positions[name_a]:
                    for pos_b in positions[name_b]:
                        if abs(pos_a - pos_b) <= 200:
                            close = True
                            break
                    if close:
                        break

                if close:
                    key = (name_a, name_b)
                    if key not in seen:
                        seen.add(key)
                        triples.append((name_a, "co-occurs-with", name_b))

        return triples
