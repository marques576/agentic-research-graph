"""
ResearchController – orchestrates the knowledge graph building loop.

Architecture
------------

    goal
      |
      v
  OntologyLearnerAgent  (once, at start — learns entity types + relation schema)
      |
      v
  PlannerAgent          (once — decomposes goal into ordered task list)
      |
      v
  ┌─────────────────────────────────────────────────────────┐
  │  AGENT LOOP (up to max_iterations)                      │
  │                                                         │
  │  Tasks distributed round-robin across iterations:       │
  │    ResearchAgent    (document retrieval/extraction)     │
  │                                                         │
  │  After research tasks each iteration:                   │
  │    GraphExplorerAgent (multi-hop inferred edges)        │
  │    HypothesisAgent    (proposes hidden relationships)   │
  │    ValidationAgent    (scores evidence; may stop loop)  │
  └─────────────────────────────────────────────────────────┘
      |
      v
  final report + graph.html

Loop exits early when ValidationAgent reports
``best_confidence >= confidence_threshold`` (default 0.75).

Each step is printed to stdout with labelled sections for observability:

  PLAN:
  TOOL USED:
  RESULT:
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Any

from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.graph_explorer_agent import GraphExplorerAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.validation_agent import ValidationAgent
from agents.ontology_learner_agent import OntologyLearnerAgent
from ingestion.multimodal_ingestion import MultimodalIngestionPipeline
from llm.llm import LLM, MockLLM
from memory.memory import AgentMemory
from ontology.ontology import DomainOntology
from tools.tools import (
    ExtractEntitiesTool,
    GraphNeighborsTool,
    GraphShortestPathTool,
    MultimodalVectorSearchTool,
    Qwen3VLRerankerTool,
    ReadDocumentTool,
    SummarizeTool,
    ToolRegistry,
)


# ---------------------------------------------------------------------------
# Rich console setup
# ---------------------------------------------------------------------------
try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable
    from rich.panel import Panel as _RichPanel
    _console = _RichConsole()
    _RICH_AVAILABLE = True
except ImportError:
    _console = None
    _RICH_AVAILABLE = False


def _rprint(markup: str) -> None:
    """Print via rich if available, else plain print."""
    if _console is not None:
        _console.print(markup)
    else:
        plain = re.sub(r"\[/?[^\]]+\]", "", markup)
        print(plain)


# ---------------------------------------------------------------------------
# Thin colour wrappers
# ---------------------------------------------------------------------------
def BOLD(t: str) -> str:    return f"[bold]{t}[/bold]"
def CYAN(t: str) -> str:    return f"[bold cyan]{t}[/bold cyan]"
def GREEN(t: str) -> str:   return f"[bold green]{t}[/bold green]"
def YELLOW(t: str) -> str:  return f"[bold yellow]{t}[/bold yellow]"
def MAGENTA(t: str) -> str: return f"[bold magenta]{t}[/bold magenta]"
def DIM(t: str) -> str:     return f"[dim]{t}[/dim]"


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

# Minimum hypothesis confidence required before hypothesis-directed retrieval
# fires.  Below this value the loop has not yet found a meaningful direction,
# so a targeted search would just repeat generic queries.
_MIN_HYPOTHESIS_CONFIDENCE: float = 0.2

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "are", "is", "be", "this", "that", "these", "those", "with", "as",
    "about", "have", "has", "had", "its", "their", "our", "your", "my",
    "which", "where", "when", "how", "what", "who", "will", "can", "may",
    "key", "main", "core", "primary", "important", "include", "includes",
    "entities", "entity", "types", "type", "corpus", "document", "documents",
    "such", "like", "e.g", "eg", "etc", "only", "focus", "focused",
    "between", "among", "across", "within", "related", "covering", "covers",
    "describes", "describe", "analyse", "analyze", "find", "discover",
})

_TYPE_SEPARATORS = re.compile(r"[,;/\s]+")
_NON_NOUN_SUFFIXES = ("ing", "tion", "sion", "ment", "ness", "ity", "ous", "ful", "ive", "ical")


def _parse_core_types_from_hint(helper_prompt: str) -> list[str]:
    """
    Extract candidate entity-type keywords from a free-form helper prompt.

    Parameters
    ----------
    helper_prompt : str
        The raw helper prompt string (any free-form sentence).

    Returns
    -------
    list[str] — candidate core type labels (lowercase singular nouns).
    """
    if not helper_prompt:
        return []

    text = helper_prompt.lower()
    raw_tokens = re.split(r"[,;/\s\-\(\)\"\']+", text)

    types: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        tok = tok.strip(".,;:()\"'").strip()
        if not tok or len(tok) < 3 or len(tok) > 25:
            continue
        if tok in _STOPWORDS:
            continue
        if tok.endswith(_NON_NOUN_SUFFIXES):
            continue
        canon = tok
        if tok.endswith("ies") and len(tok) > 4:
            canon = tok[:-3] + "y"
        elif tok.endswith("ses") and len(tok) > 4:
            canon = tok[:-2]
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            canon = tok[:-1]
        if canon not in seen:
            seen.add(canon)
            types.append(canon)

    return types


class ResearchController:
    """
    Orchestrates the complete knowledge graph building loop.

    Parameters
    ----------
    llm : LLM | None
        LLM backend.  Defaults to MockLLM if not provided.
    data_dir : str | Path
        Folder to scan for input files.
    max_iterations : int
        Number of loop iterations.  Each iteration processes a round-robin
        slice of the plan tasks so every task is always reached.
    confidence_threshold : float
        Confidence level at which the best hypothesis is considered confirmed
        and the loop exits early.  Default 0.75.
    embedding_model_id : str
        Qwen3-VL-Embedding model id.
    reranker_model_id : str | None
        Qwen3-VL-Reranker model id.  None disables re-ranking.
    embedding_kwargs : dict
        Extra kwargs forwarded to the embedder.
    ingestion_kwargs : dict
        Extra kwargs forwarded to MultimodalIngestionPipeline.
    helper_prompt : str
        One-paragraph domain hint for ontology learning.
    ontology_path : str | Path | None
        Where to save/load the learned ontology JSON.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        data_dir: str | Path = "data",
        max_iterations: int = 3,
        confidence_threshold: float = 0.75,
        embedding_model_id: str = "Qwen/Qwen3-VL-Embedding-2B",
        reranker_model_id: str | None = "Qwen/Qwen3-VL-Reranker-2B",
        embedding_kwargs: dict | None = None,
        ingestion_kwargs: dict | None = None,
        helper_prompt: str = "",
        ontology_path: str | Path | None = None,
    ) -> None:
        self.llm = llm or MockLLM()
        self.data_dir = Path(data_dir)
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.embedding_model_id = embedding_model_id
        self.reranker_model_id = reranker_model_id
        self.embedding_kwargs: dict = embedding_kwargs or {}
        self.ingestion_kwargs: dict = ingestion_kwargs or {}
        self.helper_prompt: str = helper_prompt.strip()

        if ontology_path:
            self.ontology_path: Path = Path(ontology_path).resolve()
        else:
            self.ontology_path = Path.cwd() / "ontology.json"

        self._mem: AgentMemory = AgentMemory()
        self._tools: ToolRegistry = ToolRegistry()
        self._pipeline: MultimodalIngestionPipeline | None = None
        self._ontology: DomainOntology = DomainOntology()

        self._doc_chunks: list[tuple[str, str]] = []
        self._doc_chunk_cursor: int = 0

        self._graph_html_path: Path = Path.cwd() / "graph.html"
        self._graph_last_nodes: int = 0
        self._graph_last_edges: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, goal: str) -> dict[str, Any]:
        """
        Execute the full knowledge graph building loop.

        Parameters
        ----------
        goal : str
            Natural-language description of the domain / research question.

        Returns
        -------
        dict containing the final report.
        """
        self._print_header(goal)
        start_time = time.time()

        self._mem = AgentMemory()
        self._doc_chunks = []
        self._doc_chunk_cursor = 0
        documents = self._ingest_data()

        # Build flat list of (doc_id, chunk_text) for extract_entities / summarize steps
        _STEP_CHUNK = 8000
        _STEP_OVERLAP = 500
        _REF_HEADING = re.compile(
            r'\n[ \t]*(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY'
            r'|Works Cited|WORKS CITED)[ \t]*\n',
        )
        for _did, _text in documents.items():
            _ref_match = _REF_HEADING.search(_text)
            _body = _text[:_ref_match.start()] if _ref_match else _text
            start = 0
            while start < len(_body):
                end = start + _STEP_CHUNK
                self._doc_chunks.append((_did, _body[start:end]))
                start = end - _STEP_OVERLAP

        for doc_id, text in documents.items():
            self._mem.add_document(doc_id, text)

        self._tools = self._build_tool_registry(documents)

        # --- Phase 0: Ontology learning ---
        self._learn_ontology()
        self._mem.set_ontology(self._ontology)

        # --- Phase 1: Planning ---
        planner = PlannerAgent(
            memory=self._mem,
            tool_registry=self._tools,
            llm=self.llm,
        )
        plan_result = planner.run(goal=goal)
        plan: list[dict[str, Any]] = plan_result["plan"]
        self._print_plan(plan)

        # --- Phase 2: Agent loop ---
        # Tasks are distributed round-robin so every task is reached every
        # iteration regardless of plan length.  This avoids the floor-division
        # bug where tail tasks are skipped when the loop exits early.
        for iteration in range(1, self.max_iterations + 1):
            self._mem.iteration = iteration
            self._print_section_header(f"ITERATION {iteration}")

            tasks_this_iter = self._tasks_for_iteration(plan, iteration)

            for task in tasks_this_iter:
                self._execute_task(task, goal)
                self._snapshot_graph(goal)

            # Graph exploration after research tasks
            if self._mem.graph.node_count() >= 2:
                self._run_graph_exploration(goal)
                self._snapshot_graph(goal)

            # Hypothesis generation — propose hidden relationships
            self._run_hypothesis_generation(goal)

            # Validation — score hypotheses; may trigger early exit
            should_stop = self._run_validation()

            # Hypothesis-directed retrieval: if we have a best hypothesis with
            # meaningful confidence, fire a focused search to gather more
            # targeted evidence before the next iteration begins.
            if not should_stop:
                self._run_hypothesis_directed_retrieval()

            self._snapshot_graph(goal)

            if should_stop:
                best_conf = max(
                    (h.confidence for h in self._mem.hypotheses), default=0.0
                )
                _rprint(
                    f"\n  {GREEN('✓ CONFIDENCE THRESHOLD REACHED')} "
                    f"({best_conf:.2f} ≥ {self.confidence_threshold:.2f}) "
                    f"— stopping after iteration {iteration}."
                )
                break

        # --- Phase 3: Report ---
        elapsed = time.time() - start_time
        report = self._generate_report(goal, elapsed)
        self._print_report(report)

        return report

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _ingest_data(self) -> dict[str, str]:
        """Run the multimodal ingestion pipeline over the data folder."""
        _rprint(f"\n{CYAN('INGESTION:')} scanning '{self.data_dir}' …")

        pipeline = MultimodalIngestionPipeline(
            data_dir=self.data_dir,
            embedding_model_id=self.embedding_model_id,
            embedding_kwargs=self.embedding_kwargs,
            verbose=True,
            **self.ingestion_kwargs,
        )
        documents = pipeline.run()
        self._pipeline = pipeline

        if not documents:
            _rprint(
                f"{YELLOW('WARNING')}: No content was ingested from '{self.data_dir}'.\n"
                f"  Drop supported files there and re-run.\n"
                f"  Supported: PDF, TXT, MD, JPG, PNG, MP3, WAV, MP4, DOCX, …"
            )

        return documents

    def _learn_ontology(self) -> None:
        """Load or learn the domain ontology."""
        _rprint(f"\n{CYAN('ONTOLOGY:')} loading / learning domain schema …")

        current_hash = hashlib.sha256(self.helper_prompt.encode()).hexdigest()[:16]

        if self.ontology_path.exists():
            candidate = DomainOntology.load(self.ontology_path)
            if candidate.goal_hash == current_hash:
                self._ontology = candidate
                _rprint(
                    f"  {DIM('Loaded ontology:')} {self.ontology_path.name}  "
                    f"({len(self._ontology.allowed_triples)} triples, "
                    f"{len(self._ontology.entity_types)} types)"
                )
            else:
                self._ontology = DomainOntology()
                _rprint(
                    f"  {YELLOW('Goal changed')} — discarding cached ontology, "
                    f"learning fresh schema from documents."
                )
        else:
            self._ontology = DomainOntology()
            _rprint(f"  {DIM('No saved ontology found — starting from seed.')}")

        # Stamp the current goal hash so OntologyLearnerAgent persists it on save.
        self._ontology.goal_hash = current_hash

        if self.helper_prompt:
            _rprint(f"  {DIM('Helper prompt:')} {self.helper_prompt[:80]}…")

        learner = OntologyLearnerAgent(
            memory=self._mem,
            tool_registry=self._tools,
            llm=self.llm,
            ontology=self._ontology,
            helper_prompt=self.helper_prompt,
            ontology_save_path=self.ontology_path,
        )
        result = learner.run()

        new_types   = result.get("new_entity_types", [])
        new_triples = result.get("new_triples", [])
        if new_types or new_triples:
            _rprint(
                f"  {DIM('Learned:')} {len(new_types)} new entity type(s), "
                f"{len(new_triples)} new relation triple(s)"
            )

        _rprint(f"  {GREEN(self._ontology.summary())}")

    def _build_tool_registry(self, documents: dict[str, str]) -> ToolRegistry:
        """Instantiate all tools and register them."""
        registry = ToolRegistry()

        registry.register(MultimodalVectorSearchTool(
            documents=documents,
            pipeline=self._pipeline,
            top_k=5,
            embedding_model_id=self.embedding_model_id,
            embedding_kwargs=self.embedding_kwargs,
        ))

        if self.reranker_model_id:
            registry.register(Qwen3VLRerankerTool(
                model_id=self.reranker_model_id,
            ))

        registry.register(ReadDocumentTool(documents))
        registry.register(ExtractEntitiesTool(
            llm=self.llm,
            ontology=self._ontology,
            use_spacy=False,
        ))
        registry.register(SummarizeTool(self.llm))
        registry.register(GraphNeighborsTool(memory=self._mem))
        registry.register(GraphShortestPathTool(memory=self._mem))

        return registry

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------

    def _tasks_for_iteration(
        self,
        plan: list[dict[str, Any]],
        iteration: int,
    ) -> list[dict[str, Any]]:
        """
        Return the tasks assigned to this iteration using round-robin assignment.

        Task i (0-indexed) is assigned to iteration (i % max_iterations) + 1.
        This guarantees every task is reached regardless of plan length and
        avoids the floor-division tail-drop bug.
        """
        return [
            task for i, task in enumerate(plan)
            if (i % self.max_iterations) == (iteration - 1)
        ]

    def _execute_task(self, task: dict[str, Any], goal: str) -> None:
        """Dispatch a single plan task to the ResearchAgent."""
        action = task.get("action", "")
        query = task.get("query", "")
        step = task.get("step", "?")

        _rprint(
            f"\n  {DIM(f'Step {step}:')} {CYAN(action)} "
            f"← {DIM(str(query)[:60])}"
        )

        if action in ("vector_search", "read_document", "extract_entities", "summarize"):
            agent = ResearchAgent(
                memory=self._mem,
                tool_registry=self._tools,
                llm=self.llm,
                helper_prompt=self.helper_prompt,
                ontology=self._ontology,
                use_cooccurrence=False,  # co-occurrence floods graph with 0.50 edges; real LLM extracts typed triples
            )

            if action == "read_document":
                # The planner writes natural-language queries, not doc IDs.
                # Use vector search to resolve the query to the best matching
                # document ID, then read that document.
                vs_tool = self._tools.get("vector_search")
                doc_id_to_read = str(query)
                if vs_tool is not None:
                    hits = vs_tool.run(str(query))
                    if hits:
                        doc_id_to_read = hits[0].get("doc_id", str(query))
                result = agent.run(doc_id=doc_id_to_read)
            elif action == "extract_entities":
                if self._doc_chunks:
                    idx = self._doc_chunk_cursor % len(self._doc_chunks)
                    self._doc_chunk_cursor += 1
                    chunk_did, chunk_text = self._doc_chunks[idx]
                    result = agent.run(query=chunk_text)
                else:
                    result = agent.run(query=str(query))
            elif action == "summarize":
                if self._doc_chunks:
                    idx = self._doc_chunk_cursor % len(self._doc_chunks)
                    self._doc_chunk_cursor += 1
                    _chunk_did, chunk_text = self._doc_chunks[idx]
                    tool = self._tools.get("summarize")
                    summary = tool.run(chunk_text) if tool else ""
                    self._mem.add_evidence(summary)
                    result: Any = {"summary": summary}
                elif self._mem.documents:
                    last_text = list(self._mem.documents.values())[-1]
                    tool = self._tools.get("summarize")
                    summary = tool.run(last_text) if tool else ""
                    self._mem.add_evidence(summary)
                    result = {"summary": summary}
                else:
                    result = {"summary": ""}
            elif action == "vector_search":
                raw_query = str(query)
                steered_query = (
                    f"{self.helper_prompt} {raw_query}"
                    if self.helper_prompt
                    else raw_query
                )
                result = agent.run(query=steered_query)
            else:
                result = agent.run(query=str(query))

            self._print_tool_result(action, result)

        else:
            _rprint(f"    {YELLOW('Unknown action')} '{action}' — skipping.")

    def _run_graph_exploration(self, goal: str) -> None:
        """Run the GraphExplorerAgent and print discovered paths."""
        _rprint(f"\n  {CYAN('── GRAPH EXPLORATION ──')}")
        explorer = GraphExplorerAgent(
            memory=self._mem,
            tool_registry=self._tools,
            llm=self.llm,
        )
        result = explorer.run(goal=goal)
        paths = result.get("discovered_paths", [])
        if paths:
            for p in paths[:3]:
                path_str = " → ".join(p["path"])
                _rprint(f"    {DIM('Path:')} {path_str}")
                interp = p.get("interpretation", "")
                if interp:
                    _rprint(f"    {DIM('Insight:')} {interp[:120]}")
        else:
            _rprint(f"    {DIM('No multi-hop paths found yet.')}")

    def _run_hypothesis_generation(self, goal: str) -> None:
        """Run the HypothesisAgent and print generated hypotheses."""
        _rprint(f"\n  {CYAN('── HYPOTHESIS GENERATION ──')}")
        agent = HypothesisAgent(
            memory=self._mem,
            tool_registry=self._tools,
            llm=self.llm,
            max_hypotheses=3,
        )
        result = agent.run(goal=goal)
        n = result.get("hypotheses_generated", 0)
        if n:
            _rprint(f"    {GREEN(f'{n} new hypothesis/hypotheses generated:')}")
            for h in result.get("hypotheses", []):
                conf = h.get("confidence", 0.0)
                _rprint(
                    f"    {DIM('→')} {h['statement'][:100]}  "
                    f"{DIM(f'(conf: {conf:.2f})')}"
                )
        else:
            _rprint(f"    {DIM('No new hypotheses generated.')}")

    def _run_validation(self) -> bool:
        """
        Run the ValidationAgent.

        Returns
        -------
        bool — True if the best hypothesis confidence has reached the threshold.
        """
        _rprint(f"\n  {CYAN('── VALIDATION ──')}")
        agent = ValidationAgent(
            memory=self._mem,
            tool_registry=self._tools,
            llm=self.llm,
            confidence_threshold=self.confidence_threshold,
        )
        result = agent.run()
        n = result.get("validated_count", 0)
        best_conf = result.get("best_confidence", 0.0)
        best_hyp = result.get("best_hypothesis")

        if n:
            for v in result.get("verdicts", []):
                _rprint(
                    f"    {DIM('Verdict:')} {YELLOW(v.get('verdict', '?'))}  "
                    f"{DIM(v.get('statement', '')[:80])}"
                )
                conf_before = v.get("confidence_before", 0.0)
                conf_after = v.get("confidence_after", 0.0)
                conf_after_str = f"{conf_after:.2f}"
                _rprint(
                    f"    {DIM('Confidence:')} "
                    f"{conf_before:.2f} \u2192 "
                    + GREEN(conf_after_str)
                )
        else:
            _rprint(f"    {DIM('No hypotheses to validate yet.')}")

        if best_hyp:
            _rprint(
                f"    {BOLD('Best hypothesis confidence:')} "
                f"{GREEN(f'{best_conf:.2f}')} / {self.confidence_threshold:.2f}"
            )

        return bool(result.get("should_stop", False))

    def _run_hypothesis_directed_retrieval(self) -> None:
        """
        Fire targeted ResearchAgent runs seeded by the top-2 current hypotheses.

        Running retrieval for the top 2 (rather than just the single best) avoids
        the tie-breaking problem: when two hypotheses share the highest confidence,
        only the first one (by insertion order) would ever be targeted with a
        single-best selection, potentially locking retrieval onto the wrong lead
        for the rest of the run.

        Guards
        ------
        - Skips when no hypotheses exist yet.
        - Skips when best_confidence < _MIN_HYPOTHESIS_CONFIDENCE (no useful
          signal yet — a targeted search would just repeat generic queries).
        - Skips when best_confidence >= confidence_threshold (loop is about to
          stop; extra retrieval would be wasted work).
        - Each candidate hypothesis is only targeted if its confidence is within
          0.10 of the best (avoids wasting a retrieval on a stale low-conf hyp).
        """
        if not self._mem.hypotheses:
            return

        sorted_hyps = sorted(
            self._mem.hypotheses, key=lambda h: h.confidence, reverse=True
        )
        best_conf = sorted_hyps[0].confidence

        # No signal yet, or already at/past the threshold — nothing to do.
        if best_conf < _MIN_HYPOTHESIS_CONFIDENCE or best_conf >= self.confidence_threshold:
            return

        # Take up to 2 hypotheses within 0.10 of the best confidence.
        candidates = [
            h for h in sorted_hyps[:5]
            if best_conf - h.confidence <= 0.10
        ][:2]

        for hyp in candidates:
            statement = hyp.statement
            entities = hyp.entities_involved[:4]
            entity_fragment = " ".join(entities)
            query = f"{statement} {entity_fragment}".strip()

            _rprint(f"\n  {CYAN('── HYPOTHESIS-DIRECTED RETRIEVAL ──')}")
            _rprint(
                f"    {DIM('Targeting:')} {statement[:100]}"
                f"  {DIM(f'(conf: {hyp.confidence:.2f})')}"
            )

            agent = ResearchAgent(
                memory=self._mem,
                tool_registry=self._tools,
                llm=self.llm,
                helper_prompt=self.helper_prompt,
                ontology=self._ontology,
                use_cooccurrence=False,  # avoid noisy co-occurrence edges
            )
            result = agent.run(query=query, max_docs=2)

            n_docs = len(result.get("documents_found", []))
            n_ents = len(result.get("entities_found", []))
            n_rels = len(result.get("relationships_found", []))
            _rprint(
                f"    {DIM('Found:')} {n_docs} doc(s), "
                f"{n_ents} entit(ies), "
                f"{n_rels} relationship(s) — "
                f"evidence queued for next validation cycle"
            )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _synthesize_answer(self, goal: str) -> str:
        """
        Ask the LLM to directly answer the research goal using all accumulated
        evidence, relationships, and hypotheses.

        This is a guaranteed answer path that fires regardless of whether any
        hypothesis reached the confidence threshold.

        Parameters
        ----------
        goal : str
            The original research question.

        Returns
        -------
        str — a direct natural-language answer, or "" if synthesis fails.
        """
        if self.llm is None:
            return ""

        # Build a rich evidence block from everything accumulated.
        # Keep the prompt under ~6000 chars to avoid token-limit rejections.
        _MAX_PROMPT_CHARS = 6000

        # -- Top typed relationships (exclude trivial co-occurs-with)
        typed_rels = [
            r for r in self._mem.relationships
            if r.relation_type not in ("co-occurs-with", "co_occurs_with", "related_to")
        ]
        typed_rels.sort(key=lambda r: r.confidence, reverse=True)
        rel_lines = [
            f"  {r.source} --[{r.relation_type}]--> {r.target}  (conf: {r.confidence:.2f})"
            for r in typed_rels[:30]
        ]

        # -- Best hypotheses (if any) — put these first, highest signal
        sorted_hyps = sorted(
            self._mem.hypotheses, key=lambda h: h.confidence, reverse=True
        )
        hyp_lines = [
            f"  - {h.statement}  (confidence: {h.confidence:.2f})"
            for h in sorted_hyps[:5]
        ]

        # -- Evidence snippets — most recent first, trimmed to fit budget
        evidence_lines: list[str] = []
        for i, ev in enumerate(reversed(self._mem.evidence)):
            evidence_lines.append(f"  [{i+1}] {ev[:300]}")

        parts: list[str] = [f"RESEARCH GOAL: {goal}"]
        if hyp_lines:
            parts.append("\nCURRENT HYPOTHESES (best first):\n" + "\n".join(hyp_lines))
        if rel_lines:
            parts.append("\nKEY RELATIONSHIPS:\n" + "\n".join(rel_lines))

        base = "\n".join(parts)
        # Fill remaining budget with evidence snippets
        budget = _MAX_PROMPT_CHARS - len(base) - 300  # 300 for the instruction suffix
        ev_block = ""
        if budget > 200 and evidence_lines:
            chosen: list[str] = []
            used = 0
            for line in evidence_lines:
                if used + len(line) + 1 > budget:
                    break
                chosen.append(line)
                used += len(line) + 1
            if chosen:
                ev_block = "\nEVIDENCE:\n" + "\n".join(chosen)

        context = base + ev_block

        prompt = (
            f"{context}\n\n"
            f"Using ONLY the evidence, relationships, and hypotheses listed above, "
            f"provide a direct, specific answer to the research goal:\n"
            f"  \"{goal}\"\n\n"
            f"Name the perpetrator, the method, and the motive. "
            f"If evidence is conclusive, state it directly. "
            f"If partial, state what is known and what remains uncertain.\n\n"
            f"ANSWER:"
        )

        try:
            result = self.llm.generate(prompt).strip()
            if result:
                return result
        except Exception as exc:
            self._mem.log_step("synthesize_answer_error", str(exc))
            _rprint(f"    {YELLOW('Synthesis error:')} {str(exc)[:200]}")

        # Fallback: LLM returned empty (model refused or null content).
        # Build a structured answer deterministically from what we already know.
        return self._build_fallback_answer(goal, rel_lines)

    def _build_fallback_answer(self, goal: str, rel_lines: list[str]) -> str:
        """
        Build a structured answer string without an LLM call.

        Used when the synthesis LLM call returns empty (e.g. model content
        refusal, null reasoning_content, or API timeout).  Pulls the best
        hypothesis, key relationships, and top evidence snippets into a
        readable narrative.
        """
        sorted_hyps = sorted(
            self._mem.hypotheses, key=lambda h: h.confidence, reverse=True
        )
        if not sorted_hyps:
            return ""

        best = sorted_hyps[0]
        lines: list[str] = [
            f"Goal: {goal}",
            f"",
            f"Most supported conclusion (confidence {best.confidence:.0%}):",
            f"  {best.statement}",
        ]

        if len(sorted_hyps) > 1:
            lines.append("")
            lines.append("Other hypotheses considered:")
            for h in sorted_hyps[1:3]:
                lines.append(f"  [{h.confidence:.0%}] {h.statement}")

        if rel_lines:
            lines.append("")
            lines.append("Key relationships supporting this conclusion:")
            for rl in rel_lines[:8]:
                lines.append(rl)

        top_ev = self._mem.evidence[-3:] if self._mem.evidence else []
        if top_ev:
            lines.append("")
            lines.append("Supporting evidence:")
            for ev in top_ev:
                lines.append(f"  • {ev[:200]}")

        return "\n".join(lines)

    def _generate_report(
        self,
        goal: str,
        elapsed: float,
    ) -> dict[str, Any]:
        """Compile the final structured report from memory."""
        top_relationships = sorted(
            self._mem.relationships,
            key=lambda r: r.confidence,
            reverse=True,
        )

        entities_by_type: dict[str, list[str]] = {}
        for e in self._mem.entities.values():
            entities_by_type.setdefault(e.entity_type, []).append(e.name)

        best_hypothesis = (
            max(self._mem.hypotheses, key=lambda h: h.confidence)
            if self._mem.hypotheses
            else None
        )
        best_confidence = best_hypothesis.confidence if best_hypothesis else 0.0
        conclusion = best_hypothesis.statement if best_hypothesis else ""

        _rprint(f"\n  {CYAN('── SYNTHESIZING DIRECT ANSWER ──')}")
        answer = self._synthesize_answer(goal)
        if answer:
            _rprint(f"    {DIM('Answer synthesized from')} {len(self._mem.evidence)} evidence snippets.")
        else:
            _rprint(f"    {YELLOW('Synthesis returned empty — no LLM available or error.')}")

        return {
            "goal": goal,
            "elapsed_seconds": round(elapsed, 2),
            "iterations": self._mem.iteration,
            "entities_discovered": entities_by_type,
            "relationships": [r.to_dict() for r in top_relationships],
            "hypotheses": [h.to_dict() for h in self._mem.hypotheses],
            "best_hypothesis": best_hypothesis.to_dict() if best_hypothesis else None,
            "best_confidence": best_confidence,
            "conclusion": conclusion,
            "answer": answer,
            "documents_processed": list(self._mem.documents.keys()),
            "evidence_count": len(self._mem.evidence),
            "memory_summary": self._mem.summary(),
            "ontology": self._ontology.to_dict(),
            "memory": self._mem.to_dict(),
        }

    # ------------------------------------------------------------------
    # Pretty-printing
    # ------------------------------------------------------------------

    def _print_header(self, goal: str) -> None:
        if _console is not None and _RICH_AVAILABLE:
            from rich.panel import Panel as _Panel
            _console.print()
            _console.print(_Panel(
                f"[bold]GOAL:[/bold] {goal}",
                title="[bold cyan]KNOWLEDGE GRAPH BUILDER[/bold cyan]",
                border_style="cyan",
                expand=True,
            ))
        else:
            width = 72
            _rprint("\n" + "=" * width)
            _rprint(BOLD("  KNOWLEDGE GRAPH BUILDER"))
            _rprint("=" * width)
            _rprint(f"  {BOLD('GOAL:')} {goal}")
            _rprint("=" * width + "\n")

    def _print_section_header(self, title: str) -> None:
        _rprint(f"\n[dim]{'─' * 64}[/dim]")
        _rprint(BOLD(f"  {title}"))
        _rprint(f"[dim]{'─' * 64}[/dim]")

    def _print_plan(self, plan: list[dict[str, Any]]) -> None:
        _rprint(f"\n{BOLD('PLAN:')}")
        for task in plan:
            q = task.get("query", "")
            q_str = str(q)[:55] + "…" if len(str(q)) > 55 else str(q)
            _rprint(
                f"  {DIM(str(task['step']) + '.')}  "
                f"{CYAN(task['action'])}  {DIM(q_str)}"
            )

    def _print_tool_result(self, tool_name: str, result: Any) -> None:
        _rprint(f"    {BOLD('TOOL USED:')} {tool_name}")
        if isinstance(result, dict):
            for key, val in result.items():
                if isinstance(val, list):
                    _rprint(f"    {DIM(key + ':')}  {len(val)} item(s)")
                elif isinstance(val, str) and len(val) > 80:
                    _rprint(f"    {DIM(key + ':')}  {val[:80]}…")
                else:
                    _rprint(f"    {DIM(key + ':')}  {val}")
        elif isinstance(result, list):
            _rprint(f"    {DIM('items:')}  {len(result)}")
        else:
            _rprint(f"    {DIM('result:')}  {str(result)[:100]}")

    def _print_graph_summary(self) -> None:
        """Print a Knowledge Graph Summary table and a Discovered Relationships table."""
        if _console is None or not _RICH_AVAILABLE:
            return

        from rich.table import Table as _Table

        top_nodes = self._mem.graph.top_nodes_by_degree(10)
        if top_nodes:
            kg_table = _Table(title="Knowledge Graph Summary", border_style="blue")
            kg_table.add_column("Entity", style="bold", no_wrap=True)
            kg_table.add_column("Type", style="cyan")
            kg_table.add_column("Connections", justify="right", style="green")
            for node_name, degree in top_nodes:
                node_data = self._mem.graph.get_entity(node_name) or {}
                node_type = node_data.get("type", "unknown")
                kg_table.add_row(node_name, node_type, str(degree))
            _console.print()
            _console.print(kg_table)

        all_rels = self._mem.graph.all_relationships()
        if all_rels:
            rel_table = _Table(title="Discovered Relationships", border_style="magenta")
            rel_table.add_column("Source", style="bold", no_wrap=True)
            rel_table.add_column("Relation", style="yellow")
            rel_table.add_column("Target", style="bold", no_wrap=True)
            rel_table.add_column("Confidence", justify="right", style="green")
            for rel in all_rels:
                conf_val = rel.get("confidence", 0.0)
                rel_table.add_row(
                    rel.get("source", ""),
                    rel.get("relation_type", ""),
                    rel.get("target", ""),
                    f"{conf_val:.2f}",
                )
            _console.print()
            _console.print(rel_table)

    def _print_report(self, report: dict[str, Any]) -> None:
        self._print_graph_summary()

        _rprint(f"\n{BOLD('  FINAL REPORT')}")
        _rprint(f"[dim]{'=' * 72}[/dim]")
        _rprint(f"\n  {BOLD('GOAL:')} {report['goal']}")
        _rprint(
            f"\n  {BOLD('STATS:')} "
            f"{DIM('Iterations:')} {report['iterations']}"
            f"  |  {DIM('Elapsed:')} {report['elapsed_seconds']}s"
        )

        _rprint(f"\n  {BOLD('ENTITIES DISCOVERED:')}")
        for etype, names in report["entities_discovered"].items():
            _rprint(f"    {DIM(etype + ':')}  {', '.join(names[:5])}")

        _rprint(f"\n  {BOLD('RELATIONSHIPS:')}")
        for r in report["relationships"][:10]:
            conf = r["confidence"]
            bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
            _rprint(
                f"    {r['source'][:25]}"
                f" [dim]→[{r['relation_type']}]→[/dim] "
                f"{r['target'][:25]}"
                f"  {GREEN('[' + bar + ']')} {conf:.2f}"
            )

        hypotheses = report.get("hypotheses", [])
        if hypotheses:
            _rprint(f"\n  {BOLD('HYPOTHESES:')}")
            for h in hypotheses:
                conf = h.get("confidence", 0.0)
                verdict = "validated" if h.get("validated") else "pending"
                bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
                _rprint(
                    f"    {GREEN('[' + bar + ']')} {conf:.2f}  "
                    f"{DIM(f'[{verdict}]')}  {h['statement'][:80]}"
                )

        # --- Direct synthesized answer (always printed when available) ---
        answer = report.get("answer", "")
        if answer:
            if _console is not None and _RICH_AVAILABLE:
                from rich.panel import Panel as _AnswerPanel
                _console.print()
                _console.print(_AnswerPanel(
                    answer,
                    title=f"[bold green]ANSWER: {report['goal'][:80]}[/bold green]",
                    border_style="green",
                    expand=True,
                ))
            else:
                width = 72
                _rprint(f"\n{'=' * width}")
                _rprint(GREEN(f"  ANSWER: {report['goal'][:80]}"))
                _rprint(f"{'=' * width}")
                _rprint(f"\n{answer}\n")
                _rprint(f"{'=' * width}")

        conclusion = report.get("conclusion", "")
        best_conf = report.get("best_confidence", 0.0)
        if conclusion:
            _rprint(f"\n  {BOLD('TOP HYPOTHESIS')} {DIM(f'(confidence: {best_conf:.2f})')}")
            _rprint(f"  {MAGENTA(conclusion)}")

        _rprint(
            f"\n  {BOLD('DOCUMENTS PROCESSED:')} "
            f"{', '.join(report['documents_processed'])}"
        )
        _rprint(f"  {BOLD('EVIDENCE SNIPPETS:')} {report['evidence_count']}")
        _rprint(f"\n  {DIM(report['memory_summary'])}")
        _rprint(f"[dim]{'=' * 72}[/dim]\n")

    # ------------------------------------------------------------------
    # Incremental graph snapshots
    # ------------------------------------------------------------------

    def _snapshot_graph(self, goal: str) -> None:
        """
        Overwrite ``graph.html`` whenever the knowledge graph has grown.
        Opening the file in a browser and refreshing shows the live graph.
        """
        n_nodes = self._mem.graph.node_count()
        n_edges = self._mem.graph.edge_count()

        if n_nodes == self._graph_last_nodes and n_edges == self._graph_last_edges:
            return

        self._graph_last_nodes = n_nodes
        self._graph_last_edges = n_edges

        best_hypothesis = (
            max(self._mem.hypotheses, key=lambda h: h.confidence)
            if self._mem.hypotheses
            else None
        )
        snapshot_report: dict[str, Any] = {
            "goal": goal,
            "best_confidence": best_hypothesis.confidence if best_hypothesis else 0.0,
            "conclusion": best_hypothesis.statement if best_hypothesis else "",
            "memory": {"graph": self._mem.graph.to_dict()},
        }

        self.export_html_graph(snapshot_report, self._graph_html_path)
        _rprint(
            f"    {DIM('Graph →')} {GREEN(str(self._graph_html_path))}  "
            f"({n_nodes} nodes, {n_edges} edges)"
        )

    # ------------------------------------------------------------------
    # HTML graph export
    # ------------------------------------------------------------------

    @staticmethod
    def export_html_graph(report: dict[str, Any], output_path: Path) -> Path:
        """
        Write a self-contained interactive HTML knowledge-graph visualisation.

        Uses Cytoscape.js (CDN) and inlines the full graph data as a JS
        constant — no server required, works offline after the first open.

        Parameters
        ----------
        report : dict
            Must contain ``report["memory"]["graph"]`` with ``nodes`` and
            ``edges`` lists.
        output_path : Path
            The HTML is written with the same stem and a ``.html`` extension.

        Returns
        -------
        Path — absolute path of the written HTML file.
        """
        import json as _json

        graph_data = report.get("memory", {}).get("graph", {"nodes": [], "edges": []})
        nodes: list[dict] = graph_data.get("nodes", [])
        edges: list[dict] = graph_data.get("edges", [])

        PALETTE = [
            "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
            "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        ]
        entity_types = sorted({n.get("type", "unknown") for n in nodes})
        type_colour = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(entity_types)}

        cy_elements: list[dict] = []

        degree_map: dict[str, int] = {}
        for edge in edges:
            for key in ("source", "target"):
                nid = edge.get(key, "")
                if nid:
                    degree_map[nid] = degree_map.get(nid, 0) + 1

        for node in nodes:
            name = node.get("name", "")
            ntype = node.get("type", "unknown")
            cy_elements.append({
                "data": {
                    "id": name,
                    "label": name,
                    "type": ntype,
                    "colour": type_colour.get(ntype, "#aaaaaa"),
                    "attributes": node.get("attributes", {}),
                    "degree": degree_map.get(name, 0),
                },
            })

        for edge in edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("relation_type", "")
            conf = float(edge.get("confidence", 0.5))
            cy_elements.append({
                "data": {
                    "id": f"{src}__{rel}__{tgt}",
                    "source": src,
                    "target": tgt,
                    "label": rel,
                    "confidence": conf,
                    "evidence": edge.get("evidence", ""),
                    "width": max(1, round(conf * 6)),
                },
            })

        legend_items = [
            f'<li><span class="dot" style="background:{type_colour[t]}"></span>{t}</li>'
            for t in entity_types
        ]
        legend_html = "\n".join(legend_items)

        goal      = report.get("goal", "Knowledge Graph")
        n_nodes   = len(nodes)
        n_edges   = len(edges)

        elements_json = _json.dumps(cy_elements, ensure_ascii=False, indent=2)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Knowledge Graph – {goal[:60]}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f1117; color: #e0e0e0; display: flex;
            flex-direction: column; height: 100vh; overflow: hidden; }}
    header {{ padding: 10px 18px; background: #1a1d2e; border-bottom: 1px solid #2a2d3e;
              display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
    header h1 {{ font-size: 15px; font-weight: 600; color: #7dd3fc; flex: 1;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .badges {{ display: flex; gap: 8px; flex-shrink: 0; }}
    .badge {{ background: #2a2d3e; border-radius: 12px; padding: 3px 10px;
              font-size: 12px; color: #94a3b8; }}
    .badge span {{ color: #7dd3fc; font-weight: 600; }}
    main {{ display: flex; flex: 1; overflow: hidden; }}
    #cy {{ flex: 1; background: #0f1117; }}
    aside {{ width: 240px; background: #1a1d2e; border-left: 1px solid #2a2d3e;
             overflow-y: auto; padding: 16px; flex-shrink: 0; }}
    aside h2 {{ font-size: 13px; font-weight: 600; color: #94a3b8;
                text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }}
    ul.legend {{ list-style: none; }}
    ul.legend li {{ display: flex; align-items: center; gap: 8px;
                    font-size: 13px; padding: 3px 0; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
    #tooltip {{ position: fixed; background: #1e2130; border: 1px solid #2a2d3e;
                border-radius: 8px; padding: 10px 14px; font-size: 12px;
                max-width: 280px; pointer-events: none; opacity: 0;
                transition: opacity .15s; z-index: 999; line-height: 1.6; }}
    #tooltip strong {{ color: #7dd3fc; display: block; margin-bottom: 4px; }}
    footer {{ padding: 6px 18px; background: #1a1d2e; border-top: 1px solid #2a2d3e;
              font-size: 11px; color: #475569; flex-shrink: 0; }}
  </style>
</head>
<body>
  <header>
    <h1>{goal}</h1>
    <div class="badges">
      <div class="badge"><span>{n_nodes}</span> nodes</div>
      <div class="badge"><span>{n_edges}</span> edges</div>
    </div>
  </header>
  <main>
    <div id="cy"></div>
    <aside>
      <h2>Entity types</h2>
      <ul class="legend">
        {legend_html}
      </ul>
    </aside>
  </main>
  <div id="tooltip"></div>
  <footer>Knowledge Graph Builder &nbsp;·&nbsp; drag to pan &nbsp;·&nbsp; scroll to zoom &nbsp;·&nbsp; hover for details</footer>

  <script>
    const ELEMENTS = {elements_json};

    const cy = cytoscape({{
      container: document.getElementById('cy'),
      elements: ELEMENTS,
      style: [
        {{
          selector: 'node',
          style: {{
            'background-color': 'data(colour)',
            'label': 'data(label)',
            'color': '#e0e0e0',
            'font-size': '11px',
            'text-valign': 'center',
            'text-halign': 'center',
            'text-wrap': 'wrap',
            'text-max-width': '90px',
            'width': 'mapData(degree, 0, 10, 28, 72)',
            'height': 'mapData(degree, 0, 10, 28, 72)',
            'border-width': 1.5,
            'border-color': '#2a2d3e',
            'text-outline-color': '#0f1117',
            'text-outline-width': 2,
          }}
        }},
        {{
          selector: 'edge',
          style: {{
            'width': 'data(width)',
            'line-color': '#3b4a6b',
            'target-arrow-color': '#3b4a6b',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'color': '#64748b',
            'font-size': '9px',
            'text-rotation': 'autorotate',
            'text-outline-color': '#0f1117',
            'text-outline-width': 2,
          }}
        }},
        {{
          selector: 'node:selected',
          style: {{
            'border-color': '#7dd3fc',
            'border-width': 3,
          }}
        }},
        {{
          selector: 'edge:selected',
          style: {{
            'line-color': '#7dd3fc',
            'target-arrow-color': '#7dd3fc',
          }}
        }},
      ],
    }});

    cy.layout({{
      name: 'cose',
      animate: false,
      randomize: true,
      nodeDimensionsIncludeLabels: true,
      nodeRepulsion: 400000,
      nodeOverlap: 20,
      idealEdgeLength: 150,
      edgeElasticity: 100,
      gravity: 80,
      numIter: 1000,
      initialTemp: 200,
      coolingFactor: 0.95,
      minTemp: 1.0,
    }}).run();

    cy.fit(undefined, 40);

    const tooltip = document.getElementById('tooltip');

    cy.on('mouseover', 'node', evt => {{
      const d = evt.target.data();
      const attrs = Object.entries(d.attributes || {{}})
        .map(([k, v]) => `<br/><em>${{k}}:</em> ${{v}}`).join('');
      tooltip.innerHTML = `<strong>${{d.label}}</strong>type: ${{d.type}}${{attrs}}`;
      tooltip.style.opacity = '1';
    }});

    cy.on('mouseover', 'edge', evt => {{
      const d = evt.target.data();
      const ev = d.evidence ? `<br/><em>evidence:</em> ${{d.evidence.slice(0,120)}}…` : '';
      tooltip.innerHTML =
        `<strong>${{d.label}}</strong>` +
        `${{d.source}} → ${{d.target}}<br/>` +
        `<em>confidence:</em> ${{d.confidence.toFixed(2)}}${{ev}}`;
      tooltip.style.opacity = '1';
    }});

    cy.on('mouseout', 'node edge', () => {{ tooltip.style.opacity = '0'; }});

    document.addEventListener('mousemove', e => {{
      tooltip.style.left = (e.clientX + 14) + 'px';
      tooltip.style.top  = (e.clientY + 14) + 'px';
    }});
  </script>
</body>
</html>"""

        html_path = output_path.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")
        return html_path.resolve()
