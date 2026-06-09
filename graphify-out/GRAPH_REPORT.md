# Graph Report - .  (2026-06-09)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 384 nodes · 839 edges · 13 communities (12 shown, 1 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 161 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `10575e29`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]

## God Nodes (most connected - your core abstractions)
1. `AgentMemory` - 40 edges
2. `DomainOntology` - 28 edges
3. `Any` - 27 edges
4. `Path` - 21 edges
5. `ToolRegistry` - 21 edges
6. `LLM` - 20 edges
7. `_rprint()` - 18 edges
8. `ToolRegistry` - 18 edges
9. `MultimodalIngestionPipeline` - 17 edges
10. `LLM` - 17 edges

## Surprising Connections (you probably didn't know these)
- `AgentMemory` --uses--> `AgentMemory`  [INFERRED]
  agents/base_agent.py → memory/memory.py
- `AgentMemory` --uses--> `ToolRegistry`  [INFERRED]
  agents/base_agent.py → tools/tools.py
- `ToolRegistry` --uses--> `AgentMemory`  [INFERRED]
  agents/base_agent.py → memory/memory.py
- `ToolRegistry` --uses--> `ToolRegistry`  [INFERRED]
  agents/base_agent.py → tools/tools.py
- `Any` --uses--> `AgentMemory`  [INFERRED]
  agents/base_agent.py → memory/memory.py

## Import Cycles
- None detected.

## Communities (13 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.10
Nodes (30): ResearchAgent – retrieves and analyses documents from the vector store.  Respons, Searches the document store and extracts structured knowledge.      After a run(, BOLD(), CYAN(), DIM(), GREEN(), MAGENTA(), _parse_core_types_from_hint() (+22 more)

### Community 1 - "Community 1"
Cohesion: 0.10
Nodes (36): LLM, Path, ToolRegistry, Instantiate all tools and register them., ExtractEntitiesTool, GraphNeighborsTool, GraphShortestPathTool, MultimodalVectorSearchTool (+28 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (37): Ingestion package — multimodal file processing pipeline., _chunk_text(), _embed_input_for_text(), _embed_input_for_video(), _extract_docx(), _extract_pdf_text(), _extract_pptx(), _extract_xlsx() (+29 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (21): Any, Extract relationship triples from text.          Uses an LLM-based approach guid, Heuristic: two entities that appear within 200 characters of each other, Parameters         ----------         query   : free-text search query         d, AgentMemory, Entity, Any, Shared memory system for the knowledge discovery agent.  The memory object is th (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (21): DomainOntology, Any, Path, DomainOntology – formal schema for entity classes and relationship types.  The o, Check whether a directed relationship is permitted by the ontology.          Par, Convenience boolean wrapper around validate_relationship., Register a new canonical entity type.          Parameters         ----------, Register a user-specified core entity type.          Core types are added to bot (+13 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (18): KnowledgeGraph – NetworkX-backed directed multigraph for entity-relationship sto, Add or update a directed relationship edge.          If an edge with the **same*, Return all edges that touch a given node (both in- and out-edges).          Para, Return every edge in the graph.          Returns         -------         list of, Return names of all nodes directly connected to the given node.          Include, Directed knowledge multigraph backed by NetworkX MultiDiGraph.      Nodes repres, Find the shortest directed path between two nodes.          Parameters         -, Serialise the graph to a plain dict.          Returns         -------         di (+10 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (18): AgentMemory, Any, ToolRegistry, Agent base class.  All agents share a common interface: they receive memory and, Abstract base for all research agents.      Each agent is responsible for a sing, Execute agent logic; return a structured result dict., Convenience wrapper to call a tool and handle missing tools., Any (+10 more)

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (19): ABC, ArgumentParser, LLM, MockLLM, OpenAICompatibleLLM, Any, LLM abstraction layer.  Provides a single `LLM` base class plus two concrete bac, Generic backend for any OpenAI-compatible /v1/chat/completions endpoint.      Wo (+11 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (12): Any, Path, OntologyLearnerAgent – induces a domain ontology from unstructured documents.  T, Induce ontology extensions from the current document corpus.          Returns, Pull document excerpts from memory.  Up to ``max_doc_samples``         documents, Call the LLM with the entity-proposal prompt and parse the response.          Fa, Add new canonical entity types; return list of genuinely new ones., Add new type aliases; return the genuinely new mappings. (+4 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (11): Any, Hypothesis, ValidationAgent – validates hypotheses against accumulated evidence and updates, Run LLM validation on a single hypothesis.          Updates the hypothesis in-pl, Build a focused evidence string for the hypothesis.          Prefers snippets th, Return memory relationships that touch any hypothesis entity., Validates hypotheses by scoring them against accumulated evidence.      After a, Robustly extract a JSON object from potentially noisy LLM text. (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.18
Nodes (9): Any, Hypothesis, HypothesisAgent – generates candidate relationship hypotheses from accumulated e, Build a compact, LLM-readable summary of the current memory state., Call the LLM and parse its response into Hypothesis objects., Parse LLM output into Hypothesis objects.          Handles single-object, array,, Robustly extract a JSON value from potentially noisy LLM text.          Tries fo, Generates candidate hypotheses about hidden relationships in the knowledge graph (+1 more)

## Knowledge Gaps
- **1 isolated node(s):** `install.sh script`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AgentMemory` connect `Community 3` to `Community 0`, `Community 1`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.235) - this node is a cross-community bridge._
- **Why does `DomainOntology` connect `Community 4` to `Community 8`, `Community 0`, `Community 1`?**
  _High betweenness centrality (0.181) - this node is a cross-community bridge._
- **Why does `Any` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `AgentMemory` (e.g. with `base_agent.py` and `AgentMemory`) actually correct?**
  _`AgentMemory` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `DomainOntology` (e.g. with `ontology_learner_agent.py` and `Any`) actually correct?**
  _`DomainOntology` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Any` (e.g. with `graph_explorer_agent.py` and `hypothesis_agent.py`) actually correct?**
  _`Any` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Path` (e.g. with `graph_explorer_agent.py` and `hypothesis_agent.py`) actually correct?**
  _`Path` has 19 INFERRED edges - model-reasoned connections that need verification._