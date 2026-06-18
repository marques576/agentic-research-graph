# Reference Research Pipeline

This example demonstrates the Ontology Harness substrate end-to-end with a simple entity extraction pipeline over text documents.

## What it does

1. Defines Object Types (Person, Organization, Document, Event, Location, FinancialRecord)
2. Defines Link Types (EMPLOYED_BY, KNOWS, MENTIONED_IN, etc.)
3. Reads text files from the `data/` directory
4. Uses an LLM to extract entities and relationships
5. Writes everything through `core/actions.py` with full provenance tracking
6. Demonstrates persistence, reingest guards, and the query API

## Run it

```bash
# Mock LLM — deterministic, no API key, instant
uv run python examples/research_pipeline/run.py

# Real LLM
uv run python examples/research_pipeline/run.py \
  --base-url https://opencode.ai/zen/go/v1 \
  --model deepseek-v4-pro \
  --api-key oc-...

# Re-extract entities into existing database
uv run python examples/research_pipeline/run.py --reingest

# Inspect results
uv run python -m cli.main --db examples/research_pipeline/research.db inspect

# Export as JSON
uv run python -m cli.main --db examples/research_pipeline/research.db export --pretty > export.json

# Validate integrity
uv run python -m cli.main --db examples/research_pipeline/research.db validate
```

## What was simplified from the original pipeline

This example is a deliberate simplification of the original five-agent research pipeline found in `agents/`, `controller/`, `llm/`, etc. at the repo root.

| Original feature | Status in this example |
|---|---|
| Multimodal ingestion (PDF, audio, video, images) | **Removed** — text files only. This is a feature of ingestion pipelines built on top of the substrate, not the substrate itself. |
| Embedding + vector search (Qwen3-VL, FAISS) | **Removed** — documents are read directly by filename. |
| Multi-agent blackboard loop (planner → researcher → graph explorer → hypothesis → validation) | **Simplified** — single-phase extract-and-write. The substrate demonstrates agent addressability; the multi-agent orchestration is a client concern. |
| Confidence-gated termination | **Removed** — always writes all extracted data. |
| Hypothesis generation and validation | **Removed** — not part of the substrate's responsibility. |
| Graph visualization (graph.html, Cytoscape.js) | **Removed** — use `cli export` and feed to any visualization tool. |

## What is preserved and demonstrated

- **Schema authoring** via `create_object_type` / `create_link_type` Action Types
- **Governed writes** — all entity and relationship writes go through `upsert_object` / `upsert_link` with type validation and cardinality enforcement
- **Provenance tracking** — every write is traceable to its source document and agent
- **Persistence** — data survives in SQLite across process restarts and is inspectable with `cli inspect`
- **Read API** — `get_object`, `find_objects`, `get_neighbors`, `get_provenance` all work on the persisted data
- **Integrity validation** — `cli validate` checks referential integrity and provenance completeness

## Design choice: direct core API vs MCP

This example calls `core/` directly (in-process) rather than going through the MCP server. The MCP server is designed for external agent clients (Claude Code, Claude Desktop, custom scripts). Internal code running in the same Python process should use the `core/` library directly for efficiency. Both paths use the same `ActionRegistry` with the same validation and provenance guarantees.
