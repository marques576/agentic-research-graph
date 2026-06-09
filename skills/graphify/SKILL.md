---
name: graphify
description: "Drop any folder of code, docs, papers, images, or videos into a persistent knowledge graph with community detection, god nodes, and query/path/explain tools. Works as /graphify command. Supports Hermes Agent as the AI runtime."
---

# /graphify — Hermes Agent

Turn any folder of files into a navigable knowledge graph with community detection, an honest audit trail (EXTRACTED / INFERRED / AMBIGUOUS), and outputs: interactive HTML, GraphRAG-ready JSON, and a plain-language GRAPH_REPORT.md.

## Pre-requisites

graphify is vendored at `lib/graphify/` (git submodule). Before first use:

```bash
pip install graphifyy    # OR
uv add graphifyy
```

Or install from the vendored source:

```bash
cd lib/graphify && pip install -e .
```

## Usage (Hermes Agent)

When the user types `/graphify`, follow the steps below:

```
/graphify                                             # full pipeline on current directory
/graphify <path>                                      # full pipeline on specific path
/graphify <path> --mode deep                          # thorough extraction, richer INFERRED edges
/graphify <path> --update                             # incremental — re-extract only new/changed files
/graphify <path> --cluster-only                       # rerun clustering on existing graph
/graphify <path> --no-viz                             # skip visualization, just report + JSON
/graphify <path> --svg                                # also export graph.svg
/graphify <path> --graphml                            # export graph.graphml (Gephi, yEd)
/graphify <path> --wiki                               # build agent-crawlable wiki
/graphify add <url>                                   # fetch URL, save to ./raw, update graph
/graphify add <url> --author "Name"                   # tag who wrote it
/graphify query "<question>"                          # BFS traversal — broad context
/graphify query "<question>" --dfs                    # DFS — trace a specific path
/graphify path "NodeA" "NodeB"                        # shortest path between two concepts
/graphify explain "NodeName"                          # plain-language explanation of a node
```

### Step 1 — Check for existing graph

If `graphify-out/graph.json` exists and the user's request is a natural-language question about the codebase (not an explicit rebuild command):

→ Jump straight to `graphify query "<question>"` using the terminal tool.

Otherwise, continue to Step 2.

### Step 2 — Detect files

```bash
cd <project-root>
python3 -c "
import json
from graphify.detect import collect_files
from pathlib import Path
result = collect_files(Path('<path>'))
print(json.dumps(result))
" 2>/dev/null
```

Replace `<path>` with the actual directory the user provided.

If `total_files == 0`: stop with "No supported files found in `<path>`."
If `total_words > 2000000` or `total_files > 200`: show warning + top 5 subdirectories, ask which subfolder to run on.

Otherwise proceed.

### Step 3 — Extract + Build + Cluster + Export

```bash
cd <project-root>
python3 -m graphify <path> [--mode deep] [--update] [--cluster-only]
```

This runs the full pipeline (detect → extract → build → cluster → analyze → report → export).

### Step 4 — Present results

Read and present `graphify-out/GRAPH_REPORT.md` (the key findings: god nodes, surprising connections, suggested questions). Tell the user where to find the outputs:

- `graphify-out/graph.html` — interactive graph
- `graphify-out/GRAPH_REPORT.md` — full report
- `graphify-out/graph.json` — reusable graph data (persists across sessions)

## Integration notes

- **Excluded from indexation:** The `data/` directory and `goodgraph.png` / `goodGraph.png` are gitignored and excluded from graphify's file detection (they are not analysed).
- **Existing project graph:** This project (`agentic-research-graph`) already builds knowledge graphs via its own pipeline. Use graphify to build supplementary graphs from specific code/document folders, not to duplicate the main research graph.
- **Agent memory:** After building a graph, save key node IDs and community structure to memory so future `query`/`path`/`explain` calls can use them without re-clustering.
