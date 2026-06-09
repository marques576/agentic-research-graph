---
name: graphify
description: "graphify — turn any code/docs/papers/images/videos folder into a knowledge graph. Use /graphify in OpenCode."
---

# /graphify — OpenCode

Drop any folder of code, docs, papers, images, or videos into graphify and get a queryable knowledge graph with community detection, an honest audit trail (EXTRACTED / INFERRED / AMBIGUOUS), and outputs: interactive HTML, GraphRAG-ready JSON, and a plain-language GRAPH_REPORT.md.

## Install

graphify is vendored at `lib/graphify/` (git submodule). Before first use:

```bash
pip install graphifyy
```

## Fast path — existing graph

Before doing anything else, check `graphify-out/graph.json`. If it exists AND the user's request is a natural-language question about the codebase (not an explicit rebuild):

→ Run `graphify query "<question>"` immediately. Skip detect/extract/build.

## Full pipeline

### Step 0 — GitHub repos (if URL given)

If the argument starts with `https://github.com/`:

```bash
gh repo clone <owner>/<repo> /tmp/graphify-repo-<name>
cd /tmp/graphify-repo-<name>
# Then continue with Step 1 on the cloned path
```

### Step 1 — Detect files

```bash
python3 -c "
import json, sys
sys.path.insert(0, 'lib/graphify')
from graphify.detect import collect_files
from pathlib import Path
result = collect_files(Path('<path>'))
print(json.dumps(result))
"
```

### Step 2 — Extract + Build + Cluster + Export

```bash
cd <project-root>
python3 -m graphify <path> [--mode deep] [--update] [--cluster-only] [--wiki]
```

### Step 3 — Present

Read and present `graphify-out/GRAPH_REPORT.md`. Tell the user:

- `graphify-out/graph.html` — interactive HTML graph
- `graphify-out/graph.json` — persistable graph data
- `graphify-out/wiki/` — agent-crawlable wiki (if `--wiki` used)

## Querying

When graph is already built:

```bash
graphify query "<question>"
graphify path "NodeA" "NodeB"
graphify explain "NodeName"
```

## Notes

- `data/` and `goodgraph.png` / `goodGraph.png` are excluded from graphify's file detection (gitignored).
- This project (`agentic-research-graph`) has its own research graph pipeline. Use graphify for supplementary code/docs graph building.
