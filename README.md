# Ontology Harness

**An open, local-first, agent-native ontology substrate.**

A local daemon that owns a typed object/link graph, exposes it over MCP so any agent can read and write structured knowledge with provenance and an audit trail — without needing to know or care how the graph is stored.

---

## What it is

Ontology Harness is not a knowledge graph builder. It is the substrate *under* one.

Most ontology + LLM systems share the same gap: anything can mutate the graph with no validation, no record of *why* a write happened, and no way to trace data back to its source. When you close the process, the graph vanishes.

Ontology Harness fixes that:

- **Persistent** — SQLite-backed. Data survives process restarts. Open `ontology.db` with `sqlite3` or any SQL tool.
- **Governed writes** — every mutation goes through declared Action Types with input validation and atomic transactions. No ad-hoc graph edits.
- **Provenance** — every object, link, and property value is traceable to the source (file, agent, conversation) and timestamp that produced it.
- **Agent-native** — MCP is the primary interface. Connect via Claude Code, Claude Desktop, or any MCP client and agents can read/write the ontology as naturally as they call tools.
- **Local-first** — runs entirely on your machine. No server, no cloud sync, no auth beyond filesystem permissions.

This is the "opencode for ontologies" — open-source, local-first, governed writes, agent-native by construction rather than a database agents happen to call.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/marques576/agentic-research-graph
cd agentic-research-graph
uv sync
```

### 2. Create a database

```bash
uv run python -m cli.main init --db my_ontology.db
```

### 3. Connect an agent via MCP

Add to your Claude Desktop or Claude Code MCP config:

```json
{
  "mcpServers": {
    "ontology-harness": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server.server", "--db", "my_ontology.db"]
    }
  }
}
```

The agent can now call these tools:

| Tool | Purpose |
|---|---|
| `get_schema` | List all Object Types and Link Types |
| `define_object_type` | Create a new Object Type (e.g. Person, Organization) |
| `define_link_type` | Create a new Link Type between Object Types |
| `upsert_object` | Create or update an object instance |
| `upsert_link` | Create a link between two objects |
| `delete_object` / `delete_link` | Soft-delete with provenance |
| `get_object` | Get an object with its full provenance history |
| `find_objects` | Find objects by type and property filters |
| `get_neighbors` | Get objects linked to a given object |
| `get_provenance` | Full write history for any object or link |

### 4. Define a schema and add data (via agent)

Tell the agent something like:

> Define an object type "Person" with properties: name (string, required) and age (number). Define a link type "KNOWS" from Person to Person. Then create two Person objects "Alice" and "Bob" and link them with KNOWS.

The agent will call `define_object_type`, `define_link_type`, `upsert_object`, and `upsert_link` through MCP — all validated, all tracked.

### 5. Inspect from the CLI

```bash
uv run python -m cli.main --db my_ontology.db inspect
uv run python -m cli.main --db my_ontology.db export --pretty
uv run python -m cli.main --db my_ontology.db validate
```

Every object and link you see has a non-null, queryable provenance record.

---

## Run the reference pipeline

The repository includes a reference client that demonstrates the full extraction pipeline:

```bash
# Mock LLM (deterministic, no API key needed — instant)
uv run python examples/research_pipeline/run.py

# With a real LLM
uv run python examples/research_pipeline/run.py \
  --base-url https://opencode.ai/zen/go/v1 \
  --model deepseek-v4-pro \
  --api-key oc-...

# Inspect results
uv run python -m cli.main --db examples/research_pipeline/research.db inspect
```

The pipeline reads text files from `data/`, extracts entities and relationships, and writes them through the same Action Types and provenance system. See `examples/research_pipeline/run.py` for details.

The original five-agent blackboard research pipeline (`agents/`, `controller/`, `llm/`, `tools/`, `memory/`, `ingestion/`, `graph/`, `ontology/`) remains available at the repo root as-is. It illustrates the multi-agent pattern that originally motivated the substrate.

---

## Project layout

```
/
├── core/                     ← Ontology Harness core library
│   ├── schema.py              Object Type / Link Type / property definitions + validation
│   ├── store.py                SQLite-backed persistence layer
│   ├── actions.py              Action Type registry — the only write path
│   ├── provenance.py           Provenance record shape
│   └── query.py                Read API: get, find, neighbors, traverse
│
├── mcp_server/
│   └── server.py              MCP server exposing core/ as 11 tools
│
├── cli/
│   └── main.py                Thin CLI: init, inspect, export, validate
│
├── examples/
│   └── research_pipeline/     Reference client: LLM extraction → governed writes
│
├── tests/                     pytest test suite (109 tests)
├── data/                      Text files for the reference pipeline
└── pyproject.toml
```

---

## Data model

### Object Type
A declared type of entity (e.g. `Person`, `Organization`, `Document`).
Has a name, an ordered list of properties (name, data type, required), and a description.
Data types: `string | number | boolean | datetime | reference`.

### Link Type
A declared relationship between two Object Types (e.g. `EMPLOYED_BY: Person -> Organization`).
Has a name, source/target types, cardinality (`one_to_one | one_to_many | many_to_one | many_to_many`), and a description.

### Object (instance)
An entity instance with a UUID, object type, validated properties, timestamps, and provenance.

### Link (instance)
A relationship instance with a UUID, link type, source/target object IDs, optional properties, and provenance.

### Provenance
Every object and link carries a provenance record: source, agent, action type, timestamp, and optional confidence. Nothing enters the graph without it.

### Action Types
The governance layer. Six built-in types: `create_object_type`, `create_link_type`, `upsert_object`, `upsert_link`, `delete_object`, `delete_link`. All writes go through an Action Type — there is no other write path.

---

## Design decisions

- **SQLite, not a graph DB** — survives restarts, supports concurrent readers, inspectable with zero setup. A dedicated graph DB is out of scope until query performance on tens of millions of edges becomes the bottleneck.
- **MCP is the primary interface** — this is what makes it agent-native by construction. An HTTP layer can wrap the same core library later but is not required.
- **JSON blob for properties columns** — schema flexibility, with validation happening in Python at the Action Type layer. Object Types are user-defined and dynamic, so SQL constraints can't cover the validation surface.
- **Soft deletes** — objects and links are marked deleted, never hard-deleted. Provenance history is preserved.
- **In-process core usage for the example client** — `examples/research_pipeline/` calls `core/` directly rather than going through MCP, since it runs in the same process. An external agent would use the MCP server.

---

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- No GPU, no external services, no API keys required for the core
