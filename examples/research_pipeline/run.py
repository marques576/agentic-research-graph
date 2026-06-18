"""
Reference client: demonstrates the Ontology Harness substrate with a simple
entity extraction pipeline over text documents.

This is a simplified port of the original five-agent research pipeline.
Instead of an in-memory KnowledgeGraph and ad-hoc mutation, all writes go
through the core Action Types with provenance tracking.

Usage:
    # Mock LLM (no API key, deterministic):
    uv run python examples/research_pipeline/run.py

    # With a real LLM:
    uv run python examples/research_pipeline/run.py \
        --base-url https://opencode.ai/zen/go/v1 \
        --model deepseek-v4-pro \
        --api-key oc-...

    # View the database after running:
    uv run python -m cli.main --db examples/research_pipeline/research.db inspect

What was simplified from the original pipeline:
    - No multimodal ingestion (audio/video/images) — text files only
    - No embedding/vector search — documents are read directly by filename
    - No multi-agent blackboard loop — single-phase extract-and-write
    - No hypothesis generation/validation confidence loop
    - No graph visualization export (graph.html)

What is preserved and demonstrated:
    - Schema authoring via Action Types
    - LLM-based entity and relationship extraction
    - All writes go through core/actions.py with provenance
    - Data persists in SQLite across process restarts
    - Query API for traversal and inspection
    - CLI export/validate against the same database
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on the Python path
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.store import OntologyStore
from core.actions import ActionRegistry, ActionError
from core.query import QueryAPI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ontology Harness — Reference Research Pipeline",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to SQLite database (default: research.db next to this script)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing text documents (default: data/ in repo root)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM base URL for OpenAI-compatible endpoint (omit for mock LLM)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key",
    )
    parser.add_argument(
        "--reingest",
        action="store_true",
        help="Re-extract entities even if database already has data",
    )
    return parser


class LLM:
    """Abstract LLM interface (compatible subset of original llm/llm.py)."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(LLM):
    """Deterministic mock that produces structured extraction results.

    In a real pipeline, this would be an OpenAI-compatible call.
    The mock is sufficient to demonstrate the substrate working.
    """

    def generate(self, prompt: str) -> str:
        p = prompt.lower()

        if "entity types" in p or "ontology" in p:
            return json.dumps({
                "entity_types": [
                    "person", "organization", "document", "event",
                    "location", "financial_record",
                ],
                "relation_triples": [
                    ["person", "employed_by", "organization"],
                    ["person", "knows", "person"],
                    ["person", "mentioned_in", "document"],
                    ["person", "involved_in", "event"],
                    ["organization", "located_in", "location"],
                    ["event", "occurred_at", "location"],
                    ["event", "recorded_in", "document"],
                ],
                "aliases": {
                    "individual": "person",
                    "company": "organization",
                    "firm": "organization",
                    "report": "document",
                    "incident": "event",
                },
            })

        if "entity" in p and ("extract" in p or "entities" in p):
            return json.dumps([
                {"name": "Victor Harrington", "type": "person"},
                {"name": "Alice Chen", "type": "person"},
                {"name": "Bob Harrington", "type": "person"},
                {"name": "Alpine Consulting", "type": "organization"},
                {"name": "MedTech Industries", "type": "organization"},
                {"name": "Westbrook Chronicle", "type": "organization"},
                {"name": "Police Incident Report", "type": "document"},
                {"name": "Witness Statement", "type": "document"},
                {"name": "Financial Audit Q3 1987", "type": "document"},
                {"name": "Harrington Letter", "type": "document"},
                {"name": "Lab Accident April 1987", "type": "event"},
                {"name": "Westbrook Industrial Park", "type": "location"},
            ])

        if "relationship" in p or "relation" in p:
            return json.dumps([
                {"source": "Victor Harrington", "target": "Alpine Consulting",
                 "relation": "employed_by", "confidence": 0.9},
                {"source": "Alice Chen", "target": "Alpine Consulting",
                 "relation": "employed_by", "confidence": 0.9},
                {"source": "Bob Harrington", "target": "Alpine Consulting",
                 "relation": "employed_by", "confidence": 0.8},
                {"source": "Victor Harrington", "target": "Alice Chen",
                 "relation": "knows", "confidence": 0.85},
                {"source": "Alice Chen", "target": "Bob Harrington",
                 "relation": "knows", "confidence": 0.7},
                {"source": "Victor Harrington", "target": "Police Incident Report",
                 "relation": "mentioned_in", "confidence": 1.0},
                {"source": "Alice Chen", "target": "Witness Statement",
                 "relation": "mentioned_in", "confidence": 1.0},
                {"source": "Victor Harrington", "target": "Lab Accident April 1987",
                 "relation": "involved_in", "confidence": 0.75},
                {"source": "Lab Accident April 1987", "target": "Westbrook Industrial Park",
                 "relation": "occurred_at", "confidence": 0.8},
                {"source": "Lab Accident April 1987", "target": "Police Incident Report",
                 "relation": "recorded_in", "confidence": 0.9},
                {"source": "Financial Audit Q3 1987", "target": "Alpine Consulting",
                 "relation": "mentions", "confidence": 0.85},
            ])

        return "No structured output found."


class OpenAICompatibleLLM(LLM):
    """Thin wrapper around the OpenAI-compatible chat completions endpoint."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        import requests
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        resolved_key = api_key or os.environ.get("LLM_API_KEY")
        if resolved_key:
            self._headers["Authorization"] = f"Bearer {resolved_key}"

    def generate(self, prompt: str) -> str:
        import requests
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        body = resp.json()
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        return message.get("content", "") or message.get("reasoning_content", "") or ""


def build_llm(args: argparse.Namespace) -> LLM:
    if args.base_url:
        model = args.model or "llama3"
        print(f"LLM: {args.base_url}  (model: {model})")
        return OpenAICompatibleLLM(
            model=model,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    print("LLM: mock (deterministic, no API key required)")
    return MockLLM()


def read_text_files(data_dir: Path) -> dict[str, str]:
    """Read all .txt files from the data directory."""
    documents: dict[str, str] = {}
    if not data_dir.exists():
        print(f"Warning: data directory '{data_dir}' not found.")
        return documents

    for filepath in sorted(data_dir.glob("*.txt")):
        try:
            text = filepath.read_text(encoding="utf-8")
            documents[filepath.name] = text
        except Exception as e:
            print(f"  Skipping {filepath.name}: {e}")

    return documents


def setup_schema(actions: ActionRegistry) -> None:
    """Create Object Types and Link Types for the murder mystery domain."""
    existing = actions.store.get_all_object_types()
    if existing:
        print("Schema already exists, skipping schema creation.")
        return

    print("Creating schema...")

    actions.create_object_type(
        "Person",
        [
            {"name": "name", "data_type": "string", "required": True},
            {"name": "role", "data_type": "string", "required": False},
        ],
        "A person mentioned in the corpus",
        source="schema_setup",
        agent="research_pipeline",
    )

    actions.create_object_type(
        "Organization",
        [
            {"name": "name", "data_type": "string", "required": True},
        ],
        "An organization mentioned in the corpus",
        source="schema_setup",
        agent="research_pipeline",
    )

    actions.create_object_type(
        "Document",
        [
            {"name": "name", "data_type": "string", "required": True},
        ],
        "A document in the corpus",
        source="schema_setup",
        agent="research_pipeline",
    )

    actions.create_object_type(
        "Event",
        [
            {"name": "name", "data_type": "string", "required": True},
            {"name": "date", "data_type": "string", "required": False},
        ],
        "An event described in the corpus",
        source="schema_setup",
        agent="research_pipeline",
    )

    actions.create_object_type(
        "Location",
        [
            {"name": "name", "data_type": "string", "required": True},
        ],
        "A location mentioned in the corpus",
        source="schema_setup",
        agent="research_pipeline",
    )

    actions.create_object_type(
        "FinancialRecord",
        [
            {"name": "name", "data_type": "string", "required": True},
        ],
        "A financial record",
        source="schema_setup",
        agent="research_pipeline",
    )

    # Link Types
    link_types = [
        ("EMPLOYED_BY", "Person", "Organization", "many_to_one"),
        ("KNOWS", "Person", "Person", "many_to_many"),
        ("MENTIONED_IN", "Person", "Document", "many_to_many"),
        ("INVOLVED_IN", "Person", "Event", "many_to_many"),
        ("OCCURRED_AT", "Event", "Location", "many_to_one"),
        ("RECORDED_IN", "Event", "Document", "many_to_many"),
        ("LOCATED_IN", "Organization", "Location", "many_to_many"),
        ("RELATED_TO", "Event", "Event", "many_to_many"),
    ]

    for name, src, tgt, card in link_types:
        actions.create_link_type(name, src, tgt, card, "", "schema_setup", "research_pipeline")

    print(f"  Created {len(link_types)} link types")


def _normalise_entity_name(raw: str) -> str:
    """Normalise an entity name extracted by the LLM."""
    return raw.strip().strip('"\'')


def _map_to_object_type(raw_type: str) -> str:
    """Map LLM entity type strings to canonical Object Type names."""
    mapping = {
        "person": "Person",
        "organization": "Organization",
        "organisation": "Organization",
        "company": "Organization",
        "firm": "Organization",
        "document": "Document",
        "report": "Document",
        "statement": "Document",
        "letter": "Document",
        "event": "Event",
        "incident": "Event",
        "accident": "Event",
        "location": "Location",
        "place": "Location",
        "financial_record": "FinancialRecord",
    }
    return mapping.get(raw_type.strip().lower(), raw_type)


def _map_relation_type(raw_rel: str) -> str | None:
    """Map LLM relation strings to Link Type names. Returns None if unmappable."""
    mapping = {
        "employed_by": "EMPLOYED_BY",
        "works_at": "EMPLOYED_BY",
        "employs": "EMPLOYED_BY",
        "knows": "KNOWS",
        "acquainted_with": "KNOWS",
        "mentioned_in": "MENTIONED_IN",
        "appears_in": "MENTIONED_IN",
        "involved_in": "INVOLVED_IN",
        "participated_in": "INVOLVED_IN",
        "occurred_at": "OCCURRED_AT",
        "happened_at": "OCCURRED_AT",
        "recorded_in": "RECORDED_IN",
        "documented_in": "RECORDED_IN",
        "located_in": "LOCATED_IN",
        "based_in": "LOCATED_IN",
        "related_to": "RELATED_TO",
        "mentions": "RELATED_TO",
    }
    return mapping.get(raw_rel.strip().lower().replace(" ", "_"))


def extract_entities(llm: LLM, text: str, doc_name: str) -> list[dict[str, Any]]:
    """Use LLM to extract entities from document text."""
    prompt = f"""Extract named entities from the following document. Return a JSON list of objects, each with "name" and "type" fields.

Entity types to use: person, organization, document, event, location, financial_record.

Document name: {doc_name}

Text:
{text[:6000]}

Return ONLY a JSON array, nothing else."""

    response = llm.generate(prompt)
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            data = json.loads(json_match.group())
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def extract_relationships(
    llm: LLM, text: str, doc_name: str, entity_names: list[str]
) -> list[dict[str, Any]]:
    """Use LLM to extract relationships between entities."""
    entities_str = ", ".join(entity_names[:20])
    prompt = f"""Extract relationships between entities from the following document. Return a JSON list of objects with "source", "target", "relation", and "confidence" (0-1) fields.

Known entities in this document: {entities_str}

Use these relation types: employed_by, knows, mentioned_in, involved_in, occurred_at, recorded_in, located_in, related_to.

Document name: {doc_name}

Text:
{text[:6000]}

Return ONLY a JSON array, nothing else."""

    response = llm.generate(prompt)
    try:
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            data = json.loads(json_match.group())
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def run_pipeline(args: argparse.Namespace) -> int:
    db_path = args.db or str(Path(__file__).parent / "research.db")
    data_dir = Path(args.data_dir or str(_root / "data"))

    llm = build_llm(args)

    # Initialize store and action registry
    store = OntologyStore(db_path)
    actions = ActionRegistry(store)
    query = QueryAPI(store)

    try:
        # Step 1: Set up schema
        setup_schema(actions)

        # Step 2: Check if we already have data
        existing_count = store.count_objects()
        if existing_count > 0 and not args.reingest:
            print(f"\nDatabase already has {existing_count} objects. Use --reingest to re-extract.")
            print(f"Run 'uv run python -m cli.main --db {db_path} inspect' to view.")
            return 0

        # Step 3: Read documents
        documents = read_text_files(data_dir)
        if not documents:
            print("No .txt files found in data directory. Nothing to process.")
            return 0

        print(f"\nFound {len(documents)} document(s) to process.")

        # Step 4: Use mock data if using MockLLM (since mock can't actually read text)
        if isinstance(llm, MockLLM):
            print("\nUsing deterministic mock extraction (no real LLM).")
            print("Pass --base-url and --model to use a real LLM.\n")

            # Use pre-defined extraction results
            entities = [
                {"name": "Victor Harrington", "type": "person"},
                {"name": "Alice Chen", "type": "person"},
                {"name": "Bob Harrington", "type": "person"},
                {"name": "Alpine Consulting", "type": "organization"},
                {"name": "MedTech Industries", "type": "organization"},
                {"name": "Westbrook Chronicle", "type": "organization"},
                {"name": "Police Incident Report", "type": "document"},
                {"name": "Witness Statement (Chen)", "type": "document"},
                {"name": "Financial Audit Q3 1987", "type": "document"},
                {"name": "Harrington Letter to Lawyer", "type": "document"},
                {"name": "MedTech Internal Memo", "type": "document"},
                {"name": "Lab Accident April 1987", "type": "event"},
                {"name": "Westbrook Industrial Park", "type": "location"},
            ]

            relationships = [
                {"source": "Victor Harrington", "target": "Alpine Consulting",
                 "relation": "employed_by", "confidence": 0.95},
                {"source": "Alice Chen", "target": "Alpine Consulting",
                 "relation": "employed_by", "confidence": 0.95},
                {"source": "Bob Harrington", "target": "Alpine Consulting",
                 "relation": "employed_by", "confidence": 0.85},
                {"source": "Victor Harrington", "target": "Alice Chen",
                 "relation": "knows", "confidence": 0.9},
                {"source": "Alice Chen", "target": "Bob Harrington",
                 "relation": "knows", "confidence": 0.7},
                {"source": "Victor Harrington", "target": "Police Incident Report",
                 "relation": "mentioned_in", "confidence": 1.0},
                {"source": "Alice Chen", "target": "Witness Statement (Chen)",
                 "relation": "mentioned_in", "confidence": 1.0},
                {"source": "Victor Harrington", "target": "Lab Accident April 1987",
                 "relation": "involved_in", "confidence": 0.8},
                {"source": "Lab Accident April 1987", "target": "Westbrook Industrial Park",
                 "relation": "occurred_at", "confidence": 0.85},
                {"source": "Lab Accident April 1987", "target": "Police Incident Report",
                 "relation": "recorded_in", "confidence": 0.9},
                {"source": "MedTech Industries", "target": "Westbrook Industrial Park",
                 "relation": "located_in", "confidence": 0.7},
            ]
        else:
            # Real LLM: extract from each document
            all_entities: list[dict[str, Any]] = []
            all_relationships: list[dict[str, Any]] = []
            seen_names: set[str] = set()

            for doc_name, text in documents.items():
                print(f"  Processing {doc_name} ({len(text)} chars)...")

                ents = extract_entities(llm, text, doc_name)
                for e in ents:
                    if e.get("name", "").strip() not in seen_names:
                        seen_names.add(e["name"].strip())
                        all_entities.append(e)

                entity_names = [e["name"] for e in all_entities]
                rels = extract_relationships(llm, text, doc_name, entity_names)
                all_relationships.extend(rels)

            entities = all_entities
            relationships = all_relationships

        # Step 5: Write entities through the action layer
        print(f"\nWriting {len(entities)} entities through Action Types...")
        entity_id_map: dict[str, str] = {}
        created = 0
        skipped = 0

        for ent in entities:
            name = _normalise_entity_name(ent["name"])
            ot = _map_to_object_type(ent.get("type", ""))

            if name in entity_id_map:
                skipped += 1
                continue

            try:
                result = actions.upsert_object(
                    object_type=ot,
                    properties={"name": name},
                    source="document_corpus",
                    agent="research_pipeline",
                    confidence=0.9,
                )
                entity_id_map[name] = result["id"]
                if result["created"]:
                    created += 1
            except ActionError as e:
                print(f"    Warning: could not write '{name}' ({ot}): {e}")

        print(f"  {created} created, {skipped} deduplicated")

        # Step 6: Write relationships through the action layer
        print(f"\nWriting {len(relationships)} relationships through Action Types...")
        links_created = 0
        links_skipped = 0

        for rel in relationships:
            src_name = _normalise_entity_name(rel["source"])
            tgt_name = _normalise_entity_name(rel["target"])
            rel_type = _map_relation_type(rel.get("relation", ""))

            if rel_type is None:
                continue

            src_id = entity_id_map.get(src_name)
            tgt_id = entity_id_map.get(tgt_name)

            if src_id is None or tgt_id is None:
                # Auto-create missing entities
                if src_id is None:
                    try:
                        r = actions.upsert_object(
                            "Person", {"name": src_name},
                            source="document_corpus", agent="research_pipeline",
                        )
                        src_id = r["id"]
                        entity_id_map[src_name] = src_id
                    except ActionError:
                        continue

                if tgt_id is None:
                    try:
                        r = actions.upsert_object(
                            "Person", {"name": tgt_name},
                            source="document_corpus", agent="research_pipeline",
                        )
                        tgt_id = r["id"]
                        entity_id_map[tgt_name] = tgt_id
                    except ActionError:
                        continue

            try:
                result = actions.upsert_link(
                    link_type=rel_type,
                    source_object_id=src_id,
                    target_object_id=tgt_id,
                    source="document_corpus",
                    agent="research_pipeline",
                    confidence=float(rel.get("confidence", 0.5)),
                )
                if result["created"]:
                    links_created += 1
            except ActionError as e:
                print(f"    Warning: could not link {src_name} -[{rel_type}]-> {tgt_name}: {e}")

        print(f"  {links_created} created, {links_skipped} skipped")

        # Step 7: Display results
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")

        summary = query.summary()
        print(f"\nDatabase summary:")
        print(f"  Objects:     {summary['object_count']}")
        print(f"  Links:       {summary['link_count']}")
        print(f"  Obj Types:   {summary['object_type_count']}")
        print(f"  Link Types:  {summary['link_type_count']}")

        print(f"\nEntities by type:")
        for ot_def in store.get_all_object_types():
            objs = query.find_objects(object_type=ot_def.name)
            if objs:
                names = [o["properties"].get("name", o["id"][:8]) for o in objs]
                print(f"  {ot_def.name} ({len(objs)}): {', '.join(names[:8])}")

        print(f"\nProvenance check — first 3 objects:")
        all_objs = store.find_objects()
        for obj in all_objs[:3]:
            prov = query.get_provenance(obj["id"])
            if prov:
                latest = prov[-1]
                print(f"  {obj['properties'].get('name', obj['id'][:8])} "
                      f"← {latest['source']} via {latest['agent']} "
                      f"(action: {latest['action_type']}, {latest['timestamp'][:19]})")

        print(f"\nDatabase file: {Path(db_path).resolve()}")
        print(f"Run 'uv run python -m cli.main --db {db_path} inspect' to explore.")
        print(f"Run 'uv run python -m cli.main --db {db_path} export --pretty' to dump as JSON.")

        return 0

    finally:
        store.close()


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
