"""
Thin CLI for the Ontology Harness.

Subcommands:
    init       Create a new ontology.db
    inspect    Print schema summary + data counts
    ingest     Read files from data/ and extract entities via LLM
    export     Dump full ontology as JSON to stdout
    validate   Run schema + referential integrity checks

Usage:
    uv run python -m cli.main init [--db ontology.db]
    uv run python -m cli.main ingest [--db ontology.db] [--data-dir data/] [--api-key ...] [--base-url ...]
    uv run python -m cli.main inspect [--db ontology.db]
    uv run python -m cli.main export [--db ontology.db] [--pretty]
    uv run python -m cli.main validate [--db ontology.db]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root on path
_src_root = Path(__file__).resolve().parent.parent
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from core.store import OntologyStore, SchemaVersionMismatch
from core.actions import ActionRegistry, ActionError
from core.query import QueryAPI


def cmd_init(args: argparse.Namespace) -> int:
    """Create a new empty ontology database."""
    db_path = args.db or "ontology.db"
    if Path(db_path).exists():
        print(f"Error: '{db_path}' already exists. Delete it first or use a different name.", file=sys.stderr)
        return 1

    store = OntologyStore(db_path)
    store.close()
    print(f"Created empty ontology database: {db_path}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print schema summary and data counts."""
    db_path = args.db or "ontology.db"
    if not Path(db_path).exists():
        print(f"Error: '{db_path}' does not exist. Run 'init' first.", file=sys.stderr)
        return 1

    store = OntologyStore(db_path)
    query = QueryAPI(store)

    summary = query.summary()
    schema = query.get_schema()

    print(f"Database: {Path(db_path).resolve()}")
    print(f"  Objects:     {summary['object_count']}")
    print(f"  Links:       {summary['link_count']}")
    print(f"  Obj Types:   {summary['object_type_count']}")
    print(f"  Link Types:  {summary['link_type_count']}")
    print()

    if schema["object_types"]:
        print("Object Types:")
        for ot in schema["object_types"]:
            props = ", ".join(
                f"{p['name']}:{p['data_type']}{'*' if p['required'] else ''}"
                for p in ot["properties"]
            )
            desc = f" — {ot['description']}" if ot["description"] else ""
            print(f"  {ot['name']}  ({props}){desc}")
    else:
        print("Object Types: (none defined)")

    print()

    if schema["link_types"]:
        print("Link Types:")
        for lt in schema["link_types"]:
            desc = f" — {lt['description']}" if lt["description"] else ""
            print(f"  {lt['name']}: {lt['source_type']} → {lt['target_type']}  [{lt['cardinality']}]{desc}")
    else:
        print("Link Types: (none defined)")

    store.close()
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export the full ontology as JSON to stdout."""
    db_path = args.db or "ontology.db"
    if not Path(db_path).exists():
        print(f"Error: '{db_path}' does not exist. Run 'init' first.", file=sys.stderr)
        return 1

    store = OntologyStore(db_path)
    query = QueryAPI(store)

    schema = query.get_schema()

    objects = []
    for ot in schema["object_types"]:
        objs = query.find_objects(object_type=ot["name"])
        for obj in objs:
            obj["provenance"] = query.get_provenance(obj["id"])
        objects.extend(objs)

    links = []
    for lt in schema["link_types"]:
        lks = store.find_links(link_type=lt["name"])
        for link in lks:
            link["provenance"] = query.get_provenance(link["id"])
        links.extend(lks)

    export_data = {
        "schema": schema,
        "objects": objects,
        "links": links,
        "summary": query.summary(),
    }

    indent = 2 if args.pretty else None
    json.dump(export_data, sys.stdout, indent=indent, ensure_ascii=False)
    if args.pretty:
        print()

    store.close()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run schema and referential integrity checks."""
    db_path = args.db or "ontology.db"
    if not Path(db_path).exists():
        print(f"Error: '{db_path}' does not exist. Run 'init' first.", file=sys.stderr)
        return 1

    store = OntologyStore(db_path)
    errors: list[str] = []
    warnings: list[str] = []

    object_types = {ot.name: ot for ot in store.get_all_object_types()}
    link_types = {lt.name: lt for lt in store.get_all_link_types()}

    # Check: objects have valid object_type
    for obj in store.find_objects(include_deleted=True):
        ot = obj["object_type"]
        if ot not in object_types:
            errors.append(
                f"Object {obj['id']} has undeclared object_type '{ot}'"
            )

    # Check: links have valid link_type and valid source/target objects
    for link in store.find_links(include_deleted=True):
        lt_name = link["link_type"]
        if lt_name not in link_types:
            errors.append(
                f"Link {link['id']} has undeclared link_type '{lt_name}'"
            )
        else:
            lt = link_types[lt_name]
            src_obj = store.get_object(link["source_object_id"], include_deleted=True)
            tgt_obj = store.get_object(link["target_object_id"], include_deleted=True)

            if src_obj is None:
                errors.append(
                    f"Link {link['id']} references nonexistent source object "
                    f"'{link['source_object_id']}'"
                )
            elif src_obj["object_type"] != lt.source_type:
                errors.append(
                    f"Link {link['id']}: source object {link['source_object_id']} "
                    f"has type '{src_obj['object_type']}' but link type '{lt_name}' "
                    f"expects '{lt.source_type}'"
                )

            if tgt_obj is None:
                errors.append(
                    f"Link {link['id']} references nonexistent target object "
                    f"'{link['target_object_id']}'"
                )
            elif tgt_obj["object_type"] != lt.target_type:
                errors.append(
                    f"Link {link['id']}: target object {link['target_object_id']} "
                    f"has type '{tgt_obj['object_type']}' but link type '{lt_name}' "
                    f"expects '{lt.target_type}'"
                )

    # Check: link types reference valid object types
    for lt in link_types.values():
        if lt.source_type not in object_types:
            errors.append(
                f"Link type '{lt.name}' references nonexistent source type '{lt.source_type}'"
            )
        if lt.target_type not in object_types:
            errors.append(
                f"Link type '{lt.name}' references nonexistent target type '{lt.target_type}'"
            )

    # Check: all non-deleted objects have at least one provenance record
    for obj in store.find_objects():
        prov = store.get_provenance(obj["id"], target_type="object")
        if not prov:
            warnings.append(f"Object {obj['id']} has no provenance records")

    for link in store.find_links():
        prov = store.get_provenance(link["id"], target_type="link")
        if not prov:
            warnings.append(f"Link {link['id']} has no provenance records")

    if errors:
        print(f"VALIDATION ERRORS ({len(errors)}):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)

    if warnings:
        print(f"VALIDATION WARNINGS ({len(warnings)}):", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)

    if not errors and not warnings:
        print("Validation passed — no issues found.")
        store.close()
        return 0
    elif errors:
        print(f"\nValidation FAILED with {len(errors)} error(s).")
        store.close()
        return 1
    else:
        print(f"\nValidation passed with {len(warnings)} warning(s).")
        store.close()
        return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest files from a directory, extract entities via LLM, write to store."""
    db_path = args.db or "ontology.db"
    data_dir = Path(args.data_dir or str(_src_root / "data"))
    if not Path(db_path).exists():
        print(f"Error: '{db_path}' does not exist. Run 'init' first.", file=sys.stderr)
        return 1

    store = OntologyStore(db_path)
    actions = ActionRegistry(store)
    query = QueryAPI(store)

    try:
        if not store.get_all_object_types():
            _setup_ingest_schema(actions)

        # Read files: .txt and .pdf
        docs: dict[str, str] = {}
        for suffix, reader in [(".txt", _read_txt), (".pdf", _read_pdf)]:
            for fp in sorted(data_dir.glob(f"*{suffix}")):
                try:
                    text = reader(fp)
                    if text.strip():
                        docs[fp.name] = text
                except Exception:
                    pass

        if not docs:
            print(f"No .txt or .pdf files found in {data_dir}")
            return 0

        llm = _build_ingest_llm(args)
        print(f"Ingesting {len(docs)} file(s) from {data_dir} ...")

        ents_raw: list[dict[str, Any]] = []
        rels_raw: list[dict[str, Any]] = []
        seen: set[str] = set()

        for name, text in docs.items():
            print(f"  {name} ({len(text)} chars)")

            prompt = (
                f"Extract named entities from this document as a JSON array of "
                f'{{"name","type"}} where type is one of: person, organization, '
                f"document, event, location, concept, method, dataset.\n\n"
                f"Document: {text[:6000]}\n\nJSON:"
            )
            try:
                raw = llm.generate(prompt)
                m = re.search(r"\[[\s\S]*\]", raw)
                if m:
                    for e in json.loads(m.group()):
                        n = e.get("name", "").strip()
                        if n and n not in seen:
                            seen.add(n)
                            ents_raw.append(e)
            except Exception as e:
                print(f"    entity extraction failed: {e}")

            if ents_raw:
                names = [e["name"] for e in ents_raw]
                prompt = (
                    f"Extract relationships between these entities from the document "
                    f"as a JSON array of {{'source','target','relation','confidence'}}. "
                    f"Use relation types: employs, authored_by, cites, describes, "
                    f"related_to, part_of, uses, evaluates.\n\n"
                    f"Entities: {', '.join(names[-15:])}\n"
                    f"Document: {text[:4000]}\n\nJSON:"
                )
                try:
                    raw = llm.generate(prompt)
                    m = re.search(r"\[[\s\S]*\]", raw)
                    if m:
                        for r in json.loads(m.group()):
                            rels_raw.append(r)
                except Exception as e:
                    print(f"    relationship extraction failed: {e}")

        if not ents_raw:
            print("No entities extracted.")
            return 0

        print(f"\nWriting {len(ents_raw)} entities ...")
        eid_map: dict[str, str] = {}
        _OT_MAP = {
            "person": "Person", "organization": "Organization",
            "document": "Document", "event": "Event",
            "location": "Location", "concept": "Concept",
            "method": "Concept", "dataset": "Concept",
            "financial_record": "FinancialRecord",
        }
        for e in ents_raw:
            name = e.get("name", "").strip()
            ot = _OT_MAP.get(e.get("type", "").strip().lower(), "Concept")
            if name and name not in eid_map:
                try:
                    r = actions.upsert_object(ot, {"name": name}, source=name, agent="cli_ingest")
                    eid_map[name] = r["id"]
                except ActionError as exc:
                    print(f"  skip '{name}': {exc}")

        _REL_MAP = {
            "employs": "EMPLOYED_BY", "works_at": "EMPLOYED_BY",
            "employed_by": "EMPLOYED_BY",
            "knows": "KNOWS", "authored_by": "AUTHORED_BY",
            "mentioned_in": "MENTIONED_IN", "cites": "CITES",
            "describes": "DESCRIBES", "related_to": "RELATED_TO",
            "part_of": "PART_OF", "uses": "USES", "evaluates": "EVALUATES",
        }
        linked = 0
        for r in rels_raw:
            lt = _REL_MAP.get(r.get("relation", "").strip().lower().replace(" ", "_"))
            if not lt:
                continue
            sid = eid_map.get(r.get("source", "").strip())
            tid = eid_map.get(r.get("target", "").strip())
            if sid and tid:
                try:
                    actions.upsert_link(lt, sid, tid, source="cli_ingest",
                                        agent="cli_ingest",
                                        confidence=float(r.get("confidence", 0.5)))
                    linked += 1
                except ActionError:
                    pass

        print(f"  {len(eid_map)} objects, {linked} links written")
        s = query.summary()
        print(f"\nDatabase now has {s['object_count']} objects, {s['link_count']} links")

    finally:
        store.close()
    return 0


def _read_txt(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="replace")


def _read_pdf(fp: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(fp)
    except ImportError:
        pass
    try:
        import pypdf
        reader = pypdf.PdfReader(str(fp))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except ImportError:
        pass
    raise RuntimeError("No PDF library available (install pdfminer.six or pypdf)")


def _setup_ingest_schema(actions: ActionRegistry) -> None:
    """Create default schema for document ingestion."""
    for name, props in [
        ("Person", [{"name": "name", "data_type": "string", "required": True}]),
        ("Organization", [{"name": "name", "data_type": "string", "required": True}]),
        ("Document", [{"name": "name", "data_type": "string", "required": True}]),
        ("Event", [{"name": "name", "data_type": "string", "required": True}]),
        ("Location", [{"name": "name", "data_type": "string", "required": True}]),
        ("Concept", [{"name": "name", "data_type": "string", "required": True}]),
        ("FinancialRecord", [{"name": "name", "data_type": "string", "required": True}]),
    ]:
        actions.create_object_type(name, props, "", "schema_auto", "cli_ingest")
    for name, src, tgt, card in [
        ("EMPLOYED_BY", "Person", "Organization", "many_to_one"),
        ("KNOWS", "Person", "Person", "many_to_many"),
        ("MENTIONED_IN", "Person", "Document", "many_to_many"),
        ("AUTHORED_BY", "Document", "Person", "many_to_one"),
        ("CITES", "Document", "Document", "many_to_many"),
        ("DESCRIBES", "Document", "Concept", "many_to_many"),
        ("RELATED_TO", "Concept", "Concept", "many_to_many"),
        ("PART_OF", "Concept", "Concept", "many_to_many"),
        ("USES", "Concept", "Concept", "many_to_many"),
        ("EVALUATES", "Document", "Concept", "many_to_many"),
    ]:
        actions.create_link_type(name, src, tgt, card, "", "schema_auto", "cli_ingest")


def _build_ingest_llm(args: argparse.Namespace) -> Any:
    """Build an LLM instance from CLI args."""

    class MockLLM:
        def generate(self, prompt: str) -> str:
            p = prompt.lower()
            if "relationship" in p or "relation" in p:
                return "[]"
            if "entit" not in p or "extract" not in p:
                return "[]"

            # Extract the document text from the prompt
            doc_start = prompt.find("Document:")
            if doc_start == -1:
                return "[]"
            doc_text = prompt[doc_start + 9:]

            items: list[dict[str, str]] = []
            seen_n: set[str] = set()

            # Match capitalized multi-word names (e.g. "Victor Harrington", "Alpine Consulting")
            for m in re.finditer(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b', doc_text):
                n = m.group(1).strip()
                skip = {"The", "This", "These", "Those", "Each", "Some", "Many",
                         "After", "Before", "During", "While", "Although",
                         "However", "Therefore", "Because", "Since", "Unless",
                         "Document", "Extract", "JSON", "Return", "Only", "Abstract",
                         "Introduction", "Method", "Results", "Discussion",
                         "Conclusion", "References", "Appendix"}
                if n in skip or len(n) < 4:
                    continue
                if n not in seen_n:
                    seen_n.add(n)
                    nl = n.lower()
                    if any(w in nl for w in ("inc", "corp", "llc", "ltd", "group", "consulting",
                                              "lab", "laboratory", "institute", "university",
                                              "college", "company", "industries", "bank", "trust",
                                              "partners", "associates", "systems", "solutions")):
                        t = "organization"
                    elif any(w in nl for w in ("report", "statement", "letter", "memo",
                                                "audit", "diary", "obituary", "paper",
                                                "article", "journal", "proceedings")):
                        t = "document"
                    elif any(w in nl for w in ("accident", "incident", "conference",
                                                "workshop", "summit", "war", "revolution")):
                        t = "event"
                    elif any(w in nl for w in ("park", "street", "avenue", "road",
                                                "city", "county", "state", "country",
                                                "river", "mountain", "lake", "ocean")):
                        t = "location"
                    elif any(w in nl for w in ("algorithm", "method", "model",
                                                "framework", "architecture", "theory",
                                                "dataset", "corpus", "benchmark",
                                                "transformer", "bert", "gpt", "llm",
                                                "neural", "network", "attention",
                                                "embedding", "tokenizer")):
                        t = "concept"
                    else:
                        t = "person"
                    items.append({"name": n, "type": t})

            # Match all-caps acronyms (e.g. "BERT", "GPT", "LLM", "CNN")
            for m in re.finditer(r'\b([A-Z]{2,6})\b', doc_text):
                n = m.group(1)
                if n in ("JSON", "PDF", "URL", "API", "HTTP", "ISBN", "DOI"):
                    continue
                if n not in seen_n:
                    seen_n.add(n)
                    items.append({"name": n, "type": "concept"})

            # Match quoted phrases as named entities
            for m in re.finditer(r'"([^"]{3,80})"', doc_text):
                n = m.group(1).strip()
                if n not in seen_n:
                    seen_n.add(n)
                    items.append({"name": n, "type": "concept"})

            # Match email addresses as persons
            for m in re.finditer(r'\b([\w.-]+@[\w.-]+\.\w+)\b', doc_text):
                items.append({"name": m.group(1), "type": "person"})

            # Match DOIs / URLs as documents
            for m in re.finditer(r'(10\.\d{4,}/[\w./-]+)', doc_text):
                items.append({"name": m.group(1), "type": "document"})

            return json.dumps(items[:20])

    if args.base_url:
        import requests
        model = args.model or "llama3"
        base = args.base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        key = args.api_key or os.environ.get("LLM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"

        class RealLLM:
            def generate(self, prompt: str) -> str:
                r = requests.post(
                    f"{base}/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3, "max_tokens": 4096,
                    },
                    timeout=300,
                )
                r.raise_for_status()
                msg = r.json()["choices"][0].get("message", {})
                return msg.get("content", "") or msg.get("reasoning_content", "") or "[]"

        print(f"Ingest LLM: {base} (model: {model})")
        return RealLLM()

    print("Ingest LLM: mock (regex heuristic, no API key needed)")
    return MockLLM()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ontology Harness — CLI for managing ontology databases",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite database file (default: ontology.db)",
    )

    sub = parser.add_subparsers(dest="command")

    sub_init = sub.add_parser("init", help="Create a new empty ontology database")
    sub_init.set_defaults(func=cmd_init)

    sub_inspect = sub.add_parser("inspect", help="Print schema summary and data counts")
    sub_inspect.set_defaults(func=cmd_inspect)

    sub_ingest = sub.add_parser("ingest", help="Read files from data/ and extract entities via LLM")
    sub_ingest.add_argument("--data-dir", default=None, help="Directory with .txt/.pdf files (default: data/)")
    sub_ingest.add_argument("--base-url", default=None, help="LLM API base URL (omit for regex mock)")
    sub_ingest.add_argument("--model", default=None, help="LLM model name")
    sub_ingest.add_argument("--api-key", default=None, help="LLM API key")
    sub_ingest.set_defaults(func=cmd_ingest)

    sub_export = sub.add_parser("export", help="Dump full ontology as JSON to stdout")
    sub_export.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    sub_export.set_defaults(func=cmd_export)

    sub_validate = sub.add_parser("validate", help="Run schema and referential integrity checks")
    sub_validate.set_defaults(func=cmd_validate)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except SchemaVersionMismatch as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
