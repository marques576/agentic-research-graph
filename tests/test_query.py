"""Tests for core/query.py — Read API, neighbors, and traversal."""

import os
import tempfile

import pytest

from core.store import OntologyStore
from core.actions import ActionRegistry
from core.query import QueryAPI


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = OntologyStore(path)
    yield s
    s.close()
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def actions(store):
    return ActionRegistry(store)


@pytest.fixture
def query(store):
    return QueryAPI(store)


def _setup_mystery_ontology(actions):
    """Create a small murder-mystery fixture ontology."""
    actions.create_object_type(
        "Person",
        [
            {"name": "name", "data_type": "string", "required": True},
            {"name": "role", "data_type": "string", "required": False},
        ],
        "A person involved in the case",
        source="test_setup",
        agent="test_agent",
    )
    actions.create_object_type(
        "Organization",
        [
            {"name": "name", "data_type": "string", "required": True},
        ],
        "An organization",
        source="test_setup",
        agent="test_agent",
    )
    actions.create_object_type(
        "Document",
        [
            {"name": "name", "data_type": "string", "required": True},
            {"name": "date", "data_type": "datetime", "required": False},
        ],
        "A document in the corpus",
        source="test_setup",
        agent="test_agent",
    )

    actions.create_link_type(
        "EMPLOYED_BY",
        "Person",
        "Organization",
        "many_to_one",
        "Employment relationship",
        source="test_setup",
        agent="test_agent",
    )
    actions.create_link_type(
        "KNOWS",
        "Person",
        "Person",
        "many_to_many",
        "Acquaintance relationship",
        source="test_setup",
        agent="test_agent",
    )
    actions.create_link_type(
        "MENTIONED_IN",
        "Person",
        "Document",
        "many_to_many",
        "Person mentioned in document",
        source="test_setup",
        agent="test_agent",
    )


def _populate_mystery_data(actions):
    """Create Person, Organization, Document objects and links."""
    # People
    victor = actions.upsert_object(
        "Person", {"name": "Victor Harrington", "role": "Victim"},
        source="police_report", agent="test", confidence=1.0,
    )
    alice = actions.upsert_object(
        "Person", {"name": "Alice Chen", "role": "Witness"},
        source="witness_statement", agent="test", confidence=0.95,
    )
    bob = actions.upsert_object(
        "Person", {"name": "Bob Harrington", "role": "Suspect"},
        source="police_report", agent="test", confidence=0.8,
    )

    # Organization
    alpine = actions.upsert_object(
        "Organization", {"name": "Alpine Consulting"},
        source="registry", agent="test", confidence=0.95,
    )

    # Document
    report = actions.upsert_object(
        "Document", {"name": "Police Incident Report"},
        source="file_ingest", agent="test", confidence=1.0,
    )

    # Links
    actions.upsert_link(
        "EMPLOYED_BY", victor["id"], alpine["id"],
        source="registry", agent="test",
    )
    actions.upsert_link(
        "EMPLOYED_BY", alice["id"], alpine["id"],
        source="registry", agent="test",
    )
    actions.upsert_link(
        "EMPLOYED_BY", bob["id"], alpine["id"],
        source="registry", agent="test",
    )
    actions.upsert_link(
        "KNOWS", victor["id"], alice["id"],
        source="witness_statement", agent="test",
    )
    actions.upsert_link(
        "KNOWS", alice["id"], bob["id"],
        source="witness_statement", agent="test",
    )
    actions.upsert_link(
        "MENTIONED_IN", victor["id"], report["id"],
        source="file_ingest", agent="test",
    )

    return {
        "victor": victor,
        "alice": alice,
        "bob": bob,
        "alpine": alpine,
        "report": report,
    }


class TestGetObject:
    def test_existing(self, actions, query):
        _setup_mystery_ontology(actions)
        victor = actions.upsert_object(
            "Person", {"name": "Victor Harrington"},
            source="doc", agent="test",
        )
        obj = query.get_object(victor["id"])
        assert obj is not None
        assert obj["object_type"] == "Person"
        assert obj["properties"]["name"] == "Victor Harrington"
        assert "provenance" in obj
        assert len(obj["provenance"]) == 1
        assert obj["provenance"][0]["source"] == "doc"

    def test_nonexistent(self, query):
        assert query.get_object("nonexistent-uuid") is None

    def test_deleted(self, actions, query):
        _setup_mystery_ontology(actions)
        victor = actions.upsert_object(
            "Person", {"name": "V"}, source="s", agent="a",
        )
        actions.delete_object(victor["id"], "a", "reason")
        assert query.get_object(victor["id"]) is None


class TestFindObjects:
    def test_by_type(self, actions, query):
        _setup_mystery_ontology(actions)
        data = _populate_mystery_data(actions)

        persons = query.find_objects(object_type="Person")
        assert len(persons) == 3

        orgs = query.find_objects(object_type="Organization")
        assert len(orgs) == 1

    def test_by_property(self, actions, query):
        _setup_mystery_ontology(actions)
        data = _populate_mystery_data(actions)

        results = query.find_objects(
            object_type="Person",
            property_filters={"name": "Alice Chen"},
        )
        assert len(results) == 1
        assert results[0]["properties"]["name"] == "Alice Chen"


class TestGetNeighbors:
    def test_outgoing_only(self, actions, query):
        _setup_mystery_ontology(actions)
        data = _populate_mystery_data(actions)

        neighbors = query.get_neighbors(
            data["victor"]["id"], direction="out"
        )
        neighbor_names = {n["neighbor"]["properties"]["name"] for n in neighbors}
        assert "Alice Chen" in neighbor_names
        assert "Alpine Consulting" in neighbor_names
        assert "Police Incident Report" in neighbor_names

    def test_incoming_only(self, actions, query):
        _setup_mystery_ontology(actions)
        data = _populate_mystery_data(actions)

        neighbors = query.get_neighbors(
            data["alpine"]["id"], direction="in"
        )
        neighbor_names = {n["neighbor"]["properties"]["name"] for n in neighbors}
        assert len(neighbors) == 3
        assert "Victor Harrington" in neighbor_names

    def test_filter_by_link_type(self, actions, query):
        _setup_mystery_ontology(actions)
        data = _populate_mystery_data(actions)

        neighbors = query.get_neighbors(
            data["victor"]["id"], link_type="KNOWS", direction="both"
        )
        neighbor_names = {n["neighbor"]["properties"]["name"] for n in neighbors}
        assert "Alice Chen" in neighbor_names
        # Alpine and Report should not appear since they're not via KNOWS
        assert "Alpine Consulting" not in neighbor_names

    def test_nonexistent_object(self, query):
        assert query.get_neighbors("nonexistent") == []


class TestTraverse:
    def test_single_hop(self, actions, query):
        _setup_mystery_ontology(actions)
        data = _populate_mystery_data(actions)

        result = query.traverse(data["victor"]["id"], max_depth=1)
        assert result["start"] is not None
        # victor → alice (KNOWS), victor → alpine (EMPLOYED_BY), victor → report (MENTIONED_IN)
        assert len(result["paths"]) >= 3

    def test_two_hop(self, actions, query):
        _setup_mystery_ontology(actions)
        data = _populate_mystery_data(actions)

        result = query.traverse(data["victor"]["id"], max_depth=2)
        # Should include: victor→alice→bob (2-hop path)
        two_hop_paths = [p for p in result["paths"] if p["depth"] == 2]
        assert len(two_hop_paths) > 0

        # Check we can find the path victor→alice→bob
        names_paths = []
        for p in result["paths"]:
            names = [o["properties"]["name"] for o in p["path"]]
            names_paths.append(" -> ".join(names))

        assert "Victor Harrington -> Alice Chen -> Bob Harrington" in names_paths

    def test_max_depth_clamp(self, actions, query):
        _setup_mystery_ontology(actions)
        data = _populate_mystery_data(actions)

        result = query.traverse(data["victor"]["id"], max_depth=10)
        # Should be clamped to 3
        max_depth_found = max((p["depth"] for p in result["paths"]), default=0)
        assert max_depth_found <= 3

    def test_filter_by_link_type(self, actions, query):
        _setup_mystery_ontology(actions)
        data = _populate_mystery_data(actions)

        result = query.traverse(
            data["victor"]["id"], max_depth=2, link_type="KNOWS"
        )
        # Only follows KNOWS links
        for p in result["paths"]:
            for edge in p["edges"]:
                assert edge["link_type"] == "KNOWS"

    def test_nonexistent_start(self, query):
        result = query.traverse("nonexistent")
        assert result["start"] is None
        assert result["paths"] == []


class TestGetProvenance:
    def test_object_provenance(self, actions, query):
        _setup_mystery_ontology(actions)
        victor = actions.upsert_object(
            "Person", {"name": "V"}, source="doc", agent="test",
        )
        records = query.get_provenance(victor["id"])
        assert len(records) == 1
        assert records[0]["action_type"] == "upsert_object"

    def test_link_provenance(self, actions, query):
        _setup_mystery_ontology(actions)
        a = actions.upsert_object("Person", {"name": "A"}, "s", "a")
        b = actions.upsert_object("Person", {"name": "B"}, "s", "a")
        actions.create_link_type("COLLEAGUE_OF", "Person", "Person", "many_to_many", "", "s", "a")
        link = actions.upsert_link("COLLEAGUE_OF", a["id"], b["id"], "s", "a")

        records = query.get_provenance(link["id"])
        assert len(records) == 1
        assert records[0]["action_type"] == "upsert_link"


class TestGetSchema:
    def test_empty(self, query):
        schema = query.get_schema()
        assert schema["object_types"] == []
        assert schema["link_types"] == []

    def test_populated(self, actions, query):
        _setup_mystery_ontology(actions)
        schema = query.get_schema()
        assert len(schema["object_types"]) == 3
        assert len(schema["link_types"]) == 3


class TestSummary:
    def test_populated(self, actions, query):
        _setup_mystery_ontology(actions)
        _populate_mystery_data(actions)

        s = query.summary()
        assert s["object_count"] == 5
        assert s["link_count"] == 6
        assert s["object_type_count"] == 3
        assert s["link_type_count"] == 3
