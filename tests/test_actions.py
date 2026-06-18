"""Tests for core/actions.py and core/provenance.py — Action Types and provenance."""

import os
import tempfile

import pytest

from core.store import OntologyStore
from core.actions import ActionRegistry, ActionError
from core.schema import ObjectType, PropertyDef


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


def _setup_person_org(actions):
    """Helper: create Person and Organization object types."""
    actions.create_object_type(
        name="Person",
        properties=[
            {"name": "name", "data_type": "string", "required": True},
            {"name": "age", "data_type": "number", "required": False},
        ],
        description="A human being",
        source="test",
        agent="test_agent",
    )
    actions.create_object_type(
        name="Organization",
        properties=[
            {"name": "name", "data_type": "string", "required": True},
        ],
        description="An organization",
        source="test",
        agent="test_agent",
    )


class TestCreateObjectType:
    def test_basic(self, actions):
        result = actions.create_object_type(
            name="Person",
            properties=[
                {"name": "name", "data_type": "string", "required": True},
                {"name": "age", "data_type": "number", "required": False},
            ],
            description="A person",
            source="test_file.txt",
            agent="test_agent",
        )
        assert result["name"] == "Person"
        assert len(result["properties"]) == 2

        ot = actions.store.get_object_type("Person")
        assert ot is not None
        assert ot.name == "Person"

        records = actions.store.get_provenance("Person", target_type="object_type")
        assert len(records) == 1
        assert records[0]["source"] == "test_file.txt"
        assert records[0]["agent"] == "test_agent"
        assert records[0]["action_type"] == "create_object_type"

    def test_duplicate_fails(self, actions):
        actions.create_object_type("Person", [], "", "test", "test")
        with pytest.raises(ActionError, match="already exists"):
            actions.create_object_type("Person", [], "", "test", "test")

    def test_empty_name_fails(self, actions):
        with pytest.raises(ActionError, match="must not be empty"):
            actions.create_object_type("  ", [], "", "test", "test")

    def test_invalid_data_type(self, actions):
        with pytest.raises(ValueError, match="Invalid data type"):
            actions.create_object_type(
                "T",
                [{"name": "x", "data_type": "foobar"}],
                "",
                "test",
                "test",
            )

    def test_property_defaults(self, actions):
        result = actions.create_object_type(
            "T",
            [{"name": "x", "data_type": "string"}],
            "",
            "test",
            "test",
        )
        assert result["properties"][0]["required"] is False


class TestCreateLinkType:
    def test_basic(self, actions):
        actions.create_object_type("Person", [], "", "test", "test")
        actions.create_object_type("Organization", [], "", "test", "test")

        result = actions.create_link_type(
            name="EMPLOYED_BY",
            source_type="Person",
            target_type="Organization",
            cardinality="many_to_one",
            description="Employment",
            source="test",
            agent="test_agent",
        )
        assert result["name"] == "EMPLOYED_BY"
        assert result["cardinality"] == "many_to_one"

        records = actions.store.get_provenance("EMPLOYED_BY", target_type="link_type")
        assert len(records) == 1

    def test_source_type_missing(self, actions):
        actions.create_object_type("Person", [], "", "test", "test")
        with pytest.raises(ActionError, match="does not exist"):
            actions.create_link_type(
                "EMPLOYED_BY", "Person", "MissingType", "many_to_many", "",
                "test", "test",
            )

    def test_duplicate_fails(self, actions):
        actions.create_object_type("A", [], "", "test", "test")
        actions.create_link_type("KNOWS", "A", "A", "many_to_many", "", "test", "test")
        with pytest.raises(ActionError, match="already exists"):
            actions.create_link_type("KNOWS", "A", "A", "many_to_many", "", "test", "test")


class TestUpsertObject:
    def test_create(self, actions):
        actions.create_object_type("Person", [
            {"name": "name", "data_type": "string", "required": True},
        ], "", "test", "test")

        result = actions.upsert_object(
            object_type="Person",
            properties={"name": "Alice"},
            source="doc1.txt",
            agent="research_agent",
            confidence=0.9,
        )
        assert result["created"] is True
        obj_id = result["id"]

        obj = actions.store.get_object(obj_id)
        assert obj is not None
        assert obj["properties"] == {"name": "Alice"}

        records = actions.store.get_provenance(obj_id, target_type="object")
        assert len(records) == 1
        assert records[0]["source"] == "doc1.txt"
        assert records[0]["agent"] == "research_agent"
        assert records[0]["confidence"] == 0.9
        assert records[0]["action_type"] == "upsert_object"

    def test_update(self, actions):
        actions.create_object_type("Person", [
            {"name": "name", "data_type": "string", "required": True},
        ], "", "test", "test")

        r1 = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")
        r2 = actions.upsert_object("Person", {"name": "Alicia"}, "s2", "a2", id=r1["id"])

        assert r2["created"] is False

        obj = actions.store.get_object(r1["id"])
        assert obj["properties"] == {"name": "Alicia"}

        records = actions.store.get_provenance(r1["id"], target_type="object")
        assert len(records) == 2
        assert records[1]["previous_properties"] is not None

    def test_update_deleted_fails(self, actions):
        actions.create_object_type("Person", [
            {"name": "name", "data_type": "string", "required": True},
        ], "", "test", "test")

        r = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")
        actions.delete_object(r["id"], "a", "reason")

        with pytest.raises(ActionError, match="deleted"):
            actions.upsert_object("Person", {"name": "Bob"}, "s", "a", id=r["id"])

    def test_missing_object_type(self, actions):
        with pytest.raises(ActionError, match="does not exist"):
            actions.upsert_object("MissingType", {}, "s", "a")

    def test_invalid_properties(self, actions):
        actions.create_object_type("Person", [
            {"name": "name", "data_type": "string", "required": True},
        ], "", "test", "test")

        with pytest.raises(ValueError, match="Missing required property"):
            actions.upsert_object("Person", {}, "s", "a")

    def test_wrong_property_type(self, actions):
        actions.create_object_type("Person", [
            {"name": "name", "data_type": "string", "required": True},
        ], "", "test", "test")

        with pytest.raises(ValueError, match="expected string"):
            actions.upsert_object("Person", {"name": 42}, "s", "a")

    def test_unknown_property(self, actions):
        actions.create_object_type("Person", [
            {"name": "name", "data_type": "string", "required": True},
        ], "", "test", "test")

        with pytest.raises(ValueError, match="Unknown property"):
            actions.upsert_object("Person", {"name": "Alice", "extra": "bad"}, "s", "a")


class TestUpsertLink:
    def test_basic(self, actions):
        _setup_person_org(actions)
        actions.create_link_type(
            "EMPLOYED_BY", "Person", "Organization", "many_to_one", "", "test", "test"
        )
        alice = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")
        acme = actions.upsert_object("Organization", {"name": "Acme"}, "s", "a")

        result = actions.upsert_link(
            link_type="EMPLOYED_BY",
            source_object_id=alice["id"],
            target_object_id=acme["id"],
            source="doc2.txt",
            agent="research_agent",
            confidence=0.85,
        )
        assert result["created"] is True

        records = actions.store.get_provenance(result["id"], target_type="link")
        assert len(records) == 1
        assert records[0]["confidence"] == 0.85

    def test_wrong_source_type(self, actions):
        _setup_person_org(actions)
        actions.create_link_type(
            "EMPLOYED_BY", "Person", "Organization", "many_to_one", "", "test", "test"
        )
        acme = actions.upsert_object("Organization", {"name": "Acme"}, "s", "a")
        acme2 = actions.upsert_object("Organization", {"name": "Acme2"}, "s", "a")

        with pytest.raises(ActionError, match="requires source type"):
            actions.upsert_link(
                "EMPLOYED_BY", acme["id"], acme2["id"], "s", "a"
            )

    def test_wrong_target_type(self, actions):
        _setup_person_org(actions)
        actions.create_link_type(
            "EMPLOYED_BY", "Person", "Organization", "many_to_one", "", "test", "test"
        )
        alice = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")
        bob = actions.upsert_object("Person", {"name": "Bob"}, "s", "a")

        with pytest.raises(ActionError, match="requires target type"):
            actions.upsert_link(
                "EMPLOYED_BY", alice["id"], bob["id"], "s", "a"
            )

    def test_missing_link_type(self, actions):
        with pytest.raises(ActionError, match="does not exist"):
            actions.upsert_link("MissingLink", "a", "b", "s", "a")

    def test_missing_source_object(self, actions):
        _setup_person_org(actions)
        actions.create_link_type(
            "EMPLOYED_BY", "Person", "Organization", "many_to_one", "", "test", "test"
        )
        org = actions.upsert_object("Organization", {"name": "Acme"}, "s", "a")

        with pytest.raises(ActionError, match="does not exist"):
            actions.upsert_link("EMPLOYED_BY", "nonexistent", org["id"], "s", "a")

    def test_cardinality_one_to_one_source(self, actions):
        _setup_person_org(actions)
        actions.create_link_type(
            "SPOUSE_OF", "Person", "Person", "one_to_one", "", "test", "test"
        )
        alice = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")
        bob = actions.upsert_object("Person", {"name": "Bob"}, "s", "a")
        charlie = actions.upsert_object("Person", {"name": "Charlie"}, "s", "a")

        actions.upsert_link("SPOUSE_OF", alice["id"], bob["id"], "s", "a")
        with pytest.raises(ActionError, match="cardinality one_to_one"):
            actions.upsert_link("SPOUSE_OF", alice["id"], charlie["id"], "s", "a")

    def test_cardinality_one_to_many_target(self, actions):
        _setup_person_org(actions)
        actions.create_link_type(
            "CEO_OF", "Person", "Organization", "one_to_many", "", "test", "test"
        )
        alice = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")
        bob = actions.upsert_object("Person", {"name": "Bob"}, "s", "a")
        acme = actions.upsert_object("Organization", {"name": "Acme"}, "s", "a")

        actions.upsert_link("CEO_OF", alice["id"], acme["id"], "s", "a")
        with pytest.raises(ActionError, match="cardinality one_to_many"):
            actions.upsert_link("CEO_OF", bob["id"], acme["id"], "s", "a")

    def test_many_to_many_allows_multiple(self, actions):
        _setup_person_org(actions)
        actions.create_link_type(
            "INVESTED_IN", "Person", "Organization", "many_to_many", "", "test", "test"
        )
        alice = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")
        bob = actions.upsert_object("Person", {"name": "Bob"}, "s", "a")
        acme = actions.upsert_object("Organization", {"name": "Acme"}, "s", "a")

        r1 = actions.upsert_link("INVESTED_IN", alice["id"], acme["id"], "s", "a")
        r2 = actions.upsert_link("INVESTED_IN", bob["id"], acme["id"], "s", "a")
        assert r1["created"] is True
        assert r2["created"] is True


class TestDeleteObject:
    def test_basic(self, actions):
        actions.create_object_type("Person", [
            {"name": "name", "data_type": "string", "required": True},
        ], "", "test", "test")
        r = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")

        result = actions.delete_object(r["id"], "admin", "User requested deletion")
        assert result["deleted"] is True

        assert actions.store.get_object(r["id"]) is None

        records = actions.store.get_provenance(r["id"], target_type="object")
        assert len(records) == 2
        delete_record = records[-1]
        assert delete_record["action_type"] == "delete_object"
        assert delete_record["source"] == "User requested deletion"
        assert delete_record["agent"] == "admin"

    def test_double_delete_fails(self, actions):
        actions.create_object_type("Person", [
            {"name": "name", "data_type": "string", "required": True},
        ], "", "test", "test")
        r = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")
        actions.delete_object(r["id"], "a", "reason")
        with pytest.raises(ActionError, match="already deleted"):
            actions.delete_object(r["id"], "a", "reason")

    def test_nonexistent_fails(self, actions):
        with pytest.raises(ActionError, match="does not exist"):
            actions.delete_object("nonexistent", "a", "reason")


class TestDeleteLink:
    def test_basic(self, actions):
        _setup_person_org(actions)
        actions.create_link_type(
            "KNOWS", "Person", "Person", "many_to_many", "", "test", "test"
        )
        alice = actions.upsert_object("Person", {"name": "Alice"}, "s", "a")
        bob = actions.upsert_object("Person", {"name": "Bob"}, "s", "a")
        link = actions.upsert_link("KNOWS", alice["id"], bob["id"], "s", "a")

        result = actions.delete_link(link["id"], "admin", "Outdated info")
        assert result["deleted"] is True

        records = actions.store.get_provenance(link["id"], target_type="link")
        assert len(records) == 2


class TestNoBypassWritePath:
    """Confirm that the action layer is the only write path — direct table
    writes are possible at the store level but that's a deliberate design
    choice to keep the store simple; the governance happens at the action
    layer which wraps the store.  The test verifies that only actions write
    provenance."""

    def test_direct_store_write_does_not_write_provenance(self, store):
        store.insert_object_type(ObjectType("X", [], ""))
        assert len(store.get_provenance("X", target_type="object_type")) == 0

    def test_action_write_does_write_provenance(self, actions):
        actions.create_object_type("X", [], "", "s", "a")
        assert len(actions.store.get_provenance("X", target_type="object_type")) == 1
