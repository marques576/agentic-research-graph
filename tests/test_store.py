"""Tests for core/store.py — SQLite-backed ontology store."""

import os
import tempfile

import pytest

from core.schema import ObjectType, LinkType, PropertyDef
from core.store import OntologyStore, SchemaVersionMismatch


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


class TestSchemaVersion:
    def test_new_db_gets_current_version(self, store):
        row = store._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "1"

    def test_mismatched_version_fails(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s = OntologyStore(path)
            s._conn.execute(
                "UPDATE meta SET value = '0' WHERE key = 'schema_version'"
            )
            s._conn.commit()
            s.close()

            with pytest.raises(SchemaVersionMismatch, match="schema version"):
                OntologyStore(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_reopen_preserves_data(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            s1 = OntologyStore(path)
            s1.insert_object_type(ObjectType("Test", [], ""))
            s1.close()

            s2 = OntologyStore(path)
            ot = s2.get_object_type("Test")
            assert ot is not None
            assert ot.name == "Test"
            s2.close()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


class TestObjectTypes:
    def test_insert_and_get(self, store):
        ot = ObjectType("Person", [
            PropertyDef("name", "string", True),
            PropertyDef("age", "number", False),
        ], "A person")
        store.insert_object_type(ot)

        got = store.get_object_type("Person")
        assert got is not None
        assert got.name == "Person"
        assert len(got.properties) == 2
        assert got.properties[0].name == "name"
        assert got.properties[0].data_type == "string"
        assert got.properties[0].required is True
        assert got.description == "A person"

    def test_get_nonexistent(self, store):
        assert store.get_object_type("Nonexistent") is None

    def test_get_all(self, store):
        store.insert_object_type(ObjectType("A", [], ""))
        store.insert_object_type(ObjectType("B", [], ""))
        all_types = store.get_all_object_types()
        assert len(all_types) == 2
        names = [t.name for t in all_types]
        assert names == ["A", "B"]

    def test_duplicate_name_fails(self, store):
        store.insert_object_type(ObjectType("Person", [], ""))
        with pytest.raises(Exception):
            store.insert_object_type(ObjectType("Person", [], ""))


class TestLinkTypes:
    def test_insert_and_get(self, store):
        lt = LinkType(
            "EMPLOYED_BY",
            "Person",
            "Organization",
            "many_to_one",
            "Employment link",
        )
        store.insert_link_type(lt)

        got = store.get_link_type("EMPLOYED_BY")
        assert got is not None
        assert got.name == "EMPLOYED_BY"
        assert got.source_type == "Person"
        assert got.target_type == "Organization"
        assert got.cardinality == "many_to_one"
        assert got.description == "Employment link"

    def test_get_nonexistent(self, store):
        assert store.get_link_type("Nonexistent") is None


class TestObjects:
    def test_insert_and_get(self, store):
        obj_id = store.insert_object("TestType", {"key": "value"})
        obj = store.get_object(obj_id)
        assert obj is not None
        assert obj["object_type"] == "TestType"
        assert obj["properties"] == {"key": "value"}
        assert obj["is_deleted"] is False
        assert "created_at" in obj
        assert "updated_at" in obj

    def test_update(self, store):
        obj_id = store.insert_object("TestType", {"key": "v1"})
        assert store.update_object(obj_id, {"key": "v2"}) is True
        obj = store.get_object(obj_id)
        assert obj["properties"] == {"key": "v2"}

    def test_update_nonexistent(self, store):
        assert store.update_object("nonexistent", {}) is False

    def test_update_deleted(self, store):
        obj_id = store.insert_object("T", {"x": 1})
        store.soft_delete_object(obj_id)
        assert store.update_object(obj_id, {"x": 2}) is False

    def test_soft_delete(self, store):
        obj_id = store.insert_object("T", {})
        assert store.soft_delete_object(obj_id) is True
        assert store.get_object(obj_id) is None
        assert store.get_object(obj_id, include_deleted=True) is not None
        assert store.get_object(obj_id, include_deleted=True)["is_deleted"] is True

    def test_find_objects_by_type(self, store):
        store.insert_object("Person", {"name": "Alice"})
        store.insert_object("Person", {"name": "Bob"})
        store.insert_object("Org", {"name": "Acme"})

        persons = store.find_objects(object_type="Person")
        assert len(persons) == 2
        assert all(p["object_type"] == "Person" for p in persons)

    def test_find_objects_by_property(self, store):
        store.insert_object("Person", {"name": "Alice", "age": 30})
        store.insert_object("Person", {"name": "Bob", "age": 25})

        results = store.find_objects(object_type="Person", property_filters={"name": "Alice"})
        assert len(results) == 1
        assert results[0]["properties"]["name"] == "Alice"

    def test_find_objects_hides_deleted(self, store):
        obj_id = store.insert_object("T", {})
        store.soft_delete_object(obj_id)
        results = store.find_objects()
        assert len(results) == 0

    def test_count(self, store):
        assert store.count_objects() == 0
        store.insert_object("T", {})
        store.insert_object("T", {})
        assert store.count_objects() == 2


class TestLinks:
    def test_insert_and_get(self, store):
        link_id = store.insert_link("KNOWS", "obj-1", "obj-2", {"conf": 0.9})
        link = store.get_link(link_id)
        assert link is not None
        assert link["link_type"] == "KNOWS"
        assert link["source_object_id"] == "obj-1"
        assert link["target_object_id"] == "obj-2"
        assert link["properties"] == {"conf": 0.9}

    def test_soft_delete(self, store):
        link_id = store.insert_link("KNOWS", "a", "b")
        assert store.soft_delete_link(link_id) is True
        assert store.get_link(link_id) is None

    def test_find_by_source(self, store):
        store.insert_link("KNOWS", "alice", "bob")
        store.insert_link("KNOWS", "alice", "charlie")
        store.insert_link("KNOWS", "bob", "charlie")

        from_alice = store.find_links(source_object_id="alice")
        assert len(from_alice) == 2

    def test_find_by_type(self, store):
        store.insert_link("KNOWS", "a", "b")
        store.insert_link("EMPLOYED_BY", "a", "c")

        knows = store.find_links(link_type="KNOWS")
        assert len(knows) == 1


class TestProvenance:
    def test_insert_and_get(self, store):
        store.insert_provenance(
            target_type="object",
            target_id="obj-1",
            source="test_file.txt",
            agent="test_agent",
            action_type="upsert_object",
            confidence=0.95,
            previous_properties={"old": "value"},
        )

        records = store.get_provenance("obj-1")
        assert len(records) == 1
        r = records[0]
        assert r["target_type"] == "object"
        assert r["target_id"] == "obj-1"
        assert r["source"] == "test_file.txt"
        assert r["agent"] == "test_agent"
        assert r["action_type"] == "upsert_object"
        assert r["confidence"] == 0.95

    def test_filter_by_target_type(self, store):
        store.insert_provenance("object", "id-1", "s", "a", "create_object_type")
        store.insert_provenance("link", "id-1", "s", "a", "create_link_type")

        obj_records = store.get_provenance("id-1", target_type="object")
        assert len(obj_records) == 1
        assert obj_records[0]["action_type"] == "create_object_type"

        link_records = store.get_provenance("id-1", target_type="link")
        assert len(link_records) == 1
        assert link_records[0]["action_type"] == "create_link_type"


class TestSchemaSummary:
    def test_empty(self, store):
        s = store.schema_summary()
        assert s["object_types"] == []
        assert s["link_types"] == []

    def test_populated(self, store):
        store.insert_object_type(ObjectType("Person", [], ""))
        store.insert_link_type(LinkType("KNOWS", "Person", "Person"))
        s = store.schema_summary()
        assert len(s["object_types"]) == 1
        assert len(s["link_types"]) == 1
