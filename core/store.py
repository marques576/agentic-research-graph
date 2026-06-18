"""
SQLite-backed ontology store.

Provides low-level CRUD operations on the schema, object, link, and provenance
tables.  Validation and provenance management happen at the Action Type layer
(core/actions.py), not here.

Schema version is stamped in the `meta` table.  Opening a database with a
mismatched version raises an error loudly rather than guessing.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from .schema import ObjectType, LinkType, PropertyDef

CURRENT_SCHEMA_VERSION = "1"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS object_types (
    name        TEXT PRIMARY KEY,
    properties  TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS link_types (
    name         TEXT PRIMARY KEY,
    source_type  TEXT NOT NULL,
    target_type  TEXT NOT NULL,
    cardinality  TEXT NOT NULL DEFAULT 'many_to_many',
    description  TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS objects (
    id          TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    properties  TEXT NOT NULL DEFAULT '{}',
    is_deleted  INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS links (
    id                TEXT PRIMARY KEY,
    link_type         TEXT NOT NULL,
    source_object_id  TEXT NOT NULL,
    target_object_id  TEXT NOT NULL,
    properties        TEXT NOT NULL DEFAULT '{}',
    is_deleted        INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(object_type);
CREATE INDEX IF NOT EXISTS idx_objects_deleted ON objects(is_deleted);
CREATE INDEX IF NOT EXISTS idx_links_type ON links(link_type);
CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_object_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_object_id);
CREATE INDEX IF NOT EXISTS idx_links_deleted ON links(is_deleted);

CREATE TABLE IF NOT EXISTS provenance (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type         TEXT NOT NULL CHECK(target_type IN ('object_type', 'link_type', 'object', 'link')),
    target_id           TEXT NOT NULL,
    source              TEXT NOT NULL,
    agent               TEXT NOT NULL,
    action_type         TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    confidence          REAL,
    previous_properties TEXT
);

CREATE INDEX IF NOT EXISTS idx_provenance_target ON provenance(target_type, target_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


class StoreError(Exception):
    """Base exception for store-level errors."""


class SchemaVersionMismatch(StoreError):
    """The database schema version does not match this code's version."""


class OntologyStore:
    """
    Low-level SQLite store for ontology data.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.  Created if it doesn't exist.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=OFF")

        self._create_tables()
        self._check_schema_version()

    def _create_tables(self) -> None:
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def _check_schema_version(self) -> None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()

        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (CURRENT_SCHEMA_VERSION,),
            )
            self._conn.commit()
            return

        stored = row["value"]
        if stored != CURRENT_SCHEMA_VERSION:
            raise SchemaVersionMismatch(
                f"Database schema version is '{stored}', but this code expects "
                f"'{CURRENT_SCHEMA_VERSION}'.  Migration tooling is not available yet."
            )

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Object Types
    # ------------------------------------------------------------------

    def insert_object_type(self, obj_type: ObjectType) -> None:
        self._conn.execute(
            "INSERT INTO object_types (name, properties, description, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                obj_type.name,
                json.dumps([p.to_dict() for p in obj_type.properties]),
                obj_type.description,
                _now_iso(),
            ),
        )
        self._conn.commit()

    def get_object_type(self, name: str) -> ObjectType | None:
        row = self._conn.execute(
            "SELECT * FROM object_types WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return ObjectType(
            name=row["name"],
            properties=[PropertyDef.from_dict(p) for p in json.loads(row["properties"])],
            description=row["description"],
        )

    def get_all_object_types(self) -> list[ObjectType]:
        rows = self._conn.execute(
            "SELECT * FROM object_types ORDER BY name"
        ).fetchall()
        return [
            ObjectType(
                name=row["name"],
                properties=[PropertyDef.from_dict(p) for p in json.loads(row["properties"])],
                description=row["description"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Link Types
    # ------------------------------------------------------------------

    def insert_link_type(self, link_type: LinkType) -> None:
        self._conn.execute(
            "INSERT INTO link_types (name, source_type, target_type, cardinality, description, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                link_type.name,
                link_type.source_type,
                link_type.target_type,
                link_type.cardinality,
                link_type.description,
                _now_iso(),
            ),
        )
        self._conn.commit()

    def get_link_type(self, name: str) -> LinkType | None:
        row = self._conn.execute(
            "SELECT * FROM link_types WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return LinkType(
            name=row["name"],
            source_type=row["source_type"],
            target_type=row["target_type"],
            cardinality=row["cardinality"],
            description=row["description"],
        )

    def get_all_link_types(self) -> list[LinkType]:
        rows = self._conn.execute(
            "SELECT * FROM link_types ORDER BY name"
        ).fetchall()
        return [
            LinkType(
                name=row["name"],
                source_type=row["source_type"],
                target_type=row["target_type"],
                cardinality=row["cardinality"],
                description=row["description"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Objects (instances)
    # ------------------------------------------------------------------

    def insert_object(
        self, object_type: str, properties: dict[str, Any], obj_id: str | None = None
    ) -> str:
        obj_id = obj_id or _new_id()
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO objects (id, object_type, properties, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (obj_id, object_type, json.dumps(properties), now, now),
        )
        self._conn.commit()
        return obj_id

    def update_object(self, obj_id: str, properties: dict[str, Any]) -> bool:
        now = _now_iso()
        cur = self._conn.execute(
            "UPDATE objects SET properties = ?, updated_at = ? "
            "WHERE id = ? AND is_deleted = 0",
            (json.dumps(properties), now, obj_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def soft_delete_object(self, obj_id: str) -> bool:
        now = _now_iso()
        cur = self._conn.execute(
            "UPDATE objects SET is_deleted = 1, updated_at = ? WHERE id = ? AND is_deleted = 0",
            (now, obj_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_object(self, obj_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        clause = "" if include_deleted else "AND is_deleted = 0"
        row = self._conn.execute(
            f"SELECT * FROM objects WHERE id = ? {clause}", (obj_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "object_type": row["object_type"],
            "properties": json.loads(row["properties"]),
            "is_deleted": bool(row["is_deleted"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def find_objects(
        self,
        object_type: str | None = None,
        property_filters: dict[str, Any] | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM objects WHERE 1=1"
        params: list[Any] = []

        if not include_deleted:
            query += " AND is_deleted = 0"

        if object_type is not None:
            query += " AND object_type = ?"
            params.append(object_type)

        rows = self._conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            props = json.loads(row["properties"])
            if property_filters:
                match = True
                for k, v in property_filters.items():
                    if props.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            results.append({
                "id": row["id"],
                "object_type": row["object_type"],
                "properties": props,
                "is_deleted": bool(row["is_deleted"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return results

    def count_objects(self, include_deleted: bool = False) -> int:
        clause = "" if include_deleted else "WHERE is_deleted = 0"
        row = self._conn.execute(f"SELECT COUNT(*) as cnt FROM objects {clause}").fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Links (instances)
    # ------------------------------------------------------------------

    def insert_link(
        self,
        link_type: str,
        source_object_id: str,
        target_object_id: str,
        properties: dict[str, Any] | None = None,
        link_id: str | None = None,
    ) -> str:
        link_id = link_id or _new_id()
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO links (id, link_type, source_object_id, target_object_id, "
            "properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                link_id,
                link_type,
                source_object_id,
                target_object_id,
                json.dumps(properties or {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        return link_id

    def update_link(self, link_id: str, properties: dict[str, Any]) -> bool:
        now = _now_iso()
        cur = self._conn.execute(
            "UPDATE links SET properties = ?, updated_at = ? "
            "WHERE id = ? AND is_deleted = 0",
            (json.dumps(properties), now, link_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def soft_delete_link(self, link_id: str) -> bool:
        now = _now_iso()
        cur = self._conn.execute(
            "UPDATE links SET is_deleted = 1, updated_at = ? WHERE id = ? AND is_deleted = 0",
            (now, link_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_link(self, link_id: str, include_deleted: bool = False) -> dict[str, Any] | None:
        clause = "" if include_deleted else "AND is_deleted = 0"
        row = self._conn.execute(
            f"SELECT * FROM links WHERE id = ? {clause}", (link_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "link_type": row["link_type"],
            "source_object_id": row["source_object_id"],
            "target_object_id": row["target_object_id"],
            "properties": json.loads(row["properties"]),
            "is_deleted": bool(row["is_deleted"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def find_links(
        self,
        link_type: str | None = None,
        source_object_id: str | None = None,
        target_object_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM links WHERE 1=1"
        params: list[Any] = []

        if not include_deleted:
            query += " AND is_deleted = 0"

        if link_type is not None:
            query += " AND link_type = ?"
            params.append(link_type)

        if source_object_id is not None:
            query += " AND source_object_id = ?"
            params.append(source_object_id)

        if target_object_id is not None:
            query += " AND target_object_id = ?"
            params.append(target_object_id)

        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "link_type": row["link_type"],
                "source_object_id": row["source_object_id"],
                "target_object_id": row["target_object_id"],
                "properties": json.loads(row["properties"]),
                "is_deleted": bool(row["is_deleted"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def count_links(self, include_deleted: bool = False) -> int:
        clause = "" if include_deleted else "WHERE is_deleted = 0"
        row = self._conn.execute(f"SELECT COUNT(*) as cnt FROM links {clause}").fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def insert_provenance(
        self,
        target_type: str,
        target_id: str,
        source: str,
        agent: str,
        action_type: str,
        confidence: float | None = None,
        previous_properties: dict[str, Any] | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO provenance (target_type, target_id, source, agent, "
            "action_type, timestamp, confidence, previous_properties) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                target_type,
                target_id,
                source,
                agent,
                action_type,
                _now_iso(),
                confidence,
                json.dumps(previous_properties) if previous_properties else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_provenance(
        self, target_id: str, target_type: str | None = None
    ) -> list[dict[str, Any]]:
        if target_type:
            rows = self._conn.execute(
                "SELECT * FROM provenance WHERE target_id = ? AND target_type = ? "
                "ORDER BY timestamp ASC",
                (target_id, target_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM provenance WHERE target_id = ? ORDER BY timestamp ASC",
                (target_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Schema summary
    # ------------------------------------------------------------------

    def schema_summary(self) -> dict[str, Any]:
        """Return a human-readable summary of the schema."""
        obj_types = self.get_all_object_types()
        link_types = self.get_all_link_types()
        return {
            "object_types": [
                {
                    "name": ot.name,
                    "property_count": len(ot.properties),
                    "description": ot.description,
                }
                for ot in obj_types
            ],
            "link_types": [
                {
                    "name": lt.name,
                    "source_type": lt.source_type,
                    "target_type": lt.target_type,
                    "cardinality": lt.cardinality,
                    "description": lt.description,
                }
                for lt in link_types
            ],
        }
