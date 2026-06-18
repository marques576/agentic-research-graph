"""
Action Type registry: declared, validated write operations.

Every write to the ontology goes through an Action Type.  An Action Type:
1. Has a defined input schema (what arguments it takes)
2. Validates those inputs against the current Object/Link Type schema
3. Performs one or more Object/Link creates, updates, or deletes atomically
4. Always writes a provenance record

There is no other write path.  This the one rule that must not be relaxed.

Built-in Action Types for the foundation:
- create_object_type / create_link_type — schema authoring
- upsert_object — create or update an object instance; validates properties
- upsert_link — create or update a link instance; validates source/target types
- delete_object / delete_link — soft delete with provenance
"""

from __future__ import annotations

from typing import Any, Callable

from .schema import (
    ObjectType,
    LinkType,
    PropertyDef,
    validate_data_type,
    validate_cardinality,
    validate_object_properties,
)
from .store import OntologyStore


class ActionError(Exception):
    """Raised when an action fails validation or execution."""


class ActionRegistry:
    """
    Registry of all Action Types.

    Every write to the store goes through an action in this registry.
    Actions wrap store operations with validation and provenance recording.
    """

    def __init__(self, store: OntologyStore) -> None:
        self.store = store

    # ------------------------------------------------------------------
    # create_object_type
    # ------------------------------------------------------------------

    def create_object_type(
        self,
        name: str,
        properties: list[dict[str, Any]],
        description: str,
        source: str,
        agent: str,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ActionError("Object type name must not be empty")

        existing = self.store.get_object_type(name)
        if existing is not None:
            raise ActionError(f"Object type '{name}' already exists")

        prop_defs = []
        for i, p in enumerate(properties):
            if not isinstance(p, dict):
                raise ActionError(f"Property at index {i} must be a dict")
            pname = p.get("name", "")
            if not pname:
                raise ActionError(f"Property at index {i} must have a 'name'")
            ptype = p.get("data_type", "string")
            validated_type = validate_data_type(ptype)
            prop_defs.append(PropertyDef(
                name=pname,
                data_type=validated_type,
                required=bool(p.get("required", False)),
            ))

        obj_type = ObjectType(
            name=name,
            properties=prop_defs,
            description=description,
        )

        self.store.insert_object_type(obj_type)

        self.store.insert_provenance(
            target_type="object_type",
            target_id=name,
            source=source,
            agent=agent,
            action_type="create_object_type",
        )

        return {"name": name, "properties": [p.to_dict() for p in prop_defs]}

    # ------------------------------------------------------------------
    # create_link_type
    # ------------------------------------------------------------------

    def create_link_type(
        self,
        name: str,
        source_type: str,
        target_type: str,
        cardinality: str,
        description: str,
        source: str,
        agent: str,
    ) -> dict[str, Any]:
        if not name.strip():
            raise ActionError("Link type name must not be empty")

        existing = self.store.get_link_type(name)
        if existing is not None:
            raise ActionError(f"Link type '{name}' already exists")

        if self.store.get_object_type(source_type) is None:
            raise ActionError(
                f"Source object type '{source_type}' does not exist. "
                f"Create it first with create_object_type."
            )
        if self.store.get_object_type(target_type) is None:
            raise ActionError(
                f"Target object type '{target_type}' does not exist. "
                f"Create it first with create_object_type."
            )

        validated_cardinality = validate_cardinality(cardinality)

        link_type = LinkType(
            name=name,
            source_type=source_type,
            target_type=target_type,
            cardinality=validated_cardinality,
            description=description,
        )

        self.store.insert_link_type(link_type)

        self.store.insert_provenance(
            target_type="link_type",
            target_id=name,
            source=source,
            agent=agent,
            action_type="create_link_type",
        )

        return link_type.to_dict()

    # ------------------------------------------------------------------
    # upsert_object
    # ------------------------------------------------------------------

    def upsert_object(
        self,
        object_type: str,
        properties: dict[str, Any],
        source: str,
        agent: str,
        confidence: float | None = None,
        id: str | None = None,
    ) -> dict[str, Any]:
        obj_type_def = self.store.get_object_type(object_type)
        if obj_type_def is None:
            raise ActionError(
                f"Object type '{object_type}' does not exist. "
                f"Create it first with create_object_type."
            )

        validate_object_properties(obj_type_def, properties)

        if id is not None:
            existing = self.store.get_object(id, include_deleted=True)
            if existing is not None:
                if existing["is_deleted"]:
                    raise ActionError(
                        f"Cannot upsert object '{id}': it has been deleted. "
                        f"Create a new object instead."
                    )
                previous = existing["properties"]
                self.store.update_object(id, properties)
                self.store.insert_provenance(
                    target_type="object",
                    target_id=id,
                    source=source,
                    agent=agent,
                    action_type="upsert_object",
                    confidence=confidence,
                    previous_properties=previous,
                )
                return {"id": id, "object_type": object_type, "created": False}

        obj_id = self.store.insert_object(object_type, properties, obj_id=id)
        self.store.insert_provenance(
            target_type="object",
            target_id=obj_id,
            source=source,
            agent=agent,
            action_type="upsert_object",
            confidence=confidence,
        )

        return {"id": obj_id, "object_type": object_type, "created": True}

    # ------------------------------------------------------------------
    # upsert_link
    # ------------------------------------------------------------------

    def upsert_link(
        self,
        link_type: str,
        source_object_id: str,
        target_object_id: str,
        source: str,
        agent: str,
        properties: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        lt_def = self.store.get_link_type(link_type)
        if lt_def is None:
            raise ActionError(
                f"Link type '{link_type}' does not exist. "
                f"Create it first with create_link_type."
            )

        src_obj = self.store.get_object(source_object_id)
        if src_obj is None:
            raise ActionError(
                f"Source object '{source_object_id}' does not exist "
                f"(or is deleted)."
            )

        tgt_obj = self.store.get_object(target_object_id)
        if tgt_obj is None:
            raise ActionError(
                f"Target object '{target_object_id}' does not exist "
                f"(or is deleted)."
            )

        if src_obj["object_type"] != lt_def.source_type:
            raise ActionError(
                f"Source object '{source_object_id}' has type "
                f"'{src_obj['object_type']}', but link type '{link_type}' "
                f"requires source type '{lt_def.source_type}'."
            )

        if tgt_obj["object_type"] != lt_def.target_type:
            raise ActionError(
                f"Target object '{target_object_id}' has type "
                f"'{tgt_obj['object_type']}', but link type '{link_type}' "
                f"requires target type '{lt_def.target_type}'."
            )

        if lt_def.cardinality == "one_to_one":
            existing_source = self.store.find_links(
                link_type=link_type, source_object_id=source_object_id
            )
            if existing_source:
                raise ActionError(
                    f"Link type '{link_type}' has cardinality one_to_one, "
                    f"but source '{source_object_id}' already has a link of this type."
                )
            existing_target = self.store.find_links(
                link_type=link_type, target_object_id=target_object_id
            )
            if existing_target:
                raise ActionError(
                    f"Link type '{link_type}' has cardinality one_to_one, "
                    f"but target '{target_object_id}' already has a link of this type."
                )

        if lt_def.cardinality == "one_to_many":
            existing_target = self.store.find_links(
                link_type=link_type, target_object_id=target_object_id
            )
            if existing_target:
                raise ActionError(
                    f"Link type '{link_type}' has cardinality one_to_many, "
                    f"but target '{target_object_id}' already has an incoming "
                    f"link of this type."
                )

        link_id = self.store.insert_link(
            link_type, source_object_id, target_object_id, properties or {}
        )

        self.store.insert_provenance(
            target_type="link",
            target_id=link_id,
            source=source,
            agent=agent,
            action_type="upsert_link",
            confidence=confidence,
        )

        return {
            "id": link_id,
            "link_type": link_type,
            "created": True,
        }

    # ------------------------------------------------------------------
    # delete_object
    # ------------------------------------------------------------------

    def delete_object(
        self,
        id: str,
        agent: str,
        reason: str,
    ) -> dict[str, Any]:
        obj = self.store.get_object(id)
        if obj is None:
            raise ActionError(f"Object '{id}' does not exist or is already deleted.")

        success = self.store.soft_delete_object(id)
        if not success:
            raise ActionError(f"Failed to delete object '{id}'.")

        self.store.insert_provenance(
            target_type="object",
            target_id=id,
            source=reason,
            agent=agent,
            action_type="delete_object",
            previous_properties=obj["properties"],
        )

        return {"id": id, "deleted": True}

    # ------------------------------------------------------------------
    # delete_link
    # ------------------------------------------------------------------

    def delete_link(
        self,
        id: str,
        agent: str,
        reason: str,
    ) -> dict[str, Any]:
        link = self.store.get_link(id)
        if link is None:
            raise ActionError(f"Link '{id}' does not exist or is already deleted.")

        success = self.store.soft_delete_link(id)
        if not success:
            raise ActionError(f"Failed to delete link '{id}'.")

        self.store.insert_provenance(
            target_type="link",
            target_id=id,
            source=reason,
            agent=agent,
            action_type="delete_link",
            previous_properties=link["properties"],
        )

        return {"id": id, "deleted": True}
