"""
Provenance record shape and query helpers.

Every write to the ontology store produces a provenance record that captures:
- source    — where the data came from (file path, URL, "manual entry", conversation id)
- agent     — which agent/tool/user made the write
- action_type — which declared Action Type produced this write
- timestamp — ISO 8601 UTC timestamp
- confidence — optional float 0–1 (nullable)

Provenance is non-negotiable: every object, link, and property value must be
traceable to its origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ProvenanceRecord:
    target_type: str  # "object" | "link"
    target_id: str
    source: str
    agent: str
    action_type: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: float | None = None
    previous_properties: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type,
            "target_id": self.target_id,
            "source": self.source,
            "agent": self.agent,
            "action_type": self.action_type,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "previous_properties": self.previous_properties,
        }
