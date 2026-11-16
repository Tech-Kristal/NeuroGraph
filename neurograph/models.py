"""Typed graph entities and JSON-compatible data validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any
from uuid import uuid4

from neurograph.errors import ValidationError


def _validate_properties(properties: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(properties, dict):
        raise ValidationError("Properties must be a JSON object.")

    def validate_object_keys(value: Any) -> None:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValidationError("Property object keys must be strings.")
            for nested_value in value.values():
                validate_object_keys(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                validate_object_keys(nested_value)

    validate_object_keys(properties)
    try:
        # Round-tripping rejects custom objects and non-finite numbers.
        return json.loads(json.dumps(properties, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Properties must contain JSON-compatible values: {exc}") from exc


@dataclass(frozen=True)
class Node:
    """A named knowledge item with arbitrary JSON-compatible properties."""

    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValidationError("Node ID must be a non-empty string.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValidationError("Node name must be a non-empty string.")
        object.__setattr__(self, "properties", _validate_properties(self.properties))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return {"id": self.id, "name": self.name, "properties": deepcopy(self.properties)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Build a node from a serialized mapping."""
        if not isinstance(data, dict):
            raise ValidationError("Each node must be a JSON object.")
        try:
            return cls(id=data["id"], name=data["name"], properties=data.get("properties", {}))
        except KeyError as exc:
            raise ValidationError(f"Node is missing required field: {exc.args[0]}") from exc


@dataclass(frozen=True)
class Edge:
    """A directed, labeled relationship between two node IDs."""

    source_id: str
    target_id: str
    relation: str
    properties: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        for field_name in ("id", "source_id", "target_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"Edge {field_name} must be a non-empty string.")
        if not isinstance(self.relation, str) or not self.relation.strip():
            raise ValidationError("Edge relation must be a non-empty string.")
        object.__setattr__(self, "properties", _validate_properties(self.properties))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "properties": deepcopy(self.properties),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        """Build an edge from a serialized mapping."""
        if not isinstance(data, dict):
            raise ValidationError("Each edge must be a JSON object.")
        try:
            return cls(
                id=data["id"],
                source_id=data["source_id"],
                target_id=data["target_id"],
                relation=data["relation"],
                properties=data.get("properties", {}),
            )
        except KeyError as exc:
            raise ValidationError(f"Edge is missing required field: {exc.args[0]}") from exc
