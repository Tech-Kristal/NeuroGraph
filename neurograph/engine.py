"""NetworkX-backed graph operations and JSON interchange."""

from __future__ import annotations

from collections import deque
import json
import logging
from pathlib import Path
from typing import Any, Literal, Mapping

import networkx as nx

from neurograph.errors import DuplicateError, NotFoundError, ValidationError
from neurograph.models import Edge, Node
from neurograph.storage import SQLiteRepository

logger = logging.getLogger(__name__)

_JSON_FORMAT = "neurograph"
_JSON_VERSION = 1


class GraphEngine:
    """Manage a directed knowledge graph backed by SQLite."""

    def __init__(self, database_path: str | Path = "neurograph.db") -> None:
        self._repository = SQLiteRepository(database_path)
        self._graph = nx.MultiDiGraph()
        nodes, edges = self._repository.load_all()
        for node in nodes:
            self._add_node_to_graph(node)
        for edge in edges:
            self._add_edge_to_graph(edge)

    def close(self) -> None:
        """Close the SQLite connection."""
        self._repository.close()

    def __enter__(self) -> GraphEngine:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def add_node(
        self, name: str, properties: dict[str, Any] | None = None, *, node_id: str | None = None
    ) -> Node:
        """Add a node and persist it."""
        node = (
            Node(name=name, properties={} if properties is None else properties)
            if node_id is None
            else Node(name=name, properties={} if properties is None else properties, id=node_id)
        )
        self._repository.insert_node(node)
        self._add_node_to_graph(node)
        logger.debug("Added node %s", node.id)
        return node

    def update_node(
        self,
        node_id: str,
        *,
        name: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> Node:
        """Replace a node's name and/or complete properties mapping."""
        current = self.get_node(node_id)
        updated = Node(
            id=current.id,
            name=current.name if name is None else name,
            properties=current.properties if properties is None else properties,
        )
        self._repository.update_node(updated)
        self._add_node_to_graph(updated)
        logger.debug("Updated node %s", node_id)
        return updated

    def delete_node(self, node_id: str) -> None:
        """Delete a node and all incident edges."""
        self._repository.delete_node(node_id)
        self._graph.remove_node(node_id)
        logger.debug("Deleted node %s", node_id)

    def get_node(self, node_id: str) -> Node:
        """Retrieve a node or raise :class:`NotFoundError`."""
        node = self._repository.get_node(node_id)
        if node is None:
            raise NotFoundError(f"Node not found: {node_id}")
        return node

    def all_nodes(self) -> list[Node]:
        """Return every node in stable ID order."""
        nodes, _ = self._repository.load_all()
        return nodes

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        properties: dict[str, Any] | None = None,
        *,
        edge_id: str | None = None,
    ) -> Edge:
        """Add a directed relationship between existing nodes."""
        # Check endpoints first to provide a precise error rather than a DB constraint error.
        self.get_node(source_id)
        self.get_node(target_id)
        edge = (
            Edge(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                properties={} if properties is None else properties,
            )
            if edge_id is None
            else Edge(
                id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                properties={} if properties is None else properties,
            )
        )
        self._repository.insert_edge(edge)
        self._add_edge_to_graph(edge)
        logger.debug("Added edge %s", edge.id)
        return edge

    def update_edge(
        self,
        edge_id: str,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        relation: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> Edge:
        """Replace supplied fields on an edge."""
        current = self.get_edge(edge_id)
        updated = Edge(
            id=current.id,
            source_id=current.source_id if source_id is None else source_id,
            target_id=current.target_id if target_id is None else target_id,
            relation=current.relation if relation is None else relation,
            properties=current.properties if properties is None else properties,
        )
        self.get_node(updated.source_id)
        self.get_node(updated.target_id)
        self._repository.update_edge(updated)
        self._remove_edge_from_graph(current)
        self._add_edge_to_graph(updated)
        logger.debug("Updated edge %s", edge_id)
        return updated

    def delete_edge(self, edge_id: str) -> None:
        """Delete an edge."""
        edge = self.get_edge(edge_id)
        self._repository.delete_edge(edge_id)
        self._remove_edge_from_graph(edge)
        logger.debug("Deleted edge %s", edge_id)

    def get_edge(self, edge_id: str) -> Edge:
        """Retrieve an edge or raise :class:`NotFoundError`."""
        edge = self._repository.get_edge(edge_id)
        if edge is None:
            raise NotFoundError(f"Edge not found: {edge_id}")
        return edge

    def all_edges(self) -> list[Edge]:
        """Return every edge in stable ID order."""
        _, edges = self._repository.load_all()
        return edges

    def traverse(
        self,
        start_node_id: str,
        *,
        max_depth: int | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
        include_start: bool = False,
    ) -> list[Node]:
        """Return breadth-first reachable nodes, excluding the start by default."""
        self.get_node(start_node_id)
        if max_depth is not None and (
            isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
        ):
            raise ValidationError("max_depth must be a non-negative integer or None.")
        if direction not in ("outgoing", "incoming", "both"):
            raise ValidationError("direction must be 'outgoing', 'incoming', or 'both'.")

        visited = {start_node_id}
        queue = deque([(start_node_id, 0)])
        ordered_ids = [start_node_id] if include_start else []
        while queue:
            current, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            neighbors: set[str] = set()
            if direction in ("outgoing", "both"):
                neighbors.update(self._graph.successors(current))
            if direction in ("incoming", "both"):
                neighbors.update(self._graph.predecessors(current))
            for neighbor in sorted(neighbors):
                if neighbor not in visited:
                    visited.add(neighbor)
                    ordered_ids.append(neighbor)
                    queue.append((neighbor, depth + 1))
        return [self.get_node(node_id) for node_id in ordered_ids]

    def search_nodes(
        self, *, name: str | None = None, properties: Mapping[str, Any] | None = None
    ) -> list[Node]:
        """Search by case-insensitive name substring and exact property values."""
        if name is None and properties is None:
            raise ValidationError("Provide a name query, properties, or both.")
        property_filter = dict(properties) if properties is not None else {}
        matches: list[Node] = []
        needle = name.casefold() if name is not None else None
        for node in self.all_nodes():
            if needle is not None and needle not in node.name.casefold():
                continue
            if any(node.properties.get(key) != value for key, value in property_filter.items()):
                continue
            matches.append(node)
        return matches

    def find_duplicate_nodes(self) -> list[list[Node]]:
        """Group nodes with names equal after trimming whitespace and case-folding."""
        groups: dict[str, list[Node]] = {}
        for node in self.all_nodes():
            key = node.name.strip().casefold()
            groups.setdefault(key, []).append(node)
        return [group for _, group in sorted(groups.items()) if len(group) > 1]

    def export_json(self, path: str | Path) -> Path:
        """Write the full graph to a versioned JSON document."""
        destination = Path(path)
        document = {
            "format": _JSON_FORMAT,
            "version": _JSON_VERSION,
            "nodes": [node.to_dict() for node in self.all_nodes()],
            "edges": [edge.to_dict() for edge in self.all_edges()],
        }
        destination.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return destination

    def import_json(self, path: str | Path, *, replace: bool = False) -> tuple[int, int]:
        """Import a versioned JSON graph; append by default or replace atomically."""
        source = Path(path)
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"Could not read graph JSON from {source}: {exc}") from exc
        nodes, edges = self._parse_document(document)
        self._validate_import(nodes, edges, replace=replace)

        if replace:
            self._repository.replace_all(nodes, edges)
            self._graph.clear()
        else:
            self._repository.insert_all(nodes, edges)
        for node in nodes:
            self._add_node_to_graph(node)
        for edge in edges:
            self._add_edge_to_graph(edge)
        logger.info("Imported %d nodes and %d edges from %s", len(nodes), len(edges), source)
        return len(nodes), len(edges)

    @staticmethod
    def _parse_document(document: Any) -> tuple[list[Node], list[Edge]]:
        if not isinstance(document, dict):
            raise ValidationError("Graph JSON root must be an object.")
        if document.get("format") != _JSON_FORMAT or document.get("version") != _JSON_VERSION:
            raise ValidationError("Unsupported graph JSON format or version.")
        raw_nodes = document.get("nodes")
        raw_edges = document.get("edges")
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValidationError("Graph JSON must contain node and edge arrays.")
        nodes = [Node.from_dict(item) for item in raw_nodes]
        edges = [Edge.from_dict(item) for item in raw_edges]
        return nodes, edges

    def _validate_import(self, nodes: list[Node], edges: list[Edge], *, replace: bool) -> None:
        node_ids = [node.id for node in nodes]
        edge_ids = [edge.id for edge in edges]
        if len(node_ids) != len(set(node_ids)):
            raise DuplicateError("Graph JSON contains duplicate node IDs.")
        if len(edge_ids) != len(set(edge_ids)):
            raise DuplicateError("Graph JSON contains duplicate edge IDs.")
        available_node_ids = set(node_ids)
        if not replace:
            available_node_ids.update(node.id for node in self.all_nodes())
        for edge in edges:
            if edge.source_id not in available_node_ids:
                raise ValidationError(f"Edge {edge.id} references missing source node {edge.source_id}.")
            if edge.target_id not in available_node_ids:
                raise ValidationError(f"Edge {edge.id} references missing target node {edge.target_id}.")
        if not replace:
            existing_node_ids = {node.id for node in self.all_nodes()}
            existing_edge_ids = {edge.id for edge in self.all_edges()}
            if existing_node_ids.intersection(node_ids):
                raise DuplicateError("Imported node ID conflicts with existing graph data.")
            if existing_edge_ids.intersection(edge_ids):
                raise DuplicateError("Imported edge ID conflicts with existing graph data.")

    def _add_node_to_graph(self, node: Node) -> None:
        self._graph.add_node(node.id, name=node.name, properties=node.to_dict()["properties"])

    def _add_edge_to_graph(self, edge: Edge) -> None:
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.id,
            relation=edge.relation,
            properties=edge.to_dict()["properties"],
        )

    def _remove_edge_from_graph(self, edge: Edge) -> None:
        if self._graph.has_edge(edge.source_id, edge.target_id, edge.id):
            self._graph.remove_edge(edge.source_id, edge.target_id, edge.id)
