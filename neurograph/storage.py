"""SQLite persistence for graph nodes and edges."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

from neurograph.errors import DuplicateError, NotFoundError, ValidationError
from neurograph.models import Edge, Node


class SQLiteRepository:
    """Stores graph entities in SQLite using explicit transactions."""

    def __init__(self, database_path: str | Path) -> None:
        path = Path(database_path).expanduser() if str(database_path) != ":memory:" else None
        self.database_path = str(path) if path is not None else ":memory:"
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                properties TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                properties TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            """
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()

    def load_all(self) -> tuple[list[Node], list[Edge]]:
        """Load all entities in stable ID order."""
        nodes = [
            Node(id=row["id"], name=row["name"], properties=json.loads(row["properties"]))
            for row in self._connection.execute("SELECT id, name, properties FROM nodes ORDER BY id")
        ]
        edges = [
            Edge(
                id=row["id"],
                source_id=row["source_id"],
                target_id=row["target_id"],
                relation=row["relation"],
                properties=json.loads(row["properties"]),
            )
            for row in self._connection.execute(
                "SELECT id, source_id, target_id, relation, properties FROM edges ORDER BY id"
            )
        ]
        return nodes, edges

    def insert_node(self, node: Node) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO nodes (id, name, properties) VALUES (?, ?, ?)",
                    (node.id, node.name, self._encode(node.properties)),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateError(f"Node ID already exists: {node.id}") from exc

    def update_node(self, node: Node) -> None:
        with self._connection:
            cursor = self._connection.execute(
                "UPDATE nodes SET name = ?, properties = ? WHERE id = ?",
                (node.name, self._encode(node.properties), node.id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"Node not found: {node.id}")

    def delete_node(self, node_id: str) -> None:
        with self._connection:
            cursor = self._connection.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            if cursor.rowcount == 0:
                raise NotFoundError(f"Node not found: {node_id}")

    def insert_edge(self, edge: Edge) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO edges (id, source_id, target_id, relation, properties)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        edge.id,
                        edge.source_id,
                        edge.target_id,
                        edge.relation,
                        self._encode(edge.properties),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if self.get_node(edge.source_id) is None:
                raise NotFoundError(f"Source node not found: {edge.source_id}") from exc
            if self.get_node(edge.target_id) is None:
                raise NotFoundError(f"Target node not found: {edge.target_id}") from exc
            raise DuplicateError(f"Edge ID already exists: {edge.id}") from exc

    def update_edge(self, edge: Edge) -> None:
        try:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE edges
                    SET source_id = ?, target_id = ?, relation = ?, properties = ?
                    WHERE id = ?
                    """,
                    (
                        edge.source_id,
                        edge.target_id,
                        edge.relation,
                        self._encode(edge.properties),
                        edge.id,
                    ),
                )
                if cursor.rowcount == 0:
                    raise NotFoundError(f"Edge not found: {edge.id}")
        except sqlite3.IntegrityError as exc:
            if self.get_node(edge.source_id) is None:
                raise NotFoundError(f"Source node not found: {edge.source_id}") from exc
            if self.get_node(edge.target_id) is None:
                raise NotFoundError(f"Target node not found: {edge.target_id}") from exc
            raise DuplicateError(f"Edge ID already exists: {edge.id}") from exc

    def delete_edge(self, edge_id: str) -> None:
        with self._connection:
            cursor = self._connection.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            if cursor.rowcount == 0:
                raise NotFoundError(f"Edge not found: {edge_id}")

    def get_node(self, node_id: str) -> Node | None:
        row = self._connection.execute(
            "SELECT id, name, properties FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return Node(id=row["id"], name=row["name"], properties=json.loads(row["properties"]))

    def get_edge(self, edge_id: str) -> Edge | None:
        row = self._connection.execute(
            "SELECT id, source_id, target_id, relation, properties FROM edges WHERE id = ?",
            (edge_id,),
        ).fetchone()
        if row is None:
            return None
        return Edge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation=row["relation"],
            properties=json.loads(row["properties"]),
        )

    def replace_all(self, nodes: Iterable[Node], edges: Iterable[Edge]) -> None:
        """Atomically replace stored graph contents."""
        try:
            with self._connection:
                self._connection.execute("DELETE FROM edges")
                self._connection.execute("DELETE FROM nodes")
                self._connection.executemany(
                    "INSERT INTO nodes (id, name, properties) VALUES (?, ?, ?)",
                    [(n.id, n.name, self._encode(n.properties)) for n in nodes],
                )
                self._connection.executemany(
                    """
                    INSERT INTO edges (id, source_id, target_id, relation, properties)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (e.id, e.source_id, e.target_id, e.relation, self._encode(e.properties))
                        for e in edges
                    ],
                )
        except sqlite3.IntegrityError as exc:
            raise ValidationError(f"Imported graph violates storage constraints: {exc}") from exc

    def insert_all(self, nodes: Iterable[Node], edges: Iterable[Edge]) -> None:
        """Atomically append entities, rejecting duplicate IDs or missing endpoints."""
        try:
            with self._connection:
                self._connection.executemany(
                    "INSERT INTO nodes (id, name, properties) VALUES (?, ?, ?)",
                    [(n.id, n.name, self._encode(n.properties)) for n in nodes],
                )
                self._connection.executemany(
                    """
                    INSERT INTO edges (id, source_id, target_id, relation, properties)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (e.id, e.source_id, e.target_id, e.relation, self._encode(e.properties))
                        for e in edges
                    ],
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateError(f"Imported graph conflicts with stored data: {exc}") from exc

    @staticmethod
    def _encode(value: dict[str, Any]) -> str:
        try:
            return json.dumps(value, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Value is not JSON-compatible: {exc}") from exc
