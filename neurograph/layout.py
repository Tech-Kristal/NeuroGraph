"""Graph layout and geometry helpers that do not depend on Qt."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence
import zlib

import networkx as nx

from neurograph.models import Edge, Node

Positions = dict[str, tuple[float, float]]


def node_radius(degree: int) -> float:
    """Return a node radius that grows gently with connectivity."""
    return min(26.0, 10.0 + 2.6 * math.sqrt(max(0, degree)))


def curve_offsets(edges: Iterable[Edge], spacing: float = 22.0) -> dict[str, float]:
    """Spread parallel and opposing edges so they do not draw on top of each other.

    Edges between the same two nodes (in either direction) are fanned out around the
    straight line. A lone edge gets an offset of zero and stays straight. The sign of
    an offset is relative to the direction from the lower to the higher node ID.
    """
    groups: dict[tuple[str, str], list[Edge]] = {}
    for edge in edges:
        if edge.source_id == edge.target_id:
            continue
        key = tuple(sorted((edge.source_id, edge.target_id)))
        groups.setdefault(key, []).append(edge)
    offsets: dict[str, float] = {}
    for group in groups.values():
        group.sort(key=lambda item: (item.source_id, item.relation, item.id))
        count = len(group)
        for index, edge in enumerate(group):
            offsets[edge.id] = (index - (count - 1) / 2) * spacing
    return offsets


def _stable_angle(identifier: str) -> tuple[float, float]:
    """Deterministic pseudo-random angle and jitter derived from an ID."""
    digest = zlib.crc32(identifier.encode("utf-8"))
    return math.radians(digest % 360), (digest >> 9) % 50


def _spread(count: int) -> float:
    return 240.0 + 36.0 * math.sqrt(count)


def _circle_layout(node_ids: Sequence[str]) -> Positions:
    """Dependency-free fallback used when NumPy/SciPy are unavailable."""
    count = len(node_ids)
    radius = _spread(count) * 0.8
    return {
        node_id: (
            radius * math.cos(2 * math.pi * index / count),
            radius * math.sin(2 * math.pi * index / count),
        )
        for index, node_id in enumerate(node_ids)
    }


def _full_layout(nodes: Sequence[Node], edges: Sequence[Edge]) -> Positions:
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0].id: (0.0, 0.0)}
    graph = nx.Graph()
    graph.add_nodes_from(node.id for node in nodes)
    graph.add_edges_from(
        (edge.source_id, edge.target_id) for edge in edges if edge.source_id != edge.target_id
    )
    try:
        raw = nx.spring_layout(graph, seed=37, iterations=50)
    except ImportError:
        return _circle_layout([node.id for node in nodes])
    spread = _spread(len(nodes))
    return {key: (float(value[0]) * spread, float(value[1]) * spread * 0.78) for key, value in raw.items()}


def _incremental_layout(
    nodes: Sequence[Node], edges: Sequence[Edge], previous: Mapping[str, tuple[float, float]]
) -> Positions:
    """Keep known positions and place new nodes near their neighbours.

    This keeps the picture stable when a concept is added or a filter is toggled.
    """
    positions: Positions = {node.id: previous[node.id] for node in nodes if node.id in previous}
    neighbours: dict[str, set[str]] = {}
    for edge in edges:
        neighbours.setdefault(edge.source_id, set()).add(edge.target_id)
        neighbours.setdefault(edge.target_id, set()).add(edge.source_id)
    centre_x = sum(x for x, _ in positions.values()) / len(positions)
    centre_y = sum(y for _, y in positions.values()) / len(positions)
    reach = max((math.hypot(x - centre_x, y - centre_y) for x, y in positions.values()), default=0.0)
    for node in nodes:
        if node.id in positions:
            continue
        angle, jitter = _stable_angle(node.id)
        placed = [positions[other] for other in sorted(neighbours.get(node.id, ())) if other in positions]
        if placed:
            base_x = sum(x for x, _ in placed) / len(placed)
            base_y = sum(y for _, y in placed) / len(placed)
            distance = 110.0 + jitter
        else:
            base_x, base_y = centre_x, centre_y
            distance = reach + 140.0 + jitter
        positions[node.id] = (base_x + distance * math.cos(angle), base_y + distance * math.sin(angle))
    return positions


def calculate_layout(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    previous: Mapping[str, tuple[float, float]] | None = None,
) -> Positions:
    """Return ``{node_id: (x, y)}`` for the given nodes.

    When at least half of the nodes already have a position in ``previous`` those
    positions are kept; otherwise a fresh force-directed layout is computed.
    """
    if not nodes:
        return {}
    known = {node.id for node in nodes if previous and node.id in previous}
    if previous and known and len(known) >= len(nodes) / 2:
        return _incremental_layout(nodes, edges, previous)
    return _full_layout(nodes, edges)
