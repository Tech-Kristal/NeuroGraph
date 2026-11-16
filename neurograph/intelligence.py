"""Graph-based discovery, similarity, clustering, and rule-based inference."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from typing import Iterable, Mapping

import networkx as nx

from neurograph.engine import GraphEngine
from neurograph.errors import IntelligenceError, ValidationError
from neurograph.models import Edge, Node
from neurograph.ollama import OllamaClient


@dataclass(frozen=True)
class RelationshipPath:
    """A confirmed chain of graph edges with a human-readable explanation."""

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    explanation: str

    @property
    def length(self) -> int:
        """Number of relationships in the path."""
        return len(self.edges)


@dataclass(frozen=True)
class SemanticMatch:
    """A node ranked by cosine similarity to a semantic query embedding."""

    node: Node
    score: float


@dataclass(frozen=True)
class NodeSimilarity:
    """Explainable structural and optional embedding similarity components."""

    first: Node
    second: Node
    structural_score: float
    semantic_score: float | None
    score: float


@dataclass(frozen=True)
class ConceptCluster:
    """An interpretable cluster of graph concepts and its internal edge count."""

    nodes: tuple[Node, ...]
    internal_edge_count: int
    density: float

    @property
    def node_ids(self) -> tuple[str, ...]:
        """IDs of cluster members."""
        return tuple(node.id for node in self.nodes)


@dataclass(frozen=True)
class InferredRelationship:
    """A potential transitive relationship, explicitly not a confirmed graph fact."""

    source: Node
    target: Node
    relation: str
    evidence: tuple[Edge, ...]
    confirmed: bool = False

    @property
    def explanation(self) -> str:
        """Explain the evidence chain and its unconfirmed status."""
        chain = " then ".join(
            f"{edge.relation} ({edge.source_id} -> {edge.target_id})" for edge in self.evidence
        )
        return (
            f"Potential {self.relation} relationship from {self.source.name} to "
            f"{self.target.name}, inferred from {chain}. This is not a confirmed fact."
        )


class IntelligenceEngine:
    """Provide optional intelligence features over an existing Level 1 graph."""

    def __init__(self, graph: GraphEngine, *, ollama: OllamaClient | None = None) -> None:
        self._graph = graph
        self._ollama = ollama

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        *,
        limit: int = 3,
        directed: bool = True,
    ) -> list[RelationshipPath]:
        """Return up to ``limit`` shortest simple paths and explain each edge chain."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValidationError("limit must be a positive integer.")
        source = self._graph.get_node(source_id)
        target = self._graph.get_node(target_id)
        nodes = {node.id: node for node in self._graph.all_nodes()}
        edges = self._graph.all_edges()
        edge_index: dict[tuple[str, str], list[Edge]] = {}
        graph = nx.DiGraph() if directed else nx.Graph()
        graph.add_nodes_from(nodes)
        for edge in edges:
            edge_index.setdefault((edge.source_id, edge.target_id), []).append(edge)
            graph.add_edge(edge.source_id, edge.target_id)
            if not directed:
                graph.add_edge(edge.target_id, edge.source_id)
        if source_id == target_id:
            return [RelationshipPath((source,), (), f"{source.name} is the starting and destination concept.")]
        try:
            node_paths = nx.shortest_simple_paths(graph, source_id, target_id)
            paths = []
            for node_path in node_paths:
                path_nodes = tuple(nodes[node_id] for node_id in node_path)
                edge_options = [
                    self._edge_options(edge_index, from_id, to_id, directed=directed)
                    for from_id, to_id in zip(node_path, node_path[1:])
                ]
                for path_edges in product(*edge_options):
                    paths.append(
                        RelationshipPath(
                            path_nodes,
                            path_edges,
                            self._explain_chain(path_nodes, path_edges),
                        )
                    )
                    if len(paths) == limit:
                        break
                if len(paths) == limit:
                    break
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def semantic_search(self, query: str, *, limit: int = 10) -> list[SemanticMatch]:
        """Rank graph concepts by local Ollama embedding cosine similarity."""
        ollama = self._require_ollama()
        if not query.strip():
            raise ValidationError("Semantic search query must not be empty.")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValidationError("limit must be a positive integer.")
        query_vector = ollama.embed(query)
        matches = []
        for node in self._graph.all_nodes():
            vector = ollama.embed(self._node_text(node))
            matches.append(SemanticMatch(node, self._cosine(query_vector, vector)))
        return sorted(matches, key=lambda match: (-match.score, match.node.id))[:limit]

    def similarity(self, first_id: str, second_id: str) -> NodeSimilarity:
        """Compare nodes by neighbor Jaccard similarity and optional local embeddings."""
        first = self._graph.get_node(first_id)
        second = self._graph.get_node(second_id)
        adjacency = self._adjacency()
        first_neighbors = adjacency[first_id]
        second_neighbors = adjacency[second_id]
        union = first_neighbors | second_neighbors
        structural = len(first_neighbors & second_neighbors) / len(union) if union else 0.0
        if first_id == second_id:
            structural = 1.0
        ollama = self._ollama
        if ollama is None:
            return NodeSimilarity(first, second, structural, None, structural)
        semantic = self._cosine(ollama.embed(self._node_text(first)), ollama.embed(self._node_text(second)))
        return NodeSimilarity(first, second, structural, semantic, (structural + semantic) / 2)

    def cluster_concepts(self) -> list[ConceptCluster]:
        """Group connected concepts using NetworkX greedy modularity communities."""
        nodes = self._graph.all_nodes()
        if not nodes:
            return []
        graph = nx.Graph()
        graph.add_nodes_from(node.id for node in nodes)
        edges = self._graph.all_edges()
        graph.add_edges_from((edge.source_id, edge.target_id) for edge in edges)
        if graph.number_of_edges() == 0:
            communities = [{node_id} for node_id in graph.nodes]
        else:
            communities = list(nx.community.greedy_modularity_communities(graph))
            assigned = set().union(*communities) if communities else set()
            communities.extend({node_id} for node_id in graph.nodes if node_id not in assigned)

        communities.sort(key=lambda group: tuple(sorted(group)))
        community_for = {
            node_id: index
            for index, community in enumerate(communities)
            for node_id in community
        }
        internal_edge_counts = [0] * len(communities)
        for edge in edges:
            if community_for[edge.source_id] == community_for[edge.target_id]:
                internal_edge_counts[community_for[edge.source_id]] += 1
        nodes_by_id = {node.id: node for node in nodes}
        result = []
        for index, community in enumerate(communities):
            members = tuple(nodes_by_id[node_id] for node_id in sorted(community))
            induced = graph.subgraph(community)
            result.append(
                ConceptCluster(members, internal_edge_counts[index], nx.density(induced))
            )
        return result

    def infer_transitive(
        self,
        relations: Iterable[str] | Mapping[str, str] = ("is_a", "part_of"),
    ) -> list[InferredRelationship]:
        """Infer unconfirmed two-hop transitive links without storing or overwriting facts."""
        if isinstance(relations, str):
            raise ValidationError("relations must be a collection of relation names or a mapping.")
        rules = dict(relations) if isinstance(relations, Mapping) else {r: r for r in relations}
        if any(not isinstance(key, str) or not key.strip() for key in rules):
            raise ValidationError("Transitive relation names must be non-empty strings.")
        if any(not isinstance(value, str) or not value.strip() for value in rules.values()):
            raise ValidationError("Inferred relation names must be non-empty strings.")
        edges = self._graph.all_edges()
        nodes = {node.id: node for node in self._graph.all_nodes()}
        by_source_relation: dict[tuple[str, str], list[Edge]] = {}
        confirmed = {(edge.source_id, edge.target_id, edge.relation) for edge in edges}
        for edge in edges:
            if edge.relation in rules:
                by_source_relation.setdefault((edge.source_id, edge.relation), []).append(edge)
        inferred: dict[tuple[str, str, str], InferredRelationship] = {}
        for first in edges:
            if first.relation not in rules:
                continue
            result_relation = rules[first.relation]
            for second in by_source_relation.get((first.target_id, first.relation), []):
                key = (first.source_id, second.target_id, result_relation)
                if first.source_id == second.target_id or key in confirmed:
                    continue
                inference = InferredRelationship(
                    nodes[first.source_id], nodes[second.target_id], result_relation, (first, second)
                )
                old = inferred.get(key)
                if old is None or tuple(edge.id for edge in inference.evidence) < tuple(
                    edge.id for edge in old.evidence
                ):
                    inferred[key] = inference
        return [inferred[key] for key in sorted(inferred)]

    def explain_with_ollama(self, path: RelationshipPath) -> str:
        """Ask the configured local Ollama chat model to explain a confirmed path."""
        prompt = (
            "Explain this confirmed knowledge-graph path using only the supplied facts. "
            "Do not add unsupported claims.\n\n" + path.explanation
        )
        return self._require_ollama().generate(prompt)

    def summarize_cluster(self, cluster: ConceptCluster) -> str:
        """Ask the configured local Ollama chat model to summarize cluster concepts."""
        concepts = [
            {"name": node.name, "properties": node.properties}
            for node in cluster.nodes
        ]
        prompt = (
            "Summarize the common theme, if any, in these graph-cluster concepts. "
            "Do not claim relationships not present in the list.\n\n"
            + json.dumps(concepts, ensure_ascii=False)
        )
        return self._require_ollama().generate(prompt)

    def _require_ollama(self) -> OllamaClient:
        if self._ollama is None:
            raise IntelligenceError(
                "This feature requires a configured local OllamaClient; graph features work without Ollama."
            )
        return self._ollama

    def _adjacency(self) -> dict[str, set[str]]:
        adjacency = {node.id: set() for node in self._graph.all_nodes()}
        for edge in self._graph.all_edges():
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)
        return adjacency

    @staticmethod
    def _node_text(node: Node) -> str:
        return f"{node.name}\n{json.dumps(node.properties, ensure_ascii=False, sort_keys=True)}"

    @staticmethod
    def _cosine(first: list[float], second: list[float]) -> float:
        if not first or not second or len(first) != len(second):
            raise IntelligenceError("Embedding vectors have incompatible dimensions.")
        if any(not math.isfinite(value) for value in first + second):
            raise IntelligenceError("Embedding vectors must contain finite values.")
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if first_norm == 0 or second_norm == 0:
            return 0.0
        return max(-1.0, min(1.0, sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)))

    @staticmethod
    def _edge_options(
        edge_index: Mapping[tuple[str, str], list[Edge]],
        source_id: str,
        target_id: str,
        *,
        directed: bool,
    ) -> tuple[Edge, ...]:
        candidates = list(edge_index.get((source_id, target_id), ()))
        if not directed:
            candidates.extend(edge_index.get((target_id, source_id), ()))
        if not candidates:
            raise IntelligenceError(f"Missing relationship for path step {source_id} -> {target_id}.")
        return tuple(sorted(candidates, key=lambda edge: (edge.relation.casefold(), edge.id)))

    @staticmethod
    def _explain_chain(nodes: tuple[Node, ...], edges: tuple[Edge, ...]) -> str:
        parts = []
        for source, edge, target in zip(nodes, edges, nodes[1:]):
            if edge.source_id == source.id:
                parts.append(f"{source.name} --[{edge.relation}]--> {target.name}")
            else:
                parts.append(f"{source.name} <--[{edge.relation}]-- {target.name}")
        return "Confirmed relationship chain: " + "; then ".join(parts) + "."
