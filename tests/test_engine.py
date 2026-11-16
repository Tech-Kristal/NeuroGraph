"""Tests for Level 1 graph operations and persistence."""

import json
import socket
import time

import pytest

from neurograph import (
    DuplicateError,
    GraphEngine,
    IntelligenceEngine,
    IntelligenceError,
    NotFoundError,
    OllamaClient,
    ValidationError,
)


def test_node_and_edge_crud_persist_across_reopen(tmp_path):
    database = tmp_path / "graph.sqlite"
    graph = GraphEngine(database)
    first = graph.add_node("Python", {"kind": "language"}, node_id="python")
    second = graph.add_node("NetworkX", node_id="networkx")
    edge = graph.add_edge(first.id, second.id, "uses", {"since": 2005}, edge_id="relationship")

    updated_node = graph.update_node(first.id, name="Python language", properties={"kind": "code"})
    updated_edge = graph.update_edge(edge.id, relation="depends on")
    assert updated_node.name == "Python language"
    assert updated_edge.relation == "depends on"
    graph.close()
    reopened = GraphEngine(database)
    assert reopened.get_node("python").properties == {"kind": "code"}
    assert reopened.get_edge("relationship").relation == "depends on"
    reopened.delete_edge("relationship")
    reopened.delete_node("python")
    assert reopened.all_edges() == []
    assert [node.id for node in reopened.all_nodes()] == ["networkx"]
    reopened.close()


def test_shortest_paths_preserve_parallel_relationship_alternatives(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    graph.add_node("A", node_id="a")
    graph.add_node("B", node_id="b")
    graph.add_edge("a", "b", "causes", edge_id="causes")
    graph.add_edge("a", "b", "influences", edge_id="influences")

    paths = IntelligenceEngine(graph).find_paths("a", "b", limit=2)
    assert {path.edges[0].relation for path in paths} == {"causes", "influences"}
    graph.close()


def test_undirected_path_explanation_preserves_confirmed_edge_direction(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    graph.add_node("A", node_id="a")
    graph.add_node("B", node_id="b")
    graph.add_edge("b", "a", "depends on")

    path = IntelligenceEngine(graph).find_paths("a", "b", directed=False)[0]
    assert path.explanation == "Confirmed relationship chain: A <--[depends on]-- B."
    graph.close()


def test_traversal_is_breadth_first_and_respects_depth_and_direction(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    nodes = [graph.add_node(name, node_id=name) for name in "ABCD"]
    graph.add_edge(nodes[0].id, nodes[1].id, "to")
    graph.add_edge(nodes[0].id, nodes[2].id, "to")
    graph.add_edge(nodes[1].id, nodes[3].id, "to")
    graph.add_edge(nodes[3].id, nodes[0].id, "cycle")

    assert [node.id for node in graph.traverse("A")] == ["B", "C", "D"]
    assert [node.id for node in graph.traverse("A", max_depth=1)] == ["B", "C"]
    assert [node.id for node in graph.traverse("A", max_depth=0, include_start=True)] == ["A"]
    assert [node.id for node in graph.traverse("A", direction="incoming")] == ["D", "B"]
    with pytest.raises(ValidationError):
        graph.traverse("A", max_depth="invalid")
    graph.close()


def test_search_duplicates_and_missing_entities(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    graph.add_node("  Mercury ", {"type": "planet"}, node_id="one")
    graph.add_node("mercury", {"type": "element"}, node_id="two")
    graph.add_node("Venus", {"type": "planet"}, node_id="three")

    assert {node.id for node in graph.search_nodes(name="CUR", properties={"type": "planet"})} == {"one"}
    assert {node.id for node in graph.search_nodes(properties={"type": "planet"})} == {"one", "three"}
    assert [[node.id for node in group] for group in graph.find_duplicate_nodes()] == [["one", "two"]]
    with pytest.raises(NotFoundError):
        graph.get_node("missing")
    with pytest.raises(NotFoundError):
        graph.delete_edge("missing")
    with pytest.raises(ValidationError):
        graph.search_nodes()
    graph.close()


def test_json_round_trip_and_replace_import(tmp_path):
    exported = tmp_path / "export.json"
    graph = GraphEngine(tmp_path / "source.sqlite")
    source = graph.add_node("Source", {"list": [1, 2]}, node_id="source")
    destination = graph.add_node("Destination", node_id="destination")
    graph.add_edge(source.id, destination.id, "links", edge_id="link")
    graph.export_json(exported)
    document = json.loads(exported.read_text(encoding="utf-8"))
    assert document["version"] == 1

    imported = GraphEngine(tmp_path / "target.sqlite")
    imported.add_node("Existing", node_id="existing")
    assert imported.import_json(exported) == (2, 1)
    assert imported.get_node("source").properties == {"list": [1, 2]}
    assert imported.import_json(exported, replace=True) == (2, 1)
    assert {node.id for node in imported.all_nodes()} == {"source", "destination"}
    graph.close()
    imported.close()


def test_failed_import_is_atomic_and_rejects_bad_edges(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    graph.add_node("Keep", node_id="keep")
    malformed = tmp_path / "bad.json"
    malformed.write_text(
        json.dumps(
            {
                "format": "neurograph",
                "version": 1,
                "nodes": [{"id": "new", "name": "New"}],
                "edges": [
                    {
                        "id": "bad-edge",
                        "source_id": "new",
                        "target_id": "missing",
                        "relation": "points to",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        graph.import_json(malformed, replace=True)
    assert [node.id for node in graph.all_nodes()] == ["keep"]
    with pytest.raises(DuplicateError):
        graph.add_node("Different", node_id="keep")
    graph.close()


@pytest.mark.parametrize("properties", [{"invalid": object()}, {"not finite": float("nan")}])
def test_node_rejects_non_json_properties(tmp_path, properties):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    with pytest.raises(ValidationError):
        graph.add_node("Invalid", properties)
    assert graph.all_nodes() == []
    graph.close()


def test_invalid_ids_and_non_string_property_keys_are_rejected(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    with pytest.raises(ValidationError):
        graph.add_node("Invalid ID", node_id="")
    with pytest.raises(ValidationError):
        graph.add_node("Invalid property", {"nested": {1: "not a JSON key"}})
    assert graph.all_nodes() == []
    graph.close()


def test_shortest_paths_include_explanations_alternatives_and_disconnected_case(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    for node_id in ("a", "b", "c", "d", "island"):
        graph.add_node(node_id.upper(), node_id=node_id)
    graph.add_edge("a", "b", "knows")
    graph.add_edge("b", "d", "created")
    graph.add_edge("a", "c", "uses")
    graph.add_edge("c", "d", "supports")
    paths = IntelligenceEngine(graph).find_paths("a", "d", limit=3)

    assert len(paths) == 2
    assert all(path.length == 2 for path in paths)
    assert {tuple(node.id for node in path.nodes) for path in paths} == {
        ("a", "b", "d"),
        ("a", "c", "d"),
    }
    assert all("Confirmed relationship chain" in path.explanation for path in paths)
    assert IntelligenceEngine(graph).find_paths("a", "island") == []
    assert IntelligenceEngine(graph).find_paths("a", "a")[0].length == 0
    with pytest.raises(NotFoundError):
        IntelligenceEngine(graph).find_paths("a", "missing")
    graph.close()


def test_similarity_uses_shared_neighbors_without_ollama(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    for node_id in ("a", "b", "shared", "only-a", "only-b"):
        graph.add_node(node_id, node_id=node_id)
    for source, target in (
        ("a", "shared"),
        ("a", "only-a"),
        ("b", "shared"),
        ("b", "only-b"),
    ):
        graph.add_edge(source, target, "related")
    result = IntelligenceEngine(graph).similarity("a", "b")
    assert result.structural_score == pytest.approx(1 / 3)
    assert result.semantic_score is None
    assert result.score == result.structural_score
    graph.close()


def test_semantic_search_and_embedding_similarity_use_local_client(tmp_path):
    class LocalEmbeddings:
        vectors = {
            "meaning of river": [1.0, 0.0],
            "stream\n{}": [0.9, 0.1],
            "mountain\n{}": [0.0, 1.0],
        }

        def embed(self, text):
            return self.vectors[text]

        def generate(self, prompt):
            return f"Local summary for {len(prompt)} characters."

    graph = GraphEngine(tmp_path / "graph.sqlite")
    graph.add_node("stream", node_id="stream")
    graph.add_node("mountain", node_id="mountain")
    intelligence = IntelligenceEngine(graph, ollama=LocalEmbeddings())

    matches = intelligence.semantic_search("meaning of river", limit=1)
    assert matches[0].node.id == "stream"
    assert matches[0].score > 0.99
    similarity = intelligence.similarity("stream", "mountain")
    assert similarity.semantic_score == pytest.approx(0.110431526)
    assert similarity.score == pytest.approx(0.055215763)
    graph.add_edge("stream", "mountain", "near")
    path = intelligence.find_paths("stream", "mountain")[0]
    assert intelligence.explain_with_ollama(path).startswith("Local summary")
    cluster = intelligence.cluster_concepts()[0]
    assert intelligence.summarize_cluster(cluster).startswith("Local summary")
    graph.close()


def test_clustering_handles_empty_isolated_and_connected_concepts(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    intelligence = IntelligenceEngine(graph)
    assert intelligence.cluster_concepts() == []
    for node_id in ("a", "b", "c", "d", "e", "f", "alone"):
        graph.add_node(node_id, node_id=node_id)
    for source, target in (
        ("a", "b"),
        ("b", "c"),
        ("c", "a"),
        ("d", "e"),
        ("e", "f"),
        ("f", "d"),
    ):
        graph.add_edge(source, target, "connected")
    clusters = intelligence.cluster_concepts()
    members = {frozenset(cluster.node_ids) for cluster in clusters}
    assert members == {
        frozenset({"a", "b", "c"}),
        frozenset({"d", "e", "f"}),
        frozenset({"alone"}),
    }
    graph.close()


def test_transitive_inference_is_unconfirmed_and_never_duplicates_facts(tmp_path):
    graph = GraphEngine(tmp_path / "graph.sqlite")
    for node_id in ("a", "b", "c", "d"):
        graph.add_node(node_id.upper(), node_id=node_id)
    graph.add_edge("a", "b", "is_a", edge_id="ab")
    graph.add_edge("b", "c", "is_a", edge_id="bc")
    graph.add_edge("c", "a", "is_a", edge_id="ca")
    graph.add_edge("a", "c", "is_a", edge_id="confirmed")
    graph.add_edge("c", "d", "part_of", edge_id="cd")
    graph.add_edge("d", "b", "part_of", edge_id="db")

    inferred = IntelligenceEngine(graph).infer_transitive()
    triples = {(item.source.id, item.target.id, item.relation) for item in inferred}
    assert ("a", "c", "is_a") not in triples
    assert ("a", "a", "is_a") not in triples
    assert ("a", "c", "is_a") in {
        (edge.source_id, edge.target_id, edge.relation) for edge in graph.all_edges()
    }
    assert all(not item.confirmed and "not a confirmed fact" in item.explanation for item in inferred)
    assert len(graph.all_edges()) == 6
    assert ("c", "b", "part_of") in triples
    with pytest.raises(ValidationError):
        IntelligenceEngine(graph).infer_transitive("is_a")
    graph.close()


def test_ollama_client_is_local_only_and_requires_explicit_configuration(tmp_path):
    with pytest.raises(ValidationError):
        OllamaClient(base_url="https://example.com")
    with pytest.raises(ValidationError):
        OllamaClient(base_url="http://localhost:bad")
    with pytest.raises(ValidationError):
        OllamaClient(base_url="http://localhost:11434/path")
    with pytest.raises(ValidationError):
        OllamaClient(timeout=float("nan"))
    graph = GraphEngine(tmp_path / "graph.sqlite")
    with pytest.raises(IntelligenceError, match="configured local OllamaClient"):
        IntelligenceEngine(graph).semantic_search("meaning")
    graph.close()


def test_ollama_default_model_and_base_url_normalization():
    client = OllamaClient(base_url=" http://localhost:11434/// ")
    assert client.model == "llama3:latest"
    assert client.base_url == "http://localhost:11434"


def test_ollama_client_uses_configured_local_models(monkeypatch):
    import neurograph.ollama as ollama_module

    requests = []
    responses = iter(
        [
            (200, {"embeddings": [[0.25, 0.75]]}),
            (200, {"message": {"content": "A local explanation."}}),
        ]
    )

    class Response:
        def __init__(self, status, value):
            self.status = status
            self.reason = "OK"
            self.value = value

        def read(self, amount=-1):
            return json.dumps(self.value).encode("utf-8")

    class FakeSocket:
        def settimeout(self, timeout):
            requests[-1]["response_timeout"] = timeout

    class FakeConnection:
        def __init__(self, host, port, timeout):
            self.host = host
            self.port = port
            self.connect_timeout = timeout
            self.sock = FakeSocket()

        def connect(self):
            requests.append(
                {"host": self.host, "port": self.port, "connect_timeout": self.connect_timeout}
            )

        def request(self, method, endpoint, body, headers):
            requests[-1].update(
                method=method,
                endpoint=endpoint,
                payload=json.loads(body),
                headers=headers,
            )

        def getresponse(self):
            return Response(*next(responses))

        def close(self):
            pass

    monkeypatch.setattr(ollama_module.http.client, "HTTPConnection", FakeConnection)
    client = OllamaClient(
        model="local-chat",
        embedding_model="local-embeddings",
        timeout=240,
        connection_timeout=4,
    )
    assert client.embed("concept") == [0.25, 0.75]
    assert client.generate("explain this") == "A local explanation."
    assert [request["endpoint"] for request in requests] == ["/api/embed", "/api/chat"]
    payloads = [request["payload"] for request in requests]
    assert payloads[0]["model"] == "local-embeddings"
    assert payloads[1]["model"] == "local-chat"
    assert payloads[1]["messages"] == [{"role": "user", "content": "explain this"}]
    assert payloads[1]["stream"] is False
    assert payloads[1]["keep_alive"] == "10m"
    assert all(request["headers"]["Content-Type"] == "application/json" for request in requests)
    assert all(request["connect_timeout"] == 4 for request in requests)
    assert all(request["response_timeout"] == 240 for request in requests)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"error": "model 'missing:tag' not found"}, "Ollama model 'missing:tag' was not found"),
        ({"error": "route not found"}, "HTTP 404"),
    ],
)
def test_ollama_http_404_preserves_server_error(monkeypatch, body, message):
    import neurograph.ollama as ollama_module

    class Response:
        status = 404
        reason = "Not Found"

        def read(self, amount=-1):
            return json.dumps(body).encode("utf-8")

    class FakeConnection:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(ollama_module.http.client, "HTTPConnection", FakeConnection)
    client = OllamaClient(model="missing:tag")
    with pytest.raises(IntelligenceError, match=message) as error:
        client.generate("hello")
    assert error.value.__cause__ is None
    assert "not found" in str(error.value)


def test_ollama_response_timeout_and_unavailable_connection(monkeypatch):
    import neurograph.ollama as ollama_module

    class TimedOutConnection:
        sock = None

        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            raise socket.timeout("timed out")

        def close(self):
            pass

    monkeypatch.setattr(ollama_module.http.client, "HTTPConnection", TimedOutConnection)
    with pytest.raises(IntelligenceError, match="Ollama response timed out after 240 seconds"):
        OllamaClient().generate("hello")

    class UnavailableConnection(TimedOutConnection):
        def connect(self):
            raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(ollama_module.http.client, "HTTPConnection", UnavailableConnection)
    with pytest.raises(IntelligenceError, match="Could not connect to local Ollama service"):
        OllamaClient().generate("hello")


def test_ollama_slow_response_succeeds_with_extended_timeout(monkeypatch):
    import neurograph.ollama as ollama_module

    class Response:
        status = 200
        reason = "OK"

        def read(self, amount=-1):
            return b'{"message":{"content":"finished"}}'

    class SlowConnection:
        def __init__(self, *args, timeout, **kwargs):
            assert timeout == 10
            self.sock = self
            self.response_timeout = None

        def connect(self):
            pass

        def settimeout(self, timeout):
            self.response_timeout = timeout

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            assert self.response_timeout == 180
            time.sleep(0.02)
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(ollama_module.http.client, "HTTPConnection", SlowConnection)
    assert OllamaClient(timeout=180).generate("hello") == "finished"
