"""Offscreen integration tests for the desktop application."""

import json
import os
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEventLoop, QSettings
from PyQt6.QtWidgets import QDialog, QMessageBox, QApplication

import neurograph.gui as gui_module
from neurograph.engine import GraphEngine
from neurograph.gui import MainWindow, RelationshipVisual


def _wait_until(application, predicate, timeout_ms=10000):
    elapsed = 0
    while elapsed < timeout_ms and not predicate():
        application.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
        application.processEvents()
        from PyQt6.QtTest import QTest

        QTest.qWait(20)
        elapsed += 20
    assert predicate()


def test_desktop_graph_crud_visualization_and_json_actions(tmp_path, monkeypatch):
    application = QApplication.instance() or QApplication([])
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    database = tmp_path / "desktop.sqlite"
    window = MainWindow(database)
    window.show()
    _wait_until(application, lambda: not window._workers)
    assert window.canvas.node_items == {}

    class FakeNodeDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def values(self):
            return "GUI concept", {"type": "research"}

    monkeypatch.setattr(gui_module, "NodeDialog", FakeNodeDialog)
    window._add_node()
    assert window.graph.search_nodes(name="GUI concept")
    node = window.graph.search_nodes(name="GUI concept")[0]
    _wait_until(application, lambda: node.id in window.canvas.node_items)
    assert window.stats_label.text().startswith("1 concept")

    window.graph.add_node("Second concept", node_id="second")
    window._refresh_graph()
    _wait_until(application, lambda: not window._workers)

    class FakeEdgeDialog:
        def __init__(self, nodes, parent=None, **kwargs):
            self.source = SimpleNamespace(currentData=lambda: node.id)
            self.target = SimpleNamespace(currentData=lambda: "second")
            self.relation = SimpleNamespace(text=lambda: "connects")
            self.properties = SimpleNamespace(toPlainText=lambda: '{"weight": 2}')

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(gui_module, "EdgeDialog", FakeEdgeDialog)
    window._add_edge()
    _wait_until(application, lambda: not window._workers)
    edges = window.graph.all_edges()
    assert len(edges) == 1
    assert any(isinstance(item, RelationshipVisual) for item in window.canvas.scene_model.items())
    source_visual = window.canvas.node_items[node.id]
    assert source_visual.flags() & source_visual.GraphicsItemFlag.ItemIsMovable
    before = next(
        item._start.x()
        for item in window.canvas.scene_model.items()
        if isinstance(item, RelationshipVisual)
    )
    source_visual.setPos(source_visual.x() + 30, source_visual.y())
    after = next(
        item._start.x()
        for item in window.canvas.scene_model.items()
        if isinstance(item, RelationshipVisual)
    )
    assert before != after

    window.search_box.setText("second")
    window._apply_filters()
    _wait_until(application, lambda: set(window.canvas.node_items) == {"second"})
    window.search_box.clear()
    window._apply_filters()
    _wait_until(application, lambda: set(window.canvas.node_items) == {node.id, "second"})

    window._inspect_node(node)
    _wait_until(application, lambda: "structural similarity" in window.inspector.toPlainText().casefold())
    details = window.inspector.toPlainText()
    assert "GUI concept" in details
    assert "cluster" in details.casefold()

    exported = tmp_path / "ui-export.json"
    monkeypatch.setattr(
        gui_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args: (str(exported), "JSON files (*.json)")),
    )
    window._export_json()
    assert json.loads(exported.read_text(encoding="utf-8"))["format"] == "neurograph"

    source_path = tmp_path / "source.json"
    with GraphEngine(tmp_path / "source.sqlite") as source:
        source.add_node("Imported from UI", node_id="imported")
        source.export_json(source_path)
    monkeypatch.setattr(
        gui_module.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args: (str(source_path), "JSON files (*.json)")),
    )
    window._import_json()
    _wait_until(application, lambda: not window._workers)
    assert window.graph.get_node("imported").name == "Imported from UI"

    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *args: QMessageBox.StandardButton.Yes)
    )
    relationship = next(
        item for item in window.canvas.scene_model.items() if isinstance(item, RelationshipVisual)
    )
    relationship.setSelected(True)
    window._delete_selected_edge()
    assert window.graph.all_edges() == []
    window._selected_node_id = node.id
    window._delete_selected_node()
    assert all(item.id != node.id for item in window.graph.all_nodes())

    class UnavailableOllama:
        def embed(self, text):
            raise RuntimeError("Local Ollama service unavailable")

    window.ollama_client = UnavailableOllama()
    window.intelligence._ollama = window.ollama_client
    warnings = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *args: warnings.append(args[-1]))
    )
    window.semantic_query.setText("local meaning search")
    window._semantic_search()
    _wait_until(application, lambda: not window._workers)
    assert warnings == ["Local Ollama service unavailable"]
    assert window.graph.get_node("second").name == "Second concept"

    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_chat():
        calls.append(True)
        started.set()
        release.wait(2)
        return "done"

    window._run_chat(slow_chat, lambda _result: None)
    assert started.wait(2)
    assert not window.explain_button.isEnabled()
    assert not window.summarize_button.isEnabled()
    window._run_chat(slow_chat, lambda _result: None)
    assert len(calls) == 1
    release.set()
    _wait_until(application, lambda: not window._chat_busy)
    assert window.explain_button.isEnabled()
    assert window.summarize_button.isEnabled()

    window._show_background_error("Ollama response timed out after 240 seconds")
    assert window.statusBar().currentMessage() == "Ollama timed out; increase its response timeout"
    window.close()
    application.processEvents()
    with GraphEngine(database) as reopened:
        assert reopened.get_node("second").name == "Second concept"
