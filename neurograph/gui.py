"""PyQt6 desktop interface for NeuroGraph."""

from __future__ import annotations

from collections import Counter
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

import networkx as nx
from PyQt6.QtCore import QByteArray, QSettings, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QActionGroup, QColor, QKeySequence, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from neurograph import render
from neurograph.canvas import GraphCanvas, GraphNodeVisual, RelationshipVisual, node_category
from neurograph.dialogs import EdgeDialog, NodeDialog
from neurograph.engine import GraphEngine
from neurograph.errors import GraphError, ValidationError
from neurograph.intelligence import IntelligenceEngine
from neurograph.layout import calculate_layout
from neurograph.models import Edge, Node
from neurograph.ollama import OllamaClient
from neurograph.tasks import TaskRunner
from neurograph.theme import DARK, Theme, build_stylesheet, category_colors, get_theme
from neurograph.widgets import (
    color_icon,
    combo_node_id,
    create_node_combo,
    hairline,
    section_label,
)

__all__ = [
    "EdgeDialog",
    "GraphCanvas",
    "GraphNodeVisual",
    "MainWindow",
    "NodeDialog",
    "RelationshipVisual",
    "apply_theme",
    "install_theme",
]

logger = logging.getLogger(__name__)

_MAX_VISIBLE_NODES = 1200
_ACTIVE_THEME: Theme = DARK


# --- theming -------------------------------------------------------------------


def apply_theme(application: QApplication, theme: Theme) -> None:
    """Apply a palette and style sheet for ``theme`` to the whole application."""
    global _ACTIVE_THEME
    _ACTIVE_THEME = theme
    application.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: theme.window,
        QPalette.ColorRole.WindowText: theme.text,
        QPalette.ColorRole.Base: theme.input,
        QPalette.ColorRole.AlternateBase: theme.panel,
        QPalette.ColorRole.Text: theme.text,
        QPalette.ColorRole.Button: theme.button,
        QPalette.ColorRole.ButtonText: theme.text,
        QPalette.ColorRole.ToolTipBase: theme.panel,
        QPalette.ColorRole.ToolTipText: theme.text,
        QPalette.ColorRole.Highlight: theme.accent,
        QPalette.ColorRole.HighlightedText: theme.accent_text,
        QPalette.ColorRole.Link: theme.accent,
        QPalette.ColorRole.PlaceholderText: theme.text_muted,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(theme.text_disabled))
    application.setPalette(palette)
    application.setStyleSheet(build_stylesheet(theme))


def install_theme(application: QApplication, name: str | None = None) -> None:
    """Apply the saved (or named) theme; defaults to dark."""
    if name is None:
        name = str(QSettings().value("theme", "dark"))
    apply_theme(application, get_theme(name))


# --- main window ---------------------------------------------------------------


class MainWindow(QMainWindow):
    """NeuroGraph desktop application window."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        super().__init__()
        self.database_path = (
            Path(database_path).expanduser() if database_path else self.default_database_path()
        )
        if str(self.database_path) == ":memory:":
            raise ValidationError("The desktop application requires a persistent SQLite database.")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = GraphEngine(self.database_path)
        self.ollama_client = OllamaClient(
            base_url=os.environ.get("NEUROGRAPH_OLLAMA_URL", "http://localhost:11434"),
            model=os.environ.get("NEUROGRAPH_OLLAMA_MODEL", "llama3:latest"),
            embedding_model=os.environ.get(
                "NEUROGRAPH_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
            ),
            timeout=float(os.environ.get("NEUROGRAPH_OLLAMA_TIMEOUT", "240")),
            connection_timeout=float(os.environ.get("NEUROGRAPH_OLLAMA_CONNECT_TIMEOUT", "10")),
            keep_alive=os.environ.get("NEUROGRAPH_OLLAMA_KEEP_ALIVE", "10m"),
        )
        self.intelligence = IntelligenceEngine(self.graph, ollama=self.ollama_client)
        self.tasks = TaskRunner(self)
        self.theme: Theme = _ACTIVE_THEME
        self._settings = QSettings()

        self._all_nodes: list[Node] = []
        self._all_edges: list[Edge] = []
        self._nodes_by_id: dict[str, Node] = {}
        self._category_colors: dict[str, str] = {}
        self._selected_node_id: str | None = None
        self._edge_selected = False
        self._chat_busy = False
        self._refresh_generation = 0
        self._layout_generation = 0
        self._data_version = 0
        self._layout_key: tuple[int, frozenset[str]] | None = None
        self._force_full_layout = False
        self._desired_ids: set[str] = set()
        self._pending_selection: str | None = None
        self._inspector_render: Callable[[Theme], str] | None = None
        self._result_blocks: list[Callable[[Theme], str]] = []
        self._last_paths: list[Any] = []
        self._last_clusters: list[Any] = []

        self._pending_filter = QTimer(self)
        self._pending_filter.setSingleShot(True)
        self._pending_filter.timeout.connect(self._apply_filters)

        self.setWindowTitle(f"{self.database_path.name} — NeuroGraph")
        self.resize(1360, 860)
        self.setMinimumSize(980, 620)
        self._build_ui()
        self._build_actions()
        self._connect_signals()
        self._restore_window_state()
        self._show_empty_inspector()
        self._render_results()
        self._update_actions()
        self._refresh_graph()

    @property
    def _workers(self) -> dict[int, Any]:
        """Tasks still running or awaiting delivery (empty when the UI is idle)."""
        return self.tasks.pending

    @staticmethod
    def default_database_path() -> Path:
        """Return a portable per-user application data path."""
        from PyQt6.QtCore import QStandardPaths

        location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        base = Path(location) if location else Path.home() / ".neurograph"
        return base / "knowledge.sqlite"

    # --- construction -----------------------------------------------------------

    def _build_ui(self) -> None:
        self.canvas = GraphCanvas(self.theme)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(5)
        self.splitter.addWidget(self._build_explorer())
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self._build_right_panel())
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([270, 760, 380])
        self.setCentralWidget(self.splitter)

        self.stats_label = QLabel("")
        self.db_label = QLabel(self.database_path.name)
        self.db_label.setToolTip(str(self.database_path))
        self.statusBar().addPermanentWidget(self.stats_label)
        self.statusBar().addPermanentWidget(self.db_label)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage("Loading graph…")

    def _build_explorer(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("leftPanel")
        panel.setMinimumWidth(210)
        panel.setMaximumWidth(380)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("Concepts")
        title.setObjectName("panelTitle")
        self.count_label = QLabel("")
        self.count_label.setObjectName("mutedLabel")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.count_label)
        layout.addLayout(header)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by name")
        self.search_box.setClearButtonEnabled(True)
        layout.addWidget(self.search_box)
        self.category_filter = QComboBox()
        layout.addWidget(self.category_filter)
        self.concept_list = QListWidget()
        self.concept_list.setUniformItemSizes(True)
        layout.addWidget(self.concept_list, 1)
        self.explorer_note = QLabel("")
        self.explorer_note.setObjectName("mutedLabel")
        self.explorer_note.setWordWrap(True)
        layout.addWidget(self.explorer_note)
        return panel

    def _make_browser(self) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setOpenLinks(False)
        browser.setOpenExternalLinks(False)
        browser.document().setDocumentMargin(14)
        browser.anchorClicked.connect(self._on_anchor)
        return browser

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("rightPanel")
        panel.setMinimumWidth(330)
        panel.setMaximumWidth(540)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.inspector = self._make_browser()
        self.tabs.addTab(self.inspector, "Details")
        self.analysis_panel = self._build_analysis_panel()
        self.tabs.addTab(self.analysis_panel, "Analysis")
        layout.addWidget(self.tabs)
        return panel

    def _build_analysis_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(6)

        layout.addWidget(section_label("Find paths"))
        self.path_source = create_node_combo([], {})
        self.path_target = create_node_combo([], {})
        for combo, hint in ((self.path_source, "From concept…"), (self.path_target, "To concept…")):
            line = combo.lineEdit()
            if line is not None:
                line.setPlaceholderText(hint)
            layout.addWidget(combo)
        row = QHBoxLayout()
        self.find_paths_button = QPushButton("Find paths")
        self.find_paths_button.setObjectName("primary")
        self.explain_button = QPushButton("Explain with Ollama")
        row.addWidget(self.find_paths_button)
        row.addWidget(self.explain_button)
        layout.addLayout(row)

        layout.addSpacing(8)
        layout.addWidget(hairline())
        layout.addSpacing(4)
        layout.addWidget(section_label("Semantic search (needs Ollama)"))
        self.semantic_query = QLineEdit()
        self.semantic_query.setPlaceholderText("Describe a concept or idea")
        layout.addWidget(self.semantic_query)
        self.semantic_button = QPushButton("Search by meaning")
        layout.addWidget(self.semantic_button)

        layout.addSpacing(8)
        layout.addWidget(hairline())
        layout.addSpacing(4)
        layout.addWidget(section_label("Structure"))
        row = QHBoxLayout()
        self.cluster_button = QPushButton("Cluster concepts")
        self.infer_button = QPushButton("Suggest links")
        row.addWidget(self.cluster_button)
        row.addWidget(self.infer_button)
        layout.addLayout(row)
        self.summarize_button = QPushButton("Summarize first cluster with Ollama")
        layout.addWidget(self.summarize_button)

        layout.addSpacing(8)
        layout.addWidget(hairline())
        layout.addSpacing(4)
        layout.addWidget(section_label("Results"))
        self.intelligence_results = self._make_browser()
        layout.addWidget(self.intelligence_results, 1)
        return panel

    def _make_action(
        self, text: str, slot: Callable[[], None], shortcut: str | list[str] | None = None
    ) -> QAction:
        action = QAction(text, self)
        if shortcut:
            keys = [shortcut] if isinstance(shortcut, str) else shortcut
            action.setShortcuts([QKeySequence(key) for key in keys])
        action.triggered.connect(lambda _checked=False: slot())
        return action

    def _build_actions(self) -> None:
        self.act_import = self._make_action("Import JSON…", self._import_json, "Ctrl+O")
        self.act_export = self._make_action("Export JSON…", self._export_json, "Ctrl+E")
        self.act_quit = self._make_action("Quit", self.close, "Ctrl+Q")
        self.act_add_node = self._make_action("Add concept…", self._add_node, "Ctrl+N")
        self.act_add_edge = self._make_action("Add relationship…", self._add_edge, "Ctrl+L")
        self.act_edit_node = self._make_action("Edit concept…", self._edit_selected_node, "F2")
        self.act_delete = self._make_action("Delete selection…", self._delete_selection, "Delete")
        self.act_find = self._make_action("Search concepts", self._focus_search, "Ctrl+F")
        self.act_fit = self._make_action("Fit to window", lambda: self.canvas.fit_all(), "Ctrl+0")
        self.act_zoom_in = self._make_action(
            "Zoom in", lambda: self.canvas.zoom(1.25), ["Ctrl+=", "Ctrl++"]
        )
        self.act_zoom_out = self._make_action("Zoom out", lambda: self.canvas.zoom(0.8), "Ctrl+-")
        self.act_relayout = self._make_action("Re-arrange layout", self._relayout, "Ctrl+R")
        self.act_refresh = self._make_action("Reload from database", self._refresh_graph, "F5")
        self.act_labels = QAction("Always show labels", self)
        self.act_labels.setCheckable(True)
        self.act_find_paths = self._make_action("Find paths", self._find_paths)
        self.act_cluster = self._make_action("Cluster concepts", self._cluster_graph)
        self.act_infer = self._make_action("Suggest links", self._infer_links)
        self.act_about = self._make_action("About NeuroGraph", self._about)

        self.theme_group = QActionGroup(self)
        self.theme_actions: dict[str, QAction] = {}
        for name, label in (("dark", "Dark"), ("light", "Light")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(self.theme.name == name)
            action.triggered.connect(lambda _checked=False, value=name: self._set_theme(value))
            self.theme_group.addAction(action)
            self.theme_actions[name] = action

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addActions([self.act_import, self.act_export])
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addActions([self.act_add_node, self.act_add_edge])
        edit_menu.addSeparator()
        edit_menu.addActions([self.act_edit_node, self.act_delete])
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_find)
        view_menu = menubar.addMenu("&View")
        view_menu.addActions([self.act_fit, self.act_zoom_in, self.act_zoom_out])
        view_menu.addSeparator()
        view_menu.addActions([self.act_labels, self.act_relayout, self.act_refresh])
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        theme_menu.addActions(list(self.theme_actions.values()))
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addActions([self.act_find_paths, self.act_cluster, self.act_infer])
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.act_about)

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toolbar.addActions([self.act_add_node, self.act_add_edge])
        toolbar.addSeparator()
        toolbar.addActions([self.act_edit_node, self.act_delete])
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addActions([self.act_zoom_out, self.act_zoom_in, self.act_fit])

    def _connect_signals(self) -> None:
        self.search_box.textChanged.connect(lambda _text: self._pending_filter.start(180))
        self.category_filter.currentIndexChanged.connect(lambda _index: self._apply_filters())
        self.concept_list.currentItemChanged.connect(self._on_list_current_changed)
        self.canvas.node_selected.connect(self._inspect_node)
        self.canvas.edge_selected.connect(self._inspect_edge)
        self.canvas.selection_cleared.connect(self._clear_selection_state)
        self.canvas.node_activated.connect(self._edit_node)
        self.canvas.context_requested.connect(self._show_context_menu)
        self.act_labels.toggled.connect(self.canvas.set_always_labels)
        self.find_paths_button.clicked.connect(lambda _checked=False: self._find_paths())
        self.explain_button.clicked.connect(lambda _checked=False: self._explain_first_path())
        self.semantic_button.clicked.connect(lambda _checked=False: self._semantic_search())
        self.semantic_query.returnPressed.connect(self._semantic_search)
        self.cluster_button.clicked.connect(lambda _checked=False: self._cluster_graph())
        self.infer_button.clicked.connect(lambda _checked=False: self._infer_links())
        self.summarize_button.clicked.connect(lambda _checked=False: self._summarize_first_cluster())

    # --- window state -----------------------------------------------------------

    def _restore_window_state(self) -> None:
        try:
            geometry = self._settings.value("window/geometry")
            if isinstance(geometry, QByteArray):
                self.restoreGeometry(geometry)
            sizes = self._settings.value("window/splitter")
            if sizes:
                values = [int(value) for value in sizes]
                if len(values) == 3 and all(value > 0 for value in values):
                    self.splitter.setSizes(values)
        except (TypeError, ValueError):
            logger.debug("Ignoring unreadable window settings", exc_info=True)

    def _save_window_state(self) -> None:
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/splitter", self.splitter.sizes())

    def _set_theme(self, name: str) -> None:
        theme = get_theme(name)
        application = QApplication.instance()
        if application is not None:
            apply_theme(application, theme)
        self.theme = theme
        self.theme_actions[theme.name].setChecked(True)
        self._settings.setValue("theme", theme.name)
        self.canvas.apply_theme(theme)
        self._category_colors = category_colors(
            sorted({node_category(node) for node in self._all_nodes}), theme
        )
        self._populate_category_filter()
        self._rerender_inspector()
        self._render_results()
        # Node colours are baked into the canvas items, so rebuild them (positions are kept).
        self._layout_key = None
        self._apply_filters()

    # --- loading ----------------------------------------------------------------

    def _refresh_graph(self) -> None:
        self._refresh_generation += 1
        generation = self._refresh_generation
        self.statusBar().showMessage("Loading graph…")
        path = self.database_path
        self.tasks.submit(
            lambda: self._load_graph_snapshot(path),
            lambda snapshot: self._on_graph_loaded(snapshot, generation),
            self._show_background_error,
        )

    @staticmethod
    def _load_graph_snapshot(database_path: Path) -> tuple[list[Node], list[Edge], int]:
        with GraphEngine(database_path) as graph:
            nodes = graph.all_nodes()
            edges = graph.all_edges()
        components = (
            nx.number_weakly_connected_components(MainWindow._nx_graph(nodes, edges)) if nodes else 0
        )
        return nodes, edges, components

    @staticmethod
    def _nx_graph(nodes: list[Node], edges: list[Edge]) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(node.id for node in nodes)
        graph.add_edges_from((edge.source_id, edge.target_id) for edge in edges)
        return graph

    @staticmethod
    def _calculate_layout(
        nodes: list[Node],
        edges: list[Edge],
        previous: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, tuple[float, float]]:
        return calculate_layout(nodes, edges, previous)

    def _on_graph_loaded(self, snapshot: tuple[list[Node], list[Edge], int], generation: int) -> None:
        if generation != self._refresh_generation:
            return
        nodes, edges, components = snapshot
        self._all_nodes = nodes
        self._all_edges = edges
        self._nodes_by_id = {node.id: node for node in nodes}
        self._data_version += 1
        self._category_colors = category_colors(
            sorted({node_category(node) for node in nodes}), self.theme
        )
        self._update_combos(nodes)
        self._update_stats(nodes, edges, components)
        self._populate_category_filter()
        self._apply_filters()
        self._update_actions()

    def _populate_category_filter(self) -> None:
        counts = Counter(node_category(node) for node in self._all_nodes)
        current = self.category_filter.currentData()
        self.category_filter.blockSignals(True)
        self.category_filter.clear()
        self.category_filter.addItem("All categories", None)
        for category in sorted(counts, key=str.casefold):
            color = self._category_colors.get(category, self.theme.node_neutral)
            self.category_filter.addItem(color_icon(color), f"{category}  ({counts[category]:,})", category)
        index = self.category_filter.findData(current)
        self.category_filter.setCurrentIndex(index if index >= 0 else 0)
        self.category_filter.blockSignals(False)

    def _update_stats(self, nodes: list[Node], edges: list[Edge], components: int) -> None:
        concepts = "concept" if len(nodes) == 1 else "concepts"
        relationships = "relationship" if len(edges) == 1 else "relationships"
        parts = "component" if components == 1 else "components"
        self.stats_label.setText(
            f"{len(nodes):,} {concepts}  ·  {len(edges):,} {relationships}  ·  {components:,} {parts}"
        )

    def _update_combos(self, nodes: list[Node]) -> None:
        shown = sorted(nodes[:_MAX_VISIBLE_NODES], key=lambda n: n.name.casefold())
        labels = render.node_labels(shown)
        for combo in (self.path_source, self.path_target):
            selected = combo_node_id(combo)
            combo.clear()
            for node in shown:
                combo.addItem(labels[node.id], node.id)
            combo.setCurrentIndex(combo.findData(selected) if selected else -1)

    # --- filtering, explorer and layout ---------------------------------------------

    def _apply_filters(self) -> None:
        needle = self.search_box.text().casefold().strip()
        category = self.category_filter.currentData()
        matches = [
            node
            for node in self._all_nodes
            if (not needle or needle in node.name.casefold())
            and (category is None or node_category(node) == category)
        ]
        total_matches = len(matches)
        matches = matches[:_MAX_VISIBLE_NODES]
        self._populate_explorer(matches, total_matches)
        matched_ids = {node.id for node in matches}
        self._desired_ids = matched_ids
        key = (self._data_version, frozenset(matched_ids))
        if key == self._layout_key:
            return
        self._layout_key = key
        self._layout_generation += 1
        generation = self._layout_generation
        visible_edges = [
            edge for edge in self._all_edges if edge.source_id in matched_ids and edge.target_id in matched_ids
        ]
        if not matches:
            self.canvas.display_graph([], [], {}, self._category_colors)
            self._finish_display(0)
            return
        previous = {} if self._force_full_layout else self.canvas.positions()
        self._force_full_layout = False
        self.statusBar().showMessage(f"Arranging {len(matches):,} concepts…")
        self.tasks.submit(
            lambda: self._calculate_layout(matches, visible_edges, previous),
            lambda positions: self._on_layout_ready(matches, visible_edges, positions, generation),
            self._on_layout_failed,
        )

    def _populate_explorer(self, matches: list[Node], total_matches: int) -> None:
        self.concept_list.blockSignals(True)
        self.concept_list.clear()
        for node in sorted(matches, key=lambda n: n.name.casefold()):
            category = node_category(node)
            color = self._category_colors.get(category, self.theme.node_neutral)
            item = QListWidgetItem(color_icon(color), node.name)
            item.setData(Qt.ItemDataRole.UserRole, node.id)
            item.setToolTip(category)
            self.concept_list.addItem(item)
        self.concept_list.blockSignals(False)
        self._sync_list_selection(self._selected_node_id)
        total = len(self._all_nodes)
        self.count_label.setText(
            f"{total_matches:,}" if total_matches == total else f"{total_matches:,} of {total:,}"
        )
        self.explorer_note.setText(
            f"Showing the first {_MAX_VISIBLE_NODES:,} matches. Refine the search to focus."
            if total_matches > _MAX_VISIBLE_NODES
            else ""
        )

    def _sync_list_selection(self, node_id: str | None) -> None:
        self.concept_list.blockSignals(True)
        try:
            self.concept_list.setCurrentRow(-1)
            if node_id is None:
                return
            for row in range(self.concept_list.count()):
                item = self.concept_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == node_id:
                    self.concept_list.setCurrentItem(item)
                    self.concept_list.scrollToItem(item)
                    return
        finally:
            self.concept_list.blockSignals(False)

    def _on_list_current_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None:
            self._select_node(current.data(Qt.ItemDataRole.UserRole))

    def _on_layout_ready(
        self,
        nodes: list[Node],
        edges: list[Edge],
        positions: dict[str, tuple[float, float]],
        generation: int,
    ) -> None:
        if generation != self._layout_generation:
            return
        self.canvas.display_graph(nodes, edges, positions, self._category_colors)
        self._finish_display(len(nodes))
        pending = self._pending_selection
        self._pending_selection = None
        wanted = pending or self._selected_node_id
        if wanted and wanted in self.canvas.node_items:
            self._select_node(wanted, center=pending is not None)
        elif self._selected_node_id is not None or self._edge_selected:
            self._clear_selection_state()

    def _on_layout_failed(self, message: str) -> None:
        self._layout_key = None
        self._show_background_error(message)

    def _finish_display(self, shown: int) -> None:
        total = len(self._all_nodes)
        if not total:
            self.statusBar().showMessage("The graph is empty")
        elif shown < total:
            self.statusBar().showMessage(f"Showing {shown:,} of {total:,} concepts")
        else:
            self.statusBar().showMessage("Ready")

    def _relayout(self) -> None:
        self._force_full_layout = True
        self._layout_key = None
        self._apply_filters()

    def _focus_search(self) -> None:
        self.search_box.setFocus()
        self.search_box.selectAll()

    # --- selection and inspection ---------------------------------------------------

    def _select_node(self, node_id: str, *, center: bool = True) -> None:
        node = self._nodes_by_id.get(node_id)
        if node is None:
            return
        self.canvas.select_node(node_id, center=center)
        self._inspect_node(node)

    def _show_empty_inspector(self) -> None:
        self._set_inspector(render.empty_selection)

    def _set_inspector(self, build: Callable[[Theme], str]) -> None:
        self._inspector_render = build
        self.inspector.setHtml(build(self.theme))

    def _rerender_inspector(self) -> None:
        if self._inspector_render is not None:
            self.inspector.setHtml(self._inspector_render(self.theme))

    def _clear_selection_state(self) -> None:
        self._selected_node_id = None
        self._edge_selected = False
        self._sync_list_selection(None)
        self.canvas.emphasize(set())
        self._show_empty_inspector()
        self._update_actions()

    def _update_actions(self) -> None:
        has_node = self._selected_node_id is not None
        has_edge = bool(self.canvas.selected_relationships())
        self.act_edit_node.setEnabled(has_node)
        self.act_delete.setEnabled(has_node or has_edge)
        self.act_add_edge.setEnabled(len(self._all_nodes) >= 2)

    def _ensure_in_combos(self, node: Node) -> None:
        for combo in (self.path_source, self.path_target):
            if combo.findData(node.id) < 0:
                combo.addItem(render.node_labels([node])[node.id], node.id)

    def _inspect_node(self, node: Node) -> None:
        self._selected_node_id = node.id
        self._edge_selected = False
        self._ensure_in_combos(node)
        self._sync_list_selection(node.id)
        self._update_actions()
        self._set_inspector(lambda theme: render.message(theme, f"Loading {node.name}…"))
        self.statusBar().showMessage(f"Inspecting {node.name}…")
        node_id = node.id
        database = self.database_path

        def analyze():
            with GraphEngine(database) as graph:
                selected = graph.get_node(node_id)
                nodes = graph.all_nodes()
                edges = graph.all_edges()
                by_id = {item.id: item for item in nodes}
                incoming = tuple(
                    render.Link("in", e.relation, e.source_id, by_id[e.source_id].name, e.id)
                    for e in edges
                    if e.target_id == node_id
                )
                outgoing = tuple(
                    render.Link("out", e.relation, e.target_id, by_id[e.target_id].name, e.id)
                    for e in edges
                    if e.source_id == node_id
                )
                adjacency: dict[str, set[str]] = {item.id: set() for item in nodes}
                for edge in edges:
                    adjacency[edge.source_id].add(edge.target_id)
                    adjacency[edge.target_id].add(edge.source_id)
                mine = adjacency[node_id]
                similar = []
                for item in nodes:
                    if item.id == node_id:
                        continue
                    union = mine | adjacency[item.id]
                    score = len(mine & adjacency[item.id]) / len(union) if union else 0.0
                    if score > 0:
                        similar.append((score, item.name, item.id))
                similar.sort(key=lambda row: (-row[0], row[1].casefold()))
                clusters = IntelligenceEngine(graph).cluster_concepts()
                cluster = next(
                    (
                        (index, len(result.nodes))
                        for index, result in enumerate(clusters, 1)
                        if node_id in result.node_ids
                    ),
                    None,
                )
                return selected, incoming, outgoing, tuple(similar[:5]), cluster

        def show(result) -> None:
            if self._selected_node_id != node_id:
                return
            selected, incoming, outgoing, similar, cluster = result
            category = node_category(selected)
            details = render.NodeDetails(
                selected,
                category,
                self._category_colors.get(category, self.theme.node_neutral),
                incoming,
                outgoing,
                similar,
                cluster,
            )
            self._set_inspector(lambda theme: render.node_html(theme, details))
            links = incoming + outgoing
            self.canvas.emphasize({link.node_id for link in links} | {node_id}, {link.edge_id for link in links})
            self.statusBar().showMessage("Ready")

        def failed(message: str) -> None:
            if self._selected_node_id == node_id:
                self._set_inspector(lambda theme: render.message(theme, "Could not load details", message))
                self.statusBar().showMessage("Operation failed")

        self.tasks.submit(analyze, show, failed)

    def _inspect_edge(self, edge: Edge) -> None:
        self._selected_node_id = None
        self._edge_selected = True
        self._sync_list_selection(None)
        try:
            source = self.graph.get_node(edge.source_id)
            target = self.graph.get_node(edge.target_id)
        except GraphError as exc:
            self._set_inspector(lambda theme: render.message(theme, "Could not load details", str(exc)))
            return
        self._set_inspector(lambda theme: render.edge_html(theme, edge, source, target))
        self.canvas.emphasize({source.id, target.id}, {edge.id})
        self._update_actions()

    def _on_anchor(self, url: QUrl) -> None:
        kind = url.host()
        target = unquote(url.path()[1:])
        if kind == "node":
            self._select_node(target)
        elif kind == "edit":
            node = self._nodes_by_id.get(target)
            if node is not None:
                self._edit_node(node)
        elif kind in ("pathfrom", "pathto"):
            self._set_path_endpoint(self.path_source if kind == "pathfrom" else self.path_target, target)
        elif kind == "path" and target.isdigit() and int(target) < len(self._last_paths):
            path = self._last_paths[int(target)]
            self.canvas.emphasize({n.id for n in path.nodes}, {e.id for e in path.edges})
        elif kind == "cluster" and target.isdigit() and int(target) < len(self._last_clusters):
            self.canvas.emphasize({n.id for n in self._last_clusters[int(target)].nodes})

    def _set_path_endpoint(self, combo: QComboBox, node_id: str) -> None:
        node = self._nodes_by_id.get(node_id)
        if node is None:
            return
        self._ensure_in_combos(node)
        combo.setCurrentIndex(combo.findData(node_id))
        self.tabs.setCurrentWidget(self.analysis_panel)

    def _show_context_menu(self, node: Node | None, edge: Edge | None, position) -> None:
        menu = QMenu(self)
        if node is not None:
            self._select_node(node.id, center=False)
            menu.addAction(self.act_edit_node)
            link = menu.addAction("Add relationship from here…")
            link.triggered.connect(lambda _c=False: self._add_edge(source_id=node.id))
            menu.addSeparator()
            start = menu.addAction("Find paths from here")
            start.triggered.connect(lambda _c=False: self._set_path_endpoint(self.path_source, node.id))
            end = menu.addAction("Find paths to here")
            end.triggered.connect(lambda _c=False: self._set_path_endpoint(self.path_target, node.id))
            menu.addSeparator()
            remove = menu.addAction("Delete concept…")
            remove.triggered.connect(lambda _c=False: self._delete_selected_node())
        elif edge is not None:
            self.canvas.scene_model.clearSelection()
            for item in self.canvas.edge_items:
                if item.edge.id == edge.id:
                    item.setSelected(True)
            remove = menu.addAction("Remove relationship…")
            remove.triggered.connect(lambda _c=False: self._delete_selected_edge())
        else:
            menu.addAction(self.act_add_node)
            menu.addAction(self.act_fit)
        menu.exec(position)

    # --- editing ----------------------------------------------------------------------

    def _selected_node(self) -> Node | None:
        node_id = self._selected_node_id
        if not node_id:
            QMessageBox.information(self, "Select a concept", "Select a concept in the graph first.")
            return None
        try:
            return self.graph.get_node(node_id)
        except GraphError as exc:
            self._show_error(exc)
            return None

    def _category_names(self) -> list[str]:
        return sorted({node_category(node) for node in self._all_nodes})

    def _add_node(self) -> None:
        dialog = NodeDialog(self, categories=self._category_names())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                name, properties = dialog.values()
                node = self.graph.add_node(name, properties)
                self._pending_selection = node.id
                self._refresh_graph()
            except GraphError as exc:
                self._show_error(exc)

    def _edit_selected_node(self) -> None:
        node = self._selected_node()
        if node is not None:
            self._edit_node(node)

    def _edit_node(self, node: Node) -> None:
        dialog = NodeDialog(self, node, categories=self._category_names())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                name, properties = dialog.values()
                self.graph.update_node(node.id, name=name, properties=properties)
                self._pending_selection = node.id
                self._refresh_graph()
            except GraphError as exc:
                self._show_error(exc)

    def _delete_selection(self) -> None:
        if self._selected_node_id is None and self.canvas.selected_relationships():
            self._delete_selected_edge()
        else:
            self._delete_selected_node()

    def _delete_selected_node(self) -> None:
        node = self._selected_node()
        if node is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete concept",
            f"Delete “{node.name}” and its connected relationships?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.graph.delete_node(node.id)
                self._clear_selection_state()
                self._refresh_graph()
            except GraphError as exc:
                self._show_error(exc)

    def _add_edge(self, source_id: str | None = None) -> None:
        nodes = self.graph.all_nodes()
        if len(nodes) < 2:
            QMessageBox.information(self, "Add concepts first", "At least two concepts are required.")
            return
        if source_id is None:
            source_id = self._selected_node_id
        relations = sorted({edge.relation for edge in self._all_edges})
        dialog = EdgeDialog(nodes, self, source_id=source_id, relations=relations)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                properties = json.loads(dialog.properties.toPlainText() or "{}")
                self.graph.add_edge(
                    dialog.source.currentData(),
                    dialog.target.currentData(),
                    dialog.relation.text().strip(),
                    properties,
                )
                self._refresh_graph()
            except (GraphError, ValueError) as exc:
                self._show_error(exc)

    def _delete_selected_edge(self) -> None:
        selected = self.canvas.selected_relationships()
        if not selected:
            QMessageBox.information(self, "Select a relationship", "Select a relationship in the graph first.")
            return
        edge = selected[0].edge
        try:
            answer = QMessageBox.question(
                self,
                "Remove relationship",
                f"Remove “{edge.relation}” between {self.graph.get_node(edge.source_id).name} and "
                f"{self.graph.get_node(edge.target_id).name}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.graph.delete_edge(edge.id)
                self._clear_selection_state()
                self._refresh_graph()
        except GraphError as exc:
            self._show_error(exc)

    def _import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import graph", "", "JSON files (*.json)")
        if path:
            try:
                nodes, edges = self.graph.import_json(path)
                self._refresh_graph()
                self.statusBar().showMessage(f"Imported {nodes} concepts and {edges} relationships")
            except (GraphError, OSError) as exc:
                self._show_error(exc)

    def _export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export graph", "neurograph.json", "JSON files (*.json)")
        if path:
            try:
                self.graph.export_json(path)
                self.statusBar().showMessage(f"Exported graph to {path}")
            except (GraphError, OSError) as exc:
                self._show_error(exc)

    # --- analysis ---------------------------------------------------------------------

    def _set_results(self, block: Callable[[Theme], str]) -> None:
        self._result_blocks = [block]
        self._render_results()

    def _append_results(self, block: Callable[[Theme], str]) -> None:
        self._result_blocks.append(block)
        self._render_results()

    def _render_results(self) -> None:
        self.intelligence_results.setHtml(
            render.results_document(self.theme, [block(self.theme) for block in self._result_blocks])
        )

    def _path_endpoints(self) -> tuple[str, str] | None:
        source = combo_node_id(self.path_source)
        target = combo_node_id(self.path_target)
        if not source or not target:
            self.statusBar().showMessage("Choose both concepts from the lists first")
            self.tabs.setCurrentWidget(self.analysis_panel)
            return None
        return source, target

    def _find_paths(self) -> None:
        endpoints = self._path_endpoints()
        if endpoints is None:
            return
        source, target = endpoints
        database = self.database_path

        def find():
            with GraphEngine(database) as graph:
                return IntelligenceEngine(graph).find_paths(source, target, limit=5)

        self._run_intelligence(find, self._show_paths)

    def _show_paths(self, paths: list[Any]) -> None:
        self._last_paths = list(paths)
        self._set_results(lambda theme: render.paths_block(theme, paths))
        self.tabs.setCurrentWidget(self.analysis_panel)
        if paths:
            first = paths[0]
            self.canvas.emphasize({n.id for n in first.nodes}, {e.id for e in first.edges})
        else:
            self.canvas.emphasize(set())

    def _semantic_search(self) -> None:
        query = self.semantic_query.text().strip()
        if not query:
            self.statusBar().showMessage("Enter a phrase to search by meaning")
            return
        database = self.database_path

        def search():
            with GraphEngine(database) as graph:
                return IntelligenceEngine(graph, ollama=self.ollama_client).semantic_search(query)

        self._run_intelligence(
            search, lambda matches: self._set_results(lambda theme: render.matches_block(theme, matches))
        )

    def _cluster_graph(self) -> None:
        database = self.database_path

        def cluster():
            with GraphEngine(database) as graph:
                return IntelligenceEngine(graph).cluster_concepts()

        self._run_intelligence(cluster, self._show_clusters)

    def _show_clusters(self, clusters: list[Any]) -> None:
        self._last_clusters = list(clusters)
        self._set_results(lambda theme: render.clusters_block(theme, clusters))
        self.tabs.setCurrentWidget(self.analysis_panel)

    def _infer_links(self) -> None:
        database = self.database_path

        def infer():
            with GraphEngine(database) as graph:
                return IntelligenceEngine(graph).infer_transitive()

        def show(suggestions) -> None:
            names = {node.id: node.name for node in self._all_nodes}
            self._set_results(lambda theme: render.inferences_block(theme, suggestions, names))
            self.tabs.setCurrentWidget(self.analysis_panel)

        self._run_intelligence(infer, show)

    def _explain_first_path(self) -> None:
        endpoints = self._path_endpoints()
        if endpoints is None:
            return
        source, target = endpoints
        database = self.database_path

        def explain():
            with GraphEngine(database) as graph:
                intelligence = IntelligenceEngine(graph, ollama=self.ollama_client)
                paths = intelligence.find_paths(source, target, limit=1)
                if not paths:
                    return "No confirmed path to explain."
                return intelligence.explain_with_ollama(paths[0])

        self._run_chat(
            explain,
            lambda response: self._append_results(
                lambda theme: render.text_block(theme, "Ollama explanation", response)
            ),
        )

    def _summarize_first_cluster(self) -> None:
        database = self.database_path

        def summarize():
            with GraphEngine(database) as graph:
                intelligence = IntelligenceEngine(graph, ollama=self.ollama_client)
                clusters = intelligence.cluster_concepts()
                if not clusters:
                    return "The graph has no concepts to summarize."
                return intelligence.summarize_cluster(clusters[0])

        self._run_chat(
            summarize,
            lambda response: self._append_results(
                lambda theme: render.text_block(theme, "Ollama cluster summary", response)
            ),
        )

    def _run_chat(self, function: Callable[[], Any], success: Callable[[Any], None]) -> None:
        if self._chat_busy:
            self.statusBar().showMessage("An Ollama chat request is already running")
            return
        self._chat_busy = True
        self.explain_button.setEnabled(False)
        self.summarize_button.setEnabled(False)
        self.statusBar().showMessage("Waiting for Ollama response…")

        def finish() -> None:
            self._chat_busy = False
            self.explain_button.setEnabled(True)
            self.summarize_button.setEnabled(True)

        def done(result: Any) -> None:
            finish()
            success(result)
            self.statusBar().showMessage("Ready")

        def failed(message: str) -> None:
            finish()
            self._show_background_error(message)

        self.tasks.submit(function, done, failed)

    def _run_intelligence(self, function: Callable[[], Any], success: Callable[[Any], None]) -> None:
        self.statusBar().showMessage("Working…")

        def done(result: Any) -> None:
            success(result)
            self.statusBar().showMessage("Ready")

        self.tasks.submit(function, done, self._show_background_error)

    # --- misc -------------------------------------------------------------------------

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About NeuroGraph",
            "NeuroGraph\n\nA local-first knowledge graph for organising concepts and "
            f"exploring how they relate.\n\nDatabase: {self.database_path}",
        )

    def _show_background_error(self, message: str) -> None:
        status = (
            "Ollama timed out; increase its response timeout"
            if "timed out" in message.casefold()
            else "Operation failed"
        )
        self.statusBar().showMessage(status)
        QMessageBox.warning(self, "NeuroGraph", message)

    def _show_error(self, error: Exception) -> None:
        QMessageBox.warning(self, "NeuroGraph", str(error))

    def closeEvent(self, event) -> None:
        self._save_window_state()
        self.tasks.close()
        self.graph.close()
        super().closeEvent(event)
