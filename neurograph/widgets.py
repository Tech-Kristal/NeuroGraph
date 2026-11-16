"""Small reusable Qt helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontDatabase, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QComboBox, QCompleter, QFrame, QLabel

from neurograph.models import Node

_icon_cache: dict[tuple[str, int], QIcon] = {}


def monospace_font() -> QFont:
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)


def color_icon(color: str, size: int = 14) -> QIcon:
    """A small filled circle used as a category marker."""
    key = (color, size)
    if key not in _icon_cache:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(color)))
        painter.drawEllipse(2, 2, size - 4, size - 4)
        painter.end()
        _icon_cache[key] = QIcon(pixmap)
    return _icon_cache[key]


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("sectionLabel")
    return label


def field_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("mutedLabel")
    return label


def hairline() -> QFrame:
    line = QFrame()
    line.setObjectName("hairline")
    line.setFrameShape(QFrame.Shape.NoFrame)
    return line


def create_node_combo(nodes: Sequence[Node], labels: Mapping[str, str]) -> QComboBox:
    """An editable, type-to-filter combo box whose item data is the node ID."""
    combo = QComboBox()
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    for node in nodes:
        combo.addItem(labels.get(node.id, node.name), node.id)
    completer = combo.completer()
    if completer is not None:
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    combo.setCurrentIndex(-1)
    return combo


def combo_node_id(combo: QComboBox) -> str | None:
    """Resolve the concept typed or picked in a node combo, or ``None``."""
    index = combo.findText(combo.currentText().strip())
    return combo.itemData(index) if index >= 0 else None
