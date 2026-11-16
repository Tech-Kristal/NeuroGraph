"""Dialogs for creating and editing concepts and relationships."""

from __future__ import annotations

import json
from typing import Any, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
)

from neurograph.errors import ValidationError
from neurograph.models import Node
from neurograph.render import node_labels
from neurograph.widgets import combo_node_id, create_node_combo, field_label, monospace_font


def _parse_properties(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text.strip() or "{}")
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Properties are not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError("Properties must be a JSON object, for example {\"year\": 2024}.")
    return value


class NodeDialog(QDialog):
    """Create or edit a concept: name, category and free-form JSON properties."""

    def __init__(self, parent=None, node: Node | None = None, categories: Sequence[str] = ()) -> None:
        super().__init__(parent)
        self._init_ui(node, categories)

    def _init_ui(self, node: Node | None, categories: Sequence[str]) -> None:
        self.setWindowTitle("Edit concept" if node else "New concept")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(6)

        existing_type = node.properties.get("type") if node else None
        self._type_is_category = isinstance(existing_type, str)
        self.name = QLineEdit(node.name if node else "")
        self.name.setPlaceholderText("Name of the concept")
        self.category = QComboBox()
        self.category.setEditable(True)
        self.category.addItems(sorted({c for c in categories if c != "concept"}, key=str.casefold))
        self.category.setCurrentText(existing_type if self._type_is_category else "")
        line = self.category.lineEdit()
        if line is not None:
            line.setPlaceholderText("Optional, e.g. person, topic, tool")
        remaining = {k: v for k, v in (node.properties if node else {}).items() if not (k == "type" and self._type_is_category)}
        self.properties = QPlainTextEdit(json.dumps(remaining, indent=2, ensure_ascii=False) if remaining else "")
        self.properties.setFont(monospace_font())
        self.properties.setPlaceholderText('{\n  "key": "value"\n}')
        self.properties.setMinimumHeight(150)

        for label, widget in (
            ("Name", self.name),
            ("Category", self.category),
            ("Other properties (JSON)", self.properties),
        ):
            layout.addSpacing(6)
            layout.addWidget(field_label(label))
            layout.addWidget(widget)

        self.error_label = QLabel()
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addSpacing(4)
        layout.addWidget(self.error_label)
        layout.addSpacing(8)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setText("Save changes" if node else "Add concept")
        save.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.name.setFocus()

    def values(self) -> tuple[str, dict[str, Any]]:
        properties = _parse_properties(self.properties.toPlainText())
        category = self.category.currentText().strip()
        if category:
            properties["type"] = category
        elif self._type_is_category:
            properties.pop("type", None)
        return self.name.text(), properties

    def accept(self) -> None:
        try:
            name, _ = self.values()
            if not name.strip():
                raise ValidationError("Give the concept a name.")
        except ValidationError as exc:
            self.error_label.setText(str(exc))
            self.error_label.show()
            return
        super().accept()


class EdgeDialog(QDialog):
    """Create a directed relationship between two existing concepts."""

    def __init__(
        self,
        nodes: list[Node],
        parent=None,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        relations: Sequence[str] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New relationship")
        self.setMinimumWidth(460)
        labels = node_labels(nodes)
        ordered = sorted(nodes, key=lambda n: labels[n.id].casefold())
        self.source = create_node_combo(ordered, labels)
        self.target = create_node_combo(ordered, labels)
        for combo, wanted in ((self.source, source_id), (self.target, target_id)):
            if wanted is not None and combo.findData(wanted) >= 0:
                combo.setCurrentIndex(combo.findData(wanted))
        self.relation = QLineEdit()
        self.relation.setPlaceholderText("e.g. is_a, part_of, depends_on")
        if relations:
            completer = QCompleter(sorted(set(relations), key=str.casefold), self.relation)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            self.relation.setCompleter(completer)
        self.properties = QPlainTextEdit()
        self.properties.setFont(monospace_font())
        self.properties.setPlaceholderText('{\n  "weight": 1\n}')
        self.properties.setMinimumHeight(90)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(6)
        for label, widget in (
            ("From", self.source),
            ("Relationship", self.relation),
            ("To", self.target),
            ("Properties (JSON, optional)", self.properties),
        ):
            layout.addSpacing(6)
            layout.addWidget(field_label(label))
            layout.addWidget(widget)
        self.error_label = QLabel()
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addSpacing(4)
        layout.addWidget(self.error_label)
        layout.addSpacing(8)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setText("Add relationship")
        save.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        (self.source if source_id is None else self.relation).setFocus()

    def accept(self) -> None:
        try:
            for combo, label in ((self.source, "source"), (self.target, "target")):
                node_id = combo_node_id(combo)
                if node_id is None:
                    raise ValidationError(f"Choose the {label} concept from the list.")
                combo.setCurrentIndex(combo.findData(node_id))
            if not self.relation.text().strip():
                raise ValidationError("Give the relationship a label.")
            _parse_properties(self.properties.toPlainText())
        except ValidationError as exc:
            self.error_label.setText(str(exc))
            self.error_label.show()
            return
        super().accept()
