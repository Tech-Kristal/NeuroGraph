"""Colour themes and style sheet generation.

This module deliberately has no Qt imports so it can be tested anywhere. The
GUI turns a :class:`Theme` into a ``QPalette`` and applies :func:`build_stylesheet`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from string import Template

# A muted categorical palette (Tableau 10). Chosen for distinguishability on both
# light and dark backgrounds rather than for decoration.
CATEGORY_COLORS: tuple[str, ...] = (
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#76b7b2",
    "#edc948",
    "#9c755f",
    "#ff9da7",
    "#bab0ac",
)


@dataclass(frozen=True)
class Theme:
    """Every colour used by the application, as ``#rrggbb`` strings."""

    name: str
    window: str
    panel: str
    canvas: str
    input: str
    button: str
    hover: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_disabled: str
    accent: str
    accent_hover: str
    accent_text: str
    accent_soft: str
    edge: str
    edge_label: str
    node_neutral: str
    danger: str

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


DARK = Theme(
    name="dark",
    window="#1f2023",
    panel="#26272b",
    canvas="#1a1b1e",
    input="#1b1c1f",
    button="#2e3035",
    hover="#33353b",
    border="#37393f",
    border_strong="#474a52",
    text="#d9dbe0",
    text_muted="#8f939d",
    text_disabled="#5f636c",
    accent="#5b8def",
    accent_hover="#7aa3f5",
    accent_text="#ffffff",
    accent_soft="#2c3e63",
    edge="#5b5e68",
    edge_label="#a2a6b0",
    node_neutral="#8c93a3",
    danger="#e5646b",
)

LIGHT = Theme(
    name="light",
    window="#f3f4f6",
    panel="#ffffff",
    canvas="#fafafb",
    input="#ffffff",
    button="#f3f4f6",
    hover="#eceef2",
    border="#dcdfe4",
    border_strong="#bcc1c9",
    text="#1f2328",
    text_muted="#667085",
    text_disabled="#a3a9b3",
    accent="#2f6fed",
    accent_hover="#255cc9",
    accent_text="#ffffff",
    accent_soft="#d9e5ff",
    edge="#a6acb7",
    edge_label="#6b7280",
    node_neutral="#8a93a6",
    danger="#d1383f",
)

THEMES: dict[str, Theme] = {"dark": DARK, "light": LIGHT}


def get_theme(name: str | None) -> Theme:
    """Return the named theme, falling back to dark for unknown names."""
    return THEMES.get((name or "").lower(), DARK)


def category_colors(categories: list[str], theme: Theme) -> dict[str, str]:
    """Assign stable colours to category names.

    The generic ``concept`` category (nodes without a ``type`` property) always
    gets the neutral colour so that typed nodes stand out.
    """
    colors: dict[str, str] = {}
    index = 0
    for category in sorted(categories, key=str.casefold):
        if category == "concept":
            colors[category] = theme.node_neutral
        else:
            colors[category] = CATEGORY_COLORS[index % len(CATEGORY_COLORS)]
            index += 1
    return colors


_STYLESHEET = Template(
    """
QMainWindow, QDialog { background: $window; }
QToolTip { background: $panel; color: $text; border: 1px solid $border_strong; padding: 4px 6px; }

QMenuBar { background: $panel; border-bottom: 1px solid $border; padding: 2px 4px; }
QMenuBar::item { background: transparent; padding: 4px 10px; border-radius: 4px; }
QMenuBar::item:selected { background: $hover; }
QMenu { background: $panel; border: 1px solid $border_strong; padding: 4px; }
QMenu::item { padding: 5px 28px 5px 14px; border-radius: 3px; }
QMenu::item:selected { background: $accent_soft; }
QMenu::item:disabled { color: $text_disabled; }
QMenu::separator { height: 1px; background: $border; margin: 4px 6px; }

QToolBar { background: $panel; border: none; border-bottom: 1px solid $border; padding: 4px 8px; spacing: 2px; }
QToolBar::separator { background: $border; width: 1px; margin: 5px 8px; }
QToolButton { background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 5px 11px; }
QToolButton:hover { background: $hover; }
QToolButton:pressed { background: $accent_soft; }
QToolButton:disabled { color: $text_disabled; }

QPushButton { background: $button; border: 1px solid $border_strong; border-radius: 4px; padding: 5px 14px; min-height: 18px; }
QPushButton:hover { background: $hover; }
QPushButton:pressed { background: $accent_soft; }
QPushButton:disabled { color: $text_disabled; border-color: $border; }
QPushButton:default { background: $accent; border-color: $accent; color: $accent_text; }
QPushButton:default:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton#primary { background: $accent; border-color: $accent; color: $accent_text; }
QPushButton#primary:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton#primary:disabled { background: $button; border-color: $border; color: $text_disabled; }

QLineEdit, QPlainTextEdit, QComboBox { background: $input; border: 1px solid $border_strong; border-radius: 4px; padding: 4px 8px; selection-background-color: $accent; selection-color: $accent_text; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus { border-color: $accent; }
QComboBox QAbstractItemView { background: $panel; border: 1px solid $border_strong; selection-background-color: $accent_soft; selection-color: $text; outline: 0; }

QTextBrowser { background: $panel; border: none; selection-background-color: $accent; selection-color: $accent_text; }

QListWidget { background: $panel; border: 1px solid $border; border-radius: 4px; outline: 0; padding: 2px; }
QListWidget::item { padding: 4px 6px; border-radius: 3px; }
QListWidget::item:hover { background: $hover; }
QListWidget::item:selected, QListWidget::item:selected:!active { background: $accent_soft; color: $text; }

QTabWidget::pane { border: none; border-top: 1px solid $border; background: $panel; }
QTabBar { background: $panel; }
QTabBar::tab { background: transparent; color: $text_muted; padding: 8px 16px; border: none; border-bottom: 2px solid transparent; margin-right: 2px; }
QTabBar::tab:hover { color: $text; }
QTabBar::tab:selected { color: $text; border-bottom: 2px solid $accent; }

QScrollBar:vertical { background: transparent; width: 11px; margin: 0; }
QScrollBar::handle:vertical { background: $border_strong; border-radius: 4px; min-height: 28px; margin: 1px 2px; }
QScrollBar::handle:vertical:hover { background: $text_muted; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; width: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 11px; margin: 0; }
QScrollBar::handle:horizontal { background: $border_strong; border-radius: 4px; min-width: 28px; margin: 2px 1px; }
QScrollBar::handle:horizontal:hover { background: $text_muted; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { height: 0; width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

QStatusBar { background: $panel; border-top: 1px solid $border; color: $text_muted; }
QStatusBar::item { border: none; }
QStatusBar QLabel { color: $text_muted; padding: 0 8px; }

QSplitter::handle { background: transparent; }
QSplitter::handle:hover { background: $accent_soft; }

QFrame#leftPanel { background: $panel; border: none; border-right: 1px solid $border; }
QFrame#rightPanel { background: $panel; border: none; border-left: 1px solid $border; }
QGraphicsView { border: none; background: $canvas; }

QLabel#panelTitle { font-weight: 600; }
QLabel#sectionLabel { color: $text_muted; font-weight: 600; }
QLabel#mutedLabel { color: $text_muted; }
QLabel#errorLabel { color: $danger; }
QFrame#hairline { background: $border; border: none; max-height: 1px; min-height: 1px; }
"""
)


def build_stylesheet(theme: Theme) -> str:
    """Render the application style sheet for ``theme``."""
    return _STYLESHEET.substitute(asdict(theme))
