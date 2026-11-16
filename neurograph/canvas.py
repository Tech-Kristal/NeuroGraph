"""Interactive graph canvas built on Qt's Graphics View framework."""

from __future__ import annotations

from collections import Counter
import math
from typing import Mapping

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView

from neurograph.layout import curve_offsets, node_radius
from neurograph.models import Edge, Node
from neurograph.theme import Theme

# Labels appear once the view is zoomed in this far (or when a node is active).
_NODE_LABEL_ZOOM = 0.55
_EDGE_LABEL_ZOOM = 0.9
_DIMMED_NODE = 0.18
_DIMMED_EDGE = 0.1


def node_category(node: Node) -> str:
    return str(node.properties.get("type", "concept"))


def _elide(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class CanvasStyle:
    """Colours shared by every item, so a theme switch repaints without rebuilding."""

    def __init__(self, theme: Theme) -> None:
        self.always_labels = False
        self.apply(theme)

    def apply(self, theme: Theme) -> None:
        self.theme = theme
        self.canvas = QColor(theme.canvas)
        self.text = QColor(theme.text)
        self.muted = QColor(theme.text_muted)
        self.accent = QColor(theme.accent)
        self.edge = QColor(theme.edge)
        self.edge_label = QColor(theme.edge_label)
        self.label_background = QColor(theme.canvas)
        self.label_background.setAlpha(215)


def _label_font() -> QFont:
    font = QFont()
    font.setPointSizeF(8.5)
    return font


class GraphNodeVisual(QGraphicsItem):
    """A draggable, selectable concept drawn as a filled circle with a label."""

    def __init__(self, node: Node, color: str, degree: int, style: CanvasStyle) -> None:
        super().__init__()
        self.node = node
        self.category = node_category(node)
        self.color = QColor(color)
        self.degree = degree
        self.radius = node_radius(degree)
        self.canvas_style = style
        self.relationships: list[RelationshipVisual] = []
        self.emphasis: bool | None = None
        self._hover = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        self._font = _label_font()
        metrics = QFontMetricsF(self._font)
        self._text = _elide(node.name, 30)
        self._ascent = metrics.ascent()
        width = metrics.horizontalAdvance(self._text) + 10
        self._label_rect = QRectF(-width / 2, self.radius + 6, width, metrics.height() + 4)
        noun = "connection" if degree == 1 else "connections"
        self.setToolTip(f"{node.name}\n{self.category} · {degree} {noun}")

    def boundingRect(self) -> QRectF:
        reach = self.radius + 6
        ring = QRectF(-reach, -reach, 2 * reach, 2 * reach)
        return ring.united(self._label_rect).adjusted(-1, -1, 1, 1)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addEllipse(QPointF(0, 0), self.radius + 2, self.radius + 2)
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:
        del option, widget
        style = self.canvas_style
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        transform = painter.worldTransform()
        zoom = math.hypot(transform.m11(), transform.m12())
        selected = self.isSelected()
        if selected or self._hover:
            painter.setPen(QPen(style.accent if selected else style.muted, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), self.radius + 4, self.radius + 4)
        painter.setPen(QPen(style.canvas, 2.0))
        painter.setBrush(QBrush(self.color))
        painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)
        if selected or self._hover or self.emphasis or style.always_labels or zoom >= _NODE_LABEL_ZOOM:
            rect = self._label_rect
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(style.label_background))
            painter.drawRoundedRect(rect, 3, 3)
            painter.setFont(self._font)
            painter.setPen(QPen(style.text))
            painter.drawText(QPointF(rect.left() + 5, rect.top() + 2 + self._ascent), self._text)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for relationship in self.relationships:
                relationship.update_positions()
        return result

    def hoverEnterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)


class RelationshipVisual(QGraphicsItem):
    """A labelled directed edge, drawn straight or as a gentle curve."""

    def __init__(
        self,
        edge: Edge,
        source: GraphNodeVisual,
        target: GraphNodeVisual,
        offset: float,
        style: CanvasStyle,
    ) -> None:
        super().__init__()
        self.edge = edge
        self.source = source
        self.target = target
        self.offset = offset
        self.canvas_style = style
        self.emphasis: bool | None = None
        self._hover = False
        self._start = QPointF()
        self._end = QPointF()
        self._path = QPainterPath()
        self._shape = QPainterPath()
        self._arrow = QPolygonF()
        self._label_rect = QRectF()
        self._font = _label_font()
        self._metrics = QFontMetricsF(self._font)
        self._text = _elide(edge.relation, 28)
        self.setZValue(0)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setToolTip(edge.relation)
        source.relationships.append(self)
        if source is not target:
            target.relationships.append(self)
        self.update_positions()

    @staticmethod
    def _toward(origin: QPointF, point: QPointF, radius: float) -> QPointF:
        dx, dy = point.x() - origin.x(), point.y() - origin.y()
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return QPointF(origin)
        return QPointF(origin.x() + dx / length * radius, origin.y() + dy / length * radius)

    def update_positions(self) -> None:
        self.prepareGeometryChange()
        a = self.source.scenePos()
        b = self.target.scenePos()
        path = QPainterPath()
        arrow = QPolygonF()
        if self.source is self.target:
            radius = self.source.radius
            centre = QPointF(a.x(), a.y() - radius - 13)
            path.addEllipse(centre, 14, 14)
            self._start = self._end = QPointF(a)
            label_point = QPointF(centre.x(), centre.y() - 24)
        else:
            low, high = (
                (self.source, self.target)
                if self.source.node.id <= self.target.node.id
                else (self.target, self.source)
            )
            vx = high.scenePos().x() - low.scenePos().x()
            vy = high.scenePos().y() - low.scenePos().y()
            length = math.hypot(vx, vy) or 1.0
            normal = QPointF(-vy / length, vx / length)
            control = QPointF(
                (a.x() + b.x()) / 2 + normal.x() * 2 * self.offset,
                (a.y() + b.y()) / 2 + normal.y() * 2 * self.offset,
            )
            self._start = self._toward(a, control, self.source.radius)
            self._end = self._toward(b, control, self.target.radius)
            path.moveTo(self._start)
            path.quadTo(control, self._end)
            label_point = path.pointAtPercent(0.5)
            angle = math.atan2(self._end.y() - control.y(), self._end.x() - control.x())
            size = 8.0
            arrow = QPolygonF(
                [
                    self._end,
                    QPointF(
                        self._end.x() - size * math.cos(angle - math.pi / 7),
                        self._end.y() - size * math.sin(angle - math.pi / 7),
                    ),
                    QPointF(
                        self._end.x() - size * math.cos(angle + math.pi / 7),
                        self._end.y() - size * math.sin(angle + math.pi / 7),
                    ),
                ]
            )
        width = self._metrics.horizontalAdvance(self._text) + 10
        height = self._metrics.height() + 2
        self._label_rect = QRectF(
            label_point.x() - width / 2, label_point.y() - height / 2, width, height
        )
        self._path = path
        self._arrow = arrow
        stroker = QPainterPathStroker()
        stroker.setWidth(10)
        self._shape = stroker.createStroke(path)
        self.update()

    def boundingRect(self) -> QRectF:
        return self._path.boundingRect().united(self._label_rect).adjusted(-12, -12, 12, 12)

    def shape(self) -> QPainterPath:
        return self._shape

    def paint(self, painter: QPainter, option, widget=None) -> None:
        del option, widget
        style = self.canvas_style
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        zoom = math.hypot(painter.worldTransform().m11(), painter.worldTransform().m12())
        selected = self.isSelected()
        active = selected or self._hover or bool(self.emphasis)
        color = style.accent if (selected or self.emphasis) else (style.muted if self._hover else style.edge)
        painter.setPen(QPen(color, 2.4 if selected else (1.9 if active else 1.2)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self._path)
        if not self._arrow.isEmpty():
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPolygon(self._arrow)
        if active or zoom >= _EDGE_LABEL_ZOOM:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(style.label_background))
            painter.drawRoundedRect(self._label_rect, 3, 3)
            painter.setFont(self._font)
            painter.setPen(QPen(style.accent if active else style.edge_label))
            painter.drawText(
                QPointF(self._label_rect.left() + 5, self._label_rect.top() + 1 + self._metrics.ascent()),
                self._text,
            )

    def hoverEnterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)


class GraphCanvas(QGraphicsView):
    """Pannable, zoomable graph view that reports user selection through signals."""

    node_selected = pyqtSignal(object)
    edge_selected = pyqtSignal(object)
    selection_cleared = pyqtSignal()
    node_activated = pyqtSignal(object)
    context_requested = pyqtSignal(object, object, object)

    def __init__(self, theme: Theme) -> None:
        scene = QGraphicsScene()
        super().__init__(scene)
        self.scene_model = scene
        self.style_model = CanvasStyle(theme)
        self.node_items: dict[str, GraphNodeVisual] = {}
        self.edge_items: list[RelationshipVisual] = []
        self._silent = False
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setMinimumSize(420, 360)
        self.setBackgroundBrush(QBrush(QColor(theme.canvas)))
        scene.selectionChanged.connect(self._on_selection_changed)

    # --- appearance -------------------------------------------------------

    def apply_theme(self, theme: Theme) -> None:
        self.style_model.apply(theme)
        self.setBackgroundBrush(QBrush(QColor(theme.canvas)))
        self.scene_model.update()
        self.viewport().update()

    def set_always_labels(self, enabled: bool) -> None:
        self.style_model.always_labels = enabled
        self.scene_model.update()

    # --- content ----------------------------------------------------------

    def display_graph(
        self,
        nodes: list[Node],
        edges: list[Edge],
        positions: Mapping[str, tuple[float, float]],
        colors: Mapping[str, str],
    ) -> None:
        self._silent = True
        try:
            self.scene_model.clear()
            self.node_items = {}
            self.edge_items = []
            degree: Counter[str] = Counter()
            for edge in edges:
                degree[edge.source_id] += 1
                degree[edge.target_id] += 1
            neutral = self.style_model.theme.node_neutral
            for node in nodes:
                item = GraphNodeVisual(
                    node, colors.get(node_category(node), neutral), degree[node.id], self.style_model
                )
                self.scene_model.addItem(item)
                item.setPos(*positions.get(node.id, (0.0, 0.0)))
                self.node_items[node.id] = item
            offsets = curve_offsets(edges)
            for edge in edges:
                source = self.node_items.get(edge.source_id)
                target = self.node_items.get(edge.target_id)
                if source is None or target is None:
                    continue
                visual = RelationshipVisual(edge, source, target, offsets.get(edge.id, 0.0), self.style_model)
                self.scene_model.addItem(visual)
                self.edge_items.append(visual)
            bounds = self.scene_model.itemsBoundingRect()
            self.scene_model.setSceneRect(bounds.adjusted(-500, -400, 500, 400))
        finally:
            self._silent = False
        self.fit_all()
        self.viewport().update()

    def positions(self) -> dict[str, tuple[float, float]]:
        return {node_id: (item.x(), item.y()) for node_id, item in self.node_items.items()}

    def selected_relationships(self) -> list[RelationshipVisual]:
        return [item for item in self.scene_model.selectedItems() if isinstance(item, RelationshipVisual)]

    # --- selection and emphasis -------------------------------------------

    def select_node(self, node_id: str, *, center: bool = True) -> bool:
        item = self.node_items.get(node_id)
        if item is None:
            return False
        self._silent = True
        try:
            self.scene_model.clearSelection()
            item.setSelected(True)
        finally:
            self._silent = False
        if center:
            self.centerOn(item)
        return True

    def clear_selection(self) -> None:
        self._silent = True
        try:
            self.scene_model.clearSelection()
        finally:
            self._silent = False

    def emphasize(self, node_ids: set[str], edge_ids: set[str] | None = None) -> None:
        """Dim everything except the given nodes and edges (empty sets reset)."""
        edge_ids = edge_ids or set()
        focus = bool(node_ids or edge_ids)
        for node_id, item in self.node_items.items():
            on = node_id in node_ids
            item.emphasis = True if (focus and on) else None
            item.setOpacity(1.0 if (not focus or on) else _DIMMED_NODE)
        for item in self.edge_items:
            on = item.edge.id in edge_ids
            item.emphasis = True if (focus and on) else None
            item.setOpacity(1.0 if (not focus or on) else _DIMMED_EDGE)
        self.scene_model.update()

    def _on_selection_changed(self) -> None:
        if self._silent:
            return
        try:
            items = self.scene_model.selectedItems()
        except RuntimeError:  # scene already destroyed while closing
            return
        nodes = [item for item in items if isinstance(item, GraphNodeVisual)]
        edges = [item for item in items if isinstance(item, RelationshipVisual)]
        if nodes:
            self.node_selected.emit(nodes[0].node)
        elif edges:
            self.edge_selected.emit(edges[0].edge)
        else:
            self.selection_cleared.emit()

    # --- view control -----------------------------------------------------

    def fit_all(self) -> None:
        if not self.node_items:
            self.resetTransform()
            return
        rect = self.scene_model.itemsBoundingRect().adjusted(-70, -70, 70, 70)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        current = self.transform().m11()
        if current > 1.3:  # do not blow a handful of nodes up to fill the screen
            self.scale(1.3 / current, 1.3 / current)

    def zoom(self, factor: float) -> None:
        current = self.transform().m11() or 1.0
        target = min(6.0, max(0.04, current * factor))
        self.scale(target / current, target / current)

    def wheelEvent(self, event) -> None:
        self.zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)

    def mouseDoubleClickEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, GraphNodeVisual):
            self.node_activated.emit(item.node)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event) -> None:
        item = self.itemAt(event.pos())
        node = item.node if isinstance(item, GraphNodeVisual) else None
        edge = item.edge if isinstance(item, RelationshipVisual) else None
        self.context_requested.emit(node, edge, event.globalPos())

    def drawForeground(self, painter: QPainter, rect) -> None:
        super().drawForeground(painter, rect)
        if self.node_items:
            return
        painter.save()
        painter.resetTransform()
        theme = self.style_model.theme
        area = self.viewport().rect()
        title_font = QFont()
        title_font.setPointSizeF(13)
        title_font.setWeight(QFont.Weight.DemiBold)
        hint_font = QFont()
        hint_font.setPointSizeF(9.5)
        lines = (
            (title_font, "No concepts to show", theme.text),
            (hint_font, "Add a concept with Ctrl+N, or import a graph with Ctrl+O.", theme.text_muted),
        )
        y = area.center().y() - 14
        for font, text, color in lines:
            metrics = QFontMetricsF(font)
            painter.setFont(font)
            painter.setPen(QPen(QColor(color)))
            painter.drawText(QPointF(area.center().x() - metrics.horizontalAdvance(text) / 2, y), text)
            y += metrics.height() + 6
        painter.restore()
