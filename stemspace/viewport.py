from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QToolTip, QWidget

from .project import Placement, Project, Track
from .spatial import (
    ORIGIN, Camera, Point3, Projected, Projection, azimuth_of, direction, gain_radius,
    width_arc, wrap_azimuth,
)

INK = "#e8edf6"
MUTED = "#a3b2cb"
ACCENTS = ("#ffc785", "#ee8e75", "#ffb396", "#dba8fa", "#7addce", "#8eaaff",
           "#b4e9a3", "#ffe587", "#a2b9ff", "#f7a3cf", "#a5aebe")
HIT_RADIUS = 15
HEIGHT_HANDLE_HIT = 10
DRAG_THRESHOLD = 4
AXES = ((0, "FRONT"), (90, "RIGHT"), (180, "REAR"), (-90, "LEFT"))
AXIS_LABEL_RADIUS = 1.34


def screen(p: Projected) -> QPointF:
    return QPointF(p.x, p.y)


def separation(a: QPointF, b: QPointF) -> float:
    return hypot(a.x() - b.x(), a.y() - b.y())


def translucent(color: str | QColor, alpha: int) -> QColor:
    result = QColor(color)
    result.setAlpha(alpha)
    return result


def overlap_badge(point: QPointF) -> QRectF:
    return QRectF(point.x() + 10, point.y() + 7, 24, 15)


@dataclass
class Pointer:
    mode: str
    anchor: QPointF
    camera: Camera
    row: int = -1
    candidates: tuple[int, ...] = ()
    position: Placement | None = None
    offset: QPointF | None = None
    dragging: bool = False


class OrbitViewport(QWidget):
    """Common camera, perspective, and listener room for both 3D views."""

    def __init__(self):
        super().__init__()
        self.camera = Camera()

    def reset_view(self) -> None:
        self.camera = Camera()
        self.update()

    def zoom_view(self, steps: float) -> None:
        self.camera = self.camera.dolly(steps)
        self.update()

    def zoom_in(self) -> None:
        self.zoom_view(1)

    def zoom_out(self) -> None:
        self.zoom_view(-1)

    def projection(self) -> Projection:
        return Projection(self.camera, self.width(), self.height())

    def draw_background(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#0d1522"))
        gradient.setColorAt(1, QColor("#101b2b"))
        painter.fillRect(self.rect(), gradient)

    @staticmethod
    def path(projection: Projection, points: list[Point3]) -> QPolygonF:
        return QPolygonF([screen(projection.project(p)) for p in points])

    def draw_room(self, painter: QPainter, projection: Projection) -> None:
        painter.setBrush(QColor("#121e30"))
        painter.setPen(QPen(QColor("#354963"), 1.1))
        painter.drawPolygon(self.path(projection, width_arc(0, 0, 360, 1.10, steps=72)))

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#25374f"), 1))
        for radius in (0.36, 0.72):
            painter.drawPolyline(self.path(projection, width_arc(0, 0, 360, radius, steps=48)))
        center = screen(projection.project(ORIGIN))
        for azimuth in range(0, 360, 45):
            painter.drawLine(center, screen(projection.project(direction(azimuth, 0, 1.10))))

        grid = QPen(QColor("#31455e"), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid)
        for elevation in (30, 60):
            painter.drawPolyline(self.path(projection, width_arc(0, elevation, 360, steps=56)))
        for azimuth in (0, 90):
            dome = [direction(azimuth, el) for el in range(0, 91, 5)]
            dome += [direction(azimuth + 180, el) for el in range(85, -1, -5)]
            painter.drawPolyline(self.path(projection, dome))

        painter.setPen(QColor(MUTED))
        for azimuth, name in AXES:
            point = screen(projection.project(direction(azimuth, 0, AXIS_LABEL_RADIUS)))
            painter.drawText(QRectF(point.x() - 32, point.y() - 9, 64, 18),
                             Qt.AlignmentFlag.AlignCenter, name)

        # A small directional listener glyph sits on the ear-level plane.
        arrow = [Point3(-0.056, -0.044, 0), Point3(0, 0.112, 0), Point3(0.056, -0.044, 0)]
        painter.setPen(QPen(QColor("#142032"), 1))
        painter.setBrush(QColor("#dce6fb"))
        painter.drawPolygon(self.path(projection, arrow))

    def wheelEvent(self, event) -> None:
        pixel, angle = event.pixelDelta().y(), event.angleDelta().y()
        steps = pixel / 75 if pixel else angle / 120
        if steps:
            self.zoom_view(max(-3, min(3, steps)))
            event.accept()
        else:
            super().wheelEvent(event)


class ListenerViewport(OrbitViewport):
    """Draw and pick listener-relative 3D geometry using the same camera."""

    track_selected = Signal(int)
    azimuth_changed = Signal(float)
    elevation_changed = Signal(float)

    def __init__(self):
        super().__init__()
        self.project: Project | None = None
        self.selected = -1
        self.pointer: Pointer | None = None
        self.last_click: tuple[QPointF, tuple[int, ...], int] | None = None
        self.setMinimumSize(340, 320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setAccessibleName("3D listener scene")
        self.setAccessibleDescription("Select tracks from the list or click a marker. "
                                      "Use the placement fields for keyboard editing.")

    def set_project(self, project: Project) -> None:
        self.project = project
        self.selected = -1
        self.reset_view()

    def set_selected(self, index: int) -> None:
        self.selected = index
        self.update()

    def reset_view(self) -> None:
        self.pointer = None
        self.last_click = None
        super().reset_view()

    def zoom_view(self, steps: float) -> None:
        self.last_click = None
        super().zoom_view(steps)

    def markers(self, projection: Projection) -> list[Projected]:
        tracks = self.project.tracks if self.project else []
        return [projection.project(direction(t.placement.azimuth, t.placement.elevation)) for t in tracks]

    def hit_candidates(self, cursor: QPointF, markers: list[Projected]) -> tuple[int, ...]:
        ranked = [(separation(cursor, screen(p)), p.depth, i) for i, p in enumerate(markers)]
        return tuple(i for distance, _, i in sorted(ranked) if distance <= HIT_RADIUS)

    def marker_order(self, markers: list[Projected]) -> list[int]:
        order = sorted(range(len(markers)), key=lambda i: -markers[i].depth)
        if self.selected in order:
            order.remove(self.selected)
            order.append(self.selected)
        return order

    def marker_groups(self, markers: list[Projected], order: list[int]) -> dict[int, tuple[int, ...]]:
        groups: dict[int, tuple[int, ...]] = {}
        seen: set[int] = set()
        for index in reversed(order):
            if index not in seen:
                group = self.hit_candidates(screen(markers[index]), markers)
                groups[index] = group
                seen.update(group)
        return groups

    def pick(self, cursor: QPointF, markers: list[Projected]) -> tuple[int, ...]:
        hits = self.hit_candidates(cursor, markers)
        if hits:
            return hits
        for index, group in self.marker_groups(markers, self.marker_order(markers)).items():
            if len(group) > 1 and overlap_badge(screen(markers[index])).contains(cursor):
                return group
        return ()

    def height_handle(self, markers: list[Projected]) -> QPointF | None:
        if not 0 <= self.selected < len(markers):
            return None
        point = screen(markers[self.selected])
        inset = QRectF(self.rect()).adjusted(10, 10, -10, -10)
        offsets = [QPointF(0, -33), QPointF(0, 33), QPointF(28, -24), QPointF(-28, -24)]
        for offset in offsets:
            handle = point + offset
            if inset.contains(handle) and all(separation(handle, screen(p)) > HIT_RADIUS + 3 for p in markers):
                return handle
        return None

    def track_geometry(self, painter: QPainter, projection: Projection,
                       position: Placement, color: QColor, selected: bool) -> None:
        p = position
        world = direction(p.azimuth, p.elevation)
        point = screen(projection.project(world))
        base = screen(projection.project(Point3(world.x, world.y, 0)))
        if p.elevation > 0:
            painter.setPen(QPen(translucent(color, 200 if selected else 98), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(base, point)
            painter.drawEllipse(base, 3, 3)
        if p.width > 0:
            arc = self.path(projection, width_arc(p.azimuth, p.elevation, p.width))
            painter.setPen(QPen(translucent(color, 220 if selected else 110), 1.6 if selected else 1))
            painter.drawPolyline(arc)
            painter.setBrush(QColor("#101927"))
            for endpoint in (arc[0], arc[-1]):
                painter.drawEllipse(endpoint, 3.5 if selected else 2.4, 3.5 if selected else 2.4)
            if selected and p.elevation >= 75:
                # Near zenith the physical arc contracts to a point. The floor
                # guide still shows the editable azimuth spread.
                painter.setPen(QPen(translucent(color, 152), 1, Qt.PenStyle.DashLine))
                painter.drawPolyline(self.path(projection, width_arc(p.azimuth, 0, p.width, 0.97)))

    def draw_marker(self, painter: QPainter, point: QPointF, track: Track,
                    color: QColor, selected: bool, overlap: int) -> None:
        muted = track.placement.muted
        gain = gain_radius(track.placement.gain_db)
        color = QColor("#8794a7") if muted else color
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if selected:
            reference = gain_radius(0)
            painter.setPen(QPen(QColor("#72849a"), 0.9, Qt.PenStyle.DotLine))
            painter.drawEllipse(point, reference, reference)
        painter.setPen(QPen(translucent(color, 226 if selected else 146), 1.45 if selected else 1))
        painter.drawEllipse(point, gain, gain)
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#f2f6fd") if selected else QColor("#182438"), 2 if selected else 1))
        painter.drawEllipse(point, 7, 7)
        if muted:
            painter.setPen(QPen(QColor("#202b3d"), 1.8))
            painter.drawLine(point + QPointF(-4, 4), point + QPointF(4, -4))
        if overlap > 1:
            badge = overlap_badge(point)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#25374f"))
            painter.drawRoundedRect(badge, 5, 5)
            painter.setPen(QColor(INK))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, f"×{overlap}")

    def draw_label(self, painter: QPainter, point: QPointF, text: str) -> QRectF:
        metrics = painter.fontMetrics()
        text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(120, self.width() - 24))
        width = metrics.horizontalAdvance(text) + 16
        left = point.x() + 37
        if left + width > self.width() - 6:
            left = max(6, point.x() - width - 24)
        top = max(6, min(self.height() - 29, point.y() + 10))
        bounds = QRectF(left, top, width, 23)
        painter.setPen(QPen(QColor("#405573"), 1))
        painter.setBrush(QColor("#18273b"))
        painter.drawRoundedRect(bounds, 5, 5)
        painter.setPen(QColor(INK))
        painter.drawText(bounds, Qt.AlignmentFlag.AlignCenter, text)
        return bounds

    def draw_names(self, painter: QPainter, markers: list[Projected], order: list[int],
                   occupied: list[QRectF], overlaps: dict[int, int]) -> None:
        inset = QRectF(self.rect()).adjusted(7, 7, -7, -7)
        metrics = painter.fontMetrics()
        painter.setPen(QColor("#becce0"))
        for index in reversed(order):
            if index == self.selected or overlaps.get(index, 0) != 1:
                continue
            point = screen(markers[index])
            name = metrics.elidedText(self.project.tracks[index].label, Qt.TextElideMode.ElideRight, 90)
            width = metrics.horizontalAdvance(name) + 2
            radius = max(11, gain_radius(self.project.tracks[index].placement.gain_db)) + 4
            candidates = (
                QRectF(point.x() + radius, point.y() - 10, width, 20),
                QRectF(point.x() - radius - width, point.y() - 10, width, 20),
                QRectF(point.x() - width / 2, point.y() + radius, width, 20),
                QRectF(point.x() - width / 2, point.y() - radius - 20, width, 20),
            )
            for rect in candidates:
                if inset.contains(rect) and not any(rect.intersects(other) for other in occupied):
                    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, name)
                    occupied.append(rect.adjusted(-3, -1, 3, 1))
                    break

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        self.draw_background(painter)
        projection = self.projection()
        self.draw_room(painter, projection)
        if self.project is None:
            self.draw_label(painter, QPointF(12, 14), "Open audio to place its stems")
            return

        markers = self.markers(projection)
        order = self.marker_order(markers)
        for i in order:
            track = self.project.tracks[i]
            color = QColor(ACCENTS[i % len(ACCENTS)])
            if track.placement.muted:
                color = QColor("#8794a7")
            self.track_geometry(painter, projection, track.placement, color, i == self.selected)
        overlaps = {i: len(group) for i, group in self.marker_groups(markers, order).items()}
        occupied = [QRectF(p.x - 14, p.y - 14, 28, 28) for p in markers]
        for azimuth, _ in AXES:
            axis = projection.project(direction(azimuth, 0, AXIS_LABEL_RADIUS))
            occupied.append(QRectF(axis.x - 28, axis.y - 10, 56, 20))
        for i in order:
            track = self.project.tracks[i]
            point = screen(markers[i])
            overlap = overlaps.get(i, 0)
            self.draw_marker(painter, point, track, QColor(ACCENTS[i % len(ACCENTS)]),
                             i == self.selected, overlap)
            if overlap > 1:
                occupied.append(overlap_badge(point))

        handle = self.height_handle(markers)
        if handle is not None:
            point = screen(markers[self.selected])
            painter.setPen(QPen(QColor("#a5bfdb"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(point, handle)
            painter.setPen(QPen(QColor("#b4cbe6"), 1))
            painter.setBrush(QColor("#20344e"))
            box = QRectF(handle.x() - 8, handle.y() - 8, 16, 16)
            painter.drawRoundedRect(box, 4, 4)
            occupied.append(box)
            painter.setPen(QPen(QColor("#f4f7fc"), 1))
            painter.drawLine(handle + QPointF(0, -5), handle + QPointF(0, 5))
            for sign in (-1, 1):
                tip = handle + QPointF(0, sign * 5)
                painter.drawLine(tip, handle + QPointF(-2.5, sign * 2))
                painter.drawLine(tip, handle + QPointF(2.5, sign * 2))
        if 0 <= self.selected < len(markers):
            track = self.project.tracks[self.selected]
            point = screen(markers[self.selected])
            p = track.placement
            gain = "Muted" if p.muted else f"{p.gain_db:g} dB"
            occupied.append(self.draw_label(painter, point, f"{track.label}  ·  {p.elevation:g}° up  ·  "
                                                            f"{p.width:g}° wide  ·  {gain}"))
        self.draw_names(painter, markers, order, occupied, overlaps)

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip and self.project is not None:
            markers = self.markers(self.projection())
            handle = self.height_handle(markers)
            if handle is not None and separation(event.pos(), handle) <= HEIGHT_HANDLE_HIT:
                QToolTip.showText(event.globalPos(), "Drag up or down to change elevation", self)
                return True
            hits = self.pick(event.pos(), markers)
            if hits:
                name = self.project.tracks[hits[0]].label
                detail = "Click again to cycle overlapping tracks" if len(hits) > 1 else "Drag to change azimuth"
                QToolTip.showText(event.globalPos(), f"{name} — {detail}", self)
                return True
            QToolTip.hideText()
        return super().event(event)

    def mousePressEvent(self, event) -> None:
        cursor = event.position()
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self.pointer = Pointer("orbit", cursor, self.camera)
            self.last_click = None
        elif event.button() == Qt.MouseButton.LeftButton:
            markers = self.markers(self.projection())
            handle = self.height_handle(markers)
            hits = self.pick(cursor, markers)
            if handle is not None and separation(cursor, handle) <= HEIGHT_HANDLE_HIT:
                row, mode, hits = self.selected, "elevation", ()
            elif hits:
                # For overlapping markers, drag the visible selection if its
                # center is essentially the same as the first hit's center.
                row = hits[0]
                if self.selected in hits and separation(screen(markers[row]), screen(markers[self.selected])) < 5:
                    row = self.selected
                mode = "elevation" if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else "azimuth"
            else:
                self.pointer = Pointer("orbit", cursor, self.camera)
                self.last_click = None
                return
            position = self.project.tracks[row].placement
            self.pointer = Pointer(mode, cursor, self.camera, row, hits,
                                   position, cursor - screen(markers[row]))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pointer = self.pointer
        if pointer is None:
            if self.project is not None:
                markers = self.markers(self.projection())
                handle = self.height_handle(markers)
                cursor = event.position()
                if handle is not None and separation(cursor, handle) <= HEIGHT_HANDLE_HIT:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                elif self.pick(cursor, markers):
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.unsetCursor()
            return
        delta = event.position() - pointer.anchor
        if not pointer.dragging:
            if hypot(delta.x(), delta.y()) < DRAG_THRESHOLD:
                return
            pointer.dragging = True
            self.last_click = None
            if pointer.row >= 0:
                self.track_selected.emit(pointer.row)
        if pointer.mode == "orbit":
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.camera = pointer.camera.orbit(delta.x(), delta.y())
            self.update()
            return

        position = pointer.position
        if position is None or pointer.offset is None:
            return
        if pointer.mode == "elevation":
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            travel = max(105, min(self.width(), self.height()) * 0.45)
            value = position.elevation - delta.y() * 90 / travel
            self.elevation_changed.emit(round(max(0, min(90, value)), 1))
            return

        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        target = event.position() - pointer.offset
        height = direction(position.azimuth, position.elevation).z
        hit = self.projection().on_horizontal(target.x(), target.y(), height)
        if position.elevation > 88 or hit is None or hypot(hit.x, hit.y) < 0.035:
            azimuth = wrap_azimuth(position.azimuth + delta.x() * 0.55)
        else:
            azimuth = azimuth_of(hit)
        self.azimuth_changed.emit(round(azimuth, 1))

    def mouseReleaseEvent(self, event) -> None:
        pointer = self.pointer
        self.pointer = None
        self.unsetCursor()
        if (pointer is None or pointer.dragging or not pointer.candidates
                or event.button() != Qt.MouseButton.LeftButton):
            return
        candidates = pointer.candidates
        row = pointer.row
        if self.last_click is not None:
            point, group, last = self.last_click
            if set(group) == set(candidates) and separation(point, pointer.anchor) <= 9:
                candidates = group
                row = candidates[(candidates.index(last) + 1) % len(candidates)]
        self.last_click = (pointer.anchor, candidates, row)
        self.track_selected.emit(row)

    def wheelEvent(self, event) -> None:
        if self.pointer is not None:
            event.accept()
        else:
            super().wheelEvent(event)
