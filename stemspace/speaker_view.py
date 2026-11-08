"""Read-only 3D speaker activity and channel inspection."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QToolTip, QVBoxLayout, QWidget

from .speaker_levels import FLOOR_DB, LFE, SPEAKERS, SpeakerLevels
from .spatial import Camera, Point3, direction
from .viewport import AXES, AXIS_LABEL_RADIUS, DRAG_THRESHOLD, INK, OrbitViewport, screen, separation, translucent

EAR = "#72ddc6"
HEIGHT = "#9eb4ff"
LOW = "#efc181"
DB_RANGE = (-60.0, 0.0)


def intensity(db: float) -> float:
    lo, hi = DB_RANGE
    return max(0.0, min(1.0, (db - lo) / (hi - lo)))


def level_text(db: float) -> str:
    return "≤ −90 dBFS" if db <= FLOOR_DB + 0.01 else f"{db:.1f} dBFS"


def label(text: str, name: str = "") -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setObjectName(name)
    return widget


@dataclass
class Drag:
    start: QPointF
    camera: Camera
    moved: bool = False


class SpeakerViewport(OrbitViewport):
    speaker_selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(345, 286)
        self.setMouseTracking(True)
        self.setAccessibleName("3D 7.1.4 speaker activity")
        self.setAccessibleDescription("Click a speaker to inspect it. Drag to orbit. "
                                      "Use the speaker list for keyboard selection.")
        self.world = {i: direction(*speaker.direction) for i, speaker in enumerate(SPEAKERS)
                      if speaker.direction is not None}
        self.selected = 2
        self.levels = None
        self.drag: Drag | None = None
        self.labels: dict[int, QRectF] = {}

    def set_selected(self, index: int) -> None:
        self.selected = index
        self.update()

    def set_levels(self, rms) -> None:
        self.levels = rms
        self.update()

    def reset_view(self) -> None:
        self.drag = None
        super().reset_view()

    def points(self, projection):
        return {i: projection.project(pos) for i, pos in self.world.items()}

    def pick(self, at: QPointF) -> int | None:
        points = self.points(self.projection())
        hits = [(separation(at, screen(p)), p.depth, i) for i, p in points.items()
                if separation(at, screen(p)) <= 13]
        if hits:
            return min(hits)[2]
        return next((i for i, rect in self.labels.items() if rect.contains(at)), None)

    def place_labels(self, painter: QPainter, projection, points) -> None:
        occupied = [QRectF(p.x - 10, p.y - 10, 20, 20) for p in points.values()]
        for azimuth, _ in AXES:
            p = projection.project(direction(azimuth, 0, AXIS_LABEL_RADIUS))
            occupied.append(QRectF(p.x - 23, p.y - 9, 46, 18))
        inset = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        self.labels = {}
        order = sorted(points, key=lambda i: (i != self.selected, points[i].depth))
        for i in order:
            p = points[i]
            text = SPEAKERS[i].code
            w = max(24, painter.fontMetrics().horizontalAdvance(text) + 10)
            h = 19
            gap = 13
            offsets = ((gap, -h / 2), (-gap - w, -h / 2), (-w / 2, gap),
                       (-w / 2, -h - gap), (gap, -h - gap), (-gap - w, gap),
                       (gap + 13, -h - gap), (-gap - w - 13, -h - gap),
                       (gap + 13, gap), (-gap - w - 13, gap))
            for dx, dy in offsets:
                rect = QRectF(p.x + dx, p.y + dy, w, h)
                if inset.contains(rect) and not any(rect.intersects(item) for item in occupied):
                    self.labels[i] = rect
                    occupied.append(rect.adjusted(-2, -1, 2, 1))
                    break

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        self.draw_background(painter)
        projection = self.projection()
        self.draw_room(painter, projection)
        points = self.points(projection)
        # Show a vertical guide for all four height speakers before their dots.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for i, world in self.world.items():
            if not world.z:
                continue
            foot = screen(projection.project(Point3(world.x, world.y, 0)))
            painter.setPen(QPen(translucent(HEIGHT, 166 if i == self.selected else 78),
                                1, Qt.PenStyle.DashLine))
            painter.drawLine(foot, screen(points[i]))
            painter.drawEllipse(foot, 2, 2)

        order = sorted(points, key=lambda i: (i == self.selected, -points[i].depth))
        for i in order:
            point = screen(points[i])
            color = QColor(HEIGHT if self.world[i].z else EAR)
            strength = intensity(float(self.levels[i])) if self.levels is not None else 0
            if strength > 0:
                radius = 10 + 17 * strength
                halo = QRadialGradient(point, radius)
                halo.setColorAt(0, translucent(color, round(75 + 145 * strength)))
                halo.setColorAt(1, translucent(color, 0))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(halo))
                painter.drawEllipse(point, radius, radius)
            painter.setPen(QPen(QColor(INK) if i == self.selected else translucent(color, 148),
                                1.6 if i == self.selected else 1))
            painter.setBrush(translucent(color, round(34 + 221 * strength)))
            painter.drawEllipse(point, 7, 7)

        self.place_labels(painter, projection, points)
        for i, rect in self.labels.items():
            color = HEIGHT if self.world[i].z else EAR
            point = screen(points[i])
            if separation(point, rect.center()) > 27:
                painter.setPen(QPen(translucent(color, 85), 1))
                painter.drawLine(point, rect.center())
            painter.setPen(QPen(translucent(color, 170 if i == self.selected else 66), 1))
            painter.setBrush(QColor("#263b53" if i == self.selected else "#172538"))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QColor(INK if i == self.selected else color))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, SPEAKERS[i].code)

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            hit = self.pick(QPointF(event.pos()))
            if hit is not None:
                QToolTip.showText(event.globalPos(), SPEAKERS[hit].name, self)
                return True
            QToolTip.hideText()
        return super().event(event)

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton,
                              Qt.MouseButton.MiddleButton):
            self.drag = Drag(event.position(), self.camera)
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self.drag is None:
            if self.pick(event.position()) is None:
                self.unsetCursor()
            else:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            return
        delta = event.position() - self.drag.start
        if not self.drag.moved and separation(event.position(), self.drag.start) < DRAG_THRESHOLD:
            return
        self.drag.moved = True
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.camera = self.drag.camera.orbit(delta.x(), delta.y())
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        drag, self.drag = self.drag, None
        self.unsetCursor()
        if drag is not None and not drag.moved and event.button() == Qt.MouseButton.LeftButton:
            hit = self.pick(event.position())
            if hit is not None:
                self.speaker_selected.emit(hit)


class Meter(QWidget):
    def __init__(self, color: str):
        super().__init__()
        self.color = QColor(color)
        self.value: tuple[float, float] | None = None
        self.setFixedHeight(12)

    def set_reading(self, value: tuple[float, float] | None) -> None:
        if self.value != value:
            self.value = value
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(0, 1, -1, -1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#263751"))
        painter.drawRoundedRect(bounds, 4, 4)
        if self.value is None:
            return
        level, peak = (intensity(x) for x in self.value)
        if level:
            filled = QRectF(bounds.x(), bounds.y(), bounds.width() * level, bounds.height())
            painter.setBrush(self.color)
            painter.drawRoundedRect(filled, 4, 4)
        if peak:
            x = bounds.x() + max(1, min(bounds.width() - 1, peak * bounds.width()))
            painter.setPen(QPen(QColor(INK), 1))
            painter.drawLine(QPointF(x, bounds.y() + 1), QPointF(x, bounds.bottom() - 1))


class SpeakerActivity(QWidget):
    def __init__(self):
        super().__init__()
        self.timeline: SpeakerLevels | None = None
        self.row = -1
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(15)
        left = QVBoxLayout()
        toolbar = QHBoxLayout()
        toolbar.addWidget(label("Speaker activity", "section"))
        toolbar.addStretch()
        self.view = SpeakerViewport()
        for text, name, handler in (
            ("−", "Zoom out", self.view.zoom_out),
            ("+", "Zoom in", self.view.zoom_in),
            ("Reset view", "Reset camera view", self.view.reset_view),
        ):
            button = QPushButton(text)
            button.setProperty("compact", True)
            button.setAccessibleName(name)
            button.clicked.connect(handler)
            toolbar.addWidget(button)
        left.addLayout(toolbar)
        left.addWidget(self.view, 1)
        self.note = label("Open an IAMF file to see the decoded speaker activity.", "hint")
        self.note.setWordWrap(True)
        left.addWidget(self.note)
        root.addLayout(left, 1)

        inspect = QVBoxLayout()
        panel = QWidget()
        panel.setMinimumWidth(214)
        panel.setMaximumWidth(275)
        panel.setLayout(inspect)
        inspect.setContentsMargins(0, 0, 0, 0)
        inspect.addWidget(label("Inspect speaker", "section"))
        self.selector = QComboBox()
        self.selector.setAccessibleName("Speaker to inspect")
        for item in SPEAKERS:
            self.selector.addItem(f"{item.code}  ·  {item.name}")
        self.selector.setCurrentIndex(self.view.selected)
        inspect.addWidget(self.selector)
        self.reading = label("RMS —   /   Peak —", "hint")
        inspect.addWidget(self.reading)
        self.meter = Meter(EAR)
        inspect.addWidget(self.meter)
        legend = label("● Ear level   ● Overhead", "hint")
        legend.setTextFormat(Qt.TextFormat.RichText)
        legend.setText(f"<span style='color:{EAR}'>●</span> Ear level &nbsp; "
                       f"<span style='color:{HEIGHT}'>●</span> Overhead")
        inspect.addWidget(legend)
        inspect.addSpacing(14)
        inspect.addWidget(label("LFE", "section"))
        self.lfe_text = label("RMS —   /   Peak —", "hint")
        inspect.addWidget(self.lfe_text)
        self.lfe_meter = Meter(LOW)
        inspect.addWidget(self.lfe_meter)
        footer = label("Decoded 7.1.4 signal · before monitor volume and output mapping.", "hint")
        footer.setWordWrap(True)
        inspect.addStretch()
        inspect.addWidget(footer)
        root.addWidget(panel)
        self.selector.currentIndexChanged.connect(self.select)
        self.view.speaker_selected.connect(self.selector.setCurrentIndex)

    def select(self, index: int) -> None:
        self.view.set_selected(index)
        self.update_readouts()

    def clear(self, message: str) -> None:
        self.timeline = None
        self.row = -1
        self.note.setText(message)
        self.note.setToolTip("")
        self.view.set_levels(None)
        self.update_readouts()

    def set_timeline(self, timeline: SpeakerLevels) -> None:
        self.timeline = timeline
        self.row = -1
        self.note.setText("Click a speaker to inspect it. Drag to orbit; scroll to zoom.")
        self.note.setToolTip("")
        self.set_position(0)

    def set_position(self, seconds: float) -> None:
        if self.timeline is None:
            return
        row = self.timeline.index_at(seconds)
        if row != self.row:
            self.row = row
            self.view.set_levels(self.timeline.rms[row])
            self.update_readouts()

    def update_readouts(self) -> None:
        index = self.selector.currentIndex()
        if not 0 <= index < len(SPEAKERS):
            return
        if self.timeline is None or self.row < 0:
            self.reading.setText("RMS —   /   Peak —")
            self.lfe_text.setText("RMS —   /   Peak —")
            self.meter.set_reading(None)
            self.lfe_meter.set_reading(None)
            return
        rms, peak = self.timeline.rms[self.row], self.timeline.peak[self.row]
        self.meter.color = QColor(LOW if index == LFE else HEIGHT if SPEAKERS[index].direction[1] else EAR)
        for i, text, meter in ((index, self.reading, self.meter), (LFE, self.lfe_text, self.lfe_meter)):
            reading = float(rms[i]), float(peak[i])
            text.setText(f"RMS {level_text(reading[0])}  /  Peak {level_text(reading[1])}")
            meter.set_reading(reading)
        self.meter.update()
