from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSignalBlocker, QThreadPool, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from . import __version__
from .cascade import load_recipe, separate
from .export import export_iamf
from .playback_ui import PlaybackUI
from .project import PROJECT_FILENAME, Project, default_placement, load_project
from .tools import StemSpaceError

from .viewport import ACCENTS, ListenerViewport

STYLE = """
QWidget { background: #121a28; color: #e8edf6; font-size: 13px; }
QLabel#title { font-size: 29px; font-weight: 650; }
QLabel#section { font-size: 16px; font-weight: 620; }
QLabel#hint { color: #a3b2cb; }
QLabel#tag { background: #203c3a; color: #a5ebd9; border: 1px solid #345954;
             border-radius: 5px; padding: 5px 9px; font-weight: 600; }
QPushButton { background: #223047; border: 1px solid #3b4d69; border-radius: 6px; padding: 8px 12px; }
QPushButton:hover { background: #2c3d58; }
QPushButton[compact="true"] { padding: 5px 9px; }
QPushButton#primary { background: #8ce3cc; color: #102720; border-color: #8ce3cc; font-weight: 650; }
QPushButton#primary:hover { background: #acf1df; }
QPushButton:disabled, QPushButton#primary:disabled { background: #1b2536; color: #8898b0; border-color: #2b3a51; }
QListWidget, QDoubleSpinBox { background: #182335; border: 1px solid #35445e; border-radius: 5px; }
QListWidget::item { padding: 7px 8px; }
QListWidget::item:selected { background: #2d4660; color: #ffffff; }
QDoubleSpinBox { padding: 4px; }
QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #526782; border-radius: 3px;
                       background: #182335; }
QCheckBox::indicator:checked { background: #8ce3cc; border-color: #8ce3cc; }
QTabWidget::pane { border: 1px solid #31415b; border-radius: 5px; }
QTabBar::tab { background: #1a273a; color: #b9c7dd; padding: 8px 15px; }
QTabBar::tab:selected { background: #2d4660; color: #fff; }
"""


def label(text: str, role: str = "", wrap: bool = False) -> QLabel:
    result = QLabel(text)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setObjectName(role)
    result.setWordWrap(wrap)
    return result


class Signals(QObject):
    progress = Signal(str)
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.signals = Signals()

    def run(self) -> None:
        try:
            result = self.action(self.signals.progress.emit)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        else:
            self.signals.done.emit(result)


class Window(QMainWindow):
    def __init__(self, *, audio_backend=None, audio_settings=None):
        super().__init__()
        self.project: Project | None = None
        self.job: Task | None = None
        self.dirty = False
        self.setWindowTitle(f"StemSpace {__version__}")
        self.resize(1160, 750)
        self.setMinimumSize(960, 665)

        central = QWidget()
        shell = QVBoxLayout(central)
        shell.setContentsMargins(20, 16, 20, 13)
        shell.setSpacing(12)
        self.setCentralWidget(central)

        title_row = QHBoxLayout()
        title_row.addWidget(label("StemSpace", "title"))
        title_row.addWidget(label(f"v{__version__}", "tag"))
        title_row.addStretch()
        self.new = QPushButton("New from audio…")
        self.open = QPushButton("Open project…")
        self.save = QPushButton("Save")
        self.export = QPushButton("Export IAMF…")
        self.export.setObjectName("primary")
        for button in (self.new, self.open, self.save, self.export):
            title_row.addWidget(button)
        shell.addLayout(title_row)
        self.info = label("Separate a recording or open an existing project.", "hint", wrap=True)
        shell.addWidget(self.info)

        self.tabs = QTabWidget()
        editor = QWidget()
        edit = QHBoxLayout(editor)
        edit.setContentsMargins(14, 15, 14, 15)
        edit.setSpacing(15)
        left = QVBoxLayout()
        left.addWidget(label("Tracks", "section"))
        left.addWidget(label("Check tracks to listen", "hint"))
        self.tracks = QListWidget()
        self.tracks.setMinimumWidth(156)
        self.tracks.setAccessibleName("Editable tracks")
        left.addWidget(self.tracks)
        listen_buttons = QHBoxLayout()
        self.listen_all = QPushButton("All")
        self.listen_solo = QPushButton("Solo")
        self.listen_all.setAccessibleName("Listen to all tracks")
        self.listen_solo.setAccessibleName("Listen to the selected track only")
        self.listen_all.setToolTip("Listen to all tracks")
        self.listen_solo.setToolTip("Listen only to the selected track")
        for button in (self.listen_all, self.listen_solo):
            button.setProperty("compact", True)
            listen_buttons.addWidget(button)
        left.addLayout(listen_buttons)
        edit.addLayout(left, 1)

        middle = QVBoxLayout()
        view_bar = QHBoxLayout()
        view_bar.addWidget(label("3D listener scene", "section"))
        view_bar.addStretch()
        self.zoom_out = QPushButton("−")
        self.zoom_in = QPushButton("+")
        self.reset_view = QPushButton("Reset view")
        for button, name in ((self.zoom_out, "Zoom out"), (self.zoom_in, "Zoom in"),
                             (self.reset_view, "Reset camera view")):
            button.setProperty("compact", True)
            button.setAccessibleName(name)
            button.setToolTip(name)
            view_bar.addWidget(button)
        middle.addLayout(view_bar)
        self.viewport = ListenerViewport()
        middle.addWidget(self.viewport, 1)
        middle.addWidget(label("Click a marker to select it; click again to cycle overlaps. Drag a marker "
                               "to turn it, or its height handle to raise it. Drag empty space to orbit.",
                               "hint", wrap=True))
        edit.addLayout(middle, 4)

        right = QVBoxLayout()
        self.selected = label("Placement", "section", wrap=True)
        right.addWidget(self.selected)
        self.settings = QWidget()
        form = QFormLayout(self.settings)
        form.setContentsMargins(0, 6, 0, 6)
        self.azimuth = self.number(-180, 180, "°", "Azimuth")
        self.elevation = self.number(0, 90, "°", "Elevation above ear level")
        self.width = self.number(0, 180, "°", "Stereo width")
        self.gain = self.number(-60, 12, " dB", "Track gain")
        for name, field in (("Azimuth", self.azimuth), ("Elevation", self.elevation),
                            ("Stereo width", self.width), ("Gain", self.gain)):
            form.addRow(name, field)
            field.valueChanged.connect(self.update_placement)
        self.mute = QCheckBox("Mute in export")
        form.addRow(self.mute)
        self.mute.toggled.connect(self.update_placement)
        self.reset = QPushButton("Reset this placement")
        self.reset.clicked.connect(self.reset_placement)
        form.addRow(self.reset)
        right.addWidget(self.settings)
        right.addWidget(label("Width 0° places a mono point. Increase width to keep the left and right sides apart.",
                              "hint", wrap=True))
        right.addStretch()
        self.residual = QCheckBox("Add residual as a front layer")
        self.residual.toggled.connect(self.update_residual)
        right.addWidget(self.residual)
        right.addWidget(label("Export: 7.1.4 speaker mix with height", "hint", wrap=True))
        edit.addLayout(right, 2)
        self.tabs.addTab(editor, "Placement editor")
        self.status = label("Ready", "hint", wrap=True)
        self.playback = PlaybackUI(self, self.tabs, self.status, backend=audio_backend, settings=audio_settings)
        self.tabs.addTab(self.playback.player, "IAMF Player")
        shell.addWidget(self.tabs, 1)
        shell.addWidget(self.playback.bar)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        shell.addWidget(self.progress)
        shell.addWidget(self.status)

        self.new.clicked.connect(self.new_project)
        self.open.clicked.connect(self.open_project)
        self.save.clicked.connect(self.save_project)
        self.export.clicked.connect(self.export_project)
        self.tracks.currentRowChanged.connect(self.select_track)
        self.tracks.itemChanged.connect(self.publish_monitor)
        self.listen_all.clicked.connect(self.enable_all_tracks)
        self.listen_solo.clicked.connect(self.solo_selected_track)
        self.viewport.track_selected.connect(self.tracks.setCurrentRow)
        self.viewport.azimuth_changed.connect(self.azimuth.setValue)
        self.viewport.elevation_changed.connect(self.elevation.setValue)
        self.zoom_in.clicked.connect(self.viewport.zoom_in)
        self.zoom_out.clicked.connect(self.viewport.zoom_out)
        self.reset_view.clicked.connect(self.viewport.reset_view)
        menu = self.menuBar().addMenu("File")
        menu.addAction("New from audio…", self.new_project, "Ctrl+N")
        menu.addAction("New with YAML recipe…", self.new_with_recipe)
        menu.addAction("Open project…", self.open_project, "Ctrl+O")
        menu.addAction("Save", self.save_project, "Ctrl+S")
        self.refresh_enabled()

    @staticmethod
    def number(lo: int, hi: int, suffix: str, name: str) -> QDoubleSpinBox:
        result = QDoubleSpinBox()
        result.setRange(lo, hi)
        result.setDecimals(1)
        result.setSingleStep(1)
        result.setSuffix(suffix)
        result.setAccessibleName(name)
        return result

    def refresh_enabled(self) -> None:
        idle, loaded = self.job is None, self.project is not None
        for button in (self.new, self.open):
            button.setEnabled(idle)
        for widget in (self.save, self.export, self.tracks, self.viewport, self.residual,
                       self.zoom_in, self.zoom_out, self.reset_view, self.listen_all):
            widget.setEnabled(idle and loaded)
        self.listen_solo.setEnabled(idle and loaded and self.tracks.currentRow() >= 0)
        self.settings.setEnabled(idle and loaded and self.tracks.currentRow() >= 0)
        self.progress.setVisible(not idle)

    def apply_project(self, project: Project) -> None:
        self.project = project
        self.dirty = False
        self.viewport.set_project(project)
        with QSignalBlocker(self.tracks):
            self.tracks.clear()
            for index, track in enumerate(project.tracks):
                item = QListWidgetItem(track.label)
                item.setForeground(QColor(ACCENTS[index % len(ACCENTS)]))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.tracks.addItem(item)
            self.tracks.setCurrentRow(0)
        with QSignalBlocker(self.residual):
            self.residual.setChecked(project.include_residual)
        self.info.setText(f"{Path(project.source).name}  ·  {len(project.tracks)} tracks  ·  "
                          f"{project.sample_rate:,} Hz  ·  {project.frames / project.sample_rate:.1f} s")
        self.status.setText(f"Dry reconstruction peak error: {project.reconstruction['peak_error']:.2g} FS")
        self.playback.set_project(project)
        self.select_track(0)
        self.refresh_enabled()

    def current_track(self):
        index = self.tracks.currentRow()
        if self.project and 0 <= index < len(self.project.tracks):
            return self.project.tracks[index]
        return None

    def select_track(self, index: int) -> None:
        self.viewport.set_selected(index)
        track = self.current_track()
        self.selected.setText(track.label if track else "Placement")
        self.refresh_enabled()
        if not track:
            return
        position = track.placement
        for widget, value in ((self.azimuth, position.azimuth), (self.elevation, position.elevation),
                              (self.width, position.width), (self.gain, position.gain_db)):
            with QSignalBlocker(widget):
                widget.setValue(value)
        with QSignalBlocker(self.mute):
            self.mute.setChecked(position.muted)

    def update_placement(self, *_args) -> None:
        track = self.current_track()
        if track:
            track.placement = replace(track.placement, azimuth=self.azimuth.value(),
                                      elevation=self.elevation.value(), width=self.width.value(),
                                      gain_db=self.gain.value(), muted=self.mute.isChecked())
            self.dirty = True
            self.viewport.update()
            self.status.setText("Placement changed · save the project or export.")
            self.publish_monitor()

    def reset_placement(self) -> None:
        track = self.current_track()
        if track:
            track.placement = default_placement(track.id.rsplit(".", 1)[-1])
            self.dirty = True
            self.select_track(self.tracks.currentRow())
            self.status.setText("Placement reset · save the project or export.")
            self.publish_monitor()

    def update_residual(self, checked: bool) -> None:
        if self.project:
            self.project.include_residual = checked
            self.dirty = True
            self.publish_monitor()

    def publish_monitor(self, *_args) -> None:
        if self.project is not None and self.tracks.count() == len(self.project.tracks):
            enabled = tuple(self.tracks.item(i).checkState() == Qt.CheckState.Checked
                            for i in range(self.tracks.count()))
            self.playback.monitor(self.project, enabled)

    def set_listen_selection(self, selected: int | None) -> None:
        with QSignalBlocker(self.tracks):
            for i in range(self.tracks.count()):
                on = selected is None or i == selected
                self.tracks.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self.publish_monitor()

    def enable_all_tracks(self) -> None:
        self.set_listen_selection(None)

    def solo_selected_track(self) -> None:
        if self.tracks.currentRow() >= 0:
            self.set_listen_selection(self.tracks.currentRow())

    def confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        buttons = QMessageBox.StandardButton
        result = QMessageBox.question(self, "Unsaved placements", "Save changes to this scene?",
                                      buttons.Save | buttons.Discard | buttons.Cancel, buttons.Save)
        return result == buttons.Discard or (result == buttons.Save and self.save_project())

    def save_project(self) -> bool:
        if self.project is None or self.job is not None:
            return False
        try:
            self.project.save()
        except (StemSpaceError, OSError, ValueError) as exc:
            self.show_error(str(exc))
            return False
        self.dirty = False
        self.status.setText(f"Saved {self.project.path.name}")
        return True

    def new_with_recipe(self) -> None:
        if self.job is not None:
            return
        recipe, _ = QFileDialog.getOpenFileName(self, "Choose a cascade recipe", "", "YAML (*.yaml *.yml)")
        if recipe:
            self.new_project(recipe=Path(recipe))

    def new_project(self, _checked=False, *, recipe: Path | None = None) -> None:
        if self.job is not None or not self.confirm_discard():
            return
        audio, _ = QFileDialog.getOpenFileName(
            self, "Choose a recording", "", "Audio (*.wav *.flac *.mp3 *.m4a *.aiff *.aif *.ogg *.opus);;All files (*)"
        )
        if not audio:
            return
        source = Path(audio)
        parent = QFileDialog.getExistingDirectory(self, "Choose where to create the project folder", str(source.parent))
        if parent:
            root = Path(parent) / f"{source.stem} StemSpace"
            self.start_job(lambda progress: separate(source, root, load_recipe(recipe), progress=progress),
                           "Preparing separation…")

    def open_project(self) -> None:
        if self.job is not None or not self.confirm_discard():
            return
        project, _ = QFileDialog.getOpenFileName(self, "Open a StemSpace project", "", f"StemSpace ({PROJECT_FILENAME})")
        if project:
            self.load_path(Path(project))

    def load_path(self, path: Path) -> None:
        try:
            self.apply_project(load_project(path))
        except StemSpaceError as exc:
            self.show_error(str(exc))

    def export_project(self) -> None:
        project = self.project
        if project is None or self.job is not None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export 7.1.4 IAMF speaker mix",
            str(project.root / f"{Path(project.source).stem}.iamf"), "Standalone IAMF (*.iamf)",
        )
        if not selected:
            return
        destination = Path(selected)
        if not destination.suffix:
            destination = destination.with_suffix(".iamf")
        # The save dialog confirms replacement for the exact selected filename.
        overwrite = destination == Path(selected) and destination.exists()
        if self.dirty and not self.save_project():
            return
        self.start_job(lambda progress: export_iamf(project, destination, overwrite=overwrite, progress=progress),
                       "Preparing export…")

    def start_job(self, action, text: str) -> None:
        self.job = Task(action)
        self.job.signals.progress.connect(self.status.setText)
        self.job.signals.done.connect(self.job_done)
        self.job.signals.failed.connect(self.job_failed)
        self.status.setText(text)
        self.refresh_enabled()
        QThreadPool.globalInstance().start(self.job)

    def job_done(self, result) -> None:
        self.job = None
        if isinstance(result, Project):
            self.apply_project(result)
        self.refresh_enabled()

    def job_failed(self, message: str) -> None:
        self.job = None
        self.refresh_enabled()
        self.show_error(message)

    def show_error(self, message: str) -> None:
        self.status.setText("Could not complete the operation.")
        box = QMessageBox(QMessageBox.Icon.Warning, "StemSpace", "Could not complete the operation.",
                          QMessageBox.StandardButton.Ok, self)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setInformativeText(message.splitlines()[0] if message else "Unknown error.")
        if "\n" in message:
            box.setDetailedText(message)
        box.exec()

    def closeEvent(self, event) -> None:
        if self.job is not None:
            self.status.setText("Wait for the current operation to finish before closing.")
            event.ignore()
        elif not self.confirm_discard():
            event.ignore()
        else:
            self.playback.close()
            event.accept()


def launch(project: Path | None = None) -> int:
    app = QApplication([sys.argv[0]])
    app.setApplicationName("StemSpace")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    if project is not None:
        window.load_path(project)
    return app.exec()
