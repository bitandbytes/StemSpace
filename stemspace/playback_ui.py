"""Qt transport and output preferences, shared by both listening modes."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QSettings, QSignalBlocker, QThreadPool, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from .iamf_decode import DecodeCache, DecodeCancelled
from .playback import Controls, Device, PortAudio, StemSource, Transport, WavSource
from .playback_render import LAYOUTS
from .project import Project
from .speaker_levels import AnalysisCancelled, LAYOUT as SPEAKER_LAYOUT, read_speaker_levels
from .speaker_view import SpeakerActivity
from .tools import StemSpaceError


def plain(text: str) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def timestamp(seconds: float) -> str:
    secs = max(0, int(seconds))
    return f"{secs // 60}:{secs % 60:02d}"


class Preferences:
    def __init__(self, settings=None):
        self.settings = settings if settings is not None else QSettings("StemSpace", "StemSpace")
        self.data = {"device": "", "layout": "binaural", "maps": {}, "volume": 60}
        try:
            stored = json.loads(self.settings.value("audio-output", "{}", str))
            if isinstance(stored, dict):
                for key, value in self.data.items():
                    if isinstance(stored.get(key), type(value)):
                        self.data[key] = stored[key]
        except (TypeError, ValueError):
            pass
        if self.data["layout"] not in LAYOUTS:
            self.data["layout"] = "binaural"
        self.data["volume"] = min(100, max(0, self.data["volume"]))

    def save(self) -> None:
        self.settings.setValue("audio-output", json.dumps(self.data))


def stored_map(data: dict, device: Device, layout: str) -> tuple[int, ...]:
    n = LAYOUTS[layout].channels
    maps = data["maps"].get(device.key, {})
    value = maps.get(layout) if isinstance(maps, dict) else None
    if (isinstance(value, list) and len(value) == n
            and all(type(ch) is int and 1 <= ch <= device.channels for ch in value)):
        return tuple(value)
    return tuple(range(1, n + 1))


def find_device(devices: list[Device], default: int | None, key: str) -> Device:
    device = next((d for d in devices if (d.key == key if key else d.index == default)), None)
    if device is None:
        raise StemSpaceError("The selected output is unavailable. Choose an output device in Audio Output.")
    return device


class OutputDialog(QDialog):
    def __init__(self, parent, preferences: dict, backend, test_speaker):
        super().__init__(parent)
        self.setWindowTitle("Audio Output")
        self.resize(500, 285)
        self.backend, self.test_speaker = backend, test_speaker
        self.data = deepcopy(preferences)
        self.devices, self.default = backend.devices()
        self.selected: Device | None = None
        self.spins: list[QSpinBox] = []
        outer = QVBoxLayout(self)
        form = QFormLayout()
        self.device = QComboBox()
        default_device = next((d for d in self.devices if d.index == self.default), None)
        if default_device:
            self.device.addItem(f"System default · {default_device.name}", "")
        for item in self.devices:
            self.device.addItem(f"{item.name} · {item.channels} outputs", item.key)
        self.layout_choice = QComboBox()
        form.addRow("Device", self.device)
        form.addRow("Render layout", self.layout_choice)
        outer.addLayout(form)
        self.routing = QFormLayout()
        outer.addLayout(self.routing)
        self.hint = plain("Output numbers refer to device channels. Use Test to verify each speaker.")
        self.hint.setWordWrap(True)
        outer.addWidget(self.hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        self.ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        outer.addWidget(buttons)
        index = self.device.findData(self.data["device"])
        if index >= 0:
            self.device.setCurrentIndex(index)
        self.device.currentIndexChanged.connect(self.update_device)
        self.layout_choice.currentIndexChanged.connect(self.update_layout)
        self.update_device()

    def update_device(self, *_args) -> None:
        previous = self.layout_choice.currentData() or self.data["layout"]
        key = self.device.currentData()
        available = []
        try:
            self.selected = find_device(self.devices, self.default, key or "")
        except StemSpaceError:
            self.selected = None
        if self.selected:
            for layout, info in LAYOUTS.items():
                if info.channels > self.selected.channels:
                    continue
                try:
                    self.backend.configure(self.selected, layout, tuple(range(1, info.channels + 1)))
                except (StemSpaceError, ValueError):
                    continue
                available.append(layout)
        with QSignalBlocker(self.layout_choice):
            self.layout_choice.clear()
            for layout in available:
                self.layout_choice.addItem(LAYOUTS[layout].name, layout)
            index = self.layout_choice.findData(previous)
            if index >= 0:
                self.layout_choice.setCurrentIndex(index)
        self.update_layout()

    def update_layout(self, *_args) -> None:
        while self.routing.rowCount():
            self.routing.removeRow(0)
        self.spins.clear()
        layout = self.layout_choice.currentData()
        self.ok.setEnabled(bool(self.selected and layout))
        if not self.selected or not layout:
            self.hint.setText("No supported render layout is available for this device.")
            return
        for i, (speaker, channel) in enumerate(zip(LAYOUTS[layout].speakers,
                                                   stored_map(self.data, self.selected, layout))):
            row = QWidget()
            horizontal = QHBoxLayout(row)
            horizontal.setContentsMargins(0, 0, 0, 0)
            spin = QSpinBox()
            spin.setAccessibleName(f"{speaker} output channel")
            spin.setRange(1, self.selected.channels)
            spin.setValue(channel)
            test = QPushButton("Test")
            test.setAccessibleName(f"Test {speaker}")
            test.clicked.connect(lambda _checked=False, number=i: self.test(number))
            horizontal.addWidget(spin, 1)
            horizontal.addWidget(test)
            self.spins.append(spin)
            self.routing.addRow(speaker, row)
            spin.valueChanged.connect(self.remember_map)
        self.hint.setText("Output numbers refer to device channels. Use Test to verify each speaker.")

    def remember_map(self, *_args) -> None:
        layout = self.layout_choice.currentData()
        if self.selected and layout:
            maps = self.data["maps"].get(self.selected.key)
            if not isinstance(maps, dict):
                maps = self.data["maps"][self.selected.key] = {}
            maps[layout] = [s.value() for s in self.spins]

    def current_output(self):
        if not self.selected or not self.layout_choice.currentData():
            raise StemSpaceError("Select a supported audio output.")
        return self.backend.configure(self.selected, self.layout_choice.currentData(),
                                      tuple(spin.value() for spin in self.spins))

    def test(self, speaker: int) -> None:
        try:
            output = self.current_output()
        except (StemSpaceError, ValueError) as exc:
            self.hint.setText(str(exc))
            return
        self.test_speaker(output, speaker)
        self.hint.setText(f"Testing {LAYOUTS[output.layout].speakers[speaker]} "
                          f"on device output {output.mapping[speaker]}.")

    def apply(self) -> None:
        try:
            self.current_output()
        except (StemSpaceError, ValueError) as exc:
            self.hint.setText(str(exc))
            return
        self.remember_map()
        self.data["device"] = self.device.currentData()
        self.data["layout"] = self.layout_choice.currentData()
        self.accept()


class DecodeSignals(QObject):
    done = Signal(int, object, str)


class DecodeTask(QRunnable):
    def __init__(self, cache: DecodeCache, source: Path, layout: str, generation: int):
        super().__init__()
        self.cache, self.source, self.layout, self.generation = cache, source, layout, generation
        self.signals = DecodeSignals()
        self.cancel = Event()

    def run(self) -> None:
        try:
            result = WavSource(self.cache.decode(self.source, self.layout, cancel=self.cancel), self.layout)
        except DecodeCancelled:
            self.signals.done.emit(self.generation, None, "")
        except Exception as exc:
            self.signals.done.emit(self.generation, None, str(exc))
        else:
            self.signals.done.emit(self.generation, result, "")


class SpeakerTask(QRunnable):
    def __init__(self, cache: DecodeCache, source: Path, shared_pcm: Path | None, generation: int):
        super().__init__()
        self.cache, self.source, self.shared_pcm = cache, source, shared_pcm
        self.generation = generation
        self.signals = DecodeSignals()
        self.cancel = Event()

    def run(self) -> None:
        try:
            pcm = self.shared_pcm or self.cache.decode(self.source, SPEAKER_LAYOUT, cancel=self.cancel)
            result = read_speaker_levels(pcm, self.cancel)
        except (AnalysisCancelled, DecodeCancelled):
            self.signals.done.emit(self.generation, None, "")
        except Exception as exc:
            self.signals.done.emit(self.generation, None, str(exc))
        else:
            self.signals.done.emit(self.generation, result, "")


class PlaybackUI(QObject):
    def __init__(self, parent, tabs, status, *, backend=None, settings=None):
        super().__init__(parent)
        self.window, self.tabs, self.status = parent, tabs, status
        self.backend = backend if backend is not None else PortAudio()
        self.transport = Transport(self.backend)
        self.preferences = Preferences(settings)
        self.cache = DecodeCache()
        self.stems = self.iamf = None
        self.iamf_path: Path | None = None
        self.mode = "editor"
        self.positions = {"editor": 0.0, "iamf": 0.0}
        self.token = self.generation = 0
        self.tasks: dict[int, DecodeTask] = {}
        self.speaker_generation = 0
        self.speaker_tasks: dict[int, SpeakerTask] = {}
        self.last_error = None

        self.bar = QWidget()
        horizontal = QHBoxLayout(self.bar)
        horizontal.setContentsMargins(0, 1, 0, 1)
        self.play = QPushButton("Play")
        self.play.setMinimumWidth(66)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setAccessibleName("Playback position")
        self.seek.setRange(0, 1000)
        self.seek.setTracking(False)
        self.time = plain("0:00 / 0:00")
        self.time.setMinimumWidth(94)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setAccessibleName("Monitor volume")
        self.volume.setToolTip("Monitor volume")
        self.volume.setRange(0, 100)
        self.volume.setFixedWidth(78)
        self.volume.setValue(self.preferences.data["volume"])
        self.transport.set_volume(self.volume.value() / 100)
        self.output_button = QPushButton("Audio Output…")
        self.layout_label = plain("")
        self.layout_label.setObjectName("hint")
        self.underrun_label = plain("")
        self.underrun_label.setObjectName("hint")
        self.underrun_label.setToolTip("Audio gaps reported by the output device during this playback run.")
        self.underrun_label.hide()
        horizontal.addWidget(self.play)
        horizontal.addWidget(self.seek, 1)
        horizontal.addWidget(self.time)
        horizontal.addWidget(self.underrun_label)
        horizontal.addWidget(self.layout_label)
        horizontal.addWidget(plain("Volume"))
        horizontal.addWidget(self.volume)
        horizontal.addWidget(self.output_button)

        self.player = QWidget()
        player_layout = QVBoxLayout(self.player)
        player_layout.setContentsMargins(17, 13, 17, 13)
        file_header = QHBoxLayout()
        title = plain("IAMF Player")
        title.setObjectName("section")
        file_header.addWidget(title)
        file_header.addStretch()
        self.open = QPushButton("Open IAMF…")
        file_header.addWidget(self.open)
        player_layout.addLayout(file_header)
        self.file_status = plain("Open a standalone .iamf file to hear the saved mix.")
        self.file_status.setWordWrap(True)
        self.file_status.setObjectName("hint")
        player_layout.addWidget(self.file_status)
        self.speakers = SpeakerActivity()
        player_layout.addWidget(self.speakers, 1)

        self.play.clicked.connect(self.toggle)
        self.seek.valueChanged.connect(self.scrub)
        self.volume.valueChanged.connect(self.set_volume)
        self.output_button.clicked.connect(self.choose_output)
        self.open.clicked.connect(self.choose_iamf)
        self.tabs.currentChanged.connect(self.change_tab)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    @property
    def layout(self) -> str:
        return self.preferences.data["layout"]

    def source(self):
        return self.stems if self.mode == "editor" else self.iamf

    def capture(self) -> None:
        state = self.transport.snapshot
        if state.token == self.token and state.duration:
            self.positions[self.mode] = state.position

    def attach(self) -> None:
        self.token += 1
        self.last_error = None
        self.transport.load(self.source(), self.positions[self.mode], self.token)
        self.refresh()

    def set_project(self, project: Project) -> None:
        self.stems = StemSource(project)
        self.positions["editor"] = 0
        if self.mode == "editor":
            self.attach()

    def monitor(self, project: Project, enabled: tuple[bool, ...]) -> None:
        if self.stems is not None:
            self.stems.set_controls(Controls(tuple(t.placement for t in project.tracks),
                                             enabled, project.include_residual))

    def change_tab(self, index: int) -> None:
        mode = "editor" if index == 0 else "iamf"
        if self.mode != mode:
            self.capture()
            self.mode = mode
            self.attach()

    def refresh(self) -> None:
        state = self.transport.snapshot
        active = state.token == self.token
        source = self.source()
        self.play.setEnabled(source is not None)
        self.seek.setEnabled(source is not None)
        self.play.setText("Pause" if active and state.playing else "Play")
        duration = source.frames / source.rate if source else 0
        position = state.position if active and duration else self.positions[self.mode] if duration else 0
        if not self.seek.isSliderDown():
            with QSignalBlocker(self.seek):
                self.seek.setValue(round(1000 * position / duration) if duration else 0)
        self.time.setText("Buffering…" if active and state.buffering
                          else f"{timestamp(position)} / {timestamp(duration)}")
        underruns = state.underruns if active else 0
        self.underrun_label.setVisible(underruns > 0)
        self.underrun_label.setText(f"Underruns: {underruns}" if underruns else "")
        self.layout_label.setText(LAYOUTS[self.layout].name)
        if self.mode == "iamf":
            self.speakers.set_position(position if source else self.positions["iamf"])
        if active and state.error and state.error != self.last_error:
            self.last_error = state.error
            self.status.setText(f"Playback: {state.error}")
        elif active and not state.error:
            self.last_error = None

    def set_volume(self, value: int) -> None:
        self.transport.set_volume(value / 100)
        self.preferences.data["volume"] = value
        self.preferences.save()

    def selected_output(self):
        devices, default = self.backend.devices()
        data = self.preferences.data
        device = find_device(devices, default, data["device"])
        return self.backend.configure(device, self.layout, stored_map(data, device, self.layout))

    def toggle(self) -> None:
        if self.transport.snapshot.token == self.token and self.transport.snapshot.playing:
            self.transport.pause()
        elif self.source() is not None:
            try:
                output = self.selected_output()
            except Exception as exc:
                self.status.setText(f"Playback: {exc}")
                return
            self.transport.play(output)

    def scrub(self, value: int) -> None:
        source = self.source()
        if source:
            seconds = (value / 1000) * source.frames / source.rate
            self.positions[self.mode] = seconds
            self.transport.seek(seconds)

    def choose_output(self) -> None:
        self.transport.pause()
        try:
            dialog = OutputDialog(self.window, self.preferences.data, self.backend, self.transport.test_speaker)
        except Exception as exc:
            self.status.setText(f"Audio Output: {exc}")
            return
        if not dialog.exec():
            return
        self.apply_output(dialog.data)

    def apply_output(self, data: dict) -> None:
        self.capture()
        previous = self.layout
        self.preferences.data = data
        self.preferences.save()
        if previous != self.layout and self.iamf_path is not None:
            self.render_iamf()
        self.refresh()

    def choose_iamf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self.window, "Open standalone IAMF", "", "IAMF (*.iamf)")
        if path:
            self.open_iamf(Path(path))

    def open_iamf(self, path: Path) -> None:
        self.iamf_path = path
        self.positions["iamf"] = 0
        self.speakers.clear("Speaker activity will be prepared after the file opens.")
        self.render_iamf()

    def render_iamf(self) -> None:
        if self.iamf_path is None:
            return
        self.generation += 1
        for task in self.tasks.values():
            task.cancel.set()
        self.cancel_speaker_tasks()
        self.iamf = None
        if self.mode == "iamf":
            self.attach()
        self.file_status.setText(f"Decoding {self.iamf_path.name} for {LAYOUTS[self.layout].name}…")
        task = DecodeTask(self.cache, self.iamf_path, self.layout, self.generation)
        self.tasks[self.generation] = task
        task.signals.done.connect(self.decoded)
        QThreadPool.globalInstance().start(task)

    def decoded(self, generation: int, source, error: str) -> None:
        self.tasks.pop(generation, None)
        if generation != self.generation:
            return
        if error:
            self.file_status.setText(error)
            if self.speakers.timeline is None:
                self.speakers.clear("Speaker activity is unavailable until this file can be decoded.")
            return
        if source is None:
            return
        self.iamf = source
        self.file_status.setText(f"{self.iamf_path.name} · {LAYOUTS[source.layout].name} · "
                                 f"{timestamp(source.frames / source.rate)}")
        if self.mode == "iamf":
            self.attach()
        if self.speakers.timeline is None:
            self.prepare_speakers(source)

    def cancel_speaker_tasks(self) -> None:
        self.speaker_generation += 1
        for task in self.speaker_tasks.values():
            task.cancel.set()

    def prepare_speakers(self, source: WavSource) -> None:
        self.speakers.clear("Preparing the 7.1.4 speaker levels…")
        generation = self.speaker_generation
        shared = source.path if source.layout == SPEAKER_LAYOUT else None
        task = SpeakerTask(self.cache, self.iamf_path, shared, generation)
        self.speaker_tasks[generation] = task
        task.signals.done.connect(self.speakers_ready)
        QThreadPool.globalInstance().start(task)

    def speakers_ready(self, generation: int, timeline, error: str) -> None:
        self.speaker_tasks.pop(generation, None)
        if generation != self.speaker_generation:
            return
        if error:
            self.speakers.clear("A 7.1.4 speaker view is unavailable for this file.")
            self.speakers.note.setToolTip(error)
        elif timeline is not None:
            self.speakers.set_timeline(timeline)
            if self.mode == "iamf":
                self.refresh()
            else:
                self.speakers.set_position(self.positions["iamf"])

    def close(self) -> None:
        self.timer.stop()
        self.generation += 1
        for task in self.tasks.values():
            task.cancel.set()
        self.cancel_speaker_tasks()
        self.transport.close()
        self.cache.close()
