from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from stemspace.iamf_decode import DecodeCache, DecodeCancelled
from stemspace.speaker_levels import (
    AnalysisCancelled, FLOOR_DB, LFE, SPEAKERS, read_speaker_levels,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "speaker-check.iamf"


class SpeakerLevelTest(unittest.TestCase):
    def test_rms_peak_and_tail_use_the_actual_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = np.zeros((2 * 400 + 13, 12), np.float32)
            audio[:200, 7] = 0.5
            audio[800:, LFE] = -0.25
            wav = Path(tmp) / "speakers.wav"
            sf.write(wav, audio, 8000, subtype="FLOAT")
            timeline = read_speaker_levels(wav)
            self.assertEqual(timeline.window_frames, 400)
            self.assertEqual(timeline.rms.shape, (3, 12))
            self.assertAlmostEqual(float(timeline.rms[0, 7]), 20 * np.log10(0.5 / np.sqrt(2)), places=4)
            self.assertAlmostEqual(float(timeline.peak[0, 7]), 20 * np.log10(0.5), places=4)
            self.assertAlmostEqual(float(timeline.rms[1, 7]), FLOOR_DB, places=4)
            self.assertAlmostEqual(float(timeline.rms[-1, LFE]), 20 * np.log10(0.25), places=4)
            self.assertEqual([timeline.index_at(s) for s in (-1, 0.049, 0.051, 100)], [0, 0, 1, 2])
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(AnalysisCancelled):
                read_speaker_levels(wav, cancel)

    @unittest.skipUnless(shutil.which("node"), "needs Node for IAMF")
    def test_standalone_check_isolates_every_decoded_speaker(self):
        cache = DecodeCache()
        self.addCleanup(cache.close)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(DecodeCancelled):
            cache.decode(EXAMPLE, "7.1.4", cancel=cancel)
        timeline = read_speaker_levels(cache.decode(EXAMPLE, "7.1.4"))
        expected = ("FL", "FR", "C", "LFE", "SL", "SR", "RL", "RR", "TFL", "TFR", "TRL", "TRR")
        for i, name in enumerate(expected):
            values = timeline.rms[timeline.index_at(i + 0.51)]
            loudest = int(np.argmax(values))
            self.assertEqual(SPEAKERS[loudest].code, name)
            self.assertGreater(values[loudest], -22)
            self.assertTrue(np.all(np.delete(values, loudest) < -40))
        self.assertIsNone(SPEAKERS[LFE].direction)


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from stemspace.ui import Window
    from stemspace.viewport import screen
    from test_playback import FakeBackend
except ModuleNotFoundError as exc:
    if exc.name not in {"PySide6", "soxr"}:
        raise
    QApplication = None


@unittest.skipIf(QApplication is None or not shutil.which("node"), "needs the UI extra and Node")
class PlayerViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.backend = FakeBackend()
        settings = QSettings(str(Path(self.folder.name) / "prefs.ini"), QSettings.Format.IniFormat)
        self.window = Window(audio_backend=self.backend, audio_settings=settings)
        self.window.show()
        self.window.tabs.setCurrentIndex(1)
        self.p = self.window.playback
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def wait(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            time.sleep(0.008)
        self.fail(f"Player did not reach the expected state: {self.p.file_status.text()}")

    def seek(self, seconds):
        duration = self.p.iamf.frames / self.p.iamf.rate
        self.p.seek.setValue(round(seconds / duration * 1000))
        self.wait(lambda: abs(self.p.transport.snapshot.position - seconds) < 0.025)

    def test_plays_without_a_project_while_meters_prepare_and_inspects_on_seek(self):
        started, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def delayed(path, cancel):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test did not release the analyzer.")
            return read_speaker_levels(path, cancel)

        with patch("stemspace.playback_ui.read_speaker_levels", side_effect=delayed):
            self.p.open_iamf(EXAMPLE)
            self.wait(started.is_set)
            self.assertIsNone(self.window.project)
            self.assertIsNone(self.p.speakers.timeline)
            self.p.play.click()
            self.wait(lambda: self.p.transport.snapshot.position > 0.06)
            self.assertIsNone(self.p.speakers.timeline)
            self.p.transport.pause()
            self.wait(lambda: not self.p.transport.snapshot.playing)
            release.set()
            self.wait(lambda: self.p.speakers.timeline is not None)

        self.seek(11.50)
        speakers = self.p.speakers
        self.wait(lambda: speakers.row == speakers.timeline.index_at(11.50))
        self.assertEqual(int(np.argmax(speakers.view.levels)), 11)
        point = screen(speakers.view.points(speakers.view.projection())[11]).toPoint()
        QTest.mouseClick(speakers.view, Qt.MouseButton.LeftButton, pos=point)
        self.assertEqual(speakers.selector.currentIndex(), 11)
        self.assertIn("dBFS", speakers.reading.text())
        row = speakers.row
        self.p.volume.setValue(15)
        self.p.refresh()
        self.assertEqual(speakers.row, row)
        self.seek(3.50)
        self.wait(lambda: int(np.argmax(speakers.view.levels)) == LFE)
        self.assertGreater(speakers.lfe_meter.value[0], -22)

    def test_714_reuses_pcm_and_changing_output_keeps_speaker_timeline(self):
        data = self.p.preferences.data.copy()
        data.update(device=self.backend.surround.key, layout="7.1.4")
        self.p.apply_output(data)
        with patch.object(self.p.cache, "decode", wraps=self.p.cache.decode) as decode:
            self.p.open_iamf(EXAMPLE)
            self.wait(lambda: self.p.speakers.timeline is not None)
            timeline = self.p.speakers.timeline
            self.seek(6.52)
            self.wait(lambda: self.p.speakers.row == timeline.index_at(self.p.transport.snapshot.position))
            self.assertEqual([c.args[1] for c in decode.call_args_list], ["7.1.4"])
            data = data.copy()
            data.update(device="", layout="binaural")
            self.p.apply_output(data)
            self.wait(lambda: self.p.iamf is not None and self.p.iamf.layout == "binaural")
            self.wait(lambda: self.p.transport.snapshot.token == self.p.token)
            self.assertIs(self.p.speakers.timeline, timeline)
            self.assertAlmostEqual(self.p.transport.snapshot.position, 6.52, delta=0.025)
            self.assertEqual([c.args[1] for c in decode.call_args_list], ["7.1.4", "binaural"])


if __name__ == "__main__":
    unittest.main()
