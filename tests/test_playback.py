from __future__ import annotations

import shutil
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from stemspace.export import render_speaker_bed
from stemspace.iamf_decode import DecodeCache
from stemspace.playback_render import BLOCK, LAYOUTS, MonitorRenderer, filters
from stemspace.project import Placement, Project, Track
from stemspace.tools import StemSpaceError

try:
    from stemspace.playback import (
        RENDER_AHEAD_SECONDS, Controls, Device, Output, StemSource, Transport,
        WavSource, render_blocks, route, validate_mapping,
    )
except ModuleNotFoundError as exc:
    if exc.name != "soxr":
        raise
    Output = None

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "position-check.iamf"


class RenderingTest(unittest.TestCase):
    def test_aom_channel_order_does_not_swap_side_rear_or_height(self):
        impulse = np.eye(12, dtype=np.float32)
        surround = MonitorRenderer("7.1.4").process(impulse)
        np.testing.assert_array_equal(surround.argmax(axis=1), [0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11])
        np.testing.assert_array_equal(surround.max(axis=1), 1)
        self.assertFalse(np.array_equal(MonitorRenderer("7.1").process(impulse)[4],
                                        MonitorRenderer("7.1").process(impulse)[6]))

    def test_streaming_binaural_matches_fir_across_block_boundaries(self):
        src = np.zeros((4 * BLOCK, 12), np.float32)
        src[BLOCK - 19, 5] = 0.3    # Right rear.
        src[2 * BLOCK + 27, 11] = 0.4  # Same side, elevated.
        renderer = MonitorRenderer("binaural")
        actual = np.concatenate([renderer.process(src[i:i+BLOCK]) for i in range(0, len(src), BLOCK)])
        impulse = filters()["binaural"]
        expected = np.stack([
            sum(np.convolve(src[:, ch], impulse[:, ch, ear])[:len(src)] for ch in (5, 11))
            for ear in range(2)
        ], axis=1)
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=1e-5)
        self.assertGreater(np.max(np.abs(impulse[:, 5] - impulse[:, 11])), 0.05)

    @unittest.skipUnless(shutil.which("node"), "needs Node for AOMedia WASM")
    def test_real_iamf_decodes_to_every_exposed_layout(self):
        cache = DecodeCache()
        self.addCleanup(cache.close)
        for key, layout in LAYOUTS.items():
            with self.subTest(layout=key):
                decoded = cache.decode(EXAMPLE, key)
                self.assertEqual(cache.decode(EXAMPLE, key), decoded)
                audio, sr = sf.read(decoded, always_2d=True)
                self.assertEqual((sr, audio.shape[1]), (48000, layout.channels))
                self.assertAlmostEqual(len(audio) / sr, 4, delta=0.06)
                self.assertGreater(np.max(np.abs(audio)), 0.03)
                if key == "7.1.4":
                    # Demo's elevated-right event should feed top rear right.
                    peaks = np.max(np.abs(audio[2*sr+11000:2*sr+36000]), axis=0)
                    self.assertEqual(peaks.argmax(), layout.speakers.index("Top rear right"))
        self.assertNotEqual(cache.decode(EXAMPLE, "stereo"), cache.decode(EXAMPLE, "binaural"))


class FakeStream:
    latency = 0

    def __init__(self, backend, output):
        self.backend, self.output = backend, output
        self.blocks = []

    def start(self):
        if self.backend.active:
            raise AssertionError("Two output streams are active")
        self.backend.active = self

    def write(self, data):
        assert data.shape[1] == self.output.channels and data.dtype == np.float32
        self.blocks.append(data.copy())
        time.sleep(0.001)
        return len(self.blocks) in self.backend.underrun_writes

    def stop(self):
        self.backend.active = None

    abort = stop

    def close(self):
        self.backend.active = None


class FakeBackend:
    stereo = None
    surround = None

    def __init__(self, rate=48000):
        self.active = None
        self.streams = []
        self.underrun_writes = set()
        self.rate = rate
        self.stereo = Device("Headset", 0, "Headset", 2, rate)
        self.surround = Device("Interface", 1, "Interface", 12, rate)

    def devices(self):
        return [self.stereo, self.surround], 0

    def configure(self, device, layout, mapping):
        validate_mapping(mapping, layout, device.channels)
        return Output(device, layout, mapping, self.rate)

    def open(self, output):
        result = FakeStream(self, output)
        self.streams.append(result)
        return result


class SyntheticSource:
    live = False
    rate, frames = 48000, 96000

    def __init__(self, gates=None):
        self.gates = gates or {}
        self.opened = []

    def open(self, position):
        self.opened.append(position)
        return SyntheticReader(self, position)


class SyntheticReader:
    channels = 2

    def __init__(self, source, position):
        self.source = source
        self.rate, self.frames, self.position = source.rate, source.frames, position
        self.calls = 0

    def read(self, count):
        self.calls += 1
        if self.calls in self.source.gates:
            entered, release = self.source.gates[self.calls]
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Render test did not release the reader")
        start = self.position
        count = min(count, self.frames - start)
        self.position += count
        # Gives an unambiguous value before and after a seek.
        value = 0.1 if start < 24000 else 0.6
        return np.full((count, self.channels), value, dtype=np.float32)

    def close(self):
        pass


@unittest.skipIf(Output is None, "install the UI extra to test PCM playback")
class PlaybackTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def project(self, rate=48000):
        frames = rate
        bass = np.full((frames, 2), 0.1, np.float32)
        ride = np.repeat(np.linspace(0, 0.5, frames, dtype=np.float32)[:, None], 2, axis=1)
        for filename, audio in (("bass.wav", bass), ("ride.wav", ride),
                                ("original.wav", bass + ride), ("residual.wav", bass * 0)):
            sf.write(self.root / filename, audio, rate, subtype="FLOAT")
        return Project(self.root, "recording.wav", "", {}, "original.wav", "residual.wav", rate, frames,
                       [Track("bass", "Bass", "bass.wav", Placement(0, 0)),
                        Track("ride", "Ride", "ride.wav", Placement(145, 45))], {"peak_error": 0})

    def test_listen_solo_keeps_ride_on_timeline_and_preserves_export_state(self):
        project = self.project()
        project.save()
        saved = project.path.read_bytes()
        src = StemSource(project)
        src.set_controls(Controls(src.controls.placements, (True, False), False))
        reader = src.open(5000)
        self.addCleanup(reader.close)
        self.assertFalse(reader.read(300)[:, 11].any())
        src.set_controls(Controls(src.controls.placements, (False, True), False))
        fading = reader.read(960)
        self.assertGreater(fading[0, 2], fading[-1, 2])
        self.assertGreater(fading[-1, 11], fading[0, 11])
        block = reader.read(40)
        at = 5000 + 300 + 960
        self.assertAlmostEqual(float(block[0, 11]), 0.5 * at / (project.frames - 1), places=6)
        self.assertTrue(np.all(block[:, 2] == 0))
        project.save()
        self.assertEqual(project.path.read_bytes(), saved)
        exported = self.root / "export-bed.wav"
        render_speaker_bed(project, exported)
        saved_mix, _ = sf.read(exported, always_2d=True)
        self.assertGreater(saved_mix[:, 2].max(), 0.09)  # Bass is still in export.
        self.assertGreater(saved_mix[:, 11].max(), 0.49)

    def test_playback_resamples_and_routes_top_rear(self):
        src = StemSource(self.project(rate=44100))
        src.set_controls(replace(src.controls, listen=(False, True)))
        reader = src.open(0)
        self.addCleanup(reader.close)
        backend = FakeBackend(44100)
        output = backend.configure(backend.surround, "7.1.4", tuple(range(12, 0, -1)))
        audio = np.concatenate([route(block, output) for block in render_blocks(reader, src, output)])
        self.assertLessEqual(abs(len(audio) - 44100), 1)
        self.assertEqual(np.abs(audio).max(axis=0).argmax(), 0)
        self.assertGreater(np.abs(audio[:, 0]).max(), 0.45)
        self.assertFalse(np.any(audio[:, 1:]))

    def test_routing_allows_gaps_and_rejects_duplicates(self):
        backend = FakeBackend()
        output = backend.configure(backend.surround, "stereo", (3, 8))
        actual = route(np.array([[0.1, 0.2]], np.float32), output)
        self.assertEqual(actual.shape[1], 8)
        np.testing.assert_allclose(actual[0], [0, 0, 0.1, 0, 0, 0, 0, 0.2])
        with self.assertRaisesRegex(StemSpaceError, "different"):
            backend.configure(backend.surround, "stereo", (3, 3))

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.005)
        self.fail("Playback worker did not reach the requested state.")

    def test_transport_switches_sources_and_test_tone_only_uses_mapped_output(self):
        backend = FakeBackend()
        transport = Transport(backend)
        self.addCleanup(transport.close)
        project = self.project()
        src = StemSource(project)
        transport.load(src, 0.2, 7)
        self.wait_for(lambda: transport.snapshot.token == 7)
        stereo = backend.configure(backend.stereo, "stereo", (1, 2))
        transport.play(stereo)
        self.wait_for(lambda: transport.snapshot.position > 0.22)
        transport.pause()
        self.wait_for(lambda: not transport.snapshot.playing and backend.active is None)
        self.assertGreater(transport.snapshot.position, 0.2)

        recorded = WavSource(self.root / "original.wav", "stereo")
        transport.load(recorded, 0.4, 8)
        self.wait_for(lambda: transport.snapshot.token == 8)
        self.assertEqual(transport.snapshot.position, 0.4)
        transport.play(stereo)
        self.wait_for(lambda: transport.snapshot.position > 0.42)
        transport.seek(0.7)
        self.wait_for(lambda: transport.snapshot.position >= 0.7)
        self.assertTrue(transport.snapshot.playing)

        output = backend.configure(backend.surround, "stereo", (7, 9))
        transport.test_speaker(output, 1)
        self.wait_for(lambda: backend.streams[-1].output == output
                      and bool(backend.streams[-1].blocks) and backend.active is None)
        tone = np.concatenate(backend.streams[-1].blocks)
        self.assertEqual(np.abs(tone).max(axis=0).argmax(), 8)
        self.assertFalse(np.any(tone[:, :8]))
        self.assertFalse(transport.snapshot.playing)
        self.assertIsNone(transport.snapshot.error)

    def test_writer_keeps_feeding_device_during_a_render_stall(self):
        backend = FakeBackend()
        transport = Transport(backend)
        self.addCleanup(transport.close)
        opening, release_open = threading.Event(), threading.Event()
        stalled, release_stall = threading.Event(), threading.Event()
        self.addCleanup(release_open.set)
        self.addCleanup(release_stall.set)
        capacity = max(2, int(np.ceil(backend.rate * RENDER_AHEAD_SECONDS / BLOCK)))
        source = SyntheticSource({1: (opening, release_open), capacity + 1: (stalled, release_stall)})
        transport.load(source, 0, 9)
        transport.play(backend.configure(backend.stereo, "stereo", (1, 2)))
        self.assertTrue(opening.wait(timeout=2))
        self.assertTrue(transport.snapshot.buffering)
        self.assertEqual(backend.streams, [])  # Don't start a device with nothing to play.
        release_open.set()
        self.assertTrue(stalled.wait(timeout=2))
        self.wait_for(lambda: bool(backend.streams) and len(backend.streams[0].blocks) >= 3)
        self.assertFalse(release_stall.is_set())
        self.assertFalse(transport.snapshot.buffering)
        self.assertEqual(transport.snapshot.underruns, 0)
        release_stall.set()

    def test_seek_flushes_render_ahead_and_counts_driver_underruns(self):
        backend = FakeBackend()
        backend.underrun_writes = {2}
        transport = Transport(backend)
        self.addCleanup(transport.close)
        source = SyntheticSource()
        transport.load(source, 0, 10)
        transport.play(backend.configure(backend.stereo, "stereo", (1, 2)))
        self.wait_for(lambda: transport.snapshot.underruns == 1)
        transport.seek(0.5)
        self.wait_for(lambda: len(backend.streams) == 2 and len(backend.streams[-1].blocks) >= 3)
        self.assertEqual(source.opened, [0, 24000])
        self.assertGreater(np.max(backend.streams[-1].blocks[0]), 0.15)
        self.assertLess(np.max(backend.streams[0].blocks[0]), 0.07)
        self.assertEqual(transport.snapshot.underruns, 1)  # Count from the new stream only.


if __name__ == "__main__":
    unittest.main()
