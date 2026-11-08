import math
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from stemspace.audio import reconstruct
from stemspace.cascade import load_recipe, separate
from stemspace.export import export_iamf, render_speaker_bed, speaker_gains
from stemspace.project import Placement, load_project
from stemspace.tools import StemSpaceError


class FixtureSeparator:
    """Exercise the real cascade runner without an Apple GPU."""
    identity = "fixture-separator-v1"

    def __init__(self):
        self.calls = []
        self.drums = None

    def separate(self, source, stage, output, progress):
        self.calls.append(stage.id)
        audio, rate = sf.read(source, always_2d=True, dtype="float32")
        for i, name in enumerate(stage.outputs.values(), start=1):
            if stage.id == "drumsep":
                np.testing.assert_array_equal(audio, self.drums)
            stem = audio * (i / 25)
            if name == "drums":
                self.drums = stem.copy()
            sf.write(output / f"{name}.wav", stem, rate, subtype="FLOAT")


class PanningTest(unittest.TestCase):
    def test_front_rear_overhead_and_equal_power(self):
        self.assertEqual(speaker_gains(0, 0).argmax(), 2)      # Front center.
        self.assertEqual(speaker_gains(150, 0).argmax(), 5)    # Rear right.
        self.assertEqual(speaker_gains(90, 0).argmax(), 7)     # Side right.
        self.assertEqual(speaker_gains(145, 45).argmax(), 11)  # Top rear right.
        np.testing.assert_allclose(speaker_gains(90, 90)[8:], [0.5] * 4)
        for az in (-180, -170, -67, 0, 66, 170, 180):
            for el in (0, 23, 45, 62, 90):
                gains = speaker_gains(az, el)
                self.assertEqual(gains[3], 0)                # No direct LFE feed.
                self.assertAlmostEqual(np.linalg.norm(gains), 1)
        np.testing.assert_allclose(speaker_gains(-180, 25), speaker_gains(180, 25))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs FFmpeg")
class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.wav"
        # 48 kHz mono proves both channel duplication and a sample-rate round trip.
        t = np.arange(48003) / 48000
        sf.write(self.source, .17 * np.sin(2 * math.pi * 303 * t), 48000, subtype="FLOAT")
        self.backend = FixtureSeparator()
        self.recipe = load_recipe()

    def test_cascade_cache_and_project_reconstruction(self):
        directory = self.root / "project"
        project = separate(self.source, directory, self.recipe, backend=self.backend)
        self.assertEqual(self.backend.calls, ["roformer", "drumsep"])
        self.assertEqual([track.id.rsplit(".", 1)[-1] for track in project.tracks],
                         ["bass", "kick", "snare", "toms", "hh", "ride", "crash",
                          "vocals", "guitar", "piano", "other"])
        self.assertEqual((project.sample_rate, project.frames), (48000, 48003))
        reference, _ = sf.read(project.audio_path(project.reference))
        original, _ = sf.read(self.source)
        np.testing.assert_array_equal(reference[:, 0], original)
        np.testing.assert_array_equal(reference[:, 1], original)
        self.assertGreater(project.reconstruction["peak_without_residual"], 0.01)
        self.assertLess(project.reconstruction["peak_error"], 1e-7)

        project.tracks[1].placement = Placement(144, 46)
        project.save()
        reopened = separate(self.source, directory, self.recipe, backend=self.backend)
        self.assertEqual(reopened.tracks[1].placement, Placement(144, 46))
        self.assertEqual(self.backend.calls, ["roformer", "drumsep"])

        # Changing the child recipe reruns only the child and rebuilds the scene.
        changed = replace(self.recipe, stages=(self.recipe.stages[0],
                                               replace(self.recipe.stages[1], revision="alternate")))
        new = separate(self.source, directory, changed, backend=self.backend)
        self.assertEqual(self.backend.calls, ["roformer", "drumsep", "drumsep"])
        self.assertEqual(load_project(new.path).tracks[1].placement, Placement(144, 46))

    def test_render_moves_the_same_stem_and_encodes_iamf(self):
        project = separate(self.source, self.root / "project", self.recipe, backend=self.backend)
        for track in project.tracks:
            track.placement = Placement(muted=True)
        project.tracks[1].placement = Placement(0, 0)
        bed = self.root / "speakers.wav"
        render_speaker_bed(project, bed)
        front, rate = sf.read(bed)
        self.assertEqual(rate, 48000)
        self.assertEqual(np.nonzero(np.abs(front).max(axis=0))[0].tolist(), [2])

        project.tracks[1].placement = Placement(145, 45)
        render_speaker_bed(project, bed)
        rear_top, _ = sf.read(bed)
        self.assertEqual(np.nonzero(np.abs(rear_top).max(axis=0))[0].tolist(), [11])
        np.testing.assert_array_equal(front[:, 2], rear_top[:, 11])

        target = self.root / "scene.iamf"
        export_iamf(project, target)
        original_bytes = target.read_bytes()
        self.assertTrue(original_bytes.startswith(b"\xf8\x06iamf"))
        self.assertGreater(len(original_bytes), 1000)
        project.tracks[1].placement = Placement(0, 0)
        with self.assertRaisesRegex(StemSpaceError, "already exists"):
            export_iamf(project, target)
        self.assertEqual(target.read_bytes(), original_bytes)
        export_iamf(project, target, overwrite=True)
        self.assertNotEqual(target.read_bytes(), original_bytes)
        self.assertEqual(self.backend.calls, ["roformer", "drumsep"])

    def test_check_detects_modified_residual(self):
        project = separate(self.source, self.root / "project", self.recipe, backend=self.backend)
        residual = project.audio_path(project.residual)
        samples, rate = sf.read(residual, dtype="float32")
        samples[0] += 0.01
        sf.write(residual, samples, rate, subtype="FLOAT")
        report = reconstruct(project.audio_path(project.reference),
                             [project.audio_path(t.file) for t in project.tracks],
                             residual, create=False)
        self.assertFalse(report["passed"])
        self.assertGreater(report["peak_error"], 0.009)


if __name__ == "__main__":
    unittest.main()
