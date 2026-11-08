"""Create a small generated placement check, independent of MLX."""
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

from stemspace.audio import reconstruct
from stemspace.export import export_iamf
from stemspace.project import Placement, Project, Track
from stemspace.tools import digest_file


def make_demo(folder: Path, iamf: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rate = 48000
    frames = 4 * rate
    time = np.arange(int(0.65 * rate)) / rate
    rng = np.random.default_rng(23)
    freqs = np.geomspace(210, 10000, 64)
    sound = np.sin(2 * np.pi * time[:, None] * freqs + rng.uniform(0, 2 * np.pi, len(freqs))).sum(axis=1)
    sound *= 0.25 * np.hanning(len(sound)) / np.abs(sound).max()
    tracks = []
    original = np.zeros((frames, 2), dtype="float32")
    for i, (name, position) in enumerate((
        ("Front", Placement(0, 0)),
        ("Rear right", Placement(145, 0)),
        ("High rear right", Placement(145, 45)),
        ("Left", Placement(-90, 0)),
    )):
        stem = np.zeros_like(original)
        start = int((i + 0.18) * rate)
        stem[start:start + len(sound)] = sound[:, None]
        original += stem
        filename = f"track-{i}.wav"
        sf.write(folder / filename, stem, rate, subtype="FLOAT")
        tracks.append(Track(f"demo.{i}", name, filename, position))
    reference = folder / "source.wav"
    sf.write(reference, original, rate, subtype="FLOAT")
    stats = reconstruct(reference, [folder / t.file for t in tracks], folder / "residual.wav", create=True)
    project = Project(folder, str(reference), digest_file(reference), {}, reference.name, "residual.wav",
                      rate, frames, tracks, stats)
    project.save()
    export_iamf(project, iamf, overwrite=True, progress=print)


if __name__ == "__main__":
    make_demo(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
