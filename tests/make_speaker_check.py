"""Generate a file with one isolated burst per 7.1.4 speaker."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from stemspace.export import EXPORT_RATE, encode_speaker_bed

# Play in IAMF decoder order. Write these positions into the FFmpeg WAV.
CHANNELS = (0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11)


def make_check(target: Path) -> None:
    rate = EXPORT_RATE
    bed = np.zeros((len(CHANNELS) * rate, len(CHANNELS)), dtype=np.float32)
    n = rate // 2
    time = np.arange(n) / rate
    envelope = np.ones(n)
    fade = rate // 40
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    for index, channel in enumerate(CHANNELS):
        start = index * rate + rate // 4
        frequency = 55 if channel == 3 else 430
        bed[start:start + n, channel] = 0.25 * np.sin(2 * np.pi * frequency * time) * envelope
    with TemporaryDirectory() as tmp:
        wav = Path(tmp) / "bed.wav"
        sf.write(wav, bed, rate, subtype="FLOAT", format="RF64")
        encode_speaker_bed(wav, target)


if __name__ == "__main__":
    make_check(Path(sys.argv[1]).resolve())
