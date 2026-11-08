"""Compact, file-only level timelines for a decoded 7.1.4 render."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np
import soundfile as sf

from .playback_render import LAYOUTS
from .tools import StemSpaceError

LAYOUT = "7.1.4"
WINDOW_SECONDS = 0.05
FLOOR_DB = -90.0


@dataclass(frozen=True)
class Speaker:
    name: str
    code: str
    direction: tuple[int, int] | None


# Index in the decoded WAV, not the order of StemSpace's FFmpeg export bed.
# Keep side/rear names explicit; LFE has no scene position.
GEOMETRY = {
    "Front left": ("FL", (-30, 0)),
    "Front right": ("FR", (30, 0)),
    "Center": ("C", (0, 0)),
    "LFE": ("LFE", None),
    "Side left": ("SL", (-90, 0)),
    "Side right": ("SR", (90, 0)),
    "Rear left": ("RL", (-150, 0)),
    "Rear right": ("RR", (150, 0)),
    "Top front left": ("TFL", (-35, 45)),
    "Top front right": ("TFR", (35, 45)),
    "Top rear left": ("TRL", (-145, 45)),
    "Top rear right": ("TRR", (145, 45)),
}
SPEAKERS = tuple(Speaker(name, *GEOMETRY[name]) for name in LAYOUTS[LAYOUT].speakers)
LFE = next(i for i, speaker in enumerate(SPEAKERS) if speaker.direction is None)


class AnalysisCancelled(Exception):
    pass


@dataclass(frozen=True)
class SpeakerLevels:
    sample_rate: int
    window_frames: int
    rms: np.ndarray
    peak: np.ndarray

    def index_at(self, seconds: float) -> int:
        if not math.isfinite(seconds) or seconds <= 0:
            return 0
        return min(len(self.rms) - 1, int(seconds * self.sample_rate / self.window_frames))


def dbfs(amplitude: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(amplitude, 10 ** (FLOOR_DB / 20)))


def read_speaker_levels(path: Path, cancel: Event | None = None) -> SpeakerLevels:
    """Analyze PCM in bounded batches; keep only RMS and sample peak per window."""
    with sf.SoundFile(path) as wav:
        if wav.channels != len(SPEAKERS) or not wav.frames:
            raise StemSpaceError("Speaker activity requires nonempty 7.1.4 PCM.")
        size = max(1, round(wav.samplerate * WINDOW_SECONDS))
        windows = (wav.frames + size - 1) // size
        rms = np.empty((windows, wav.channels), np.float32)
        peak = np.empty_like(rms)
        row = 0
        while row < windows:
            if cancel is not None and cancel.is_set():
                raise AnalysisCancelled
            audio = wav.read(size * 48, dtype="float32", always_2d=True)
            expected = min(size * 48, wav.frames - row * size)
            if len(audio) != expected or not np.isfinite(audio).all():
                raise StemSpaceError("The decoded speaker signal has missing or invalid samples.")
            full, partial = divmod(len(audio), size)
            if full:
                blocks = audio[:full * size].reshape(full, size, wav.channels)
                rms[row:row + full] = dbfs(np.sqrt(np.mean(blocks * blocks, axis=1)))
                peak[row:row + full] = dbfs(np.max(np.abs(blocks), axis=1))
                row += full
            if partial:
                tail = audio[-partial:]
                rms[row] = dbfs(np.sqrt(np.mean(tail * tail, axis=0)))
                peak[row] = dbfs(np.max(np.abs(tail), axis=0))
                row += 1
        rms.setflags(write=False)
        peak.setflags(write=False)
        return SpeakerLevels(wav.samplerate, size, rms, peak)
