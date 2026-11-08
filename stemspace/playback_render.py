"""Playback layouts and the live 7.1.4-to-monitor render."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .tools import StemSpaceError

MONITOR_RATE = 48000
BLOCK = 512
VENDOR = Path(__file__).with_name("vendor")
FRONT = ("Front left", "Front right", "Center", "LFE")
SIDE = ("Side left", "Side right")
BACK = ("Rear left", "Rear right")
HEIGHT = ("Top front left", "Top front right", "Top rear left", "Top rear right")


@dataclass(frozen=True)
class Layout:
    id: str
    name: str
    decoder_id: int
    speakers: tuple[str, ...]

    @property
    def channels(self) -> int:
        return len(self.speakers)


# The WASM decoder's channelOrdering=0 (IAMF) puts sides before rears. The
# exporter uses FFmpeg's order (rears before sides). The measured preview
# transforms include this permutation, so both sources enter routing alike.
LAYOUTS = {
    item.id: item for item in (
        Layout("binaural", "Binaural headphones", 14, ("Left ear", "Right ear")),
        Layout("stereo", "Stereo speakers", 0, FRONT[:2]),
        Layout("5.1", "5.1 speakers", 1, FRONT + ("Surround left", "Surround right")),
        Layout("7.1", "7.1 speakers", 8, FRONT + SIDE + BACK),
        Layout("7.1.4", "7.1.4 speakers", 9, FRONT + SIDE + BACK + HEIGHT),
    )
}


@lru_cache(maxsize=1)
def filters() -> dict[str, np.ndarray]:
    with np.load(VENDOR / "monitor-filters.npz", allow_pickle=False) as data:
        result = {layout: data[layout] for layout in LAYOUTS}
    for key, filt in result.items():
        if (filt.ndim != 3 or filt.shape[1:] != (12, LAYOUTS[key].channels)
                or not np.isfinite(filt).all()):
            raise StemSpaceError(f"The packaged monitor filter for {key} is invalid.")
    return result


class MonitorRenderer:
    """AOMedia speaker render measured from lossless 7.1.4 impulses.

    The speaker transforms are instantaneous. Headphones use a partitioned
    convolution and carry the response tail across successive blocks.
    """

    def __init__(self, layout: str):
        self.impulse = filters()[layout]
        self.channels = self.impulse.shape[2]
        self.matrix = self.impulse[0] if len(self.impulse) == 1 else None
        if self.matrix is None:
            partitions = (len(self.impulse) + BLOCK - 1) // BLOCK
            padded = np.zeros((partitions * BLOCK, 12, self.channels), dtype=np.float32)
            padded[:len(self.impulse)] = self.impulse
            bank = padded.reshape(partitions, BLOCK, 12, self.channels)
            bank = np.fft.rfft(bank, n=2 * BLOCK, axis=1).astype(np.complex64)
            self.history = np.zeros(bank.shape[:3], dtype=np.complex64)
            # Each frequency now has a contiguous dot product over speakers
            # and partitions instead of striding through the filter bank.
            self.bank = np.ascontiguousarray(bank.transpose(1, 3, 0, 2)).reshape(
                BLOCK + 1, self.channels, partitions * 12
            )
            self.tail = np.zeros((BLOCK, self.channels), dtype=np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim != 2 or audio.shape[1] != 12 or len(audio) > BLOCK:
            raise ValueError("Monitor renderer expects up to one 12-channel block.")
        if self.matrix is not None:
            return np.einsum("ni,io->no", audio, self.matrix, optimize=False)
        self.history[1:] = self.history[:-1]
        self.history[0] = np.fft.rfft(audio, n=2 * BLOCK, axis=0)
        history = np.ascontiguousarray(self.history.transpose(1, 0, 2)).reshape(BLOCK + 1, -1)
        spectrum = np.einsum("fok,fk->fo", self.bank, history, optimize=False)
        block = np.fft.irfft(spectrum, n=2 * BLOCK, axis=0).astype(np.float32, copy=False)
        block[:BLOCK] += self.tail
        self.tail[:] = block[BLOCK:]
        return block[:len(audio)]
