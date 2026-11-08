"""Reproduce the live monitor filters from the packaged AOMedia WASM decoder.

Usage: python tests/rebuild_monitor_filters.py /tmp/monitor-filters.npz
Requires Node and an FFmpeg build with raw IAMF muxing and FLAC encoding.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from stemspace.export import MAPPING
from stemspace.iamf_decode import DecodeCache
from stemspace.playback_render import LAYOUTS, MONITOR_RATE
from stemspace.tools import ffmpeg, run

SLOT = 16384
OFFSET = 2048
LEVEL = 0.75


def encode_lossless(audio: np.ndarray, directory: Path) -> Path:
    bed, target = directory / "probe.wav", directory / "probe.iamf"
    sf.write(bed, audio, MONITOR_RATE, subtype="FLOAT", format="RF64")
    filters = ["[0:a]asplit=7" + "".join(f"[s{i}]" for i in range(7))]
    filters.extend(f'[s{i}]channelmap={ch}:{"stereo" if "|" in ch else "mono"}[{name}]'
                   for i, (name, ch) in enumerate(MAPPING))
    element = ("type=iamf_audio_element:id=1:" + ":".join(f"st={i}" for i in range(7))
               + ":audio_element_type=channel,layer=ch_layout=7.1.4")
    presentation = (
        "type=iamf_mix_presentation:id=3:stg=0:annotations=en-us=Calibration,"
        "submix=parameter_id=100:parameter_rate=48000:default_mix_gain=0.0"
        "|element=stg=0:headphones_rendering_mode=binaural:annotations=en-us=Speaker mix:"
        "parameter_id=101:parameter_rate=48000:default_mix_gain=0.0"
        "|layout=sound_system=stereo:integrated_loudness=-24:digital_peak=-2"
        "|layout=sound_system=7.1.4:integrated_loudness=-24:digital_peak=-2"
    )
    args = ffmpeg("-loglevel", "error", "-y", "-ch_layout", "7.1.4", "-i", str(bed),
                  "-filter_complex", ";".join(filters))
    for name, _ in MAPPING:
        args.extend(("-map", f"[{name}]"))
    args.extend(("-stream_group", element, "-stream_group", presentation))
    for i in range(7):
        args.extend(("-streamid", f"{i}:{i}"))
    # Use a power-of-two FLAC frame size supported by the binaural WASM path.
    run([*args, "-c:a", "flac", "-frame_size", "1024", "-f", "iamf", str(target)],
        "Encode lossless calibration")
    return target


def rebuild(output: Path) -> None:
    inputs = np.zeros((12 * SLOT, 12), dtype=np.float32)
    for channel in range(12):
        inputs[channel * SLOT + OFFSET, channel] = LEVEL
    cache = DecodeCache()
    try:
        with TemporaryDirectory() as tmp:
            iamf = encode_lossless(inputs, Path(tmp))
            data = {}
            for layout, spec in LAYOUTS.items():
                audio, rate = sf.read(cache.decode(iamf, layout), always_2d=True, dtype="float32")
                if rate != MONITOR_RATE or audio.shape != (len(inputs), spec.channels):
                    raise ValueError(f"Unexpected calibration PCM for {layout}.")
                length = 8192 if layout == "binaural" else 1
                filt = np.empty((length, 12, spec.channels), dtype=np.float32)
                for c in range(12):
                    start = c * SLOT + OFFSET
                    if np.any(audio[start-512:start]) or np.any(audio[start+length:start+length+512]):
                        raise ValueError(f"Unexpected calibration timing for {layout}, speaker {c}.")
                    filt[:, c, :] = audio[start:start+length] / LEVEL
                data[layout] = filt
            np.savez_compressed(output, **data)
    finally:
        cache.close()


if __name__ == "__main__":
    rebuild(Path(sys.argv[1]))
