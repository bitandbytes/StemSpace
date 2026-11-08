from __future__ import annotations

import json
import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import soundfile as sf

from .tools import StemSpaceError, executable, ffmpeg, run

BLOCK = 65536
RECONSTRUCTION_TOLERANCE = 1e-6


def canonical_wav(source: Path, target: Path) -> None:
    """Decode the first stream at its original sample rate into stereo float WAV."""
    probe = executable("ffprobe", "STEMSPACE_FFPROBE")
    response = run([probe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                    "stream=channels", "-of", "json", str(source)], "Inspect source audio")
    streams = json.loads(response).get("streams", [])
    channels = streams[0].get("channels") if streams else None
    if channels not in (1, 2):
        raise StemSpaceError("Choose an audio file with a mono or stereo first audio stream.")
    filters = ["-af", "pan=stereo|c0=c0|c1=c0"] if channels == 1 else []
    run(ffmpeg("-loglevel", "error", "-y", "-i", str(source), "-map", "0:a:0",
               *filters, "-c:a", "pcm_f32le", "-rf64", "auto", str(target)), "Decode source audio")
    info = sf.info(target)
    if not info.frames or info.channels != 2:
        raise StemSpaceError("The source did not decode to a nonempty stereo recording.")


def convert_wav(source: Path, target: Path, rate: int) -> None:
    info = sf.info(source)
    if not info.frames or info.channels not in (1, 2):
        raise StemSpaceError(f"Expected a nonempty mono or stereo WAV: {source.name}")
    if info.channels == 2 and info.samplerate == rate and info.subtype == "FLOAT":
        import shutil
        shutil.copyfile(source, target)
        return
    filters = ["-af", "pan=stereo|c0=c0|c1=c0"] if info.channels == 1 else []
    run(ffmpeg("-loglevel", "error", "-y", "-i", str(source), *filters,
               "-ar", str(rate), "-c:a", "pcm_f32le", "-rf64", "auto", str(target)),
        f"Prepare {source.name} at {rate} Hz")


def read_exact(reader: sf.SoundFile, frames: int, name: str) -> np.ndarray:
    data = reader.read(frames, dtype="float64", always_2d=True)
    if len(data) != frames:
        raise StemSpaceError(f"Unexpected end of {name}. Run separation again.")
    if not np.isfinite(data).all():
        raise StemSpaceError(f"{name} contains non-finite audio samples.")
    return data


def align_wav(source: Path, target: Path, frames: int, rate: int) -> None:
    source_rate = sf.info(source).samplerate
    tmp = target.with_name(f"{target.stem}.resampled.wav")
    convert_wav(source, tmp, rate)
    info = sf.info(tmp)
    # A round trip between integer frame clocks can differ at the tail only.
    tolerance = max(2, math.ceil(rate / source_rate) + 1)
    if abs(info.frames - frames) > tolerance:
        raise StemSpaceError(
            f"{source.name} has {info.frames} frames after resampling, expected {frames}. "
            "The separator must preserve the input timeline."
        )
    if info.frames == frames:
        tmp.replace(target)
        return
    with sf.SoundFile(tmp) as reader, sf.SoundFile(
        target, "w", samplerate=rate, channels=2, subtype="FLOAT", format="WAVEX"
    ) as writer:
        remaining = min(info.frames, frames)
        while remaining:
            n = min(BLOCK, remaining)
            writer.write(read_exact(reader, n, source.name))
            remaining -= n
        if frames > info.frames:
            writer.write(np.zeros((frames - info.frames, 2)))
    tmp.unlink()


def check_timeline(reader: sf.SoundFile, rate: int, frames: int, name: str) -> None:
    if (reader.samplerate, reader.frames, reader.channels) != (rate, frames, 2):
        raise StemSpaceError(f"{name} does not match the project's stereo sample timeline.")


def reconstruct(reference: Path, stems: list[Path], residual: Path, *, create: bool) -> dict:
    """Check the on-disk float residual using double-precision accumulation."""
    def process(write: bool) -> tuple[float, float, float]:
        pre_peak = post_peak = square_sum = 0.0
        with ExitStack() as stack:
            original = stack.enter_context(sf.SoundFile(reference))
            rate, total = original.samplerate, original.frames
            if not total or original.channels != 2:
                raise StemSpaceError("The reconstruction reference must be nonempty stereo.")
            readers = [stack.enter_context(sf.SoundFile(p)) for p in stems]
            for reader, path in zip(readers, stems):
                check_timeline(reader, rate, total, path.name)
            if write:
                rest = stack.enter_context(sf.SoundFile(residual, "w", samplerate=rate,
                                             channels=2, subtype="FLOAT", format="WAVEX"))
            else:
                rest = stack.enter_context(sf.SoundFile(residual))
                check_timeline(rest, rate, total, residual.name)
            for start in range(0, total, BLOCK):
                n = min(BLOCK, total - start)
                error = read_exact(original, n, reference.name)
                for reader, path in zip(readers, stems):
                    error -= read_exact(reader, n, path.name)
                pre_peak = max(pre_peak, float(np.max(np.abs(error))))
                if write:
                    rest.write(error)
                else:
                    error -= read_exact(rest, n, residual.name)
                    post_peak = max(post_peak, float(np.max(np.abs(error))))
                    square_sum += float(np.sum(error * error))
        return pre_peak, post_peak, math.sqrt(square_sum / (total * 2))
    if create:
        process(True)
    pre, peak, rms = process(False)
    return {"peak_without_residual": pre, "peak_error": peak, "rms_error": rms,
            "tolerance": RECONSTRUCTION_TOLERANCE, "passed": peak <= RECONSTRUCTION_TOLERANCE}
