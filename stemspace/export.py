from __future__ import annotations

import math
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from .audio import BLOCK, check_timeline, read_exact
from .project import Placement, Project
from .tools import Progress, StemSpaceError, ffmpeg, quiet, run

EXPORT_RATE = 48000
# FFmpeg channel order: FL FR FC LFE BL BR SL SR TFL TFR TBL TBR.
# Listener coordinates: positive azimuth turns right; elevation increases upward.
EAR_RING = ((-150, 4), (-90, 6), (-30, 0), (0, 2), (30, 1), (90, 7), (150, 5))
HEIGHT_RING = ((-145, 10), (-35, 8), (35, 9), (145, 11))
MAPPING = (("front", "0|1"), ("side", "6|7"), ("back", "4|5"), ("top_front", "8|9"),
           ("top_back", "10|11"), ("center", "2"), ("lfe", "3"))
LAYOUT = "7.1.4"


def ring_gains(azimuth: float, speakers: tuple[tuple[int, int], ...]) -> np.ndarray:
    start = speakers[0][0]
    angle = (azimuth - start) % 360 + start
    closed = (*speakers, (start + 360, speakers[0][1]))
    result = np.zeros(12)
    for (a, i), (b, j) in zip(closed, closed[1:]):
        if a <= angle <= b:
            if angle == a:
                result[i] = 1
            elif angle == b:
                result[j] = 1
            else:
                x = (angle - a) / (b - a) * math.pi / 2
                result[i], result[j] = math.cos(x), math.sin(x)
            return result
    raise StemSpaceError("Invalid speaker azimuth.")


def speaker_gains(azimuth: float, elevation: float) -> np.ndarray:
    """Equal-power panning between adjacent speakers and the two height planes."""
    ear, top = ring_gains(azimuth, EAR_RING), ring_gains(azimuth, HEIGHT_RING)
    if elevation <= 45:
        angle = elevation / 45 * math.pi / 2
        if elevation == 45:
            return top
        return math.cos(angle) * ear + math.sin(angle) * top
    ceiling = np.zeros(12)
    ceiling[[8, 9, 10, 11]] = 0.5
    ratio = (elevation - 45) / 45
    result = (1 - ratio) * top + ratio * ceiling
    return result / np.linalg.norm(result)


def stereo_matrix(position: Placement) -> np.ndarray:
    """Map stereo to a point or to a pair around its center azimuth."""
    az, el = position.azimuth, position.elevation
    if position.width == 0:
        matrix = np.stack([speaker_gains(az, el)] * 2) * 0.5
    else:
        offset = position.width * 0.5
        matrix = np.stack([speaker_gains(az - offset, el), speaker_gains(az + offset, el)])
        # Maintain center-level gain as the pair narrows and energy when the pair spreads.
        matrix /= np.linalg.norm(matrix.sum(axis=0))
    return matrix * (10 ** (position.gain_db / 20))


def render_speaker_bed(project: Project, output: Path) -> None:
    selected = [(track.file, track.placement) for track in project.tracks
                if not track.placement.muted]
    if project.include_residual:
        selected.append((project.residual, Placement(width=90)))
    matrices = [stereo_matrix(position) for _, position in selected]
    with ExitStack() as stack:
        readers = [stack.enter_context(sf.SoundFile(project.audio_path(name))) for name, _ in selected]
        for reader, (name, _) in zip(readers, selected):
            check_timeline(reader, project.sample_rate, project.frames, name)
        writer = stack.enter_context(sf.SoundFile(
            output, "w", channels=12, samplerate=project.sample_rate, subtype="FLOAT", format="RF64"
        ))
        for start in range(0, project.frames, BLOCK):
            n = min(BLOCK, project.frames - start)
            mixed = np.zeros((n, 12))
            for reader, matrix, (name, _) in zip(readers, matrices, selected):
                mixed += read_exact(reader, n, name) @ matrix
            writer.write(mixed)


@dataclass(frozen=True)
class Loudness:
    integrated: float
    digital_peak: float


def measure(bed: Path, *, stereo: bool) -> Loudness:
    resample = f"aresample={EXPORT_RATE}" + (":ochl=stereo" if stereo else "")
    result = run(ffmpeg("-nostats", "-loglevel", "info", "-ch_layout", LAYOUT, "-i", str(bed),
                        "-af", f"{resample},ebur128=peak=sample:framelog=verbose",
                        "-f", "null", "-"), "Measure export loudness")
    summary = result.rsplit("Summary:", 1)[-1]
    integrated = re.search(r"\bI:\s*(-?inf|-?\d+(?:\.\d+)?)\s+LUFS", summary)
    peak = re.search(r"\bPeak:\s*(-?inf|-?\d+(?:\.\d+)?)\s+dBFS", summary)
    if not integrated or not peak:
        raise StemSpaceError("FFmpeg did not report integrated loudness and sample peak.")
    return Loudness(float(integrated[1]), float(peak[1]))


def mix_layout(system: str, stats: Loudness, attenuation: float) -> str:
    loudness = max(-120, min(0, stats.integrated + attenuation))
    peak = max(-120, min(0, stats.digital_peak + attenuation))
    return f"layout=sound_system={system}:integrated_loudness={loudness:.2f}:digital_peak={peak:.2f}"


def encode_speaker_bed(bed: Path, target: Path) -> float:
    surround, stereo = measure(bed, stereo=False), measure(bed, stereo=True)
    attenuation = min(0.0, -2.0 - max(surround.digital_peak, stereo.digital_peak))
    filters = [f"[0:a]aresample={EXPORT_RATE},volume={attenuation:.2f}dB,"
               f"asplit={len(MAPPING)}" + "".join(f"[src{i}]" for i in range(len(MAPPING)))]
    filters.extend(f"[src{i}]channelmap={channels}:{'stereo' if '|' in channels else 'mono'}[{label}]"
                   for i, (label, channels) in enumerate(MAPPING))
    element = ("type=iamf_audio_element:id=1:"
               + ":".join(f"st={i}" for i in range(len(MAPPING)))
               + f":audio_element_type=channel,layer=ch_layout={LAYOUT}")
    presentation = (
        "type=iamf_mix_presentation:id=3:stg=0:annotations=en-us=StemSpace,"
        f"submix=parameter_id=100:parameter_rate={EXPORT_RATE}:default_mix_gain=0.0"
        f"|element=stg=0:headphones_rendering_mode=binaural:annotations=en-us=Speaker mix:"
        f"parameter_id=101:parameter_rate={EXPORT_RATE}:default_mix_gain=0.0"
        f"|{mix_layout('stereo', stereo, attenuation)}|{mix_layout(LAYOUT, surround, attenuation)}"
    )
    args = ffmpeg("-loglevel", "error", "-y", "-ch_layout", LAYOUT, "-i", str(bed),
                  "-filter_complex", ";".join(filters))
    for label, _ in MAPPING:
        args.extend(("-map", f"[{label}]"))
    args.extend(("-stream_group", element, "-stream_group", presentation))
    for i, (_, channels) in enumerate(MAPPING):
        args.extend(("-streamid", f"{i}:{i}", f"-b:a:{i}", "160k" if "|" in channels else "96k"))
    run([*args, "-c:a", "libopus", "-f", "iamf", str(target)],
        "Encode 7.1.4 IAMF (requires an IAMF-enabled FFmpeg with libopus)")
    with target.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8 or header[:6] != b"\xf8\x06iamf":
        raise StemSpaceError("The encoder did not produce a standalone IAMF sequence.")
    return attenuation


def export_iamf(project: Project, output: Path, *, overwrite: bool = False,
                progress: Progress = quiet) -> float:
    """Export the verified 7.1.4 path; the editable objects stay in the project."""
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".iamf":
        raise StemSpaceError("Choose a destination ending in .iamf.")
    if output.exists() and not overwrite:
        raise StemSpaceError(f"Output already exists: {output}. Choose a new name or use --overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".stemspace-export-", dir=output.parent) as tmp:
        bed = Path(tmp) / "speakers.wav"
        encoded = Path(tmp) / "sequence.iamf"
        progress("Rendering the placements into a 7.1.4 speaker mix…")
        render_speaker_bed(project, bed)
        progress("Encoding the 48 kHz Opus IAMF file…")
        attenuation = encode_speaker_bed(bed, encoded)
        if overwrite:
            encoded.replace(output)
        else:
            try:
                output.hardlink_to(encoded)
            except FileExistsError as exc:
                raise StemSpaceError(f"Output already exists: {output}") from exc
    progress(f"Saved {output.name} (7.1.4 speaker mix).")
    return attenuation
