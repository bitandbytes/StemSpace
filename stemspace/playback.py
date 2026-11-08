"""Shared PCM sources, device routing, and a single playback worker."""

from __future__ import annotations

import math
import queue
import threading
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
import soxr

from .audio import check_timeline
from .export import stereo_matrix
from .playback_render import BLOCK, LAYOUTS, MONITOR_RATE, MonitorRenderer
from .project import Placement, Project
from .tools import StemSpaceError

RENDER_AHEAD_SECONDS = 0.085
DEVICE_LATENCY_SECONDS = 0.075
QUEUE_POLL_SECONDS = 0.01


@dataclass(frozen=True)
class Device:
    key: str
    index: int
    name: str
    channels: int
    default_rate: int


@dataclass(frozen=True)
class Output:
    device: Device
    layout: str
    mapping: tuple[int, ...]
    rate: int

    @property
    def channels(self) -> int:
        return max(self.mapping)


def validate_mapping(mapping: tuple[int, ...], layout: str, maximum: int) -> None:
    if len(mapping) != LAYOUTS[layout].channels or any(
        type(ch) is not int or not 1 <= ch <= maximum for ch in mapping
    ):
        raise StemSpaceError("Assign each speaker to a device channel in range.")
    if len(set(mapping)) != len(mapping):
        raise StemSpaceError("Assign a different output channel to each speaker.")


def route(audio: np.ndarray, output: Output) -> np.ndarray:
    if audio.shape[1] != len(output.mapping):
        raise StemSpaceError("The rendered channel count does not match the output layout.")
    if output.mapping == tuple(range(1, output.channels + 1)):
        return audio
    pcm = np.zeros((len(audio), output.channels), np.float32)
    pcm[:, [ch - 1 for ch in output.mapping]] = audio
    return pcm


class PortAudio:
    @staticmethod
    def module():
        try:
            import sounddevice
            return sounddevice
        except (ImportError, OSError) as exc:
            raise StemSpaceError(f"Audio output is unavailable: {exc}") from exc

    def devices(self) -> tuple[list[Device], int | None]:
        sd = self.module()
        hosts = sd.query_hostapis()
        items = []
        seen = {}
        for index, data in enumerate(sd.query_devices()):
            if data["max_output_channels"] <= 0:
                continue
            host = hosts[data["hostapi"]]["name"]
            identity = f"{host}: {data['name']}"
            seen[identity] = seen.get(identity, 0) + 1
            key = f"{identity} #{seen[identity]}"
            items.append(Device(key, index, str(data["name"]), int(data["max_output_channels"]),
                                int(round(data["default_samplerate"]))))
        default = sd.default.device[1]
        return items, None if default is None or default < 0 else int(default)

    def configure(self, device: Device, layout: str, mapping: tuple[int, ...]) -> Output:
        validate_mapping(mapping, layout, device.channels)
        sd = self.module()
        error = None
        for rate in dict.fromkeys((MONITOR_RATE, device.default_rate)):
            if rate < 8000:
                continue
            try:
                sd.check_output_settings(device=device.index, channels=max(mapping),
                                         samplerate=rate, dtype="float32")
            except (sd.PortAudioError, ValueError) as exc:
                error = exc
            else:
                return Output(device, layout, mapping, rate)
        raise StemSpaceError(f"{device.name} cannot open {max(mapping)} outputs at a supported rate: {error}")

    def open(self, output: Output):
        return self.module().OutputStream(device=output.device.index, channels=output.channels,
                                          samplerate=output.rate, dtype="float32",
                                          latency=DEVICE_LATENCY_SECONDS)


class Reader(Protocol):
    rate: int
    channels: int
    frames: int
    position: int

    def read(self, count: int) -> np.ndarray: ...
    def close(self) -> None: ...


class Source(Protocol):
    rate: int
    frames: int
    live: bool

    def open(self, position: int) -> Reader: ...


@dataclass(frozen=True)
class Controls:
    placements: tuple[Placement, ...]
    listen: tuple[bool, ...]
    residual: bool


class StemSource:
    live = True

    def __init__(self, project: Project):
        self.rate, self.frames = project.sample_rate, project.frames
        self.paths = tuple(project.audio_path(t.file) for t in project.tracks) + (
            project.audio_path(project.residual),
        )
        self.controls = Controls(tuple(t.placement for t in project.tracks),
                                 (True,) * len(project.tracks), project.include_residual)

    def set_controls(self, controls: Controls) -> None:
        if len(controls.listen) != len(self.paths) - 1 or len(controls.placements) != len(controls.listen):
            raise ValueError("Monitor controls must match the opened stems.")
        # Publish a single immutable snapshot. The audio worker only reads it.
        self.controls = controls

    def open(self, position: int) -> Reader:
        return StemReader(self, position)


class StemReader:
    channels = 12

    def __init__(self, source: StemSource, position: int):
        self.source = source
        self.rate, self.frames, self.position = source.rate, source.frames, position
        self.stack = ExitStack()
        try:
            self.readers = [self.stack.enter_context(sf.SoundFile(p)) for p in source.paths]
            for reader, path in zip(self.readers, source.paths):
                check_timeline(reader, self.rate, self.frames, path.name)
        except Exception:
            self.stack.close()
            raise
        self.controls = source.controls
        self.desired = self.matrices(self.controls)
        self.current = self.desired.copy()
        self.ramp_left = 0

    @staticmethod
    def matrices(controls: Controls) -> np.ndarray:
        positions = (*controls.placements, Placement(width=90))
        enabled = (*controls.listen, controls.residual)
        return np.asarray([stereo_matrix(p) if on else np.zeros((2, 12)) for p, on in
                           zip(positions, enabled)], dtype=np.float32)

    def read(self, count: int) -> np.ndarray:
        count = min(count, self.frames - self.position)
        controls = self.source.controls
        if controls != self.controls:
            self.controls = controls
            self.desired = self.matrices(controls)
            self.ramp_left = max(1, round(self.rate * 0.020))
        audio = np.zeros((count, 12), dtype=np.float32)
        if not count:
            return audio
        if self.ramp_left:
            phases = np.minimum(1, np.arange(1, count + 1, dtype=np.float32) / self.ramp_left)[:, None]
        for i, reader in enumerate(self.readers):
            current, desired = self.current[i], self.desired[i]
            if not np.any(current) and not np.any(desired):
                continue
            if reader.tell() != self.position:
                reader.seek(self.position)
            samples = reader.read(count, dtype="float32", always_2d=True)
            if len(samples) != count or not np.isfinite(samples).all():
                raise StemSpaceError("A project stem is missing samples or contains invalid PCM.")
            audio += np.einsum("ni,io->no", samples, current, optimize=False)
            if self.ramp_left and not np.array_equal(current, desired):
                audio += np.einsum("ni,io->no", samples, desired - current, optimize=False) * phases
        if self.ramp_left:
            self.current += (self.desired - self.current) * min(1, count / self.ramp_left)
            self.ramp_left = max(0, self.ramp_left - count)
            if not self.ramp_left:
                self.current[:] = self.desired
        self.position += count
        return audio

    def close(self) -> None:
        self.stack.close()


class WavSource:
    live = False

    def __init__(self, path: Path, layout: str):
        self.path, self.layout = path, layout
        info = sf.info(path)
        self.rate, self.frames = info.samplerate, info.frames
        if info.channels != LAYOUTS[layout].channels:
            raise StemSpaceError("The decoded file does not match the selected layout.")

    def open(self, position: int) -> Reader:
        return WavReader(self.path, position)


class WavReader:
    def __init__(self, path: Path, position: int):
        self.reader = sf.SoundFile(path)
        self.rate, self.frames, self.channels = self.reader.samplerate, self.reader.frames, self.reader.channels
        self.position = self.reader.seek(position)

    def read(self, count: int) -> np.ndarray:
        result = self.reader.read(count, dtype="float32", always_2d=True)
        self.position += len(result)
        return result

    def close(self) -> None:
        self.reader.close()


def resampler(source: int, target: int, channels: int):
    return None if source == target else soxr.ResampleStream(
        source, target, channels, dtype="float32", quality="MQ"
    )


def render_blocks(reader: Reader, source: Source, output: Output):
    """Yield device-rate PCM in renderer order, before routing to device pins."""
    if isinstance(source, WavSource) and source.layout != output.layout:
        raise StemSpaceError("Decode this IAMF file for the selected render layout first.")
    layout = LAYOUTS[output.layout]
    if source.live:
        renderer = MonitorRenderer(output.layout)
        convert_in = resampler(reader.rate, MONITOR_RATE, reader.channels)
        render_rate = MONITOR_RATE
        pending = np.empty((0, reader.channels), np.float32)
    else:
        renderer = convert_in = None
        render_rate = reader.rate
    convert_out = resampler(render_rate, output.rate, layout.channels)
    count = max(1, round(reader.rate * BLOCK / MONITOR_RATE))
    while reader.position < reader.frames:
        audio = reader.read(count)
        if not len(audio):
            raise StemSpaceError("Audio ended earlier than expected.")
        last = reader.position >= reader.frames
        if convert_in is not None:
            audio = convert_in.resample_chunk(audio, last=last)
        if renderer is not None:
            if len(pending):
                audio = np.concatenate((pending, audio))
            end = len(audio) if last else len(audio) // BLOCK * BLOCK
            chunks = [renderer.process(audio[i:min(i + BLOCK, end)]) for i in range(0, end, BLOCK)]
            pending = audio[end:]
            audio = np.concatenate(chunks) if chunks else np.empty((0, layout.channels), np.float32)
        if convert_out is not None:
            audio = convert_out.resample_chunk(audio, last=last)
        # SRC flushes can produce several blocks; keep writes short.
        for i in range(0, len(audio), BLOCK):
            yield np.ascontiguousarray(audio[i:i + BLOCK], dtype=np.float32)


class RenderAhead:
    """Own the reader and render on a producer thread, with bounded backpressure."""

    def __init__(self, source: Source, frame: int, output: Output):
        self.capacity = max(2, math.ceil(output.rate * RENDER_AHEAD_SECONDS / BLOCK))
        self.pcm: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.capacity)
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self.produce, args=(source, frame, output),
                                       name="StemSpace render ahead", daemon=True)
        self.thread.start()

    def produce(self, source: Source, frame: int, output: Output) -> None:
        try:
            with ExitStack() as stack:
                if self.cancelled.is_set():
                    return
                reader = source.open(frame)
                stack.callback(reader.close)
                blocks = iter(render_blocks(reader, source, output))
                while not self.cancelled.is_set():
                    try:
                        audio = next(blocks)
                    except StopIteration:
                        return
                    pcm = np.ascontiguousarray(route(audio, output), dtype=np.float32)
                    while not self.cancelled.is_set():
                        try:
                            self.pcm.put(pcm, timeout=QUEUE_POLL_SECONDS)
                        except queue.Full:
                            continue
                        break
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()

    @property
    def ready(self) -> bool:
        return self.pcm.full() or self.done.is_set()

    def take(self) -> np.ndarray | None:
        if self.done.is_set() and self.pcm.empty():
            if self.error is not None:
                raise self.error
            raise StopIteration
        try:
            return self.pcm.get(timeout=QUEUE_POLL_SECONDS)
        except queue.Empty:
            return None

    def cancel(self) -> None:
        self.cancelled.set()


@dataclass(frozen=True)
class Snapshot:
    token: int = 0
    position: float = 0
    duration: float = 0
    playing: bool = False
    error: str | None = None
    buffering: bool = False
    underruns: int = 0


class Transport:
    def __init__(self, backend=None):
        self.backend = backend or PortAudio()
        self.snapshot = Snapshot()
        self.commands: queue.Queue = queue.Queue()
        self.source: Source | None = None
        self.output: Output | None = None
        self.stream = None
        self.ahead: RenderAhead | None = None
        self.start = self.written = 0
        self.volume = self.current_volume = 0.60
        self.thread = threading.Thread(target=self.worker, name="StemSpace playback", daemon=True)
        self.thread.start()

    def load(self, source: Source | None, position: float, token: int) -> None:
        self.commands.put(("load", source, position, token))

    def play(self, output: Output) -> None:
        self.commands.put(("play", output))

    def pause(self) -> None:
        self.commands.put(("pause",))

    def seek(self, seconds: float) -> None:
        self.commands.put(("seek", seconds))

    def set_volume(self, value: float) -> None:
        self.volume = min(1, max(0, value))

    def test_speaker(self, output: Output, speaker: int) -> None:
        self.commands.put(("tone", output, speaker))

    def close(self) -> None:
        self.commands.put(("close",))
        self.thread.join(timeout=2)

    def stop_stream(self, *, drained: bool = False) -> None:
        stream, ahead = self.stream, self.ahead
        self.stream = self.ahead = None
        self.snapshot = replace(self.snapshot, playing=False, buffering=False)
        if ahead is not None:
            ahead.cancel()
        if stream is not None:
            try:
                stream.stop() if drained else stream.abort()
            finally:
                stream.close()

    def position(self) -> float:
        if self.stream is None or self.output is None:
            return self.snapshot.position
        latency = float(getattr(self.stream, "latency", 0))
        return min(self.snapshot.duration,
                   self.start + max(0, self.written / self.output.rate - latency))

    def start_stream(self, output: Output) -> None:
        if self.source is None:
            return
        if self.snapshot.position >= self.snapshot.duration:
            self.snapshot = replace(self.snapshot, position=0)
        self.output = output
        frame = min(self.source.frames, round(self.snapshot.position * self.source.rate))
        self.start, self.written = frame / self.source.rate, 0
        self.current_volume = 0
        self.snapshot = replace(self.snapshot, playing=True, buffering=True, error=None, underruns=0)
        self.ahead = RenderAhead(self.source, frame, output)

    def write_block(self) -> None:
        ahead = self.ahead
        if self.stream is None and not ahead.ready:
            ahead.done.wait(QUEUE_POLL_SECONDS)
            return
        try:
            audio = ahead.take()
        except StopIteration:
            self.stop_stream(drained=True)
            self.snapshot = replace(self.snapshot, position=self.snapshot.duration)
            return
        if audio is None:
            return
        if self.stream is None:
            self.stream = self.backend.open(self.output)
            self.stream.start()
            self.snapshot = replace(self.snapshot, buffering=False)
        ramp_frames = max(1, round(self.output.rate * 0.018))
        target = self.volume
        phase = np.minimum(1, np.arange(1, len(audio) + 1, dtype=np.float32) / ramp_frames)[:, None]
        gains = self.current_volume + (target - self.current_volume) * phase
        self.current_volume = float(gains[-1, 0])
        # Fixed monitor level; saturation is only a last guard for hot mixes.
        audio = np.multiply(audio, gains, dtype=np.float32)
        np.clip(audio, -0.99, 0.99, out=audio)
        underflowed = bool(self.stream.write(audio))
        self.written += len(audio)
        self.snapshot = replace(self.snapshot, position=self.position(),
                                underruns=self.snapshot.underruns + int(underflowed))

    def tone(self, output: Output, speaker: int) -> None:
        if not 0 <= speaker < len(output.mapping):
            raise StemSpaceError("Unknown speaker channel.")
        name = LAYOUTS[output.layout].speakers[speaker]
        count = round(output.rate * 0.35)
        wave = np.sin(np.arange(count) * math.tau * (55 if name == "LFE" else 550) / output.rate)
        fade = min(count // 2, round(output.rate * 0.035))
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
        stream = self.backend.open(output)
        try:
            stream.start()
            for offset in range(0, count, BLOCK):
                segment = wave[offset:offset + BLOCK]
                audio = np.zeros((len(segment), output.channels), dtype=np.float32)
                audio[:, output.mapping[speaker] - 1] = segment * 0.08
                stream.write(audio)
            stream.stop()
        finally:
            stream.close()

    def worker(self) -> None:
        while True:
            try:
                try:
                    command = self.commands.get(block=self.ahead is None)
                except queue.Empty:
                    self.write_block()
                    continue
                match command:
                    case ("close",):
                        try:
                            self.stop_stream()
                        finally:
                            return
                    case ("load", source, seconds, token):
                        self.stop_stream()
                        self.source = source
                        duration = source.frames / source.rate if source else 0
                        self.snapshot = Snapshot(token, max(0, min(seconds, duration)), duration)
                    case ("play", output):
                        if not self.snapshot.playing:
                            self.start_stream(output)
                    case ("pause",):
                        position = self.position()
                        self.stop_stream()
                        self.snapshot = replace(self.snapshot, position=position, error=None)
                    case ("seek", seconds):
                        playing = self.snapshot.playing
                        self.stop_stream()
                        position = max(0, min(seconds, self.snapshot.duration))
                        self.snapshot = replace(self.snapshot, position=position, error=None)
                        if playing and position < self.snapshot.duration:
                            self.start_stream(self.output)
                    case ("tone", output, channel):
                        position = self.position()
                        self.stop_stream()
                        self.snapshot = replace(self.snapshot, position=position, error=None)
                        self.tone(output, channel)
            except Exception as exc:
                try:
                    self.stop_stream()
                except Exception:
                    pass
                self.snapshot = replace(self.snapshot, error=str(exc), playing=False)
