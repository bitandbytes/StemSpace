"""Decode standalone IAMF to disk with the bundled AOMedia decoder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

from .playback_render import LAYOUTS, VENDOR
from .tools import StemSpaceError, executable


class DecodeCancelled(Exception):
    pass


class DecodeCache:
    def __init__(self) -> None:
        self.temp = TemporaryDirectory(prefix="stemspace-iamf-")
        self._queue = threading.Lock()
        self._state = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._closed = False

    @staticmethod
    def communicate(process: subprocess.Popen, cancel: threading.Event | None):
        if cancel is None:
            return process.communicate(timeout=195)
        deadline = time.monotonic() + 195
        while True:
            if cancel.is_set():
                raise DecodeCancelled
            try:
                return process.communicate(timeout=0.12)
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise

    def decode(self, source: Path, layout: str, *, cancel: threading.Event | None = None) -> Path:
        source = source.expanduser().resolve()
        if source.suffix.lower() != ".iamf":
            raise StemSpaceError("Choose a standalone .iamf file.")
        stat = source.stat()
        with source.open("rb") as stream:
            header = stream.read(16)
        if len(header) < 8 or header[0] >> 3 != 31 or b"iamf" not in header[1:]:
            raise StemSpaceError("The file does not start with a standalone IAMF sequence.")
        key = hashlib.sha256(repr((str(source), stat.st_size, stat.st_mtime_ns, layout)).encode()).hexdigest()
        target = Path(self.temp.name) / f"{key}.wav"
        partial = target.with_suffix(".tmp")
        selected = LAYOUTS[layout]
        with self._queue:
            with self._state:
                if self._closed:
                    raise StemSpaceError("IAMF playback is closing.")
                if cancel is not None and cancel.is_set():
                    raise DecodeCancelled
                if target.is_file():
                    return target
                node = executable("node", "STEMSPACE_NODE")
                self._process = process = subprocess.Popen(
                    [node, str(VENDOR / "decode.mjs"), str(source), str(partial), str(selected.decoder_id)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, errors="replace",
                )
            try:
                try:
                    out, err = self.communicate(process, cancel)
                except subprocess.TimeoutExpired as exc:
                    process.kill()
                    process.communicate()
                    raise StemSpaceError("IAMF decoding timed out.") from exc
                except DecodeCancelled:
                    process.kill()
                    process.communicate()
                    raise
                if cancel is not None and cancel.is_set():
                    raise DecodeCancelled
                if process.returncode:
                    detail = err.strip().splitlines()[0] if err.strip() else f"exit {process.returncode}"
                    raise StemSpaceError(f"Cannot decode {source.name} for {selected.name}: {detail}")
                info = sf.info(partial)
                metadata = json.loads(out)
                if (not info.frames or info.channels != selected.channels
                        or metadata["numChannels"] != selected.channels
                        or info.samplerate != metadata["sampleRate"]):
                    raise StemSpaceError(f"The decoder did not return {selected.channels} channels for {selected.name}.")
                partial.replace(target)
                return target
            finally:
                partial.unlink(missing_ok=True)
                with self._state:
                    self._process = None

    def close(self) -> None:
        with self._state:
            self._closed = True
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
        with self._queue:
            self.temp.cleanup()
