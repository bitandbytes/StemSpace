from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

Progress = Callable[[str], None]


class StemSpaceError(Exception):
    """An error the CLI or UI can show without a traceback."""


def quiet(_: str) -> None:
    pass


def executable(name: str, env: str) -> str:
    configured = os.environ.get(env)
    binary = shutil.which(configured or name)
    if binary is None:
        raise StemSpaceError(f"Could not find {configured or name}. Install it or set {env} to its path.")
    return binary


def run(args: list[str], action: str, *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(args, cwd=cwd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", check=False)
    except OSError as exc:
        raise StemSpaceError(f"{action}: {exc}") from exc
    if result.returncode:
        details = "\n".join(result.stdout.strip().splitlines()[-18:])
        raise StemSpaceError(f"{action} failed (exit {result.returncode}).\n{details}")
    return result.stdout


def ffmpeg(*args: str) -> list[str]:
    return [executable("ffmpeg", "STEMSPACE_FFMPEG"), "-hide_banner", "-nostdin", *args]


def digest_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
