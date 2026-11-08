from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .tools import StemSpaceError, write_json

PROJECT_FILENAME = "project.json"
SCHEMA_VERSION = 1
LABELS = {"hh": "Hi-hat", "other": "Other"}
DEFAULTS = {
    "bass": (0, 0, 0), "kick": (0, 0, 0), "snare": (-12, 0, 0),
    "toms": (35, 5, 20), "hh": (-110, 35, 0), "ride": (125, 45, 0),
    "crash": (-145, 45, 25), "vocals": (0, 0, 0),
    "guitar": (-45, 0, 22), "piano": (45, 0, 34), "other": (0, 0, 75),
}


@dataclass(frozen=True)
class Placement:
    azimuth: float = 0
    elevation: float = 0
    width: float = 0
    gain_db: float = 0
    muted: bool = False

    def __post_init__(self) -> None:
        for field, lo, hi in (("azimuth", -180, 180), ("elevation", 0, 90),
                              ("width", 0, 180), ("gain_db", -60, 12)):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not lo <= value <= hi):
                raise StemSpaceError(f"{field} must be a finite number from {lo} to {hi}.")
        if not isinstance(self.muted, bool):
            raise StemSpaceError("Muted must be true or false.")


def default_placement(stem: str) -> Placement:
    return Placement(*DEFAULTS.get(stem, (0, 0, 30)))


@dataclass
class Track:
    id: str
    label: str
    file: str
    placement: Placement


@dataclass
class Project:
    root: Path
    source: str
    source_hash: str
    recipe: dict
    reference: str
    residual: str
    sample_rate: int
    frames: int
    tracks: list[Track]
    reconstruction: dict
    include_residual: bool = False

    @property
    def path(self) -> Path:
        return self.root / PROJECT_FILENAME

    def audio_path(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel:
            raise StemSpaceError("The project contains an invalid audio path.")
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise StemSpaceError(f"Audio path must stay inside the project folder: {rel}")
        absolute = (self.root / path).resolve()
        if not absolute.is_relative_to(self.root.resolve()):
            raise StemSpaceError(f"Audio path must stay inside the project folder: {rel}")
        return absolute

    def save(self) -> None:
        data = {key: value for key, value in vars(self).items() if key not in ("root", "tracks")}
        data.update(schema_version=SCHEMA_VERSION, tracks=[asdict(track) for track in self.tracks])
        write_json(self.path, data)


def load_project(path: Path, *, check_audio: bool = True) -> Project:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / PROJECT_FILENAME
    if path.name != PROJECT_FILENAME:
        raise StemSpaceError(f"Select a project folder or its {PROJECT_FILENAME} file.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.pop("schema_version", None) != SCHEMA_VERSION:
            raise StemSpaceError("This project file uses an unsupported version.")
        rows = data.pop("tracks")
        tracks = [Track(**{**row, "placement": Placement(**row["placement"])}) for row in rows]
        if not tracks or len({track.id for track in tracks}) != len(tracks):
            raise StemSpaceError("The project must have tracks with unique IDs.")
        project = Project(root=path.parent, tracks=tracks, **data)
        if (not isinstance(project.frames, int) or project.frames <= 0
                or not isinstance(project.sample_rate, int) or project.sample_rate <= 0
                or not isinstance(project.include_residual, bool)):
            raise StemSpaceError("The project contains invalid timeline or render settings.")
        for name in [project.reference, project.residual, *(track.file for track in tracks)]:
            audio = project.audio_path(name)
            if check_audio and not audio.is_file():
                raise StemSpaceError(f"Missing project audio: {name}")
        return project
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise StemSpaceError(f"Cannot open {path}: {exc}") from exc
