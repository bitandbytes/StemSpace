from __future__ import annotations

import importlib.metadata
import json
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import soundfile as sf
import yaml

from .audio import align_wav, canonical_wav, convert_wav, reconstruct
from .project import LABELS, PROJECT_FILENAME, Project, Track, default_placement, load_project
from .tools import Progress, StemSpaceError, digest_file, digest_json, executable, quiet, run

CACHE_VERSION = 1
ID = re.compile(r"[a-z][a-z0-9_-]*\Z")
MLX_OPTION_FLAGS = {
    "mdxc_segment_size": "--mdxc_segment_size",
    "mdxc_overlap": "--mdxc_overlap",
    "mdxc_batch_size": "--mdxc_batch_size",
}


@dataclass(frozen=True)
class Stage:
    id: str
    input: str
    model: str
    sample_rate: int
    outputs: dict[str, str]  # Model output label -> stable stem ID.
    options: dict[str, int]
    revision: str


@dataclass(frozen=True)
class Recipe:
    stages: tuple[Stage, ...]
    scene: tuple[str, ...]

    def metadata(self) -> dict:
        return {"stages": [asdict(stage) for stage in self.stages], "scene": list(self.scene)}


def load_recipe(path: Path | None = None) -> Recipe:
    try:
        raw = (path.read_text(encoding="utf-8") if path is not None
               else files("stemspace").joinpath("cascade.yaml").read_text(encoding="utf-8"))
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise StemSpaceError(f"Cannot read cascade recipe: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"stages", "scene"}:
        raise StemSpaceError("A recipe must define stages and scene.")
    if not isinstance(data["stages"], list) or not data["stages"]:
        raise StemSpaceError("A recipe needs at least one stage.")
    refs = {"source"}
    stage_ids: set[str] = set()
    stages = []
    for row in data["stages"]:
        if not isinstance(row, dict) or set(row) - {
            "id", "input", "model", "sample_rate", "outputs", "options", "revision"
        }:
            raise StemSpaceError("A cascade stage contains invalid fields.")
        name, parent, model, outputs = (row.get(key) for key in ("id", "input", "model", "outputs"))
        if not isinstance(name, str) or not ID.fullmatch(name) or name in stage_ids or name == "source":
            raise StemSpaceError(f"Invalid or duplicate stage ID: {name!r}")
        if not isinstance(parent, str) or parent not in refs:
            raise StemSpaceError(f"{name}: input must be source or an output of a previous stage.")
        if (not isinstance(model, str) or not model or not model[0].isalnum()
                or model != Path(model).name or "\\" in model):
            raise StemSpaceError(f"{name}: provide a registered MLX model filename or ID.")
        if isinstance(outputs, list) and all(isinstance(s, str) for s in outputs):
            if len(outputs) != len(set(outputs)):
                raise StemSpaceError(f"{name}: duplicate output labels.")
            outputs = {s: s for s in outputs}
        if (not isinstance(outputs, dict) or not outputs
                or any(not isinstance(k, str) or not k or not isinstance(v, str)
                       or not ID.fullmatch(v) or v == "residual" for k, v in outputs.items())
                or len({k.lower() for k in outputs}) != len(outputs)
                or len(set(outputs.values())) != len(outputs)):
            raise StemSpaceError(f"{name}: provide unique model labels and lowercase stem IDs.")
        rate = row.get("sample_rate", 44100)
        options = row.get("options", {})
        revision = row.get("revision", "")
        if type(rate) is not int or not 8000 <= rate <= 192000:
            raise StemSpaceError(f"{name}: invalid sample rate.")
        if (not isinstance(options, dict) or any(
                key not in MLX_OPTION_FLAGS or type(value) is not int or value < 1
                for key, value in options.items()
        )):
            raise StemSpaceError(f"{name}: options must be positive integers for {', '.join(MLX_OPTION_FLAGS)}.")
        if not isinstance(revision, str):
            raise StemSpaceError(f"{name}: revision must be a string.")
        stage_ids.add(name)
        refs.update(f"{name}.{stem}" for stem in outputs.values())
        stages.append(Stage(name, parent, model, rate, outputs, options, revision))
    scene = data["scene"]
    if (not isinstance(scene, list) or not scene
            or any(not isinstance(ref, str) or ref == "source" or ref not in refs for ref in scene)
            or len(set(scene)) != len(scene)):
        raise StemSpaceError("Scene must select unique outputs of the configured stages.")
    return Recipe(tuple(stages), tuple(scene))


class MLXBackend:
    @property
    def identity(self) -> str:
        try:
            version = importlib.metadata.version("mlx-audio-separator")
        except importlib.metadata.PackageNotFoundError:
            version = "external"
        configured = executable("mlx-audio-separator", "STEMSPACE_MLX_SEPARATOR")
        binary = Path(configured).resolve()
        return f"mlx-audio-separator:{version}:{binary}:{digest_file(binary)}"

    def separate(self, source: Path, stage: Stage, output: Path, progress: Progress) -> None:
        binary = executable("mlx-audio-separator", "STEMSPACE_MLX_SEPARATOR")
        args = [binary, str(source), "-m", stage.model, "--output_format", "WAV",
                "--output_dir", str(output), "--sample_rate", str(stage.sample_rate),
                "--normalization", "1.0", "--amplification", "0",
                "--custom_output_names", json.dumps(stage.outputs)]
        for key, value in stage.options.items():
            args.extend((MLX_OPTION_FLAGS[key], str(value)))
        if "mdxc_segment_size" in stage.options:
            args.append("--mdxc_override_model_segment_size")
        progress(f"Running {stage.model} on the Apple GPU…")
        run(args, f"Separate {stage.id} with {stage.model}")


def cache_directory(target: Path, build: Callable[[Path], None],
                    valid: Callable[[Path], bool]) -> None:
    if valid(target):
        return
    if target.exists():
        raise StemSpaceError(f"Incomplete or damaged cache: {target}. Move that folder aside and retry.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".work-", dir=target.parent) as work:
        candidate = Path(work) / "result"
        candidate.mkdir()
        build(candidate)
        if not valid(candidate):
            raise StemSpaceError(f"Could not build {target.name}.")
        try:
            candidate.rename(target)
        except OSError:
            if not valid(target):
                raise


def valid_wavs(folder: Path, names: list[str], *, rate: int | None = None,
               frames: int | None = None) -> bool:
    try:
        for name in names:
            info = sf.info(folder / f"{name}.wav")
            if (info.channels != 2 or not info.frames or info.subtype != "FLOAT"
                    or (rate is not None and info.samplerate != rate)
                    or (frames is not None and info.frames != frames)):
                return False
        return True
    except (OSError, RuntimeError):
        return False


def separate(source: Path, root: Path, recipe: Recipe, *, backend=None,
             progress: Progress = quiet, replace: bool = False) -> Project:
    source, root = source.expanduser().resolve(), root.expanduser().resolve()
    if not source.is_file():
        raise StemSpaceError(f"Audio file not found: {source}")
    progress("Reading the input recording…")
    source_hash = digest_file(source)
    prior = load_project(root, check_audio=False) if (root / PROJECT_FILENAME).exists() else None
    if prior is not None and prior.source_hash != source_hash and not replace:
        raise StemSpaceError("That project already belongs to another recording. Choose a new folder "
                             "or use --replace with the CLI.")
    if prior is not None and prior.source_hash != source_hash:
        prior = None
    backend = backend if backend is not None else MLXBackend()
    cache = root / "cache"
    ref_key = digest_json({"version": CACHE_VERSION, "source": source_hash})
    reference_dir = cache / "reference" / ref_key
    cache_directory(reference_dir, lambda d: canonical_wav(source, d / "source.wav"),
                    lambda d: valid_wavs(d, ["source"]))
    reference = reference_dir / "source.wav"
    base = sf.info(reference)
    outputs: dict[str, tuple[Path, str]] = {"source": (reference, ref_key)}
    backend_id = None
    for stage in recipe.stages:
        parent, parent_key = outputs[stage.input]
        if backend_id is None:
            backend_id = backend.identity
        key = digest_json({"version": CACHE_VERSION, "stage": asdict(stage),
                           "input": parent_key, "backend": backend_id})
        destination = cache / "stages" / key
        names = list(stage.outputs.values())

        def build(directory: Path) -> None:
            prepared = directory.parent / "input.wav"
            raw = directory.parent / "raw"
            raw.mkdir()
            convert_wav(parent, prepared, stage.sample_rate)
            frames = sf.info(prepared).frames
            backend.separate(prepared, stage, raw, progress)
            for stem in names:
                output = raw / f"{stem}.wav"
                if not output.is_file():
                    seen = ", ".join(p.name for p in sorted(raw.iterdir())) or "(none)"
                    raise StemSpaceError(
                        f"{stage.id} did not produce {stem}.wav. Check the recipe's output labels. "
                        f"Files written: {seen}"
                    )
                align_wav(output, directory / output.name, frames, stage.sample_rate)

        if valid_wavs(destination, names, rate=stage.sample_rate):
            progress(f"Reusing cached {stage.id} stems.")
        else:
            cache_directory(destination, build, lambda d: valid_wavs(d, names, rate=stage.sample_rate))
        for name in names:
            outputs[f"{stage.id}.{name}"] = (destination / f"{name}.wav", f"{key}:{name}")

    scene_key = digest_json({"version": CACHE_VERSION, "reference": ref_key,
                             "tracks": [outputs[ref][1] for ref in recipe.scene]})
    scene_dir = cache / "scenes" / scene_key
    ids = [ref.rsplit(".", 1)[-1] for ref in recipe.scene]
    file_ids = [name if ids.count(name) == 1 else ref
                for ref, name in zip(recipe.scene, ids)]

    def build_scene(directory: Path) -> None:
        progress("Aligning the final tracks and writing the residual…")
        for ref, name in zip(recipe.scene, file_ids):
            align_wav(outputs[ref][0], directory / f"{name}.wav", base.frames, base.samplerate)
        report = reconstruct(reference, [directory / f"{name}.wav" for name in file_ids],
                             directory / "residual.wav", create=True)
        if not report["passed"]:
            raise StemSpaceError(f"Dry reconstruction exceeds tolerance: {report['peak_error']:.3g} FS.")

    cache_directory(scene_dir, build_scene, lambda d: valid_wavs(
        d, [*file_ids, "residual"], rate=base.samplerate, frames=base.frames
    ))
    old_positions = {track.id: track.placement for track in prior.tracks} if prior else {}
    tracks = [
        Track(ref, LABELS.get(name, name.replace("_", " ").title())
              if ids.count(name) == 1 else f"{name.title()} ({ref.split('.')[0]})",
              str((scene_dir / f"{file_id}.wav").relative_to(root)),
              old_positions.get(ref, default_placement(name)))
        for ref, name, file_id in zip(recipe.scene, ids, file_ids)
    ]
    residual = scene_dir / "residual.wav"
    progress("Checking that the dry tracks and residual reconstruct the input…")
    report = reconstruct(reference, [root / track.file for track in tracks], residual, create=False)
    if not report["passed"]:
        raise StemSpaceError(f"Dry reconstruction exceeds tolerance: {report['peak_error']:.3g} FS.")
    project = Project(root, str(source), source_hash, recipe.metadata(), str(reference.relative_to(root)),
                      str(residual.relative_to(root)), base.samplerate, base.frames, tracks, report,
                      prior.include_residual if prior else False)
    project.save()
    progress(f"Ready: {len(tracks)} editable tracks.")
    return project
