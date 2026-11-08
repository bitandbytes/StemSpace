from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .audio import reconstruct
from .cascade import load_recipe, separate
from .export import export_iamf
from .project import load_project
from .tools import StemSpaceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stemspace")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    ui = commands.add_parser("ui", help="Open the desktop placement editor")
    ui.add_argument("project", type=Path, nargs="?")
    extract = commands.add_parser("separate", help="Run the MLX cascade and create a project")
    extract.add_argument("audio", type=Path)
    extract.add_argument("project", type=Path, help="Project directory")
    extract.add_argument("--recipe", type=Path, help="Use a custom YAML cascade")
    extract.add_argument("--replace", action="store_true", help="Replace project settings for a different song")
    check = commands.add_parser("check", help="Verify the dry tracks and residual against the source")
    check.add_argument("project", type=Path)
    export = commands.add_parser("export", help="Save a 7.1.4 standalone IAMF file")
    export.add_argument("project", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        if args.command in (None, "ui"):
            try:
                from .ui import launch
            except ModuleNotFoundError as exc:
                if exc.name in {"PySide6", "soxr"}:
                    raise StemSpaceError('Install the desktop extra: python -m pip install ".[ui]"') from exc
                raise
            return launch(getattr(args, "project", None))
        if args.command == "separate":
            project = separate(args.audio, args.project, load_recipe(args.recipe), progress=log,
                               replace=args.replace)
            print(project.path)
        elif args.command == "check":
            project = load_project(args.project)
            report = reconstruct(project.audio_path(project.reference),
                                 [project.audio_path(track.file) for track in project.tracks],
                                 project.audio_path(project.residual), create=False)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 1
        elif args.command == "export":
            gain = export_iamf(load_project(args.project), args.output, overwrite=args.overwrite, progress=log)
            if gain:
                log(f"Export headroom adjustment: {gain:.1f} dB.")
    except (StemSpaceError, OSError, RuntimeError) as exc:
        log(str(exc))
        return 1
    return 0
