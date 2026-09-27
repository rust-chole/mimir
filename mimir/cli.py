"""Command line: ``python -m mimir run VIDEO`` (plus ``models`` and ``doctor``)."""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

from mimir.config import Settings, parse_speaker_names
from mimir.core import log
from mimir.errors import MimirError, QualityGateError


def _provider(args: argparse.Namespace):
    if args.replay:
        from mimir.models.replay import ReplayProvider

        return ReplayProvider(args.replay)
    from mimir.models.openai_provider import OpenAIProvider

    provider = OpenAIProvider()
    if args.record:
        from mimir.models.replay import RecordingProvider

        provider = RecordingProvider(provider, args.record)
    return provider


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings()
    identity = dataclasses.replace(
        settings.identity,
        interactive=bool(args.interactive),
        speaker_names=parse_speaker_names(args.speaker_names or ""),
        creator=args.creator or settings.identity.creator,
        entities=tuple(e.strip() for e in (args.entities.split(",") if args.entities else settings.identity.entities)
                       if e.strip()),
    )
    story = dataclasses.replace(settings.story, story_index=args.story_index)
    vision = dataclasses.replace(settings.vision, observer=not args.no_observer)
    qc = dataclasses.replace(settings.qc, reviewer=bool(args.reviewer))
    output = dataclasses.replace(settings.output, fps=int(args.fps or 0))
    changes = {"identity": identity, "story": story, "vision": vision, "qc": qc, "output": output}
    if args.workspace:
        changes["workspace"] = Path(args.workspace).resolve()
    if args.output:
        changes["output_dir"] = Path(args.output).resolve()
    if args.sfx_library:
        changes["effects"] = dataclasses.replace(settings.effects, sfx_library=str(Path(args.sfx_library).resolve()))
    return settings.with_(**changes)


def cmd_run(args: argparse.Namespace) -> int:
    from mimir.pipeline import run_job

    log.set_quiet(args.quiet)
    settings = settings_from_args(args)
    rerun = [s.strip() for s in (args.rerun or "").split(",") if s.strip()]
    try:
        result = run_job(args.video, settings, _provider(args), force=args.force, rerun=rerun)
    except QualityGateError as error:
        print(f"QUALITY GATE FAILED: {error}\nreport: {error.report_path}", file=sys.stderr)
        return 3
    except MimirError as error:
        print(f"MIMIR ERROR: {error}", file=sys.stderr)
        return 2
    print(f"\nShort: {result.published_video}\nManifest: {result.manifest}")
    return 0


def cmd_models(_: argparse.Namespace) -> int:
    settings = Settings()
    for role, route in sorted(settings.routes.items()):
        print(f"{role:24s} {route.model}" + (f" [{route.effort}]" if route.effort else ""))
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    import os

    ok = True
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        print(f"{tool:10s} {found or 'MISSING'}")
        ok &= bool(found)
    if shutil.which("ffmpeg"):
        from mimir.media.ffmpeg import filter_has

        for name in ("ass", "sidechaincompress", "loudnorm"):
            present = filter_has(name)
            print(f"filter {name:18s} {'ok' if present else 'MISSING'}")
            ok &= present
    try:
        import cv2
        import numpy

        print(f"opencv     {cv2.__version__} (numpy {numpy.__version__})")
        ok &= hasattr(cv2, "CascadeClassifier")
        from mimir.vision.faces import load_detector

        detector, fallback = load_detector()
        print(f"faces      {detector.name}" + (f" (FALLBACK: {fallback})" if fallback else ""))
    except ImportError as error:
        print(f"opencv     MISSING ({error})")
        ok = False
    print(f"OPENAI_API_KEY {'set' if os.getenv('OPENAI_API_KEY') else 'not set (needed unless --replay)'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mimir", description="MIMIR: VOD -> vertical Short with a peak cold open")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="produce a Short from a VOD")
    run.add_argument("video")
    run.add_argument("--workspace")
    run.add_argument("--output")
    run.add_argument("--story-index", type=int, default=None, help="force the Nth ranked story candidate")
    run.add_argument("--speaker-names", help="confirmed names, e.g. S1=Alex,S2=Sam")
    run.add_argument("--entities", help="verified names/terms for caption spelling, comma separated")
    run.add_argument("--creator", help="verified creator name")
    run.add_argument("--interactive", action="store_true", help="ask for speaker names using speaker_preview audio")
    run.add_argument("--force", action="store_true", help="re-run every stage")
    run.add_argument("--rerun", help="comma separated stages to re-run (dependents follow via signatures)")
    run.add_argument("--replay", help="serve model responses from a recorded directory (regression runs)")
    run.add_argument("--record", help="record every model response into this directory")
    run.add_argument("--no-observer", action="store_true", help="disable the bounded multimodal observer")
    run.add_argument("--reviewer", action="store_true", help="enable the bounded multimodal final reviewer")
    run.add_argument("--sfx-library")
    run.add_argument("--fps", type=int, default=0)
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)
    sub.add_parser("models", help="print model routing").set_defaults(func=cmd_models)
    sub.add_parser("doctor", help="check the local toolchain").set_defaults(func=cmd_doctor)
    args = parser.parse_args(argv)
    return int(args.func(args))
