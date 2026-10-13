"""``python -m ai.caption_eval <command>``: build the gold set, run systems, score, compare."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ai.caption_eval import cases as case_store
from ai.caption_eval import runner, systems


def _names(value: str | None) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _cmd_prepare(args: argparse.Namespace) -> int:
    common = dict(root=args.root, names=_names(args.names), entities=_names(args.entities), language=args.language,
                  force=args.force)
    targets: list[tuple[str, Path]] = []
    if args.clip:
        targets.append(("clip", Path(args.clip)))
    if args.final:
        final = Path(args.final)
        if final.is_dir():
            found = sorted(final.glob("*_short.mp4"))
            if not found:
                print(f"no *_short.mp4 in {final}")
                return 1
            targets += [("final", path) for path in found]
        else:
            targets.append(("final", final))
    if not targets:
        print("prepare needs --clip PATH or --final PATH (a published *_short.mp4 or the folder holding them)")
        return 2
    if args.id and len(targets) > 1:
        print("--id names ONE case; drop it when preparing a folder")
        return 2
    failed = 0
    for kind, path in targets:
        try:
            if kind == "clip":
                case = case_store.prepare_from_clip(path, case_id=args.id, profile=args.profile, **common)
            else:
                case = case_store.prepare_from_final(path, case_id=args.id, timeline=args.timeline,
                                                     profile=args.profile, **common)
        except FileExistsError as error:
            print(f"= {path.name}: {error}")
            continue
        except Exception as error:
            failed += 1
            print(f"! {path.name}: {type(error).__name__}: {error}")
            continue
        draft = "with MIMIR's captions as the draft" if case.published_words() is not None else \
            "WITHOUT a draft (no pipeline profile found; pass --profile to get one)"
        print(f"+ {case.case_id}: {case.duration:.1f} s, {draft}\n    edit: {case.gold_path}")
    print("\nNext: listen to each audio.wav, correct gold.txt word for word and delete its TODO line.")
    return 1 if failed else 0


def _cmd_status(args: argparse.Namespace) -> int:
    print(runner.status_table(args.root))
    return 0


def _cmd_systems(_args: argparse.Namespace) -> int:
    print(systems.SYSTEM_HELP)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    overrides = runner.env_from_pairs(args.set or [])
    name, results = runner.run_system(args.system, name=args.name, overrides=overrides,
                                      only=_names(args.cases) or None, jobs=max(1, args.jobs), force=args.force,
                                      root=args.root)
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"run '{name}': {len(results)} case(s), {errors} error(s)")
    result = runner.score_run(name, root=args.root)
    aggregate = result["aggregate"]
    wer = aggregate.get("wer")
    print(f"scored {aggregate.get('cases', 0)} case(s) with a finished gold: WER "
          f"{'n/a' if wer is None else f'{100 * wer:.1f}%'}; report: {runner.run_dir(name, args.root) / 'report.md'}")
    return 1 if errors else 0


def _cmd_score(args: argparse.Namespace) -> int:
    for name in args.names:
        result = runner.score_run(name, root=args.root)
        wer = result["aggregate"].get("wer")
        print(f"{name}: {result['aggregate'].get('cases', 0)} case(s), WER "
              f"{'n/a' if wer is None else f'{100 * wer:.1f}%'} -> {runner.run_dir(name, args.root) / 'report.md'}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    print(runner.compare(args.names, root=args.root))
    print(f"(saved to {case_store.eval_root(args.root) / 'compare.md'})")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from ai.caption_eval import editor

    editor.serve(args.root, port=args.port, open_browser=not args.no_browser)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from ai.caption_eval import sheet

    path, count = sheet.export_sheet(root=args.root, only=_names(args.cases) or None, out=args.out)
    print(f"{count} short(s) -> {path}")
    print("Correct each block word for word, mark it [x], then `import` it here or send the file as it is.")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from ai.caption_eval import sheet

    rows = sheet.import_sheet(args.path, root=args.root)
    for row in rows:
        print(f"{row['case_id']}: {row['status']}")
    return 1 if any(r["status"].startswith("unknown") for r in rows) else 0


def _cmd_score_sheet(args: argparse.Namespace) -> int:
    from ai.caption_eval import sheet

    result = sheet.score_sheet(args.path, out=args.out)
    aggregate = result["aggregate"]
    wer = aggregate.get("wer")
    print(f"scored {aggregate.get('cases', 0)} short(s) from the sheet: WER "
          f"{'n/a' if wer is None else f'{100 * wer:.1f}%'}; report: {result['report_path']}")
    return 0


def _cmd_align_gold(args: argparse.Namespace) -> int:
    rows = runner.align_gold(root=args.root, only=_names(args.cases) or None, force=args.force)
    for row in rows:
        print(f"{row['case_id']}: {row['status']}")
    return 1 if any(r["status"].startswith("failed") for r in rows) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ai.caption_eval",
                                     description="Measure caption accuracy on real shorts against human gold.")
    parser.add_argument("--root", default=None, help="evaluation folder (default vod_output/caption_eval, "
                                                     "or MIMIR_CAPTION_EVAL_DIR)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="make a case (audio + MIMIR's captions as a draft gold) from a short")
    p.add_argument("--clip", help="an edited clip (vod_output/edited_clips/...; kept with --keep-temp) or any media")
    p.add_argument("--final", help="a published *_short.mp4, or the folder holding them (vod_output/final)")
    p.add_argument("--profile", help="the speaker profile JSON (found automatically when possible)")
    p.add_argument("--timeline", help="the final *.timeline.json of a published short (found automatically)")
    p.add_argument("--id", help="case id (default: from the file name)")
    p.add_argument("--names", help="verified names said in the short, comma separated")
    p.add_argument("--entities", help="other verified terms (game, brands), comma separated")
    p.add_argument("--language", help="caption language (default: the profile's, else CAPTION_LANGUAGE, else en)")
    p.add_argument("--force", action="store_true", help="rebuild audio/draft of an existing case (never gold.txt)")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("serve", help="open the local gold editor in the browser (audio + clickable words)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="only print the address")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("export", help="one text file with every short's draft transcript to correct")
    p.add_argument("--out", help="where to write it (default <root>/gold_sheet.txt)")
    p.add_argument("--cases", help="only these case ids, comma separated")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("import", help="write a corrected gold sheet back into the cases")
    p.add_argument("path")
    p.set_defaults(func=_cmd_import)

    p = sub.add_parser("score-sheet", help="score MIMIR's captions against a corrected gold sheet (text only)")
    p.add_argument("path")
    p.add_argument("--out", help="report path (default next to the sheet)")
    p.set_defaults(func=_cmd_score_sheet)

    p = sub.add_parser("status", help="list cases and whether their gold is finished")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("systems", help="list the systems that can be run and scored")
    p.set_defaults(func=_cmd_systems)

    p = sub.add_parser("run", help="run a system on every case (cached) and score it")
    p.add_argument("system", help="published | stack | ear:... (see `systems`)")
    p.add_argument("--name", help="run name (default: from the system and overrides)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="environment override for this run, e.g. MIMIR_CAPTION_PRIMARY_PROVIDER=qwen3_asr")
    p.add_argument("--cases", help="only these case ids, comma separated")
    p.add_argument("--jobs", type=int, default=1, help="cases in parallel (default 1)")
    p.add_argument("--force", action="store_true", help="ignore cached hypotheses")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("score", help="re-score runs against the current gold (no model call)")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("compare", help="compare runs side by side")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("align-gold", help="machine reference word times for finished gold (forced aligner)")
    p.add_argument("--cases", help="only these case ids, comma separated")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_align_gold)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
