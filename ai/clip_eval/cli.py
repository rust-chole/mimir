"""``python -m ai.clip_eval <command>``: label videos blind, run variants of Stage 2, score, compare."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from ai.clip_eval import cases as case_store
from ai.clip_eval import pricing, runner, variants

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts"}


def _names(value: str | None) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _videos(paths: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            out += sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file())
        else:
            out.append(path)
    return out


def _cmd_prepare(args: argparse.Namespace) -> int:
    videos = _videos(args.videos)
    if not videos:
        print("prepare needs a video file (or a folder of videos)")
        return 2
    if args.id and len(videos) > 1:
        print("--id names ONE case; drop it when preparing several videos")
        return 2
    failed = 0
    for video in videos:
        try:
            case = case_store.prepare(video, root=args.root, case_id=args.id, split=args.split,
                                      transcript=args.transcript, visual_report=args.visual_report,
                                      use_visual_report=not args.no_visual_report, proxy=args.proxy,
                                      refresh=args.refresh)
        except FileExistsError as error:
            print(f"= {video.name}: {error}")
            continue
        except Exception as error:
            failed += 1
            print(f"! {video.name}: {type(error).__name__}: {error}")
            continue
        transcript = "transcript frozen" if case.has_transcript() else "NO transcript yet"
        visual = "visual facts frozen" if case.visual_report_path else "no visual-facts report (transcript-only Stage 2)"
        print(f"+ {case.case_id} [{case.split}]: {case_store.clock(case.duration)}, {transcript}, {visual}")
    print("\nNext: python -m ai.clip_eval serve   (label each video blind; or export / import a text sheet)")
    return 1 if failed else 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        print(f"video not found: {video}")
        return 2
    duration = case_store.probe_duration(video)
    if not duration:
        print(f"ffprobe could not read the duration of {video}")
        return 2
    env = runner.child_env(variants.Variant(variants.BASELINE, {}))
    text_model = (env.get("MIMIR_VOD_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe").strip()
    prices = pricing.load_prices()
    estimate = pricing.estimate_transcribe(prices, duration, text_model=text_model)
    target = case_store.default_transcript_path(video)
    print(f"Stage 1 for {video.name} ({case_store.clock(duration)}): {text_model} + whisper-1, "
          f"~${estimate['expected']:.2f} (cap needed: ${estimate['worst']:.2f}) -> {target}")
    if target.is_file():
        print("note: a transcript already exists there and would be replaced")
    if args.dry_run:
        print("--dry-run: nothing was run, no model was called.")
        return 0
    if estimate["unpriced"]:
        print(f"Refused: no price for {', '.join(estimate['unpriced'])} in {pricing.PRICING_PATH}. No model was called.")
        return 2
    if args.max_usd is None:
        print("No model was called. Add --max-usd <dollars> to run it.")
        return 2
    if estimate["worst"] > args.max_usd + 1e-9:
        print(f"Refused: up to ${estimate['worst']:.2f} is above --max-usd {args.max_usd:.2f}. No model was called.")
        return 2
    work = case_store.eval_root(args.root) / "tmp" / f"transcribe_{time.time_ns()}"
    job = {"mode": "transcribe", "work_dir": str(work), "video": str(video), "budget_usd": args.max_usd,
           "reserved_usd": estimate["worst"]}
    result = runner.spawn(job, env, case_store.PROJECT_ROOT, work / "transcribe.log")
    print(f"{result.get('status')}: {result.get('error') or result.get('transcript')} (log {work / 'transcribe.log'})")
    if result.get("status") == "ok":
        print(f"Next: python -m ai.clip_eval prepare {case_store.ps_quote(video)} --refresh")
    return 0 if result.get("status") == "ok" else 1


def _cmd_status(args: argparse.Namespace) -> int:
    print(runner.status_table(args.root))
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    found = {c.case_id: c for c in case_store.list_cases(args.root, args.cases)}
    for case_id in args.cases:
        if case_id not in found:
            print(f"! no case {case_id}")
            continue
        case_store.set_split(found[case_id], args.split)
        print(f"{case_id}: {args.split}")
    return 0 if len(found) == len(args.cases) else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    from ai.clip_eval import editor

    editor.serve(args.root, port=args.port, open_browser=not args.no_browser, annotator=args.annotator)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from ai.clip_eval import sheet

    path, count = sheet.export_sheet(root=args.root, only=_names(args.cases) or None, out=args.out,
                                     annotator=args.annotator, window=args.window)
    print(f"{count} video(s) -> {path}")
    print("Watch each video, write its moment: / never: lines, mark it [x], then `import` the file.")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    from ai.clip_eval import sheet

    rows = sheet.import_sheet(args.path, root=args.root, annotator=args.annotator)
    for row in rows:
        print(f"{row['case_id']}: {row['status']}")
    return 1 if any(r["status"].startswith(("unknown", "not saved")) for r in rows) else 0


def _cmd_harvest(args: argparse.Namespace) -> int:
    from ai.clip_eval import reviews

    rows = reviews.harvest(root=args.root, final_dir=args.final, rejected_dir=args.rejected)
    for row in rows:
        where = (f"{case_store.clock(row['range']['start'])}-{case_store.clock(row['range']['end'])}"
                 + (" (uncertain)" if row["range"].get("uncertain") else "")) if row.get("range") else "range unknown"
        print(f"{row.get('case_id') or '(no case)'}: {row['decision']} - {where} - {Path(row['qc_file']).name}")
    print(f"{len(rows)} review(s); weak labels are reported apart from the golden labels and never shown while "
          "labeling.")
    return 0


def _cmd_variants(args: argparse.Namespace) -> int:
    print(variants.describe(variants.list_variants(args.root)))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    plan, _records, code = runner.run_variant(
        args.variant, split=args.split, only=_names(args.cases) or None, repeats=args.repeats, stage=args.stage,
        max_usd=args.max_usd, dry_run=args.dry_run, name=args.name, force=args.force,
        include_unlabeled=args.include_unlabeled, keep_workspace=args.keep_workspace, root=args.root)
    if args.dry_run or code == 2:
        return code
    if not (plan.directory / "run.json").is_file():
        return code
    result = runner.score_run(plan.name, root=args.root)
    for split in case_store.SPLITS:
        aggregate = result["splits"][split]["aggregate"]
        if aggregate["videos"]:
            hit = aggregate["metrics"]["hit1"]["mean"]
            recall = aggregate["metrics"]["recall3"]["mean"]
            print(f"{split}: {aggregate['videos']} video(s): HIT@1 {'n/a' if hit is None else f'{hit:.2f}'}, "
                  f"Recall@3 {'n/a' if recall is None else f'{recall:.2f}'}")
    print(f"report: {plan.directory / 'report.md'}")
    return code


def _cmd_score(args: argparse.Namespace) -> int:
    for raw in args.names:
        name = runner.resolve_run(raw, args.root)
        result = runner.score_run(name, root=args.root)
        dev = result["splits"]["dev"]["aggregate"]
        print(f"{name}: dev {dev['videos']} / holdout {result['splits']['holdout']['aggregate']['videos']} video(s) -> "
              f"{runner.run_dir(name, args.root) / 'report.md'}")
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    print(runner.list_runs(args.root))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from ai.clip_eval import judge

    name_a, name_b = runner.resolve_run(args.run_a, args.root), runner.resolve_run(args.run_b, args.root)
    judge_result = None
    code = 0
    if args.judge:
        judge_result = judge.compare_runs(runner.score_run(name_a, root=args.root),
                                          runner.score_run(name_b, root=args.root), root=args.root,
                                          max_usd=args.max_usd, dry_run=args.dry_run)
        print(f"judge {judge_result['model']}: {judge_result['planned_calls']} uncached call(s), "
              f"~${judge_result['planned_usd']:.3f}; spent ${judge_result['spent_usd']:.3f}")
        if judge_result.get("refused"):
            print(f"judge: {judge_result['refused']}")
            code = 2 if not args.dry_run else 0
    text, decision = runner.compare(name_a, name_b, root=args.root, gate=args.gate, threshold=args.gate_threshold,
                                    judge_result=judge_result, out=args.out)
    print(text)
    print(f"(saved to {Path(args.out) if args.out else case_store.eval_root(args.root) / 'compare.md'})")
    if decision is not None:
        print(f"gate: {'PASS' if decision['passed'] else 'FAIL'}")
        return 0 if decision["passed"] else 1
    return code


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from ai.clip_eval import calibrate

    if args.runs:
        if len(args.runs) != 2:
            print("calibrate takes two runs (baseline and candidate), or none to label / judge the existing set")
            return 2
        names = [runner.resolve_run(n, args.root) for n in args.runs]
        added = calibrate.add_pairs(runner.score_run(names[0], root=args.root),
                                    runner.score_run(names[1], root=args.root), root=args.root, seed=args.seed)
        print(f"{added} new blind pair(s); {len(calibrate.load_pairs(args.root))} in the set "
              f"(aim for {calibrate.TARGET_PAIRS}).")
    if args.judge:
        result = calibrate.run_judge(root=args.root, max_usd=args.max_usd, dry_run=args.dry_run)
        print(f"judge {result['model']}: {result['planned_calls']} uncached call(s), ~${result['planned_usd']:.3f}; "
              f"spent ${result['spent_usd']:.3f}" + (f"; {result['refused']}" if result.get("refused") else ""))
        if result["n"]:
            print(f"{result['n']} pair(s) with a human and a judge verdict: agreement {result['agreement']:.2f}, "
                  f"Cohen's kappa {result['kappa']:.2f} -> judge {'TRUSTED' if result['trusted'] else 'UNTRUSTED'} "
                  f"(needs kappa >= {calibrate.TRUST_KAPPA} on >= 10 pairs)")
        else:
            print("no pair has both a human and a judge verdict yet")
        return 2 if result.get("refused") and not args.dry_run else 0
    if not calibrate.load_pairs(args.root):
        print("no calibration pairs yet: python -m ai.clip_eval calibrate <baseline run> <candidate run>")
        return 2
    if args.text:
        labeled = calibrate.text_session(root=args.root)
        print(f"{labeled} pair(s) answered; {len(calibrate.human_labels(args.root))} in total.")
        return 0
    from ai.clip_eval import editor

    editor.serve(args.root, port=args.port, open_browser=not args.no_browser, path="/calibrate")
    return 0


def _cmd_feedback(args: argparse.Namespace) -> int:
    from ai.clip_eval import guard

    path = guard.write_feedback(runner.resolve_run(args.run, args.root), root=args.root, out=args.out)
    print(f"dev-only feedback (checked: no holdout data) -> {path}")
    return 0


def _money(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-usd", type=float, default=None,
                        help="hard spending cap in USD for this command; without it no paid call is made")
    parser.add_argument("--dry-run", action="store_true", help="print the estimated cost and exit (no model call)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ai.clip_eval",
                                     description="Measure how good MIMIR's clip picks are against blind human labels.")
    parser.add_argument("--root", default=None, help="evaluation folder (default vod_output/clip_eval, "
                                                     "or MIMIR_CLIP_EVAL_DIR)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="make a case per source video (frozen transcript + visual facts; no model call)")
    p.add_argument("videos", nargs="+", help="source video(s) or a folder of videos")
    p.add_argument("--id", help="case id (default: from the file name)")
    p.add_argument("--split", choices=case_store.SPLITS, default="dev")
    p.add_argument("--transcript", help="Stage 1 transcript (default vod_output/transcripts/<video stem>.json)")
    p.add_argument("--visual-report", help="visual-facts report (default vod_output/video_brain/<stem>_video_report_v6.json)")
    p.add_argument("--no-visual-report", action="store_true", help="run Stage 2 transcript-only for this case")
    p.add_argument("--proxy", action="store_true", help="also make a small browser-playable proxy.mp4 (FFmpeg)")
    p.add_argument("--refresh", action="store_true", help="re-freeze the inputs of an existing case (labels kept)")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("transcribe", help="Stage 1 for a video without a transcript (PAID)")
    p.add_argument("video")
    _money(p)
    p.set_defaults(func=_cmd_transcribe)

    p = sub.add_parser("status", help="cases, splits, transcripts and whether their labels are finished")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("split", help="move cases to dev or holdout")
    p.add_argument("split", choices=case_store.SPLITS)
    p.add_argument("cases", nargs="+")
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("serve", help="open the local label editor (video + transcript, blind)")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true", help="only print the address")
    p.add_argument("--annotator", help="who is labeling (default MIMIR_CLIP_EVAL_ANNOTATOR or 'primary')")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("export", help="one text sheet: transcripts in 30 s windows + empty label blocks")
    p.add_argument("--out", help="where to write it (default <root>/label_sheet.txt)")
    p.add_argument("--cases", help="only these case ids, comma separated")
    p.add_argument("--annotator")
    p.add_argument("--window", type=float, default=30.0, help="transcript window in seconds (default 30)")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("import", help="write a filled label sheet back into the cases")
    p.add_argument("path")
    p.add_argument("--annotator", help="whose labels these are (default: the sheet's 'annotator:' line)")
    p.set_defaults(func=_cmd_import)

    p = sub.add_parser("harvest-reviews", help="weak labels from Accept/Reject in published shorts' review sheets")
    p.add_argument("--final", help="folder of published shorts (default vod_output/final)")
    p.add_argument("--rejected", help="folder of QC rejects (default vod_output/rejected)")
    p.set_defaults(func=_cmd_harvest)

    p = sub.add_parser("variants", help="list the variants (env-file overrides) that can be run")
    p.set_defaults(func=_cmd_variants)

    p = sub.add_parser("run", help="run a variant on the cases (cached per case and repeat) and score it")
    p.add_argument("--variant", default=variants.BASELINE, help="baseline or <root>/variants/<name>.env")
    p.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    p.add_argument("--cases", help="only these case ids, comma separated")
    p.add_argument("--repeats", type=int, default=1, help="runs per case (LLM output varies; 3 shows the spread)")
    p.add_argument("--stage", choices=("clip", "full"), default="clip",
                   help="clip = Stage 2 on the frozen transcript (default); full = the whole pipeline (expensive)")
    _money(p)
    p.add_argument("--name", help="run name (default <variant>__<configuration hash>)")
    p.add_argument("--force", action="store_true", help="ignore cached results of this run")
    p.add_argument("--include-unlabeled", action="store_true", help="also run cases whose labels are not finished")
    p.add_argument("--keep-workspace", action="store_true", help="--stage full: keep the isolated code copy")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("score", help="re-score runs against the current labels (no model call)")
    p.add_argument("names", nargs="+")
    p.set_defaults(func=_cmd_score)

    p = sub.add_parser("runs", help="list runs")
    p.set_defaults(func=_cmd_runs)

    p = sub.add_parser("compare", help="compare a baseline run (A) with a candidate run (B)")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--gate", action="store_true", help="exit 1 if holdout HIT@1 or Recall@3 drops by > threshold")
    p.add_argument("--gate-threshold", type=float, default=0.1)
    p.add_argument("--judge", action="store_true", help="also ask the pairwise LLM judge (dev videos only; PAID)")
    _money(p)
    p.add_argument("--out", help="report path (default <root>/compare.md)")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("calibrate", help="blind human verdicts on judge pairs -> agreement and Cohen's kappa")
    p.add_argument("runs", nargs="*", help="baseline and candidate run whose dev pairs to add")
    p.add_argument("--text", action="store_true", help="label in the terminal instead of the browser")
    p.add_argument("--judge", action="store_true", help="let the judge answer the pairs and compute kappa (PAID)")
    _money(p)
    p.add_argument("--seed", type=int, default=None, help="side assignment seed (default random)")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("feedback", help="write the dev-only, AI-readable per-video feedback of a run")
    p.add_argument("run")
    p.add_argument("--out")
    p.set_defaults(func=_cmd_feedback)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
