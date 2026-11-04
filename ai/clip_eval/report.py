"""Human-readable clip evaluation reports (markdown). Dev and holdout are always reported apart.

These reports contain holdout results and are for people only. The AI-readable export is
``guard.write_feedback`` (dev only, checked in code).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from ai.clip_eval import cases as case_store
from ai.clip_eval import metrics

HOLDOUT_BANNER = ("> **Contains holdout results: for people only.** Never paste this file into an AI chat, a "
                  "prompt or an optimizer. The AI-readable export is `python -m ai.clip_eval feedback <run>` "
                  "(dev videos only).")
SMALL_SET = 20
ROWS = ("hit1", "hit1_03", "hit1_07", "recall1", "recall3", "map", "ap_03", "ap_05", "ap_07", "best_tiou",
        "covers_great", "negative_top", "negative_any", "bad_top", "mid_word_start", "mid_word_end", "duration_ok",
        "in_target", "schema_ok")
STABILITY_ROWS = (("top_change_rate", "published clip changes between repeats", False),
                  ("repeat_tiou", "tIoU between repeats (published clip)", True))


def _f(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _usd(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):.3f}"


def _secs(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} s"


def _label(name: str) -> tuple[str, bool]:
    if name in metrics.METRICS:
        return metrics.METRICS[name][1], metrics.METRICS[name][2]
    for key, label, higher in STABILITY_ROWS:
        if key == name:
            return label, higher
    return name, True


def _spread(row: Mapping[str, Any]) -> str:
    spread = row.get("spread")
    if not spread:
        return "-"
    return f"± {spread['std']:.2f} (min {spread['min']:.2f}, max {spread['max']:.2f}, {spread['repeats']} repeats)"


def _ci(ci: Sequence[float] | None) -> str:
    return "n/a" if not ci else f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def _routing(configuration: Mapping[str, Any]) -> str:
    routing = configuration.get("routing") or {}
    scout, judge, esc = (routing.get(k) or {} for k in ("scout", "judge", "escalation"))
    return (f"scout `{scout.get('model')}` [{scout.get('effort')}], judge `{judge.get('model')}` "
            f"[{judge.get('effort')}], close calls [{esc.get('effort')}]")


def _noise(videos: int) -> list[str]:
    if videos == 0:
        return []
    if videos < SMALL_SET:
        return [f"> Only {videos} video(s): one video moves a rate by {1 / videos:.2f}. Read the confidence "
                "intervals; a difference inside them may be noise.", ""]
    return []


def _split_section(split: str, data: Mapping[str, Any]) -> list[str]:
    aggregate = data.get("aggregate") or {}
    videos = int(aggregate.get("videos", 0) or 0)
    lines = [f"## {split} ({videos} video{'s' if videos != 1 else ''})", ""]
    if not videos:
        return lines + ["No scored video in this split.", ""]
    lines += _noise(videos)
    lines += ["| metric | mean over videos | 95% CI (bootstrap over videos) | spread across repeats |",
              "| --- | ---: | --- | --- |"]
    for name in ROWS:
        row = (aggregate.get("metrics") or {}).get(name) or {}
        label, higher = _label(name)
        arrow = "higher is better" if higher else "lower is better"
        lines.append(f"| {label} ({arrow}) | {_f(row.get('mean'))} | {_ci(row.get('ci95'))} | {_spread(row)} |")
    for name, label, _higher in STABILITY_ROWS:
        row = (aggregate.get("metrics") or {}).get(name) or {}
        lines.append(f"| {label} | {_f(row.get('mean'))} | {_ci(row.get('ci95'))} | - |")
    boundary = aggregate.get("boundary") or {}
    start, end = boundary.get("start") or {}, boundary.get("end") or {}
    within = boundary.get("within_s") or {}
    lines += ["", f"- Boundary error vs the matched great moment ({start.get('n', 0)} clips that overlap one): "
                  f"start median {_secs(start.get('median_s'))} (p90 {_secs(start.get('p90_s'))}), end median "
                  f"{_secs(end.get('median_s'))} (p90 {_secs(end.get('p90_s'))}); both ends within 2 / 5 / 10 s: "
                  f"{_f(within.get('2.0'))} / {_f(within.get('5.0'))} / {_f(within.get('10.0'))}",
              f"- Cost {_usd(aggregate.get('cost_usd_per_video'))} and {_secs(aggregate.get('seconds_per_video'))} "
              "per video and repeat"]
    human = data.get("human_ceiling")
    if human:
        lines.append(f"- Human ceiling ({human['pairs']} second-annotator pair(s)): tIoU between annotators on the "
                     f"same great moment {_f(human.get('tiou_mean'))}, HIT@0.5 {_f(human.get('hit_05'))}")
    weak = data.get("weak")
    if weak and weak.get("cases"):
        lines.append(f"- Weak labels (published shorts a person accepted / rejected; NOT golden labels): "
                     f"{weak['accepted_repicked']}/{weak['accepted']} accepted picks picked again, "
                     f"{weak['rejected_repicked']}/{weak['rejected']} rejected picks picked again")
    lines += ["", "### Per video", "",
              "| video | HIT@1 | R@3 | mAP | best tIoU | start / end error | mid-word | never-pick | changes | feedback |",
              "| --- | ---: | ---: | ---: | ---: | --- | :---: | :---: | ---: | --- |"]
    for case in sorted(data.get("cases") or [], key=lambda c: (c["metrics"].get("best_tiou") or 0.0)):
        m = case["metrics"]
        starts, ends = case.get("start_errors_s") or [], case.get("end_errors_s") or []
        errors = (f"{metrics._mean(starts):+.1f} / {metrics._mean(ends):+.1f} s" if starts else "no overlap")
        midword = "yes" if (m.get("mid_word_start") or 0) + (m.get("mid_word_end") or 0) > 0 else "-"
        never = "YES" if m.get("negative_top") else "-"
        feedback = str(case.get("feedback", "")).replace("|", "/")[:400]
        lines.append(f"| {case['case_id']} | {_f(m.get('hit1'))} | {_f(m.get('recall3'))} | {_f(m.get('map'))} | "
                     f"{_f(m.get('best_tiou'))} | {errors} | {midword} | {never} | "
                     f"{_f(m.get('top_change_rate'))} | {feedback} |")
    lines.append("")
    return lines


def run_report(result: Mapping[str, Any]) -> str:
    configuration = result.get("configuration") or {}
    overrides = result.get("variant_env") or {}
    code = ", ".join(f"`{str(m.get('module', '')).rsplit('.', 1)[-1]}` {str(m.get('sha256') or '-')[:8]}"
                     for m in configuration.get("code") or [])
    prices = result.get("prices") or {}
    lines = [f"# Clip eval: {result.get('run')}", "", HOLDOUT_BANNER, "",
             f"- Variant: `{result.get('variant')}`" + (
                 " with " + ", ".join(f"`{k}={v}`" for k, v in sorted(overrides.items())) if overrides
                 else " (no overrides: code defaults + .env)"),
             f"- Stage: `{result.get('stage')}`; {_routing(configuration)}",
             f"- Code: git `{result.get('git_commit') or 'unknown'}`; {code}; configuration "
             f"`{str(result.get('configuration_hash') or '')[:12]}`",
             f"- Spent on this run so far: {_usd(result.get('spent_usd'))}"
             + ("" if prices.get("verified") else " (prices unverified and set high on purpose; see "
                                                  "ai/clip_eval/pricing.json)"),
             f"- Scored {result.get('scored_at')} against the current labels (metrics v{result.get('metrics_version')})",
             ""]
    for split in case_store.SPLITS:
        lines += _split_section(split, (result.get("splits") or {}).get(split) or {})
    skipped = result.get("skipped") or []
    if skipped:
        lines += ["## Not scored", ""] + [f"- {row['case_id']}: {row['reason']}" for row in skipped] + [""]
    return "\n".join(lines) + "\n"


# ============================================================
# COMPARISON
# ============================================================

def configuration_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """What differs between two runs' configurations (routing, settings, env, code)."""
    out: list[str] = []
    ca, cb = a.get("configuration") or {}, b.get("configuration") or {}
    for role in ("scout", "judge", "escalation"):
        ra, rb = (ca.get("routing") or {}).get(role) or {}, (cb.get("routing") or {}).get(role) or {}
        if ra != rb:
            out.append(f"{role}: `{ra.get('model')}` [{ra.get('effort')}] -> `{rb.get('model')}` [{rb.get('effort')}]")
    for group in ("settings", "constraints", "env"):
        ga, gb = ca.get(group) or {}, cb.get(group) or {}
        for key in sorted(set(ga) | set(gb)):
            if ga.get(key) != gb.get(key):
                out.append(f"{group} {key}: `{ga.get(key)}` -> `{gb.get(key)}`")
    ea, eb = a.get("variant_env") or {}, b.get("variant_env") or {}
    for key in sorted(set(ea) | set(eb)):
        if ea.get(key) != eb.get(key):
            out.append(f"variant {key}: `{ea.get(key, '(unset)')}` -> `{eb.get(key, '(unset)')}`")
    code_a = {m.get("module"): m.get("sha256") for m in ca.get("code") or []}
    code_b = {m.get("module"): m.get("sha256") for m in cb.get("code") or []}
    for module in sorted(set(code_a) | set(code_b)):
        if code_a.get(module) != code_b.get(module):
            out.append(f"code `{module}`: {str(code_a.get(module) or '-')[:8]} -> {str(code_b.get(module) or '-')[:8]}")
    if a.get("stage") != b.get("stage"):
        out.append(f"stage: {a.get('stage')} -> {b.get('stage')}")
    return out


def _clip_text(case: case_store.Case | None, top: Sequence[float] | None, limit: int = 45) -> str:
    if case is None or not top:
        return "(no clip)"
    words = case_store.text_between(case.words(), float(top[0]), float(top[1])).split()
    if len(words) > limit:
        words = words[: limit // 2] + ["..."] + words[-(limit // 2):]
    text = " ".join(words) or "(no transcript words)"
    return f"{case_store.clock(top[0])}-{case_store.clock(top[1])}: \"{text}\""


def _tops(case_row: Mapping[str, Any]) -> list[Sequence[float]]:
    return [r.get("top") for r in case_row.get("repeat_rows") or [] if r.get("top")]


def _compare_split(split: str, a: Mapping[str, Any], b: Mapping[str, Any],
                   cases: Mapping[str, case_store.Case]) -> list[str]:
    rows_a = {c["case_id"]: c for c in ((a.get("splits") or {}).get(split) or {}).get("cases") or []}
    rows_b = {c["case_id"]: c for c in ((b.get("splits") or {}).get(split) or {}).get("cases") or []}
    common = sorted(set(rows_a) & set(rows_b))
    lines = [f"## {split} ({len(common)} video{'s' if len(common) != 1 else ''} scored by both runs)", ""]
    only = sorted(set(rows_a) ^ set(rows_b))
    if only:
        lines += [f"> Scored by only one run (left out of every number below): {', '.join(only)}", ""]
    if not common:
        return lines + ["Nothing to compare in this split.", ""]
    lines += _noise(len(common))
    agg_a = ((a.get("splits") or {}).get(split) or {}).get("aggregate") or {}
    agg_b = ((b.get("splits") or {}).get(split) or {}).get("aggregate") or {}
    lines += [f"| metric | A `{a.get('run')}` | B `{b.get('run')}` | B - A | 95% CI of B - A | |",
              "| --- | --- | --- | ---: | --- | --- |"]
    for name in [*ROWS, *(r[0] for r in STABILITY_ROWS)]:
        label, higher = _label(name)
        va = {k: rows_a[k]["metrics"].get(name) for k in common}
        vb = {k: rows_b[k]["metrics"].get(name) for k in common}
        delta = metrics.paired_delta(va, vb)
        if delta["delta"] is None:
            continue
        ma = metrics._mean(va.values())
        mb = metrics._mean(vb.values())
        better = (delta["delta"] > 0) == higher
        verdict = "may be noise" if delta["maybe_noise"] else ("better" if better else "WORSE")
        if abs(delta["delta"]) < 1e-9:
            verdict = "same"
        sa = _spread(((agg_a.get("metrics") or {}).get(name) or {})).split(" (")[0]
        sb = _spread(((agg_b.get("metrics") or {}).get(name) or {})).split(" (")[0]
        lines.append(f"| {label} | {_f(ma)} {sa if sa != '-' else ''} | {_f(mb)} {sb if sb != '-' else ''} | "
                     f"{delta['delta']:+.2f} | {_ci(delta['ci95'])} | {verdict} |")
    cost_a = metrics._mean(rows_a[k].get("cost_usd") for k in common)
    cost_b = metrics._mean(rows_b[k].get("cost_usd") for k in common)
    time_a = metrics._mean(rows_a[k].get("elapsed_s") for k in common)
    time_b = metrics._mean(rows_b[k].get("elapsed_s") for k in common)
    lines += [f"| cost per video and repeat | {_usd(cost_a)} | {_usd(cost_b)} | | | |",
              f"| seconds per video and repeat | {_secs(time_a)} | {_secs(time_b)} | | | |", "",
              "### Per video", "", "| video | HIT@1 A -> B | R@3 A -> B | best tIoU A -> B | A published | B published |",
              "| --- | --- | --- | --- | --- | --- |"]

    def delta_key(case_id: str) -> tuple[float, float, float]:
        ma, mb = rows_a[case_id]["metrics"], rows_b[case_id]["metrics"]
        return tuple((mb.get(k) or 0.0) - (ma.get(k) or 0.0) for k in ("hit1", "best_tiou", "recall3"))  # type: ignore

    for case_id in sorted(common, key=delta_key):
        ma, mb = rows_a[case_id]["metrics"], rows_b[case_id]["metrics"]
        top_a, top_b = _tops(rows_a[case_id]), _tops(rows_b[case_id])
        span = (lambda t: f"{case_store.clock(t[0][0])}-{case_store.clock(t[0][1])}" if t else "-")
        lines.append(f"| {case_id} | {_f(ma.get('hit1'))} -> {_f(mb.get('hit1'))} | {_f(ma.get('recall3'))} -> "
                     f"{_f(mb.get('recall3'))} | {_f(ma.get('best_tiou'))} -> {_f(mb.get('best_tiou'))} | "
                     f"{span(top_a)} | {span(top_b)} |")
    regressed = [c for c in sorted(common, key=delta_key) if any(v < -1e-9 for v in delta_key(c))][:3]
    lines += ["", "### Most regressed", ""]
    if not regressed:
        lines.append("No video got worse on HIT@1, best tIoU or Recall@3.")
    for case_id in regressed:
        dk = delta_key(case_id)
        case = cases.get(case_id)
        lines.append(f"**{case_id}**: HIT@1 {dk[0]:+.2f}, best tIoU {dk[1]:+.2f}, Recall@3 {dk[2]:+.2f}")
        gold = case.gold() if case else None
        for moment in (gold.great if gold else [])[:3]:
            lines.append(f"- great moment {case_store.clock(moment.start)}-{case_store.clock(moment.end)}"
                         + (f": {moment.note}" if moment.note else ""))
        top_a, top_b = _tops(rows_a[case_id]), _tops(rows_b[case_id])
        lines.append(f"- A published {_clip_text(case, top_a[0] if top_a else None)}")
        lines.append(f"- B published {_clip_text(case, top_b[0] if top_b else None)}")
        lines.append(f"- B feedback: {rows_b[case_id].get('feedback', '')[:500]}")
        lines.append("")
    lines.append("")
    return lines


def judge_section(judge: Mapping[str, Any]) -> list[str]:
    lines = ["## LLM judge (Tier 2, dev videos only)", ""]
    trust = judge.get("calibration") or {}
    if judge.get("trusted"):
        lines.append(f"Judge `{judge.get('model')}` is calibrated: Cohen's kappa {_f(trust.get('kappa'))} on "
                     f"{trust.get('n')} human-labeled pairs (>= 0.6).")
    else:
        lines.append(f"**UNTRUSTED**: judge `{judge.get('model')}` "
                     + (f"has kappa {_f(trust.get('kappa'))} on {trust.get('n')} pairs (< 0.6)" if trust.get("n")
                        else "is not calibrated (run `calibrate`)")
                     + ". These verdicts are shown for information; the gate ignores them.")
    lines += ["", f"- Pairs judged (both orders each): {judge.get('pairs', 0)}; B wins {judge.get('wins', 0)}, "
                  f"A wins {judge.get('losses', 0)}, ties {judge.get('ties', 0)} "
                  f"(of which order-disagreements {judge.get('order_disagreements', 0)}, identical clips "
                  f"{judge.get('identical', 0)})",
              f"- Cost {_usd(judge.get('spent_usd'))}"]
    if judge.get("undecided"):
        why = ("dry run" if judge.get("dry_run") else judge.get("refused") or "not asked")
        lines.append(f"- Not asked: {judge['undecided']} pair(s) ({why}); {judge.get('planned_calls', 0)} uncached "
                     f"call(s) would cost ~{_usd(judge.get('planned_usd'))}")
    lines.append("")
    rows = judge.get("rows") or []
    if rows:
        lines += ["| video | repeat | verdict | A then B | B then A | reason (A then B) |", "| --- | ---: | --- | --- | --- | --- |"]
        for row in rows:
            lines.append(f"| {row['case_id']} | {row['repeat']} | {row['verdict'] or 'not asked'} | "
                         f"{row.get('ab') or '-'} | {row.get('ba') or '-'} | "
                         f"{str(row.get('reason', '')).replace('|', '/')[:200]} |")
    return lines + [""]


def compare_report(a: Mapping[str, Any], b: Mapping[str, Any], *, cases: Mapping[str, case_store.Case],
                   gate: Mapping[str, Any] | None = None, judge: Mapping[str, Any] | None = None) -> str:
    lines = [f"# Clip eval comparison: A `{a.get('run')}` vs B `{b.get('run')}`", "", HOLDOUT_BANNER, "",
             "A is the reference (baseline), B the candidate. Same videos, same labels, same metrics; "
             "every number is computed only over videos both runs scored.", "",
             "What differs between the runs:", ""]
    diff = configuration_diff(a, b)
    lines += [f"- {row}" for row in diff] or ["- nothing recorded (same configuration: differences are LLM noise)"]
    lines.append("")
    for split in case_store.SPLITS:
        lines += _compare_split(split, a, b, cases)
    if judge:
        lines += judge_section(judge)
    if gate is not None:
        lines += ["## Regression gate (holdout)", ""]
        if not gate.get("checks"):
            lines.append(f"**FAIL**: {gate.get('reason')}")
        else:
            lines.append(f"**{'PASS' if gate['passed'] else 'FAIL'}**: B may not lose more than {gate['threshold']:.2f} "
                         f"on holdout HIT@1 or Recall@3 ({gate['videos']} holdout videos).")
            for check in gate["checks"]:
                lines.append(f"- {check['label']}: {check['delta']:+.2f} (95% CI {_ci(check['ci95'])})"
                             + ("  <- FAIL" if check["failed"] else ""))
            if gate.get("judge"):
                judge_check = gate["judge"]
                lines.append("- LLM judge: " + (judge_check["ignored"] if "ignored" in judge_check else
                                                f"net loss rate {judge_check['net_loss_rate']:+.2f}"
                                                + ("  <- FAIL" if judge_check["failed"] else "")))
        lines.append("")
    return "\n".join(lines) + "\n"
