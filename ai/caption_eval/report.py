"""Human-readable caption evaluation reports (markdown)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _crit(aggregate: Mapping[str, Any], kind: str) -> str:
    row = (aggregate.get("critical") or {}).get(kind) or {}
    return f"{_pct(row.get('rate'))} ({row.get('errors', 0)}/{row.get('total', 0)})"


def _timing(aggregate: Mapping[str, Any]) -> tuple[str, str, str]:
    timing = aggregate.get("timing") or {}
    if not timing:
        return "n/a", "n/a", "n/a"
    refs = ", ".join(timing.get("references") or [])
    return (f"{timing['median_ms']:.0f} ms (p90 {timing['p90_ms']:.0f} ms)", _pct(timing.get("within_100ms")),
            f"{timing['words']} words vs {refs} reference")


SUMMARY_ROWS = (
    ("WER (lower is better)", lambda a: _pct(a.get("wer"))),
    ("substitutions / deletions / insertions", lambda a: f"{a.get('sub', 0)} / {a.get('del', 0)} / {a.get('ins', 0)}"),
    ("critical word error rate", lambda a: _pct(a.get("critical_error_rate"))),
    ("names wrong or lost", lambda a: _crit(a, "name")),
    ("numbers wrong or lost", lambda a: _crit(a, "number")),
    ("negations wrong or lost", lambda a: _crit(a, "negation")),
    ("invented numbers / negations", lambda a: str(a.get("critical_insertions", 0))),
    ("dropped phrases (>=3 words)", lambda a: f"{a.get('dropped_phrases', 0)} ({a.get('dropped_phrase_words', 0)} words)"),
    ("invented phrases (>=3 words)", lambda a: f"{a.get('invented_phrases', 0)} ({a.get('invented_phrase_words', 0)} words)"),
    ("word onset error, median", lambda a: _timing(a)[0]),
    ("onsets within 100 ms", lambda a: _timing(a)[1]),
    ("'uncertain' flags that were real errors", lambda a: _pct(a.get("uncertain_flag_precision"))),
    ("real errors flagged 'uncertain'", lambda a: _pct(a.get("uncertain_flag_recall"))),
    ("coverage flags on a real error", lambda a: f"{_pct(a.get('coverage_flag_precision'))} of {a.get('coverage_flags', 0)}"),
    ("dropped phrases a coverage flag pointed at", lambda a: _pct(a.get("dropped_phrases_flagged"))),
)


def run_report(result: Mapping[str, Any]) -> str:
    aggregate = result.get("aggregate") or {}
    overrides = result.get("overrides") or {}
    lines = [f"# Caption eval: {result.get('run')}", "",
             f"- System: `{result.get('system')}`" + (f" with {', '.join(f'`{k}={v}`' for k, v in overrides.items())}"
                                                       if overrides else ""),
             f"- Code: `{result.get('git_commit') or 'unknown'}` - scored {result.get('scored_at')}",
             f"- Cases scored: {aggregate.get('cases', 0)} ({aggregate.get('gold_words', 0)} gold words); "
             f"skipped: {len(result.get('skipped') or [])}", ""]
    if len(result.get("configurations") or []) > 1:
        lines += [f"> WARNING: the scored hypotheses come from {len(result['configurations'])} different "
                  "configurations (settings or caption code changed between runs). Re-run with `--force` "
                  "before comparing.", ""]
    timing = aggregate.get("timing") or {}
    if timing:
        lines.append(f"- Timing: {_timing(aggregate)[2]}")
        lines.append("")
    lines += ["## Summary", "", "| metric | value |", "| --- | --- |"]
    lines += [f"| {label} | {fn(aggregate)} |" for label, fn in SUMMARY_ROWS]
    lines += ["", "## Per case", "",
              "| case | gold words | WER | S | D | I | critical | dropped | invented | onset median | degradations |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    for row in sorted(result.get("cases") or [], key=lambda r: -float(r.get("wer", 0.0))):
        crit = row.get("critical") or {}
        crit_errors = sum(int(v.get("errors", 0)) for v in crit.values())
        crit_total = sum(int(v.get("total", 0)) for v in crit.values())
        onsets = sorted(row.get("onset_errors_s") or [])
        median = f"{1000 * onsets[len(onsets) // 2]:.0f} ms" if onsets else "n/a"
        lines.append(f"| {row['case_id']} | {row['gold_words']} | {_pct(row['wer'])} | {row['sub']} | {row['del']} | "
                     f"{row['ins']} | {crit_errors}/{crit_total} | {len(row.get('dropped_phrases') or [])} | "
                     f"{len(row.get('invented_phrases') or [])} | {median} | "
                     f"{'; '.join(row.get('degradations') or []) or '-'} |")
    subs = aggregate.get("top_substitutions") or []
    if subs:
        lines += ["", "## Most frequent substitutions (gold -> system)", "", "| gold | system | count |",
                  "| --- | --- | ---: |"]
        lines += [f"| {a} | {b} | {n} |" for a, b, n in subs]
    lines += ["", "## Errors to listen to", ""]
    for row in result.get("cases") or []:
        interesting = [e for e in row.get("errors") or [] if e.get("kind") or e["op"] != "ins"][:15]
        phrases = row.get("dropped_phrases") or []
        if not interesting and not phrases:
            continue
        lines.append(f"### {row['case_id']}")
        for phrase in phrases:
            lines.append(f"- {_at(phrase.get('at_s'))} DROPPED {phrase['words']} words: \"{phrase['text']}\"")
        for error in interesting:
            kind = f" [{error['kind']}]" if error.get("kind") else ""
            if error["op"] == "sub":
                what = f"\"{error['gold']}\" heard as \"{error['system']}\""
            elif error["op"] == "del":
                what = f"\"{error['gold']}\" missing"
            else:
                what = f"\"{error['system']}\" added"
            lines.append(f"- {_at(error.get('at_s'))} {what}{kind} - context: \"{error.get('context', '')}\"")
        lines.append("")
    skipped = result.get("skipped") or []
    if skipped:
        lines += ["## Skipped", ""] + [f"- {row['case_id']}: {row['reason']}" for row in skipped]
    return "\n".join(lines) + "\n"


def _at(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    minutes, rest = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(minutes)}:{rest:04.1f}"


def compare_report(results: Sequence[Mapping[str, Any]]) -> str:
    names = [str(r.get("run")) for r in results]
    lines = ["# Caption eval comparison", "",
             "Same cases, same gold, same normalization. Lower is better for every error rate.", "",
             "| metric | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
    for label, fn in SUMMARY_ROWS:
        lines.append(f"| {label} | " + " | ".join(fn(r.get("aggregate") or {}) for r in results) + " |")
    lines.append("| cases scored | " + " | ".join(str((r.get("aggregate") or {}).get("cases", 0)) for r in results)
                 + " |")
    by_case: dict[str, dict[str, float]] = {}
    for result in results:
        for row in result.get("cases") or []:
            by_case.setdefault(row["case_id"], {})[str(result.get("run"))] = float(row["wer"])
    common = [c for c, v in by_case.items() if len(v) == len(results)]
    if len(results) >= 2 and common:
        base = names[0]
        lines += ["", f"## Per-case WER (delta vs `{base}`)", "",
                  "| case | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names)]
        for case_id in sorted(common, key=lambda c: by_case[c][names[-1]] - by_case[c][base]):
            values = by_case[case_id]
            cells = [_pct(values[base])] + [
                f"{_pct(values[n])} ({100 * (values[n] - values[base]):+.1f})" for n in names[1:]]
            lines.append(f"| {case_id} | " + " | ".join(cells) + " |")
        worse = [c for c in common for n in names[1:] if by_case[c][n] - by_case[c][base] > 0.02]
        if worse:
            lines += ["", f"Regressions (> 2 WER points worse than `{base}` on a case): " + ", ".join(sorted(set(worse)))]
    if len(common) < len(by_case):
        lines += ["", f"Note: {len(by_case) - len(common)} case(s) were scored by only some runs and are left out of "
                      "the per-case table; the summary rows are only comparable when 'cases scored' match."]
    return "\n".join(lines) + "\n"
