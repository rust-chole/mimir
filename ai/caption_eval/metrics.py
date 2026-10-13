"""Scoring a caption hypothesis against a human gold transcript.

Normalization is the caption stack's own definition of a harmless difference
(``ai.caption_stack.normalize``): case, punctuation, fillers, "ok"/"okay",
"gonna"/"going to", "twenty-one"/"21" never count as errors; another word, a lost
negation, another number or a misspelled name always does.

Metrics per case (micro-summed over cases in ``aggregate``):

    WER                 (substitutions + deletions + insertions) / gold words
    critical errors     gold names / numbers / negations substituted or dropped,
                        plus inserted numbers / negations
    dropped phrases     runs of >= PHRASE_RUN consecutive gold words the system lost
    invented phrases    runs of >= PHRASE_RUN consecutive words the system added
    onset timing        |system start - gold start| of correctly heard words
                        (only when the case has gold word times)
    uncertain flags     precision / recall of the words the system marked uncertain
    coverage flags      share of the stack's speech-coverage flags that sit on a real
                        error, and share of dropped phrases they point at

Gold markup: lines starting with ``#`` are comments; ``[...]`` marks speech even a
human cannot make out, and matches any system words (or none) at no cost.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai.caption_stack import lexical, normalize

METRICS_VERSION = 1
PHRASE_RUN = 3
WILDCARD = "<*>"
_WILDCARD_TOKEN = "zzmimirwildcardzz"
_BRACKETS = re.compile(r"\[[^\]]*\]")
NEGATIONS = frozenset({"not", "no", "never", "nobody", "nothing", "none", "neither", "nor", "nowhere"})
_NOT_NAMES = frozenset({"i", "i'm", "i'll", "i've", "i'd", "ok", "okay", "oh", "god"})
FLAG_NEAR_S = 0.30


@dataclass(frozen=True)
class Token:
    key: str                     # canonical comparison key (WILDCARD for gold [...] markup)
    display: str                 # the source token it came from
    source: int                  # index of that source token
    start: float | None = None
    end: float | None = None
    flagged: bool = False        # the system marked this word uncertain
    kind: str = ""               # gold only: name | number | negation | ""


def gold_body(text: str) -> str:
    """Gold file text without comment lines."""
    return "\n".join(line for line in str(text).splitlines() if not line.lstrip().startswith("#"))


def gold_display_tokens(text: str) -> list[str]:
    """Display tokens of a gold text; every ``[...]`` becomes one wildcard token."""
    return lexical.display_tokens(_BRACKETS.sub(f" {_WILDCARD_TOKEN} ", gold_body(text)))


def _sentence_initial(tokens: Sequence[str], index: int) -> bool:
    return index == 0 or tokens[index - 1].rstrip().endswith((".", "!", "?", ":", '"'))


def _canonical(displays: Sequence[str], *, times: Sequence[tuple[float | None, float | None]] | None = None,
               flags: Sequence[bool] | None = None, kinds: Sequence[str] | None = None) -> list[Token]:
    keys: list[str] = []
    owners: list[int] = []
    for index, token in enumerate(displays):
        if token == _WILDCARD_TOKEN:
            keys.append(WILDCARD)
            owners.append(index)
            continue
        for key in [*lexical.token_keys(token), *normalize.symbol_keys(token)]:
            keys.append(key)
            owners.append(index)
    out: list[Token] = []
    for key, owner in normalize.canonical_pairs(keys, owners):
        start, end = (times[owner] if times else (None, None))
        display = "[...]" if key == WILDCARD else displays[owner]
        out.append(Token(key, display, owner, start, end, bool(flags[owner]) if flags else False,
                         kinds[owner] if kinds and key != WILDCARD else ""))
    return out


def gold_tokens(text: str, *, names: Sequence[str] = (),
                times: Sequence[tuple[float, float]] | None = None) -> list[Token]:
    """Canonical gold tokens with their critical class (name / number / negation)."""
    displays = gold_display_tokens(text)
    name_keys = {k for name in names for k in lexical.keys_of(lexical.display_tokens(name))}
    kinds: list[str] = []
    for index, token in enumerate(displays):
        keys = lexical.token_keys(token)
        letters = [ch for ch in token if ch.isalpha()]
        capitalized = bool(letters) and letters[0].isupper()
        if any(k in name_keys for k in keys) or (
                capitalized and not _sentence_initial(displays, index)
                and not set(keys) <= _NOT_NAMES and not (token.isupper() and len(letters) <= 3)):
            kinds.append("name")
        elif any(any(ch.isdigit() for ch in k) for k in normalize.canonical_sequence(keys)):
            kinds.append("number")
        else:
            kinds.append("")
    wild_free = [t for t in displays if t != _WILDCARD_TOKEN]
    full_times: list[tuple[float | None, float | None]] | None = None
    if times is not None and len(times) == len(wild_free):
        iterator = iter(times)
        full_times = [(None, None) if t == _WILDCARD_TOKEN else tuple(next(iterator)) for t in displays]
    tokens = _canonical(displays, times=full_times, kinds=kinds)
    # A negation is a canonical key ("don't" -> do, not): classify after canonicalization.
    return [Token(t.key, t.display, t.source, t.start, t.end, t.flagged,
                  "negation" if t.key in NEGATIONS and t.kind != "name" else t.kind) for t in tokens]


def hypothesis_tokens(words: Sequence[Mapping[str, Any]]) -> list[Token]:
    """Canonical tokens of system words (``word``, optional ``start``/``end`` and ``status``/``uncertain``)."""
    displays: list[str] = []
    times: list[tuple[float | None, float | None]] = []
    flags: list[bool] = []
    for word in words:
        parts = lexical.display_tokens(str(word.get("word", "")))
        start = word.get("start")
        end = word.get("end")
        flagged = str(word.get("status", "")) == "uncertain" or bool(word.get("uncertain"))
        for part in parts:
            displays.append(part)
            times.append((float(start) if start is not None else None, float(end) if end is not None else None))
            flags.append(flagged)
    return _canonical(displays, times=times, flags=flags)


# ============================================================
# ALIGNMENT (Levenshtein with a gold wildcard)
# ============================================================

def align(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, int | None, int | None]]:
    """Minimum edit alignment -> [(op, ref index, hyp index)]; op in equal | sub | del | ins | wild."""
    n, m = len(ref), len(hyp)
    inf = 10 ** 9
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = j, "ins"
    for i in range(1, n + 1):
        wild = ref[i - 1] == WILDCARD
        row, prev = cost[i], cost[i - 1]
        brow = back[i]
        row[0], brow[0] = (prev[0], "wskip") if wild else (prev[0] + 1, "del")
        for j in range(1, m + 1):
            if wild:
                best, how = prev[j], "wskip"                      # the wildcard absorbs nothing more
                if row[j - 1] < best:
                    best, how = row[j - 1], "wtake"               # ... or one more system word
            else:
                best, how = prev[j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1), "diag"
                if prev[j] + 1 < best:
                    best, how = prev[j] + 1, "del"
                if row[j - 1] + 1 < best:
                    best, how = row[j - 1] + 1, "ins"
            row[j], brow[j] = best, how
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        how = back[i][j]
        if how == "diag":
            ops.append(("equal" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif how == "del":
            ops.append(("del", i - 1, None))
            i -= 1
        elif how == "ins":
            ops.append(("ins", None, j - 1))
            j -= 1
        elif how == "wtake":
            ops.append(("wild", i - 1, j - 1))
            j -= 1
        else:                                                    # wskip
            i -= 1
    ops.reverse()
    return ops


# ============================================================
# SCORE
# ============================================================

@dataclass
class CaseScore:
    case_id: str
    gold_words: int = 0
    hyp_words: int = 0
    sub: int = 0
    dele: int = 0
    ins: int = 0
    critical: dict[str, list[int]] = field(default_factory=lambda: {"name": [0, 0], "number": [0, 0],
                                                                     "negation": [0, 0]})
    critical_insertions: int = 0
    dropped_phrases: list[dict[str, Any]] = field(default_factory=list)
    invented_phrases: list[dict[str, Any]] = field(default_factory=list)
    onset_errors: list[float] = field(default_factory=list)
    timing_reference: str = ""
    flagged: int = 0
    flagged_wrong: int = 0
    wrong_hyp: int = 0
    wrong_hyp_flagged: int = 0
    coverage_flags: int = 0
    coverage_flags_on_errors: int = 0
    dropped_phrases_flagged: int = 0
    substitutions: Counter = field(default_factory=Counter)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def errors_total(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def wer(self) -> float:
        return self.errors_total / max(1, self.gold_words)

    def to_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "gold_words": self.gold_words, "hyp_words": self.hyp_words,
                "sub": self.sub, "del": self.dele, "ins": self.ins, "wer": round(self.wer, 4),
                "critical": {k: {"errors": v[0], "total": v[1]} for k, v in self.critical.items()},
                "critical_insertions": self.critical_insertions,
                "dropped_phrases": self.dropped_phrases, "invented_phrases": self.invented_phrases,
                "onset_errors_s": [round(x, 4) for x in self.onset_errors], "timing_reference": self.timing_reference,
                "uncertain_flags": {"flagged": self.flagged, "flagged_wrong": self.flagged_wrong,
                                    "wrong": self.wrong_hyp, "wrong_flagged": self.wrong_hyp_flagged},
                "coverage_flags": {"flags": self.coverage_flags, "on_errors": self.coverage_flags_on_errors,
                                   "dropped_phrases_flagged": self.dropped_phrases_flagged},
                "substitutions": [[a, b, n] for (a, b), n in self.substitutions.most_common()],
                "errors": self.errors}


def _phrase_runs(ops: Sequence[tuple[str, int | None, int | None]], kind: str) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for index, (op, _i, _j) in enumerate(ops):
        if op == kind:
            current.append(index)
            continue
        if op in ("equal", "sub", "wild") or (op == "del" and kind == "ins") or (op == "ins" and kind == "del"):
            if len(current) >= PHRASE_RUN:
                runs.append(current)
            current = []
    if len(current) >= PHRASE_RUN:
        runs.append(current)
    return runs


def _error_time(op: tuple[str, int | None, int | None], ref: Sequence[Token], hyp: Sequence[Token],
                ops: Sequence[tuple[str, int | None, int | None]], position: int) -> float | None:
    """Where an error is on the case clock (gold time, else the system's, else a neighbour's)."""
    _kind, i, j = op
    if i is not None and ref[i].start is not None:
        return ref[i].start
    if j is not None and hyp[j].start is not None:
        return hyp[j].start
    for step in range(1, 6):                                       # a deletion: the nearest system word
        for k in (position - step, position + step):
            if 0 <= k < len(ops) and ops[k][2] is not None and hyp[ops[k][2]].start is not None:
                return hyp[ops[k][2]].start
    return None


def _context(ref: Sequence[Token], i: int | None, width: int = 4) -> str:
    if i is None or not ref:
        return ""
    lo, hi = max(0, i - width), min(len(ref), i + width + 1)
    return " ".join(t.display for t in ref[lo:hi])


def score_case(case_id: str, gold: Sequence[Token], hyp: Sequence[Token], *,
               coverage: Mapping[str, Any] | None = None, timing_reference: str = "") -> CaseScore:
    ops = align([t.key for t in gold], [t.key for t in hyp])
    score = CaseScore(case_id, gold_words=sum(1 for t in gold if t.key != WILDCARD), hyp_words=len(hyp),
                      timing_reference=timing_reference)
    wrong_hyp_positions: set[int] = set()
    error_times: list[float] = []
    for position, (op, i, j) in enumerate(ops):
        g = gold[i] if i is not None else None
        h = hyp[j] if j is not None else None
        if g is not None and g.kind:
            score.critical[g.kind][1] += 1
        if op == "equal":
            if g is not None and h is not None and g.start is not None and h.start is not None:
                score.onset_errors.append(abs(h.start - g.start))
            continue
        if op == "wild":
            continue
        if op == "sub":
            score.sub += 1
            score.substitutions[(g.display, h.display)] += 1
        elif op == "del":
            score.dele += 1
        else:
            score.ins += 1
            if h is not None and (h.key in NEGATIONS or any(ch.isdigit() for ch in h.key)):
                score.critical_insertions += 1
        if g is not None and g.kind:
            score.critical[g.kind][0] += 1
        if j is not None:
            wrong_hyp_positions.add(j)
        at = _error_time((op, i, j), gold, hyp, ops, position)
        if at is not None:
            error_times.append(at)
        if len(score.errors) < 200:
            score.errors.append({"op": op, "gold": g.display if g else "", "system": h.display if h else "",
                                 "kind": g.kind if g else "", "at_s": round(at, 2) if at is not None else None,
                                 "context": _context(gold, i if i is not None else _nearest_ref(ops, position))})

    for kind, target in (("del", score.dropped_phrases), ("ins", score.invented_phrases)):
        for run in _phrase_runs(ops, kind):
            words = [gold[ops[k][1]].display if kind == "del" else hyp[ops[k][2]].display for k in run]
            at = _error_time(ops[run[0]], gold, hyp, ops, run[0])
            end = _error_time(ops[run[-1]], gold, hyp, ops, run[-1])
            target.append({"words": len(words), "text": " ".join(words), "at_s": round(at, 2) if at is not None
                           else None, "end_s": round(end, 2) if end is not None else None})

    score.flagged = sum(1 for t in hyp if t.flagged)
    score.flagged_wrong = sum(1 for j, t in enumerate(hyp) if t.flagged and j in wrong_hyp_positions)
    score.wrong_hyp = len(wrong_hyp_positions)
    score.wrong_hyp_flagged = sum(1 for j in wrong_hyp_positions if hyp[j].flagged)

    intervals = coverage_intervals(coverage or {})
    score.coverage_flags = len(intervals)
    score.coverage_flags_on_errors = sum(
        1 for a, b in intervals if any(a - FLAG_NEAR_S <= t <= b + FLAG_NEAR_S for t in error_times))
    for phrase in score.dropped_phrases:
        a = phrase["at_s"]
        b = phrase["end_s"] if phrase["end_s"] is not None else a
        if a is not None and any(not (b < x - FLAG_NEAR_S or a > y + FLAG_NEAR_S) for x, y in intervals):
            score.dropped_phrases_flagged += 1
    return score


def _nearest_ref(ops: Sequence[tuple[str, int | None, int | None]], position: int) -> int | None:
    for step in range(0, 8):
        for k in (position - step, position + step):
            if 0 <= k < len(ops) and ops[k][1] is not None:
                return ops[k][1]
    return None


def coverage_intervals(coverage: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Case-clock intervals of the stack's speech-coverage flags."""
    out: list[tuple[float, float]] = []
    for key in ("uncovered_speech", "stretched_words", "unsupported_words"):
        for row in coverage.get(key, []) or []:
            try:
                out.append((float(row["start"]), float(row["end"])))
            except (KeyError, TypeError, ValueError):
                continue
    return out


# ============================================================
# AGGREGATE
# ============================================================

def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def aggregate(scores: Sequence[CaseScore]) -> dict[str, Any]:
    """Micro-summed metrics over cases (every gold word weighs the same)."""
    gold = sum(s.gold_words for s in scores)
    sub, dele, ins = sum(s.sub for s in scores), sum(s.dele for s in scores), sum(s.ins for s in scores)
    critical = {kind: [sum(s.critical[kind][0] for s in scores), sum(s.critical[kind][1] for s in scores)]
                for kind in ("name", "number", "negation")}
    crit_errors = sum(v[0] for v in critical.values()) + sum(s.critical_insertions for s in scores)
    crit_total = sum(v[1] for v in critical.values())
    onsets = [x for s in scores for x in s.onset_errors]
    flagged = sum(s.flagged for s in scores)
    wrong = sum(s.wrong_hyp for s in scores)
    dropped = [p for s in scores for p in s.dropped_phrases]
    cov_flags = sum(s.coverage_flags for s in scores)
    subs: Counter = Counter()
    for s in scores:
        subs.update(s.substitutions)
    result: dict[str, Any] = {
        "cases": len(scores), "gold_words": gold, "sub": sub, "del": dele, "ins": ins,
        "wer": _ratio(sub + dele + ins, gold),
        "case_wer_mean": round(statistics.fmean(s.wer for s in scores), 4) if scores else None,
        "critical": {k: {"errors": v[0], "total": v[1], "rate": _ratio(v[0], v[1])} for k, v in critical.items()},
        "critical_error_rate": _ratio(crit_errors, crit_total),
        "critical_insertions": sum(s.critical_insertions for s in scores),
        "dropped_phrases": len(dropped), "dropped_phrase_words": sum(p["words"] for p in dropped),
        "invented_phrases": sum(len(s.invented_phrases) for s in scores),
        "invented_phrase_words": sum(p["words"] for s in scores for p in s.invented_phrases),
        "uncertain_flag_precision": _ratio(sum(s.flagged_wrong for s in scores), flagged),
        "uncertain_flag_recall": _ratio(sum(s.wrong_hyp_flagged for s in scores), wrong),
        "coverage_flags": cov_flags,
        "coverage_flag_precision": _ratio(sum(s.coverage_flags_on_errors for s in scores), cov_flags),
        "dropped_phrases_flagged": _ratio(sum(s.dropped_phrases_flagged for s in scores), len(dropped)),
        "top_substitutions": [[a, b, n] for (a, b), n in subs.most_common(25)],
    }
    if onsets:
        ordered = sorted(onsets)
        result["timing"] = {
            "words": len(onsets), "mean_ms": round(1000 * statistics.fmean(onsets), 1),
            "median_ms": round(1000 * statistics.median(onsets), 1),
            "p90_ms": round(1000 * ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 1),
            "within_100ms": _ratio(sum(1 for x in onsets if x <= 0.100), len(onsets)),
            "within_250ms": _ratio(sum(1 for x in onsets if x <= 0.250), len(onsets)),
            "references": sorted({s.timing_reference for s in scores if s.onset_errors}),
        }
    return result
