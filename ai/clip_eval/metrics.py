"""Scoring MIMIR's clip picks against blind human labels (Tier 1: deterministic, no model call).

Adapted from the moment-retrieval / highlight-detection literature (QVHighlights: HIT@1, mAP at
temporal-IoU thresholds). A prediction is a ranked list of clips on the source clock: the clip
the pipeline publishes first (``ranked_clips`` mirrors ``shorts_pipeline._select_clip``), then
the other candidates in Stage 2's order. ``great`` moments are the positives.

Per video and repeat:

    HIT@1 (tIoU >= 0.3 / 0.5 / 0.7)   the published clip overlaps a great moment that well
    Recall@1, Recall@3 (tIoU >= 0.5)  share of great moments captured by the top 1 / 3 clips
    AP at tIoU 0.3 / 0.5 / 0.7        ActivityNet-style interpolated AP (greedy one-to-one matching);
                                      mAP = their mean
    best tIoU                         the published clip vs its best great moment
    covers a great moment             the published clip holds >= 90 % of a great moment (a longer
                                      clip that contains the whole moment scores low on tIoU)
    start / end error                 signed seconds vs the best-matching great moment (+ = later)
    never-pick violation              the published clip (or any candidate) overlaps a never-pick range
    cuts mid-word                     the published clip starts / ends inside a transcript word
    constraint compliance             duration inside Stage 2's own bounds, inside the video, valid schema
    lands on a 'bad' moment           informational: tIoU >= 0.5 with a moment graded 'bad'

Across repeats: how often the published clip changes and the tIoU between repeats. Across videos:
mean over videos, the spread across repeats, and a bootstrap confidence interval (few videos are
noisy). Every per-video score carries a plain-text ``feedback`` line for humans and, later, for a
reflective optimizer (dev split only; see ``guard``).
"""
from __future__ import annotations

import bisect
import math
import random
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence

from ai.clip_eval.cases import Labels, clock

METRICS_VERSION = 1
THRESHOLDS = (0.3, 0.5, 0.7)
MATCH_TIOU = 0.5
COVER_FRACTION = 0.9
NEGATIVE_OVERLAP_S = 0.25
MID_WORD_EPS_S = 0.05
SAME_CLIP_TIOU = 0.5
# Mirrors ai/editor/clip_analyzer.py (MIN/MAX_CLIP_DURATION, TARGET_CLIP_MIN/MAX, ANALYZER_VERSION); a run
# records the values the analyzer it ran actually had, and those win.
DEFAULT_CONSTRAINTS = {"min_duration": 12.0, "max_duration": 55.0, "target_min": 22.0, "target_max": 42.0,
                       "analyzer_version": 2}

Span = tuple[float, float]


# ============================================================
# INTERVALS
# ============================================================

def overlap(a: Span, b: Span) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def tiou(a: Span, b: Span) -> float:
    """Temporal IoU of two [start, end] intervals; 0 for empty / inverted intervals."""
    la, lb = max(0.0, a[1] - a[0]), max(0.0, b[1] - b[0])
    inter = overlap(a, b) if la > 0 and lb > 0 else 0.0
    union = la + lb - inter
    return inter / union if union > 0 else 0.0


def _span(row: Any) -> Span:
    return float(row.start), float(row.end)


# ============================================================
# PREDICTIONS
# ============================================================

def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ranked_clips(analysis: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Stage 2's candidates in the order the pipeline would use them.

    The first one is what ``shorts_pipeline._select_clip(analysis, None)`` publishes (Terra's
    ``terra_selected`` clip, else the score / coverage / duration fallback); the rest keep Stage 2's
    list order. Candidates without a usable start/end are dropped (and fail ``schema_problems``)."""
    clips = [c for c in (analysis or {}).get("clips") or [] if isinstance(c, Mapping)]
    positions = list(enumerate(clips, start=1))
    if not positions:
        return []
    selected = [item for item in positions if item[1].get("terra_selected") is True]
    if selected:
        top = selected[0]
    else:
        def score_key(item: tuple[int, Mapping[str, Any]]) -> tuple[float, int, float, int]:
            position, clip = item
            score = _num(clip.get("score")) or 0.0
            duration = clip.get("duration")
            if duration is None and isinstance(clip.get("source"), Mapping):
                duration = clip["source"].get("duration")
            duration = _num(duration)
            covered = clip.get("covered_moment_ids", [])
            return (score, len(covered) if isinstance(covered, list) else 0,
                    -abs((999.0 if duration is None else duration) - 32.0), -position)
        top = sorted(positions, key=score_key, reverse=True)[0]
    order = [top] + [item for item in positions if item[0] != top[0]]
    out: list[dict[str, Any]] = []
    for position, clip in order:
        start, end = _num(clip.get("start")), _num(clip.get("end"))
        if start is None or end is None or end <= start:
            continue
        out.append({"position": position, "start": round(start, 3), "end": round(end, 3),
                    "duration": round(end - start, 3), "title": str(clip.get("title", ""))[:120],
                    "terra_selected": clip.get("terra_selected") is True, "score": _num(clip.get("score"))})
    return out


def schema_problems(analysis: Any, constraints: Mapping[str, Any] | None = None) -> list[str]:
    """What makes a Stage 2 output unusable or malformed (empty list = valid)."""
    constraints = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    if not isinstance(analysis, Mapping):
        return ["analysis is not a JSON object"]
    problems: list[str] = []
    expected = constraints.get("analyzer_version")
    if expected is not None and analysis.get("version") != expected:
        problems.append(f"analysis version {analysis.get('version')!r} != analyzer version {expected!r}")
    clips = analysis.get("clips")
    if not isinstance(clips, list) or not clips:
        return problems + ["no clips"]
    selected = 0
    for index, clip in enumerate(clips, start=1):
        if not isinstance(clip, Mapping):
            problems.append(f"clip {index} is not an object")
            continue
        start, end, duration = _num(clip.get("start")), _num(clip.get("end")), _num(clip.get("duration"))
        if start is None or end is None or end <= start:
            problems.append(f"clip {index}: start/end missing or end <= start")
        elif duration is not None and abs(duration - (end - start)) > 0.1:
            problems.append(f"clip {index}: duration {duration} != end - start {end - start:.3f}")
        selected += clip.get("terra_selected") is True
    if selected > 1:
        problems.append(f"{selected} clips are marked terra_selected")
    return problems


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def average_precision(predictions: Sequence[Span], truths: Sequence[Span], threshold: float) -> float | None:
    """Interpolated AP (ActivityNet / QVHighlights): ranked predictions, greedy one-to-one matching.

    None when there is no positive (the metric is undefined); 0.0 for no predictions."""
    if not truths:
        return None
    if not predictions:
        return 0.0
    used = [False] * len(truths)
    hits: list[int] = []
    for prediction in predictions:
        best, best_iou = -1, -1.0
        for index, truth in enumerate(truths):
            if used[index]:
                continue
            value = tiou(prediction, truth)
            if value >= threshold and value > best_iou:
                best, best_iou = index, value
        if best >= 0:
            used[best] = True
        hits.append(1 if best >= 0 else 0)
    recall, precision, cum = [], [], 0
    for rank, hit in enumerate(hits, start=1):
        cum += hit
        recall.append(cum / len(truths))
        precision.append(cum / rank)
    mrec, mprec = [0.0, *recall, 1.0], [0.0, *precision, 0.0]
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    return sum((mrec[i] - mrec[i - 1]) * mprec[i] for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1])


def recall_at(predictions: Sequence[Span], truths: Sequence[Span], k: int, threshold: float = MATCH_TIOU
              ) -> float | None:
    if not truths:
        return None
    top = list(predictions[:k])
    return sum(1 for truth in truths if any(tiou(p, truth) >= threshold for p in top)) / len(truths)


class WordIndex:
    """Transcript words for mid-word checks (binary search over ~10^4-10^5 words)."""

    def __init__(self, words: Sequence[tuple[float, float, str]]) -> None:
        self.words = sorted(words)
        self.starts = [w[0] for w in self.words]

    def word_at(self, t: float, eps: float = MID_WORD_EPS_S) -> tuple[float, float, str] | None:
        """The word ``t`` falls strictly inside (``eps`` from either edge counts as a clean cut)."""
        index = bisect.bisect_right(self.starts, t) - 1
        for j in (index, index - 1):
            if 0 <= j < len(self.words):
                start, end, word = self.words[j]
                if start + eps < t < end - eps:
                    return self.words[j]
        return None


def score_repeat(clips: Sequence[Mapping[str, Any]], labels: Labels, words: WordIndex | None, *,
                 duration: float | None, constraints: Mapping[str, Any] | None = None,
                 schema: Sequence[str] = ()) -> dict[str, Any]:
    """Every Tier 1 metric for one Stage 2 output (``clips`` from ``ranked_clips``) vs one video's labels."""
    constraints = {**DEFAULT_CONSTRAINTS, **(constraints or {})}
    preds: list[Span] = [(float(c["start"]), float(c["end"])) for c in clips]
    great = [_span(m) for m in labels.great]
    top = preds[0] if preds else None
    row: dict[str, Any] = {"top": list(top) if top else None, "candidates": len(preds),
                           "schema_ok": not schema and bool(preds), "schema_problems": list(schema)}
    if great:
        row["hit1"] = {str(t): (1.0 if top and max(tiou(top, g) for g in great) >= t else 0.0) for t in THRESHOLDS}
        row["recall1"] = recall_at(preds, great, 1)
        row["recall3"] = recall_at(preds, great, 3)
        row["ap"] = {str(t): average_precision(preds, great, t) for t in THRESHOLDS}
        row["map"] = sum(row["ap"].values()) / len(THRESHOLDS)
        row["best_tiou"] = max(tiou(top, g) for g in great) if top else 0.0
        row["covers_great"] = 1.0 if top and any(
            overlap(top, g) >= COVER_FRACTION * (g[1] - g[0]) for g in great) else 0.0
    else:
        row.update(hit1=None, recall1=None, recall3=None, ap=None, map=None, best_tiou=None, covers_great=None)
    match = None
    if top:
        scored = [(tiou(top, _span(m)), i, m) for i, m in enumerate(labels.moments) if m.grade == "great"]
        scored = [s for s in scored if s[0] > 0]
        if scored:
            value, index, moment = max(scored, key=lambda s: s[0])
            match = {"moment": index + 1, "tiou": round(value, 4), "start": moment.start, "end": moment.end,
                     "note": moment.note}
            row["start_error_s"] = round(top[0] - moment.start, 3)
            row["end_error_s"] = round(top[1] - moment.end, 3)
    row["matched"] = match
    row.setdefault("start_error_s", None)
    row.setdefault("end_error_s", None)
    negatives = [_span(n) for n in labels.negatives]
    row["negative_top"] = (1.0 if top and any(overlap(top, n) > NEGATIVE_OVERLAP_S for n in negatives) else 0.0)
    row["negative_any"] = (1.0 if any(overlap(p, n) > NEGATIVE_OVERLAP_S for p in preds for n in negatives)
                           else 0.0)
    row["bad_top"] = (1.0 if top and any(tiou(top, _span(m)) >= MATCH_TIOU for m in labels.moments
                                         if m.grade == "bad") else 0.0)
    cut_start = words.word_at(top[0]) if (words and top) else None
    cut_end = words.word_at(top[1]) if (words and top) else None
    row["mid_word_start"] = None if not (words and words.words and top) else (1.0 if cut_start else 0.0)
    row["mid_word_end"] = None if not (words and words.words and top) else (1.0 if cut_end else 0.0)
    row["mid_word_words"] = [cut_start[2] if cut_start else None, cut_end[2] if cut_end else None]
    if top:
        length = top[1] - top[0]
        inside = top[0] >= -0.05 and (not duration or top[1] <= float(duration) + 0.5)
        row["duration_ok"] = 1.0 if (constraints["min_duration"] - 0.05 <= length
                                     <= constraints["max_duration"] + 0.05 and inside) else 0.0
        row["in_target"] = 1.0 if constraints["target_min"] <= length <= constraints["target_max"] else 0.0
        row["top_duration_s"] = round(length, 3)
    else:
        row.update(duration_ok=0.0, in_target=0.0, top_duration_s=None)
    row["feedback"] = feedback_for(row, labels, constraints)
    return row


def stability(tops: Sequence[Sequence[float] | None]) -> dict[str, Any]:
    """How much the published clip moves between repeats of the same configuration."""
    spans = [(float(t[0]), float(t[1])) for t in tops if t]
    if len(spans) < 2:
        return {"repeats": len(spans), "top_change_rate": None, "repeat_tiou": None, "distinct_tops": len(spans)}
    pairs = [(a, b) for i, a in enumerate(spans) for b in spans[i + 1:]]
    values = [tiou(a, b) for a, b in pairs]
    distinct: list[Span] = []
    for span in spans:
        if not any(tiou(span, other) >= SAME_CLIP_TIOU for other in distinct):
            distinct.append(span)
    return {"repeats": len(spans), "top_change_rate": sum(1 for v in values if v < SAME_CLIP_TIOU) / len(values),
            "repeat_tiou": sum(values) / len(values), "distinct_tops": len(distinct)}


# ============================================================
# FEEDBACK (plain text per video: humans now, a reflective optimizer later, dev only)
# ============================================================

def feedback_for(row: Mapping[str, Any], labels: Labels, constraints: Mapping[str, Any]) -> str:
    top = row.get("top")
    if not top:
        return "Stage 2 returned no usable clip" + (
            f" ({'; '.join(row.get('schema_problems') or [])})." if row.get("schema_problems") else ".")
    parts: list[str] = []
    length = top[1] - top[0]
    head = f"Published clip {clock(top[0])}-{clock(top[1])} ({length:.1f} s)"
    match = row.get("matched")
    great = labels.great
    if match:
        note = f" '{match['note']}'" if match.get("note") else ""
        parts.append(f"{head} matches great moment {match['moment']}{note} ({clock(match['start'])}-"
                     f"{clock(match['end'])}) with tIoU {match['tiou']:.2f}.")
        start_err, end_err = row.get("start_error_s") or 0.0, row.get("end_error_s") or 0.0
        if abs(start_err) >= 1.0:
            parts.append(f"It starts {abs(start_err):.1f} s {'after' if start_err > 0 else 'before'} the labeled start"
                         + (" (the setup is cut)." if start_err > 0 else "."))
        if abs(end_err) >= 1.0:
            parts.append(f"It ends {abs(end_err):.1f} s {'before' if end_err < 0 else 'after'} the labeled end"
                         + (" (the payoff/reaction is cut)." if end_err < 0 else "."))
    elif great:
        nearest = min(great, key=lambda m: min(abs(m.start - top[0]), abs(m.end - top[1])))
        note = f" '{nearest.note}'" if nearest.note else ""
        parts.append(f"{head} overlaps no great moment; the nearest is{note} at {clock(nearest.start)}-"
                     f"{clock(nearest.end)}.")
    else:
        parts.append(f"{head}; this video has no great moment labeled.")
    if row.get("bad_top"):
        parts.append("It lands on a moment the annotator graded 'bad'.")
    if great and row.get("recall3") is not None and row["recall3"] < 1.0:
        lost = round((1.0 - float(row["recall3"])) * len(great))
        parts.append(f"{lost} of {len(great)} great moments are not among the top 3 candidates.")
    if row.get("negative_top"):
        parts.append("It overlaps a never-pick range.")
    elif row.get("negative_any"):
        parts.append("Another candidate overlaps a never-pick range.")
    words = row.get("mid_word_words") or [None, None]
    if words[0]:
        parts.append(f"It starts mid-word ('{words[0]}').")
    if words[1]:
        parts.append(f"It ends mid-word ('{words[1]}').")
    if not row.get("duration_ok"):
        parts.append(f"Its length is outside {constraints['min_duration']:g}-{constraints['max_duration']:g} s "
                     "or it leaves the video.")
    if row.get("schema_problems"):
        parts.append("Schema: " + "; ".join(row["schema_problems"][:3]) + ".")
    return " ".join(parts)


# ============================================================
# AGGREGATION
# ============================================================

def _rate(key: str) -> Callable[[Mapping[str, Any]], float | None]:
    return lambda row: row.get(key)


def _nested(key: str, sub: str) -> Callable[[Mapping[str, Any]], float | None]:
    return lambda row: (row.get(key) or {}).get(sub) if isinstance(row.get(key), Mapping) else None


# name -> (extractor, label, higher is better)
METRICS: dict[str, tuple[Callable[[Mapping[str, Any]], float | None], str, bool]] = {
    "hit1": (_nested("hit1", "0.5"), "HIT@1 (tIoU >= 0.5)", True),
    "hit1_03": (_nested("hit1", "0.3"), "HIT@1 (tIoU >= 0.3)", True),
    "hit1_07": (_nested("hit1", "0.7"), "HIT@1 (tIoU >= 0.7)", True),
    "recall1": (_rate("recall1"), "Recall@1 (tIoU >= 0.5)", True),
    "recall3": (_rate("recall3"), "Recall@3 (tIoU >= 0.5)", True),
    "map": (_rate("map"), "mAP (tIoU 0.3/0.5/0.7)", True),
    "ap_03": (_nested("ap", "0.3"), "AP @ tIoU 0.3", True),
    "ap_05": (_nested("ap", "0.5"), "AP @ tIoU 0.5", True),
    "ap_07": (_nested("ap", "0.7"), "AP @ tIoU 0.7", True),
    "best_tiou": (_rate("best_tiou"), "mean best tIoU (published clip)", True),
    "covers_great": (_rate("covers_great"), "published clip holds >= 90% of a great moment", True),
    "negative_top": (_rate("negative_top"), "never-pick violation (published clip)", False),
    "negative_any": (_rate("negative_any"), "never-pick violation (any candidate)", False),
    "bad_top": (_rate("bad_top"), "published clip on a 'bad' moment", False),
    "mid_word_start": (_rate("mid_word_start"), "starts mid-word", False),
    "mid_word_end": (_rate("mid_word_end"), "ends mid-word", False),
    "duration_ok": (_rate("duration_ok"), "duration inside Stage 2 bounds", True),
    "in_target": (_rate("in_target"), "duration inside the target range", True),
    "schema_ok": (lambda row: 1.0 if row.get("schema_ok") else 0.0, "valid Stage 2 output", True),
}
GATE_METRICS = ("hit1", "recall3")


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def percentile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values)
    if not clean:
        return None
    position = (len(clean) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return clean[low] + (clean[high] - clean[low]) * (position - low)


def bootstrap_ci(values: Sequence[float], *, resamples: int = 2000, seed: int = 7, level: float = 0.95
                 ) -> list[float] | None:
    """Percentile bootstrap CI of the mean over videos (None below two videos)."""
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 2:
        return None
    rng = random.Random(seed)
    n = len(clean)
    means = sorted(sum(clean[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    tail = (1.0 - level) / 2.0
    return [round(percentile(means, tail), 4), round(percentile(means, 1.0 - tail), 4)]


def case_summary(case_id: str, split: str, repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean over repeats of every metric for one video, plus its stability."""
    out: dict[str, Any] = {"case_id": case_id, "split": split, "repeats": len(repeats),
                           "metrics": {name: _mean(fn(r) for r in repeats) for name, (fn, _l, _h) in METRICS.items()},
                           "stability": stability([r.get("top") for r in repeats])}
    out["metrics"]["top_change_rate"] = out["stability"]["top_change_rate"]
    out["metrics"]["repeat_tiou"] = out["stability"]["repeat_tiou"]
    out["start_errors_s"] = [r["start_error_s"] for r in repeats if r.get("start_error_s") is not None]
    out["end_errors_s"] = [r["end_error_s"] for r in repeats if r.get("end_error_s") is not None]
    out["cost_usd"] = _mean(r.get("cost_usd") for r in repeats)
    out["elapsed_s"] = _mean(r.get("elapsed_s") for r in repeats)
    feedback = []
    for row in repeats:
        if row.get("feedback") and row["feedback"] not in feedback:
            feedback.append(row["feedback"])
    stab = out["stability"]
    if stab["repeats"] >= 2 and stab["distinct_tops"] > 1:
        feedback.append(f"The published clip changed across repeats ({stab['distinct_tops']} different clips in "
                        f"{stab['repeats']} repeats).")
    out["feedback"] = " | ".join(feedback)
    return out


def aggregate(cases: Sequence[Mapping[str, Any]], per_repeat: Mapping[str, Sequence[Mapping[str, Any]]]
              ) -> dict[str, Any]:
    """Split-level numbers: mean over videos, spread across repeats, bootstrap CI over videos."""
    out: dict[str, Any] = {"videos": len(cases), "metrics": {}}
    names = [*METRICS, "top_change_rate", "repeat_tiou"]
    max_repeats = max((len(rows) for rows in per_repeat.values()), default=0)
    for name in names:
        values = [c["metrics"].get(name) for c in cases if c["metrics"].get(name) is not None]
        row: dict[str, Any] = {"mean": _mean(values), "n": len(values), "ci95": bootstrap_ci(values)}
        if name in METRICS and max_repeats >= 2:
            fn = METRICS[name][0]
            by_repeat = []
            for index in range(max_repeats):
                means = [fn(rows[index]) for rows in per_repeat.values() if len(rows) > index]
                value = _mean(means)
                if value is not None:
                    by_repeat.append(value)
            if len(by_repeat) >= 2:
                row["spread"] = {"std": statistics.stdev(by_repeat), "min": min(by_repeat), "max": max(by_repeat),
                                 "repeats": len(by_repeat)}
        out["metrics"][name] = row
    # start and end errors exist together (both only when the published clip overlaps a great moment)
    pairs = [(abs(s), abs(e)) for c in cases for s, e in zip(c.get("start_errors_s") or [],
                                                             c.get("end_errors_s") or [])]
    starts, ends = [p[0] for p in pairs], [p[1] for p in pairs]
    out["boundary"] = {"start": {"median_s": percentile(starts, 0.5), "p90_s": percentile(starts, 0.9),
                                 "n": len(starts)},
                       "end": {"median_s": percentile(ends, 0.5), "p90_s": percentile(ends, 0.9), "n": len(ends)},
                       "within_s": {str(t): _mean([1.0 if s <= t and e <= t else 0.0 for s, e in pairs])
                                    for t in (2.0, 5.0, 10.0)}}
    out["cost_usd_per_video"] = _mean(c.get("cost_usd") for c in cases)
    out["seconds_per_video"] = _mean(c.get("elapsed_s") for c in cases)
    return out


def paired_delta(a: Mapping[str, float | None], b: Mapping[str, float | None]) -> dict[str, Any]:
    """Mean of (b - a) over the videos both runs scored, with a paired bootstrap CI."""
    common = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    diffs = [float(b[k]) - float(a[k]) for k in common]
    ci = bootstrap_ci(diffs)
    return {"videos": len(common), "delta": _mean(diffs), "ci95": ci,
            "maybe_noise": ci is None or (ci[0] <= 0.0 <= ci[1]), "per_video": dict(zip(common, diffs))}


# ============================================================
# HUMAN CEILING / JUDGE AGREEMENT
# ============================================================

def human_agreement(gold: Labels, other: Labels) -> dict[str, Any]:
    """How well two annotators agree on the same great moments (a ceiling for model scores)."""
    pairs = []
    for moment in gold.great:
        candidates = [(tiou(_span(moment), _span(o)), o) for o in other.great]
        candidates = [c for c in candidates if c[0] > 0]
        if candidates:
            value, match = max(candidates, key=lambda c: c[0])
            pairs.append((value, abs(moment.start - match.start), abs(moment.end - match.end)))
    return {"annotator": other.annotator, "gold_great": len(gold.great), "other_great": len(other.great),
            "matched": len(pairs), "tiou_mean": _mean(p[0] for p in pairs),
            "hit_05": (sum(1 for p in pairs if p[0] >= MATCH_TIOU) / len(gold.great)) if gold.great else None,
            "start_diff_median_s": percentile([p[1] for p in pairs], 0.5),
            "end_diff_median_s": percentile([p[2] for p in pairs], 0.5)}


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Cohen's kappa between two raters over categorical labels (e.g. judge vs human: A / B / tie)."""
    n = len(pairs)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    agree = sum(1 for x, y in pairs if x == y) / n
    categories = {x for x, _ in pairs} | {y for _, y in pairs}
    expected = sum((sum(1 for x, _ in pairs if x == c) / n) * (sum(1 for _, y in pairs if y == c) / n)
                   for c in categories)
    kappa = 1.0 if expected >= 1.0 and agree >= 1.0 else (0.0 if expected >= 1.0 else (agree - expected) /
                                                              (1.0 - expected))
    return {"n": n, "agreement": agree, "expected_agreement": expected, "kappa": kappa}
