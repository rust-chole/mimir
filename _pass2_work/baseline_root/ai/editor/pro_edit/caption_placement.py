"""Deterministic caption placement: one position per page (or per cluster of
simultaneous pages), never frame by frame.

Pages that overlap in time (dual/multi-speaker lanes) form a cluster and move
as one stack, so lanes can never overprint. Each cluster is scored for every
candidate zone of the output aspect:

    total = platform_overlap + face_overlap + story_region + story_subject
          + action_region + layout_region + ui_overlap
          + activity_overlap + edge + aesthetic_bias        (per cluster)
          + placement_switch                                 (between clusters)

Hard constraints: the whole stack inside the frame (with the platform edge
padding) and outside hard platform regions; otherwise the zone is invalid.
Evidence priority is encoded in the weights: story region > story subject >
face (= layout face-cam region) > action region > static UI / text-like
occupancy > activity > aesthetic bias. Layout classification may scale the UI
and activity weights (never above the face weight). A Viterbi pass over clusters with a switch
penalty keeps positions stable (a switch has to buy a material reduction).
With no evidence the bias makes every page BOTTOM, i.e. exactly V3.1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai.editor.pro_edit.caption_occupancy import VisualOccupancyMap
from ai.editor.pro_edit.caption_platform import GENERIC, NormBox, PlatformSafeZoneProfile

PLACEMENT_VERSION = 2

# Evidence weights (priority story region > story subject > face > activity > bias).
# Costs are coverage fractions: covering ~15% of a face (chin/mouth) already
# outweighs moving from the V3.1 bottom anchor to another zone.
W_STORY_REGION = 6.0
W_STORY_SUBJECT = 5.0
W_FACE = 4.0
W_ACTION = 3.5                     # story span x activity / detector hotspot (inferred, below faces)
W_UI = 3.0                         # persistent TEXT_LIKE / HUD occupancy (x persistence)
W_LAYOUT_REGION = 4.0              # face-cam box of a GAMEPLAY_FACE_CAM layout (a face)
W_ACTIVITY = 1.0
MAX_LAYOUT_MULTIPLIER = 1.5
EDGE_COST = 0.1
PREFERRED_BONUS = 0.2
SWITCH_COST = 0.8
RELAXED_SWITCH_COST = 0.4          # across a scene change or a >= 1.5 s caption gap
RELAX_GAP_S = 1.5
BUSY_THRESHOLD = 0.35              # activity left under the chosen position -> "busy background"
EVIDENCE_PAD_S = 0.15
MAX_SAMPLES = 24


@dataclass(frozen=True)
class ZoneSpec:
    name: str
    kind: str          # "bottom" (V3.1 anchor) | "center" | "top"
    value: float       # center / top ratio (unused for bottom)
    bias: float


ZONES_BY_PROFILE: Mapping[str, tuple[ZoneSpec, ...]] = {
    "vertical": (ZoneSpec("bottom", "bottom", 0.0, 0.0), ZoneSpec("lower_middle", "center", 0.60, 0.35),
                 ZoneSpec("upper_middle", "center", 0.40, 0.55), ZoneSpec("top", "top", 0.10, 0.75)),
    "landscape": (ZoneSpec("bottom", "bottom", 0.0, 0.0), ZoneSpec("top", "top", 0.06, 0.6)),
    "square": (ZoneSpec("bottom", "bottom", 0.0, 0.0), ZoneSpec("top", "top", 0.07, 0.6)),
    "generic_preserve": (ZoneSpec("bottom", "bottom", 0.0, 0.0), ZoneSpec("lower_middle", "center", 0.62, 0.35),
                         ZoneSpec("top", "top", 0.08, 0.7)),
}


@dataclass(frozen=True)
class TimedBox:
    start: float
    end: float
    box: NormBox                # OUTPUT-frame normalized
    kind: str                   # face | story_region | story_subject | action_region | layout_region
    weight: float = 1.0         # evidence confidence multiplier (action regions)


@dataclass(frozen=True)
class PlacementEvidence:
    boxes: tuple[TimedBox, ...] = ()
    occupancy: VisualOccupancyMap | None = None
    scene_changes: tuple[float, ...] = ()
    platform: PlatformSafeZoneProfile = GENERIC
    sources: tuple[str, ...] = ()
    ui: Any = None                              # TEXT_LIKE persistence map (window / region_score)
    multipliers: Mapping[str, float] = field(default_factory=dict)   # layout: {"ui": x, "activity": x}
    layout: str = ""

    @property
    def empty(self) -> bool:
        return not self.boxes and self.occupancy is None and self.ui is None \
            and not self.platform.reserved_regions and not self.platform.preferred_caption_regions

    def multiplier(self, key: str) -> float:
        value = float(self.multipliers.get(key, 1.0))
        return min(MAX_LAYOUT_MULTIPLIER, max(0.0, value))


@dataclass(frozen=True)
class PageGeometry:
    """A page at its V3.1 (bottom) position: line boxes in output pixels."""

    page_id: int
    start: float
    end: float
    boxes_px: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class PlacementDecision:
    cluster_id: int
    page_ids: tuple[int, ...]
    zone: str
    delta_px: float
    cost: float
    breakdown: Mapping[str, float]
    busy_background: bool = False
    fallback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"cluster": self.cluster_id, "pages": list(self.page_ids), "zone": self.zone,
                "delta_px": round(self.delta_px, 1), "cost": round(self.cost, 4),
                "breakdown": {k: round(v, 4) for k, v in self.breakdown.items() if abs(v) > 1e-9},
                "busy_background": self.busy_background, "fallback": self.fallback}


@dataclass
class PlacementReport:
    policy: str
    zones: dict[str, int] = field(default_factory=dict)
    switches: int = 0
    clusters: int = 0
    fallbacks: list[str] = field(default_factory=list)
    evidence: tuple[str, ...] = ()
    decisions: list[PlacementDecision] = field(default_factory=list)
    layout: str = ""

    def to_dict(self, limit: int = 200) -> dict[str, Any]:
        return {"version": PLACEMENT_VERSION, "policy": self.policy, "clusters": self.clusters,
                "zones": dict(self.zones), "switches": self.switches, "fallbacks": list(self.fallbacks),
                "evidence": list(self.evidence), "layout": self.layout,
                "decisions": [d.to_dict() for d in self.decisions[:limit]]}


def cluster_pages(pages: Sequence[PageGeometry]) -> list[list[PageGeometry]]:
    """Pages that are on screen at the same time share one position."""
    clusters: list[list[PageGeometry]] = []
    current: list[PageGeometry] = []
    current_end = -math.inf
    for page in sorted(pages, key=lambda p: (p.start, p.page_id)):
        if current and page.start < current_end - 1e-6:
            current.append(page)
            current_end = max(current_end, page.end)
        else:
            if current:
                clusters.append(current)
            current, current_end = [page], page.end
    if current:
        clusters.append(current)
    return clusters


def _norm(box: tuple[float, float, float, float], width: int, height: int, dy: float) -> NormBox:
    x0, y0, x1, y1 = box
    return NormBox(x0 / width, (y0 + dy) / height, x1 / width, (y1 + dy) / height)


def _inter(a: NormBox, b: NormBox) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def zone_delta(zone: ZoneSpec, stack_top: float, stack_bottom: float, height: int) -> float:
    """Vertical shift (px) of the whole caption stack for ``zone``."""
    if zone.kind == "bottom":
        return 0.0
    if zone.kind == "top":
        return zone.value * height - stack_top
    return zone.value * height - (stack_top + stack_bottom) / 2.0


class _ClusterScorer:
    def __init__(self, cluster: Sequence[PageGeometry], evidence: PlacementEvidence, width: int, height: int):
        self.cluster = cluster
        self.evidence = evidence
        self.width, self.height = width, height
        start = min(p.start for p in cluster)
        end = max(p.end for p in cluster)
        count = max(2, min(MAX_SAMPLES, int(math.ceil((end - start) / 0.25))))
        self.times = [start + (end - start) * (i + 0.5) / count for i in range(count)]
        self.activity = ([evidence.occupancy.window(t - 0.25, t + 0.25) for t in self.times]
                         if evidence.occupancy is not None else None)
        self.ui = ([evidence.ui.window(t - 0.25, t + 0.25) for t in self.times]
                   if evidence.ui is not None else None)

    def visible(self, t: float) -> list[PageGeometry]:
        return [p for p in self.cluster if p.start <= t < p.end] or list(self.cluster[:1])

    def score(self, dy: float) -> tuple[float, dict[str, float]] | None:
        platform = self.evidence.platform
        pad = platform.minimum_edge_padding
        parts = {"platform": 0.0, "face": 0.0, "story_region": 0.0, "story_subject": 0.0, "action_region": 0.0,
                 "layout_region": 0.0, "ui": 0.0, "activity": 0.0, "edge": 0.0, "preferred": 0.0}
        weights = {"face": W_FACE, "story_region": W_STORY_REGION, "story_subject": W_STORY_SUBJECT,
                   "action_region": W_ACTION, "layout_region": W_LAYOUT_REGION}
        for page in self.cluster:
            for raw in page.boxes_px:
                box = _norm(raw, self.width, self.height, dy)
                if box.y0 < pad - 1e-9 or box.y1 > 1.0 - pad + 1e-9 or box.x0 < -1e-9 or box.x1 > 1.0 + 1e-9:
                    return None
                for region in platform.reserved_regions:
                    if region.mode == "hard" and _inter(region.box, box) > 0:
                        return None
        for index, t in enumerate(self.times):
            boxes = [_norm(raw, self.width, self.height, dy) for page in self.visible(t) for raw in page.boxes_px]
            area = sum(b.area for b in boxes) or 1.0
            for region in platform.reserved_regions:
                parts["platform"] += region.weight * sum(_inter(region.box, b) for b in boxes) / area
            if platform.preferred_caption_regions:
                inside = all(any(pref.contains(b) for pref in platform.preferred_caption_regions) for b in boxes)
                parts["preferred"] += -PREFERRED_BONUS if inside else PREFERRED_BONUS
            for timed in self.evidence.boxes:
                if not timed.start - EVIDENCE_PAD_S <= t <= timed.end + EVIDENCE_PAD_S:
                    continue
                covered = min(1.0, sum(_inter(timed.box, b) for b in boxes) / max(timed.box.area, 1e-9))
                kind = timed.kind if timed.kind in weights else "face"
                parts[kind] += weights[kind] * covered * min(1.0, max(0.0, timed.weight))
            if self.ui is not None:
                values = self.ui[index]
                parts["ui"] += W_UI * self.evidence.multiplier("ui") * sum(
                    self.evidence.ui.region_score((b.x0, b.y0, b.x1, b.y1), values) * b.area
                    for b in boxes) / area
            if self.activity is not None:
                values = self.activity[index]
                parts["activity"] += W_ACTIVITY * self.evidence.multiplier("activity") * sum(
                    self.evidence.occupancy.region_score((b.x0, b.y0, b.x1, b.y1), values) * b.area
                    for b in boxes) / area
            if any(b.y0 < 2 * pad or b.y1 > 1.0 - 2 * pad for b in boxes):
                parts["edge"] += EDGE_COST
        samples = float(len(self.times))
        parts = {key: value / samples for key, value in parts.items()}
        return sum(parts.values()), parts


def solve_placement(pages: Sequence[PageGeometry], evidence: PlacementEvidence, *, width: int, height: int,
                    profile_name: str, policy: str = "auto") -> tuple[dict[int, float], PlacementReport]:
    """page_id -> vertical shift (px) + report. ``policy == "bottom"`` keeps V3.1."""
    report = PlacementReport(policy=policy, evidence=evidence.sources, layout=evidence.layout)
    clusters = cluster_pages(pages)
    report.clusters = len(clusters)
    zones = ZONES_BY_PROFILE.get(profile_name, ZONES_BY_PROFILE["generic_preserve"])
    if policy != "auto":
        zones = zones[:1]
    all_boxes = [b for p in pages for b in p.boxes_px]
    if not all_boxes:
        return {}, report
    stack_top = min(b[1] for b in all_boxes)
    stack_bottom = max(b[3] for b in all_boxes)
    deltas = [zone_delta(zone, stack_top, stack_bottom, height) for zone in zones]

    table: list[list[tuple[float, dict[str, float]] | None]] = []
    for cluster in clusters:
        scorer = _ClusterScorer(cluster, evidence, width, height)
        row = []
        for zone, dy in zip(zones, deltas):
            scored = scorer.score(dy)
            if scored is not None:
                total, parts = scored
                parts["bias"] = zone.bias
                scored = (total + zone.bias, parts)
            row.append(scored)
        table.append(row)

    def switch_cost(prev: list[PageGeometry], nxt: list[PageGeometry]) -> float:
        boundary_a = max(p.end for p in prev)
        boundary_b = min(p.start for p in nxt)
        relaxed = boundary_b - boundary_a >= RELAX_GAP_S or any(
            boundary_a - 0.1 <= t <= boundary_b + 0.1 for t in evidence.scene_changes)
        return RELAXED_SWITCH_COST if relaxed else SWITCH_COST

    # Viterbi over clusters; a cluster with no valid zone is forced to bottom (V3.1).
    inf = math.inf
    best: list[list[float]] = []
    back: list[list[int]] = []
    forced: set[int] = set()
    for c, row in enumerate(table):
        costs = [entry[0] if entry is not None else inf for entry in row]
        if all(v == inf for v in costs):
            costs = [0.0] + [inf] * (len(row) - 1)
            forced.add(c)
        if c == 0:
            best.append(costs)
            back.append([0] * len(row))
            continue
        step = switch_cost(clusters[c - 1], clusters[c])
        current, pointers = [], []
        for z, cost in enumerate(costs):
            options = [best[-1][p] + (0.0 if p == z else step) for p in range(len(row))]
            chosen = min(range(len(options)), key=lambda p: (options[p], p))
            current.append(options[chosen] + cost)
            pointers.append(chosen)
        best.append(current)
        back.append(pointers)
    path = [min(range(len(best[-1])), key=lambda z: (best[-1][z], z))]
    for c in range(len(clusters) - 1, 0, -1):
        path.append(back[c][path[-1]])
    path.reverse()

    shifts: dict[int, float] = {}
    previous_zone = None
    for c, (cluster, z) in enumerate(zip(clusters, path)):
        zone = zones[z]
        entry = table[c][z]
        cost, parts = (entry if entry is not None else (0.0, {}))
        fallback = "no_valid_zone_kept_bottom" if c in forced else ""
        if fallback:
            report.fallbacks.append(f"cluster {c}: {fallback}")
        busy = evidence.occupancy is not None and \
            parts.get("activity", 0.0) / (W_ACTIVITY * evidence.multiplier("activity") or 1.0) >= BUSY_THRESHOLD
        decision = PlacementDecision(c, tuple(p.page_id for p in cluster), zone.name, deltas[z], cost, parts, busy,
                                     fallback)
        report.decisions.append(decision)
        report.zones[zone.name] = report.zones.get(zone.name, 0) + 1
        if previous_zone is not None and zone.name != previous_zone:
            report.switches += 1
        previous_zone = zone.name
        for page in cluster:
            shifts[page.page_id] = deltas[z]
    return shifts, report
