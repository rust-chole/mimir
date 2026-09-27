"""Camera paths: easing, cut-aware transitions, follow smoothing and invariants.

Per output frame the camera is either a single window (``Window``) or the
stacked facecam/gameplay layout (``StackState``). Inside a shot every change is
eased; at shot cuts and timeline segment boundaries the framing may change
instantly (it is a cut already). Following a moving subject uses a dead zone,
exponential smoothing and a speed cap, so detector noise never shakes the frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence, Union

from mimir.edit.framing import Geometry, Window


@dataclass(frozen=True)
class StackState:
    top: tuple[float, float, float, float]      # facecam source box (panel aspect)
    bottom: Window                              # gameplay window (bottom panel aspect)


State = Union[Window, StackState]


def smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def ease_out_cubic(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return 1 - (1 - u) ** 3


EASINGS: dict[str, Callable[[float], float]] = {"smooth": smoothstep, "punch": ease_out_cubic}


def lerp_window(a: Window, b: Window, u: float) -> Window:
    # interpolate h in log space so zoom speed feels uniform
    h = math.exp(math.log(a.h) + (math.log(b.h) - math.log(a.h)) * u)
    return Window(a.cx + (b.cx - a.cx) * u, a.cy + (b.cy - a.cy) * u, h)


def lerp_state(a: State, b: State, u: float) -> State:
    if isinstance(a, Window) and isinstance(b, Window):
        return lerp_window(a, b, u)
    if isinstance(a, StackState) and isinstance(b, StackState):
        top = tuple(x + (y - x) * u for x, y in zip(a.top, b.top))
        return StackState(top, lerp_window(a.bottom, b.bottom, u))  # type: ignore[arg-type]
    return b


def follow(goals: Sequence[Window], fps: int, *, deadzone: float, max_speed: float, geo: Geometry,
           tau: float = 0.40) -> list[Window]:
    """Dead-zone + exponential smoothing + speed cap on the window center (zoom unchanged)."""
    if not goals:
        return []
    alpha = 1 - math.exp(-1.0 / (fps * tau))
    step_cap = max_speed / fps
    out = [goals[0]]
    x, y = goals[0].cx, goals[0].cy
    for goal in goals[1:]:
        tx, ty = goal.cx, goal.cy
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if dist <= deadzone:
            tx, ty = x, y
        else:
            scale = (dist - deadzone) / dist
            tx, ty = x + dx * scale, y + dy * scale
        nx, ny = x + (tx - x) * alpha, y + (ty - y) * alpha
        step = math.hypot(nx - x, ny - y)
        if step > step_cap:
            nx, ny = x + (nx - x) * step_cap / step, y + (ny - y) * step_cap / step
        x, y = nx, ny
        out.append(Window(x, y, goal.h))
    return out


@dataclass
class SpanPlan:
    span_id: str
    frames: tuple[int, int]
    goals: list[State]
    hard_start: bool
    easing: str
    transition_frames: int
    follow: bool


def build_path(plans: Sequence[SpanPlan]) -> tuple[list[State], list[str]]:
    states: list[State] = []
    phases: list[str] = []
    previous: State | None = None
    for plan in plans:
        f0, f1 = plan.frames
        count = f1 - f0
        hard = plan.hard_start or previous is None or type(previous) is not type(plan.goals[0])
        length = 0 if hard else min(count, max(1, plan.transition_frames))
        start_state = previous
        for k in range(count):
            goal = plan.goals[k]
            if k < length and start_state is not None:
                u = EASINGS[plan.easing]((k + 1) / length)
                states.append(lerp_state(start_state, goal, u))
                phases.append("transition")
            else:
                states.append(goal)
                phases.append("cut" if (k == 0 and hard) else ("follow" if plan.follow else "hold"))
        previous = states[-1]
    return states, phases


def center_steps(states: Sequence[State], phases: Sequence[str]) -> list[float]:
    """Per-frame center movement (source widths) for frames that are neither cuts nor transitions."""
    steps = []
    for index in range(1, len(states)):
        a, b = states[index - 1], states[index]
        if phases[index] in ("cut", "transition"):
            continue
        wa = a.bottom if isinstance(a, StackState) else a
        wb = b.bottom if isinstance(b, StackState) else b
        steps.append(math.hypot(wb.cx - wa.cx, wb.cy - wa.cy))
    return steps
