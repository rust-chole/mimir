from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)


DEFAULT_TERRA_MODEL = "gpt-5.6-terra"
DEFAULT_LUNA_MODEL = "gpt-5.6-luna"
# Astra 6: the strongest tier. Spent only where the artifact itself is made:
# caption lexical truth and the edit director. Never on scouting/classification.
DEFAULT_ASTRA_MODEL = "gpt-6-astra"

_ALLOWED_REASONING_EFFORTS = {
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
}


def _env_model(name: str, default: str) -> str:
    value = str(os.getenv(name, default)).strip()
    return value or default


def _env_effort(name: str, default: str) -> str:
    value = str(os.getenv(name, default)).strip().lower()
    if value not in _ALLOWED_REASONING_EFFORTS:
        raise RuntimeError(
            f"{name} geçersiz: {value!r}. Geçerli değerler: "
            + ", ".join(sorted(_ALLOWED_REASONING_EFFORTS))
        )
    return value


# ============================================================
# MIMIR ROUTING V2
# ============================================================
# The strongest intelligence goes where the published artifact is created:
#   CAPTIONS (Astra decides WHAT was said from the acoustic evidence; the
#             measured word clock alone decides WHEN)
#   EDITING  (Astra decides editorial INTENT; the deterministic compiler owns
#             every crop / zoom / pixel)
# Story judgement stays on Terra; scouting, drafting, classification and
# factual support stay on Luna. No model reviews the final short: a human does.
# Every role remains individually overrideable from .env.

# Caption lexical truth: disputed-span resolution (caption stage) and the
# verified-name decisions written into the frozen caption truth JSON.
CAPTION_JUDGE_MODEL = _env_model("MIMIR_CAPTION_JUDGE_MODEL", DEFAULT_ASTRA_MODEL)
CAPTION_JUDGE_REASONING_EFFORT = _env_effort("MIMIR_CAPTION_JUDGE_REASONING", "high")

# AI Edit Director (Pro Edit planner): editorial intent only (enums, subject ids,
# story-span ids, word ids). Crop coordinates / zoom / geometry are never its output.
EDIT_DIRECTOR_MODEL = _env_model("MIMIR_EDIT_DIRECTOR_MODEL", DEFAULT_ASTRA_MODEL)
EDIT_DIRECTOR_REASONING_EFFORT = _env_effort("MIMIR_EDIT_DIRECTOR_REASONING", "high")

EDITOR_MODEL = _env_model("MIMIR_EDITOR_MODEL", DEFAULT_TERRA_MODEL)

# Clip pipeline: Luna scouts/builds candidates; Terra owns final selection and
# final story coverage.  This removes multiple unnecessary Terra-high calls
# without giving up the final editorial authority.
CLIP_SCOUT_MODEL = _env_model("MIMIR_CLIP_SCOUT_MODEL", DEFAULT_LUNA_MODEL)
CLIP_SCOUT_REASONING_EFFORT = _env_effort("MIMIR_CLIP_SCOUT_REASONING", "medium")
CLIP_JUDGE_MODEL = _env_model("MIMIR_CLIP_JUDGE_MODEL", EDITOR_MODEL)
CLIP_JUDGE_REASONING_EFFORT = _env_effort("MIMIR_CLIP_JUDGE_REASONING", "medium")
CLIP_JUDGE_ESCALATION_REASONING_EFFORT = _env_effort("MIMIR_CLIP_JUDGE_ESCALATION_REASONING", "high")

# Backward-compatible aliases used by metadata/older callers.
CLIP_MODEL = CLIP_JUDGE_MODEL
CLIP_REASONING_EFFORT = CLIP_JUDGE_REASONING_EFFORT

# Teaser selection is constrained by real words + deterministic audio/visual
# peaks, so Luna-medium is the default.  The downstream intro judge can still
# reject a weak teaser/intro pair.
TEASER_MODEL = _env_model("MIMIR_TEASER_MODEL", DEFAULT_LUNA_MODEL)
TEASER_REASONING_EFFORT = _env_effort("MIMIR_TEASER_REASONING", "low")
TEASER_REVIEW_REASONING_EFFORT = _env_effort("MIMIR_TEASER_REVIEW_REASONING", "medium")

# Intro: Luna writes candidates/repairs; Terra is used only as the hard final
# 8/10 judge.
INTRO_DRAFT_MODEL = _env_model("MIMIR_INTRO_DRAFT_MODEL", DEFAULT_LUNA_MODEL)
INTRO_DRAFT_REASONING_EFFORT = _env_effort("MIMIR_INTRO_DRAFT_REASONING", "medium")
INTRO_JUDGE_MODEL = _env_model("MIMIR_INTRO_JUDGE_MODEL", EDITOR_MODEL)
INTRO_JUDGE_REASONING_EFFORT = _env_effort("MIMIR_INTRO_JUDGE_REASONING", "medium")

# Backward-compatible aliases.
INTRO_MODEL = INTRO_JUDGE_MODEL
INTRO_REASONING_EFFORT = INTRO_JUDGE_REASONING_EFFORT

# Speaker-role classification is a tiny text-only decision over diarization
# segments. Luna-low is deliberately used here: no audio/video generation and
# no expensive editorial reasoning.
SPEAKER_ROLE_MODEL = _env_model("MIMIR_SPEAKER_ROLE_MODEL", DEFAULT_LUNA_MODEL)
SPEAKER_ROLE_REASONING_EFFORT = _env_effort("MIMIR_SPEAKER_ROLE_REASONING", "low")

# Meme editorial decisions are low-risk/reversible; Luna-medium is enough.
MEME_MODEL = _env_model("MIMIR_MEME_MODEL", DEFAULT_LUNA_MODEL)
MEME_REASONING_EFFORT = _env_effort("MIMIR_MEME_REASONING", "medium")
MEME_DISCOVERY_MODEL = _env_model("MIMIR_MEME_DISCOVERY_MODEL", DEFAULT_LUNA_MODEL)
MEME_DISCOVERY_REASONING_EFFORT = _env_effort("MIMIR_MEME_DISCOVERY_REASONING", "low")

# Gemini quota fallback is factual visual support, not final editorial judgement.
# Use Luna-medium for frame batches; Terra still judges the evidence later.
VISUAL_SUPPORT_MODEL = _env_model("MIMIR_VISUAL_SUPPORT_MODEL", DEFAULT_LUNA_MODEL)
VISUAL_SUPPORT_REASONING_EFFORT = _env_effort("MIMIR_VISUAL_SUPPORT_REASONING", "medium")


def model_plan() -> dict[str, dict[str, str]]:
    return {
        "caption_judge": {"model": CAPTION_JUDGE_MODEL, "effort": CAPTION_JUDGE_REASONING_EFFORT},
        "edit_director": {"model": EDIT_DIRECTOR_MODEL, "effort": EDIT_DIRECTOR_REASONING_EFFORT},
        "editor": {"model": EDITOR_MODEL, "effort": "story authority"},
        "clip_scout": {"model": CLIP_SCOUT_MODEL, "effort": CLIP_SCOUT_REASONING_EFFORT},
        "clip_judge": {"model": CLIP_JUDGE_MODEL, "effort": CLIP_JUDGE_REASONING_EFFORT},
        "clip_judge_escalation": {"model": CLIP_JUDGE_MODEL, "effort": CLIP_JUDGE_ESCALATION_REASONING_EFFORT},
        "teaser": {"model": TEASER_MODEL, "effort": TEASER_REASONING_EFFORT},
        "teaser_review": {"model": TEASER_MODEL, "effort": TEASER_REVIEW_REASONING_EFFORT},
        "intro_draft": {"model": INTRO_DRAFT_MODEL, "effort": INTRO_DRAFT_REASONING_EFFORT},
        "intro_judge": {"model": INTRO_JUDGE_MODEL, "effort": INTRO_JUDGE_REASONING_EFFORT},
        "speaker_role": {"model": SPEAKER_ROLE_MODEL, "effort": SPEAKER_ROLE_REASONING_EFFORT},
        "meme": {"model": MEME_MODEL, "effort": MEME_REASONING_EFFORT},
        "meme_discovery": {"model": MEME_DISCOVERY_MODEL, "effort": MEME_DISCOVERY_REASONING_EFFORT},
        "visual_support": {"model": VISUAL_SUPPORT_MODEL, "effort": VISUAL_SUPPORT_REASONING_EFFORT},
    }


def print_model_plan() -> None:
    plan = model_plan()
    print()
    print("🤖 MIMIR MODEL PLANI (Astra: captions + editing)")
    print("-" * 66)
    print(f"CAPTION JUDGE/ASTRA: {plan['caption_judge']['model']} [{plan['caption_judge']['effort']}] (text only; clock is measured)")
    print(f"EDIT DIRECTOR/ASTRA: {plan['edit_director']['model']} [{plan['edit_director']['effort']}] (intent only; compiler owns geometry)")
    print(f"CLIP SCOUT / LUNA : {plan['clip_scout']['model']} [{plan['clip_scout']['effort']}]")
    print(f"CLIP JUDGE / TERRA: {plan['clip_judge']['model']} [{plan['clip_judge']['effort']}] -> ambiguous only [{plan['clip_judge_escalation']['effort']}]")
    print(f"TEASER / LUNA     : {plan['teaser']['model']} [{plan['teaser']['effort']}] -> review [{plan['teaser_review']['effort']}]")
    print(f"INTRO DRAFT / LUNA: {plan['intro_draft']['model']} [{plan['intro_draft']['effort']}]")
    print(f"INTRO JUDGE/TERRA : {plan['intro_judge']['model']} [{plan['intro_judge']['effort']}]")
    print(f"SPEAKER ROLE/LUNA : {plan['speaker_role']['model']} [{plan['speaker_role']['effort']}]")
    print(f"MEME / LUNA       : {plan['meme']['model']} [{plan['meme']['effort']}]")
    print(f"MEME DISCOVERY    : {plan['meme_discovery']['model']} [{plan['meme_discovery']['effort']}]")
    print(f"VISUAL SUPPORT    : {plan['visual_support']['model']} [{plan['visual_support']['effort']}]")


def validate_model_plan() -> None:
    required = {
        "CAPTION_JUDGE_MODEL": CAPTION_JUDGE_MODEL,
        "EDIT_DIRECTOR_MODEL": EDIT_DIRECTOR_MODEL,
        "EDITOR_MODEL": EDITOR_MODEL,
        "CLIP_SCOUT_MODEL": CLIP_SCOUT_MODEL,
        "CLIP_JUDGE_MODEL": CLIP_JUDGE_MODEL,
        "TEASER_MODEL": TEASER_MODEL,
        "INTRO_DRAFT_MODEL": INTRO_DRAFT_MODEL,
        "INTRO_JUDGE_MODEL": INTRO_JUDGE_MODEL,
        "SPEAKER_ROLE_MODEL": SPEAKER_ROLE_MODEL,
        "MEME_MODEL": MEME_MODEL,
        "MEME_DISCOVERY_MODEL": MEME_DISCOVERY_MODEL,
        "VISUAL_SUPPORT_MODEL": VISUAL_SUPPORT_MODEL,
    }
    empty = [name for name, value in required.items() if not str(value).strip()]
    if empty:
        raise RuntimeError("Boş model ayarı var: " + ", ".join(empty))


if __name__ == "__main__":
    validate_model_plan()
    print_model_plan()
