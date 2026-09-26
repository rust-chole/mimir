"""Editorial shot intents (the director's whole vocabulary)."""
from __future__ import annotations

from enum import Enum


class Intent(str, Enum):
    HOLD = "HOLD"                          # keep the current framing (a deliberate non-move)
    WIDE_CONTEXT = "WIDE_CONTEXT"          # everything story-relevant visible
    TWO_SHOT = "TWO_SHOT"                  # two (or more) participants together
    SPEAKER_MEDIUM = "SPEAKER_MEDIUM"      # the active speaker, shoulders and hands
    SPEAKER_PUNCH = "SPEAKER_PUNCH"        # quick push onto the speaker at a peak line
    REACTION = "REACTION"                  # the reacting person
    ACTION_REGION = "ACTION_REGION"        # where the physical action/object is
    GAMEPLAY_PRIORITY = "GAMEPLAY_PRIORITY"  # gameplay owns the frame (facecam kept)
    SCREEN_PRIORITY = "SCREEN_PRIORITY"    # on-screen UI/text owns the frame (never cropped)


# Downgrade ladder used by validation when evidence cannot support an intent.
DOWNGRADE = {
    Intent.SPEAKER_PUNCH: Intent.SPEAKER_MEDIUM,
    Intent.SPEAKER_MEDIUM: Intent.WIDE_CONTEXT,
    Intent.REACTION: Intent.WIDE_CONTEXT,
    Intent.TWO_SHOT: Intent.WIDE_CONTEXT,
    Intent.ACTION_REGION: Intent.WIDE_CONTEXT,
    Intent.GAMEPLAY_PRIORITY: Intent.WIDE_CONTEXT,
    Intent.SCREEN_PRIORITY: Intent.WIDE_CONTEXT,
    Intent.HOLD: Intent.WIDE_CONTEXT,
    Intent.WIDE_CONTEXT: Intent.WIDE_CONTEXT,
}

SINGLE_SUBJECT = {Intent.SPEAKER_MEDIUM, Intent.SPEAKER_PUNCH, Intent.REACTION}
INTENSITIES = ("subtle", "normal", "strong")
