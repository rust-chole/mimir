"""ASR prompts (lexical ears only; the timing ear never receives a prompt)."""
from __future__ import annotations

from typing import Sequence

TRANSCRIPTION_PROMPT = """
This is English gaming, livestream, streamer, YouTube or Twitch content.
Transcribe exactly what is spoken. Do not summarize, rewrite grammar, censor profanity,
remove repetitions or invent speech. Preserve slang, profanity, repeated words, brand names,
usernames, game names, streamer vocabulary and unfinished sentences.
The word "chat" usually refers to the livestream audience; do not change it into the name
"Chad" unless the audio clearly refers to a person named Chad.
Only transcribe speech that is actually audible.
""".strip()


def names_context(names: Sequence[str]) -> str:
    clean = [n for n in dict.fromkeys(" ".join(str(n).split()) for n in names) if n]
    if not clean:
        return ""
    return ("VERIFIED SPELLINGS (reference only): " + ", ".join(clean[:12]) + ". If the audio clearly says one "
            "of these, use this spelling exactly. Never force a name when the audio says something else.")


def micro_prompt(before: str, after: str, instruction: str, names: Sequence[str]) -> str:
    parts = ["Transcribe exactly the speech in this short excerpt. Audio always wins over context.",
             instruction.strip()]
    if before.strip():
        parts.append(f"Preceding words (context only): {before.strip()[-240:]}")
    if after.strip():
        parts.append(f"Following words (context only): {after.strip()[:240]}")
    context = names_context(names)
    if context:
        parts.append(context)
    return "\n".join(p for p in parts if p)
