"""Transcript truth.

Four separate authorities, never mixed:

    lexical truth   what was said        -> ASR consensus (``verify``) + evidence-gated name lock
    timing truth    when it was said     -> the immutable word clock (``align``) + backward-only PCM guard
    speaker truth   who said it          -> ``mimir.speakers`` (diarization + evidence assignment)
    identity truth  optional real names  -> ``mimir.speakers.identity`` (user-confirmed only)

Correcting spelling changes only ``text``; word ids, times and speakers are
verified unchanged after every lexical operation.
"""
