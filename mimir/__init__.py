"""MIMIR: VOD -> vertical Short with a mandatory peak cold open and a complete causal story.

One production path:

    Source Video -> Media Probe -> Transcript Truth -> Story Discovery -> Story Package
    -> Speaker Resolution -> Caption Truth -> Selected-Short Visual Analysis
    -> Cold Open Planner -> Canonical Timeline -> Edit Context -> AI Edit Director
    -> Edit Plan Validator -> Deterministic Edit Compiler -> Captions / Effects
    -> Render -> Final Quality Control -> Publish

``mimir.pipeline`` declares the stages; ``mimir.core.runner`` only orchestrates
them (signatures, caching, artifacts). Each subsystem package owns its truth.
"""

__version__ = "1.0.0"
