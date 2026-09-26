"""Canonical timeline: explicit source-range -> output-range mapping (frame quantized).

Source time is never rewritten. Every downstream layer (captions, camera,
effects, audio, QC) maps through :class:`mimir.timeline.schema.Timeline`.
"""
