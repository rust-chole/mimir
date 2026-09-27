"""Edit architecture: WHAT MUST THE VIEWER SEE RIGHT NOW?

    EditContextBuilder -> AI Edit Director -> EditPlan -> Validation -> Deterministic Edit Compiler

The director chooses editorial INTENT per span (a small vocabulary); it never
produces coordinates or commands. The compiler computes every crop, zoom,
position, movement, smoothing, timing and bound deterministically, and never
crops out story-critical people, actions, objects or gameplay to zoom onto a
speaker.
"""
