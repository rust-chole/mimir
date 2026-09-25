"""Local subject tracking for Pro Edit (no network, no per-frame API calls).

Responsibilities are split on purpose:

    frames.py       DETECTION INPUT   sparse frame decode (ffmpeg rawvideo pipe)
    detectors.py    DETECTION         YuNet (if model file present) / OpenCV Haar
    association.py  TRACK ASSOCIATION IoU + center distance + appearance matching
    tracker.py      TEMPORAL TRACKING sparse detect, optical-flow propagation,
                                      re-detect triggers, lost hold, reacquisition
    provider.py     CACHED PROVIDER   runs the tracker once per paced clip and
                                      stores the standard subject sidecar

Speaker <-> track association lives in ../speaker_link.py; smoothing and
hysteresis in ../subjects.py; composition in ../framing.py.
"""
