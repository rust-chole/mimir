"""Speaker truth (who spoke each word) and optional identity truth (real names).

Audio speaker, visible face track and real identity are separate facts:
diarization produces stable anonymous ids (S1, S2, ...); a face track is linked
to a speaker only by audiovisual evidence (``mimir.vision.speaker_link``);
a real name exists only when the user confirmed it (``identity``).
"""
