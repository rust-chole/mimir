"""Verified entity name lock (caption truth, after identity resolution).

Caption truth owns WHO is in a clip and how a VERIFIED person/entity name is
spelled. Acoustic ASR cannot separate homophonic spellings of a name (a
non-rhotic "-er" and "-a" ending, a doubled consonant, an alternative vowel),
so a confirmed participant can be written with a common variant spelling even
when every ASR ear agrees. This module is the general, evidence-gated
canonicalizer that runs on the identity-resolved final profile words:

    verified roster entity (human-confirmed participant / user-verified creator)
  + the ASR token is a close phonetic/lexical confusion of that name
  + evidence that THIS token refers to that entity (direct address,
    self-introduction, name syntax spoken by a co-participant, an alias already
    resolved by strong evidence in the same clip, or an independent ASR ear
    that emitted the verified spelling)
  + no competing roster entity is as close, the token is not an ordinary
    word that the audio can equally mean, and it is not part of ANOTHER
    capitalized full name (a different real person)
  + independent ears (with their token probabilities, when reported) do not
    confidently hear a different word
  -> only the word TEXT becomes the canonical spelling.

Nothing here is specific to a person, creator, clip or language fixture: the
roster comes only from the profile (human identity checkpoint) and from
explicitly verified extra entities (for example the ``--creator`` name).

Word id (list index), edited_start/edited_end, speaker metadata and word
order are never touched; provenance is recorded in ``word["name_lock"]`` and in
the profile's ``participant_name_lock`` audit. Every run restores the original
ASR spelling of previously locked words first, so the result is a pure
function of (ASR words, roster, evidence). Anything uncertain fails closed: the
ASR spelling is kept and the rejection reason is recorded.

Pure functions, no I/O except ``apply_to_profile_file``; deterministic.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

NAME_LOCK_VERSION = 3

# Evidence weights (deterministic gate). A correction needs at least one
# REFERENCE signal (see ``Candidate.reference_signal``) AND a total score of
# at least CORRECTION_THRESHOLD, and no veto.
WEIGHT_DIRECT_ADDRESS = 0.60
WEIGHT_SELF_INTRODUCTION = 0.60
WEIGHT_ALT_EXACT_INDEPENDENT = 0.50   # an unprompted ASR ear emitted the verified spelling
WEIGHT_CONFIRMED_ALIAS = 0.35         # same ASR variant already resolved by strong evidence in this clip
WEIGHT_CO_PARTICIPANT_SPEAKER = 0.30
WEIGHT_NAME_SYNTAX = 0.25
WEIGHT_ALT_EXACT_PROMPTED = 0.20      # a name-prompted ear emitted it (prompt-induced risk)
WEIGHT_KNOWN_NAME_FLAG = 0.15
WEIGHT_ALT_CONSISTENT = 0.10          # independent ears heard a confusable form of the name, nothing else
CORRECTION_THRESHOLD = 0.50
MIN_CONFUSION = 0.80          # candidate threshold (phonetic homophones score >= 0.85)
MIN_LEXICAL_CONFUSION = 0.80  # spelling-only similarity without a phonetic match
PHONETIC_MIN_RATIO_OTHER_ONSET = 0.75   # phonetic match with a different onset needs close spelling
AMBIGUITY_MARGIN = 0.10
CONTEXT_WINDOW = 6
LOW_ALT_PROBABILITY = 0.5     # an ear's token probability below this is weak evidence
HIGH_ALT_PROBABILITY = 0.8    # an ear confidently hearing another word vetoes a correction

STRONG_EVIDENCE = frozenset({"direct_address", "self_introduction", "confirmed_alias", "asr_alternative_exact"})
# A reference made only by syntax / by a co-participant (a third-person mention)
# additionally needs discriminating support: MIMIR's known-name detector flagged
# the token, or an ear wrote the verified spelling. Ears that merely repeat the
# confusable form cannot tell the two names apart and do not count.
SUPPORT_EVIDENCE = frozenset({"known_name_flag", "asr_alternative_prompted"})
REFERENCE_EVIDENCE = frozenset({"direct_address", "self_introduction", "name_syntax", "confirmed_alias",
                                "asr_alternative_exact"})
EVIDENCE_WEIGHTS: Mapping[str, float] = {
    "direct_address": WEIGHT_DIRECT_ADDRESS,
    "self_introduction": WEIGHT_SELF_INTRODUCTION,
    "asr_alternative_exact": WEIGHT_ALT_EXACT_INDEPENDENT,
    "confirmed_alias": WEIGHT_CONFIRMED_ALIAS,
    "co_participant_speaker": WEIGHT_CO_PARTICIPANT_SPEAKER,
    "name_syntax": WEIGHT_NAME_SYNTAX,
    "asr_alternative_prompted": WEIGHT_ALT_EXACT_PROMPTED,
    "known_name_flag": WEIGHT_KNOWN_NAME_FLAG,
    "asr_alternatives_consistent": WEIGHT_ALT_CONSISTENT,
}

_TRUSTED_SOURCE_MARKERS = ("manual", "human", "voice_calibrated")
_PLACEHOLDER_NAMES = frozenset({"speaker", "unknown", "main", "secondary", "tertiary", "a", "b", "c", "x"})
_SECOND_PERSON = frozenset({"you", "your", "youre", "yours", "yourself", "ya", "u", "yall"})
_GREETINGS = frozenset({"hey", "hi", "hello", "yo", "oh", "thanks", "thank", "bye", "sorry", "please", "dear",
                        "okay", "ok", "welcome", "congrats", "morning", "night", "listen", "look", "wait"})
_PERSON_BEFORE = frozenset({"to", "with", "for", "and", "at", "from", "about", "than", "tell", "told", "ask",
                            "asked", "call", "called", "named", "meet", "met", "love", "loves", "hate", "miss",
                            "thank", "thanks", "hi", "hey", "yo", "dear", "is", "was", "its", "thats", "whos",
                            "like", "behind", "beside", "near", "invite", "invited", "date", "kiss", "hug",
                            "introduce", "introducing", "on", "by", "sees", "saw", "see", "watch"})
_PERSON_AFTER = frozenset({"is", "was", "said", "says", "say", "told", "asked", "wants", "wanted", "will",
                           "would", "can", "could", "did", "does", "just", "and", "thinks", "knows", "likes",
                           "loves", "here", "there", "has", "had", "went", "came", "gonna", "isnt", "wasnt",
                           "doesnt", "didnt", "cant", "wont", "never", "always", "really", "right"})
_SELF_INTRO = (("i", "am"), ("im",), ("my", "name", "is"), ("name", "is"), ("call", "me"))
_THIRD_INTRO = (("this", "is"), ("thats",), ("that", "is"), ("its",), ("it", "is"), ("meet",), ("say", "hi", "to"))
_SENTENCE_END = (".", "!", "?")
_VOCATIVE_PUNCT = (",", ":", ".", "!", "?", ";")

# High-frequency ENGLISH words (general spoken/written frequency, not derived
# from any clip). A token that is an ordinary word is lexically plausible as
# itself; context alone can never turn it into a name. Only independent
# acoustic evidence (an unprompted ear emitting the verified spelling) or an
# alias resolved elsewhere in the clip can.
COMMON_WORDS = frozenset("""
a about above across act actually add after again against age ago agree ah ahead air all allow almost alone
along already alright also always am amazing among amount an and angry animal another answer any anybody
anymore anyone anything anyway apart appear are area arm around art as ask asked asking at attack away awesome
baby back bad bag ball bank bar base be bear beat beautiful became because become bed been before began begin
behind being believe below best bet better between big bill bit black blood blow blue board boat body book
born both bottom bought box boy bra brain break bring bro broke brother brought brown build building built
burn business busy but buy by call called calm came can cannot car card care carry case cat catch caught cause
center certain chair chance change charge chat check child children choose city class clean clear close
cold color come comes coming common cook cool copy corner cost could count country couple course cover crazy
cross cry cup cut dad damn dance dark date daughter day dead deal dear death decide deep did die different
dinner do does dog doing done door double down draw dream dress drink drive drop dry dude during each ear
early earth easy eat eight either else end enough even evening ever every everybody everyone everything
exactly example eye face fact fair fall family far farm fast father fear feel feeling feet fell felt few
field fight figure fill final finally find fine finger finish fire first fish five floor fly follow food
foot for force forget forgot form forward found four free friend friends from front full fun funny game
gave get gets getting girl give given go god goes going gold gone good got great green ground group grow
guess guy guys had hair half hand happen happened happy hard has hate have having he head hear heard heart
heavy held hell hello help her here hey hi high hill him himself his hit hold hole home hope horse hot hour
house how huge human hundred hurt i ice idea if important in inside instead into is it its itself job join
joke jump just keep kept key kid kids kill kind king knew know known lady land language large last late
later laugh lay lead learn least leave left leg less let letter life light like line list listen little
live living long look looked looking lose lost lot love low lucky made main make man many mark matter may
maybe me mean meet men met middle might mile mind mine minute miss mom money month moon more morning most
mother move movie much music must my myself nah name near need never new next nice night nine no nobody
none nope nor normal north not note nothing now number of off office often oh ok okay old on once one only
open or order other our out outside over own page paid pain paper part party pass past pay people person
pick picture piece place plan play player please point police poor power pretty probably problem pull
push put question quick quiet quite rain ran rather reach read ready real really reason red remember rest
rich ride right ring rise road rock room round rule run said same sat save saw say says school sea second
see seem seen sell send sense serious set seven several shall she ship shoe shop short shot should shout
show shut sick side sign simple since sing sister sit six size sleep slow small smile snow so some
somebody someone something sometimes son song soon sorry sound south space speak special spend stand star
start state stay step still stop story street strong stuff such summer sun sure surprise sweet table take
taken talk talking tall tell ten than thank thanks that the their them then there these they thing things
think third this those though thought three through throw time tired to today together told tomorrow
tonight too took top total touch town tree tried true trust truth try trying turn twelve twenty two under
understand until up upon us use used very wait walk wall want wanted war warm was wash watch water way we
wear week weird well went were west what whatever when where whether which while white who whole why wide
wife will win wind window wish with without woman women won wonder word work world worry would write wrong
yeah year yellow yes yet yo you young your yourself
""".split())


def letters(value: str) -> str:
    """Letters-only casefold form (Turkish and other alphabets kept)."""
    return "".join(ch for ch in str(value).casefold() if ch.isalpha())


def phonetic_key(value: str) -> str:
    """Compact deterministic name key (homophone-tolerant, English-leaning).

    A final schwa/rhotic ending (-a, -er, -ar, -or, -ah, -uh, -re) collapses
    to one symbol so a non-rhotic ``-er`` and an ``-a`` spelling of the same
    name share a key; interior vowels are dropped, doubled consonants merge.
    """
    w = letters(value)
    if not w:
        return ""
    for ending in ("er", "ar", "or", "re", "ah", "uh", "a"):
        if len(w) > len(ending) + 1 and w.endswith(ending):
            w = w[: -len(ending)] + "@"
            break
    if w.startswith("kn") or w.startswith("gn"):
        w = w[1:]
    if w.startswith("wr"):
        w = w[1:]
    for old, new in (("tch", "ch"), ("dg", "j"), ("ph", "f"), ("ck", "k"), ("qu", "kw"), ("q", "k"), ("x", "ks"),
                     ("z", "s"), ("wh", "w"), ("gh", "g")):
        w = w.replace(old, new)
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
    w = w.replace("ch", "X").replace("sh", "X").replace("th", "0")
    head, tail = w[0], w[1:]
    tail = "".join(ch for ch in tail if ch not in "aeiouyhw")
    key = head + tail
    collapsed = [key[0]]
    for ch in key[1:]:
        if ch != collapsed[-1]:
            collapsed.append(ch)
    return "".join(collapsed)


def confusion_score(asr_value: str, canonical_value: str) -> float:
    """0 = unrelated; >= MIN_CONFUSION = plausible ASR confusion; 1 = same spelling.

    The compact phonetic key drops interior vowels, so on its own it would
    also pair names whose leading syllables sound different (another vowel
    after the same first consonant). A phonetic match therefore also needs the
    same onset (first two letters, the same gate as the caption stage's
    known-name detector) or a high spelling similarity.
    """
    a, b = letters(asr_value), letters(canonical_value)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a[0] != b[0] or abs(len(a) - len(b)) > 3 or min(len(a), len(b)) < 3:
        return 0.0
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    same_onset = a[:2] == b[:2]
    key_a, key_b = phonetic_key(a), phonetic_key(b)
    if key_a == key_b and len(key_b) >= 2 and (same_onset or ratio >= PHONETIC_MIN_RATIO_OTHER_ONSET):
        return round(max(0.85, ratio), 4)
    return round(ratio, 4) if ratio >= MIN_LEXICAL_CONFUSION and same_onset else 0.0


# ============================================================
# ROSTER (verified entities only)
# ============================================================

@dataclass(frozen=True)
class Participant:
    """One token of a verified name (a two-word name yields two tokens)."""

    raw_speaker: str          # diarization id of the person, "" when the entity is not a mapped speaker
    name: str                 # canonical spelling of this name token
    key: str                  # letters-only casefold
    evidence: str             # where the verification comes from
    speaks_in_clip: bool
    word_count: int
    kind: str = "participant"         # participant | creator | entity
    identity: str = ""                # full verified name (tokens of one identity never compete)

    def to_dict(self) -> dict[str, Any]:
        return {"raw_speaker": self.raw_speaker, "name": self.name, "key": self.key, "evidence": self.evidence,
                "speaks_in_clip": self.speaks_in_clip, "word_count": self.word_count,
                "phonetic_key": phonetic_key(self.key), "kind": self.kind, "identity": self.identity}


# Kept for readers of the V1 audit / callers of the V1 name.
Entity = Participant


@dataclass(frozen=True)
class ExtraEntity:
    """A verified name that is not a human-confirmed speaker label (e.g. ``--creator``)."""

    name: str
    kind: str = "creator"
    evidence: str = "user_verified"
    raw_speaker: str = ""


def _display_spelling(raw_name: str) -> str:
    """Human-entered names are often upper case: captions use Title case,
    but a deliberately mixed-case entry is kept as typed."""
    name = str(raw_name).strip()
    if name.isupper() or name.islower():
        return name[:1].upper() + name[1:].lower()
    return name


def _name_tokens(name: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", str(name))


def confirmed_participants(profile: Mapping[str, Any],
                           extra_entities: Sequence[ExtraEntity] = ()) -> tuple[Participant, ...]:
    """Verified roster: human-confirmed speaker names + explicitly verified extra entities.

    Same trust rule as the caption renderer (captions V24): the validated
    identity map written after the human voice checkpoint, or display labels
    whose naming source is manual / human / voice-calibrated. Automatic A/B
    placeholders are never entities. Extra entities come only from explicit
    user input (never from ASR-derived story text).
    """
    confirmed: dict[str, tuple[str, str]] = {}
    validated = profile.get("validated_identity_map")
    if isinstance(validated, Mapping):
        for raw, name in validated.items():
            if str(raw).strip() and str(name).strip():
                confirmed[str(raw).strip()] = (str(name).strip(), "validated_identity_map")
    names = profile.get("speaker_names") if isinstance(profile.get("speaker_names"), Mapping) else {}
    source = str(names.get("source", "")).casefold()
    labels = profile.get("display_labels")
    if isinstance(labels, Mapping) and any(marker in source for marker in _TRUSTED_SOURCE_MARKERS):
        for raw, name in labels.items():
            if str(raw).strip() and str(name).strip():
                confirmed.setdefault(str(raw).strip(), (str(name).strip(), "human_voice_checkpoint"))
    counts: dict[str, int] = {}
    for word in profile.get("words", []) or []:
        if isinstance(word, Mapping):
            raw = str(word.get("speaker_raw") or "").strip()
            if raw:
                counts[raw] = counts.get(raw, 0) + 1
    roster: list[Participant] = []
    seen: set[tuple[str, str]] = set()
    for raw, (name, evidence) in sorted(confirmed.items()):
        for token in _name_tokens(name):
            key = letters(token)
            if len(key) < 3 or key in _PLACEHOLDER_NAMES or (raw, key) in seen:
                continue
            seen.add((raw, key))
            roster.append(Participant(raw, _display_spelling(token), key, evidence, counts.get(raw, 0) > 0,
                                      counts.get(raw, 0), "participant", " ".join(name.split()).casefold()))
    participant_identities = {p.identity for p in roster}
    participant_keys = {p.key for p in roster}
    for extra in extra_entities:
        full = " ".join(str(extra.name).split())
        if not full or full.casefold() in participant_identities:
            continue
        raw = str(extra.raw_speaker or "").strip()
        for token in _name_tokens(full):
            key = letters(token)
            if len(key) < 3 or key in _PLACEHOLDER_NAMES or key in participant_keys or ("", key) in seen:
                continue
            seen.add(("", key))
            roster.append(Participant(raw, _display_spelling(token), key, extra.evidence,
                                      counts.get(raw, 0) > 0 if raw else False, counts.get(raw, 0) if raw else 0,
                                      extra.kind, full.casefold()))
    return tuple(roster)


# ============================================================
# ASR ALTERNATIVES (independent evidence per word position)
# ============================================================

@dataclass(frozen=True)
class AltObservation:
    """What one independent ASR ear heard at a word position."""

    token: str
    source: str
    prompted: bool            # the ear was told the verified names (prompt-induced risk)
    probability: float | None = None   # token probability when the backend reports logprobs


def _micro_source_prompted(source: str) -> bool:
    """Caption micro evidence order (speaker_caption_support): asr_1 raw acoustic and
    asr_2 enhanced acoustic are unprompted; later ears carry name context."""
    return str(source) not in ("asr_1", "asr_2")


def alternatives_from_caption_quality(profile: Mapping[str, Any],
                                      words: Sequence[Any] | None = None) -> dict[int, list[AltObservation]]:
    """Independent ASR candidates recorded by the caption stage, mapped to word ids.

    Only single-slot spans are used (every candidate phrase is one token), so
    a phrase can never be mis-aligned onto the wrong word. Mapping is by audio
    time: the word must lie inside the micro window, spell the span's phrase,
    and be the closest such word to the window centre. ``words`` defaults to
    the profile words (pass the ASR-restored words when a lock was applied).
    """
    quality = profile.get("caption_quality") if isinstance(profile.get("caption_quality"), Mapping) else {}
    micro = quality.get("micro_accuracy") if isinstance(quality.get("micro_accuracy"), Mapping) else {}
    rows_words = list(words if words is not None else (profile.get("words", []) or []))
    result: dict[int, list[AltObservation]] = {}
    for detail in micro.get("details", []) or []:
        if not isinstance(detail, Mapping):
            continue
        span = detail.get("span") or []
        window = detail.get("audio_window") or []
        if len(span) != 2 or len(window) != 2:
            continue
        try:
            if int(span[1]) - int(span[0]) != 1:
                continue
            w0, w1 = float(window[0]), float(window[1])
        except (TypeError, ValueError):
            continue
        votes = [row for row in detail.get("candidate_votes", []) or [] if isinstance(row, Mapping)]
        phrases = [(str(row.get("phrase", "")).split(), [str(s) for s in row.get("sources", []) or []])
                   for row in votes]
        if not phrases or any(len(tokens) > 1 for tokens, _ in phrases):
            continue
        target = str(detail.get("selected_phrase") or "").strip()
        if not target:
            ranked = sorted(votes, key=lambda row: -int(row.get("votes", 0) or 0))
            target = str(ranked[0].get("phrase", "")).strip() if ranked else ""
        if not letters(target):
            continue
        centre = (w0 + w1) / 2.0
        best: tuple[float, int] | None = None
        for index, word in enumerate(rows_words):
            if not isinstance(word, Mapping) or letters(str(word.get("word", ""))) != letters(target):
                continue
            try:
                start, end = float(word.get("edited_start", -1)), float(word.get("edited_end", -1))
            except (TypeError, ValueError):
                continue
            if w0 - 1e-3 <= start and end <= w1 + 1e-3:
                distance = abs((start + end) / 2.0 - centre)
                if best is None or distance < best[0]:
                    best = (distance, index)
        if best is None:
            continue
        bucket = result.setdefault(best[1], [])
        for tokens, sources in phrases:
            for source in sources:
                bucket.append(AltObservation(tokens[0] if tokens else "", source, _micro_source_prompted(source)))
    return result


def merge_alternatives(*maps: Mapping[int, Sequence[AltObservation]]) -> dict[int, list[AltObservation]]:
    merged: dict[int, list[AltObservation]] = {}
    for mapping in maps:
        for index, rows in (mapping or {}).items():
            merged.setdefault(int(index), []).extend(rows)
    return merged


# ============================================================
# CONTEXT EVIDENCE
# ============================================================

_TOKEN = re.compile(r"^(?P<pre>[^\w]*)(?P<core>[^\W\d_]+)(?P<poss>(?:['’]s|['’])?)(?P<post>[^\w]*)$")


def _canon(token: str) -> str:
    return letters(token.replace("’", "'").replace("'", ""))


def _clause_initial(texts: Sequence[str], index: int) -> bool:
    return index == 0 or texts[index - 1].rstrip().endswith(_SENTENCE_END + (",", ":", ";"))


def _near(texts: Sequence[str], index: int, vocabulary: frozenset[str], *, before: int = CONTEXT_WINDOW,
          after: int = CONTEXT_WINDOW) -> bool:
    window = [_canon(t) for t in texts[max(0, index - before):index]] + \
             [_canon(t) for t in texts[index + 1:index + 1 + after]]
    return any(w in vocabulary for w in window)


def _preceded_by(texts: Sequence[str], index: int, phrases: Iterable[tuple[str, ...]]) -> bool:
    for phrase in phrases:
        n = len(phrase)
        if index >= n and tuple(_canon(t) for t in texts[index - n:index]) == phrase:
            return True
    return False


@dataclass
class Candidate:
    index: int
    token: str
    core: str
    participant: Participant
    confusion: float
    evidence: list[str] = field(default_factory=list)
    score: float = 0.0
    veto: str = ""

    @property
    def reference_signal(self) -> bool:
        return any(e in REFERENCE_EVIDENCE for e in self.evidence)

    # V1 name of the same property (audit readers).
    context_signal = reference_signal

    @property
    def strong(self) -> bool:
        return any(e in STRONG_EVIDENCE for e in self.evidence)


def _flagged_tokens(profile: Mapping[str, Any]) -> set[str]:
    """ASR tokens MIMIR's own known-name detector flagged as possible spelling mismatches."""
    quality = profile.get("caption_quality") if isinstance(profile.get("caption_quality"), Mapping) else {}
    found: set[str] = set()
    for row in quality.get("known_name_flagged_spans", []) or []:
        reason = str(row.get("reason", "")) if isinstance(row, Mapping) else ""
        match = re.search(r"mismatch:\s*(.+?)\s+vs\s+(\S+)", reason)
        if match:
            found.add(_canon(match.group(1)) + "|" + letters(match.group(2)))
    return found


def _alternative_evidence(candidate: Candidate, observations: Sequence[AltObservation]) -> tuple[list[str], str]:
    """(evidence, veto) from independent ears at this word position.

    Lexical confidence is combined with agreement: an unprompted ear that
    emitted the verified spelling with a LOW token probability is only weak
    (prompt-level) evidence, and an unprompted ear that confidently heard a
    different word vetoes the correction even alone.
    """
    if not observations:
        return [], ""
    key = candidate.participant.key
    evidence: list[str] = []
    independent = [o for o in observations if not o.prompted]
    exact_independent = [o for o in independent if letters(o.token) == key]
    if any(o.probability is None or o.probability >= LOW_ALT_PROBABILITY for o in exact_independent):
        evidence.append("asr_alternative_exact")
    elif exact_independent or any(letters(o.token) == key for o in observations if o.prompted):
        evidence.append("asr_alternative_prompted")
    heard = [(letters(o.token), o.probability) for o in independent if letters(o.token)]
    contradicting = [(h, p) for h, p in heard if h != key and confusion_score(h, key) < MIN_CONFUSION
                     and confusion_score(h, candidate.core) < MIN_CONFUSION]
    if heard and not contradicting and "asr_alternative_exact" not in evidence:
        evidence.append("asr_alternatives_consistent")
    confident_other = any(p is not None and p >= HIGH_ALT_PROBABILITY for _, p in contradicting)
    veto = ("independent_ear_heard_a_different_word"
            if contradicting and (len(contradicting) >= len(heard) / 2 or confident_other) else "")
    return evidence, veto


def _other_full_name(texts: Sequence[str], index: int, participant: Participant) -> bool:
    """The token is part of ANOTHER capitalized full name (followed by a
    capitalized surname, or preceded by a capitalized first name, neither of
    which belongs to the verified identity): a different person, never the
    verified entity."""
    identity_keys = {letters(t) for t in _name_tokens(participant.identity)} if participant.identity else set()

    def name_like(position: int) -> bool:
        if not 0 <= position < len(texts):
            return False
        match = _TOKEN.match(texts[position])
        if not match:
            return False
        core = match.group("core")
        key = letters(core)
        return (core[:1].isupper() and len(key) >= 2 and key not in COMMON_WORDS and key not in identity_keys
                and key not in _PLACEHOLDER_NAMES)

    token = texts[index].rstrip()
    following = not token.endswith(_SENTENCE_END + (",", ":", ";")) and name_like(index + 1)
    previous = (index > 0 and not texts[index - 1].rstrip().endswith(_SENTENCE_END + (",", ":", ";"))
                and not _clause_initial(texts, index - 1) and name_like(index - 1))
    return following or previous


def _evaluate(words: Sequence[Mapping[str, Any]], index: int, candidate: Candidate, roster: Sequence[Participant],
              flagged: set[str], confirmed_aliases: set[tuple[str, str]],
              observations: Sequence[AltObservation] = ()) -> Candidate:
    texts = [str(w.get("word", "")) for w in words]
    token = candidate.token
    speaker = str(words[index].get("speaker_raw") or "").strip()
    participant = candidate.participant
    evidence = candidate.evidence
    evidence.clear()
    stripped = token.rstrip()
    capitalized = candidate.core[:1].isupper()
    vocative_punct = stripped.endswith(_VOCATIVE_PUNCT)
    greeting = _preceded_by(texts, index, ((g,) for g in _GREETINGS))
    second_after = _near(texts, index, _SECOND_PERSON, before=0)
    second_near = _near(texts, index, _SECOND_PERSON)
    if (vocative_punct and (second_after or _clause_initial(texts, index) or greeting)) or (greeting and second_near):
        evidence.append("direct_address")
    self_intro = _preceded_by(texts, index, _SELF_INTRO)
    speaker_ids = {p.raw_speaker for p in roster if p.raw_speaker}
    if self_intro and participant.raw_speaker and speaker == participant.raw_speaker:
        evidence.append("self_introduction")
    if self_intro and speaker and speaker in speaker_ids and speaker != participant.raw_speaker:
        candidate.veto = "another_confirmed_participant_introduces_themself_with_this_name"
    if participant.raw_speaker and speaker and speaker in speaker_ids and speaker != participant.raw_speaker:
        evidence.append("co_participant_speaker")
    before = _canon(texts[index - 1]) if index > 0 else ""
    after = _canon(texts[index + 1]) if index + 1 < len(texts) else ""
    if (before in _PERSON_BEFORE or after in _PERSON_AFTER or candidate.token.find("'") >= 0
            or candidate.token.find("’") >= 0 or _preceded_by(texts, index, _THIRD_INTRO)):
        evidence.append("name_syntax")
    if (candidate.core.casefold(), participant.key) in confirmed_aliases:
        evidence.append("confirmed_alias")
    if f"{letters(candidate.core)}|{participant.key}" in flagged:
        evidence.append("known_name_flag")
    alt_evidence, alt_veto = _alternative_evidence(candidate, observations)
    evidence.extend(e for e in alt_evidence if e not in evidence)
    if alt_veto and "direct_address" not in evidence:
        candidate.veto = candidate.veto or alt_veto
    if _other_full_name(texts, index, participant) and "asr_alternative_exact" not in evidence:
        candidate.veto = candidate.veto or "part_of_another_full_name"
    if not capitalized and "direct_address" not in evidence:
        candidate.veto = candidate.veto or "not_a_proper_noun"
    if letters(candidate.core) in COMMON_WORDS and not ({"asr_alternative_exact", "confirmed_alias"} & set(evidence)):
        # An ordinary word is plausible as itself (also at a sentence start,
        # where capitalization says nothing): only independent acoustic
        # evidence or an alias resolved elsewhere may turn it into a name.
        candidate.veto = candidate.veto or "ordinary_word_without_acoustic_evidence"
    candidate.score = round(sum(EVIDENCE_WEIGHTS.get(e, 0.0) for e in evidence), 4)
    return candidate


def _decided(candidate: Candidate) -> bool:
    """Deterministic gate: no veto, a reference signal, enough weight, and either
    strong evidence or support for a third-person mention."""
    if candidate.veto or not candidate.reference_signal or candidate.score < CORRECTION_THRESHOLD:
        return False
    return candidate.strong or bool(SUPPORT_EVIDENCE & set(candidate.evidence))


def _replacement(token: str, canonical: str) -> str | None:
    match = _TOKEN.match(token)
    if not match:
        return None
    core = match.group("core")
    if len(core) > 1 and core.isupper():
        spelled = canonical.upper()
    elif core[:1].isupper():
        spelled = canonical
    else:
        spelled = canonical.lower()
    return f"{match.group('pre')}{spelled}{match.group('poss')}{match.group('post')}"


# ============================================================
# LOCK
# ============================================================

@dataclass
class NameLockResult:
    words: list[dict[str, Any]]
    audit: dict[str, Any]
    changed: bool


# Truth-side flags written only by the Caption Truth V6 stage (ai/editor/caption_truth.py).
TRUTH_V6_MARKER = "caption_truth_v6"
TRUTH_V6_UNCERTAIN = "caption_uncertain"


def clear_truth_v6_flags(words: Sequence[Any]) -> list[Any]:
    """Copies without the flags a previous Caption Truth V6 run added (text/time/speaker untouched)."""
    cleared: list[Any] = []
    for word in words:
        if isinstance(word, Mapping) and TRUTH_V6_MARKER in word:
            copy = dict(word)
            copy.pop(TRUTH_V6_MARKER, None)
            copy.pop(TRUTH_V6_UNCERTAIN, None)
            cleared.append(copy)
        else:
            cleared.append(dict(word) if isinstance(word, Mapping) else word)
    return cleared


def restore_original_spelling(words: Sequence[Any]) -> list[Any]:
    """Copies of the words with every previous lock undone (text only)."""
    restored: list[Any] = []
    for word in words:
        if isinstance(word, Mapping):
            copy = dict(word)
            provenance = copy.pop("name_lock", None)
            if isinstance(provenance, Mapping) and str(provenance.get("from", "")).strip():
                copy["word"] = str(provenance["from"])
            restored.append(copy)
        else:
            restored.append(word)
    return restored


def lock_participant_names(
    profile: Mapping[str, Any],
    *,
    extra_entities: Sequence[ExtraEntity] = (),
    alternatives: Mapping[int, Sequence[AltObservation]] | None = None,
) -> NameLockResult:
    """Return corrected copies of the profile words + audit (input not mutated).

    ``changed`` is True when the final words differ from the profile's words.
    """
    original_words = [dict(w) if isinstance(w, Mapping) else w for w in (profile.get("words", []) or [])]
    words = restore_original_spelling(original_words)
    roster = confirmed_participants(profile, extra_entities)
    observations = merge_alternatives(alternatives_from_caption_quality(profile, words), alternatives or {})
    base_audit: dict[str, Any] = {
        "version": NAME_LOCK_VERSION, "roster": [p.to_dict() for p in roster],
        "policy": ("verified entity + close phonetic/lexical confusion + reference evidence + no competing entity "
                   "+ not an ordinary word without acoustic evidence -> spelling only; word id/time/speaker/order "
                   "unchanged; ambiguity fails closed"),
    }
    if str(profile.get("status", "")) != "ok" or not roster:
        audit = base_audit | {"status": "not_applicable" if not roster else "no_final_words",
                              "reason": "no verified entity names" if not roster else "profile not ok",
                              "corrections": [], "rejected": [], "new_corrections": 0}
        return NameLockResult(words, audit, words != original_words)

    flagged = _flagged_tokens(profile)
    keys = {p.key for p in roster}
    candidates: list[Candidate] = []
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            continue
        token = str(word.get("word", ""))
        match = _TOKEN.match(token)
        if not match:
            continue
        core = match.group("core")
        core_key = letters(core)
        if core_key in keys or len(core_key) < 3:
            continue            # already a verified spelling (or a competing verified name), or too short
        scored = sorted(((confusion_score(core_key, p.key), p) for p in roster), key=lambda row: (-row[0], row[1].key))
        scored = [(s, p) for s, p in scored if s >= MIN_CONFUSION]
        if not scored:
            continue
        best_score, best = scored[0]
        candidate = Candidate(index, token, core, best, best_score)
        rivals = [s for s, p in scored[1:] if p.identity != best.identity and p.key != best.key]
        if rivals and best_score - rivals[0] < AMBIGUITY_MARGIN:
            candidate.veto = "ambiguous_between_entities"
        candidates.append(candidate)

    corrections: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    confirmed_aliases: set[tuple[str, str]] = set()
    pending = candidates
    for pass_index in (1, 2):   # pass 2 may use aliases confirmed by strong evidence in pass 1
        remaining: list[Candidate] = []
        for candidate in pending:
            candidate.veto = candidate.veto if candidate.veto == "ambiguous_between_entities" else ""
            _evaluate(words, candidate.index, candidate, roster, flagged, confirmed_aliases,
                      observations.get(candidate.index, ()))
            decided = _decided(candidate)
            if not decided:
                remaining.append(candidate)
                continue
            replacement = _replacement(candidate.token, candidate.participant.name)
            if replacement is None or replacement == candidate.token:
                remaining.append(candidate)
                continue
            word = words[candidate.index]
            provenance = {"from": candidate.token, "to": replacement, "participant": candidate.participant.raw_speaker,
                          "canonical": candidate.participant.name, "entity_kind": candidate.participant.kind,
                          "confusion": candidate.confusion, "score": candidate.score,
                          "evidence": list(candidate.evidence), "pass": pass_index, "version": NAME_LOCK_VERSION}
            word["word"] = replacement
            word["name_lock"] = provenance
            corrections.append(provenance | {"index": candidate.index})
            own_strong = candidate.strong and "confirmed_alias" not in candidate.evidence
            if own_strong or {"co_participant_speaker", "name_syntax"} <= set(candidate.evidence):
                # Resolved on its own evidence: the same ASR variant elsewhere in
                # this clip may use it (never a chain of aliases).
                confirmed_aliases.add((candidate.core.casefold(), candidate.participant.key))
        pending = remaining
        if not confirmed_aliases:
            break
    for candidate in pending:
        reason = candidate.veto or ("no_reference_evidence" if not candidate.reference_signal
                                    else f"score_below_threshold:{candidate.score}"
                                    if candidate.score < CORRECTION_THRESHOLD else "insufficient_support")
        rejected.append({"index": candidate.index, "token": candidate.token, "near": candidate.participant.name,
                         "entity_kind": candidate.participant.kind, "confusion": candidate.confusion,
                         "evidence": list(candidate.evidence), "score": candidate.score, "reason": reason})
    audit = base_audit | {"status": "corrected" if corrections else "clean", "corrections": corrections,
                          "new_corrections": len(corrections), "rejected": rejected}
    return NameLockResult(words, audit, words != original_words)


def verify_lock_invariants(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> None:
    """Only word text may change: id (position), time and speaker fields are identical."""
    if len(before) != len(after):
        raise ValueError("name lock changed the word count")
    for index, (a, b) in enumerate(zip(before, after)):
        for key in set(a) | set(b):
            if key in ("word", "name_lock"):
                continue
            if a.get(key) != b.get(key):
                raise ValueError(f"name lock changed {key} of word {index}")


def apply_to_profile_file(
    profile_path: str | Path,
    *,
    extra_entities: Sequence[ExtraEntity] = (),
    alternatives: Mapping[int, Sequence[AltObservation]] | None = None,
) -> dict[str, Any]:
    """Apply the lock to a final speaker profile JSON in place (atomic, idempotent).

    Returns the audit (with ``changed``). The file is rewritten only when the
    words or the audit actually differ, so an unchanged re-run keeps the
    profile fingerprint (downstream stage caches stay valid). Flags left by a
    previous Caption Truth V6 run are removed (this is the V6-off path).
    """
    path = Path(profile_path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("speaker profile root is not an object")
    stored_words = list(profile.get("words", []) or [])
    had_truth_v6 = any(isinstance(w, Mapping) and TRUTH_V6_MARKER in w for w in stored_words) \
        or "caption_truth" in profile
    profile["words"] = clear_truth_v6_flags(stored_words)
    profile.pop("caption_truth", None)
    before = [dict(w) for w in profile["words"] if isinstance(w, dict)]
    result = lock_participant_names(profile, extra_entities=extra_entities, alternatives=alternatives)
    after = [w for w in result.words if isinstance(w, dict)]
    verify_lock_invariants(before, after)
    changed = result.changed or had_truth_v6
    audit = dict(result.audit) | {"changed": changed}
    stored = {k: v for k, v in audit.items() if k != "changed"}
    if changed or profile.get("participant_name_lock") != stored:
        profile["words"] = result.words
        profile["participant_name_lock"] = stored
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    return audit
