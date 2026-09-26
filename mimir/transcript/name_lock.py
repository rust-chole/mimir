"""Evidence-gated canonical spelling of VERIFIED names (lexical truth only).

ASR cannot separate homophone spellings of a name (non-rhotic "-er" vs "-a",
doubled consonants), so a confirmed participant may be written as a common
variant even when every ear agrees. A token becomes the verified spelling only
when ALL of the following hold:

* the name is verified (user-confirmed speaker name, declared entity/creator);
* the token is a close phonetic/lexical confusion of it, with no equally close
  competing verified name;
* there is evidence the token REFERS to that entity (direct address,
  self-introduction, name syntax, an alias already resolved in the clip, an
  independent ear that emitted the verified spelling, ...);
* no veto: an ordinary English word needs acoustic evidence, a token inside
  another capitalized full name is another person, a confident independent ear
  that heard a different word blocks the change.

Only ``text`` changes; id, times and speaker are verified unchanged.
Uncertain cases fail closed (ASR spelling kept, reason recorded).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping, Sequence

MIN_CONFUSION = 0.80
PHONETIC_MIN_RATIO_OTHER_ONSET = 0.75
AMBIGUITY_MARGIN = 0.10
CONTEXT_WINDOW = 6
CORRECTION_THRESHOLD = 0.50
LOW_ALT_PROBABILITY = 0.5
HIGH_ALT_PROBABILITY = 0.8

WEIGHTS = {
    "direct_address": 0.60,
    "self_introduction": 0.60,
    "asr_alternative_exact": 0.50,
    "confirmed_alias": 0.35,
    "co_participant_speaker": 0.30,
    "name_syntax": 0.25,
    "asr_alternative_prompted": 0.20,
    "known_name_flag": 0.15,
    "asr_alternatives_consistent": 0.10,
}
STRONG = frozenset({"direct_address", "self_introduction", "confirmed_alias", "asr_alternative_exact"})
SUPPORT = frozenset({"known_name_flag", "asr_alternative_prompted"})
REFERENCE = frozenset({"direct_address", "self_introduction", "name_syntax", "confirmed_alias",
                       "asr_alternative_exact"})

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
_CLAUSE_END = _SENTENCE_END + (",", ":", ";")
_VOCATIVE_PUNCT = (",", ":", ".", "!", "?", ";")
_TOKEN = re.compile(r"^(?P<pre>[^\w]*)(?P<core>[^\W\d_]+?)(?P<poss>['’]s)?(?P<post>[^\w]*)$")

# High-frequency English words (general frequency, not derived from any clip):
# an ordinary word is plausible as itself; context alone never turns it into a name.
COMMON_WORDS = frozenset("""
a about above across act actually add after again against age ago agree ah ahead air all allow almost alone along
already alright also always am amazing among an and angry animal another answer any anybody anymore anyone anything
anyway apart are area arm around art as ask asked at away awesome baby back bad bag ball bank bar base be bear beat
beautiful because become bed been before begin behind being believe below best bet better between big bit black
blood blow blue board boat body book born both bottom box boy brain break bring bro broke brother brought brown
build burn business busy but buy by call called calm came can car card care carry case cat catch caught cause
certain chair chance change chat check child city class clean clear close cold color come coming cook cool corner
cost could count country couple course cover crazy cross cry cut dad damn dance dark date day dead deal dear death
decide deep did die different dinner do does dog doing done door down draw dream dress drink drive drop dude during
each early easy eat either else end enough even ever every everybody everyone everything exactly eye face fact fair
fall family far fast father fear feel fell felt few fight fill final finally find fine fire first fish five floor fly
follow food foot for forget forgot four free friend from front full fun funny game gave get getting girl give go god
goes going gold gone good got great green ground group grow guess guy guys had hair half hand happen happened happy
hard has hate have he head hear heard heart heavy hell hello help her here hey hi high him his hit hold hole home
hope hot hour house how huge hurt i idea if in inside into is it its job join joke jump just keep kept key kid kids
kill kind king knew know lady land large last late later laugh lead learn least leave left leg less let life light
like line listen little live long look looking lose lost lot love low lucky made make man many mark matter may maybe
me mean meet men met middle might mind mine minute miss mom money month more morning most mother move much music
must my myself nah name near need never new next nice night nine no nobody none nope nor normal not nothing now
number of off oh ok okay old on once one only open or order other our out over own paid pain part party pass past
pay people person pick piece place plan play player please point police poor power pretty probably problem pull push
put question quick quiet quite ran rather reach read ready real really reason red remember rest rich ride right ring
road rock room round rule run said same sat save saw say says school second see seem seen sell send serious set
seven shall she shot should show shut sick side sign since sing sister sit six size sleep slow small smile so some
somebody someone something sometimes son song soon sorry sound speak stand star start stay step still stop story
street strong stuff such sun sure surprise sweet take talk talking tell ten than thank thanks that the their them
then there these they thing things think this those though thought three through throw time tired to today together
told tomorrow tonight too took top touch town tree tried true trust truth try turn twelve twenty two under understand
until up us use used very wait walk wall want wanted war warm was watch water way we wear week weird well went were
what whatever when where which while white who whole why wide wife will win wind window wish with without woman
women won wonder word work world worry would write wrong yeah year yellow yes yet yo you young your yourself
""".split())


def letters(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalpha())


def phonetic_key(value: str) -> str:
    """Homophone-tolerant key: final schwa/rhotic endings collapse, interior vowels drop, doubles merge."""
    w = letters(value)
    if not w:
        return ""
    for ending in ("er", "ar", "or", "re", "ah", "uh", "a"):
        if len(w) > len(ending) + 1 and w.endswith(ending):
            w = w[: -len(ending)] + "@"
            break
    for prefix in ("kn", "gn", "wr"):
        if w.startswith(prefix):
            w = w[1:]
    for old, new in (("tch", "ch"), ("dg", "j"), ("ph", "f"), ("ck", "k"), ("qu", "kw"), ("q", "k"), ("x", "ks"),
                     ("z", "s"), ("wh", "w"), ("gh", "g")):
        w = w.replace(old, new)
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
    w = w.replace("ch", "X").replace("sh", "X").replace("th", "0")
    key = w[0] + "".join(ch for ch in w[1:] if ch not in "aeiouyhw")
    collapsed = [key[0]]
    for ch in key[1:]:
        if ch != collapsed[-1]:
            collapsed.append(ch)
    return "".join(collapsed)


def confusion_score(asr_value: str, canonical_value: str) -> float:
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
    return round(ratio, 4) if ratio >= MIN_CONFUSION and same_onset else 0.0


@dataclass(frozen=True)
class RosterEntry:
    name: str             # canonical spelling of ONE name token
    key: str              # letters-only casefold
    identity: str         # full verified name
    speaker: str          # anonymous speaker id when the name belongs to a participant
    kind: str             # participant | creator | entity


def _display(name: str) -> str:
    name = " ".join(str(name).split())
    if name.isupper() or name.islower():
        return " ".join(part[:1].upper() + part[1:].lower() for part in name.split())
    return name


def build_roster(identities: Mapping[str, Mapping[str, Any]], entities: Sequence[str], creator: str
                 ) -> list[RosterEntry]:
    roster: list[RosterEntry] = []
    seen: set[str] = set()

    def add(full: str, speaker: str, kind: str) -> None:
        full = _display(full)
        for part in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", full):
            key = letters(part)
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)
            roster.append(RosterEntry(part, key, full, speaker, kind))

    for speaker, row in identities.items():
        if row.get("confirmed") and str(row.get("name", "")).strip():
            add(str(row["name"]), speaker, "participant")
    if creator.strip():
        add(creator, "", "creator")
    for entity in entities:
        if entity.strip():
            add(entity, "", "entity")
    return roster


@dataclass
class Candidate:
    index: int
    core: str
    entry: RosterEntry
    confusion: float
    evidence: list[str] = field(default_factory=list)
    score: float = 0.0
    veto: str = ""

    @property
    def decided(self) -> bool:
        if self.veto or not any(e in REFERENCE for e in self.evidence) or self.score < CORRECTION_THRESHOLD:
            return False
        return any(e in STRONG for e in self.evidence) or bool(SUPPORT & set(self.evidence))


def _canon(token: str) -> str:
    return letters(token.replace("’", "'").replace("'", ""))


def _preceded_by(texts: Sequence[str], index: int, phrases: Iterable[tuple[str, ...]]) -> bool:
    for phrase in phrases:
        n = len(phrase)
        if index >= n and tuple(_canon(t) for t in texts[index - n:index]) == phrase:
            return True
    return False


def _near(texts: Sequence[str], index: int, vocabulary: frozenset[str], before: int, after: int) -> bool:
    window = [_canon(t) for t in texts[max(0, index - before):index]] + \
             [_canon(t) for t in texts[index + 1:index + 1 + after]]
    return any(w in vocabulary for w in window)


def _clause_initial(texts: Sequence[str], index: int) -> bool:
    return index == 0 or texts[index - 1].rstrip().endswith(_CLAUSE_END)


def _other_full_name(texts: Sequence[str], index: int, entry: RosterEntry) -> bool:
    identity_keys = {letters(t) for t in entry.identity.split()}

    def name_like(position: int) -> bool:
        if not 0 <= position < len(texts):
            return False
        match = _TOKEN.match(texts[position])
        if not match:
            return False
        core = match.group("core")
        key = letters(core)
        return core[:1].isupper() and len(key) >= 2 and key not in COMMON_WORDS and key not in identity_keys

    following = not texts[index].rstrip().endswith(_CLAUSE_END) and name_like(index + 1)
    previous = (index > 0 and not texts[index - 1].rstrip().endswith(_CLAUSE_END)
                and not _clause_initial(texts, index - 1) and name_like(index - 1))
    return following or previous


def _alternative_evidence(candidate: Candidate, observations: Sequence[Mapping[str, Any]]) -> tuple[list[str], str]:
    if not observations:
        return [], ""
    key = candidate.entry.key
    evidence: list[str] = []
    independent = [o for o in observations if not o.get("prompted")]
    exact_independent = [o for o in independent if letters(o["token"]) == key]
    if any(o.get("probability") is None or o["probability"] >= LOW_ALT_PROBABILITY for o in exact_independent):
        evidence.append("asr_alternative_exact")
    elif exact_independent or any(letters(o["token"]) == key for o in observations if o.get("prompted")):
        evidence.append("asr_alternative_prompted")
    heard = [(letters(o["token"]), o.get("probability")) for o in independent if letters(o["token"])]
    contradicting = [(h, p) for h, p in heard
                     if h != key and confusion_score(h, key) < MIN_CONFUSION
                     and confusion_score(h, candidate.core) < MIN_CONFUSION]
    if heard and not contradicting and "asr_alternative_exact" not in evidence:
        evidence.append("asr_alternatives_consistent")
    confident_other = any(p is not None and p >= HIGH_ALT_PROBABILITY for _, p in contradicting)
    veto = ("independent_ear_heard_a_different_word"
            if contradicting and (len(contradicting) >= len(heard) / 2 or confident_other) else "")
    return evidence, veto


def _evaluate(words: Sequence[Mapping[str, Any]], candidate: Candidate, roster: Sequence[RosterEntry],
              flagged: set[str], aliases: set[tuple[str, str]]) -> Candidate:
    texts = [str(w["text"]) for w in words]
    index = candidate.index
    entry = candidate.entry
    speaker = str(words[index].get("speaker") or "")
    evidence = candidate.evidence
    evidence.clear()
    stripped = texts[index].rstrip()
    greeting = _preceded_by(texts, index, ((g,) for g in _GREETINGS))
    second_after = _near(texts, index, _SECOND_PERSON, 0, CONTEXT_WINDOW)
    second_near = _near(texts, index, _SECOND_PERSON, CONTEXT_WINDOW, CONTEXT_WINDOW)
    if (stripped.endswith(_VOCATIVE_PUNCT) and (second_after or _clause_initial(texts, index) or greeting)) \
            or (greeting and second_near):
        evidence.append("direct_address")
    self_intro = _preceded_by(texts, index, _SELF_INTRO)
    roster_speakers = {e.speaker for e in roster if e.speaker}
    if self_intro and entry.speaker and speaker == entry.speaker:
        evidence.append("self_introduction")
    if self_intro and speaker in roster_speakers and speaker != entry.speaker:
        candidate.veto = "another_verified_participant_introduces_themself"
    if entry.speaker and speaker and speaker in roster_speakers and speaker != entry.speaker:
        evidence.append("co_participant_speaker")
    before = _canon(texts[index - 1]) if index > 0 else ""
    after = _canon(texts[index + 1]) if index + 1 < len(texts) else ""
    if before in _PERSON_BEFORE or after in _PERSON_AFTER or "'" in texts[index] or "’" in texts[index] \
            or _preceded_by(texts, index, _THIRD_INTRO):
        evidence.append("name_syntax")
    if (candidate.core.casefold(), entry.key) in aliases:
        evidence.append("confirmed_alias")
    if f"{letters(candidate.core)}|{entry.key}" in flagged:
        evidence.append("known_name_flag")
    alt_evidence, alt_veto = _alternative_evidence(candidate, words[index].get("alternatives") or [])
    evidence.extend(e for e in alt_evidence if e not in evidence)
    if alt_veto and "direct_address" not in evidence:
        candidate.veto = candidate.veto or alt_veto
    if _other_full_name(texts, index, entry) and "asr_alternative_exact" not in evidence:
        candidate.veto = candidate.veto or "part_of_another_full_name"
    if not candidate.core[:1].isupper() and "direct_address" not in evidence:
        candidate.veto = candidate.veto or "not_a_proper_noun"
    if letters(candidate.core) in COMMON_WORDS and not ({"asr_alternative_exact", "confirmed_alias"} & set(evidence)):
        candidate.veto = candidate.veto or "ordinary_word_without_acoustic_evidence"
    candidate.score = round(sum(WEIGHTS.get(e, 0.0) for e in evidence), 4)
    return candidate


def _replacement(token: str, canonical: str) -> str | None:
    match = _TOKEN.match(token)
    if not match:
        return None
    core = match.group("core")
    spelled = canonical.upper() if len(core) > 1 and core.isupper() else (
        canonical if core[:1].isupper() else canonical.lower())
    return f"{match.group('pre')}{spelled}{match.group('poss') or ''}{match.group('post')}"


def lock_names(words: Sequence[Mapping[str, Any]], roster: Sequence[RosterEntry], flagged: set[str] = frozenset()
               ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (words with canonical spellings, audit). Pure and deterministic."""
    output = [dict(w) for w in words]
    audit: dict[str, Any] = {"roster": [e.__dict__ for e in roster], "corrections": [], "rejected": []}
    if not roster:
        return output, audit
    candidates: list[Candidate] = []
    for index, word in enumerate(output):
        text = str(word["text"])
        if len(text.split()) != 1:
            continue
        match = _TOKEN.match(text)
        if not match:
            continue
        core = match.group("core")
        key = letters(core)
        if any(e.key == key for e in roster):
            continue
        scored = sorted(((confusion_score(core, e.name), e) for e in roster), key=lambda r: -r[0])
        scored = [row for row in scored if row[0] >= MIN_CONFUSION]
        if not scored:
            continue
        best_score, best = scored[0]
        rivals = [e for s, e in scored[1:] if best_score - s <= AMBIGUITY_MARGIN and e.identity != best.identity]
        if rivals:
            audit["rejected"].append({"index": index, "token": text, "reason": "competing_verified_names",
                                      "names": [best.name] + [r.name for r in rivals]})
            continue
        candidates.append(Candidate(index, core, best, best_score))
    aliases: set[tuple[str, str]] = set()
    for _round in range(2):  # second round may use aliases resolved by strong evidence in the first
        for candidate in candidates:
            candidate.veto = ""
            _evaluate(output, candidate, roster, flagged, aliases)
        for candidate in candidates:
            if candidate.decided and any(e in STRONG - {"confirmed_alias"} for e in candidate.evidence):
                aliases.add((candidate.core.casefold(), candidate.entry.key))
    for candidate in candidates:
        word = output[candidate.index]
        row = {"index": candidate.index, "word_id": word.get("id"), "token": word["text"],
               "name": candidate.entry.name, "confusion": candidate.confusion, "evidence": list(candidate.evidence),
               "score": candidate.score, "veto": candidate.veto}
        replacement = _replacement(str(word["text"]), candidate.entry.name) if candidate.decided else None
        if replacement and replacement != word["text"]:
            word["original_text"] = word.get("original_text") or word["text"]
            word["text"] = replacement
            word["lexical_source"] = "name_lock"
            word["name_lock"] = {k: row[k] for k in ("name", "evidence", "score", "confusion")}
            audit["corrections"].append({**row, "to": replacement})
        else:
            audit["rejected"].append({**row, "reason": candidate.veto or "insufficient_evidence"})
    verify_invariants(words, output)
    return output, audit


def verify_invariants(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> None:
    if len(before) != len(after):
        raise AssertionError("name lock changed the word count")
    for a, b in zip(before, after):
        for key in ("id", "start", "end", "speaker"):
            if a.get(key) != b.get(key):
                raise AssertionError(f"name lock changed {key} of word {a.get('id')}")
