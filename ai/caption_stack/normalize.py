"""Harmless lexical differences, defined once.

Two hearings of the same speech may differ without either being a caption error:

    case, punctuation, apostrophe glyph, hyphenation      ``lexical.token_keys`` already
    a hesitation filler one hearing wrote, one did not    "uh", "um", "hmm"
    spelling variants of one word                         "ok" / "okay", "alright" / "all right"
    informal and standard contractions                    "gonna" / "going to", "don't" / "do not"
    number words vs digits, compounds included            "twenty-one" / "21", "a hundred" / "100"

``canonical_sequence`` maps comparison keys (the output of ``lexical.token_keys``) onto
one canonical form. The A/B dispute finder uses it so that a harmless difference never
opens a dispute (the primary words stay exactly as heard), and the caption evaluation
scores WER on it, so a system is neither rewarded nor punished for a difference that is
not an error. Ambiguous forms are deliberately left alone: "'s" (is / has / possessive),
"'d" (would / had), "ya" (you / yeah). Meaning-changing differences (a negation, another
number, another word) always stay different.

Display text is never changed by this module: it only decides what counts as "the same".
"""
from __future__ import annotations

from typing import Sequence

NORMALIZE_VERSION = 1

FILLERS = frozenset({"uh", "uhh", "uhhh", "um", "umm", "ummm", "uhm", "er", "erm", "hmm", "hmmm", "hm", "mm",
                     "mmm", "mhm"})

_VARIANTS: dict[str, tuple[str, ...]] = {
    # one word, several spellings
    "ok": ("okay",), "okey": ("okay",), "alright": ("all", "right"), "aight": ("all", "right"),
    "yea": ("yeah",), "yeh": ("yeah",), "cuz": ("because",), "cos": ("because",), "cause": ("because",),
    # informal contractions
    "gonna": ("going", "to"), "wanna": ("want", "to"), "gotta": ("got", "to"), "hafta": ("have", "to"),
    "kinda": ("kind", "of"), "sorta": ("sort", "of"), "lotta": ("lot", "of"), "outta": ("out", "of"),
    "lemme": ("let", "me"), "gimme": ("give", "me"), "tryna": ("trying", "to"), "dunno": ("do", "not", "know"),
    # unambiguous standard contractions
    "don't": ("do", "not"), "doesn't": ("does", "not"), "didn't": ("did", "not"), "isn't": ("is", "not"),
    "aren't": ("are", "not"), "wasn't": ("was", "not"), "weren't": ("were", "not"), "haven't": ("have", "not"),
    "hasn't": ("has", "not"), "hadn't": ("had", "not"), "won't": ("will", "not"), "wouldn't": ("would", "not"),
    "can't": ("can", "not"), "cannot": ("can", "not"), "couldn't": ("could", "not"),
    "shouldn't": ("should", "not"), "mustn't": ("must", "not"), "needn't": ("need", "not"),
    "i'm": ("i", "am"), "you're": ("you", "are"), "we're": ("we", "are"), "they're": ("they", "are"),
    "i've": ("i", "have"), "you've": ("you", "have"), "we've": ("we", "have"), "they've": ("they", "have"),
    "i'll": ("i", "will"), "you'll": ("you", "will"), "we'll": ("we", "will"), "they'll": ("they", "will"),
    "he'll": ("he", "will"), "she'll": ("she", "will"), "it'll": ("it", "will"), "let's": ("let", "us"),
    "y'all": ("you", "all"),
    # a unit the other hearing wrote as a symbol ("$1" -> dollars)
    "dollar": ("dollars",),
}

# ``vod_processor.canonical_word`` already maps zero..twenty to digits.
_TENS = {"thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_TENS_VALUES = frozenset({20, 30, 40, 50, 60, 70, 80, 90})
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}


def _small(key: str) -> int | None:
    if key.isdigit() and len(key) <= 2:
        return int(key)
    return _TENS.get(key)


def _parse_number(keys: Sequence[str], start: int) -> tuple[int | None, int]:
    """(value, keys used) of a spelled number phrase starting at ``start``; (None, 0) if none.

    ``[a|N] hundred [and] N``, ``N thousand ...``, ``tens units`` ("twenty one" -> 21).
    Only multi-key phrases are collapsed: a single key is already canonical, and two
    plain numbers in a row ("one two three") stay separate words.
    """
    index, total, group, last, compound = start, 0, 0, "", False
    while index < len(keys):
        key = keys[index]
        small = _small(key)
        if small is not None:
            if last == "small":
                break
            value, nxt = small, index + 1
            if small in _TENS_VALUES and nxt < len(keys):
                unit = _small(keys[nxt])
                if unit is not None and 1 <= unit <= 9:
                    value, nxt, compound = value + unit, nxt + 1, True
            group = group + value if last == "hundred" else value
            compound = compound or bool(last)
            index, last = nxt, "small"
            continue
        if key == "a" and not last and index + 1 < len(keys) and keys[index + 1] in _SCALES:
            group, index, last = 1, index + 1, "small"
            continue
        if key in _SCALES and last == "small":
            if _SCALES[key] == 100:
                if group >= 100:
                    break
                group, last = group * 100, "hundred"
            else:
                total, group, last = total + group * _SCALES[key], 0, "scale"
            compound, index = True, index + 1
            if (index + 1 < len(keys) and keys[index] == "and" and _small(keys[index + 1]) is not None):
                index += 1
            continue
        break
    if not compound or index - start < 2:
        return None, 0
    return total + group, index - start


def canonical_pairs(keys: Sequence[str], owners: Sequence[int], *, drop_fillers: bool = True
                    ) -> list[tuple[str, int]]:
    """``canonical_sequence`` that remembers, per canonical key, the owner of its first source key."""
    expanded: list[tuple[str, int]] = []
    for key, owner in zip(keys, owners):
        key = str(key)
        if drop_fillers and key in FILLERS:
            continue
        if key in _TENS:
            key = str(_TENS[key])               # "fifty" == "50" (zero..twenty are digits already)
        expanded.extend((part, owner) for part in _VARIANTS.get(key, (key,)))
    plain = [key for key, _owner in expanded]
    out: list[tuple[str, int]] = []
    index = 0
    while index < len(expanded):
        value, used = _parse_number(plain, index)
        if used:
            out.append((str(value), expanded[index][1]))
            index += used
        else:
            out.append(expanded[index])
            index += 1
    return out


def canonical_sequence(keys: Sequence[str], *, drop_fillers: bool = True) -> list[str]:
    """Comparison keys -> canonical keys (variants expanded, fillers dropped, numbers collapsed)."""
    return [key for key, _owner in canonical_pairs(keys, range(len(keys)), drop_fillers=drop_fillers)]


def harmless(left_keys: Sequence[str], right_keys: Sequence[str]) -> bool:
    """True when two key sequences differ only in ways that are never a caption error."""
    return canonical_sequence(left_keys) == canonical_sequence(right_keys)


def symbol_keys(token: str) -> list[str]:
    """Spoken words a written symbol stands for (``50%`` -> percent, ``$5`` -> dollars)."""
    text = str(token)
    extra: list[str] = []
    if "%" in text:
        extra.append("percent")
    if "$" in text:
        extra.append("dollars")
    return extra
