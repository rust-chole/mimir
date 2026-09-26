"""Word normalization shared by alignment, verification and name lock."""
from __future__ import annotations

import re

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}


def normalize_word(text: str) -> str:
    return re.sub(r"[^\w']+", "", str(text).strip().casefold().replace("’", "'"))


def canonical_word(text: str) -> str:
    """Comparison key: casefold, punctuation-free, apostrophes dropped, number words as digits."""
    value = normalize_word(text).replace("'", "")
    return NUMBER_WORDS.get(value, value)


def tokenize(text: str) -> list[str]:
    """Visible tokens; a punctuation-only token attaches to the previous word."""
    result: list[str] = []
    for token in re.findall(r"\S+", str(text)):
        if not normalize_word(token):
            if result:
                result[-1] += token
            continue
        result.append(token)
    return result


def canon_phrase(tokens) -> tuple[str, ...]:
    return tuple(value for value in (canonical_word(t) for t in tokens) if value)


def preserve_case_and_punctuation(source: str, replacement: str) -> str:
    """Replace a token's lexical core while keeping its punctuation and case pattern."""
    match = re.match(r"^(\W*)([\w']+)(\W*)$", str(source))
    if not match:
        return replacement
    prefix, core, suffix = match.groups()
    value = replacement
    if len(core) > 1 and core.isupper():
        value = value.upper()
    elif core[:1].isupper():
        value = value[:1].upper() + value[1:]
    return f"{prefix}{value}{suffix}"
