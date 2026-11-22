"""Text cleaning for speech (standard library only).

clean_for_speech() turns display text into text the phonemizer reads naturally:
quotes and dashes are normalised, abbreviations and numbers are spelled out, markup is removed.
The display text is kept separately (Unit.text) so subtitles show the author's words.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple, Union

from .numbers import (
    digits_to_words, int_to_words, ordinal_words, pluralize_year_words, roman_to_int, year_words,
)

Pronunciations = List[Tuple[Pattern, str]]

# ---------------------------------------------------------------- patterns

_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_FOOTNOTE = re.compile(r"\[\d{1,3}\]")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?", re.M)
_MD_EMPH = re.compile(r"\*+|(?<!\w)_+|_+(?!\w)")

_ABBREVIATIONS: List[Tuple[Pattern, str]] = [
    (re.compile(r"\bMr\."), "Mister"),
    (re.compile(r"\bMrs\."), "Missus"),
    (re.compile(r"\bMs\."), "Miss"),
    (re.compile(r"\bDr\."), "Doctor"),
    (re.compile(r"\bProf\."), "Professor"),
    (re.compile(r"\bJr\."), "Junior"),
    (re.compile(r"\bSr\."), "Senior"),
    (re.compile(r"\bvs\.?(?=\s)", re.I), "versus"),
    (re.compile(r"\be\.g\.", re.I), "for example"),
    (re.compile(r"\bi\.e\.", re.I), "that is"),
    (re.compile(r"\bNo\.\s*(?=\d)"), "Number "),
    (re.compile(r"\bapprox\.", re.I), "approximately"),
]
_ETC = re.compile(r"\betc\.", re.I)

_CUR = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
}

_NUM_RE = re.compile(
    r"""
      (?P<cur>[$£€])(?P<camt>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<cfrac>\d+))?(?:\s+(?P<cscale>thousand|million|billion|trillion)\b)?
    | (?P<hh>\d{1,2}):(?P<mm>[0-5]\d)(?!\d)(?:\s?(?P<ap>[ap]\.?m\b\.?))?
    | (?P<pnum>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<pfrac>\d+))?\s?%
    | (?P<ord>\d+)(?:st|nd|rd|th)\b
    | (?P<dec>(?:1[1-9]|20)\d0)'?s\b
    | (?<![\w.])(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?(?!\w)
    """,
    re.X | re.I,
)

_AFTER_MONTH = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec)\.?\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?$", re.I)
_DIGIT_RANGE =re.compile(r"(?<![\d-])(\d{1,4})\s?[–-]\s?(\d{1,4})(?![\d-])")
_HEADING_ROMAN = re.compile(r"^(chapter|part|book|act|section|volume)\s+([IVXLCDM]+)\b", re.I)


# ---------------------------------------------------------------- numbers

def _number_words(int_part: str, frac: Optional[str] = None) -> str:
    digits = int_part.replace(",", "")
    if len(digits) > 1 and digits.startswith("0") and "," not in int_part:
        words = digits_to_words(digits)  # 007 -> zero zero seven
    else:
        words = int_to_words(int(digits))
    if frac:
        words += " point " + digits_to_words(frac)
    return words


def _speak_currency(m: "re.Match") -> str:
    unit = _CUR[m.group("cur")]
    amt = int(m.group("camt").replace(",", ""))
    frac, scale = m.group("cfrac"), m.group("cscale")
    if scale:
        words = _number_words(m.group("camt"), frac)
        return f"{words} {scale} {unit[1]}"
    if frac and len(frac) == 2:
        cents = int(frac)
        parts = []
        if amt or not cents:
            parts.append(f"{int_to_words(amt)} {unit[0] if amt == 1 else unit[1]}")
        if cents:
            parts.append(f"{int_to_words(cents)} {unit[2] if cents == 1 else unit[3]}")
        return " ".join(parts)
    words = _number_words(m.group("camt"), frac)
    return f"{words} {unit[0] if (amt == 1 and not frac) else unit[1]}"


def _speak_time(m: "re.Match") -> str:
    h, mi = int(m.group("hh")), int(m.group("mm"))
    ap = m.group("ap")
    tail = ""
    if ap:
        letters = re.sub(r"[^ap]", "", ap.lower())[0]
        tail = f" {letters} m"
    if mi == 0:
        words = int_to_words(h) + ("" if ap else " o'clock")
    elif mi < 10:
        words = f"{int_to_words(h)} oh {int_to_words(mi)}"
    else:
        words = f"{int_to_words(h)} {int_to_words(mi)}"
    words += tail
    if ap and ap.endswith("."):  # keep a sentence-final period that the a.m./p.m. pattern consumed
        rest = m.string[m.end():]
        if not rest.strip() or re.match(r"\s+[A-Z\"']", rest):
            words += "."
    return words


def _number_handler(m: "re.Match") -> str:
    if m.group("cur"):
        return _speak_currency(m)
    if m.group("hh"):
        return _speak_time(m)
    if m.group("pnum"):
        return _number_words(m.group("pnum"), m.group("pfrac")) + " percent"
    if m.group("ord"):
        return ordinal_words(int(m.group("ord")))
    if m.group("dec"):
        return pluralize_year_words(year_words(int(m.group("dec"))))
    num, frac = m.group("num"), m.group("frac")
    digits = num.replace(",", "")
    if frac is None and "," not in num and len(digits) == 4:
        value = int(digits)
        if 1100 <= value <= 2099 or (1000 <= value <= 9999 and _AFTER_MONTH.search(m.string[:m.start()])):
            return year_words(value)  # 2113 after "August" is a year too (dates in science fiction)
    return _number_words(num, frac)


def expand_numbers(text: str) -> str:
    text = _DIGIT_RANGE.sub(lambda m: f"{m.group(1)} to {m.group(2)}", text)
    return _NUM_RE.sub(_number_handler, text)


# ---------------------------------------------------------------- main entry

def normalize_punctuation(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[“”„«»]", '"', text)
    text = re.sub(r"[‘’‚′]", "'", text)
    text = text.replace("…", "...")
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"[   ​﻿]", " ", text)
    return text


def _fix_commas(text: str) -> str:
    text = re.sub(r",(\s*,)+", ",", text)
    text = re.sub(r",\s*([.!?;:])", r"\1", text)
    text = re.sub(r"^[\s,]+", "", text)
    return text


def clean_for_speech(text: str, pronunciations: Optional[Pronunciations] = None) -> str:
    text = normalize_punctuation(text)
    text = _URL.sub("link", text)
    text = _FOOTNOTE.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_QUOTE.sub("", text)
    text = _MD_EMPH.sub("", text)

    for pattern, repl in _ABBREVIATIONS:
        text = pattern.sub(repl, text)
    text = _ETC.sub(lambda m: "et cetera" + ("." if not text[m.end():].strip() else ""), text)

    # dashes become pauses; leading dialogue dashes disappear
    text = re.sub(r"^\s*[—–-]+\s*", "", text)
    text = re.sub(r"\s*(?:—|--+)\s*", ", ", text)
    text = re.sub(r"\s+[–-]\s+", ", ", text)
    text = re.sub(r"(?<!\d)–(?!\d)", ", ", text)

    text = expand_numbers(text)

    text = text.replace("&", " and ").replace("@", " at ")
    text = re.sub(r"[|~^<>\\]", " ", text)
    text = text.replace('"', "")
    text = re.sub(r"(?<!\w)'|'(?!\w)", "", text)
    text = re.sub(r"\s*[(\[{]\s*", ", ", text)
    text = re.sub(r"\s*[)\]}]\s*", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _fix_commas(text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*$", "", text)

    if pronunciations:
        for pattern, repl in pronunciations:
            text = pattern.sub(repl, text)
    return text.strip()


_CANONICAL_ROMAN = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")
_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def decap(text: str) -> str:
    """'THIS WRECKAGE' -> 'This Wreckage' so the phonemizer reads words, not letters. Roman numerals stay."""
    if not text.isupper():
        return text

    def fix(m: "re.Match") -> str:
        w = m.group(0)
        if len(w) == 1 or (len(w) <= 6 and _CANONICAL_ROMAN.fullmatch(w) and roman_to_int(w)):
            return w
        return w.capitalize()

    return _WORD.sub(fix, text)


def clean_heading(title: str, pronunciations: Optional[Pronunciations] = None) -> str:
    """Like clean_for_speech, but also reads roman numerals after Chapter/Part/Book/... and
    turns ALL-CAPS titles into Title Case."""
    def _roman(m: "re.Match") -> str:
        value = roman_to_int(m.group(2))
        if value == 0 or m.group(2) != m.group(2).upper():
            return m.group(0)
        return f"{m.group(1)} {int_to_words(value)}"

    title = title.strip()
    was_upper = title.isupper()
    title = _HEADING_ROMAN.sub(_roman, title)
    if was_upper:
        title = _WORD.sub(lambda m: m.group(0).capitalize() if len(m.group(0)) > 1 and m.group(0).isupper()
                          and not (_CANONICAL_ROMAN.fullmatch(m.group(0)) and roman_to_int(m.group(0)))
                          else m.group(0), title)
    return clean_for_speech(title, pronunciations)


def has_speech(text: str) -> bool:
    return any(c.isalnum() for c in text)


# ---------------------------------------------------------------- pronunciation dictionary

def compile_pronunciations(mapping: Dict[str, str]) -> Pronunciations:
    out: Pronunciations = []
    for key in sorted(mapping, key=len, reverse=True):
        if not key.strip():
            continue
        out.append((re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)", re.I), mapping[key]))
    return out


def load_pronunciations(path: Union[str, Path, None]) -> Pronunciations:
    """JSON file of {"Hermione": "Her my oh knee", ...}. Matching is case-insensitive, whole words."""
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return compile_pronunciations({str(k): str(v) for k, v in data.items()})
