"""Sentence splitting and chunking rules (standard library only).

Rules, in order:
  1. A sentence ends at . ! ? or an ellipsis (plus any closing quote/bracket) followed by a new sentence start
     (capital letter, digit or opening quote). Lower-case continuations never split ("... etc. and so on").
  2. Not a boundary: titles (Mr. Dr.), initials (J. R. Tolkien), dotted acronyms (e.g.), soft abbreviations
     (etc. vs. No. Fig. and months) unless a capital letter follows.
  3. Newlines: blank lines separate paragraphs. A single newline is either a soft wrap (hard-wrapped ebooks are
     unwrapped) or a real line break (one unit per line); wrap="auto" decides per document.
  4. A sentence longer than max_words is split at ; : -- , or before a conjunction, nearest the middle,
     never leaving a piece shorter than min_words.
"""

import re
from typing import List, Optional

CLOSERS = "\"'”’)]»"
OPENERS = "\"'“‘([«"

_CANDIDATE = re.compile(r"[.!?…]+[" + re.escape(CLOSERS) + r"]*(?=\s|$)")

TITLE_ABBR = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "fr", "rev", "hon", "gen", "col", "capt",
    "lt", "sgt", "cmdr", "maj", "supt", "messrs", "mme", "mlle",
}
SOFT_ABBR = {
    "etc", "vs", "no", "fig", "vol", "inc", "ltd", "co", "dept", "approx", "est", "ca", "cf", "al",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
SENTENCE_ENDING_ACRONYMS = {"u.s", "u.k", "u.s.a", "a.m", "p.m", "a.d", "b.c", "d.c"}

TERMINAL = ".!?…"
_SCENE_BREAK = re.compile(r"^[\s*#=~_•·.\-—]{3,}$")
_CONJUNCTIONS = {"and", "but", "or", "so", "because", "which", "that", "while", "although", "when", "then",
                 "until", "where", "whereas", "though", "if"}


def is_scene_break(line: str) -> bool:
    s = line.strip()
    return bool(s) and bool(_SCENE_BREAK.match(s)) and not any(c.isalnum() for c in s)


def ends_with_terminal(text: str) -> bool:
    return text.rstrip().rstrip(CLOSERS).endswith(tuple(TERMINAL))


# ---------------------------------------------------------------- sentences

def _first_letter_after_openers(rest: str) -> str:
    return rest.lstrip(OPENERS)[:1]


def _is_boundary(text: str, m: "re.Match") -> bool:
    rest = text[m.end():].lstrip()
    if not rest:
        return True
    nxt = rest[0]
    first = _first_letter_after_openers(rest)
    if nxt in OPENERS:
        # an opening quote or bracket starts a sentence; an apostrophe only before a capital ("'tis" does not)
        if nxt in "'‘":
            starts_new = first.isupper() or first.isdigit()
        else:
            starts_new = bool(first)
    else:
        starts_new = nxt.isupper() or nxt.isdigit()
    if not starts_new:
        return False

    core = m.group(0).rstrip(CLOSERS)
    if core != ".":
        return True  # ! ? ... followed by a new sentence start

    prev = re.search(r"([A-Za-z][A-Za-z.]*)$", text[:m.start()])
    if not prev:
        return True
    word = prev.group(1)
    lw = word.lower()
    if lw in TITLE_ABBR or lw in ("e.g", "i.e"):
        return False
    if len(word) == 1 and word.isupper() and word not in ("I", "A"):
        return False  # initial: J. R. R. Tolkien
    if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", word):
        return lw in SENTENCE_ENDING_ACRONYMS
    if lw in SOFT_ABBR and not first.isupper():
        return False
    return True


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out: List[str] = []
    start = 0
    for m in _CANDIDATE.finditer(text):
        if _is_boundary(text, m):
            piece = text[start:m.end()].strip()
            if piece:
                out.append(piece)
            start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


# ---------------------------------------------------------------- long sentences

def _candidates(words: List[str], min_words: int):
    """Split points (number of words in the first half) grouped by strength."""
    strong, comma, conj = [], [], []
    n = len(words)
    for i in range(min_words, n - min_words + 1):
        w = words[i - 1]
        bare = w.rstrip(CLOSERS)
        if bare.endswith((";", ":", "—", "--")) or w in ("—", "--"):
            strong.append(i)
        elif bare.endswith(","):
            comma.append(i)
        elif words[i].lower().strip(",") in _CONJUNCTIONS:
            conj.append(i)
    return strong, comma, conj


def split_long(sentence: str, max_words: int, min_words: int = 4) -> List[str]:
    words = sentence.split()
    if max_words <= 0 or len(words) <= max_words:
        return [sentence]
    n = len(words)
    min_words = max(1, min(min_words, n // 2))
    strong, comma, conj = _candidates(words, min_words)
    mid = n // 2
    cut: Optional[int] = None
    for tier in (strong, comma, conj):
        if tier:
            cut = min(tier, key=lambda i: abs(i - mid))
            break
    if cut is None:
        cut = mid
    left, right = " ".join(words[:cut]), " ".join(words[cut:])
    return split_long(left, max_words, min_words) + split_long(right, max_words, min_words)


# ---------------------------------------------------------------- newlines / paragraphs

def split_paragraphs(text: str) -> List[List[str]]:
    """Blank-line separated paragraphs, each as a list of stripped non-empty lines."""
    paragraphs: List[List[str]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if lines:
            paragraphs.append(lines)
    return paragraphs


def detect_wrap(paragraphs: List[List[str]]) -> str:
    """'hard' when single newlines are soft wraps of a fixed-width text file, else 'lines'."""
    lengths, nonterminal = [], 0
    for lines in paragraphs:
        for line in lines[:-1]:
            lengths.append(len(line))
            if not ends_with_terminal(line):
                nonterminal += 1
    if len(lengths) < 5:
        return "lines"
    ordered = sorted(lengths)
    maxlen = ordered[int(0.95 * (len(ordered) - 1))]
    near_max = sum(1 for n in lengths if n >= 0.8 * maxlen) / len(lengths)
    if maxlen >= 50 and nonterminal / len(lengths) >= 0.5 and near_max >= 0.55:
        return "hard"
    return "lines"


def join_wrapped(lines: List[str]) -> str:
    """Join soft-wrapped lines; 'exam-' + 'ple' becomes 'example'."""
    out = lines[0]
    for line in lines[1:]:
        if re.search(r"[A-Za-z]-$", out) and line[:1].islower():
            out = out[:-1] + line
        else:
            out = out + " " + line
    return out
