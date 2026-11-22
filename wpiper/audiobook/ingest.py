"""Reading a book into chapters (standard library only).

Input can be a .txt/.md file (chapters found by heading lines) or a folder of text files (one chapter each).
EPUB/PDF are left for later: epub needs only zipfile + html.parser, pdf needs a library.
"""

import re
from pathlib import Path
from typing import List, Optional, Union

from .config import BookConfig, Chapter
from .numbers import roman_to_int
from .splitter import is_scene_break

_HEADING = re.compile(
    r"""^\s*(?:
        (?:chapter|part|book|section|act|volume)\s+(?:[ivxlcdm]+|\d+|[a-z]+(?:[- ][a-z]+)?)\b[^\n]{0,60}
      | prologue | epilogue | preface | foreword | introduction | afterword | acknowledg(?:e)?ments? | appendix\b[^\n]{0,40}
      | \#{1,2}\s+[^\n]+
    )\s*$""",
    re.I | re.X,
)


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def _is_heading_line(lines: List[str], i: int) -> bool:
    line = lines[i].strip()
    if not line or len(line) > 80 or not _HEADING.match(line):
        return False
    prev_blank = i == 0 or not lines[i - 1].strip()
    next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()
    return prev_blank and (next_blank or len(lines[i + 1].strip()) <= 60)


_DIGITS_ONLY = re.compile(r"\d{1,4}")
_ROMAN_ONLY = re.compile(r"[IVXLCDM]{1,8}")
MAX_NUMBERED_CHAPTERS = 150      # more "chapters" than this are page numbers
MIN_CHAPTER_WORDS = 2            # fewer words than this under a heading: contents-list entry, not a chapter
MAX_CAPS_CHAPTERS = 60          # more caps titles than this are section labels / datelines, not chapters


def _isolated(lines: List[str], i: int) -> bool:
    return (i == 0 or not lines[i - 1].strip()) and (i + 1 >= len(lines) or not lines[i + 1].strip())


def _is_caps_title(s: str) -> bool:
    """ALL-CAPS line that reads as a title: PART I, THIS WRECKAGE, 12 AUGUST 2113."""
    letters = sum(c.isalpha() for c in s)
    return (3 <= letters and len(s) <= 60 and len(s.split()) <= 10 and s == s.upper()
            and not s.endswith((".", "!", "?", ",", ";", ":")) and not is_scene_break(s))


def _is_sequence(values: List[int]) -> bool:
    """True when values count up (1,2,3...). A restart at 1 is fine (several parts, a table of contents
    followed by the chapters); other jumps are tolerated only rarely."""
    if len(values) < 3:
        return False
    pairs = list(zip(values, values[1:]))
    good = sum(1 for a, b in pairs if b == a + 1)
    restarts = sum(1 for a, b in pairs if b == 1 and a != 0)
    other = len(pairs) - good - restarts
    return good / len(pairs) >= 0.7 and other <= max(1, len(pairs) // 10)


def split_chapters(text: str, default_title: str = "Book", cfg: Optional[BookConfig] = None) -> List[Chapter]:
    """Split one text into chapters at heading lines. No headings -> a single chapter.

    Beyond "Chapter N" style headings:
      * lines holding only a number (or roman numeral) become chapters when they count up 1, 2, 3 ...;
        any other digit-only line is a page number and is blanked out (drop_page_numbers);
      * ALL-CAPS title lines become chapters when the book has no other headings (e.g. part titles),
        otherwise they stay in the text as spoken sub-headings.
    """
    cfg = cfg or BookConfig()
    lines = text.split("\n")
    marks = []  # (line index, title, lines consumed by the heading)
    i = 0
    while i < len(lines):
        if _is_heading_line(lines, i):
            title = lines[i].strip().lstrip("#").strip()
            used = 1
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            after = lines[i + 2].strip() if i + 2 < len(lines) else ""
            # "Chapter 1" / "The Beginning" on two lines: join them
            if nxt and len(nxt) <= 60 and not nxt.endswith((".", "!", "?")) and not after:
                title = f"{title}. {nxt}"
                used = 2
            marks.append((i, title, used))
            i += used
        else:
            i += 1
    taken = {m[0] + k for m in marks for k in range(m[2])}
    standard = len(marks)

    # ---- number-only lines: chapter numbers or page numbers
    digit_lines = [(j, int(lines[j].strip())) for j in range(len(lines))
                   if j not in taken and _isolated(lines, j) and _DIGITS_ONLY.fullmatch(lines[j].strip())]
    roman_lines = [(j, roman_to_int(lines[j].strip())) for j in range(len(lines))
                   if j not in taken and _isolated(lines, j) and _ROMAN_ONLY.fullmatch(lines[j].strip())
                   and roman_to_int(lines[j].strip())]
    # runs that count up on neighbouring lines (1,2,3 ... a few lines apart) are a table of contents: blank them
    def drop_contents_runs(group):
        runs: list = []
        for item in group:
            if runs and item[1] == runs[-1][-1][1] + 1:
                runs[-1].append(item)
            else:
                runs.append([item])
        kept_items = []
        for run in runs:
            gaps = sorted(b[0] - a[0] for a, b in zip(run, run[1:]))
            if len(run) >= 3 and gaps[len(gaps) // 2] <= 3:
                for j, _ in run:
                    lines[j] = ""
            else:
                kept_items += run
        return kept_items

    digit_lines = drop_contents_runs(digit_lines)
    roman_lines = drop_contents_runs(roman_lines)
    chapter_numbers = []
    # real "Chapter N" headings mean bare numbers are more likely page numbers; Prologue/Book/Part do not
    chapter_style = sum(1 for m in marks if re.match(r"(?i)chapter\b", m[1]))
    if cfg.number_headings and chapter_style < 2:
        for group in (digit_lines, roman_lines):
            vals = [v for _, v in group]
            if len(group) <= MAX_NUMBERED_CHAPTERS and _is_sequence(vals):
                chapter_numbers = group
                break
    for j, value in chapter_numbers:
        label = cfg.number_label.strip()
        marks.append((j, f"{label} {value}" if label else str(value), 1))
        taken.add(j)
    if cfg.drop_page_numbers:
        for j, _ in digit_lines:
            if j not in taken:
                lines[j] = ""

    # ---- ALL-CAPS title lines
    caps = [j for j in range(len(lines))
            if j not in taken and _isolated(lines, j) and lines[j].strip() and _is_caps_title(lines[j].strip())]
    promote = (cfg.caps_headings == "chapters"
               or (cfg.caps_headings == "auto" and not marks and 2 <= len(caps) <= MAX_CAPS_CHAPTERS))
    if promote:
        for j in caps:
            marks.append((j, lines[j].strip(), 1))
    marks.sort(key=lambda m: m[0])

    # a heading with (almost) nothing under it is a table-of-contents entry: drop it and blank its line
    kept = []
    for k, (start, title, used) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        if len(" ".join(lines[start + used:end]).split()) < MIN_CHAPTER_WORDS:
            for t in range(start, start + used):
                lines[t] = ""
        else:
            kept.append((start, title, used))
    marks = kept

    if not marks:
        return [Chapter(0, default_title, "\n".join(lines).strip(), has_heading=False)]

    chapters: List[Chapter] = []
    front = "\n".join(lines[: marks[0][0]]).strip()
    if front:
        chapters.append(Chapter(len(chapters), "Front matter", front, has_heading=False))
    for k, (start, title, used) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        body = "\n".join(lines[start + used:end]).strip()
        chapters.append(Chapter(len(chapters), title, body, has_heading=True))
    return chapters


def load_book(path: Union[str, Path], cfg: Optional[BookConfig] = None) -> List[Chapter]:
    path = Path(path)
    if path.is_dir():
        files = sorted((p for p in path.iterdir() if p.suffix.lower() in (".txt", ".md")), key=_natural_key)
        if not files:
            raise FileNotFoundError(f"No .txt or .md files in {path}")
        chapters: List[Chapter] = []
        for f in files:
            text = read_text_file(f).strip()
            if not text:
                continue
            title = re.sub(r"^\d+[\s._-]*", "", f.stem).replace("_", " ").strip() or f.stem
            chapters.append(Chapter(len(chapters), title, text, has_heading=True))
        return chapters
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in (".txt", ".md", ""):
        raise ValueError(f"Unsupported input '{path.suffix}'. Use .txt/.md or a folder of text files.")
    return split_chapters(read_text_file(path), default_title=path.stem.replace("_", " "), cfg=cfg)
