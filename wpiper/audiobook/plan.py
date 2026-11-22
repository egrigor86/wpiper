"""Chapters -> synthesis units. Pure text work: no model, no audio."""

import re
from typing import List, Optional

from .config import BookConfig, Chapter, Unit
from .splitter import (
    detect_wrap, ends_with_terminal, is_scene_break, join_wrapped, split_long, split_paragraphs, split_sentences,
)
from .textclean import Pronunciations, clean_for_speech, clean_heading, decap, has_speech

_SUBHEADING_MAX_WORDS = 12


def _finish_spoken(spoken: str, ends_sentence: bool) -> str:
    """Give the phonemizer a clause ending so intonation falls (sentence) or holds (fragment)."""
    if not spoken:
        return spoken
    last = spoken[-1]
    if ends_sentence and last not in ".!?…":
        return spoken + "."
    if not ends_sentence and last not in ",;:":
        return spoken.rstrip(".!?…") + ","
    return spoken


def build_units(chapters: List[Chapter], cfg: BookConfig, pron: Optional[Pronunciations] = None) -> List[Unit]:
    all_paragraphs = [split_paragraphs(ch.text) for ch in chapters]
    wrap = cfg.wrap
    if wrap == "auto":
        wrap = detect_wrap([p for chapter in all_paragraphs for p in chapter])

    units: List[Unit] = []
    sent_id = 0

    def add(chapter: int, para: int, sent: int, kind: str, text: str, spoken: str, ends_sentence: bool) -> Unit:
        u = Unit(idx=len(units), chapter=chapter, para=para, sent=sent, kind=kind, text=text, spoken=spoken,
                 ends_sentence=ends_sentence)
        units.append(u)
        return u

    for ch, paragraphs in zip(chapters, all_paragraphs):
        first_of_chapter = len(units)
        para_no = 0

        if cfg.speak_headings and ch.has_heading and ch.title:
            spoken = _finish_spoken(clean_heading(ch.title, pron), True)
            if has_speech(spoken):
                add(ch.index, para_no, sent_id, "heading", ch.title, spoken, True).ends_para = True
                sent_id += 1
                para_no += 1

        for lines in paragraphs:
            if len(lines) == 1 and is_scene_break(lines[0]):
                add(ch.index, para_no, sent_id, "scene_break", lines[0], "", True).ends_para = True
                sent_id += 1
                para_no += 1
                continue

            if wrap == "hard":
                segments = [join_wrapped(lines)]
            else:
                segments = lines

            para_first = len(units)
            for seg in segments:
                subheading = (
                    len(segments) == 1 and not ends_with_terminal(seg) and len(seg.split()) <= _SUBHEADING_MAX_WORDS
                    and len(paragraphs) > 1
                )
                for sentence in split_sentences(seg):
                    pieces = split_long(sentence, cfg.max_words, cfg.min_words)
                    for j, piece in enumerate(pieces):
                        last_piece = j == len(pieces) - 1
                        # short ALL-CAPS lines (part titles, datelines) are read as words, not spelled out
                        spoken = clean_for_speech(decap(piece) if len(piece.split()) <= 12 else piece, pron)
                        if not has_speech(spoken):
                            continue
                        spoken = _finish_spoken(spoken, last_piece)
                        add(ch.index, para_no, sent_id, "heading" if subheading else "sentence", piece, spoken,
                            last_piece)
                    sent_id += 1
            if len(units) > para_first:
                units[-1].ends_para = True
            para_no += 1

        if len(units) > first_of_chapter:
            units[-1].ends_chapter = True
            units[-1].ends_para = True

    for i, u in enumerate(units):
        u.idx = i
    return units


def estimate_hours(units: List[Unit], words_per_second: float = 2.6) -> float:
    words = sum(len(u.spoken.split()) for u in units)
    return words / words_per_second / 3600.0
