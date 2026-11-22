"""Subtitle (.srt) generation.

Timing source today: sentence-level sample offsets from the assembler (exact for the audio we wrote).
Inside one unit that has to be shown as several cues, time is shared out by character count, which is an
estimate. A timed ONNX export (per-phoneme durations) can replace `_split_text_times` later without
changing anything else here.
"""

import re
from dataclasses import dataclass
from typing import List

from .config import BookConfig, Placed


@dataclass
class Cue:
    start: float
    end: float
    text: str
    para: int
    chapter: int


def fmt_time(seconds: float) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _balanced_pieces(text: str, limit: int) -> List[str]:
    """Split text into the fewest roughly equal pieces of at most `limit` chars, preferring cuts after punctuation."""
    if len(text) <= limit:
        return [text]
    words = text.split()
    ends, pos = [], 0
    for w in words:
        pos += len(w) + (1 if ends else 0)
        ends.append(pos)
    k = -(-len(text) // limit)
    while k <= len(words):
        cuts: List[int] = []
        for j in range(1, k):
            target = len(text) * j / k
            best = None
            for i in range(cuts[-1] + 1 if cuts else 0, len(words) - 1):
                score = abs(ends[i] - target) - (6 if words[i].endswith((",", ";", ":", ".", "!", "?")) else 0)
                if best is None or score < best[0]:
                    best = (score, i)
            if best is None:
                break
            cuts.append(best[1])
        if len(cuts) == k - 1:
            pieces, prev = [], 0
            for c in cuts:
                pieces.append(" ".join(words[prev:c + 1]))
                prev = c + 1
            pieces.append(" ".join(words[prev:]))
            if all(len(p) <= limit for p in pieces):
                return pieces
        k += 1
    return [text]


def _split_text_times(text: str, t0: float, t1: float, limit: int) -> List[tuple]:
    """Break text longer than `limit` chars into balanced pieces; time is shared out by character count."""
    pieces = _balanced_pieces(text, limit)
    if len(pieces) == 1:
        return [(text, t0, t1)]
    total = sum(len(p) + 1 for p in pieces)
    out, t = [], t0
    for p in pieces:
        dt = (t1 - t0) * (len(p) + 1) / total
        out.append((p, t, t + dt))
        t += dt
    return out


def wrap_lines(text: str, max_chars: int, max_lines: int) -> str:
    if len(text) <= max_chars or max_lines < 2:
        return text
    words = text.split()
    best, best_score = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        # fits in two lines if both are short enough; otherwise take the most even split rather than one long line
        score = (max(len(a), len(b)) > max_chars, abs(len(a) - len(b)))
        if best_score is None or score < best_score:
            best, best_score = (a, b), score
    if best:
        return best[0] + "\n" + best[1]
    return text


def build_cues(placed: List[Placed], sr: int, cfg: BookConfig, offset_samples: int = 0,
               total_samples: int = 0) -> List[Cue]:
    # a little under chars x lines so that a full cue still wraps into lines of at most srt_max_chars
    limit = cfg.srt_max_chars * cfg.srt_max_lines - 6
    pieces: List[Cue] = []
    for p in placed:
        u = p.unit
        text = re.sub(r"\s+", " ", u.text).strip()
        if not u.spoken or not text or p.end <= p.start:
            continue
        t0, t1 = (p.start + offset_samples) / sr, (p.end + offset_samples) / sr
        for piece_text, a, b in _split_text_times(text, t0, t1, limit):
            pieces.append(Cue(a, b, piece_text, u.para, u.chapter))

    # a cue that would flash by is joined with its neighbour when both fit
    merged: List[Cue] = []
    for cue in pieces:
        if merged:
            prev = merged[-1]
            joined = prev.text + " " + cue.text
            if (prev.para == cue.para and prev.chapter == cue.chapter
                    and (prev.end - prev.start < cfg.srt_min_sec or cue.end - cue.start < cfg.srt_min_sec)
                    and len(joined) <= limit and cue.end - prev.start <= cfg.srt_max_sec):
                prev.text, prev.end = joined, cue.end
                continue
        merged.append(Cue(cue.start, cue.end, cue.text, cue.para, cue.chapter))

    linger = cfg.srt_linger_ms / 1000.0
    last_allowed = (total_samples + offset_samples) / sr if total_samples else float("inf")
    for i, cue in enumerate(merged):
        nxt = merged[i + 1].start if i + 1 < len(merged) else last_allowed
        cue.end = min(cue.end + linger, nxt)
    return merged


def write_srt(cues: List[Cue], path, cfg: BookConfig, shift: float = 0.0, first: int = 1) -> int:
    """Write cues (optionally shifted by -shift seconds, e.g. a chapter file). Returns the number written."""
    lines, n = [], 0
    for cue in cues:
        start, end = cue.start - shift, cue.end - shift
        if end <= 0:
            continue
        n += 1
        lines.append(str(first + n - 1))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(wrap_lines(cue.text, cfg.srt_max_chars, cfg.srt_max_lines))
        lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return n
