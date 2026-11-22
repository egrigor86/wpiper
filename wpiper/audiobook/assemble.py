"""Joining cached clips into one master timeline.

The timeline is counted in samples, so subtitle times are exact for the audio we write:
a unit's start/end are the sample offsets of its trimmed clip, and every pause is an integer number of samples.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import BookConfig, Chapter, Placed, Unit
from .synth import ClipCache


def trim_clip(clip: np.ndarray, sr: int, cfg: BookConfig) -> np.ndarray:
    """Cut leading/trailing silence (relative to the clip's own peak) and apply a 4 ms fade. Returns float32."""
    if clip.size == 0:
        return np.zeros(0, dtype=np.float32)
    mag = np.abs(clip.astype(np.int32))
    peak = int(mag.max())
    if peak == 0:
        return np.zeros(0, dtype=np.float32)
    thr = max(int(peak * 10 ** (cfg.trim_db / 20.0)), 8)
    idx = np.flatnonzero(mag > thr)
    head = int(sr * cfg.trim_head_ms / 1000)
    tail = int(sr * cfg.trim_tail_ms / 1000)
    start = max(0, int(idx[0]) - head)
    end = min(clip.size, int(idx[-1]) + 1 + tail)
    out = clip[start:end].astype(np.float32)
    fade = min(int(sr * 0.004), out.size // 2)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out


def pause_ms(u: Unit, cfg: BookConfig) -> float:
    if u.kind == "scene_break":
        ms = cfg.pause_scene
    elif u.ends_chapter:
        ms = 0  # the chapter gap is added where the next chapter starts
    elif u.kind == "heading":
        ms = cfg.pause_heading
    elif u.ends_para:
        ms = cfg.pause_paragraph
    elif u.ends_sentence:
        ms = cfg.pause_sentence
    else:
        ms = cfg.pause_fragment
    return ms * cfg.pause_scale


def book_gain(units: List[Unit], keys: Dict[int, str], cache: ClipCache, cfg: BookConfig) -> float:
    """One gain for the whole book so that near-maximum peaks reach cfg.peak_target."""
    peaks = []
    for u in units:
        key = keys.get(u.idx)
        if key is None:
            continue
        clip = cache.load(key)
        if clip is not None and clip.size:
            peaks.append(int(np.abs(clip.astype(np.int32)).max()))
    if not peaks:
        return 1.0
    ref = float(np.percentile(peaks, 99.5))
    if ref <= 0:
        return 1.0
    return float(min(max(cfg.peak_target * 32767.0 / ref, 0.25), 12.0))


def assemble(
    units: List[Unit],
    keys: Dict[int, str],
    cache: ClipCache,
    sr: int,
    cfg: BookConfig,
    writer,
    gain: float,
    chapters: Optional[List[Chapter]] = None,
    progress: Optional[Callable[[str, int, int, str], None]] = None,
) -> Tuple[List[Placed], List[dict], int]:
    """Stream the master audio into `writer` (anything with .write(int16 array)).
    Returns (placements, chapter spans, total samples)."""
    titles = {c.index: c.title for c in (chapters or [])}
    placed: List[Placed] = []
    spans: List[dict] = []
    cursor = 0
    prev_chapter: Optional[int] = None

    def silence(ms: float) -> None:
        nonlocal cursor
        n = int(round(sr * ms / 1000.0))
        while n > 0:
            step = min(n, sr * 10)
            writer.write(np.zeros(step, dtype=np.int16))
            cursor += step
            n -= step

    for i, u in enumerate(units):
        if u.chapter != prev_chapter:
            if prev_chapter is not None:
                silence(cfg.pause_chapter * cfg.pause_scale)
            spans.append({"index": u.chapter, "title": titles.get(u.chapter, f"Chapter {u.chapter + 1}"),
                          "start": cursor})
            prev_chapter = u.chapter

        start = cursor
        key = keys.get(u.idx)
        if key is not None:
            clip = cache.load(key)
            audio = trim_clip(clip, sr, cfg) if clip is not None else np.zeros(0, dtype=np.float32)
            if audio.size:
                out = np.clip(audio * gain, -32768, 32767).astype(np.int16)
                writer.write(out)
                cursor += out.size
        placed.append(Placed(u, start, cursor))
        silence(pause_ms(u, cfg))
        if progress and (i % 50 == 0 or i == len(units) - 1):
            progress("assemble", i + 1, len(units), "")

    for k, span in enumerate(spans):
        span["end"] = spans[k + 1]["start"] if k + 1 < len(spans) else cursor
    return placed, spans, cursor
