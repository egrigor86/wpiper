"""End to end: text file -> chapters -> units -> cached clips -> master audio + subtitles -> encoded files.

Everything the web server needs is here as plain functions (progress callback, cancel event, optional lock and
preloaded voices), so a FastAPI route can call build_book() in a worker thread.
"""

import io
import json
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .assemble import assemble, book_gain
from .config import BookConfig, Chapter
from .export import (
    EXT, WavWriter, cut_chapter, encode_master, find_ffmpeg, safe_name, write_ffmetadata,
)
from .ingest import load_book
from .plan import build_units, estimate_hours
from .srt import build_cues, write_srt
from .synth import Cancelled, ClipCache, render_units
from .textclean import load_pronunciations

Progress = Optional[Callable[[str, int, int, str], None]]


def plan_book(source: Union[str, Path, List[Chapter]], cfg: BookConfig):
    chapters = source if isinstance(source, list) else load_book(source, cfg)
    pron = load_pronunciations(cfg.pronunciations)
    units = build_units(chapters, cfg, pron)
    if cfg.limit:
        units = units[: cfg.limit]
        if units:
            units[-1].ends_chapter = True
    return chapters, units


class _MemWriter:
    def __init__(self):
        self.parts: list = []

    def write(self, samples) -> None:
        self.parts.append(np.array(samples, dtype=np.int16))


def sample_wav(text: str, cfg: BookConfig, voice, cache: ClipCache, lock=None, max_units: int = 3) -> bytes:
    """Audition: the first few units of `text`, cleaned, paced and joined exactly like a real build (WAV bytes)."""
    cfg = replace(cfg, speak_headings=False, limit=max_units)
    chapters, units = plan_book([Chapter(0, "sample", text, False)], cfg)
    if not units:
        raise ValueError("Nothing to speak in that text.")
    keys, sr, _ = render_units(units, cfg, cache, None, None, lock, {"narrator": voice})
    writer = _MemWriter()
    assemble(units, keys, cache, sr, cfg, writer, book_gain(units, keys, cache, cfg), chapters)
    pcm = np.concatenate(writer.parts) if writer.parts else np.zeros(0, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def write_units(path: Path, units) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for u in units:
            f.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")


def _cut_wav(master: Path, out: Path, start: int, end: int) -> None:
    with wave.open(str(master), "rb") as src, wave.open(str(out), "wb") as dst:
        dst.setnchannels(src.getnchannels())
        dst.setsampwidth(src.getsampwidth())
        dst.setframerate(src.getframerate())
        src.setpos(start)
        remaining = end - start
        while remaining > 0:
            n = min(remaining, 1 << 20)
            dst.writeframes(src.readframes(n))
            remaining -= n


def build_book(
    source: Union[str, Path, List[Chapter]],
    out_dir: Union[str, Path],
    cfg: BookConfig,
    progress: Progress = None,
    cancel: Optional[threading.Event] = None,
    lock=None,
    voices: Optional[Dict] = None,
    dry_run: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
) -> dict:
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    chapters, units = plan_book(source, cfg)
    write_units(work / "units.jsonl", units)
    words = sum(len(u.spoken.split()) for u in units)
    summary: dict = {
        "chapters": len(chapters), "units": len(units), "spoken_words": words,
        "estimated_hours": round(estimate_hours(units), 2), "outputs": [],
    }
    if dry_run or not units:
        return summary
    if not cfg.voice:
        raise ValueError("cfg.voice is required (path to a .onnx voice)")

    cache = ClipCache(cache_dir or out_dir / "cache")
    keys, sr, stats = render_units(units, cfg, cache, progress, cancel, lock, voices)
    summary["synth"] = stats
    if stats["audio_seconds"]:
        summary["realtime_factor"] = round(stats["render_seconds"] / stats["audio_seconds"], 4)

    base = safe_name(cfg.title or (Path(source).stem if not isinstance(source, list) else "audiobook"))
    formats = [f.lower() for f in cfg.formats]
    for f in formats:
        if f not in EXT:
            raise ValueError(f"Unknown format '{f}' (use: {', '.join(EXT)})")
    master = out_dir / f"{base}.wav" if "wav" in formats else work / "master.wav"

    gain = book_gain(units, keys, cache, cfg)
    with WavWriter(master, sr) as writer:
        placed, spans, total = assemble(units, keys, cache, sr, cfg, writer, gain, chapters, progress)
    summary["gain"] = round(gain, 3)
    summary["duration_seconds"] = round(total / sr, 2)

    # subtitles
    cues = build_cues(placed, sr, cfg, offset_samples=int(sr * cfg.srt_offset_ms / 1000), total_samples=total)
    srt_path = out_dir / f"{base}.srt"
    summary["cues"] = write_srt(cues, srt_path, cfg)
    summary["outputs"] += [str(master) if "wav" in formats else None, str(srt_path)]

    # exact sentence starts for the web player (sample offsets, so as accurate as the audio itself)
    (work / "timeline.json").write_text(json.dumps([{
        "s": round(pl.start / sr, 3), "e": round(pl.end / sr, 3), "c": pl.unit.chapter, "k": pl.unit.kind, "t": pl.unit.text,
    } for pl in placed if pl.unit.kind != "scene_break"], ensure_ascii=False), encoding="utf-8")

    (work / "chapters.json").write_text(
        json.dumps([{**s, "start_sec": s["start"] / sr, "end_sec": s["end"] / sr} for s in spans], indent=1),
        encoding="utf-8")

    # encoded formats
    ffmpeg = find_ffmpeg(cfg)
    needs_ffmpeg = [f for f in formats if f != "wav"] or (cfg.split_chapters and cfg.loudnorm)
    if needs_ffmpeg and not ffmpeg:
        raise RuntimeError("ffmpeg not found. Install it, put it on PATH, or set FFMPEG_EXE / --ffmpeg.")
    book_title = cfg.title or base.replace("_", " ")
    for fmt in formats:
        if fmt == "wav":
            continue
        if progress:
            progress("encode", 0, 1, fmt)
        meta = None
        if fmt == "m4b":
            meta = work / "chapters.ffmetadata"
            write_ffmetadata(meta, spans, sr, cfg, book_title)
        out = out_dir / f"{base}{EXT[fmt]}"
        encode_master(master, out, fmt, cfg, sr, ffmpeg, book_title, meta)
        summary["outputs"].append(str(out))

    if cfg.split_chapters:
        ch_dir = out_dir / "chapters"
        ch_dir.mkdir(exist_ok=True)
        for n, span in enumerate(spans, start=1):
            stem = f"{n:02d}_{safe_name(span['title'])}"
            for fmt in formats:
                out = ch_dir / f"{stem}{EXT[fmt]}"
                if fmt == "wav" and not cfg.loudnorm:
                    _cut_wav(master, out, span["start"], span["end"])
                else:
                    cut_chapter(master, out, fmt, span["start"], span["end"], sr, cfg, ffmpeg, span["title"])
                summary["outputs"].append(str(out))
            chapter_cues = [c for c in cues if c.chapter == span["index"]]
            write_srt(chapter_cues, ch_dir / f"{stem}.srt", cfg, shift=span["start"] / sr)

    if "wav" not in formats:
        try:
            master.unlink()
        except OSError:
            pass
    summary["outputs"] = [o for o in summary["outputs"] if o]
    summary["seconds"] = round(time.perf_counter() - t_start, 2)
    (work / "report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
