"""Writing the finished audio: WAV with the standard library, everything else through ffmpeg."""

import functools
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import BookConfig

EXT = {"wav": ".wav", "flac": ".flac", "mp3": ".mp3", "opus": ".opus", "m4b": ".m4b"}


class WavWriter:
    """Append-only 16-bit mono WAV (the standard-library wave module patches the header on close)."""

    def __init__(self, path: Path, sample_rate: int):
        self.path = Path(path)
        self._w = wave.open(str(self.path), "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(sample_rate)
        self.frames = 0

    def write(self, samples: np.ndarray) -> None:
        self._w.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())
        self.frames += samples.size

    def close(self) -> None:
        self._w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@functools.lru_cache(maxsize=None)
def _ffmpeg_works(path: str) -> bool:
    """True if this executable is a modern ffmpeg (very old builds reject -hide_banner)."""
    try:
        proc = subprocess.run([path, "-hide_banner", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=15)
        return proc.returncode == 0 and b"ffmpeg version" in proc.stdout[:200]
    except (OSError, subprocess.SubprocessError):
        return False


def _path_candidates() -> List[str]:
    names = ["ffmpeg.exe", "ffmpeg"] if os.name == "nt" else ["ffmpeg"]
    found: List[str] = []
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        for name in names:
            p = Path(folder.strip('"')) / name
            if p.is_file() and str(p) not in found:
                found.append(str(p))
    return found


def find_ffmpeg(cfg: Optional[BookConfig] = None) -> Optional[str]:
    """--ffmpeg, then $FFMPEG_EXE, then every ffmpeg on PATH; the first one that actually works wins.
    (A stale copy earlier on PATH, e.g. a 2013 build in a Python Scripts folder, is skipped.)"""
    explicit = [c for c in ((cfg.ffmpeg if cfg else ""), os.environ.get("FFMPEG_EXE", "")) if c]
    for candidate in explicit + _path_candidates():
        resolved = candidate if Path(candidate).is_file() else shutil.which(candidate)
        if resolved and _ffmpeg_works(resolved):
            return resolved
    return None


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + proc.stderr.decode(errors="replace").strip()[-800:])


def _codec_args(fmt: str, cfg: BookConfig, sample_rate: int) -> List[str]:
    rate = 24000 if fmt in ("opus", "m4b") and sample_rate not in (8000, 12000, 16000, 24000, 48000) else sample_rate
    args = ["-ac", "1", "-ar", str(rate)]
    if fmt == "mp3":
        args += ["-c:a", "libmp3lame"] + ["-b:a", cfg.bitrate or "64k"]  # constant bitrate: browsers seek VBR mp3 by estimate and drift
    elif fmt == "opus":
        args += ["-c:a", "libopus", "-b:a", cfg.bitrate or "32k"]
    elif fmt == "m4b":
        args += ["-c:a", "aac", "-b:a", cfg.bitrate or "64k"]
    elif fmt == "flac":
        args += ["-c:a", "flac"]
    return args


def _meta_args(cfg: BookConfig, title: str) -> List[str]:
    args = ["-metadata", f"title={title}"]
    if cfg.author:
        args += ["-metadata", f"artist={cfg.author}", "-metadata", f"album_artist={cfg.author}"]
    if cfg.title:
        args += ["-metadata", f"album={cfg.title}"]
    return args


def safe_name(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^\w\- ]+", "", text, flags=re.U).strip().replace(" ", "_")
    return (s or "untitled")[:limit]


def write_ffmetadata(path: Path, spans: List[dict], sample_rate: int, cfg: BookConfig, title: str) -> None:
    lines = [";FFMETADATA1", f"title={title}"]
    if cfg.author:
        lines.append(f"artist={cfg.author}")
    for s in spans:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(s['start'] * 1000 / sample_rate)}", f"END={int(s['end'] * 1000 / sample_rate)}",
                  f"title={s['title']}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def encode_master(master: Path, out: Path, fmt: str, cfg: BookConfig, sample_rate: int, ffmpeg: str,
                  title: str, meta_file: Optional[Path] = None) -> Path:
    """Encode the full master wav to `fmt`. m4b gets chapter markers from `meta_file`."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(master)]
    if fmt == "m4b" and meta_file is not None:
        cmd += ["-i", str(meta_file), "-map_metadata", "1", "-map_chapters", "1"]
    if cfg.loudnorm:
        cmd += ["-af", "loudnorm=I=-18:TP=-1.5:LRA=11"]
    cmd += _codec_args(fmt, cfg, sample_rate) + _meta_args(cfg, title)
    if fmt == "m4b":
        cmd += ["-f", "ipod"]
    cmd.append(str(out))
    _run(cmd)
    return out


def cut_chapter(master: Path, out: Path, fmt: str, start: int, end: int, sample_rate: int, cfg: BookConfig,
                ffmpeg: str, title: str) -> Path:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(master),
           "-ss", f"{start / sample_rate:.6f}", "-to", f"{end / sample_rate:.6f}"]
    if cfg.loudnorm:
        cmd += ["-af", "loudnorm=I=-18:TP=-1.5:LRA=11"]
    if fmt == "wav":
        cmd += ["-c:a", "pcm_s16le", "-ac", "1"]
    else:
        cmd += _codec_args(fmt, cfg, sample_rate)
    cmd += _meta_args(cfg, title)
    cmd.append(str(out))
    _run(cmd)
    return out
