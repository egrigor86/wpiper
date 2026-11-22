"""Audiobook web UI + API for wpiper_server (served under /audiobook).

Hooked in from wpiper_server.py with:
    from wpiper.audiobook import server as audiobook_server; audiobook_server.register(app, sys.modules[__name__])
Env:
    WPIPER_ADMIN_KEY      if set, /audiobook/api/* needs header X-Admin-Key (the page has a field for it)
    WPIPER_AUDIOBOOK_DIR  where jobs and the clip cache live (default: <server folder>/audiobook_jobs)

Only JSON in and out (no multipart), so no extra packages are needed. The page reads a text file in the
browser and posts its text. Synthesis shares the server's espeak/onnx lock one sentence at a time, so
/speak and /v1/audio/speech keep answering while a book is being built.
"""

import hmac
import json
import os
import re
import shutil
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from .config import BookConfig, Chapter
from .export import EXT, find_ffmpeg
from .ingest import split_chapters
from .jobs import Job, JobManager
from .pipeline import build_book, plan_book, sample_wav
from .synth import AudiobookVoice, ClipCache

HTML = Path(__file__).with_name("ui.html")
PLAYER = Path(__file__).with_name("player.html")
_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
MAX_TEXT_BYTES = 40 * 1024 * 1024
ADMIN_KEY = os.environ.get("WPIPER_ADMIN_KEY")

# user-settable BookConfig fields: name -> (type, min, max)  (strings use `choices`)
_INTS = {"max_words": (10, 200), "min_words": (1, 20), "pause_sentence": (0, 5000), "pause_fragment": (0, 3000),
         "pause_paragraph": (0, 8000), "pause_heading": (0, 8000), "pause_scene": (0, 10000),
         "pause_chapter": (0, 20000), "srt_offset_ms": (-2000, 2000), "threads": (0, 64)}
_FLOATS = {"noise_scale": (0.0, 2.0), "length_scale": (0.3, 3.0), "noise_w": (0.0, 2.0), "pause_scale": (0.0, 3.0)}
_BOOLS = {"speak_headings", "number_headings", "drop_page_numbers", "loudnorm"}
_CHOICES = {"wrap": {"auto", "hard", "lines"}, "caps_headings": {"auto", "chapters", "inline", "off"}}
_STRS = {"number_label": 30, "bitrate": 8}


def require_key(x_admin_key: str = Header(default=""), key: str = Query(default="")):
    """Header for API calls; ?key= only so that <a href> downloads and <audio src> can authenticate too."""
    if ADMIN_KEY and not (hmac.compare_digest(x_admin_key, ADMIN_KEY) or hmac.compare_digest(key, ADMIN_KEY)):
        raise HTTPException(401, "Invalid or missing admin key.")


def make_config(settings: dict, voice_path: Path, providers) -> BookConfig:
    """Validate user settings into a BookConfig; unknown keys are ignored, numbers are range-checked."""
    cfg = BookConfig(voice=str(voice_path), providers=tuple(providers))
    for name, value in (settings or {}).items():
        if value is None or value == "":
            continue
        try:
            if name in _INTS:
                lo, hi = _INTS[name]
                setattr(cfg, name, int(min(max(int(value), lo), hi)))
            elif name in _FLOATS:
                lo, hi = _FLOATS[name]
                setattr(cfg, name, float(min(max(float(value), lo), hi)))
            elif name in _BOOLS:
                setattr(cfg, name, bool(value))
            elif name in _CHOICES:
                if value not in _CHOICES[name]:
                    raise ValueError(f"{name} must be one of {sorted(_CHOICES[name])}")
                setattr(cfg, name, value)
            elif name in _STRS:
                setattr(cfg, name, str(value)[: _STRS[name]])
        except (TypeError, ValueError) as e:
            raise HTTPException(400, f"Bad setting '{name}': {e}")
    return cfg


class BuildRequest(BaseModel):
    title: str = Field(default="", max_length=200)
    author: str = Field(default="", max_length=200)
    voice: str = ""
    text: str
    settings: dict = {}
    formats: List[str] = ["mp3"]
    split_chapters: bool = False


class PreviewRequest(BaseModel):
    title: str = ""
    text: str
    settings: dict = {}
    limit: int = 150


class SampleRequest(BaseModel):
    voice: str = ""
    text: str
    settings: dict = {}


def _ffmpeg_info(cfg: Optional[BookConfig] = None) -> dict:
    exe = find_ffmpeg(cfg)
    if not exe:
        return {"found": False}
    try:
        out = subprocess.run([exe, "-hide_banner", "-version"], stdout=subprocess.PIPE, timeout=10).stdout
        version = out.decode(errors="replace").splitlines()[0]
    except Exception:
        version = ""
    return {"found": True, "path": exe, "version": version}


def register(app, srv) -> None:
    """srv is the running wpiper_server module (voices dir, lock, provider list)."""
    root = Path(os.environ.get("WPIPER_AUDIOBOOK_DIR") or srv.BASE_DIR / "audiobook_jobs")
    cache = ClipCache(root / "_cache")
    voice_cache: Dict[str, AudiobookVoice] = {}
    router = APIRouter(prefix="/audiobook/api", dependencies=[Depends(require_key)])

    def voice_path(name: str) -> Path:
        name = name or srv.default_voice()
        if not NAME_RE.match(name):
            raise HTTPException(400, "Voice name may only contain letters, digits, _ . -")
        p = srv.VOICES_DIR / f"{name}.onnx"
        if not p.exists() or not p.with_suffix(".onnx.json").exists():
            raise HTTPException(404, f"Voice '{name}' not found in {srv.VOICES_DIR}")
        return p

    def get_voice(path: Path, threads: int) -> AudiobookVoice:
        """Loaded under the server lock (espeak initialisation is not thread-safe). Keeps the last two voices."""
        key = f"{path}|{path.stat().st_mtime_ns}|{threads}"
        with srv._lock:
            if key not in voice_cache:
                while len(voice_cache) >= 2:
                    voice_cache.pop(next(iter(voice_cache)))
                voice_cache[key] = AudiobookVoice(path, srv.PROVIDERS, threads)
            return voice_cache[key]

    def run_job(job: Job, input_path: Path, out_dir: Path, cancel, progress) -> dict:
        req = job.request
        cfg = make_config(req.get("settings", {}), voice_path(job.voice), srv.PROVIDERS)
        cfg.title, cfg.author = req.get("title", ""), req.get("author", "")
        cfg.formats = tuple(req.get("formats") or ["wav"])
        cfg.split_chapters = bool(req.get("split_chapters"))
        voice = get_voice(Path(cfg.voice), cfg.threads)
        return build_book(input_path, out_dir, cfg, progress=progress, cancel=cancel, lock=srv._lock,
                          voices={"narrator": voice}, cache_dir=cache.root)

    jobs = JobManager(root, run_job)

    # ------------------------------------------------------------ info
    @router.get("/info")
    def info():
        names = srv.discover_voices()
        return {"voices": names, "default": srv.default_voice(), "formats": list(EXT),
                "ffmpeg": _ffmpeg_info(), "providers": list(srv.PROVIDERS),
                "defaults": {f.name: getattr(BookConfig(), f.name) for f in fields(BookConfig)
                             if f.name in {*_INTS, *_FLOATS, *_BOOLS, *_CHOICES, *_STRS}},
                "auth_required": bool(ADMIN_KEY)}

    # ------------------------------------------------------------ preview / audition
    @router.post("/preview")
    def preview(req: PreviewRequest):
        if len(req.text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise HTTPException(413, "Text is too large.")
        cfg = make_config(req.settings, Path("preview.onnx"), srv.PROVIDERS)
        chapters = split_chapters(req.text.replace("\r\n", "\n"), req.title or "Book", cfg)
        _, units = plan_book(chapters, cfg)
        per_chapter: Dict[int, dict] = {}
        for ch in chapters:
            per_chapter[ch.index] = {"index": ch.index, "title": ch.title, "units": 0, "words": 0}
        for u in units:
            row = per_chapter.get(u.chapter)
            if row is not None:
                row["units"] += 1
                row["words"] += len(u.spoken.split())
        words = sum(r["words"] for r in per_chapter.values())
        return {
            "chapters": [r for r in per_chapter.values() if r["units"]],
            "units": len(units), "words": words,
            "estimated_hours": round(words / 2.6 / 3600, 2),
            "fragments": sum(1 for u in units if not u.ends_sentence),
            "sample": [{"chapter": u.chapter, "kind": u.kind, "text": u.text, "spoken": u.spoken}
                       for u in units[: max(1, min(req.limit, 500))]],
        }

    @router.post("/sample")
    def sample(req: SampleRequest):
        if len(req.text) > 20000:
            raise HTTPException(413, "Audition text is limited to 20,000 characters.")
        cfg = make_config(req.settings, voice_path(req.voice), srv.PROVIDERS)
        try:
            voice = get_voice(Path(cfg.voice), cfg.threads)
            wav = sample_wav(req.text, cfg, voice, cache, lock=srv._lock)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return Response(wav, media_type="audio/wav")

    # ------------------------------------------------------------ jobs
    @router.post("/jobs")
    def create_job(req: BuildRequest):
        if not req.text.strip():
            raise HTTPException(400, "Text is empty.")
        if len(req.text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise HTTPException(413, "Text is too large.")
        formats = [f.lower() for f in req.formats] or ["wav"]
        bad = [f for f in formats if f not in EXT]
        if bad:
            raise HTTPException(400, f"Unknown format(s) {bad}; use {list(EXT)}")
        if any(f != "wav" for f in formats) and not find_ffmpeg():
            raise HTTPException(400, "ffmpeg is needed for formats other than wav and was not found. "
                                     "Install it, or set FFMPEG_EXE.")
        path = voice_path(req.voice)
        make_config(req.settings, path, srv.PROVIDERS)  # validate now, not an hour into the job
        title = req.title.strip() or "Audiobook"
        job = jobs.submit(req.text, title, path.stem,
                          {"title": title, "author": req.author, "settings": req.settings, "formats": formats,
                           "split_chapters": req.split_chapters, "chars": len(req.text)})
        return job.public()

    @router.get("/jobs")
    def list_jobs():
        return [j.public() for j in jobs.list()]

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "No such job.")
        return {**job.public(), "files": jobs.files(job_id)}

    @router.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = jobs.cancel(job_id)
        if not job:
            raise HTTPException(404, "No such job.")
        return job.public()

    @router.post("/jobs/{job_id}/resume")
    def resume_job(job_id: str):
        job = jobs.resume(job_id)
        if not job:
            raise HTTPException(409, "Only interrupted, cancelled or failed jobs can be resumed.")
        return job.public()

    @router.delete("/jobs/{job_id}")
    def delete_job(job_id: str):
        if not jobs.delete(job_id):
            raise HTTPException(409, "Job not found, or still queued/running (cancel it first).")
        return {"deleted": job_id}

    @router.get("/jobs/{job_id}/download/{rel:path}")
    def download(job_id: str, rel: str):
        p = jobs.file_path(job_id, rel)
        if p is None:
            raise HTTPException(404, "No such file.")
        return FileResponse(p, filename=p.name)

    @router.get("/jobs/{job_id}/stream/{rel:path}")
    def stream(job_id: str, rel: str):
        """Same file as /download but inline, so <audio> can seek (range requests)."""
        p = jobs.file_path(job_id, rel)
        if p is None:
            raise HTTPException(404, "No such file.")
        return FileResponse(p)

    @router.get("/jobs/{job_id}/player")
    def player_data(job_id: str):
        """Everything the player page needs: audio files, chapter times, subtitle cues."""
        job = jobs.get(job_id)
        if not job or job.status != "done":
            raise HTTPException(404, "No finished job with that id.")
        files = [f["path"] for f in jobs.files(job_id) if "/" not in f["path"]]
        audio = [f for f in files if f.lower().endswith((".mp3", ".opus", ".ogg", ".flac", ".wav", ".m4b", ".m4a"))]
        audio.sort(key=lambda f: [".mp3", ".opus", ".ogg", ".m4b", ".m4a", ".flac", ".wav"].index(Path(f).suffix.lower()))
        srt = next((f for f in files if f.lower().endswith(".srt")), None)
        cues = []
        if srt:
            text = jobs.file_path(job_id, srt).read_text(encoding="utf-8-sig", errors="replace")
            for block in re.split(r"(?:\r?\n){2,}", text.strip()):
                lines = block.splitlines()
                if len(lines) < 3:
                    continue
                m = _TIME.search(lines[1])
                if m:
                    g = list(map(int, m.groups()))
                    cues.append({"s": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                                 "e": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                                 "t": " ".join(lines[2:])})
        tl_file = jobs._dir(job_id) / "out" / "work" / "timeline.json"
        if tl_file.is_file():   # exact sentence starts; preferred over subtitle cues (whose mid-sentence splits are estimates)
            try:
                cues = [{"s": u["s"], "e": u["e"], "t": u["t"]} for u in json.loads(tl_file.read_text(encoding="utf-8"))]
            except (ValueError, KeyError):
                pass
        chapters = []
        ch_file = jobs._dir(job_id) / "out" / "work" / "chapters.json"
        if ch_file.is_file():
            try:
                chapters = [{"title": c.get("title", ""), "start": c["start_sec"], "end": c["end_sec"]}
                            for c in json.loads(ch_file.read_text(encoding="utf-8"))]
            except (ValueError, KeyError):
                chapters = []
        return {"id": job_id, "title": job.title, "author": job.request.get("author", ""),
                "audio": audio, "chapters": chapters, "cues": cues,
                "duration": (job.summary or {}).get("duration_seconds", 0)}

    # ------------------------------------------------------------ cache
    @router.get("/cache")
    def cache_info():
        files = list(cache.root.rglob("*.npy"))
        return {"clips": len(files), "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1)}

    @router.delete("/cache")
    def clear_cache():
        if any(j.status in ("queued", "running") for j in jobs.list()):
            raise HTTPException(409, "Wait for running jobs to finish before clearing the cache.")
        shutil.rmtree(cache.root, ignore_errors=True)
        cache.root.mkdir(parents=True, exist_ok=True)
        return {"cleared": True}

    app.include_router(router)

    @app.get("/audiobook/player", include_in_schema=False)
    def player_page():
        return HTMLResponse(PLAYER.read_text(encoding="utf-8"))

    @app.get("/audiobook", include_in_schema=False)
    def page():
        return HTMLResponse(HTML.read_text(encoding="utf-8"))
