"""FastAPI server for wpiper ONNX voices (no piper.exe needed).

Run:
    python wpiper_server.py            # http://127.0.0.1:8035
Env:
    WPIPER_VOICES_DIR   folder of <name>.onnx + <name>.onnx.json   (default: ./voices)
    WPIPER_DEFAULT      default voice name                         (default: first voice found)
    WPIPER_HOST         interface to listen on                     (default: 127.0.0.1; 0.0.0.0 = whole network)
    WPIPER_PORT         port                                       (default: 8035)
    WPIPER_ADMIN_KEY    if set, /admin/* and /audiobook/api/* need this key (set it before exposing the server)
    WPIPER_PROVIDERS    comma list of onnxruntime providers        (default: CPUExecutionProvider)
    FFMPEG_EXE          only needed for response_format=aac
"""

import io
import json
import os
import re
import subprocess
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import soxr
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from wpiper.synthesize import Voice

BASE_DIR = Path(__file__).parent.resolve()
VOICES_DIR = Path(os.environ.get("WPIPER_VOICES_DIR") or BASE_DIR / "voices")
DEFAULT_VOICE_NAME = os.environ.get("WPIPER_DEFAULT", "")
PROVIDERS = tuple(os.environ.get("WPIPER_PROVIDERS", "CPUExecutionProvider").split(","))
LOG_FILE = BASE_DIR / "logs" / "generation_log.jsonl"
FFMPEG_EXE = os.environ.get("FFMPEG_EXE") or "ffmpeg"
_ffmpeg_path: Optional[str] = None


def ffmpeg_exe() -> str:
    """First working ffmpeg ($FFMPEG_EXE, then PATH), skipping stale copies; same lookup as the audiobook maker."""
    global _ffmpeg_path
    if _ffmpeg_path is None:
        try:
            from wpiper.audiobook.export import find_ffmpeg

            _ffmpeg_path = find_ffmpeg() or FFMPEG_EXE
        except Exception:  # noqa: BLE001
            _ffmpeg_path = FFMPEG_EXE
    return _ffmpeg_path

# response_format -> (soundfile format, subtype, media type); aac goes through ffmpeg
SF_FORMATS = {
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": ("RAW", "PCM_16", "audio/pcm"),
}
SUPPORTED_FORMATS = [*SF_FORMATS, "aac"]

app = FastAPI(title="wpiper TTS (ONNX)")

_voices: Dict[str, Voice] = {}
_lock = threading.Lock()  # espeak phonemizer is not thread-safe; also serializes ORT runs


def discover_voices() -> List[str]:
    return sorted(
        p.stem for p in VOICES_DIR.glob("*.onnx") if p.with_suffix(".onnx.json").exists()
    )


def default_voice() -> Optional[str]:
    names = discover_voices()
    if DEFAULT_VOICE_NAME in names:
        return DEFAULT_VOICE_NAME
    return names[0] if names else None


def resolve_voice(requested: Optional[str]) -> str:
    names = discover_voices()
    if not names:
        raise HTTPException(500, f"No voices found in {VOICES_DIR} (need <name>.onnx + <name>.onnx.json).")
    if requested and requested in names:
        return requested
    return default_voice()


def get_voice(name: str) -> Voice:
    if name not in _voices:
        _voices[name] = Voice(VOICES_DIR / f"{name}.onnx", providers=PROVIDERS)
    return _voices[name]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def encode(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    if fmt == "aac":
        wav = encode(audio, sample_rate, "wav")
        try:
            proc = subprocess.run(
                [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", "-",
                 "-f", "adts", "-codec:a", "aac", "-"],
                input=wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
            )
        except FileNotFoundError:
            raise HTTPException(500, f"ffmpeg not found ('{FFMPEG_EXE}'); needed for aac.")
        if proc.returncode != 0 or not proc.stdout:
            raise HTTPException(500, f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")
        return proc.stdout
    if fmt in ("opus", "pcm") and sample_rate != 24000:  # opus needs a supported rate; pcm matches OpenAI's 24 kHz
        audio, sample_rate = soxr.resample(audio, sample_rate, 24000), 24000
    sf_fmt, subtype, _ = SF_FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format=sf_fmt, subtype=subtype)
    return buf.getvalue()


def synth(text: str, voice_name: str, length_scale=None, noise_scale=None, noise_w=None, seed=None):
    try:
        with _lock:
            voice = get_voice(voice_name)
            audio = voice.synthesize(text, noise_scale, length_scale, noise_w, seed)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Synthesis failed for voice '{voice_name}': {e}")
    return audio, voice.sample_rate


class OpenAISpeechRequest(BaseModel):
    model: Optional[str] = None             # accepted, ignored
    input: str
    voice: Optional[str] = None             # any name in voices/; falls back to default
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0            # >1 faster; mapped to length_scale = 1/speed
    # wpiper extras
    noise_scale: Optional[float] = None
    noise_w: Optional[float] = None
    seed: Optional[int] = None


@app.post("/v1/audio/speech")
def openai_speech(req: OpenAISpeechRequest):
    text = clean_text(req.input or "")
    if not text:
        raise HTTPException(400, {"error": {"message": "input is required.", "type": "invalid_request_error"}})

    fmt = (req.response_format or "mp3").lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(400, f"Unsupported response_format '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}")
    speed = req.speed or 1.0
    if not 0.25 <= speed <= 4.0:
        raise HTTPException(400, "speed must be between 0.25 and 4.0.")

    voice_name = resolve_voice(req.voice)
    audio, sr = synth(text, voice_name, 1.0 / speed, req.noise_scale, req.noise_w, req.seed)
    data = encode(audio, sr, fmt)

    media_type = SF_FORMATS[fmt][2] if fmt in SF_FORMATS else "audio/aac"
    slug = "_".join(re.sub(r"[^a-zA-Z0-9\s]", "", text).split()[:6]).lower() or "speech"
    ext = "pcm" if fmt == "pcm" else fmt
    headers = {
        "Content-Disposition": f'attachment; filename="{int(time.time())}_{slug}.{ext}"',
        "X-Sample-Rate": str(24000 if fmt in ("opus", "pcm") else sr),
        "X-Voice": voice_name,
    }
    return StreamingResponse(io.BytesIO(data), media_type=media_type, headers=headers)


@app.get("/v1/audio/voices")
def list_voices():
    return JSONResponse({
        "voices": discover_voices(),
        "default": default_voice(),
        "supported_response_formats": SUPPORTED_FORMATS,
    })


# ===== Legacy routes (same contract as the old piper.exe server) =====
class TTSRequest(BaseModel):
    text: str
    username: Optional[str] = None
    voice: Optional[str] = None           # explicit voice takes priority
    sample_filename: Optional[str] = None  # kept for compatibility; unused


def pick_voice_name(username: str, explicit_voice: Optional[str]) -> str:
    names = discover_voices()
    if not names:
        raise HTTPException(500, f"No voices found in {VOICES_DIR} (need <name>.onnx + <name>.onnx.json).")
    if explicit_voice and explicit_voice in names:
        return explicit_voice
    if username in names:
        return username
    return default_voice()


def log_generation(entry: Dict):
    entry = {**entry, "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never break synthesis


@app.post("/speak")
def speak(req: TTSRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Text is required.")
    message = clean_text(req.text)
    username = (req.username or "default").strip().lower()
    voice_name = pick_voice_name(username, req.voice)

    audio, sr = synth(message, voice_name)
    data = encode(audio, sr, "wav")

    slug = "_".join(re.sub(r"[^a-zA-Z0-9\s]", "", message).split()[:6]).lower() or "speech"
    log_generation({"username": username, "voice": voice_name, "text_len": len(message),
                    "preview": message[:100], "slug": slug})
    headers = {"Content-Disposition": f'inline; filename="{int(time.time())}_{slug}.wav"'}
    return StreamingResponse(io.BytesIO(data), media_type="audio/wav", headers=headers)


@app.get("/profiles")
def get_profiles():
    return JSONResponse({"available_profiles": discover_voices(), "default_profile": default_voice(),
                         "voices_dir": str(VOICES_DIR.resolve())})


@app.get("/voices")
def get_voices():
    return JSONResponse({"voices": discover_voices(), "default_profile": default_voice(),
                         "voices_dir": str(VOICES_DIR.resolve())})


@app.get("/debug")
def debug():
    if not discover_voices():
        return JSONResponse({"status": "no_voices"}, status_code=500)
    voice_name = default_voice()
    try:
        audio, sr = synth("debug test.", voice_name)
    except HTTPException as e:
        return JSONResponse({"status": "error", "detail": e.detail}, status_code=500)
    return {"status": "ok", "voice": voice_name, "samples": int(len(audio)), "sample_rate": sr}


@app.get("/logs/summary")
def logs_summary(limit: int = 1000):
    if not LOG_FILE.exists():
        return {"entries": 0, "by_voice": {}, "by_user": {}}
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-limit:]
    by_voice: Dict[str, int] = {}
    by_user: Dict[str, int] = {}
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        v, u = obj.get("voice", "unknown"), obj.get("username", "unknown")
        by_voice[v] = by_voice.get(v, 0) + 1
        by_user[u] = by_user.get(u, 0) + 1
    return {"entries": len(lines), "by_voice": by_voice, "by_user": by_user}


# Dashboard (GET /) and /admin/* routes live in wpiper_dashboard.py
import sys
import wpiper_dashboard

wpiper_dashboard.register(app, sys.modules[__name__])

# Audiobook maker (page at GET /audiobook, API under /audiobook/api); never allowed to stop the TTS server starting
try:
    from wpiper.audiobook import server as audiobook_server

    audiobook_server.register(app, sys.modules[__name__])
except Exception as _e:  # noqa: BLE001
    print(f"audiobook module not loaded: {_e}")

if __name__ == "__main__":
    import uvicorn

    names = discover_voices()
    print(f"Voices in {VOICES_DIR}: {', '.join(names) or 'none'} (default: {default_voice()})")
    uvicorn.run(app, host=os.environ.get("WPIPER_HOST", "127.0.0.1"), port=int(os.environ.get("WPIPER_PORT", "8035")))
