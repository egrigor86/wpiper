"""Sentence rendering with an on-disk clip cache.

AudiobookVoice reuses wpiper.synthesize.Voice (phonemizer + id map) but
  * builds its onnxruntime session with explicit thread settings, and
  * does NOT peak-normalise each clip (Voice.synthesize does), so loudness stays consistent across sentences.
Clips are cached as int16 .npy files keyed by (voice, text, scales). The cache is what makes a re-run
reproducible and resumable: onnxruntime's noise sampling cannot be seeded, so a re-synthesis would differ.
"""

import hashlib
import json
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from ..synthesize import Voice
from .config import BookConfig, Unit


class AudiobookVoice(Voice):
    def __init__(self, model, providers=("CPUExecutionProvider",), threads: int = 0):
        model = Path(model)
        config = json.loads(model.with_suffix(".onnx.json").read_text(encoding="utf-8"))
        self.model_path = model
        self.sample_rate = config["audio"]["sample_rate"]
        self.espeak_voice = config["espeak"]["voice"]
        self.id_map = config["phoneme_id_map"]
        self.defaults = config.get("inference", {})
        from ..phonemize_espeak import EspeakPhonemizer
        self.phonemizer = EspeakPhonemizer()
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = int(threads)
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=opts, providers=list(providers))
        st = model.stat()
        self.voice_id = f"{model.stem}|{st.st_size}|{st.st_mtime_ns}"

    def scales(self, noise_scale=None, length_scale=None, noise_w=None) -> Tuple[float, float, float]:
        d = self.defaults
        return (
            float(d.get("noise_scale", 0.667) if noise_scale is None else noise_scale),
            float(d.get("length_scale", 1.0) if length_scale is None else length_scale),
            float(d.get("noise_w", 0.8) if noise_w is None else noise_w),
        )

    def render(self, text: str, scales: Tuple[float, float, float]) -> np.ndarray:
        """Raw float32 audio (no normalisation)."""
        ids = np.array([self.phoneme_ids(text)], dtype=np.int64)
        if ids.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        out = self.session.run(
            None,
            {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
             "scales": np.array(scales, dtype=np.float32)},
        )[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)


def to_int16(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)


class ClipCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(voice_id: str, spoken: str, scales: Tuple[float, float, float]) -> str:
        blob = json.dumps([voice_id, spoken, [round(s, 4) for s in scales]], ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npy"

    def load(self, key: str) -> Optional[np.ndarray]:
        p = self.path(key)
        if not p.exists():
            return None
        try:
            return np.load(p)
        except Exception:
            return None  # truncated by a crash: re-render

    def save(self, key: str, clip: np.ndarray) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, clip)
        tmp.replace(p)


class Cancelled(Exception):
    pass


def render_units(
    units: List[Unit],
    cfg: BookConfig,
    cache: ClipCache,
    progress: Optional[Callable[[str, int, int, str], None]] = None,
    cancel: Optional[threading.Event] = None,
    lock=None,
    voices: Optional[Dict[str, AudiobookVoice]] = None,
) -> Tuple[Dict[int, str], int, Dict[str, float]]:
    """Render (or load from cache) every unit. Returns ({unit idx: cache key}, sample_rate, stats).
    Clips stay on disk (a 20 hour book is about 3 GB); the assembler loads them one at a time.

    `lock` is an optional context manager held around each single sentence so a server can keep serving
    other requests during a long book. `voices` lets a caller pass an already-loaded voice (server use).
    Multi-speaker scaffold: units carry `speaker`; only the "narrator" voice is used for now.
    """
    lock = lock if lock is not None else nullcontext()
    voices = voices if voices is not None else {}
    if "narrator" not in voices:
        voices["narrator"] = AudiobookVoice(cfg.voice, cfg.providers, cfg.threads)
    voice = voices["narrator"]
    scales = voice.scales(cfg.noise_scale, cfg.length_scale, cfg.noise_w)

    keys: Dict[int, str] = {}
    todo = [u for u in units if u.spoken]
    stats = {"cached": 0, "rendered": 0, "render_seconds": 0.0, "audio_seconds": 0.0}
    import time

    for n, u in enumerate(todo, start=1):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        key = cache.key(voice.voice_id, u.spoken, scales)
        if cache.path(key).exists():
            stats["cached"] += 1
        else:
            t0 = time.perf_counter()
            with lock:
                raw = voice.render(u.spoken, scales)
            clip = to_int16(raw)
            stats["render_seconds"] += time.perf_counter() - t0
            stats["audio_seconds"] += len(clip) / voice.sample_rate
            stats["rendered"] += 1
            cache.save(key, clip)
        keys[u.idx] = key
        if progress and (n % 10 == 0 or n == len(todo)):
            progress("synth", n, len(todo), u.spoken[:60])
    return keys, voice.sample_rate, stats
