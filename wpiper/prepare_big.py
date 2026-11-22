"""Prepare a large single-speaker dataset (LJSpeech-style metadata) for training.

Reads `path|text` lines, resamples/trims each clip in parallel, phonemizes the
text, and writes the memory-mapped format described in wpiper.bigdata. Safe to
re-run: output goes to <out>.partial and is renamed only when complete.

Usage:
    python -m wpiper.prepare_big --csv E:/.../tts_dataset/metadata.csv --out data/myvoice

Next: python -m wpiper.make_split --data data/myvoice
"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import platform
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import PhonemeType, PiperConfig
from .hparams import QUALITIES, QUALITY_SAMPLE_RATE
from .phoneme_ids import DEFAULT_PHONEME_ID_MAP, phonemes_to_ids

_LOGGER = logging.getLogger("prepare_big")
MARKUP = re.compile(r"\{\{|\}\}|\[\[|\]\]|\||<[^>]+>|https?://|www\.")

_W: dict = {}


def _init_worker(cfg: dict) -> None:
    from .phonemize_espeak import EspeakPhonemizer

    _W.update(cfg)
    _W["phonemizer"] = EspeakPhonemizer()


def _process(job):
    """One clip: returns (idx, status, audio int16 | None, phoneme ids | None, unknown phonemes)."""
    import librosa
    import soundfile as sf

    idx, path, text = job
    c = _W
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # unreadable file
        return idx, f"unreadable: {exc}", None, None, None
    audio = audio.mean(axis=1)
    if sr != c["sample_rate"]:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=c["sample_rate"], res_type="soxr_hq")
    if not len(audio) or float(np.abs(audio).max()) < 0.01:
        return idx, "silent", None, None, None

    if c["trim_db"] > 0:
        _, (start, end) = librosa.effects.trim(audio, top_db=c["trim_db"])
        keep = int(c["keep_sec"] * c["sample_rate"])
        audio = audio[max(0, start - keep) : min(len(audio), end + keep)]

    seconds = len(audio) / c["sample_rate"]
    if seconds < c["min_sec"]:
        return idx, "too_short", None, None, None
    if seconds > c["max_sec"]:
        return idx, "too_long", None, None, None

    sentences = _W["phonemizer"].phonemize(c["espeak_voice"], text)
    unknown = Counter(p for p in itertools.chain(*sentences) if p not in DEFAULT_PHONEME_ID_MAP)
    ids = list(itertools.chain(*(phonemes_to_ids(s) for s in sentences)))
    if len(ids) < 4:
        return idx, "no_phonemes", None, None, None

    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    return idx, "ok", pcm, np.asarray(ids, dtype=np.int16), unknown


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", required=True, type=Path, help="metadata: path|text (paths relative to the csv)")
    p.add_argument("--audio-dir", type=Path, help="override base directory for audio paths")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--espeak-voice", default="en-us")
    p.add_argument("--quality", choices=QUALITIES, default="medium",
                   help="model size this dataset is for; sets the default sample rate (x-low/low 16000, medium/high 22050)")
    p.add_argument("--language", help="language tag recorded in config.json, e.g. de_DE (--espeak-voice does the phonemizing)")
    p.add_argument("--sample-rate", type=int, help="default depends on --quality")
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=30.0)
    p.add_argument("--trim-db", type=float, default=40.0, help="0 disables trimming")
    p.add_argument("--keep-sec", type=float, default=0.25, help="silence kept at each end after trimming")
    p.add_argument("--num-symbols", type=int, default=256)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="only the first N rows (for a quick test)")
    args = p.parse_args()
    args.sample_rate = args.sample_rate or QUALITY_SAMPLE_RATE[args.quality]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    base = args.audio_dir or args.csv.parent
    partial = args.out.with_name(args.out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    rows, rejects = [], []
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        for n, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            src, _, text = line.partition("|")
            text = " ".join(text.split())
            if MARKUP.search(text):
                rejects.append((src, "markup", text))
            elif not text:
                rejects.append((src, "empty_text", ""))
            else:
                rows.append((src, text))
    if args.limit:
        rows = rows[: args.limit]
    _LOGGER.info("%d usable rows, %d rejected before audio (markup/empty)", len(rows), len(rejects))

    cfg = {k: getattr(args, k) for k in ("sample_rate", "min_sec", "max_sec", "trim_db", "keep_sec", "espeak_voice")}
    jobs = ((i, str(base / src), text) for i, (src, text) in enumerate(rows))

    kept = []  # (row index)
    a_off = p_off = 0
    audio_off, audio_len, ph_off, ph_len = [], [], [], []
    unknown_total: Counter = Counter()
    status_counts: Counter = Counter()
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool, \
            open(partial / "audio.bin", "wb") as fa, open(partial / "phonemes.bin", "wb") as fp:
        for done, (idx, status, pcm, ids, unknown) in enumerate(pool.imap(_process, jobs, chunksize=8), start=1):
            status_counts[status.split(":")[0]] += 1
            if status != "ok":
                rejects.append((rows[idx][0], status, rows[idx][1]))
            else:
                fa.write(pcm.tobytes())
                fp.write(ids.tobytes())
                kept.append(idx)
                audio_off.append(a_off)
                audio_len.append(len(pcm))
                ph_off.append(p_off)
                ph_len.append(len(ids))
                a_off += len(pcm)
                p_off += len(ids)
                unknown_total.update(unknown)
            if done % 500 == 0 or done == len(rows):
                rate = done / (time.time() - t0)
                _LOGGER.info("%d/%d  %.0f clips/s  ETA %.0f min  kept %d (%.1f h)",
                             done, len(rows), rate, (len(rows) - done) / rate / 60,
                             len(kept), a_off / args.sample_rate / 3600)

    np.savez(partial / "index.npz",
             audio_off=np.asarray(audio_off, dtype=np.int64), audio_len=np.asarray(audio_len, dtype=np.int32),
             ph_off=np.asarray(ph_off, dtype=np.int64), ph_len=np.asarray(ph_len, dtype=np.int32))
    with open(partial / "meta.tsv", "w", encoding="utf-8", newline="") as f:
        for k, idx in enumerate(kept):
            f.write(f"{k}\t{rows[idx][0]}\t{rows[idx][1]}\n")
    with open(partial / "rejected.tsv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(rejects)

    config = PiperConfig(
        num_symbols=args.num_symbols, num_speakers=1, sample_rate=args.sample_rate,
        espeak_voice=args.espeak_voice, phoneme_id_map=DEFAULT_PHONEME_ID_MAP,
        phoneme_type=PhonemeType.ESPEAK, piper_version="1.3.0",
        quality=args.quality, language_code=args.language,
    )
    (partial / "config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    import librosa
    import soundfile
    import torch

    from .phonemize_espeak import EspeakPhonemizer

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "args": {k: str(v) for k, v in vars(args).items()},
        "source_csv_sha256": sha256(args.csv),
        "rows_in_csv": len(rows) + sum(1 for r in rejects if r[1] in ("markup", "empty_text")),
        "clips_kept": len(kept),
        "hours_kept": round(a_off / args.sample_rate / 3600, 3),
        "rejected_by_reason": dict(Counter(r[1].split(":")[0] for r in rejects)),
        "phonemes_not_in_map": dict(unknown_total),
        "phonemizer_backend": EspeakPhonemizer().backend,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "librosa": librosa.__version__,
                     "soundfile": soundfile.__version__, "torch": torch.__version__},
        "seconds": round(time.time() - t0),
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.out.exists():
        shutil.rmtree(args.out)
    partial.rename(args.out)
    _LOGGER.info("done: %d clips, %.2f h -> %s", len(kept), manifest["hours_kept"], args.out)
    _LOGGER.info("rejected: %s", manifest["rejected_by_reason"])
    if unknown_total:
        _LOGGER.warning("phonemes missing from id map (dropped): %s", dict(unknown_total))


if __name__ == "__main__":
    main()
