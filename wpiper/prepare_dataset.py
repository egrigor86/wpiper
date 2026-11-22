"""Prepare a single-speaker LJSpeech-style dataset for VITS training.

Reads `id|text` metadata and writes, per utterance, a cache file holding the
phoneme ids, the 22.05 kHz audio and the linear spectrogram, plus a Piper
config.json and a deterministic train/val/test split.

Usage:
    python -m wpiper.prepare_dataset --csv christopher/metadata.csv \
        --audio-dir christopher/wavs --out data/christopher --name christopher
"""

import argparse
import csv
import itertools
import json
import logging
import random
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch

from .config import PhonemeType, PiperConfig
from .hparams import QUALITIES, QUALITY_SAMPLE_RATE
from .mel_processing import spectrogram_torch
from .phoneme_ids import DEFAULT_PHONEME_ID_MAP, phonemes_to_ids
from .phonemize_espeak import EspeakPhonemizer

_LOGGER = logging.getLogger("prepare_dataset")


def trim_edges(audio: np.ndarray, sample_rate: int, top_db: float, keep_sec: float):
    """Cut leading/trailing silence, keeping `keep_sec` on each side."""
    _, (start, end) = librosa.effects.trim(audio, top_db=top_db)
    keep = int(keep_sec * sample_rate)
    return audio[max(0, start - keep) : min(len(audio), end + keep)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--name", default="voice")
    parser.add_argument("--espeak-voice", default="en-us")
    parser.add_argument("--quality", choices=QUALITIES, default="medium",
                        help="model size this dataset is for; sets the default sample rate (x-low/low 16000, medium/high 22050)")
    parser.add_argument("--language", help="language tag recorded in config.json, e.g. de_DE")
    parser.add_argument("--sample-rate", type=int, help="default depends on --quality")
    parser.add_argument("--filter-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--num-symbols", type=int, default=256)
    parser.add_argument("--val", type=int, default=20, help="validation utterances")
    parser.add_argument("--test", type=int, default=5, help="test utterances")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--trim-db", type=float, default=40.0)
    parser.add_argument("--keep-sec", type=float, default=0.25)
    args = parser.parse_args()
    args.sample_rate = args.sample_rate or QUALITY_SAMPLE_RATE[args.quality]

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cache_dir = args.out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = PiperConfig(
        num_symbols=args.num_symbols,
        num_speakers=1,
        sample_rate=args.sample_rate,
        espeak_voice=args.espeak_voice,
        phoneme_id_map=DEFAULT_PHONEME_ID_MAP,
        phoneme_type=PhonemeType.ESPEAK,
        piper_version="1.3.0",
        quality=args.quality,
        language_code=args.language,
    )
    (args.out / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    phonemizer = EspeakPhonemizer()
    _LOGGER.info("phonemizer backend: %s", phonemizer.backend)

    rows = []
    with open(args.csv, "r", encoding="utf-8", newline="") as csv_file:
        for row in csv.reader(csv_file, delimiter="|"):
            if len(row) >= 2:
                rows.append((row[0], row[-1].strip()))

    unknown: Counter = Counter()
    utt_ids: list[str] = []
    seconds = 0.0
    max_phonemes = 0
    max_frames = 0

    for utt_id, text in rows:
        audio_path = args.audio_dir / utt_id
        if not audio_path.exists():
            audio_path = args.audio_dir / f"{utt_id}.wav"
        if not audio_path.exists():
            _LOGGER.warning("missing audio: %s", audio_path)
            continue

        sentences = phonemizer.phonemize(args.espeak_voice, text)
        for phoneme in itertools.chain(*sentences):
            if phoneme not in DEFAULT_PHONEME_ID_MAP:
                unknown[phoneme] += 1
        phoneme_ids = list(
            itertools.chain(*(phonemes_to_ids(s) for s in sentences))
        )

        audio, _ = librosa.load(audio_path, sr=args.sample_rate, mono=True)
        if not args.no_trim:
            audio = trim_edges(audio, args.sample_rate, args.trim_db, args.keep_sec)

        spec = spectrogram_torch(
            y=torch.from_numpy(audio).unsqueeze(0),
            n_fft=args.filter_length,
            sampling_rate=args.sample_rate,
            hop_size=args.hop_length,
            win_size=args.win_length,
            center=False,
        ).squeeze(0)

        torch.save(
            {
                "text": text,
                "phoneme_ids": torch.LongTensor(phoneme_ids),
                "audio": torch.from_numpy(audio),
                "spec": spec,
            },
            cache_dir / f"{utt_id}.pt",
        )

        utt_ids.append(utt_id)
        seconds += len(audio) / args.sample_rate
        max_phonemes = max(max_phonemes, len(phoneme_ids))
        max_frames = max(max_frames, spec.size(1))

    rng = random.Random(args.seed)
    shuffled = utt_ids[:]
    rng.shuffle(shuffled)
    test = sorted(shuffled[: args.test])
    val = sorted(shuffled[args.test : args.test + args.val])
    train = sorted(shuffled[args.test + args.val :])
    (args.out / "split.json").write_text(
        json.dumps({"train": train, "val": val, "test": test}, indent=2),
        encoding="utf-8",
    )

    _LOGGER.info(
        "%s: %d utterances, %.1f min, max %d phoneme ids, max %d frames",
        args.name, len(utt_ids), seconds / 60, max_phonemes, max_frames,
    )
    _LOGGER.info("split: %d train / %d val / %d test", len(train), len(val), len(test))
    if unknown:
        _LOGGER.warning("phonemes missing from id map (dropped): %s", dict(unknown))


if __name__ == "__main__":
    main()
