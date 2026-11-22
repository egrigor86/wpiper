"""Text to speech with an exported ONNX voice.

Usage:
    python -m wpiper.synthesize --model voices/christopher.onnx --text "Hello there." --out out.wav
    python -m wpiper.synthesize --model voices/christopher.onnx --text "..." --out out.wav --noise-w 1.0 --seed 3
    python -m wpiper.synthesize --model voices/christopher.onnx --text "..." --out pace.wav --sweep
    python -m wpiper.synthesize --model voices/christopher.onnx --file samples.txt --out voices/e100
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from .phoneme_ids import phonemes_to_ids
from .phonemize_espeak import EspeakPhonemizer


class Voice:
    def __init__(self, model: Path, providers=("CPUExecutionProvider",)):
        model = Path(model)
        config = json.loads(model.with_suffix(".onnx.json").read_text(encoding="utf-8"))
        self.sample_rate = config["audio"]["sample_rate"]
        self.espeak_voice = config["espeak"]["voice"]
        self.id_map = config["phoneme_id_map"]
        self.defaults = config.get("inference", {})
        self.phonemizer = EspeakPhonemizer()
        self.session = ort.InferenceSession(str(model), providers=list(providers))

    def phoneme_ids(self, text: str) -> list[int]:
        sentences = self.phonemizer.phonemize(self.espeak_voice, text)
        return list(itertools.chain(*(phonemes_to_ids(s, self.id_map) for s in sentences)))

    def synthesize(self, text, noise_scale=None, length_scale=None, noise_w=None, seed=None):
        """noise_w: duration randomness (pacing variation); length_scale: >1 slower."""
        d = self.defaults
        scales = np.array(
            [
                d.get("noise_scale", 0.667) if noise_scale is None else noise_scale,
                d.get("length_scale", 1.0) if length_scale is None else length_scale,
                d.get("noise_w", 0.8) if noise_w is None else noise_w,
            ],
            dtype=np.float32,
        )
        ids = np.array([self.phoneme_ids(text)], dtype=np.int64)
        if seed is not None:
            np.random.seed(seed)  # onnxruntime RandomNormalLike has no seed input; best effort
        audio = self.session.run(
            None,
            {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64), "scales": scales},
        )[0].squeeze()
        return audio / max(0.01, float(np.abs(audio).max()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--text")
    p.add_argument("--file", type=Path, help="one sentence per line; --out is then a folder")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--noise-scale", type=float)
    p.add_argument("--length-scale", type=float)
    p.add_argument("--noise-w", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--sweep", action="store_true", help="render noise-w 0, 0.4, 0.8, 1.2 into <out>_nw*.wav")
    args = p.parse_args()

    voice = Voice(args.model)
    if args.file:
        args.out.mkdir(parents=True, exist_ok=True)
        lines = [l.strip() for l in args.file.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, line in enumerate(lines, start=1):
            audio = voice.synthesize(line, args.noise_scale, args.length_scale, args.noise_w)
            sf.write(args.out / f"{i:02d}.wav", audio, voice.sample_rate)
            print(f"{i:02d}  {len(audio) / voice.sample_rate:5.2f}s  {line[:60]}")
        return
    if not args.text:
        p.error("give --text or --file")
    if args.sweep:
        for nw in (0.0, 0.4, 0.8, 1.2):
            audio = voice.synthesize(args.text, args.noise_scale, args.length_scale, nw)
            out = args.out.with_name(f"{args.out.stem}_nw{nw}{args.out.suffix}")
            sf.write(out, audio, voice.sample_rate)
            print(f"{out}  {len(audio) / voice.sample_rate:.2f}s")
        return

    audio = voice.synthesize(args.text, args.noise_scale, args.length_scale, args.noise_w, args.seed)
    sf.write(args.out, audio, voice.sample_rate)
    print(f"{args.out}  {len(audio) / voice.sample_rate:.2f}s")


if __name__ == "__main__":
    main()
