"""Export a trained generator to ONNX, in Piper's input/output layout.

Usage:
    python -m wpiper.export_onnx --checkpoint runs/christopher/last.ckpt \
        --config data/christopher/config.json --output voices/christopher.onnx

Inputs: input [1, phonemes] int64, input_lengths [1] int64,
        scales [3] float (noise_scale, length_scale, noise_w).
Output: [1, 1, samples] float. A voices/christopher.onnx.json is written beside it.
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from .hparams import Hparams, build_models

_LOGGER = logging.getLogger("export_onnx")
OPSET_VERSION = 15


def load_generator(checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hp = Hparams.from_dict(ckpt["hparams"])
    model_g, _ = build_models(hp)
    state = {k[8:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model_g.")}
    model_g.load_state_dict(state)
    model_g.eval()
    return model_g, hp, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="config.json from prepare_dataset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(1234)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model_g, hp, ckpt = load_generator(args.checkpoint)
    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    multi = model_g.n_speakers > 1

    def infer_forward(text, text_lengths, scales, sid=None):
        return model_g.infer(
            text, text_lengths, sid=sid,
            noise_scale=scales[0], length_scale=scales[1], noise_scale_w=scales[2],
        )[0].unsqueeze(1)

    # Keep a handle on the real infer for the numerical check below.
    reference = model_g.infer
    model_g.forward = infer_forward  # type: ignore[method-assign]

    sequences = torch.randint(0, hp.num_symbols, (1, 50), dtype=torch.long)
    lengths = torch.LongTensor([sequences.size(1)])
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy = (sequences, lengths, scales) + ((torch.LongTensor([0]),) if multi else ())
    input_names = ["input", "input_lengths", "scales"] + (["sid"] if multi else [])

    torch.onnx.export(
        model=model_g,
        args=dummy,
        f=str(args.output),
        opset_version=OPSET_VERSION,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 2: "time"},
        },
        dynamo=False,
    )
    _LOGGER.info("exported %s (%.1f MB), epoch %s", args.output, args.output.stat().st_size / 2**20, ckpt.get("epoch"))

    if args.config:
        shutil.copy(args.config, args.output.with_suffix(".onnx.json"))
        _LOGGER.info("config copied to %s", args.output.with_suffix(".onnx.json"))

    if args.no_verify:
        return

    # With noise at zero both paths are deterministic, so they must agree.
    import onnxruntime as ort

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ids = torch.randint(1, 100, (1, 40), dtype=torch.long)
    lens = torch.LongTensor([ids.size(1)])
    with torch.no_grad():
        want = reference(ids, lens, noise_scale=0.0, length_scale=1.0, noise_scale_w=0.0)[0].numpy()
    got = session.run(
        None,
        {"input": ids.numpy(), "input_lengths": lens.numpy(),
         "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
    )[0]
    got = got.reshape(want.shape)
    _LOGGER.info("verify: torch %s vs onnx %s, max abs diff %.2e", want.shape, got.shape, np.abs(want - got).max())
    assert np.abs(want - got).max() < 1e-3, "ONNX output differs from PyTorch"

    # Length scale must change duration
    long = session.run(
        None,
        {"input": ids.numpy(), "input_lengths": lens.numpy(),
         "scales": np.array([0.0, 1.5, 0.0], dtype=np.float32)},
    )[0]
    _LOGGER.info("verify: length_scale 1.0 -> %d samples, 1.5 -> %d samples", got.size, long.size)


if __name__ == "__main__":
    main()
