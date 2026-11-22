"""Checks for the ported VITS code: alignment search and a forward/backward pass.

Usage: python tests/test_vits.py [data_dir]
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wpiper.bigdata import BigData, is_big_format  # noqa: E402
from wpiper.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch  # noqa: E402
from wpiper.vits.commons import slice_segments  # noqa: E402
from wpiper.vits.losses import (  # noqa: E402
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from wpiper.vits.models import MultiPeriodDiscriminator, SynthesizerTrn  # noqa: E402
from wpiper.vits.monotonic_align import maximum_path  # noqa: E402


def best_score(value: np.ndarray) -> float:
    """Independent DP: frame y picks text position x, x steps by 0 or 1 per frame."""
    t_y, t_x = value.shape
    best = np.full((t_y, t_x), -np.inf)
    best[0, 0] = value[0, 0]
    for y in range(1, t_y):
        for x in range(t_x):
            prev = best[y - 1, x]
            if x > 0:
                prev = max(prev, best[y - 1, x - 1])
            best[y, x] = prev + value[y, x]
    return best[-1, -1]


def test_alignment() -> None:
    rng = np.random.default_rng(0)
    b, max_y, max_x = 6, 60, 25
    t_ys = rng.integers(20, max_y + 1, b)
    t_xs = np.array([rng.integers(3, min(max_x, ty) + 1) for ty in t_ys])

    neg_cent = torch.from_numpy(rng.normal(size=(b, max_y, max_x)).astype(np.float32))
    mask = torch.zeros(b, max_y, max_x)
    for i in range(b):
        mask[i, : t_ys[i], : t_xs[i]] = 1

    path = maximum_path(neg_cent.clone(), mask).numpy()

    for i in range(b):
        p = path[i, : t_ys[i], : t_xs[i]]
        assert (p.sum(1) == 1).all(), "each frame must map to exactly one symbol"
        pos = p.argmax(1)
        assert pos[0] == 0 and pos[-1] == t_xs[i] - 1
        assert set(np.diff(pos)) <= {0, 1}, "path must be monotonic, step <= 1"
        assert path[i].sum() == t_ys[i], "nothing outside the valid region"
        got = float((neg_cent[i, : t_ys[i], : t_xs[i]].numpy() * p).sum())
        want = best_score(neg_cent[i, : t_ys[i], : t_xs[i]].numpy())
        assert abs(got - want) < 1e-3, (got, want)
    print("alignment: OK (valid, monotonic, optimal on", b, "random cases)")


def load_items(data_dir: Path, n: int) -> list:
    """First n clips as dicts with phoneme_ids, audio and spec, from either dataset format."""
    if is_big_format(data_dir):
        data = BigData(data_dir)
        items = []
        for i in range(min(n, len(data))):
            u = data.utterance(i)
            spec = spectrogram_torch(u.audio.unsqueeze(0), 1024, 22050, 256, 1024, center=False)[0]
            items.append({"phoneme_ids": u.phoneme_ids, "audio": u.audio, "spec": spec})
        return items
    files = sorted((data_dir / "cache").glob("*.pt"))[:n]
    return [torch.load(f, weights_only=True) for f in files]


def load_batch(data_dir: Path, n: int, device):
    items = load_items(data_dir, n)
    if not items:
        raise SystemExit(f"no clips found in {data_dir} (need cache/*.pt or audio.bin)")
    n = len(items)
    items.sort(key=lambda u: u["spec"].size(1), reverse=True)

    x = torch.zeros(n, max(len(u["phoneme_ids"]) for u in items), dtype=torch.long)
    spec = torch.zeros(n, 513, max(u["spec"].size(1) for u in items))
    audio = torch.zeros(n, 1, max(len(u["audio"]) for u in items))
    x_len, spec_len = torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long)
    for i, u in enumerate(items):
        x[i, : len(u["phoneme_ids"])] = u["phoneme_ids"]
        x_len[i] = len(u["phoneme_ids"])
        spec[i, :, : u["spec"].size(1)] = u["spec"]
        spec_len[i] = u["spec"].size(1)
        audio[i, 0, : len(u["audio"])] = u["audio"]
    return [t.to(device) for t in (x, x_len, spec, spec_len, audio)]


def test_model(data_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr, n_fft, hop, segment = 22050, 1024, 256, 8192

    g = SynthesizerTrn(
        n_vocab=256,
        spec_channels=n_fft // 2 + 1,
        segment_size=segment // hop,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        n_speakers=1,
        gin_channels=0,
        use_sdp=True,
    ).to(device)
    d = MultiPeriodDiscriminator(use_spectral_norm=False).to(device)
    n_g = sum(p.numel() for p in g.parameters()) / 1e6
    n_d = sum(p.numel() for p in d.parameters()) / 1e6
    print(f"device {device}: generator {n_g:.1f}M params, discriminator {n_d:.1f}M params")

    x, x_len, spec, spec_len, audio = load_batch(data_dir, 8, device)
    audio = torch.nn.functional.pad(audio, (0, max(0, segment - audio.size(-1))))

    t0 = time.time()
    y_hat, l_length, attn, ids_slice, _, z_mask, (_z, z_p, m_p, logs_p, _m_q, logs_q) = g(
        x, x_len, spec, spec_len, None
    )
    mel = spec_to_mel_torch(spec, n_fft, 80, sr, 0.0, None)
    y_mel = slice_segments(mel, ids_slice, segment // hop)
    y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), n_fft, 80, sr, hop, n_fft, 0.0, None)
    y = slice_segments(audio, ids_slice * hop, segment)
    y_hat = y_hat[..., : y.shape[-1]]

    y_d_r, y_d_g, fmap_r, fmap_g = d(y, y_hat)
    loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * 45
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
    loss_dur = torch.sum(l_length.float())
    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, _ = generator_loss(y_d_g)
    loss_g = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    loss_g.backward(retain_graph=True)

    y_d_r, y_d_g, _, _ = d(y, y_hat.detach())
    loss_d, _, _ = discriminator_loss(y_d_r, y_d_g)
    loss_d.backward()

    losses = dict(mel=loss_mel, kl=loss_kl, dur=loss_dur, fm=loss_fm, gen=loss_gen, d=loss_d)
    print({k: round(float(v), 3) for k, v in losses.items()}, f"{time.time() - t0:.1f}s")
    assert all(torch.isfinite(v) for v in losses.values()), "non-finite loss"
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in g.parameters()
        if p.requires_grad and p.grad is not None
    )
    print("alignment shape", tuple(attn.shape), "| y_hat", tuple(y_hat.shape))

    g.eval()
    with torch.no_grad():
        audio_out, *_ = g.infer(
            x[:1, : int(x_len[0])], x_len[:1], noise_scale=0.667, length_scale=1.0, noise_scale_w=0.8
        )
    print("infer ok, output samples:", audio_out.shape[-1])
    print("model: OK")


if __name__ == "__main__":
    test_alignment()
    test_model(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "christopher")
