"""Plain-PyTorch VITS training loop (no Lightning, no CLI).

Usage:
    python -m wpiper.train --data data/christopher --out runs/christopher
    python -m wpiper.train --data data/christopher --out runs/christopher --resume runs/christopher/last.ckpt

Checkpoints use Piper's Lightning layout (state_dict keys "model_g.*" and
"model_d.*"), so they can be exchanged with Piper tooling later.
"""

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from torch.nn import functional as F

from .dataset import make_loaders
from .hparams import QUALITIES, Hparams, build_models
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from .vits.commons import slice_segments
from .vits.losses import discriminator_loss, feature_loss, generator_loss, kl_loss

_LOGGER = logging.getLogger("train")


def compute_losses(model_g, model_d, batch, hp, autocast_ctx):
    """Returns (generator loss dict, discriminator loss), mirroring Piper's step."""
    if batch.spectrograms is None:
        # Large-dataset format stores audio only; the spectrogram is cheap on the GPU.
        with torch.no_grad():
            batch.spectrograms = spectrogram_torch(
                batch.audios.squeeze(1), hp.filter_length, hp.sample_rate,
                hp.hop_length, hp.win_length, center=False,
            )

    with autocast_ctx():
        y_hat, l_length, _attn, ids_slice, _x_mask, z_mask, stats = model_g(
            batch.phoneme_ids, batch.phoneme_lengths,
            batch.spectrograms, batch.spectrogram_lengths, None,
        )
        _z, z_p, m_p, logs_p, _m_q, logs_q = stats

        segment_frames = hp.segment_size // hp.hop_length
        mel = spec_to_mel_torch(
            batch.spectrograms, hp.filter_length, hp.mel_channels,
            hp.sample_rate, hp.mel_fmin, hp.mel_fmax,
        )
        y_mel = slice_segments(mel, ids_slice, segment_frames)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(), hp.filter_length, hp.mel_channels, hp.sample_rate,
            hp.hop_length, hp.win_length, hp.mel_fmin, hp.mel_fmax,
        )
        y = slice_segments(batch.audios, ids_slice * hp.hop_length, hp.segment_size)
        y_hat = y_hat[..., : y.shape[-1]]

        _y_d_r, y_d_g, fmap_r, fmap_g = model_d(y, y_hat)

    with torch.autocast(y.device.type, enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel.float(), y_hat_mel.float()) * hp.c_mel
        loss_kl = kl_loss(z_p.float(), logs_q.float(), m_p.float(), logs_p.float(), z_mask) * hp.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_g)
        total = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

    parts = dict(g=total, mel=loss_mel, kl=loss_kl, dur=loss_dur, fm=loss_fm, adv=loss_gen)

    # Discriminator step uses the detached generator output
    with autocast_ctx():
        y_d_r, y_d_g, _, _ = model_d(y, y_hat.detach())
    with torch.autocast(y.device.type, enabled=False):
        loss_d, _, _ = discriminator_loss(y_d_r, y_d_g)

    return parts, loss_d


@torch.no_grad()
def synthesize_samples(model_g, test_utts, hp, device, out_dir: Path, scales=(0.667, 1.0, 0.8)):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_g.eval()
    for utt in test_utts:
        x = utt.phoneme_ids.unsqueeze(0).to(device)
        x_len = torch.LongTensor([x.size(1)]).to(device)
        audio = model_g.infer(
            x, x_len, noise_scale=scales[0], length_scale=scales[1], noise_scale_w=scales[2]
        )[0].squeeze().float().cpu()
        audio = audio / max(0.01, audio.abs().max().item())
        sf.write(out_dir / f"{utt.utt_id}.wav", audio.numpy(), hp.sample_rate)
    model_g.train()


def save_checkpoint(path, model_g, model_d, opts, scheds, epoch, step, hp):
    state = {f"model_g.{k}": v for k, v in model_g.state_dict().items()}
    state.update({f"model_d.{k}": v for k, v in model_d.state_dict().items()})
    torch.save(
        {
            "state_dict": state,
            "optimizer_states": [o.state_dict() for o in opts],
            "lr_schedulers": [s.state_dict() for s in scheds],
            "epoch": epoch,
            "global_step": step,
            "hparams": hp.to_dict(),
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--resume", type=Path, help="continue a run: weights, optimizers, epoch counter")
    p.add_argument("--init-from", type=Path, help="start a new run (epoch 0) from another run's weights, e.g. fine-tuning")
    p.add_argument("--batch-size", type=int, default=32, help="batch size (maximum, for the large format)")
    p.add_argument("--max-batch-frames", type=int, default=24000,
                   help="large format only: padded spectrogram frames per batch (batch size x longest clip)")
    p.add_argument("--max-epochs", type=int, default=10000)
    p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke tests)")
    p.add_argument("--precision", choices=["32", "bf16", "16"], default="32")
    p.add_argument("--val-every", type=int, default=5, help="epochs")
    p.add_argument("--sample-every", type=int, default=50, help="epochs")
    p.add_argument("--save-every", type=int, default=10, help="epochs, writes last.ckpt")
    p.add_argument("--keep-every", type=int, default=100, help="epochs, keeps epoch-N.ckpt")
    p.add_argument("--num-workers", type=int, default=0, help="data loader processes (large format: 4 is plenty)")
    p.add_argument("--log-every", type=int, default=50, help="steps between progress records for the viewer")
    p.add_argument("--quality", choices=QUALITIES,
                   help="model size for a NEW run; default is the quality in the dataset's config.json "
                        "(medium for older datasets). Resumed runs keep their checkpoint's size")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.data / "config.json").read_text(encoding="utf-8"))
    data_hp = dict(sample_rate=config["audio"]["sample_rate"], num_symbols=config["num_symbols"],
                   num_speakers=config["num_speakers"])
    # Architecture: a resumed/initialised run keeps its checkpoint's, otherwise --quality, otherwise
    # the quality prepare_* recorded in config.json (runs from before presets are all "medium").
    ref_path = args.resume or args.init_from
    ref_ckpt = torch.load(ref_path, map_location="cpu", weights_only=False) if ref_path else None
    if ref_ckpt is not None:
        ref_hp = Hparams.from_dict(ref_ckpt["hparams"])
        if args.quality and args.quality != ref_hp.quality:
            if args.resume:
                raise SystemExit(f"--resume: checkpoint is quality {ref_hp.quality!r}, not {args.quality!r}. "
                                 "A different size needs a new run from scratch.")
            _LOGGER.warning("init-from checkpoint is %r but --quality %s was asked for; "
                            "the shapes will not match", ref_hp.quality, args.quality)
        quality = args.quality or ref_hp.quality
    else:
        quality = args.quality or config["audio"].get("quality", "medium")
    hp = Hparams.for_quality(quality, **data_hp)
    _LOGGER.info("model quality %s (hidden %d, filter %d, upsample %s)", hp.quality, hp.hidden_channels,
                 hp.filter_channels, hp.upsample_rates)
    model_g, model_d = build_models(hp)
    model_g.to(device).train()
    model_d.to(device).train()

    opts = [
        torch.optim.AdamW(model_g.parameters(), lr=hp.learning_rate, betas=hp.betas, eps=hp.eps),
        torch.optim.AdamW(model_d.parameters(), lr=hp.learning_rate_d, betas=hp.betas_d, eps=hp.eps),
    ]
    scheds = [
        torch.optim.lr_scheduler.ExponentialLR(opts[0], gamma=hp.lr_decay),
        torch.optim.lr_scheduler.ExponentialLR(opts[1], gamma=hp.lr_decay_d),
    ]

    epoch, step = 0, 0
    if args.resume:
        ckpt = ref_ckpt
        sd = ckpt["state_dict"]
        model_g.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("model_g.")})
        model_d.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("model_d.")})
        for o, s in zip(opts, ckpt["optimizer_states"]):
            o.load_state_dict(s)
        for s, st in zip(scheds, ckpt["lr_schedulers"]):
            s.load_state_dict(st)
        epoch, step = ckpt["epoch"], ckpt["global_step"]
        _LOGGER.info("resumed from %s at epoch %d, step %d", args.resume, epoch, step)
    elif args.init_from:
        sd = ref_ckpt["state_dict"]
        for model, prefix in ((model_g, "model_g."), (model_d, "model_d.")):
            try:
                missing = model.load_state_dict(
                    {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}, strict=False
                )
            except RuntimeError as exc:
                raise SystemExit(f"--init-from {args.init_from} does not fit a {hp.quality!r} model "
                                 f"(different size?): {str(exc).splitlines()[0]}")
            _LOGGER.info("init-from %s: %s (missing %d, unexpected %d)", prefix, args.init_from,
                         len(missing.missing_keys), len(missing.unexpected_keys))

    train_loader, val_loader, test_utts = make_loaders(
        args.data, args.batch_size, hp.segment_size, args.num_workers,
        hop_length=hp.hop_length, max_batch_frames=args.max_batch_frames, seed=args.seed,
    )
    _LOGGER.info(
        "device %s | %d train / %d val | batch %d (%d steps/epoch) | precision %s",
        device, len(train_loader.dataset), len(val_loader.dataset), args.batch_size,
        len(train_loader), args.precision,
    )

    amp_dtype = {"16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)

    def autocast_ctx():
        if amp_dtype is None or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=amp_dtype)

    use_scaler = args.precision == "16" and device.type == "cuda"
    scalers = [torch.amp.GradScaler("cuda", enabled=use_scaler) for _ in range(2)]

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(args.out / "tb")
    except ImportError:
        pass

    def log_metric(**record) -> None:
        """Append one JSON line for the run viewer."""
        with open(args.out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.time(), **record}) + "\n")

    log_metric(type="start", epoch=epoch, max_epochs=args.max_epochs,
               batch_size=args.batch_size, precision=args.precision, data=str(args.data.resolve()),
               init_from=str(args.init_from) if args.init_from else None, quality=hp.quality,
               max_batch_frames=args.max_batch_frames, command=" ".join(sys.argv))

    done = False
    while epoch < args.max_epochs and not done:
        t0 = time.time()
        sums: dict = {}
        nb = 0
        sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = batch.to(device)
            parts, loss_d = compute_losses(model_g, model_d, batch, hp, autocast_ctx)

            opts[0].zero_grad(set_to_none=True)
            scalers[0].scale(parts["g"]).backward()
            scalers[0].step(opts[0])
            scalers[0].update()

            opts[1].zero_grad(set_to_none=True)
            scalers[1].scale(loss_d).backward()
            scalers[1].step(opts[1])
            scalers[1].update()

            step += 1
            nb += 1
            for k, v in {**parts, "d": loss_d}.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            if nb % args.log_every == 0:
                log_metric(type="progress", epoch=epoch + 1, batch=nb, batches=len(train_loader),
                           step=step, g=sums["g"] / nb)
                _LOGGER.info("  epoch %d: %d/%d batches, g=%.2f", epoch + 1, nb, len(train_loader), sums["g"] / nb)
            if args.max_steps and step >= args.max_steps:
                done = True
                break

        for s in scheds:
            s.step()
        epoch += 1

        final = done or epoch >= args.max_epochs
        avg = {k: v / max(1, nb) for k, v in sums.items()}
        msg = " ".join(f"{k}={v:.3f}" for k, v in avg.items())
        mem = f" mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G" if device.type == "cuda" else ""
        _LOGGER.info("epoch %d step %d %s (%.1fs%s)", epoch, step, msg, time.time() - t0, mem)
        log_metric(type="train", epoch=epoch, step=step, sec=round(time.time() - t0, 2), **avg)
        if writer:
            for k, v in avg.items():
                writer.add_scalar(f"train/{k}", v, epoch)

        if epoch % args.val_every == 0 or final:
            model_g.eval()
            model_d.eval()
            val = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    parts, _ = compute_losses(model_g, model_d, batch.to(device), hp, autocast_ctx)
                    val += float(parts["g"]) * batch.phoneme_ids.size(0)
            val /= len(val_loader.dataset)
            model_g.train()
            model_d.train()
            _LOGGER.info("epoch %d val_loss=%.3f", epoch, val)
            log_metric(type="val", epoch=epoch, loss=val)
            if writer:
                writer.add_scalar("val/loss_g", val, epoch)

        if epoch % args.sample_every == 0 or final:
            synthesize_samples(model_g, test_utts, hp, device, args.out / "samples" / f"epoch{epoch}")

        if epoch % args.save_every == 0 or final:
            save_checkpoint(args.out / "last.ckpt", model_g, model_d, opts, scheds, epoch, step, hp)
        if epoch % args.keep_every == 0:
            save_checkpoint(args.out / f"epoch-{epoch}.ckpt", model_g, model_d, opts, scheds, epoch, step, hp)

    _LOGGER.info("finished at epoch %d, step %d", epoch, step)


if __name__ == "__main__":
    main()
