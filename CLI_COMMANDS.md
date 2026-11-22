# wpiper CLI reference

Run everything from the wpiper folder with the venv active (`wpiper_venv\Scripts\activate`).
Commands are written as `python -m wpiper.<module>`. In PowerShell, use `` ` `` for line continuation; in cmd, `^`.

Typical pipeline:

```
prepare_big  ->  make_split  ->  train  ->  export_onnx  ->  synthesize
                                   \-> viewer (watch progress / samples)
```

---

## 1. Prepare a dataset

### `wpiper.prepare_big` - large single-speaker dataset (memory-mapped format)
Reads `path|text` metadata, resamples/trims in parallel, phonemizes with espeak, writes the dataset.
Safe to re-run: output goes to `<out>.partial` and is renamed only when complete.

```bash
python -m wpiper.prepare_big --csv "E:\path\metadata.csv" --out data\myvoice
python -m wpiper.prepare_big --csv metadata.csv --out data\test --limit 200      # quick smoke test
```

| Flag | Default | Meaning |
|---|---|---|
| `--csv` | required | metadata file, `path\|text` per line (paths relative to the csv) |
| `--audio-dir` | - | override the base directory for audio paths |
| `--out` | required | output dataset folder |
| `--espeak-voice` | `en-us` | espeak phonemizer voice |
| `--sample-rate` | `22050` | target sample rate |
| `--min-sec` / `--max-sec` | `0.5` / `30.0` | drop clips outside this duration |
| `--trim-db` | `40.0` | silence trim threshold (0 disables) |
| `--keep-sec` | `0.25` | silence kept at each end after trimming |
| `--num-symbols` | `256` | phoneme symbol count |
| `--workers` | `12` | parallel processes (lower this on a small PC) |
| `--limit` | `0` | only first N rows (0 = all) |

### `wpiper.prepare_dataset` - small single-speaker dataset (per-utterance cache files)
Reads `id|text` metadata; writes cache files, Piper `config.json`, and a train/val/test split.

```bash
python -m wpiper.prepare_dataset --csv christopher\metadata.csv --audio-dir christopher\wavs --out data\christopher --name christopher
```

| Flag | Default | Meaning |
|---|---|---|
| `--csv`, `--audio-dir`, `--out` | required | metadata, wav folder, output folder |
| `--name` | `voice` | voice name |
| `--espeak-voice` | `en-us` | phonemizer voice |
| `--sample-rate` | `22050` | |
| `--filter-length` / `--hop-length` / `--win-length` | `1024` / `256` / `1024` | STFT settings |
| `--num-symbols` | `256` | |
| `--val` / `--test` | `20` / `5` | held-out utterance counts |
| `--seed` | `1234` | split seed |
| `--no-trim` | off | disable silence trimming |
| `--trim-db` / `--keep-sec` | `40.0` / `0.25` | trim settings |

### `wpiper.make_split` - filter + train/val/test split (for `prepare_big` datasets)
Drops clips whose characters-per-second is outside a percentile band (usually bad audio), writes `split.json`.

```bash
python -m wpiper.make_split --data data\myvoice
python -m wpiper.make_split --data data\myvoice --cps-pct 0.5 99.5 --val 200 --test 10
python -m wpiper.make_split --data data\myvoice --cps-pct 0 100                  # no filtering
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | required | prepared dataset folder |
| `--cps-pct LOW HIGH` | `0.3 99.7` | keep clips inside these chars/sec percentiles |
| `--val` / `--test` | `20` / `10` | held-out clip counts |
| `--seed` | `1234` | |

### `scripts/add_wav_ext.py` - fix metadata filenames
Adds `.wav` (and optional folder prefix) to the filename in each metadata row.

```bash
python scripts\add_wav_ext.py metadata.csv                  # writes metadata_wav.csv
python scripts\add_wav_ext.py metadata.csv --prefix wavs/   # wavs/<name>.wav|text
python scripts\add_wav_ext.py metadata.csv --out fixed.csv
```

---

## 2. Train - `wpiper.train`

```bash
# new run from scratch
python -u -m wpiper.train --data data\myvoice --out runs\myvoice --precision bf16 --batch-size 48

# resume the same run (weights + optimizers + epoch counter)
python -u -m wpiper.train --data data\myvoice --out runs\myvoice --resume runs\myvoice\last.ckpt --precision bf16 --batch-size 48 --max-epochs 200

# fine-tune: new run (epoch 0) starting from another run's weights
python -u -m wpiper.train --data data\kim --out runs\kim --init-from runs\myvoice\last.ckpt --precision bf16 --batch-size 12 --max-epochs 140

# smoke test (also fine for CPU PC)
python -u -m wpiper.train --data data\test --out runs\smoke --max-steps 20 --batch-size 2
```

Full example used here:
```bash
python -u -m wpiper.train --data data\myvoice --out runs\myvoice --resume runs\myvoice\last.ckpt --precision bf16 --batch-size 48 --max-epochs 200 --val-every 1 --sample-every 5 --save-every 5 --keep-every 10 --log-every 50
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | required | prepared dataset folder |
| `--out` | required | run folder (checkpoints, samples, `metrics.jsonl`) |
| `--resume` | - | continue a run: weights, optimizers, epoch counter |
| `--init-from` | - | new run from another checkpoint's weights (fine-tuning) |
| `--batch-size` | `32` | max batch size |
| `--max-batch-frames` | `24000` | large format: padded frames per batch (batch size x longest clip) |
| `--max-epochs` | `10000` | stop at this epoch |
| `--max-steps` | `0` | stop after N steps (0 = off; for smoke tests) |
| `--precision` | `32` | `32`, `bf16`, or `16` (bf16/16 only apply on CUDA; CPU always runs fp32) |
| `--val-every` | `5` | epochs between validations |
| `--sample-every` | `50` | epochs between sample wavs (`runs/<n>/samples/epochN`) |
| `--save-every` | `10` | epochs between writes of `last.ckpt` |
| `--keep-every` | `100` | epochs between permanent `epoch-N.ckpt` |
| `--num-workers` | `0` | data loader processes (4 is plenty) |
| `--log-every` | `50` | steps between progress records for the viewer |
| `--seed` | `1234` | |

Note: training auto-uses CUDA if available, otherwise CPU. CPU training is very slow - use it only to verify the install.

---

## 3. Monitor - `wpiper.viewer`
Local web UI for training curves, samples and renders.

```bash
python -m wpiper.viewer                          # http://127.0.0.1:8765
python -m wpiper.viewer --port 9000 --data data\christopher
```

| Flag | Default | Meaning |
|---|---|---|
| `--port` | `8765` | |
| `--data` | `data\christopher` | dataset dir used for reference clips |

---

## 4. Export - `wpiper.export_onnx`
Exports the generator to ONNX in Piper's layout and writes `<output>.json` beside it.

```bash
python -m wpiper.export_onnx --checkpoint runs\myvoice\epoch-200.ckpt --config data\myvoice\config.json --output voices\myvoice_e200.onnx
python -m wpiper.export_onnx --checkpoint runs\christopher\last.ckpt --config data\christopher\config.json --output voices\christopher.onnx --no-verify
```

| Flag | Meaning |
|---|---|
| `--checkpoint` | required, `.ckpt` from a run |
| `--config` | `config.json` from the dataset |
| `--output` | required, `.onnx` path |
| `--no-verify` | skip the ONNX-vs-PyTorch check |

ONNX I/O: inputs `input [1, phonemes] int64`, `input_lengths [1] int64`, `scales [3] float` (noise_scale, length_scale, noise_w); output `[1, 1, samples] float`.

---

## 5. Inference - `wpiper.synthesize`
Text to speech with an exported ONNX voice (runs on CPU via onnxruntime).

```bash
# single sentence
python -m wpiper.synthesize --model voices\christopher.onnx --text "Hello there." --out out.wav

# tweak voice/pace, fixed seed
python -m wpiper.synthesize --model voices\christopher.onnx --text "Hello there." --out out.wav --noise-w 1.0 --seed 3

# pacing sweep: renders noise-w 0, 0.4, 0.8, 1.2 into out_nw*.wav
python -m wpiper.synthesize --model voices\christopher.onnx --text "Hello there." --out out.wav --sweep

# batch: one sentence per line, --out is a folder
python -m wpiper.synthesize --model voices\christopher.onnx --file samples.txt --out voices\e100
```

| Flag | Meaning |
|---|---|
| `--model` | required, `.onnx` voice |
| `--text` | sentence to speak |
| `--file` | text file, one sentence per line (`--out` becomes a folder) |
| `--out` | required, wav path (or folder with `--file`) |
| `--noise-scale` | voice variation (typical ~0.667) |
| `--length-scale` | speaking speed (higher = slower; 1.0 = normal) |
| `--noise-w` | rhythm/pacing variation (typical ~0.8) |
| `--seed` | reproducible output |
| `--sweep` | render several noise-w values |

---

## 6. Utilities

```bash
# fix metadata rows that leave out the .wav extension -> metadata_wav.csv
python scripts\add_wav_ext.py metadata.csv

# check the install
python -c "import torch, onnxruntime; print(torch.__version__, torch.cuda.is_available(), onnxruntime.get_available_providers())"

# training curves (each run writes runs\<name>\tb)
tensorboard --logdir runs
```

## 7. CPU-only quick check

```bash
setup_wpiper_cpu.bat
wpiper_venv\Scripts\activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"     # expect 2.6.0+cpu False
python -m wpiper.synthesize --model voices\christopher.onnx --text "Testing on CPU." --out cpu_test.wav
```
