# wpiper: Piper-style VITS training on plain Windows

wpiper is a Windows-native trainer for Piper-style VITS voices. It runs in the existing Matcha
virtualenv and never installs the `piper-tts` / `piper_train` packages. Its output is a Piper-layout
ONNX model, plus a Lightning-layout `.ckpt` for resuming and fine-tuning.

Environment it was built and tested on: Windows 11 Home, Python 3.10.0 (MSVC build), RTX 4060 Ti 16 GB,
CUDA 12.4 build of torch, 32 GB RAM, 16 cores, no C compiler installed.

## 1. Why upstream Piper does not train on Windows

Upstream Piper training depends on things that are painful or impossible on a stock Windows setup:

| Upstream dependency | Problem on Windows | What wpiper does instead |
|---|---|---|
| `monotonic_align` Cython extension (built with `build_monotonic_align.sh`) | Needs a C compiler and a bash build script | `wpiper/vits/monotonic_align.py`: a numba `@njit` port of `core.pyx` with the same `maximum_path(neg_cent, mask)` interface |
| `piper-phonemize` / `espeakbridge` built from C source | No Windows wheel for the version needed; building needs CMake and espeak-ng sources | Uses the **prebuilt** `espeakbridge.pyd` and `espeak-ng-data/` found in `old_repos/piper_onnx_u` (a prebuilt Windows build). If the bridge is missing, falls back to ctypes calls into an installed eSpeak NG DLL |
| PyTorch Lightning training module and CLI | Heavy, version-sensitive, and DataLoader worker/spawn behaviour on Windows is fragile | A plain PyTorch loop in `train.py` (manual two-optimizer GAN step). Checkpoints are still written in Lightning layout so they stay interchangeable |
| Multiprocessing DataLoader workers (fork assumed) | Windows only has `spawn`; workers re-import everything and were slow or hung | Default `--num-workers 0`. Data is memory-mapped and spectrograms are computed on the GPU, so loading is not the bottleneck |
| Per-file `.pt`/`.wav` datasets | Does not scale to 20k+ clips; tiny random reads on a slow external drive | `bigdata.py`: one `audio.bin` (int16) and one `phonemes.bin`, memory-mapped, plus an index |

### 1.1 Phonemization (`wpiper/phonemize_espeak.py`)

- Two backends with identical output format: `bridge` (the prebuilt Piper `espeakbridge.pyd`; exact
  Piper phonemes) and `ctypes` (the installed eSpeak NG 1.52 DLL).
- The installed 1.52 DLL does not export `espeak_TextToPhonemesWithTerminator`. The ctypes backend
  recovers clause terminators (`, . ? !`) by inspecting the text espeak consumed. The pointer stops one
  character past the terminator, so the look-ahead character is dropped before the check.
- ctypes still differs from Piper in one place (no `ʲ` glide between some vowels; 58 mismatches in the
  comparison test), so **do not mix ids from the two backends in one dataset**. The bridge is
  preferred; `manifest.json` records which backend built a dataset.
- The phoneme-to-id map (161 symbols, `num_symbols` 256, BOS/EOS/PAD with PAD interspersed) is copied
  from piper1-gpl (`phoneme_ids.py`, `const.py`), so ids match Piper's.

### 1.2 Monotonic alignment search (`wpiper/vits/monotonic_align.py`)

The Cython inner loops (`maximum_path_each` / `maximum_path_c`) are reimplemented with numba
(`@njit`, `parallel=True`, `prange` over the batch, `cache=True`). The first call compiles (a few
seconds); later runs load the cache. The Matcha venv already ships `core.cp310-win_amd64.pyd` built
for Matcha's own copy of this function, but wpiper uses the numba version so it does not depend on
a Matcha import.

### 1.3 Model code (`wpiper/vits/`)

`attentions.py`, `commons.py`, `losses.py`, `models.py`, `modules.py`, `transforms.py` come from
piper1-gpl's VITS implementation. `models.py` is `SynthesizerTrn` with the Piper layout:
`TextEncoder` (relative-position transformer, window 4), `PosteriorEncoder`, `ResidualCouplingBlock`
(flow), `StochasticDurationPredictor`, HiFi-GAN `Generator`, and `MultiPeriodDiscriminator`.
Defaults (`hparams.py`) match Piper "medium": 22050 Hz, n_fft 1024, hop 256, win 1024, 80 mel bins,
hidden 192, 6 layers, segment 8192 samples.

### 1.4 Training loop (`wpiper/train.py`)

- Two AdamW optimizers (G lr 2e-4, D lr 1e-4), per-epoch `ExponentialLR`.
- Losses: mel L1 (weight 45), KL, stochastic duration loss, feature matching, adversarial
  (least-squares). `--precision bf16` uses autocast; 12 GB VRAM at batch 48.
- Spectrograms are computed on the GPU from the audio, so the loader only returns int16 audio slices
  and phoneme ids.
- **Length bucketing**: `BucketBatchSampler` groups clips of similar length under a padded-frame
  budget (`--max-batch-frames`, default 24000) so 25-30 s clips and short clips both fit; batch size
  is capped by `--batch-size`.
- Checkpoints: `{state_dict: model_g.* / model_d.*, optimizer_states, lr_schedulers, epoch,
  global_step, hparams}` (Piper Lightning layout). `--resume` restores everything. `--init-from`
  loads only model weights into a fresh run (epoch 0), which is the fine-tune path.
- Output goes to `metrics.jsonl` (read by the viewer), TensorBoard in `tb/`, and sample WAVs every
  N epochs. `last.ckpt` is also written when the run ends.

### 1.5 Datasets

- Small (<~1 h): `prepare_dataset.py` writes one `.pt` per clip (`data/christopher`).
- Large: `prepare_big.py` then `make_split.py` (see section 3). Reads the source drive read-only,
  resamples to 22050 Hz, trims silence (40 dB, keeps 0.25 s each end), keeps clips 0.5-30 s, and
  rejects rows containing wiki/HTML markup. Writes to `<out>.partial` and renames only when
  complete; `manifest.json` records arguments, source CSV SHA-256, library versions, and counts.
- `make_split.py` drops clips whose characters-per-second fall outside a percentile band
  (default 0.3-99.7), and writes the filtered clips to `filtered.tsv` for review.

### 1.6 Export (`wpiper/export_onnx.py`, `wpiper/synthesize.py`)

ONNX opset 15, inputs `input`, `input_lengths`, `scales` = `[noise_scale, length_scale, noise_w]`,
matching Piper's runtime. Export verifies ONNX against PyTorch (max abs diff ~7e-6 on the last
check). `noise_w` controls duration variability, which is the main reason for this project: the
stochastic duration predictor gives varied pacing where Matcha's deterministic one does not.

### 1.7 Other Windows notes

- PowerShell 5.1 turns native stderr into `NativeCommandError`; run training from Git Bash, or
  redirect with `*>`, or just use `cmd`/Terminal.
- External drive: do not run a large random-access scan of the source drive while another job reads
  it; contention made reads drop to about 1 clip/s. Prepare once, then train from the fast local copy.
- Checkpoints are ~800 MB each; the trainer keeps rolling saves plus every Nth permanently.

## 2. Package versions

These are the versions the port was first built with (Python 3.10). The release is pinned and tested
on Python 3.12 instead; see `requirements_gpu.txt` / `requirements_cpu.txt` for the current list. Original core:

| Package | Version | Used for |
|---|---|---|
| Python | 3.10.0 | |
| torch / torchaudio | 2.6.0+cu124 | model, training, STFT/mel on GPU |
| numpy | 1.24.3 | |
| numba | 0.66.0 (llvmlite 0.48.0) | monotonic alignment search |
| scipy | 1.15.3 | |
| librosa | 0.11.0 | trimming, resampling |
| soxr | 1.1.0 | `soxr_hq` resampler |
| soundfile | 0.14.0 | WAV read/write |
| onnx | 1.22.0 | export |
| onnxruntime | 1.23.2 | synthesis and export verification |
| tensorboard | 2.21.0 | training curves |
| matplotlib | 3.10.9 | plots |
| tqdm | 4.69.1 | |

Present in the venv from Matcha but not used by wpiper: lightning / pytorch-lightning 2.6.5,
phonemizer 3.3.0, Cython 3.2.9, matcha-tts 0.0.7.2.

Non-pip binaries bundled in `wpiper/`: `espeakbridge.pyd` and `espeak-ng-data/` (from
a prebuilt Windows build of piper1-gpl). Optional fallback: eSpeak NG 1.52 installed to `C:\Program Files\eSpeak NG`
(or set `ESPEAK_NG_DIR`).

## 3. Reproducing the large-dataset run

```
set PY=wpiper_venv\Scripts\python.exe
%PY% -m wpiper.prepare_big --csv "path\to\tts_dataset\metadata.csv" --out data\myvoice
%PY% -m wpiper.make_split --data data\myvoice
%PY% -u -m wpiper.train --data data\myvoice --out runs\myvoice --precision bf16 --batch-size 48 --max-epochs 30 --val-every 1 --sample-every 2 --save-every 2 --keep-every 10
```

Result on the 23,418-row source: 23,416 clips / 33.4 h kept (30 markup rows, 2 clips over 30 s
rejected); split 23,064 train / 200 val / 10 test (142 dropped by chars/sec). Prep took 7 min at 59
clips/s. Training: 556 steps/epoch, about 9.5 min/epoch, final val loss about 27.8 at epoch 30.

Continue: add `--resume runs\myvoice\last.ckpt --max-epochs 100`.
Fine-tune on another voice: `--init-from runs\myvoice\epoch-N.ckpt` on a new `--out`.
Export: `%PY% -m wpiper.export_onnx --checkpoint runs\myvoice\epoch-30.ckpt --config data\myvoice\config.json --output voices\myvoice_e30.onnx`.

## 4. Why this was possible: Piper (VITS) and Matcha-TTS share a lineage

Both are descendants of the Glow-TTS/VITS line of work from the same research community, and the
Matcha codebase in this folder is built from the same building blocks. What is shared:

1. **Same front end contract.** Phonemes in, 22050 Hz / 80-bin mel frames at hop 256, n_fft 1024,
   win 1024. Matcha
   even uses `f_max` 8000, while Piper's default is the Nyquist frequency; the only audio-parameter difference.
2. **Same alignment method.** Both learn text-to-frame alignment with Monotonic Alignment Search
   (from Glow-TTS), with no external aligner. Matcha ships `matcha/utils/monotonic_align`
   (`core.pyx`) and Piper ships the same Cython function; this shared function is what I ported to numba.
3. **Same utility vocabulary.** `sequence_mask`, `generate_path`, `convert_pad_shape`, and
   `duration_loss` in `matcha/utils/model.py` correspond to the same-named helpers in VITS
   `commons.py`. The data pipeline (padded text batches, length masks, mel spectrograms) works the same way.
4. **Same style of text encoder.** Both use a stack of convolution + multi-head self-attention
   + FFN layers with LayerNorm over phoneme embeddings, producing per-phoneme statistics
   (Matcha `TextEncoder` and VITS `TextEncoder`). They differ in the position encoding (Matcha: rotary;
   VITS: relative-position window 4), but the data flowing in and out is the same shape.
5. **Same HiFi-GAN family for audio.** Matcha uses a HiFi-GAN vocoder (`matcha/hifigan/models.py`
   has `Generator`, `ResBlock1/2`, `MultiPeriodDiscriminator`, `MultiScaleDiscriminator`).
   VITS' decoder `Generator` is a HiFi-GAN generator with the same ResBlock design, and its
   discriminator is the same MPD. So the neural-vocoder code, weight-norm layers, and the GAN
   training pattern were already familiar and tested in this environment.
6. **Same toolchain and espeak phonemes.** Matcha's text cleaners call espeak-ng through
   `phonemizer`; Piper uses the same espeak-ng IPA phoneme set. Both run on plain PyTorch, so the
   Matcha venv (torch 2.6 CUDA 12.4, numpy, librosa, soundfile, onnx, onnxruntime, tensorboard) already
   had almost everything needed. Only numba had to be added (and was already present).

What is different, and why moving to VITS fixes the pacing problem:

| | Matcha-TTS | Piper / VITS |
|---|---|---|
| Duration model | deterministic `DurationPredictor` (one length per phoneme) | `StochasticDurationPredictor` (flow-based; sampled with `noise_w`) |
| Acoustic model | text encoder + conditional flow-matching U-Net decoder -> mel | text encoder + posterior encoder + normalizing flow -> latent |
| Waveform | separate HiFi-GAN vocoder | integrated HiFi-GAN generator, trained end-to-end |
| Prior on alignment | fixed unit-variance Gaussian prior | learned latent prior with flow |
| Control | `length_scale`, temperature | `length_scale`, `noise_scale`, `noise_w` |

Because Matcha predicts one fixed duration per phoneme, identical text always has identical pacing.
The VITS stochastic duration predictor samples durations, so `noise_w` gives natural variation in
rhythm. The components both systems share (front end, alignment, HiFi-GAN, data layout) are what made
a Windows port of the Piper side a matter of porting a few compiled pieces rather than rewriting a model.

## 5. Verified and known gaps

Verified:

- `--resume` continues runs correctly, restoring optimizer state, schedulers, epoch and step.
- `--init-from` fine-tunes from an existing checkpoint (for example the Christopher checkpoint).
- Exported ONNX voices work in standard Piper runtimes and libraries, and Piper voices work in wpiper.
- The core also runs on Linux (tested in Colab) after small changes to `phonemize_espeak.py`, though
  only Windows is supported.

Known gaps:

- DataLoader workers (`--num-workers > 0`) hung once in a big-format test while other heavy
  software was running; not re-tested. Default 0 is recommended.
- The viewer (`python -m wpiper.viewer`) shows live metrics and samples for any run; its
  reference-clip panel only reads the small `.pt` format so far.
- ONNX Runtime's random sampling ignores `--seed`, so two runs of the same text can differ slightly.
