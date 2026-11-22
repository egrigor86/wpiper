# wpiper

Train, run and serve [Piper](https://github.com/OHF-voice/piper1-gpl)-style VITS text-to-speech voices
on plain Windows: no WSL, no C compiler, no Docker.

- **Trainer**: prepare a dataset, train from scratch or fine-tune, preview checkpoints, export to ONNX.
  Use the web app (`start_trainer_web.bat`), the desktop GUI (`trainer_gui.bat`) or the CLI.
- **Synthesizer**: ONNX voices in Piper layout (`name.onnx` + `name.onnx.json`). The output works in
  Piper itself, and Piper voices work here.
- **TTS server**: an OpenAI-compatible `/v1/audio/speech` endpoint, plus a dashboard at `/`.
- **Audiobook maker**: turns a `.txt`/`.md` book (or a folder of chapters) into WAV, MP3, Opus, FLAC
  or M4B with chapters and `.srt` subtitles. It has a web page at `/audiobook` and a CLI.
- **Australian English (`en-au`)**: a new espeak-ng voice with Australian vowels (the "G'day" /
  "no worries" sounds, flat TRAP in *dance* and *sample*, and so on). Use it to train voices with an
  Australian accent.

## Pretrained voice

A ready-made English voice, **Christopher**, is on Hugging Face:
[Egrigor/wpiper-christopher](https://huggingface.co/Egrigor/wpiper-christopher). It has
the ONNX voice plus the full training checkpoint, which makes a good base for fine-tuning your own voice.
Download `christopher.onnx` and `christopher.onnx.json` into `voices\` to follow the examples below.
Any other Piper voice works too.

## Requirements

- Windows 10/11, 64-bit
- Python 3.12, 64-bit (tested). 3.11 should also work.
- For training: an NVIDIA GPU is strongly recommended. The CPU build works but is slow.
- Optional: [ffmpeg](https://ffmpeg.org/) on `PATH` for MP3/Opus/M4B audiobooks and AAC server output.

espeak-ng comes bundled (`wpiper\espeakbridge.pyd` + `wpiper\espeak-ng-data\`), so you don't need to
install it separately.

## Install

```
git clone https://github.com/egrigor86/wpiper.git
cd wpiper
setup_wpiper_gpu.bat      (NVIDIA GPU)
setup_wpiper_cpu.bat      (no GPU / synthesis only)
```

Each script creates `wpiper_venv\`, installs the pinned requirements, and installs the `wpiper`
package itself in editable mode.

Using it only as a library for synthesis and audiobooks? A light install skips torch:

```
pip install .                 # numpy, onnxruntime, soundfile
pip install .[server]         # + FastAPI server
```

## Quick start

### Speak some text

```
wpiper_venv\Scripts\activate
python -m wpiper.synthesize --model voices\christopher.onnx --text "Hello there, how are you going?" --out hello.wav
```

### Run the server and the audiobook maker

```
start_server.bat
```

- `http://127.0.0.1:8035/` is the dashboard
- `http://127.0.0.1:8035/audiobook` is the audiobook maker
- `POST /v1/audio/speech` is OpenAI-compatible (`input`, `voice`, `response_format`: wav/mp3/flac/opus/pcm/aac, `speed`)

```
curl -X POST http://127.0.0.1:8035/v1/audio/speech -H "Content-Type: application/json" ^
     -d "{\"input\": \"Hello there.\", \"voice\": \"christopher\", \"response_format\": \"wav\"}" -o out.wav
```

Server settings are environment variables:

| Variable | Default | |
|---|---|---|
| `WPIPER_VOICES_DIR` | `.\voices` | folder of `name.onnx` + `name.onnx.json` |
| `WPIPER_DEFAULT` | first voice found | default voice name |
| `WPIPER_HOST` | `127.0.0.1` | use `0.0.0.0` to serve your whole network |
| `WPIPER_PORT` | `8035` | |
| `WPIPER_ADMIN_KEY` | unset | protects `/admin/*` and `/audiobook/api/*`; **set this before you use `WPIPER_HOST=0.0.0.0`** |
| `WPIPER_PROVIDERS` | `CPUExecutionProvider` | onnxruntime providers, comma-separated |
| `WPIPER_AUDIOBOOK_DIR` | `.\audiobook_jobs` | audiobook jobs and clip cache |
| `FFMPEG_EXE` | `ffmpeg` | |

### Make an audiobook from the command line

```
python -m wpiper.audiobook book.txt --voice voices\christopher.onnx --out out\mybook --format mp3,m4b --title "My Book"
```

Chapters come from heading lines (`Chapter 1`, `# Title`, `I`, `II`, ...) or from one file per chapter
in a folder. Numbers, years and Roman numerals are spelled out. `--pron words.json` takes a
`{"word": "replacement"}` map for names the voice gets wrong. Run with `--help` to see every option
(pauses, pace, wrapping, headings).

### Train a voice

The easiest way is the trainer web app (`start_trainer_web.bat`, then `http://127.0.0.1:8100`). It
walks you through import, clean-up, prepare, train, preview and export. The same steps on the CLI:

```
python -m wpiper.prepare_big --csv mydata\metadata.csv --out data\myvoice
python -m wpiper.make_split  --data data\myvoice
python -m wpiper.train       --data data\myvoice --out runs\myvoice --init-from christopher-epoch250.ckpt --precision bf16
python -m wpiper.export_onnx --checkpoint runs\myvoice\last.ckpt --config data\myvoice\config.json --output voices\myvoice.onnx
```

`metadata.csv` is `relative\path.wav|transcript` per line. `--init-from` fine-tunes from the
Christopher checkpoint, which needs far less data and time than training from scratch. Leave it out
to train from scratch. Add `--espeak-voice en-au --language en_AU` to `prepare_big` for an
Australian voice.

All the options are in [CLI_COMMANDS.md](CLI_COMMANDS.md). How the Windows port works is in
[docs/WINDOWS_PORT.md](docs/WINDOWS_PORT.md).

## Australian English

Pick **English (Australia)** in the trainer, or pass `--espeak-voice en-au --language en_AU` to
`prepare_big`. The voice's `.onnx.json` then records `"espeak": {"voice": "en-au"}`, and synthesis and
the audiobook maker phonemize with the Australian rules automatically.

The phoneme symbols are the same set Piper already uses, so en-AU datasets can fine-tune from an
English base checkpoint such as Christopher's. `samples_au.txt` has test sentences. The espeak-ng source changes behind
the voice (and how to rebuild the data) are in [espeak-ng-en-au/](espeak-ng-en-au/README.md).

## Other tools

- `python -m wpiper.viewer --data data\myvoice`: a browser view of training progress and samples.
- `scripts\add_wav_ext.py`: fixes metadata files that leave out the `.wav` extension.

## Tests

```
python -m unittest tests.test_audiobook_text -v     # audiobook text pipeline, no model needed
python tests\test_vits.py data\myvoice              # one training step + inference on a prepared dataset
```

## Licence and credits

wpiper is released under the **GNU GPL v3 or later** (see [LICENSE](LICENSE)), because it builds on
GPL-3 code:

- [piper1-gpl](https://github.com/OHF-voice/piper1-gpl) (Open Home Foundation): phoneme ID map,
  model layout and the prebuilt `espeakbridge` module.
- [espeak-ng](https://github.com/espeak-ng/espeak-ng): phonemizer and `espeak-ng-data`, with the
  added en-AU voice (source changes in `espeak-ng-en-au/`).
- [VITS](https://github.com/jaywalnut310/vits) (MIT): the model code in `wpiper/vits/`.

Only use voices and recordings you have the rights to. Don't clone someone's voice without their
consent.
