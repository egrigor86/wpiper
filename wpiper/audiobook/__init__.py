"""Audiobook builder for wpiper voices.

    python -m wpiper.audiobook book.txt --voice voices\\christopher.onnx --out out\\mybook --format mp3

Layout:
    ingest.py      book file/folder -> chapters
    splitter.py    sentence + newline + long-sentence rules
    textclean.py   text cleaning for speech (numbers.py spells numbers)
    plan.py        chapters -> synthesis units
    synth.py       sentence rendering + clip cache (no per-clip peak normalisation)
    assemble.py    trim, pauses, sample-exact timeline, one gain for the book
    srt.py         subtitle cues
    export.py      WAV writer + ffmpeg encodes (mp3, opus, flac, m4b with chapters)
    pipeline.py    build_book(): everything above, with progress/cancel/lock hooks for the server
Dependencies: numpy and onnxruntime (already needed by wpiper) and the ffmpeg executable for non-WAV formats.
"""

from .config import BookConfig, Chapter, Unit
from .pipeline import build_book, plan_book

__all__ = ["BookConfig", "Chapter", "Unit", "build_book", "plan_book"]
