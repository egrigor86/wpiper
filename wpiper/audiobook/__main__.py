"""Command line: python -m wpiper.audiobook INPUT --voice voices\\christopher.onnx --out out\\book [options]"""

import argparse
import sys
import time
from pathlib import Path

from .config import BookConfig
from .pipeline import build_book


def _progress():
    state = {"t": 0.0}

    def cb(stage: str, done: int, total: int, msg: str) -> None:
        now = time.time()
        if now - state["t"] < 1.0 and done != total:
            return
        state["t"] = now
        print(f"\r[{stage}] {done}/{total} {msg[:50]:<50}", end="", flush=True)
        if done == total:
            print()

    return cb


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m wpiper.audiobook", description=__doc__.split("\n")[0])
    p.add_argument("input", help=".txt/.md file, or a folder of text files (one chapter each)")
    p.add_argument("--voice", help="voice .onnx (single narrator)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--title", default="")
    p.add_argument("--author", default="")
    p.add_argument("--format", default="wav", help="comma list: wav,flac,mp3,opus,m4b (non-wav needs ffmpeg)")
    p.add_argument("--split-chapters", action="store_true", help="also write one file + .srt per chapter")
    p.add_argument("--loudnorm", action="store_true", help="ffmpeg loudness normalisation on encoded files")
    p.add_argument("--bitrate", default="", help="mp3/opus/m4b bitrate, e.g. 64k")
    p.add_argument("--ffmpeg", default="", help="path to ffmpeg (default: $FFMPEG_EXE or PATH)")

    p.add_argument("--max-words", type=int, default=60, help="split sentences longer than this")
    p.add_argument("--wrap", choices=["auto", "hard", "lines"], default="auto",
                   help="single newlines: hard = soft wrap (join), lines = real line breaks")
    p.add_argument("--no-headings", action="store_true", help="do not speak chapter headings")
    p.add_argument("--pron", default="", help="pronunciation JSON {word: replacement}")
    p.add_argument("--caps-headings", choices=["auto", "chapters", "inline", "off"], default="auto",
                   help="ALL-CAPS title lines: auto = chapters only if the book has no other headings")
    p.add_argument("--no-number-headings", action="store_true",
                   help="do not treat lines holding only 1, 2, 3 (or I, II, III) as chapters")
    p.add_argument("--number-label", default="Chapter", help="spoken before a bare chapter number ('' = number only)")
    p.add_argument("--keep-page-numbers", action="store_true", help="read digit-only lines instead of dropping them")

    p.add_argument("--noise-scale", type=float)
    p.add_argument("--length-scale", type=float, help=">1 slower")
    p.add_argument("--noise-w", type=float)
    p.add_argument("--threads", type=int, default=0, help="onnxruntime threads (0 = all cores)")

    p.add_argument("--pause-sentence", type=int, default=350)
    p.add_argument("--pause-paragraph", type=int, default=800)
    p.add_argument("--pause-chapter", type=int, default=2000)
    p.add_argument("--pause-scale", type=float, default=1.0, help="multiply every pause")
    p.add_argument("--srt-offset-ms", type=int, default=0, help="shift subtitles (e.g. for encoder delay)")

    p.add_argument("--limit", type=int, default=0, help="only the first N units (smoke test)")
    p.add_argument("--dry-run", action="store_true", help="split and clean only; write work/units.jsonl")
    args = p.parse_args(argv)

    if not args.dry_run and not args.voice:
        p.error("--voice is required unless --dry-run")

    cfg = BookConfig(
        voice=args.voice or "", noise_scale=args.noise_scale, length_scale=args.length_scale, noise_w=args.noise_w,
        threads=args.threads, max_words=args.max_words, wrap=args.wrap, speak_headings=not args.no_headings,
        pronunciations=args.pron, caps_headings=args.caps_headings, number_headings=not args.no_number_headings,
        number_label=args.number_label, drop_page_numbers=not args.keep_page_numbers, pause_sentence=args.pause_sentence, pause_paragraph=args.pause_paragraph,
        pause_chapter=args.pause_chapter, pause_scale=args.pause_scale, srt_offset_ms=args.srt_offset_ms,
        title=args.title, author=args.author, formats=tuple(f.strip() for f in args.format.split(",") if f.strip()),
        split_chapters=args.split_chapters, loudnorm=args.loudnorm, bitrate=args.bitrate, ffmpeg=args.ffmpeg,
        limit=args.limit,
    )
    summary = build_book(Path(args.input), Path(args.out), cfg, progress=_progress(), dry_run=args.dry_run)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
