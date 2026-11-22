"""Add a .wav extension (and optional folder prefix) to the filename at the start of each metadata row.

Usage:
    python scripts/add_wav_ext.py metadata.csv                 # writes metadata_wav.csv
    python scripts/add_wav_ext.py metadata.csv --prefix wavs/  # wavs/<name>.wav|text
"""

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, help="default: <name>_wav.csv next to the input")
    p.add_argument("--prefix", default="", help="folder to put before the filename, e.g. wavs/")
    args = p.parse_args()

    out = args.out or args.csv.with_name(args.csv.stem + "_wav" + args.csv.suffix)
    n = 0
    with open(args.csv, "r", encoding="utf-8", newline="") as src, \
            open(out, "w", encoding="utf-8", newline="") as dst:
        for line in src:
            body = line.rstrip("\r\n")
            if not body:
                continue
            name, sep, text = body.partition("|")
            if not name.lower().endswith(".wav"):
                name += ".wav"
            dst.write(f"{args.prefix}{name}{sep}{text}\n")
            n += 1
    print(f"{n} rows -> {out}")


if __name__ == "__main__":
    main()
