"""Filter a prepared large dataset and write split.json (train/val/test clip indices).

Filters drop clips whose characters-per-second falls outside a percentile band
(usually bad synthesis: truncated or looping audio). Everything is deterministic
for a given --seed, and the settings are stored in split.json.

Usage:
    python -m wpiper.make_split --data data/myvoice
    python -m wpiper.make_split --data data/myvoice --cps-pct 0.5 99.5 --val 200 --test 10
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .bigdata import BigData


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--cps-pct", nargs=2, type=float, default=[0.3, 99.7], metavar=("LOW", "HIGH"),
                   help="keep clips whose chars/second is inside these percentiles; 0 100 disables")
    p.add_argument("--val", type=int, default=20)
    p.add_argument("--test", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    data = BigData(args.data)
    meta = data.meta()
    seconds = data.audio_len / data.sample_rate
    chars = np.array([len(t) for _, t in meta])
    cps = chars / seconds

    lo, hi = np.percentile(cps, args.cps_pct)
    keep = (cps >= lo) & (cps <= hi)
    dropped = np.where(~keep)[0]
    with open(args.data / "filtered.tsv", "w", encoding="utf-8") as f:
        for i in dropped:
            f.write(f"{i}\t{cps[i]:.1f}\t{seconds[i]:.1f}\t{meta[i][1]}\n")

    ids = [int(i) for i in np.where(keep)[0]]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    test = sorted(ids[: args.test])
    val = sorted(ids[args.test : args.test + args.val])
    train = sorted(ids[args.test + args.val :])

    split = {
        "train": train, "val": val, "test": test,
        "settings": {"cps_percentiles": args.cps_pct, "cps_range": [round(float(lo), 2), round(float(hi), 2)],
                     "seed": args.seed, "dropped": int(len(dropped))},
    }
    (args.data / "split.json").write_text(json.dumps(split), encoding="utf-8")

    hours = lambda idx: seconds[idx].sum() / 3600  # noqa: E731
    print(f"chars/sec band [{lo:.1f}, {hi:.1f}] kept {keep.sum()} of {len(keep)}; details in {args.data / 'filtered.tsv'}")
    print(f"train {len(train)} clips {hours(train):.2f} h | val {len(val)} {hours(val):.2f} h | test {len(test)} {hours(test):.3f} h")


if __name__ == "__main__":
    main()
