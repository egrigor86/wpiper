"""Tiny local web viewer for training runs, samples and renders.

Usage:
    python -m wpiper.viewer                 # http://127.0.0.1:8765
    python -m wpiper.viewer --port 9000 --data data/christopher

Reads runs/<name>/metrics.jsonl (written by wpiper.train) and falls back to
parsing a training log (runs_<name>.log or runs/<name>/train.log).
"""

import argparse
import io
import json
import mimetypes
import re
import statistics
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("viewer.html")
ACTIVE_SECONDS = 120

TRAIN_RE = re.compile(
    r"(\d\d):(\d\d):(\d\d) epoch (\d+) step (\d+) g=([\d.]+) mel=([\d.]+) kl=([\d.]+) "
    r"dur=([\d.]+) fm=([\d.]+) adv=([\d.]+) d=([\d.]+) \(([\d.]+)s"
)
VAL_RE = re.compile(r"epoch (\d+) val_loss=([\d.]+)")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse_log(path: Path):
    train, val = [], []
    day_offset, last_clock = 0, None
    for line in read_text(path).splitlines():
        m = TRAIN_RE.search(line)
        if m:
            h, mi, s = int(m[1]), int(m[2]), int(m[3])
            clock = h * 3600 + mi * 60 + s
            if last_clock is not None and clock < last_clock - 3600:
                day_offset += 86400
            last_clock = clock
            train.append({
                "epoch": int(m[4]), "step": int(m[5]), "g": float(m[6]), "mel": float(m[7]),
                "kl": float(m[8]), "dur": float(m[9]), "fm": float(m[10]), "adv": float(m[11]),
                "d": float(m[12]), "sec": float(m[13]),
                "clock": clock + day_offset,
            })
            continue
        m = VAL_RE.search(line)
        if m:
            val.append({"epoch": int(m[1]), "loss": float(m[2])})
    return train, val, None


def parse_jsonl(path: Path):
    train, val, max_epochs = [], [], None
    for line in read_text(path).splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "start":
            max_epochs = r.get("max_epochs")
        elif r.get("type") == "train":
            train.append({**r, "clock": r["time"]})
        elif r.get("type") == "val":
            val.append({"epoch": r["epoch"], "loss": r["loss"]})
    return train, val, max_epochs


def find_source(name: str, run_dir: Path):
    jsonl = run_dir / "metrics.jsonl"
    if jsonl.exists():
        return jsonl, parse_jsonl
    for log in (run_dir / "train.log", ROOT / f"runs_{name}.log", ROOT / f"{name}.log"):
        if log.exists():
            return log, parse_log
    return None, None


def run_summary(name: str) -> dict:
    run_dir = ROOT / "runs" / name
    source, parser = find_source(name, run_dir)
    train, val, max_epochs = parser(source) if source else ([], [], None)

    gaps = [b["clock"] - a["clock"] for a, b in zip(train[-21:], train[-20:]) if 0 < b["clock"] - a["clock"] < 600]
    sec_per_epoch = statistics.median(gaps) if gaps else None

    samples = {}
    sample_root = run_dir / "samples"
    if sample_root.exists():
        for d in sample_root.iterdir():
            if d.is_dir() and d.name.startswith("epoch"):
                samples[int(d.name[5:])] = sorted(p.stem for p in d.glob("*.wav"))

    ckpts = [
        {"name": p.name, "mb": round(p.stat().st_size / 2**20), "mtime": p.stat().st_mtime}
        for p in sorted(run_dir.glob("*.ckpt"))
    ]
    touched = [p.stat().st_mtime for p in [source] + [run_dir / "last.ckpt"] if p and p.exists()]
    last_update = max(touched) if touched else None
    return {
        "name": name,
        "active": bool(last_update and time.time() - last_update < ACTIVE_SECONDS),
        "last_update": last_update,
        "epoch": train[-1]["epoch"] if train else 0,
        "max_epochs": max_epochs,
        "sec_per_epoch": sec_per_epoch,
        "train": {k: [t.get(k) for t in train] for k in ("epoch", "step", "g", "mel", "kl", "dur", "fm", "adv", "d")},
        "val": [[v["epoch"], v["loss"]] for v in val],
        "samples": {str(k): v for k, v in sorted(samples.items())},
        "checkpoints": ckpts,
    }


def list_runs() -> list:
    runs_dir = ROOT / "runs"
    names = sorted(p.name for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.exists() else []
    out = []
    for n in names:
        s = run_summary(n)
        out.append({k: s[k] for k in ("name", "active", "epoch", "max_epochs", "last_update")})
    return out


def list_renders() -> list:
    """Folders under voices/ that hold numbered or named wavs (batch renders)."""
    voices = ROOT / "voices"
    texts_file = ROOT / "samples.txt"
    texts = [l.strip() for l in read_text(texts_file).splitlines() if l.strip()] if texts_file.exists() else []
    out = []
    if voices.exists():
        for d in sorted(p for p in voices.iterdir() if p.is_dir()):
            files = sorted(p.name for p in d.glob("*.wav"))
            if not files:
                continue
            items = []
            for f in files:
                m = re.fullmatch(r"(\d+)\.wav", f)
                text = texts[int(m[1]) - 1] if m and 0 < int(m[1]) <= len(texts) else Path(f).stem
                items.append({"file": f, "text": text})
            out.append({"name": d.name, "items": items, "mtime": d.stat().st_mtime})
    return out


class Reference:
    """Held-out test clips (text + original audio) from the dataset cache."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._cache = None

    def load(self):
        if self._cache is None:
            self._cache = {}
            split = self.data_dir / "split.json"
            if split.exists():
                import torch

                ids = json.loads(split.read_text(encoding="utf-8"))["test"]
                config = json.loads((self.data_dir / "config.json").read_text(encoding="utf-8"))
                for i in ids:
                    d = torch.load(self.data_dir / "cache" / f"{i}.pt", weights_only=True)
                    self._cache[i] = {"text": d["text"], "audio": d["audio"].numpy(), "sr": config["audio"]["sample_rate"]}
        return self._cache

    def texts(self) -> dict:
        return {k: v["text"] for k, v in self.load().items()}

    def wav_bytes(self, utt_id: str):
        import soundfile as sf

        item = self.load().get(utt_id)
        if item is None:
            return None
        buf = io.BytesIO()
        sf.write(buf, item["audio"], item["sr"], format="WAV")
        return buf.getvalue()


def make_handler(reference: Reference):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, status, body: bytes, ctype: str, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj):
            self._send(200, json.dumps(obj).encode("utf-8"), "application/json")

        def _file(self, path: Path, base: Path):
            try:
                path = path.resolve()
                path.relative_to(base.resolve())
            except (OSError, ValueError):
                return self._send(404, b"not found", "text/plain")
            if not path.is_file():
                return self._send(404, b"not found", "text/plain")
            data = path.read_bytes()
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._bytes_with_range(data, ctype)

        def _bytes_with_range(self, data: bytes, ctype: str):
            rng = self.headers.get("Range")
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng or "")
            if m and (m[1] or m[2]):
                start = int(m[1]) if m[1] else max(0, len(data) - int(m[2]))
                end = int(m[2]) if m[1] and m[2] else len(data) - 1
                end = min(end, len(data) - 1)
                return self._send(
                    206, data[start : end + 1], ctype,
                    {"Content-Range": f"bytes {start}-{end}/{len(data)}", "Accept-Ranges": "bytes"},
                )
            self._send(200, data, ctype, {"Accept-Ranges": "bytes"})

        def do_GET(self):
            parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
            try:
                if not parts:
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if parts[0] == "api":
                    if parts[1:] == ["runs"]:
                        return self._json(list_runs())
                    if len(parts) == 3 and parts[1] == "run":
                        return self._json(run_summary(parts[2]))
                    if parts[1:] == ["renders"]:
                        return self._json(list_renders())
                    if parts[1:] == ["reference"]:
                        return self._json(reference.texts())
                if parts[0] == "samples" and len(parts) == 4:  # /samples/<run>/<epoch>/<utt>.wav
                    return self._file(ROOT / "runs" / parts[1] / "samples" / f"epoch{parts[2]}" / parts[3], ROOT / "runs")
                if parts[0] == "renders" and len(parts) == 3:
                    return self._file(ROOT / "voices" / parts[1] / parts[2], ROOT / "voices")
                if parts[0] == "reference" and len(parts) == 2:
                    data = reference.wav_bytes(Path(parts[1]).stem)
                    if data is not None:
                        return self._bytes_with_range(data, "audio/wav")
                self._send(404, b"not found", "text/plain")
            except Exception as exc:  # surface errors to the page instead of dropping the connection
                self._send(500, str(exc).encode("utf-8"), "text/plain")

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "myvoice", help="dataset dir for reference clips")
    args = p.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Reference(args.data)))
    print(f"viewer on http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
