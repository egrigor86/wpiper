"""wpiper trainer web app: import -> clean -> prepare -> train -> monitor -> manage runs.

Run:
    python trainer_server.py            # http://127.0.0.1:8100
    python trainer_server.py --host 0.0.0.0 --port 8100

Standalone: it only shells out to the existing wpiper CLI modules (prepare_dataset, prepare_big, make_split,
train, export_onnx) and reads their output folders; nothing in wpiper/ is modified. Every long step is a
subprocess logging to trainer_state/jobs/<id>.log, so jobs survive a page reload or a server restart.
There is no authentication and it can read/delete local folders, so keep it on localhost.
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import psutil
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

ROOT = Path(__file__).parent.resolve()
DATA, RUNS, VOICES = ROOT / "data", ROOT / "runs", ROOT / "voices"
STATE = ROOT / "trainer_state"
DRAFTS, JOBS_DIR, JOBS_FILE = STATE / "drafts", STATE / "jobs", STATE / "jobs.json"
PAGE = ROOT / "trainer.html"

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
MARKUP = re.compile(r"\{\{|\}\}|\[\[|\]\]|\||<[^>]+>|https?://|www\.")  # same filter prepare_big applies
ODD_CHARS = re.compile(r"[^\w\s.,;:!?'\"()\-–—…‘’“”/&%$°+=]")
BLOCKING = {"missing", "empty_text", "markup", "duplicate"}
WIN = os.name == "nt"
FLAGS = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)) if WIN else 0

app = FastAPI(title="wpiper trainer")


def nm(name: str, what: str = "name") -> str:
    if not NAME_RE.match(name or ""):
        raise HTTPException(400, f"Invalid {what}: use letters, digits, _ . - (max 80).")
    return name


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def dir_size_mb(path: Path) -> float:
    total = 0
    for dp, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return round(total / 1e6, 1)


def hist(values, bins=30):
    if len(values) == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [round(float(e), 3) for e in edges], "counts": [int(c) for c in counts]}


# ============================ jobs ============================
jobs: Dict[str, dict] = {}
jobs_lock = threading.RLock()
Step = Union[List[str], Callable[[], None]]


def save_jobs():
    with jobs_lock:
        keep = sorted(jobs.values(), key=lambda j: j["started"], reverse=True)[:100]
        try:
            STATE.mkdir(parents=True, exist_ok=True)
            JOBS_FILE.write_text(json.dumps(keep, indent=1), encoding="utf-8")
        except OSError:
            pass


def proc_alive(pid, ctime) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running() and abs(p.create_time() - (ctime or 0)) < 2 and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def kill_tree(pid):
    try:
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            try:
                c.kill()
            except psutil.Error:
                pass
        p.kill()
    except psutil.Error:
        pass


def finish(job, status, rc=None):
    with jobs_lock:
        job.update(status=status, ended=time.time(), returncode=rc, pid=None)
    save_jobs()


def run_job(job, steps: List[Step]):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    rc = 0
    try:
        with open(job["log"], "ab") as lf:
            for i, step in enumerate(steps):
                if job.get("stop"):
                    break
                job["step"] = i
                if callable(step):
                    try:
                        step()
                    except Exception as e:
                        lf.write(f"\nERROR: {e}\n".encode())
                        rc = 1
                        break
                    continue
                lf.write(f"\n$ python {' '.join(step)}\n".encode())
                lf.flush()
                p = subprocess.Popen([sys.executable, "-u", *step], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                     env=env, creationflags=FLAGS)
                try:
                    job.update(pid=p.pid, ctime=psutil.Process(p.pid).create_time())
                except psutil.Error:
                    pass
                save_jobs()
                rc = p.wait()
                if rc != 0:
                    break
    except Exception as e:  # e.g. could not open the log
        rc = 1
        job["error"] = str(e)
    finish(job, "stopped" if job.get("stop") else ("done" if rc == 0 else "failed"), rc)


def watch_job(job):
    """Re-attach to a job that was running when the server restarted: we only know when it ends."""
    while proc_alive(job.get("pid"), job.get("ctime")):
        time.sleep(2)
    finish(job, "stopped" if job.get("stop") else "ended")


def start_job(kind: str, label: str, steps: List[Step], meta: Optional[dict] = None) -> dict:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    job = {"id": jid, "kind": kind, "label": label, "status": "running", "started": time.time(), "ended": None,
           "returncode": None, "pid": None, "ctime": None, "log": str(JOBS_DIR / f"{jid}.log"),
           "meta": meta or {}, "step": 0, "steps": len(steps), "stop": False}
    with jobs_lock:
        jobs[jid] = job
    save_jobs()
    threading.Thread(target=run_job, args=(job, steps), daemon=True).start()
    return job


def load_jobs():
    for j in read_json(JOBS_FILE, []):
        if j.get("status") == "running":
            if proc_alive(j.get("pid"), j.get("ctime")):
                threading.Thread(target=watch_job, args=(j,), daemon=True).start()
            else:
                j.update(status="ended", ended=j.get("ended") or time.time(), pid=None)
        jobs[j["id"]] = j


def running_jobs(kind=None, **meta):
    with jobs_lock:
        return [j for j in jobs.values() if j["status"] == "running" and (kind is None or j["kind"] == kind)
                and all(j["meta"].get(k) == v for k, v in meta.items())]


def public(job):
    return {k: v for k, v in job.items() if k not in ("ctime",)}


@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        return [public(j) for j in sorted(jobs.values(), key=lambda j: j["started"], reverse=True)]


@app.get("/api/jobs/{jid}/log")
def job_log(jid: str, since: int = 0):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Unknown job.")
    try:
        with open(job["log"], "rb") as f:
            size = f.seek(0, 2)
            if since > size:
                since = 0
            start = max(since, size - 200_000) if since == 0 else since
            f.seek(start)
            data = f.read()
    except OSError:
        return {"text": "", "next": 0, "job": public(job)}
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return {"text": text, "next": start + len(data), "job": public(job)}


@app.post("/api/jobs/{jid}/stop")
def stop_job(jid: str):
    job = jobs.get(jid)
    if not job or job["status"] != "running":
        raise HTTPException(404, "No such running job.")
    job["stop"] = True
    if job.get("pid"):
        kill_tree(job["pid"])
    return {"stopping": jid}


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Unknown job.")
    if job["status"] == "running":
        raise HTTPException(409, "Stop the job first.")
    with jobs_lock:
        jobs.pop(jid, None)
    Path(job["log"]).unlink(missing_ok=True)
    save_jobs()
    return {"deleted": jid}


# ============================ system ============================
@app.get("/api/system")
def system():
    gpu = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                             creationflags=FLAGS).stdout.strip().splitlines()[0].split(", ")
        gpu = {"name": out[0], "mem_used_mb": int(out[1]), "mem_total_mb": int(out[2]),
               "util": int(out[3]), "temp": int(out[4])}
    except Exception:
        pass
    du = shutil.disk_usage(ROOT)
    return {"gpu": gpu, "disk_free_gb": round(du.free / 1e9, 1), "cpus": os.cpu_count(),
            "python": sys.executable, "training": [public(j) for j in running_jobs("train")]}


# ============================ datasets ============================
# Mirrors wpiper.hparams (not imported: this server may run without torch).
QUALITIES = [
    {"id": "x-low", "label": "x-low (16 kHz, ~7M params, smallest and fastest)", "sample_rate": 16000},
    {"id": "low", "label": "low (16 kHz, ~24M params)", "sample_rate": 16000},
    {"id": "medium", "label": "medium (22.05 kHz, ~24M params)", "sample_rate": 22050},
    {"id": "high", "label": "high (22.05 kHz, ~36M params, slowest)", "sample_rate": 22050},
]
QUALITY_IDS = {q["id"] for q in QUALITIES}
# (espeak voice, language tag, label, espeak dictionary file it needs)
LANGUAGES = [
    ("en-us", "en_US", "English (US)", "en"), ("en-gb", "en_GB", "English (UK)", "en"), ("en-au", "en_AU", "English (Australia)", "en"),
    ("de", "de_DE", "German", "de"), ("fr-fr", "fr_FR", "French", "fr"), ("es", "es_ES", "Spanish (Spain)", "es"),
    ("es-419", "es_MX", "Spanish (Latin America)", "es"), ("it", "it_IT", "Italian", "it"),
    ("pt", "pt_PT", "Portuguese (Portugal)", "pt"), ("pt-br", "pt_BR", "Portuguese (Brazil)", "pt"),
    ("nl", "nl_NL", "Dutch", "nl"), ("pl", "pl_PL", "Polish", "pl"), ("ru", "ru_RU", "Russian", "ru"),
    ("uk", "uk_UA", "Ukrainian", "uk"), ("cs", "cs_CZ", "Czech", "cs"), ("sk", "sk_SK", "Slovak", "sk"),
    ("sv", "sv_SE", "Swedish", "sv"), ("da", "da_DK", "Danish", "da"), ("nb", "nb_NO", "Norwegian", "no"),
    ("fi", "fi_FI", "Finnish", "fi"), ("hu", "hu_HU", "Hungarian", "hu"), ("ro", "ro_RO", "Romanian", "ro"),
    ("el", "el_GR", "Greek", "el"), ("tr", "tr_TR", "Turkish", "tr"), ("ca", "ca_ES", "Catalan", "ca"),
    ("bg", "bg_BG", "Bulgarian", "bg"), ("hr", "hr_HR", "Croatian", "hr"), ("sr", "sr_RS", "Serbian", "sr"),
    ("sl", "sl_SI", "Slovenian", "sl"), ("lt", "lt_LT", "Lithuanian", "lt"), ("lv", "lv_LV", "Latvian", "lv"),
    ("ar", "ar_JO", "Arabic", "ar"), ("fa", "fa_IR", "Persian", "fa"), ("he", "he_IL", "Hebrew", "he"),
    ("hi", "hi_IN", "Hindi", "hi"), ("vi", "vi_VN", "Vietnamese", "vi"), ("id", "id_ID", "Indonesian", "id"),
    ("cmn", "zh_CN", "Chinese (Mandarin)", "cmn"), ("ja", "ja_JP", "Japanese", "ja"), ("ko", "ko_KR", "Korean", "ko"),
    ("sw", "sw_KE", "Swahili", "sw"), ("cy", "cy_GB", "Welsh", "cy"), ("is", "is_IS", "Icelandic", "is"),
]


@app.get("/api/options")
def options():
    """Model sizes and phonemizer languages for the dropdowns; languages without data in espeak-ng-data are flagged."""
    data = ROOT / "wpiper" / "espeak-ng-data"
    langs = [{"espeak": v, "code": c, "label": lbl, "available": (data / f"{dic}_dict").exists()}
             for v, c, lbl, dic in LANGUAGES]
    return {"qualities": QUALITIES, "languages": langs}


def dataset_kind(d: Path) -> str:
    if (d / "index.npz").exists():
        return "big"
    if (d / "cache").is_dir():
        return "small"
    return "unknown"


def dataset_info(d: Path) -> dict:
    kind = dataset_kind(d)
    cfg = read_json(d / "config.json", {}) or {}
    info = {"name": d.name, "kind": kind, "has_split": (d / "split.json").exists(),
            "sample_rate": (cfg.get("audio") or {}).get("sample_rate"),
            "espeak": (cfg.get("espeak") or {}).get("voice"), "size_mb": dir_size_mb(d),
            "quality": (cfg.get("audio") or {}).get("quality"), "language": (cfg.get("language") or {}).get("code"),
            "clips": None, "hours": None, "created": None, "train": None, "val": None, "test": None}
    if kind == "big":
        man = read_json(d / "manifest.json", {}) or {}
        info.update(clips=man.get("clips_kept"), hours=man.get("hours_kept"), created=man.get("created"))
    elif kind == "small":
        info["clips"] = sum(1 for _ in (d / "cache").glob("*.pt"))
    split = read_json(d / "split.json") if info["has_split"] else None
    if split:
        for k in ("train", "val", "test"):
            info[k] = len(split.get(k, []))
    return info


@app.get("/api/datasets")
def list_datasets():
    if not DATA.exists():
        return []
    return [dataset_info(d) for d in sorted(DATA.iterdir()) if d.is_dir() and not d.name.endswith(".partial")]


def dset(name: str) -> Path:
    d = DATA / nm(name)
    if not d.is_dir():
        raise HTTPException(404, f"Dataset '{name}' not found.")
    return d


def _stats_big(d: Path) -> dict:
    idx = np.load(d / "index.npz")
    cfg = read_json(d / "config.json", {}) or {}
    sr = (cfg.get("audio") or {}).get("sample_rate", 22050)
    secs = idx["audio_len"] / sr
    texts = []
    with open(d / "meta.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            texts.append(parts[2] if len(parts) > 2 else "")
    chars = np.array([len(t) for t in texts])
    cps = chars / np.maximum(secs, 1e-6)
    split = read_json(d / "split.json") if (d / "split.json").exists() else None
    out = {"kind": "big", "clips": len(secs), "hours": round(float(secs.sum() / 3600), 3), "sample_rate": sr,
           "secs": {"mean": round(float(secs.mean()), 2), "median": round(float(np.median(secs)), 2),
                    "min": round(float(secs.min()), 2), "max": round(float(secs.max()), 2)},
           "chars_mean": round(float(chars.mean()), 1), "cps_mean": round(float(cps.mean()), 1),
           "hist_secs": hist(secs), "hist_cps": hist(cps), "hist_phonemes": hist(idx["ph_len"])}
    if split:
        sets = {k: np.array(split.get(k, []), dtype=int) for k in ("train", "val", "test")}
        out["split"] = {k: {"clips": int(len(v)), "hours": round(float(secs[v].sum() / 3600), 3)} for k, v in sets.items()}
        out["split_settings"] = split.get("settings")
    man = read_json(d / "manifest.json", {}) or {}
    out["manifest"] = {k: man.get(k) for k in ("created", "rows_in_csv", "rejected_by_reason", "phonemes_not_in_map", "seconds")}
    out["manifest"]["args"] = man.get("args")

    def pick(order):
        return [{"i": int(i), "secs": round(float(secs[i]), 2), "cps": round(float(cps[i]), 1), "text": texts[i][:160]}
                for i in order]
    out["outliers"] = {"shortest": pick(np.argsort(secs)[:8]), "longest": pick(np.argsort(-secs)[:8]),
                       "highest_cps": pick(np.argsort(-cps)[:8]), "lowest_cps": pick(np.argsort(cps)[:8])}
    for key, fname in (("rejected", "rejected.tsv"), ("filtered", "filtered.tsv")):
        rows = []
        p = d / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                rows = [line.rstrip("\n").split("\t") for line in f]
        out[key] = {"count": len(rows), "sample": rows[:30]}
    return out


def _stats_small(d: Path) -> dict:
    import torch  # lazy: the server stays light unless a small dataset is inspected

    cfg = read_json(d / "config.json", {}) or {}
    sr = (cfg.get("audio") or {}).get("sample_rate", 22050)
    files = sorted((d / "cache").glob("*.pt"))
    secs, chars, ph = [], [], []
    for f in files[:3000]:
        item = torch.load(f, map_location="cpu", weights_only=False)
        secs.append(len(item["audio"]) / sr)
        chars.append(len(item["text"]))
        ph.append(len(item["phoneme_ids"]))
    secs, chars = np.array(secs), np.array(chars)
    split = read_json(d / "split.json") if (d / "split.json").exists() else None
    cps = chars / np.maximum(secs, 1e-6)
    return {"kind": "small", "clips": len(files), "sampled": len(secs), "hours": round(float(secs.sum() / 3600), 3),
            "sample_rate": sr,
            "secs": {"mean": round(float(secs.mean()), 2), "median": round(float(np.median(secs)), 2),
                     "min": round(float(secs.min()), 2), "max": round(float(secs.max()), 2)},
            "chars_mean": round(float(chars.mean()), 1), "cps_mean": round(float(cps.mean()), 1),
            "hist_secs": hist(secs), "hist_cps": hist(cps), "hist_phonemes": hist(ph),
            "split": {k: {"clips": len(split.get(k, [])), "hours": None} for k in ("train", "val", "test")} if split else None,
            "outliers": None, "rejected": {"count": 0, "sample": []}, "filtered": {"count": 0, "sample": []},
            "manifest": {}}


_stats_cache: Dict[str, tuple] = {}


@app.get("/api/datasets/{name}/stats")
def dataset_stats(name: str):
    d = dset(name)
    stamp = max((p.stat().st_mtime for p in d.iterdir() if p.is_file()), default=0)
    if name in _stats_cache and _stats_cache[name][0] == stamp:
        return _stats_cache[name][1]
    kind = dataset_kind(d)
    if kind == "big":
        stats = _stats_big(d)
    elif kind == "small":
        stats = _stats_small(d)
    else:
        raise HTTPException(400, "Not a prepared dataset (no index.npz or cache/).")
    stats["name"] = name
    _stats_cache[name] = (stamp, stats)
    return stats


@app.get("/api/datasets/{name}/clip/{i}")
def dataset_clip(name: str, i: int):
    d = dset(name)
    if dataset_kind(d) != "big":
        raise HTTPException(400, "Clip playback is only available for big-format datasets.")
    idx = np.load(d / "index.npz")
    if not 0 <= i < len(idx["audio_len"]):
        raise HTTPException(404, "No such clip.")
    sr = (read_json(d / "config.json", {}) or {}).get("audio", {}).get("sample_rate", 22050)
    pcm = np.memmap(d / "audio.bin", dtype=np.int16, mode="r")[idx["audio_off"][i]: idx["audio_off"][i] + idx["audio_len"][i]]
    buf = io.BytesIO()
    sf.write(buf, np.array(pcm), sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.post("/api/datasets/{name}/split")
def make_split(name: str, body: dict):
    d = dset(name)
    if dataset_kind(d) != "big":
        raise HTTPException(400, "make_split applies to big-format datasets only.")
    lo, hi = float(body.get("cps_low", 0.3)), float(body.get("cps_high", 99.7))
    args = ["-m", "wpiper.make_split", "--data", str(d), "--cps-pct", str(lo), str(hi),
            "--val", str(int(body.get("val", 20))), "--test", str(int(body.get("test", 10))),
            "--seed", str(int(body.get("seed", 1234)))]
    return public(start_job("split", f"make_split {name}", [args], {"dataset": name}))


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    d = dset(name)
    if running_jobs("train", dataset=name) or running_jobs("prepare", out=name):
        raise HTTPException(409, "A running job is using this dataset.")
    shutil.rmtree(d, ignore_errors=True)
    _stats_cache.pop(name, None)
    if d.exists():
        raise HTTPException(500, "Could not fully delete the folder (files in use?).")
    return {"deleted": name}


# ============================ drafts (import + cleanup) ============================
_dur_cache: Dict[str, tuple] = {}
_analysis: Dict[str, tuple] = {}
_draft_mem: Dict[str, dict] = {}
_draft_lock = threading.RLock()


def draft_path(name: str) -> Path:
    return DRAFTS / nm(name) / "draft.json"


def get_draft(name: str) -> dict:
    with _draft_lock:
        if name not in _draft_mem:
            d = read_json(draft_path(name))
            if d is None:
                raise HTTPException(404, f"Draft '{name}' not found.")
            d["rev"] = 0
            _draft_mem[name] = d
        return _draft_mem[name]


def save_draft(d: dict):
    d["rev"] += 1
    p = draft_path(d["name"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: v for k, v in d.items() if k != "rev"}, ensure_ascii=False), encoding="utf-8")


def clip_seconds(path: Path):
    key = str(path)
    try:
        mt = path.stat().st_mtime
    except OSError:
        return None
    hit = _dur_cache.get(key)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        i = sf.info(str(path))
        dur = i.frames / i.samplerate
    except Exception:
        dur = -1.0  # exists but unreadable
    _dur_cache[key] = (mt, dur)
    return dur


def analyze(d: dict) -> dict:
    cached = _analysis.get(d["name"])
    if cached and cached[0] == d["rev"]:
        return cached[1]
    adir = Path(d["audio_dir"])
    rows = d["rows"]
    with ThreadPoolExecutor(16) as ex:
        durs = list(ex.map(lambda r: clip_seconds(adir / r["src"]), rows))
    seen, issues = set(), []
    for r, dur in zip(rows, durs):
        iss = []
        if dur is None:
            iss.append("missing")
        elif dur < 0:
            iss.append("unreadable")
        elif dur < 0.5:
            iss.append("too_short")
        elif dur > 30:
            iss.append("too_long")
        text = r["text"]
        if not text.strip():
            iss.append("empty_text")
        else:
            if MARKUP.search(text):
                iss.append("markup")
            if ODD_CHARS.search(text):
                iss.append("odd_chars")
            if dur and dur > 0.5 and not (5 <= len(text) / dur <= 25):
                iss.append("cps_odd")
        if r["src"] in seen:
            iss.append("duplicate")
        seen.add(r["src"])
        issues.append(iss)
    result = {"dur": durs, "issues": issues}
    _analysis[d["name"]] = (d["rev"], result)
    return result


def draft_summary(d: dict) -> dict:
    a = analyze(d)
    counts: Dict[str, int] = {}
    for iss in a["issues"]:
        for k in iss:
            counts[k] = counts.get(k, 0) + 1
    good = [x for x in a["dur"] if x and x > 0]
    return {"name": d["name"], "audio_dir": d["audio_dir"], "rows": len(d["rows"]), "issues": counts,
            "hours_found": round(sum(good) / 3600, 3), "created": d.get("created"), "source": d.get("source")}


@app.get("/api/drafts")
def list_drafts():
    if not DRAFTS.exists():
        return []
    out = []
    for p in sorted(DRAFTS.iterdir()):
        d = read_json(p / "draft.json")
        if d:
            out.append({"name": d["name"], "rows": len(d["rows"]), "audio_dir": d["audio_dir"],
                        "created": d.get("created"), "source": d.get("source")})
    return out


def parse_metadata(text: str, has_header: bool) -> List[dict]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if has_header and lines:
        lines = lines[1:]
    delim = "|" if any("|" in ln for ln in lines[:50]) else ("\t" if any("\t" in ln for ln in lines[:50]) else ",")
    rows = []
    for ln in lines:
        parts = ln.split(delim)
        if len(parts) < 2:
            continue
        # LJSpeech has id|raw|normalized; like prepare_dataset, use the last column as the text
        text_col = parts[-1] if delim == "|" else delim.join(parts[1:])
        rows.append({"src": parts[0].strip().replace("\\", "/"), "text": " ".join(text_col.split())})
    return rows


@app.post("/api/drafts")
def create_draft(body: dict):
    name = nm(body.get("name", ""), "draft name")
    if draft_path(name).exists() and not body.get("overwrite"):
        raise HTTPException(409, f"Draft '{name}' already exists.")
    mode = body.get("mode")
    report: dict = {}
    if mode == "folder":
        adir = Path(body.get("audio_dir", ""))
        if not adir.is_dir():
            raise HTTPException(400, f"Folder not found: {adir}")
        files = adir.rglob("*") if body.get("recursive") else adir.glob("*")
        wavs = sorted(p for p in files if p.suffix.lower() == ".wav")
        rows, no_txt = [], []
        for w in wavs:
            txt = next((t for t in (w.with_suffix(".txt"), w.with_suffix(".TXT")) if t.exists()), None)
            if not txt:
                no_txt.append(w.name)
                continue
            text = " ".join(txt.read_text(encoding="utf-8-sig", errors="replace").split())
            rows.append({"src": w.relative_to(adir).as_posix(), "text": text})
        wav_stems = {w.with_suffix("").relative_to(adir).as_posix().lower() for w in wavs}
        txts = adir.rglob("*.txt") if body.get("recursive") else adir.glob("*.txt")
        no_wav = [t.name for t in txts if t.with_suffix("").relative_to(adir).as_posix().lower() not in wav_stems]
        report = {"wavs_without_txt": len(no_txt), "txt_without_wav": len(no_wav),
                  "examples": (no_txt[:5] + no_wav[:5])}
    elif mode == "metadata":
        text = body.get("metadata_text") or ""
        mpath = body.get("metadata_path")
        if not text and mpath:
            p = Path(mpath)
            if not p.is_file():
                raise HTTPException(400, f"Metadata file not found: {p}")
            text = p.read_text(encoding="utf-8-sig", errors="replace")
        if not text.strip():
            raise HTTPException(400, "Provide metadata text, an uploaded file, or a metadata path.")
        adir = Path(body.get("audio_dir") or (Path(mpath).parent if mpath else ""))
        if not adir.is_dir():
            raise HTTPException(400, f"Audio folder not found: {adir}")
        rows = parse_metadata(text, bool(body.get("has_header")))
    else:
        raise HTTPException(400, "mode must be 'folder' or 'metadata'.")
    if not rows:
        raise HTTPException(400, "No rows found. Check the delimiter (| tab or comma) / that .txt files match the .wav names.")
    d = {"name": name, "audio_dir": str(adir.resolve()), "rows": rows, "source": mode, "created": time.strftime("%Y-%m-%d %H:%M")}
    d["rev"] = 0
    with _draft_lock:
        _draft_mem[name] = d
    save_draft(d)
    return {**draft_summary(d), "report": report}


@app.get("/api/drafts/{name}")
def draft_rows(name: str, offset: int = 0, limit: int = 50, filter: str = "all", q: str = ""):
    d = get_draft(name)
    a = analyze(d)
    sel = []
    ql = q.lower()
    for i, (r, iss) in enumerate(zip(d["rows"], a["issues"])):
        if filter == "problems" and not (BLOCKING & set(iss) or {"too_short", "too_long", "unreadable"} & set(iss)):
            continue
        if filter not in ("all", "problems") and filter not in iss:
            continue
        if ql and ql not in r["src"].lower() and ql not in r["text"].lower():
            continue
        sel.append(i)
    page = sel[offset: offset + max(1, min(limit, 200))]
    return {"summary": draft_summary(d), "total": len(sel), "offset": offset,
            "rows": [{"i": i, **d["rows"][i], "dur": a["dur"][i], "issues": a["issues"][i]} for i in page]}


@app.put("/api/drafts/{name}/rows/{i}")
def edit_row(name: str, i: int, body: dict):
    d = get_draft(name)
    if not 0 <= i < len(d["rows"]):
        raise HTTPException(404, "No such row.")
    if "src" in body:
        d["rows"][i]["src"] = str(body["src"]).strip().replace("\\", "/")
    if "text" in body:
        d["rows"][i]["text"] = " ".join(str(body["text"]).split())
    save_draft(d)
    return draft_summary(d)


@app.delete("/api/drafts/{name}/rows/{i}")
def delete_row(name: str, i: int):
    d = get_draft(name)
    if not 0 <= i < len(d["rows"]):
        raise HTTPException(404, "No such row.")
    d["rows"].pop(i)
    save_draft(d)
    return draft_summary(d)


@app.get("/api/drafts/{name}/audio/{i}")
def draft_audio(name: str, i: int):
    d = get_draft(name)
    if not 0 <= i < len(d["rows"]):
        raise HTTPException(404, "No such row.")
    p = Path(d["audio_dir"]) / d["rows"][i]["src"]
    if not p.is_file() or p.suffix.lower() not in (".wav", ".flac"):
        raise HTTPException(404, "Audio file not found.")
    return FileResponse(p)


@app.post("/api/drafts/{name}/fix")
def fix_draft(name: str, body: dict):
    d = get_draft(name)
    adir = Path(d["audio_dir"])
    action, changed = body.get("action"), 0
    rows = d["rows"]
    exists = lambda s: (adir / s).is_file()  # noqa: E731
    if action == "set_audio_dir":
        p = Path(body.get("audio_dir", ""))
        if not p.is_dir():
            raise HTTPException(400, f"Folder not found: {p}")
        d["audio_dir"] = str(p.resolve())
    elif action == "auto_paths":  # try the common mistakes for each row whose file is not found
        for r in rows:
            if exists(r["src"]):
                continue
            base = r["src"].rsplit("/", 1)[-1]
            for cand in (r["src"] + ".wav", base, base + ".wav", "wavs/" + base, "wavs/" + base + ".wav"):
                if exists(cand):
                    r["src"], changed = cand, changed + 1
                    break
    elif action == "add_ext":  # what scripts/add_wav_ext.py does: .wav on names without an extension
        for r in rows:
            if not re.search(r"\.[A-Za-z0-9]{2,4}$", r["src"]):
                r["src"], changed = r["src"] + ".wav", changed + 1
    elif action == "strip_dir":
        for r in rows:
            if "/" in r["src"]:
                r["src"], changed = r["src"].rsplit("/", 1)[-1], changed + 1
    elif action == "add_prefix":
        prefix = str(body.get("prefix", ""))
        for r in rows:
            if prefix and not r["src"].startswith(prefix):
                r["src"], changed = prefix + r["src"], changed + 1
    elif action == "remove_prefix":
        prefix = str(body.get("prefix", ""))
        for r in rows:
            if prefix and r["src"].startswith(prefix):
                r["src"], changed = r["src"][len(prefix):], changed + 1
    elif action == "fix_quotes":
        table = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
        for r in rows:
            new = r["text"].translate(table).replace("…", "...")
            if new != r["text"]:
                r["text"], changed = new, changed + 1
    elif action == "drop":
        kinds = set(body.get("kinds") or [])
        if not kinds:
            raise HTTPException(400, "Give the issue kinds to drop.")
        a = analyze(d)
        keep = [r for r, iss in zip(rows, a["issues"]) if not kinds & set(iss)]
        changed = len(rows) - len(keep)
        d["rows"] = keep
    else:
        raise HTTPException(400, f"Unknown action '{action}'.")
    save_draft(d)
    return {"changed": changed, "summary": draft_summary(d)}


@app.delete("/api/drafts/{name}")
def delete_draft(name: str):
    nm(name)
    _draft_mem.pop(name, None)
    _analysis.pop(name, None)
    shutil.rmtree(DRAFTS / name, ignore_errors=True)
    return {"deleted": name}


@app.post("/api/drafts/{name}/prepare")
def prepare(name: str, body: dict):
    d = get_draft(name)
    out_name = nm(body.get("out_name", ""), "dataset name")
    out = DATA / out_name
    if (out.exists() or running_jobs("prepare", out=out_name)) and not body.get("overwrite"):
        raise HTTPException(409, f"Dataset '{out_name}' already exists.")
    mode = body.get("mode", "big")
    f = lambda k, default, cast=float: cast(body.get(k, default))  # noqa: E731
    min_sec, max_sec = f("min_sec", 0.5), f("max_sec", 30.0)

    a = analyze(d)
    kept = []
    for r, dur, iss in zip(d["rows"], a["dur"], a["issues"]):
        if BLOCKING & set(iss) or (dur is None or dur < 0) or not (min_sec <= dur <= max_sec):
            continue
        kept.append(r)
    if len(kept) < 4:
        raise HTTPException(400, f"Only {len(kept)} usable rows after dropping problem rows; fix the draft first.")
    csv_path = DRAFTS / name / "prepare_metadata.csv"
    csv_path.write_text("".join(f"{r['src']}|{r['text']}\n" for r in kept), encoding="utf-8")

    espeak, sr = str(body.get("espeak_voice", "en-us")), str(f("sample_rate", 22050, int))
    quality = str(body.get("quality") or "medium")
    if quality not in QUALITY_IDS:
        raise HTTPException(400, f"quality must be one of {sorted(QUALITY_IDS)}.")
    language = str(body.get("language") or "").strip()
    if language and not re.fullmatch(r"[A-Za-z]{2,8}([_-][A-Za-z0-9]{1,8})?", language):
        raise HTTPException(400, "language tag should look like de_DE.")
    lang_args = ["--quality", quality] + (["--language", language] if language else [])
    steps: List[Step] = []
    if body.get("overwrite") and out.exists():
        steps.append(lambda: shutil.rmtree(out, ignore_errors=True))
    if mode == "big":
        steps.append(["-m", "wpiper.prepare_big", "--csv", str(csv_path), "--audio-dir", d["audio_dir"], "--out", str(out),
                      "--espeak-voice", espeak, "--sample-rate", sr, "--min-sec", str(min_sec), "--max-sec", str(max_sec), *lang_args,
                      "--trim-db", str(f("trim_db", 40.0)), "--keep-sec", str(f("keep_sec", 0.25)),
                      "--workers", str(f("workers", min(12, os.cpu_count() or 4), int))])
        steps.append(["-m", "wpiper.make_split", "--data", str(out), "--cps-pct", str(f("cps_low", 0.3)), str(f("cps_high", 99.7)),
                      "--val", str(f("val", 20, int)), "--test", str(f("test", 10, int)), "--seed", str(f("seed", 1234, int))])
    elif mode == "small":
        args = ["-m", "wpiper.prepare_dataset", "--csv", str(csv_path), "--audio-dir", d["audio_dir"], "--out", str(out),
                "--name", out_name, "--espeak-voice", espeak, "--sample-rate", sr, *lang_args, "--val", str(f("val", 20, int)),
                "--test", str(f("test", 5, int)), "--seed", str(f("seed", 1234, int)),
                "--trim-db", str(f("trim_db", 40.0)), "--keep-sec", str(f("keep_sec", 0.25))]
        if f("trim_db", 40.0) <= 0:
            args.append("--no-trim")

        def make_cache_dirs():
            # prepare_dataset saves cache/<id>.pt without creating subfolders, and ids may be paths like wavs/x.wav
            for r in kept:
                (out / "cache" / r["src"]).parent.mkdir(parents=True, exist_ok=True)

        steps.append(make_cache_dirs)
        steps.append(args)
    else:
        raise HTTPException(400, "mode must be 'big' or 'small'.")
    job = start_job("prepare", f"prepare {name} -> {out_name} ({mode}, {len(kept)} clips)", steps, {"out": out_name, "draft": name})
    return {"job": public(job), "clips": len(kept), "dropped": len(d["rows"]) - len(kept)}


# ============================ runs ============================
def rdir(run: str) -> Path:
    d = RUNS / nm(run, "run name")
    if not d.is_dir():
        raise HTTPException(404, f"Run '{run}' not found.")
    return d


def read_metrics(path: Path) -> List[dict]:
    recs = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return recs


def tail_records(path: Path, nbytes=200_000) -> List[dict]:
    try:
        with open(path, "rb") as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - nbytes))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[1:] if size > nbytes else lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def ckpt_list(d: Path):
    return sorted(({"file": c.name, "size_mb": round(c.stat().st_size / 1e6, 1), "modified": int(c.stat().st_mtime)}
                   for c in d.glob("*.ckpt")), key=lambda c: c["modified"], reverse=True)


def run_quality(d: Path):
    """Model size recorded in the run's start records; None for runs from before size presets."""
    try:
        with open(d / "metrics.jsonl", encoding="utf-8", errors="replace") as f:
            for _, line in zip(range(50), f):
                if '"quality"' in line:
                    return json.loads(line).get("quality")
        return "medium"  # started before size presets existed, when medium was the only size
    except (OSError, ValueError):
        return None


@app.get("/api/runs")
def list_runs():
    if not RUNS.exists():
        return []
    out = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        tail = tail_records(d / "metrics.jsonl")
        last = lambda t: next((r for r in reversed(tail) if r.get("type") == t), None)  # noqa: E731
        train, val, prog = last("train"), last("val"), last("progress")
        mt = (d / "metrics.jsonl").stat().st_mtime if (d / "metrics.jsonl").exists() else None
        out.append({"name": d.name, "checkpoints": ckpt_list(d), "size_mb": dir_size_mb(d), "quality": run_quality(d),
                    "epoch": (prog or train or {}).get("epoch"), "val_loss": (val or {}).get("loss"),
                    "g": (train or {}).get("g"), "last_activity": mt,
                    "running": bool(running_jobs("train", run=d.name))})
    return out


@app.get("/api/runs/{run}/metrics")
def run_metrics(run: str, max_points: int = 600):
    d = rdir(run)
    recs = read_metrics(d / "metrics.jsonl")
    starts = [r for r in recs if r.get("type") == "start"]
    train = [r for r in recs if r.get("type") == "train"]
    val = [r for r in recs if r.get("type") == "val"]
    prog = next((r for r in reversed(recs) if r.get("type") == "progress"), None)
    if len(train) > max_points:
        stride = len(train) / max_points
        train = [train[int(i * stride)] for i in range(max_points)] + [train[-1]]
    start = starts[-1] if starts else {}
    # seconds per epoch from consecutive epoch-end records
    ends = [r for r in recs if r.get("type") == "train"][-12:]
    per_epoch = None
    if len(ends) >= 2 and ends[-1]["epoch"] > ends[0]["epoch"]:
        per_epoch = (ends[-1]["time"] - ends[0]["time"]) / (ends[-1]["epoch"] - ends[0]["epoch"])
    cur = (prog or (train[-1] if train else {})).get("epoch")
    maxe = start.get("max_epochs")
    eta = (maxe - cur) * per_epoch if per_epoch and maxe and cur is not None and maxe > cur else None
    mt = (d / "metrics.jsonl").stat().st_mtime if (d / "metrics.jsonl").exists() else None
    return {"run": run, "start": start, "train": train, "val": val, "progress": prog, "sec_per_epoch": per_epoch,
            "eta_sec": eta, "last_activity": mt, "running": [public(j) for j in running_jobs("train", run=run)],
            "checkpoints": ckpt_list(d), "dataset": Path(start.get("data", "")).name or None}


@app.get("/api/runs/{run}/samples")
def run_samples(run: str):
    s = rdir(run) / "samples"
    if not s.is_dir():
        return []
    eps = []
    for e in s.iterdir():
        m = re.fullmatch(r"epoch(\d+)", e.name)
        if e.is_dir() and m:
            eps.append({"epoch": int(m.group(1)), "dir": e.name, "files": sorted(p.name for p in e.glob("*.wav"))})
    return sorted(eps, key=lambda e: e["epoch"], reverse=True)


@app.get("/api/runs/{run}/samples/{epoch_dir}/{file}")
def run_sample_file(run: str, epoch_dir: str, file: str):
    p = rdir(run) / "samples" / nm(epoch_dir) / nm(file)
    if not p.is_file():
        raise HTTPException(404, "Sample not found.")
    return FileResponse(p, media_type="audio/wav")


@app.delete("/api/runs/{run}")
def delete_run(run: str):
    d = rdir(run)
    if running_jobs("train", run=run):
        raise HTTPException(409, "This run is training; stop it first.")
    shutil.rmtree(d, ignore_errors=True)
    if d.exists():
        raise HTTPException(500, "Could not fully delete the run (files in use?).")
    return {"deleted": run}


@app.delete("/api/runs/{run}/checkpoints/{file}")
def delete_ckpt(run: str, file: str):
    d = rdir(run)
    p = d / nm(file)
    if p.suffix != ".ckpt" or not p.is_file():
        raise HTTPException(404, "Checkpoint not found.")
    if running_jobs("train", run=run):
        raise HTTPException(409, "This run is training; stop it first.")
    p.unlink()
    return {"deleted": file}


@app.post("/api/runs/{run}/export")
def export_run(run: str, body: dict):
    """Convert a checkpoint to voices/<name>.onnx (+ .onnx.json) with wpiper.export_onnx."""
    d = rdir(run)
    ckpt = d / nm(body.get("checkpoint", ""), "checkpoint")
    if ckpt.suffix != ".ckpt" or not ckpt.is_file():
        raise HTTPException(404, "Checkpoint not found.")
    start = next((r for r in reversed(read_metrics(d / "metrics.jsonl")) if r.get("type") == "start"), {})
    dataset = body.get("dataset") or Path(start.get("data", "")).name
    cfg = DATA / dataset / "config.json" if dataset else None
    if not cfg or not cfg.is_file():
        raise HTTPException(400, "Could not find the dataset config.json for this run; choose a dataset.")
    name = nm(body.get("name", ""), "voice name")
    onnx, js = VOICES / f"{name}.onnx", VOICES / f"{name}.onnx.json"
    if (onnx.exists() or js.exists()) and not body.get("overwrite"):
        raise HTTPException(409, f"Voice '{name}' already exists.")
    work = VOICES / f"__export_{name}"

    def prep():
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)

    def install():
        if not (work / f"{name}.onnx").exists() or not (work / f"{name}.onnx.json").exists():
            raise RuntimeError("export produced no model")
        os.replace(work / f"{name}.onnx.json", js)
        os.replace(work / f"{name}.onnx", onnx)
        shutil.rmtree(work, ignore_errors=True)
        print(f"installed voices/{name}.onnx")

    steps: List[Step] = [prep, ["-m", "wpiper.export_onnx", "--checkpoint", str(ckpt), "--config", str(cfg),
                               "--output", str(work / f"{name}.onnx")], install]
    return public(start_job("export", f"export {run}/{ckpt.name} -> voices/{name}", steps, {"run": run, "voice": name}))


# ============================ training ============================
@app.post("/api/train")
def train(body: dict):
    dataset = nm(body.get("dataset", ""), "dataset")
    d = DATA / dataset
    if not (d / "split.json").exists():
        raise HTTPException(400, f"Dataset '{dataset}' has no split.json; run make_split (big) or prepare again.")
    run = nm(body.get("run", ""), "run name")
    out = RUNS / run
    mode = body.get("mode", "scratch")
    if running_jobs("train"):
        raise HTTPException(409, "A training job is already running (one at a time). Stop it first.")
    if shutil.disk_usage(ROOT).free < 2e9:
        raise HTTPException(409, "Less than 2 GB free disk space; checkpoints are large.")

    ints = {"batch_size": 32, "max_epochs": 10000, "val_every": 5, "sample_every": 50, "save_every": 10,
            "keep_every": 100, "num_workers": 0, "log_every": 50, "seed": 1234}
    args = ["-m", "wpiper.train", "--data", str(d), "--out", str(out)]
    for key, default in ints.items():
        try:
            v = int(body.get(key, default))
        except (TypeError, ValueError):
            raise HTTPException(400, f"{key} must be a whole number.")
        if v < 0 or (v == 0 and key in ("batch_size", "max_epochs")):
            raise HTTPException(400, f"{key} out of range.")
        args += ["--" + key.replace("_", "-"), str(v)]
    if body.get("max_batch_frames"):
        args += ["--max-batch-frames", str(int(body["max_batch_frames"]))]
    quality = str(body.get("quality") or "")
    if quality and quality not in QUALITY_IDS:
        raise HTTPException(400, f"quality must be one of {sorted(QUALITY_IDS)}.")
    precision = str(body.get("precision", "bf16"))
    if precision not in ("32", "bf16", "16"):
        raise HTTPException(400, "precision must be 32, bf16 or 16.")
    args += ["--precision", precision]

    has_ckpt = out.is_dir() and any(out.glob("*.ckpt"))
    if mode == "scratch":
        if has_ckpt:
            raise HTTPException(409, f"Run '{run}' already has checkpoints: use a new name, or choose Resume.")
    elif mode == "resume":
        quality = ""  # a resumed run keeps its checkpoint's size
        ck = out / nm(body.get("checkpoint", "last.ckpt"), "checkpoint")
        if not ck.is_file():
            raise HTTPException(404, f"{ck.name} not found in run '{run}'.")
        args += ["--resume", str(ck)]
    elif mode == "finetune":
        if has_ckpt:
            raise HTTPException(409, f"Run '{run}' already has checkpoints: fine-tuning starts a new run; use a new name.")
        base = (RUNS / body.get("base", "")).resolve()
        if base.suffix != ".ckpt" or base.parent.parent != RUNS.resolve() or not base.is_file():
            raise HTTPException(404, "Base checkpoint not found (expected <run>/<file>.ckpt under runs/).")
        args += ["--init-from", str(base)]
    else:
        raise HTTPException(400, "mode must be scratch, resume or finetune.")
    if quality:
        args += ["--quality", quality]
    out.mkdir(parents=True, exist_ok=True)
    return public(start_job("train", f"train {run} on {dataset} ({mode})", [args], {"run": run, "dataset": dataset}))


# ============================ page ============================
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PAGE, media_type="text/html")


load_jobs()

if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    print(f"wpiper trainer: http://{a.host}:{a.port}   (data={DATA}, runs={RUNS})")
    uvicorn.run(app, host=a.host, port=a.port)
