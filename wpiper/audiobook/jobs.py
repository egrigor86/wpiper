"""Background job queue for audiobook builds (standard library only, no web framework).

One worker thread runs jobs one at a time. Every job lives in <root>/<id>/:
    job.json    state (status, progress, settings, summary)     input.txt   the submitted text
    out/        build_book() output (audio, .srt, chapters/)
State survives a server restart: a job that was queued or running comes back as "interrupted" and can be resumed
(the clip cache makes that cheap, finished sentences are not rendered again).
"""

import json
import queue
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

TERMINAL = ("done", "error", "cancelled", "interrupted")


@dataclass
class Job:
    id: str
    title: str
    voice: str
    status: str = "queued"          # queued | running | done | error | cancelled | interrupted
    created: float = 0.0
    started: float = 0.0
    finished: float = 0.0
    stage: str = ""                 # synth | assemble | encode
    done: int = 0
    total: int = 0
    message: str = ""
    error: str = ""
    request: dict = field(default_factory=dict)   # settings, formats, ... as submitted
    summary: dict = field(default_factory=dict)

    def public(self) -> dict:
        d = asdict(self)
        d["request"] = {k: v for k, v in self.request.items() if k != "text"}
        return d


# runner(job, input_path, out_dir, cancel_event, progress_callback) -> summary dict
Runner = Callable[[Job, Path, Path, threading.Event, Callable[[str, int, int, str], None]], dict]


class JobManager:
    def __init__(self, root: Path, runner: Runner):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self._jobs: Dict[str, Job] = {}
        self._cancel: Dict[str, threading.Event] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.RLock()
        self._last_save = 0.0
        self._load()
        self._worker = threading.Thread(target=self._loop, name="audiobook-worker", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ persistence
    def _dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _save(self, job: Job) -> None:
        d = self._dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(job), indent=1), encoding="utf-8")
        tmp.replace(d / "job.json")

    def _load(self) -> None:
        for p in sorted(self.root.glob("*/job.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                job = Job(**data)
            except Exception:
                continue
            if job.status in ("queued", "running"):
                job.status, job.message = "interrupted", "server restarted - resume to continue"
                self._save(job)
            self._jobs[job.id] = job

    # ------------------------------------------------------------ api
    def submit(self, text: str, title: str, voice: str, request: dict) -> Job:
        job = Job(id=time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4], title=title, voice=voice,
                  created=time.time(), request=request)
        d = self._dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "input.txt").write_text(text, encoding="utf-8")
        with self._lock:
            self._jobs[job.id] = job
            self._save(job)
        self._cancel[job.id] = threading.Event()
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        with self._lock:
            if job.status == "queued":
                job.status, job.finished, job.message = "cancelled", time.time(), "cancelled before it started"
                self._save(job)
            elif job.status == "running":
                self._cancel.setdefault(job_id, threading.Event()).set()
                job.message = "cancelling..."
        return job

    def resume(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job or job.status not in ("interrupted", "cancelled", "error"):
            return None
        with self._lock:
            job.status, job.error, job.message, job.finished = "queued", "", "queued (resumed)", 0.0
            self._save(job)
        self._cancel[job_id] = threading.Event()
        self._queue.put(job_id)
        return job

    def delete(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status in ("queued", "running"):
            return False
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(self._dir(job_id), ignore_errors=True)
        return True

    def files(self, job_id: str) -> List[dict]:
        out = self._dir(job_id) / "out"
        if not out.exists():
            return []
        items = []
        for p in sorted(out.rglob("*")):
            rel = p.relative_to(out)
            if p.is_file() and rel.parts[0] != "work":
                items.append({"path": rel.as_posix(), "size": p.stat().st_size})
        return items

    def file_path(self, job_id: str, rel: str) -> Optional[Path]:
        base = (self._dir(job_id) / "out").resolve()
        target = (base / rel).resolve()
        if base in target.parents and target.is_file() and target.relative_to(base).parts[0] != "work":
            return target
        return None

    # ------------------------------------------------------------ worker
    def _progress_cb(self, job: Job):
        def cb(stage: str, done: int, total: int, msg: str) -> None:
            job.stage, job.done, job.total, job.message = stage, done, total, msg
            now = time.time()
            if now - self._last_save > 3:
                self._last_save = now
                with self._lock:
                    self._save(job)
        return cb

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self._jobs.get(job_id)
            if not job or job.status != "queued":
                continue
            cancel = self._cancel.setdefault(job_id, threading.Event())
            with self._lock:
                job.status, job.started, job.message = "running", time.time(), ""
                job.stage, job.done, job.total = "", 0, 0
                self._save(job)
            try:
                d = self._dir(job_id)
                summary = self.runner(job, d / "input.txt", d / "out", cancel, self._progress_cb(job))
                job.summary, job.status, job.message = summary, "done", "finished"
            except Exception as e:  # Cancelled is an Exception subclass in synth.py
                if cancel.is_set() or type(e).__name__ == "Cancelled":
                    job.status, job.message = "cancelled", "cancelled"
                else:
                    job.status, job.error, job.message = "error", f"{type(e).__name__}: {e}", "failed"
            finally:
                job.finished = time.time()
                with self._lock:
                    self._save(job)
