"""Simple launcher for wpiper: pick dataset + base checkpoint, train, audition checkpoints, export to voices.

Run from the wpiper folder (or use trainer_gui.bat). Settings are remembered in trainer_gui_settings.json.
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk

try:
    import winsound
except ImportError:  # not Windows
    winsound = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SETTINGS_FILE = ROOT / "trainer_gui_settings.json"
SCRATCH = "(from scratch)"
AUTO_QUALITY = "auto"
QUALITY_CHOICES = [AUTO_QUALITY, "x-low", "low", "medium", "high"]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

DEFAULTS = {
    "dataset": "",
    "base": SCRATCH,
    "mode": "finetune",
    "run_name": "",
    "batch_size": "12",
    "max_epochs": "140",
    "precision": "bf16",
    "val_every": "1",
    "sample_every": "5",
    "save_every": "5",
    "keep_every": "10",
    "num_workers": "0",
    "log_every": "50",
    "max_batch_frames": "24000",
    "quality": AUTO_QUALITY,
    "seed": "1234",
    "test_run": "",
    "test_dataset": "",
    "test_ckpt": "",
    "test_file": "samples.txt",
    "noise_scale": "0.667",
    "length_scale": "1.0",
    "noise_w": "0.8",
    "voice_name": "",
}

TRAIN_FIELDS = [
    ("batch_size", "Batch size"),
    ("max_epochs", "Max epochs"),
    ("precision", "Precision"),
    ("val_every", "Validate every (epochs)"),
    ("sample_every", "Samples every (epochs)"),
    ("save_every", "Save last.ckpt every (epochs)"),
    ("keep_every", "Keep epoch-N.ckpt every (epochs)"),
    ("num_workers", "Data loader workers"),
    ("log_every", "Log every (steps)"),
    ("max_batch_frames", "Max batch frames"),
    ("seed", "Seed"),
]


def ckpt_sort_key(path: Path):
    stem = path.stem
    if stem == "last":
        return (0, 0)
    try:
        return (1, -int(stem.split("-")[-1]))
    except ValueError:
        return (2, 0)


def list_datasets() -> list[str]:
    data = ROOT / "data"
    if not data.is_dir():
        return []
    return sorted(d.name for d in data.iterdir() if d.is_dir() and (d / "config.json").exists())


def list_runs() -> list[str]:
    runs = ROOT / "runs"
    if not runs.is_dir():
        return []
    return sorted(d.name for d in runs.iterdir() if d.is_dir() and any(d.glob("*.ckpt")))


def list_ckpts(run: str) -> list[str]:
    return [p.name for p in sorted((ROOT / "runs" / run).glob("*.ckpt"), key=ckpt_sort_key)]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("wpiper trainer")
        self.geometry("900x720")
        self.settings = dict(DEFAULTS)
        try:
            self.settings.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        self.v = {k: tk.StringVar(value=str(val)) for k, val in self.settings.items()}

        self.proc = None
        self.log_queue = queue.Queue()
        self.ui_queue = queue.Queue()
        self.voice_cache = {}
        self.play_stop = threading.Event()
        self.test_wavs = []
        self.busy = False
        self.prev_dataset = self.v["dataset"].get()

        self.python = sys.executable
        if self.python.lower().endswith("pythonw.exe"):
            self.python = self.python[: -len("pythonw.exe")] + "python.exe"

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.train_tab = ttk.Frame(nb, padding=10)
        self.test_tab = ttk.Frame(nb, padding=10)
        nb.add(self.train_tab, text="Train")
        nb.add(self.test_tab, text="Test & Export")
        self.build_train_tab()
        self.build_test_tab()
        self.refresh_lists()

        for var in self.v.values():
            var.trace_add("write", lambda *_: self.save_settings())
        self.v["dataset"].trace_add("write", self.on_dataset_change)
        self.v["test_run"].trace_add("write", self.on_test_run_change)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(200, self.poll)

    # ---------- settings ----------
    def save_settings(self):
        try:
            SETTINGS_FILE.write_text(json.dumps({k: v.get() for k, v in self.v.items()}, indent=2), encoding="utf-8")
        except OSError:
            pass

    def on_close(self):
        self.play_stop.set()
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("Training running", "Training is still running. Stop it and quit?"):
                return
            self.stop_training()
        self.save_settings()
        self.destroy()

    # ---------- train tab ----------
    def build_train_tab(self):
        t = self.train_tab
        t.columnconfigure(1, weight=1)
        t.columnconfigure(3, weight=1)

        ttk.Label(t, text="Dataset (data/)").grid(row=0, column=0, sticky="w", pady=3)
        self.dataset_cb = ttk.Combobox(t, textvariable=self.v["dataset"], state="readonly")
        self.dataset_cb.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(t, text="Run name (runs/)").grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(t, textvariable=self.v["run_name"]).grid(row=0, column=3, sticky="ew", padx=6)

        ttk.Label(t, text="Base checkpoint").grid(row=1, column=0, sticky="w", pady=3)
        self.base_cb = ttk.Combobox(t, textvariable=self.v["base"], state="readonly")
        self.base_cb.grid(row=1, column=1, columnspan=2, sticky="ew", padx=6)
        ttk.Button(t, text="Refresh lists", command=self.refresh_lists).grid(row=1, column=3, sticky="w", padx=6)

        modes = ttk.Frame(t)
        modes.grid(row=2, column=0, columnspan=4, sticky="w", pady=3)
        ttk.Radiobutton(modes, text="Fine-tune: new run from base weights (epoch 0)", value="finetune",
                        variable=self.v["mode"]).pack(side="left")
        ttk.Radiobutton(modes, text="Resume: continue base checkpoint (keeps epoch counter)", value="resume",
                        variable=self.v["mode"]).pack(side="left", padx=16)

        box = ttk.LabelFrame(t, text="Parameters", padding=8)
        box.grid(row=3, column=0, columnspan=4, sticky="ew", pady=8)
        for i, (key, label) in enumerate(TRAIN_FIELDS):
            r, c = divmod(i, 2)
            ttk.Label(box, text=label).grid(row=r, column=c * 2, sticky="w", pady=2, padx=(0 if c == 0 else 24, 0))
            if key == "precision":
                w = ttk.Combobox(box, textvariable=self.v[key], values=["32", "bf16", "16"], width=8, state="readonly")
            else:
                w = ttk.Entry(box, textvariable=self.v[key], width=10)
            w.grid(row=r, column=c * 2 + 1, sticky="w", padx=6)
        qr = len(TRAIN_FIELDS) // 2 + 1
        ttk.Label(box, text="Model size (new runs; auto = dataset / base)").grid(row=qr, column=0, sticky="w", pady=2)
        ttk.Combobox(box, textvariable=self.v["quality"], values=QUALITY_CHOICES, width=8, state="readonly").grid(
            row=qr, column=1, sticky="w", padx=6)

        buttons = ttk.Frame(t)
        buttons.grid(row=4, column=0, columnspan=4, sticky="w")
        self.start_btn = ttk.Button(buttons, text="Start training", command=self.start_training)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop_training, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(buttons, text="Open viewer", command=self.open_viewer).pack(side="left", padx=6)
        self.status = ttk.Label(buttons, text="Idle")
        self.status.pack(side="left", padx=12)

        t.rowconfigure(5, weight=1)
        logf = ttk.Frame(t)
        logf.grid(row=5, column=0, columnspan=4, sticky="nsew", pady=(8, 0))
        self.log = tk.Text(logf, height=10, wrap="none", state="disabled", font=("Consolas", 9))
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

    def refresh_lists(self):
        datasets = list_datasets()
        self.dataset_cb["values"] = datasets
        if self.v["dataset"].get() not in datasets:
            self.v["dataset"].set(datasets[0] if datasets else "")
        bases = [SCRATCH]
        for run in list_runs():
            bases += [f"{run}/{c}" for c in list_ckpts(run)]
        self.base_cb["values"] = bases
        if self.v["base"].get() not in bases:
            self.v["base"].set(SCRATCH)
        self.test_dataset_cb["values"] = datasets
        runs = list_runs()
        self.test_run_cb["values"] = runs
        if self.v["test_run"].get() not in runs:
            self.v["test_run"].set(runs[0] if runs else "")
        self.on_test_run_change()

    def on_dataset_change(self, *_):
        ds = self.v["dataset"].get()
        if self.v["run_name"].get() in ("", self.prev_dataset):
            self.v["run_name"].set(ds)
        self.prev_dataset = ds

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        if int(self.log.index("end-1c").split(".")[0]) > 3000:
            self.log.delete("1.0", "1000.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start_training(self):
        ds, run = self.v["dataset"].get(), self.v["run_name"].get().strip()
        if not ds or not run:
            messagebox.showerror("Missing", "Choose a dataset and a run name.")
            return
        try:
            nums = {k: int(self.v[k].get()) for k, _ in TRAIN_FIELDS if k != "precision"}
        except ValueError:
            messagebox.showerror("Invalid", "All numeric parameters must be whole numbers.")
            return
        data_dir = ROOT / "data" / ds
        if (data_dir / "audio.bin").exists() and not (data_dir / "split.json").exists():
            messagebox.showerror("No split", f"data\\{ds} has no split.json. Run wpiper.make_split on it first.")
            return
        out_dir = ROOT / "runs" / run
        base = self.v["base"].get()
        mode = self.v["mode"].get()
        if base == SCRATCH:
            mode = None
        elif mode == "finetune" and (out_dir / "last.ckpt").exists():
            if not messagebox.askyesno(
                "Run exists",
                f"runs\\{run} already has checkpoints. A new fine-tune starts at epoch 0 and may overwrite them.\n\nContinue?",
            ):
                return

        cmd = [self.python, "-u", "-m", "wpiper.train", "--data", f"data\\{ds}", "--out", f"runs\\{run}"]
        if mode:
            cmd += ["--init-from" if mode == "finetune" else "--resume", "runs\\" + base.replace("/", "\\")]
        quality = self.v["quality"].get()
        if quality in QUALITY_CHOICES and quality != AUTO_QUALITY and mode != "resume":
            cmd += ["--quality", quality]  # a resumed run keeps its checkpoint's size
        cmd += [
            "--precision", self.v["precision"].get(),
            "--batch-size", str(nums["batch_size"]),
            "--max-epochs", str(nums["max_epochs"]),
            "--val-every", str(nums["val_every"]),
            "--sample-every", str(nums["sample_every"]),
            "--save-every", str(nums["save_every"]),
            "--keep-every", str(nums["keep_every"]),
            "--num-workers", str(nums["num_workers"]),
            "--log-every", str(nums["log_every"]),
            "--max-batch-frames", str(nums["max_batch_frames"]),
            "--seed", str(nums["seed"]),
        ]
        self.append_log("\n> " + " ".join(cmd) + "\n")
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
            )
        except OSError as e:
            messagebox.showerror("Could not start", str(e))
            return
        threading.Thread(target=self.pump_output, args=(self.proc,), daemon=True).start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.configure(text=f"Training runs\\{run} (pid {self.proc.pid})")
        self.save_settings()

    def pump_output(self, proc):
        for line in proc.stdout:
            self.log_queue.put(line)

    def stop_training(self):
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=NO_WINDOW)

    def open_viewer(self):
        ds = self.v["dataset"].get() or "christopher"
        subprocess.Popen([self.python, "-m", "wpiper.viewer", "--data", f"data\\{ds}"], cwd=ROOT,
                         creationflags=NO_WINDOW)
        self.after(1500, lambda: webbrowser.open("http://127.0.0.1:8765"))

    def poll(self):
        try:
            while True:
                self.append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                self.ui_queue.get_nowait()()
        except queue.Empty:
            pass
        if self.proc is not None and self.proc.poll() is not None:
            self.append_log(f"\n[training exited with code {self.proc.returncode}]\n")
            self.proc = None
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status.configure(text="Idle")
            self.refresh_lists()
        self.after(300, self.poll)

    # ---------- test & export tab ----------
    def build_test_tab(self):
        t = self.test_tab
        t.columnconfigure(1, weight=1)
        t.columnconfigure(3, weight=1)

        ttk.Label(t, text="Run").grid(row=0, column=0, sticky="w", pady=3)
        self.test_run_cb = ttk.Combobox(t, textvariable=self.v["test_run"], state="readonly")
        self.test_run_cb.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(t, text="Checkpoint").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.test_ckpt_cb = ttk.Combobox(t, textvariable=self.v["test_ckpt"], state="readonly")
        self.test_ckpt_cb.grid(row=0, column=3, sticky="ew", padx=6)

        ttk.Label(t, text="Config from dataset").grid(row=1, column=0, sticky="w", pady=3)
        self.test_dataset_cb = ttk.Combobox(t, textvariable=self.v["test_dataset"], state="readonly")
        self.test_dataset_cb.grid(row=1, column=1, sticky="ew", padx=6)
        ttk.Label(t, text="Sentences file").grid(row=1, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(t, textvariable=self.v["test_file"]).grid(row=1, column=3, sticky="ew", padx=6)

        sc = ttk.Frame(t)
        sc.grid(row=2, column=0, columnspan=4, sticky="w", pady=3)
        for key, label in (("noise_scale", "Noise scale"), ("length_scale", "Length scale"), ("noise_w", "Noise W")):
            ttk.Label(sc, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(sc, textvariable=self.v[key], width=7).pack(side="left", padx=(0, 16))

        bar = ttk.Frame(t)
        bar.grid(row=3, column=0, columnspan=4, sticky="w", pady=6)
        self.render_btn = ttk.Button(bar, text="Render test samples", command=self.render_samples)
        self.render_btn.pack(side="left")
        ttk.Button(bar, text="Play selected", command=self.play_selected).pack(side="left", padx=6)
        ttk.Button(bar, text="Play all", command=self.play_all).pack(side="left")
        ttk.Button(bar, text="Stop audio", command=self.stop_audio).pack(side="left", padx=6)
        self.test_status = ttk.Label(bar, text="")
        self.test_status.pack(side="left", padx=12)

        t.rowconfigure(4, weight=1)
        lf = ttk.Frame(t)
        lf.grid(row=4, column=0, columnspan=4, sticky="nsew")
        self.sentences = tk.Listbox(lf, activestyle="none", exportselection=False)
        sb = ttk.Scrollbar(lf, command=self.sentences.yview)
        self.sentences.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.sentences.pack(side="left", fill="both", expand=True)
        self.sentences.bind("<Double-Button-1>", lambda _e: self.play_selected())

        ex = ttk.LabelFrame(t, text="Export to voices/", padding=8)
        ex.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ex.columnconfigure(1, weight=1)
        ttk.Label(ex, text="Voice name").grid(row=0, column=0, sticky="w")
        ttk.Entry(ex, textvariable=self.v["voice_name"]).grid(row=0, column=1, sticky="ew", padx=6)
        self.export_btn = ttk.Button(ex, text="Export checkpoint", command=self.export_voice)
        self.export_btn.grid(row=0, column=2)

    def on_test_run_change(self, *_):
        run = self.v["test_run"].get()
        ckpts = list_ckpts(run) if run else []
        self.test_ckpt_cb["values"] = ckpts
        if self.v["test_ckpt"].get() not in ckpts:
            self.v["test_ckpt"].set(ckpts[0] if ckpts else "")
        datasets = list_datasets()
        if run in datasets:
            self.v["test_dataset"].set(run)
        elif self.v["test_dataset"].get() not in datasets:
            self.v["test_dataset"].set(self.v["dataset"].get())

    def selected_ckpt(self):
        run, name = self.v["test_run"].get(), self.v["test_ckpt"].get()
        if not run or not name:
            messagebox.showerror("Missing", "Choose a run and a checkpoint.")
            return None
        return ROOT / "runs" / run / name

    def ensure_onnx(self, ckpt: Path) -> Path:
        """Export ckpt to a cached onnx in runs/<run>/gui_onnx (re-exported if the ckpt changed)."""
        out_dir = ckpt.parent / "gui_onnx"
        out_dir.mkdir(exist_ok=True)
        onnx = out_dir / f"{ckpt.stem}_{int(ckpt.stat().st_mtime)}.onnx"
        if onnx.exists() and onnx.with_suffix(".onnx.json").exists():
            return onnx
        for old in out_dir.glob(f"{ckpt.stem}_*"):
            old.unlink(missing_ok=True)
        ds = self.v["test_dataset"].get()
        config = ROOT / "data" / ds / "config.json"
        if not config.exists():
            raise RuntimeError(f"No config.json for dataset '{ds}'.")
        r = subprocess.run(
            [self.python, "-m", "wpiper.export_onnx", "--checkpoint", str(ckpt), "--config", str(config),
             "--output", str(onnx)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW,
        )
        if r.returncode != 0:
            raise RuntimeError("Export failed:\n" + (r.stderr or r.stdout)[-1500:])
        return onnx

    def run_task(self, fn, status_label):
        if self.busy:
            return
        self.busy = True
        self.render_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.test_status.configure(text=status_label)

        def work():
            try:
                msg = fn()
                err = None
            except Exception as e:  # shown to the user
                msg, err = "", str(e)

            def done():
                self.busy = False
                self.render_btn.configure(state="normal")
                self.export_btn.configure(state="normal")
                self.test_status.configure(text=msg)
                if err:
                    messagebox.showerror("Error", err)

            self.ui_queue.put(done)

        threading.Thread(target=work, daemon=True).start()

    def render_samples(self):
        ckpt = self.selected_ckpt()
        if not ckpt:
            return
        sentence_file = Path(self.v["test_file"].get())
        sentence_file = sentence_file if sentence_file.is_absolute() else ROOT / sentence_file
        if not sentence_file.exists():
            messagebox.showerror("Missing", f"{sentence_file} not found.")
            return
        try:
            scales = [float(self.v[k].get()) for k in ("noise_scale", "length_scale", "noise_w")]
        except ValueError:
            messagebox.showerror("Invalid", "Noise scale, length scale and noise W must be numbers.")
            return
        lines = [l.strip() for l in sentence_file.read_text(encoding="utf-8").splitlines() if l.strip()]

        def work():
            import soundfile as sf
            from wpiper.synthesize import Voice

            self.ui_queue.put(lambda: self.test_status.configure(text="Exporting checkpoint..."))
            onnx = self.ensure_onnx(ckpt)
            if onnx not in self.voice_cache:
                self.voice_cache = {onnx: Voice(onnx)}
            voice = self.voice_cache[onnx]
            out = onnx.parent.parent / "gui_tests" / onnx.stem
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            wavs = []
            for i, line in enumerate(lines, 1):
                self.ui_queue.put(lambda i=i: self.test_status.configure(text=f"Rendering {i}/{len(lines)}..."))
                audio = voice.synthesize(line, *scales)
                wav = out / f"{i:02d}.wav"
                sf.write(wav, audio, voice.sample_rate)
                wavs.append(wav)

            def show():
                self.test_wavs = wavs
                self.sentences.delete(0, "end")
                for i, line in enumerate(lines, 1):
                    self.sentences.insert("end", f"{i:02d}  {line}")

            self.ui_queue.put(show)
            return f"Rendered {len(wavs)} samples from {ckpt.name}. Double-click a sentence to play."

        self.run_task(work, "Working...")

    def play_wav(self, path):
        if winsound:
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)

    def play_selected(self):
        sel = self.sentences.curselection()
        if sel and sel[0] < len(self.test_wavs):
            self.play_stop.set()
            self.play_wav(self.test_wavs[sel[0]])

    def play_all(self):
        if not self.test_wavs or not winsound:
            return
        self.play_stop.set()
        stop = self.play_stop = threading.Event()

        def work():
            import soundfile as sf

            for i, wav in enumerate(self.test_wavs):
                if stop.is_set():
                    return
                self.ui_queue.put(lambda i=i: (self.sentences.selection_clear(0, "end"), self.sentences.selection_set(i),
                                               self.sentences.see(i)))
                info = sf.info(str(wav))
                winsound.PlaySound(str(wav), winsound.SND_FILENAME | winsound.SND_ASYNC)
                if stop.wait(info.duration + 0.4):
                    return

        threading.Thread(target=work, daemon=True).start()

    def stop_audio(self):
        self.play_stop.set()
        if winsound:
            winsound.PlaySound(None, winsound.SND_PURGE)

    def export_voice(self):
        ckpt = self.selected_ckpt()
        if not ckpt:
            return
        name = self.v["voice_name"].get().strip()
        if not name:
            stem = ckpt.stem.replace("epoch-", "e")
            name = f"{ckpt.parent.name}_{stem}"
            self.v["voice_name"].set(name)
        dest = ROOT / "voices" / f"{name}.onnx"
        if dest.exists() and not messagebox.askyesno("Overwrite?", f"voices\\{dest.name} already exists. Overwrite?"):
            return

        def work():
            onnx = self.ensure_onnx(ckpt)
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(onnx, dest)
            shutil.copy2(onnx.with_suffix(".onnx.json"), dest.with_suffix(".onnx.json"))
            return f"Exported {ckpt.name} -> voices\\{dest.name}"

        self.run_task(work, "Exporting...")


if __name__ == "__main__":
    App().mainloop()
