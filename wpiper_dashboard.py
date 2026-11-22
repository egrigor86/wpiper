"""Dashboard + admin routes for wpiper_server (served at GET /).

Hooked in from wpiper_server.py with:
    import wpiper_dashboard; wpiper_dashboard.register(app, sys.modules[__name__])
Env:
    WPIPER_ADMIN_KEY   if set, /admin/* requires header X-Admin-Key (the dashboard has a field for it)
"""

import gc
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import onnxruntime as ort
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse

HTML = Path(__file__).with_name("dashboard.html")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
ADMIN_KEY = os.environ.get("WPIPER_ADMIN_KEY")


def require_admin(x_admin_key: str = Header(default="")):
    if ADMIN_KEY and not hmac.compare_digest(x_admin_key, ADMIN_KEY):
        raise HTTPException(401, "Invalid or missing admin key.")


def register(app, srv) -> None:
    """srv is the running wpiper_server module (passed in so __main__ state isn't duplicated)."""
    config_file = srv.BASE_DIR / "server_config.json"
    if config_file.exists():
        try:
            srv.DEFAULT_VOICE_NAME = json.loads(config_file.read_text(encoding="utf-8")).get(
                "default", srv.DEFAULT_VOICE_NAME)
        except Exception:
            pass

    router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])

    def paths(name: str):
        if not NAME_RE.match(name):
            raise HTTPException(400, "Voice name may only contain letters, digits, _ . -")
        return srv.VOICES_DIR / f"{name}.onnx", srv.VOICES_DIR / f"{name}.onnx.json"

    def unload(name: str):
        with srv._lock:
            srv._voices.pop(name, None)
        gc.collect()  # let onnxruntime release the file handle (Windows locks it)

    def describe(name: str) -> dict:
        onnx, js = paths(name)
        info = {"name": name, "size_mb": round(onnx.stat().st_size / 1e6, 1),
                "modified": int(onnx.stat().st_mtime), "loaded": name in srv._voices,
                "default": name == srv.default_voice()}
        try:
            cfg = json.loads(js.read_text(encoding="utf-8"))
            info["sample_rate"] = cfg["audio"]["sample_rate"]
            info["espeak"] = cfg["espeak"]["voice"]
            info["defaults"] = cfg.get("inference", {})
        except Exception as e:
            info["error"] = f"bad config: {e}"
        return info

    @router.get("/voices")
    def voices():
        return [describe(n) for n in srv.discover_voices()]

    @router.get("/status")
    def status():
        return {"providers": list(srv.PROVIDERS), "available_providers": ort.get_available_providers(),
                "voices_dir": str(srv.VOICES_DIR), "default": srv.default_voice(),
                "loaded": sorted(srv._voices), "auth_required": bool(ADMIN_KEY)}

    @router.post("/reload")
    def reload():
        with srv._lock:
            srv._voices.clear()
        gc.collect()
        return {"voices": srv.discover_voices()}

    @router.post("/default")
    def set_default(body: dict):
        name = body.get("name", "")
        if name not in srv.discover_voices():
            raise HTTPException(404, f"Unknown voice '{name}'.")
        srv.DEFAULT_VOICE_NAME = name
        config_file.write_text(json.dumps({"default": name}, indent=2), encoding="utf-8")
        return {"default": name}

    @router.put("/voices/{name}/{kind}")
    async def upload(name: str, kind: str, request: Request):
        """Upload the .onnx.json first, then the .onnx; the onnx upload validates and activates the voice."""
        if kind not in ("json", "onnx"):
            raise HTTPException(404, "kind must be 'json' or 'onnx'")
        onnx, js = paths(name)
        srv.VOICES_DIR.mkdir(parents=True, exist_ok=True)
        data = await request.body()
        if not data:
            raise HTTPException(400, "Empty upload.")
        if kind == "json":
            try:
                cfg = json.loads(data)
                cfg["audio"]["sample_rate"], cfg["espeak"]["voice"], cfg["phoneme_id_map"]
            except Exception as e:
                raise HTTPException(400, f"Not a valid voice config: {e!r}")
            js.with_name(js.name + ".part").write_bytes(data)
            return {"stored": "config (pending onnx)"}

        pending = js.with_name(js.name + ".part")
        if not pending.exists() and not js.exists():
            raise HTTPException(400, "Upload the .onnx.json before the .onnx.")
        tmp_onnx = onnx.with_name(onnx.name + ".part")
        tmp_onnx.write_bytes(data)
        # validate against a scratch pair so a bad upload never replaces a working voice
        scratch = srv.VOICES_DIR / f"__check_{int(time.time())}.onnx"
        scratch_js = scratch.with_suffix(".onnx.json")
        try:
            scratch.write_bytes(data)
            scratch_js.write_bytes(pending.read_bytes() if pending.exists() else js.read_bytes())
            voice = srv.Voice(scratch, providers=srv.PROVIDERS)
            voice.synthesize("Test.")
            del voice
        except Exception as e:
            raise HTTPException(400, f"Model failed validation: {e}")
        finally:
            gc.collect()
            for p in (scratch, scratch_js):
                p.unlink(missing_ok=True)
        unload(name)
        os.replace(tmp_onnx, onnx)
        if pending.exists():
            os.replace(pending, js)
        return describe(name)

    @router.get("/checkpoints")
    def checkpoints():
        runs, data = srv.BASE_DIR / "runs", srv.BASE_DIR / "data"
        found = []
        for ck in runs.glob("*/*.ckpt") if runs.exists() else []:
            run = ck.parent.name
            found.append({"path": f"{run}/{ck.name}", "run": run, "file": ck.name,
                          "size_mb": round(ck.stat().st_size / 1e6, 1), "modified": int(ck.stat().st_mtime),
                          "has_config": (data / run / "config.json").exists()})
        return sorted(found, key=lambda c: c["modified"], reverse=True)

    @router.post("/convert")
    def convert(body: dict):
        """Export runs/<run>/<file>.ckpt to voices/<name>.onnx using data/<run>/config.json."""
        runs, data = (srv.BASE_DIR / "runs").resolve(), srv.BASE_DIR / "data"
        ckpt = (runs / body.get("checkpoint", "")).resolve()
        if ckpt.suffix != ".ckpt" or ckpt.parent.parent != runs or not ckpt.is_file():
            raise HTTPException(404, "Checkpoint not found under runs/<run>/.")
        config = data / ckpt.parent.name / "config.json"
        if not config.is_file():
            raise HTTPException(400, f"Missing {config.relative_to(srv.BASE_DIR)} (needed for the voice config).")
        name = body.get("name", "")
        onnx, js = paths(name)
        exists = onnx.exists() or js.exists()
        if exists and not body.get("overwrite"):
            raise HTTPException(409, f"'{name}' already exists.")

        srv.VOICES_DIR.mkdir(parents=True, exist_ok=True)
        work = srv.VOICES_DIR / f"__convert_{name}"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "wpiper.export_onnx", "--checkpoint", str(ckpt),
                 "--config", str(config), "--output", str(work / f"{name}.onnx")],
                cwd=srv.BASE_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
            if proc.returncode != 0:
                tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-8:])
                raise HTTPException(500, f"Export failed:\n{tail}")
            if not (work / f"{name}.onnx").exists() or not (work / f"{name}.onnx.json").exists():
                raise HTTPException(500, "Export finished but produced no model/config.")
            unload(name)
            os.replace(work / f"{name}.onnx.json", js)
            os.replace(work / f"{name}.onnx", onnx)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "Export timed out after 15 minutes.")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return describe(name)

    @router.post("/rename")
    def rename(body: dict):
        old, new = body.get("name", ""), body.get("new", "")
        if old not in srv.discover_voices():
            raise HTTPException(404, f"Unknown voice '{old}'.")
        o_onnx, o_js = paths(old)
        n_onnx, n_js = paths(new)
        if n_onnx.exists() or n_js.exists():
            raise HTTPException(409, f"'{new}' already exists.")
        unload(old)
        try:
            o_onnx.rename(n_onnx)
            o_js.rename(n_js)
        except OSError as e:
            raise HTTPException(500, f"Rename failed: {e}")
        if srv.DEFAULT_VOICE_NAME == old:
            srv.DEFAULT_VOICE_NAME = new
            config_file.write_text(json.dumps({"default": new}, indent=2), encoding="utf-8")
        return describe(new)

    @router.delete("/voices/{name}")
    def delete(name: str):
        if name not in srv.discover_voices():
            raise HTTPException(404, f"Unknown voice '{name}'.")
        onnx, js = paths(name)
        unload(name)
        try:
            onnx.unlink()
            js.unlink()
        except OSError as e:
            raise HTTPException(500, f"Could not delete (file in use?): {e}")
        return {"deleted": name}

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(HTML, media_type="text/html")
