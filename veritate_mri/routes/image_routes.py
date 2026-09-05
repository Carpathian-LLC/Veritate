# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - IDEA 24's F0 falsifier surface: decoder latency and peak activation bytes at a
#   target output resolution, at random weights. Synchronous like /trainers/sysprobe,
#   because the caller sizes the cost through arms/reps/resolution and a researcher
#   runs it deliberately rather than the UI polling it.
# - Cost is the caller's to choose but not without bound: MAX_EDGE stops a mistyped
#   resolution from allocating the box out from under a training run.
# - /images/sets lists the picture sets and codecs the Images flow can train on;
#   /images/ingest collects photos from a folder on this machine into a set, in a
#   background thread with a status route, because hashing a photo library is not a
#   request-sized job. One ingest at a time: two writers on one set is a corrupt set.
# - /images/pick_folder opens the operating system's own folder chooser on the machine
#   the dashboard runs on and returns the path. A browser cannot hand a server an
#   absolute directory path, and uploading a photo library through HTTP to a server on
#   the same disk is the wrong shape entirely; the dashboard is local, so the OS dialog
#   is the honest picker. The request blocks while the dialog is open.
# - /images/generate runs every generation mode (text, variation, inpaint, expand,
#   unconditional) through image_sample against a trained image model, and returns a
#   PNG. Synchronous: a few forward passes on one window. The last model loaded stays
#   resident so a second picture does not pay the load again. /images/models lists what
#   can generate.
# - /images/caption/* is the captioning stage: a vision teacher describes every picture
#   in a set into <image>.txt sidecars, in a background thread with live progress, a
#   stop, and a one-picture preview so the prompt and model are checked before the
#   whole set is spent on them. /images/caption/options gives the form its choices.
# - /images/mri/<model> is what the Models tab shows for an image model: every
#   checkpoint's probe metrics (image_probe) plus the PNGs it wrote, served by step.
# veritate_mri/routes/image_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time

import numpy as np
from flask import request, send_from_directory
from readers import images as images_reader
from readers import trainers as trainers_reader

from ._common import safe_name
from ._common import safe_route as _safe

# ------------------------------------------------------------------------------------
# Constants

LOG_TAIL_LINES = 14        # run-log lines the Training tab shows for an image run
# Guard, not a tunable: 8K wide is past any target in IDEA 24 and a typo one digit
# longer would allocate tens of GB in the control arm.
MAX_EDGE = 8192
# A set name is a directory name under data/images/; nothing that could walk out of it.
SET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
INGEST_IDLE, INGEST_RUNNING, INGEST_OK, INGEST_FAILED = "idle", "running", "ok", "failed"
DIALOG_PROMPT    = "Choose the folder that holds your photos (subfolders are included)"
DIALOG_TIMEOUT_S = 600
MAX_SOURCE_BYTES = 64 * 1024 * 1024
_MODEL_CACHE = {}      # (name, step) -> (model, codec, geometry); one entry
_MODEL_LOCK = threading.Lock()
CAPTION_IDLE, CAPTION_RUNNING, CAPTION_OK, CAPTION_FAILED, CAPTION_STOPPED = (
    "idle", "running", "ok", "failed", "stopped")
_CAPTION = {"status": CAPTION_IDLE, "set": None, "total": 0, "done": 0, "failed": 0, "samples": [],
            "report": None, "error": None, "started_at": None, "finished_at": None, "stop": False}
_CAPTION_LOCK = threading.Lock()

_INGEST = {"status": INGEST_IDLE, "set": None, "sources": None, "report": None,
           "error": None, "started_at": None, "finished_at": None, "done": 0, "total": 0}
_INGEST_LOCK = threading.Lock()
PROMPT_IDLE, PROMPT_RUNNING, PROMPT_OK, PROMPT_FAILED, PROMPT_STOPPED = "idle", "running", "ok", "failed", "stopped"
PROMPT_MAX_STEPS = 12          # checkpoints drawn per request, evenly spaced, the last always included
_PROMPT = {"status": PROMPT_IDLE, "model": None, "caption": "", "seed": 0, "passes": 8, "mode": "text",
           "steps": [], "results": [], "error": None, "stop": False, "started_at": None, "finished_at": None}
_PROMPT_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions


def ingest_state():
    with _INGEST_LOCK:
        return dict(_INGEST)


def _run_ingest(set_name, sources, min_edge, caption_from_folder, copy):
    from tools import ingest_images

    def progress(done, total):
        with _INGEST_LOCK:
            _INGEST.update(done=int(done), total=int(total))

    try:
        report = ingest_images.ingest(set_name, sources, min_edge=min_edge, copy=copy,
                                      caption_from_folder=caption_from_folder, progress=progress)
        with _INGEST_LOCK:
            _INGEST.update(status=INGEST_OK, report=report, finished_at=time.time())
    except Exception as e:
        with _INGEST_LOCK:
            _INGEST.update(status=INGEST_FAILED, error=type(e).__name__ + ": " + str(e),
                           finished_at=time.time())


def start_ingest(body):
    """Validate and launch. Returns (payload, http_status)."""
    set_name = str(body.get("set") or "").strip()
    if not SET_NAME.match(set_name):
        return {"ok": False, "error": "set name must be letters, digits, _ . - (got "
                + repr(set_name) + ")"}, 400
    sources = body.get("sources")
    if isinstance(sources, str):
        sources = [sources]
    sources = [os.path.expanduser(str(x).strip()) for x in (sources or []) if str(x).strip()]
    if not sources:
        return {"ok": False, "error": "sources: give at least one folder on this machine"}, 400
    missing = [x for x in sources if not os.path.exists(x)]
    if missing:
        return {"ok": False, "error": "not found on this machine: " + ", ".join(missing)}, 400
    min_edge = int(body.get("min_edge") or 0) or None
    with _INGEST_LOCK:
        if _INGEST["status"] == INGEST_RUNNING:
            return {"ok": False, "error": "an ingest is already running (set "
                    + str(_INGEST["set"]) + ")", "state": dict(_INGEST)}, 409
        _INGEST.update(status=INGEST_RUNNING, set=set_name, sources=sources, report=None,
                       error=None, started_at=time.time(), finished_at=None, done=0, total=0)
    kwargs = {"min_edge": min_edge} if min_edge else {}
    from tools import ingest_images
    kwargs.setdefault("min_edge", ingest_images.MIN_EDGE_DEFAULT)
    threading.Thread(target=_run_ingest, name="images:ingest", daemon=True,
                     args=(set_name, sources, kwargs["min_edge"],
                           bool(body.get("caption_from_folder", True)),
                           bool(body.get("copy", False)))).start()
    return {"ok": True, "state": ingest_state()}, 200


def pick_folder(prompt=DIALOG_PROMPT):
    """The OS folder chooser, on the machine the dashboard runs on. Returns
    {ok, path} | {ok: False, cancelled} | {ok: False, unavailable, error}."""
    system = platform.system()
    safe = prompt.replace('"', "'")
    try:
        if system == "Darwin":
            script = 'POSIX path of (choose folder with prompt "' + safe + '")'
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                               timeout=DIALOG_TIMEOUT_S)
            if r.returncode != 0:
                if "-128" in r.stderr or "canceled" in r.stderr.lower():
                    return {"ok": False, "cancelled": True}
                return {"ok": False, "error": r.stderr.strip() or "folder dialog failed"}
            return {"ok": True, "path": r.stdout.strip().rstrip("/")}
        if system == "Windows":
            ps = ("Add-Type -AssemblyName System.Windows.Forms; "
                  "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                  "$d.Description = '" + safe.replace("'", "''") + "'; "
                  "if ($d.ShowDialog() -eq 'OK') { Write-Output $d.SelectedPath }")
            r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                               text=True, timeout=DIALOG_TIMEOUT_S)
            path = r.stdout.strip()
            return {"ok": True, "path": path} if path else {"ok": False, "cancelled": True}
        for cmd in (["zenity", "--file-selection", "--directory", "--title", prompt],
                    ["kdialog", "--getexistingdirectory", os.path.expanduser("~"), "--title", prompt]):
            if shutil.which(cmd[0]):
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=DIALOG_TIMEOUT_S)
                path = r.stdout.strip()
                if r.returncode == 0 and path:
                    return {"ok": True, "path": path}
                return {"ok": False, "cancelled": True}
        return {"ok": False, "unavailable": True,
                "error": "no folder dialog on this system; type the path instead"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "the folder dialog timed out"}
    except OSError as e:
        return {"ok": False, "unavailable": True, "error": str(e)}


def list_image_models():
    """[{name, steps, height, width, codec}] for every model dir whose config says image."""
    from readers import checkpoints, models, paths
    out = []
    for name in models.list_models():
        try:
            with open(paths.config_path(name), encoding="utf-8") as handle:
                cfg = json.load(handle)
        except (OSError, ValueError):
            continue
        if cfg.get("training") != "image":
            continue
        steps = checkpoints.list_steps(name)
        if not steps:
            continue
        ta = cfg.get("training_args") or {}
        out.append({"name": name, "steps": steps, "height": ta.get("height"), "width": ta.get("width"),
                    "codec": ta.get("codec")})
    return out


def _resident_model(name, step, device):
    from veritate_core.plugin import image_sample
    with _MODEL_LOCK:
        key = (name, int(step) if step else None)
        hit = _MODEL_CACHE.get(key)
        if hit is None:
            model, codec, geometry, real_step = image_sample.load_image_model(name, step)
            model.to(device)
            hit = (model, codec, geometry, real_step)
            _MODEL_CACHE.clear()
            _MODEL_CACHE[key] = hit
        return hit


def _decode_source(value):
    if not value:
        return None
    text = str(value)
    if "," in text and text.strip().startswith("data:"):
        text = text.split(",", 1)[1]
    raw = base64.b64decode(text, validate=False)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("source image over " + str(MAX_SOURCE_BYTES // (1024 * 1024)) + " MB")
    return raw


def generate_image(body):
    """Validate and run one generation. Returns (payload, http_status)."""
    from veritate_core.plugin import hardware, image_sample
    name = str(body.get("model") or "").strip()
    if not name:
        return {"ok": False, "error": "missing 'model'"}, 400
    mode = str(body.get("mode") or "text").strip()
    if mode not in image_sample.MODES:
        return {"ok": False, "error": "mode must be one of " + ", ".join(image_sample.MODES)}, 400
    try:
        source = _decode_source(body.get("image"))
    except (ValueError, TypeError) as e:
        return {"ok": False, "error": "bad source image: " + str(e)}, 400
    if mode in image_sample.SOURCE_MODES and source is None:
        return {"ok": False, "error": "mode " + mode + " needs a source image"}, 400
    rect = body.get("rect")
    if mode == "inpaint" and not (isinstance(rect, list | tuple) and len(rect) == 4):
        return {"ok": False, "error": "inpaint needs rect [x0, y0, x1, y1] as fractions"}, 400
    device = hardware.pick_device()
    started = time.time()
    try:
        model, codec, geometry, step = _resident_model(name, body.get("step"), device)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        return {"ok": False, "error": type(e).__name__ + ": " + str(e)}, 400
    png, info = image_sample.generate(
        model, codec, geometry, mode=mode, caption=str(body.get("caption") or ""), source=source,
        strength=float(body.get("strength", image_sample.DEFAULT_STRENGTH)),
        rect=tuple(rect) if rect else None,
        expand=float(body.get("expand", image_sample.DEFAULT_EXPAND)),
        passes=int(body.get("passes", image_sample.DEFAULT_PASSES)),
        temperature=float(body.get("temperature", 1.0)), seed=int(body.get("seed", 0)), device=device)
    info.update({"ok": True, "model": name, "step": step, "device": device,
                 "seconds": round(time.time() - started, 3),
                 "png": base64.b64encode(png).decode("ascii")})
    return info, 200


def caption_state():
    with _CAPTION_LOCK:
        return {k: v for k, v in _CAPTION.items() if k != "stop"}


def caption_options():
    """Everything the captions block needs to render: styles, providers, the configured
    teacher, and the job state."""
    from runtime import settings as settings_mod
    from tools import caption_images
    try:
        from teacher import providers as teacher_providers
    except ImportError:
        from veritate_mri.teacher import providers as teacher_providers
    s = settings_mod.get()
    provs = sorted(teacher_providers.PROVIDERS.values(), key=lambda p: (0 if p["kind"] == "api" else 1, p["id"]))
    return {"ok": True,
            "styles": [{"id": k, "label": v["label"], "prompt": v["prompt"]} for k, v in caption_images.STYLES.items()],
            "providers": [{"id": p["id"], "name": p.get("display_name", p["id"]), "kind": p["kind"]} for p in provs],
            "current": {"provider": s.get("teacher_provider") or "", "model": s.get("teacher_model") or ""},
            "defaults": {"max_words": caption_images.DEFAULT_MAX_WORDS, "max_edge": caption_images.DEFAULT_MAX_EDGE,
                         "concurrency": caption_images.DEFAULT_CONCURRENCY, "style": caption_images.DEFAULT_STYLE},
            "state": caption_state()}


def _caption_args(body):
    from tools import caption_images
    set_name = str(body.get("set") or "").strip()
    if not SET_NAME.match(set_name) or not os.path.isdir(paths_image_set_dir(set_name)):
        raise ValueError("pick a set of pictures first")
    style = str(body.get("style") or caption_images.DEFAULT_STYLE)
    prompt = str(body.get("prompt") or "")
    caption_images.prompt_for(style, prompt, int(body.get("max_words") or caption_images.DEFAULT_MAX_WORDS))
    return {"set_name": set_name, "provider": (body.get("provider") or None), "model": (body.get("model") or None),
            "style": style, "prompt": prompt or None,
            "max_words": int(body.get("max_words") or caption_images.DEFAULT_MAX_WORDS),
            "max_edge": int(body.get("max_edge") or caption_images.DEFAULT_MAX_EDGE),
            "concurrency": int(body.get("concurrency") or caption_images.DEFAULT_CONCURRENCY),
            "overwrite": bool(body.get("overwrite", False)), "limit": int(body.get("limit") or 0)}


def paths_image_set_dir(set_name):
    from readers import paths
    return paths.image_set_dir(set_name)


def preview_caption(body):
    """Caption one picture now, so the prompt and model are judged before the set is
    spent on them. Returns the caption and a small thumbnail."""
    from tools import caption_images

    from veritate_core.plugin import get_teacher_client
    try:
        args = _caption_args(body)
    except ValueError as e:
        return {"ok": False, "error": str(e)}, 400
    _set_dir, images = caption_images.set_images(args["set_name"])
    if not images:
        return {"ok": False, "error": "the set holds no pictures"}, 400
    wanted = str(body.get("name") or "")
    path = next((p for p in images if os.path.basename(p) == wanted), None) if wanted else None
    if path is None:
        path = next((p for p in images if not os.path.isfile(caption_images.sidecar_for(p))), images[0])
    client = get_teacher_client(args["provider"], args["model"])
    if client is None:
        return {"ok": False, "error": "no teacher configured: pick a provider and a vision model"}, 400
    started = time.time()
    prompt = caption_images.prompt_for(args["style"], args["prompt"], args["max_words"])
    try:
        caption = caption_images.caption_one(client, path, prompt, args["max_edge"], args["max_words"])
    except Exception as e:
        return {"ok": False, "error": type(e).__name__ + ": " + str(e), "name": os.path.basename(path)}, 502
    return {"ok": True, "name": os.path.basename(path), "caption": caption, "prompt": prompt,
            "model": getattr(client, "model", args["model"]), "seconds": round(time.time() - started, 2),
            "thumbnail": "data:image/jpeg;base64," + caption_images.prepare_image(path, 256)}, 200


def _run_caption(args):
    from tools import caption_images

    def progress(done, total, name, caption):
        with _CAPTION_LOCK:
            _CAPTION["done"] = done
            _CAPTION["total"] = total
            if caption is None:
                _CAPTION["failed"] += 1
            else:
                _CAPTION["samples"] = (_CAPTION["samples"] + [{"name": name, "caption": caption}])[-5:]

    try:
        report = caption_images.caption_set(progress=progress, should_stop=lambda: _CAPTION["stop"], **args)
        with _CAPTION_LOCK:
            _CAPTION.update(status=CAPTION_STOPPED if report["stopped"] else CAPTION_OK, report=report,
                            finished_at=time.time())
    except Exception as e:
        with _CAPTION_LOCK:
            _CAPTION.update(status=CAPTION_FAILED, error=type(e).__name__ + ": " + str(e), finished_at=time.time())


def start_caption(body):
    try:
        args = _caption_args(body)
    except ValueError as e:
        return {"ok": False, "error": str(e)}, 400
    with _CAPTION_LOCK:
        if _CAPTION["status"] == CAPTION_RUNNING:
            return {"ok": False, "error": "a captioning run is already going (set " + str(_CAPTION["set"]) + ")"}, 409
        _CAPTION.update(status=CAPTION_RUNNING, set=args["set_name"], total=0, done=0, failed=0, samples=[],
                        report=None, error=None, started_at=time.time(), finished_at=None, stop=False)
    threading.Thread(target=_run_caption, name="images:caption", daemon=True, args=(args,)).start()
    return {"ok": True, "state": caption_state()}, 200


def stop_caption():
    with _CAPTION_LOCK:
        _CAPTION["stop"] = True
    return {"ok": True, "state": caption_state()}


def mri_payload(name):
    from readers import checkpoints, paths

    from veritate_core.plugin import image_probe
    with open(paths.config_path(name), encoding="utf-8") as handle:
        cfg = json.load(handle)
    ta = cfg.get("training_args") or {}
    rows = image_probe.read(name)
    return {"ok": True, "model": name, "training": cfg.get("training"),
            "geometry": {k: ta.get(k) for k in ("height", "width", "patch", "planes", "image_code_bytes", "seq")},
            "codec": ta.get("codec"), "image_set": ta.get("image_set"),
            "checkpoint_steps": checkpoints.list_steps(name), "steps": rows}


def prompt_state():
    with _PROMPT_LOCK:
        return dict(_PROMPT)


def _pick_steps(all_steps, wanted, limit=PROMPT_MAX_STEPS):
    """The checkpoints to draw at: the asked ones, else all, thinned evenly to `limit`
    with the last kept."""
    steps = sorted({int(x) for x in wanted}) if wanted else list(all_steps)
    steps = [x for x in steps if x in set(all_steps)]
    if len(steps) <= limit:
        return steps
    idx = [round(i * (len(steps) - 1) / (limit - 1)) for i in range(limit)]
    return [steps[i] for i in sorted(set(idx))]


def _run_prompt(name, steps, caption, seed, passes, mode, source, strength):
    from veritate_core.plugin import hardware, image_sample
    device = hardware.pick_device()
    try:
        for step in steps:
            if _PROMPT["stop"]:
                break
            t0 = time.time()
            model, codec, geometry, real_step = _resident_model(name, step, device)
            with_words, _ = image_sample.generate_codes(
                model, codec, geometry, mode=mode, caption=caption, source=source, strength=strength,
                passes=passes, seed=seed, device=device)
            without, _ = image_sample.generate_codes(
                model, codec, geometry, mode=mode, caption="", source=source, strength=strength,
                passes=passes, seed=seed, device=device)
            steering = float(np.mean(np.asarray(with_words) != np.asarray(without))) if caption else 0.0
            h, w = geometry["height"], geometry["width"]
            result = {"step": int(real_step),
                      "png": base64.b64encode(image_sample.decode_png(codec, with_words, h, w)).decode("ascii"),
                      "uncond_png": base64.b64encode(image_sample.decode_png(codec, without, h, w)).decode("ascii"),
                      "steering": steering, "seconds": round(time.time() - t0, 2)}
            with _PROMPT_LOCK:
                _PROMPT["results"] = _PROMPT["results"] + [result]
        with _PROMPT_LOCK:
            _PROMPT.update(status=PROMPT_STOPPED if _PROMPT["stop"] else PROMPT_OK, finished_at=time.time())
    except Exception as e:
        with _PROMPT_LOCK:
            _PROMPT.update(status=PROMPT_FAILED, error=type(e).__name__ + ": " + str(e), finished_at=time.time())


def start_prompt(name, body):
    """Draw the same words (and optional photo) at several checkpoints of one image model, in
    the background. Each result carries the picture with the words and the same-seed picture
    without them; `steering` is the share of cells the words changed."""
    from readers import checkpoints

    from veritate_core.plugin import image_sample
    all_steps = checkpoints.list_steps(name)
    if not all_steps:
        return {"ok": False, "error": "no checkpoint for " + name}, 400
    caption = str(body.get("caption") or "").strip()
    mode = str(body.get("mode") or "text").strip()
    if mode not in ("text", "variation"):
        return {"ok": False, "error": "mode must be text or variation"}, 400
    try:
        source = _decode_source(body.get("image"))
    except (ValueError, TypeError) as e:
        return {"ok": False, "error": "bad source image: " + str(e)}, 400
    if mode == "variation" and source is None:
        return {"ok": False, "error": "variation needs a photo"}, 400
    steps = _pick_steps(all_steps, body.get("steps") or [])
    if not steps:
        return {"ok": False, "error": "none of the asked steps has a checkpoint"}, 400
    with _PROMPT_LOCK:
        if _PROMPT["status"] == PROMPT_RUNNING:
            return {"ok": False, "error": "a prompt run is already going"}, 409
        passes = max(1, min(int(body.get("passes", image_sample.DEFAULT_PASSES)), image_sample.MAX_PASSES))
        _PROMPT.update(status=PROMPT_RUNNING, model=name, caption=caption, seed=int(body.get("seed", 0)),
                       passes=passes, mode=mode, steps=steps, results=[], error=None, stop=False,
                       started_at=time.time(), finished_at=None)
        args = (name, steps, caption, _PROMPT["seed"], _PROMPT["passes"], mode, source,
                float(body.get("strength", image_sample.DEFAULT_STRENGTH)))
    threading.Thread(target=_run_prompt, name="images:prompt", daemon=True, args=args).start()
    return {"ok": True, "state": prompt_state()}, 200


def stop_prompt():
    with _PROMPT_LOCK:
        _PROMPT["stop"] = True
    return {"ok": True, "state": prompt_state()}


def _running_run_name(run):
    """The model dir the runner's current args resolve to, or None."""
    args = (run or {}).get("args") or {}
    if args.get("resume"):
        return str(args["resume"])
    try:
        from training import save
        return save.compose_name(args.get("name") or "", args.get("size") or "")
    except (ValueError, ImportError):
        return None


def _log_tail(n):
    from training import trainer_runner
    try:
        with open(trainer_runner.RUN_LOG_FILE, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    return [ln for ln in lines if ln.strip()][-n:]


def live_payload(name):
    """What the Training tab shows for an image run: the stage the trainer is in
    (progress.json, written from the first second), the device, the latest probe, the
    checkpoints on disk and the tail of the run log. Works before config.json exists,
    which is most of a first run's wall clock."""
    from readers import checkpoints, paths
    from training import trainer_runner

    from veritate_core.plugin import image_probe, image_progress
    model_dir = paths.model_dir(name)
    progress = image_progress.read(model_dir)
    cfg = None
    if os.path.isfile(paths.config_path(name)):
        with open(paths.config_path(name), encoding="utf-8") as handle:
            cfg = json.load(handle)
    ta = (cfg or {}).get("training_args") or {}
    steps = image_probe.read(name)
    run = trainer_runner.state()
    this_run = _running_run_name(run) == name and run.get("plugin_id") == trainers_reader.IMAGE_TRAINER_ID
    ckpt_steps = checkpoints.list_steps(name)
    last_ckpt_at = None
    if ckpt_steps:
        try:
            last_ckpt_at = os.path.getmtime(paths.checkpoint_path(name, ckpt_steps[-1]))
        except OSError:
            last_ckpt_at = None
    return {"ok": True, "model": name, "progress": progress,
            "training": (cfg or {}).get("training") or ("image" if progress else None),
            "geometry": {k: ta.get(k) for k in ("height", "width", "patch", "planes", "image_code_bytes", "seq")},
            "image_set": ta.get("image_set"), "codec": ta.get("codec"),
            "checkpoint_steps": ckpt_steps, "last_checkpoint_at": last_ckpt_at,
            "latest_probe": steps[-1] if steps else None, "probe_steps": len(steps),
            "running": bool(this_run and run.get("status") == trainer_runner.STATUS_RUNNING),
            "run_status": run.get("status") if this_run else None,
            "log_tail": _log_tail(LOG_TAIL_LINES) if this_run else []}


def register(app):
    @app.route("/images/mri/<path:name>")
    def images_mri(name):
        from readers import models
        if not safe_name(name) or not models.exists(name):
            return {"ok": False, "error": "no such model"}, 404
        return _safe("images", lambda: mri_payload(name))

    @app.route("/images/mri/<path:name>/prompt", methods=["POST"])
    def images_mri_prompt(name):
        from readers import models
        if not safe_name(name) or not models.exists(name):
            return {"ok": False, "error": "no such model"}, 404
        return _safe("images", lambda: start_prompt(name, request.get_json(silent=True) or {}))

    @app.route("/images/mri/<path:name>/prompt/status")
    def images_mri_prompt_status(name):
        st = prompt_state()
        if st.get("model") != name:
            return {"ok": True, "state": {"status": PROMPT_IDLE, "model": name, "results": []}}
        return {"ok": True, "state": st}

    @app.route("/images/mri/<path:name>/prompt/stop", methods=["POST"])
    def images_mri_prompt_stop(_name):
        return _safe("images", stop_prompt)

    @app.route("/images/live/<path:name>")
    def images_live(name):
        from readers import paths
        if not safe_name(name) or not os.path.isdir(paths.model_dir(name)):
            return {"ok": False, "error": "no such run"}, 404
        return _safe("images", lambda: live_payload(name))

    @app.route("/images/mri/<path:name>/<int:step>/<string:filename>")
    def images_mri_file(name, step, filename):
        from readers import models, paths

        from veritate_core.plugin import image_probe
        if not safe_name(name) or not models.exists(name) or filename not in image_probe.FILES:
            return {"ok": False, "error": "no such file"}, 404
        directory = os.path.join(paths.hook_step_dir(name, step), image_probe.IMAGE_DIR)
        if not os.path.isfile(os.path.join(directory, filename)):
            return {"ok": False, "error": "not written at this step"}, 404
        return send_from_directory(directory, filename, max_age=0)

    @app.route("/images/caption/options")
    def images_caption_options():
        return _safe("images", caption_options)

    @app.route("/images/caption/preview", methods=["POST"])
    def images_caption_preview():
        return _safe("images", lambda: preview_caption(request.get_json(silent=True) or {}))

    @app.route("/images/caption", methods=["POST"])
    def images_caption():
        """Caption every uncaptioned picture in a set. Body: {set, provider?, model?, style?,
        prompt?, max_words?, max_edge?, concurrency?, overwrite?, limit?}."""
        return _safe("images", lambda: start_caption(request.get_json(silent=True) or {}))

    @app.route("/images/caption/status")
    def images_caption_status():
        return _safe("images", lambda: {"ok": True, **caption_state()})

    @app.route("/images/caption/stop", methods=["POST"])
    def images_caption_stop():
        return _safe("images", stop_caption)

    @app.route("/images/models")
    def images_models():
        """Image models on this machine that can generate."""
        return _safe("images", lambda: {"ok": True, "models": list_image_models()})

    @app.route("/images/generate", methods=["POST"])
    def images_generate():
        """Generate a picture. Body: {model, step?, mode, caption?, image? (base64),
        strength?, rect?, expand?, passes?, temperature?, seed?}. Returns a PNG, base64."""
        return _safe("images", lambda: generate_image(request.get_json(silent=True) or {}))

    @app.route("/images/pick_folder", methods=["POST"])
    def images_pick_folder():
        """Open the OS folder chooser here and return the chosen path."""
        return _safe("images", pick_folder)

    @app.route("/images/sets")
    def images_sets():
        """The picture sets and fitted codecs on this machine, for the Images flow."""
        return _safe("images", lambda: {"ok": True, "sets": images_reader.list_sets(),
                                        "codecs": images_reader.list_codecs()})

    @app.route("/images/ingest", methods=["POST"])
    def images_ingest():
        """Collect photos from folders on this machine into data/images/<set>/.
        Body: {set, sources: [path, ...], min_edge?, caption_from_folder?, copy?}."""
        return _safe("images", lambda: start_ingest(request.get_json(silent=True) or {}))

    @app.route("/images/ingest/status")
    def images_ingest_status():
        return _safe("images", lambda: {"ok": True, **ingest_state()})

    @app.route("/images/decode_bench", methods=["POST"])
    def images_decode_bench():
        """F0 for IDEA 24. Decode one frame per arm at random weights and report
        latency, achieved GF/s and peak activation bytes. The peak is the deciding
        number: the design's rule is that no tensor whose extent is the output
        resolution may be materialized, and the `conv_full` arm is the control that
        breaks it. No weights are loaded and nothing is saved."""
        def _do():
            from veritate_core.plugin import image_decode

            body = request.get_json(silent=True) or {}
            height = int(body.get("height", 1080))
            width  = int(body.get("width", 1920))
            if not (0 < height <= MAX_EDGE and 0 < width <= MAX_EDGE):
                return {"ok": False, "error": "height and width must be in 1.." + str(MAX_EDGE)}, 400
            kwargs = {k: body[k] for k in
                      ("arms", "latent_ch", "mlp_width", "tile", "patch", "code_emb",
                       "patch_hidden", "band", "conv_ch", "grid_div", "warmup", "reps",
                       "device", "seed") if k in body}
            if "arms" in kwargs:
                kwargs["arms"] = tuple(kwargs["arms"])
            if "conv_ch" in kwargs:
                kwargs["conv_ch"] = tuple(kwargs["conv_ch"])
            return {"ok": True, "report": image_decode.bench(height, width, **kwargs)}
        return _safe("images", _do)
