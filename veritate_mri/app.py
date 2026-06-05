# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - flask app for the live MRI. routes only. all on-disk reads go through readers/.
# - serves the static dashboard, exposes per-model artifacts, drives the two backends.
# veritate_mri/app.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import errno
import os
import sys
import threading
import time

from flask import Flask, request, send_from_directory
from werkzeug.serving import WSGIRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

from readers import checkpoints, config as cfg_reader, models
from runtime import logs as logmod
from runtime import lifecycle
from runtime import sys_metrics
from runtime import settings as settings_mod
from runtime import heartbeat as heartbeat_mod
from training import build_runner
from training import trainer_runner as plugin_runner
from training.sync import app_sync as app_sync_mod

# ------------------------------------------------------------------------------------
# Constants

STATIC_DIR = os.path.join(HERE, "web")

# Power-save mode. Set by `python veritate.py --minimal` (propagated via env so
# it survives the venv re-exec and any lifecycle restart). Disables brain
# eager-load, idle watcher, heartbeat/analytics, platform sync, and sys-metrics
# warm. Read-only training/log/sys-state routes still work — the user sees
# train.csv, log ring, and CPU/mem just fine.
MINIMAL = os.environ.get("VERITATE_MINIMAL") == "1"

from routes._common import auto_thread_count


app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
# Preserve JSON key insertion order in responses (Flask 2.2 default is True,
# Flask 3+ default is False; pin explicitly so plugin manifests render in the
# order their authors wrote them regardless of Flask version).
try:
    app.json.sort_keys = False
except AttributeError:
    app.config["JSON_SORT_KEYS"] = False
app.config["BRAIN"] = None
app.config["C_EXE"] = None
app.config["C_MODEL"] = None
app.config["C_SUBPROCESS"] = None
app.config["BRAIN_LAST_USED"] = 0.0

# ------------------------------------------------------------------------------------
# Functions

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/multimind")
def multimind_page():
    return send_from_directory(STATIC_DIR, "multimind.html")


@app.errorhandler(Exception)
def _route_exception_to_log(e):
    """Catch any uncaught exception in any route. Logs to the dashboard ring
    so users see it in the Logs tab, returns parseable JSON so the frontend's
    r.json() doesn't choke on Flask's HTML 500 page (WebKit reports that as
    'string did not match the expected pattern' which is opaque)."""
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        # Flask's own 404/405/etc. JSON-ify them too so the same contract
        # holds (no HTML reaching r.json()).
        logmod.warn("route", f"HTTP {e.code} {request.method} {request.path}")
        return ({"ok": False, "error": e.description or e.name, "status": e.code}, e.code)
    msg = f"{type(e).__name__}: {e}"
    logmod.error("route", f"{request.method} {request.path} -> {msg}")
    return ({"ok": False, "error": msg}, 500)


from routes._brain import (
    resolve_pytorch_model as _resolve_pytorch_model,
    load_pytorch_brain   as _load_pytorch_brain,
)


def _pytorch_idle_watcher():
    """Background daemon. When pytorch_load_mode == 'on_demand' and the brain
    has been idle longer than pytorch_idle_unload_secs, unload it. Skips while
    a generation/neuron lookup holds brain.lock so we never unload mid-stream."""
    while True:
        time.sleep(30)
        try:
            s = settings_mod.get()
            if s.get("pytorch_load_mode") != "on_demand":
                continue
            brain = app.config.get("BRAIN")
            if brain is None:
                continue
            if brain.lock.locked():
                continue
            idle_for = time.time() - (app.config.get("BRAIN_LAST_USED") or 0)
            if idle_for >= float(s.get("pytorch_idle_unload_secs") or 600):
                app.config["BRAIN"] = None
                app.config["BRAIN_MODEL"] = None
                app.config["BRAIN_STEP"]  = None
                logmod.ok("backends", f"pytorch auto-unloaded (idle {int(idle_for)}s)")
        except Exception as e:
            logmod.error("backends", f"idle watcher: {e}")


from routes import (
    atlas_routes, backends_routes, corpus_routes, engine_routes,
    lifecycle_routes, logs_routes, mesh_routes, models_routes, multimind_routes,
    trainers_routes, pruning_routes, runs_routes, settings_routes, sys_routes,
    teacher_routes, train_routes, wiki_routes,
)
atlas_routes.register(app)
backends_routes.register(app)
corpus_routes.register(app)
engine_routes.register(app)
lifecycle_routes.register(app)
logs_routes.register(app)
mesh_routes.register(app)
models_routes.register(app)
multimind_routes.register(app)
trainers_routes.register(app)
pruning_routes.register(app)
runs_routes.register(app)
settings_routes.register(app)
sys_routes.register(app)
teacher_routes.register(app)
train_routes.register(app)
wiki_routes.register(app)

_mesh_role = (settings_mod.get().get("mesh_role") or "off").lower()
if _mesh_role in ("hub", "both"):
    try:
        from veritate_mesh import hub as _mesh_hub
        _mesh_hub.register(app)
        _mesh_hub.start_workers()
    except ImportError as _e:
        logmod.error("mesh", f"hub import failed: {_e}")
if _mesh_role in ("node", "both"):
    try:
        from veritate_mesh import node as _mesh_node
        _mesh_node.register(app)
        _mesh_node.start_workers()
    except ImportError as _e:
        logmod.error("mesh", f"node import failed: {_e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="auto", help="default model for both backends. 'auto' picks the freshest.")
    ap.add_argument("--step",    type=int, default=None)
    ap.add_argument("--port",    type=int, default=8001)
    ap.add_argument("--threads", type=int, default=0,
                    help="pytorch CPU threads. 0 = auto: physical cores capped at 16.")
    args = ap.parse_args()

    name = _resolve_pytorch_model(args.model)
    if name is not None:
        app.config["DEFAULT_MODEL"] = name
        app.config["DEFAULT_STEP"]  = args.step if args.step is not None else checkpoints.latest_step(name)
    threads = args.threads if args.threads and args.threads > 0 else auto_thread_count()
    app.config["DEFAULT_THREADS"] = threads
    logmod.info("run", f"default model: {name or '(none)'}")
    logmod.info("run", f"pytorch threads: {threads}{' (auto)' if not args.threads else ''}")

    def _close_c_for_rebuild():
        sub = app.config.get("C_SUBPROCESS")
        if sub is None:
            return
        try:
            sub.close()
        except Exception:
            pass
        app.config["C_SUBPROCESS"] = None
        logmod.warn("build", "closed C engine subprocess to release binary lock")
    build_runner.set_pre_build_hook(_close_c_for_rebuild)

    if MINIMAL:
        logmod.info("run", "MINIMAL mode: skipping idle watcher, sys-warm, app-sync, brain eager-load (heartbeat still active)")
    else:
        threading.Thread(target=_pytorch_idle_watcher, name="pytorch-idle-watcher", daemon=True).start()
        sys_metrics.warm()

    def _enrich_with_config(out, name):
        """Attach model_name + n_params + shape summary to a training payload
        when the model has a config.json. Shared by the primary and fallback
        detectors so both ship the same envelope shape."""
        if not name or not models.exists(name):
            return
        try:
            cfg   = cfg_reader.load(name) or {}
            shape = cfg.get("shape") or {}
            out["model_name"] = name
            out["n_params"]   = int(cfg.get("n_params_total") or 0) or None
            keep = ("hidden", "layers", "ffn", "heads", "seq", "n_predict", "rope_base")
            summary = {k: shape[k] for k in keep if k in shape}
            if summary:
                out["shape"] = summary
        except Exception:
            pass

    def _detect_csv_based_training():
        """Fallback detector: any models/<name>/train.csv touched within the
        last CSV_ACTIVE_WINDOW seconds is treated as a live training run.
        Catches direct-script trainers (e.g. tools/coral/run_coral.py) that
        bypass plugin_runner. The canonical save.append_train_row() contract
        is what we rely on — any trainer that writes train.csv per the
        contract gets picked up automatically. Without this fallback, presence
        pings would falsely report "idle" for the entire duration of such a
        run and the server would flip the device offline mid-training."""
        CSV_ACTIVE_WINDOW = 120  # seconds (log_every=25 at ~1s/step => ~25s between writes)
        try:
            root = os.path.join(REPO_ROOT, "models")
            if not os.path.isdir(root):
                return None
        except OSError:
            return None
        now    = time.time()
        latest = None  # (mtime, name)
        try:
            for entry in os.listdir(root):
                csv_path = os.path.join(root, entry, "train.csv")
                try:
                    mt = os.path.getmtime(csv_path)
                except OSError:
                    continue
                if now - mt > CSV_ACTIVE_WINDOW:
                    continue
                if latest is None or mt > latest[0]:
                    latest = (mt, entry)
        except OSError:
            return None
        if latest is None:
            return None
        mt, name = latest
        out = {
            "plugin_id":  "direct-script",
            "started_at": int(mt),
        }
        _enrich_with_config(out, name)
        return out

    def _heartbeat_training():
        # Primary: trainer_runner-managed subprocess (plugins launched via the
        # dashboard's Training tab or plugin_runner.start). Enriched payload:
        # plugin id + started_at + model name + shape/params. The heartbeat
        # tier logic decides which fields actually ship (analytics tier: full;
        # minimal: only "training_active" presence).
        st = plugin_runner.state()
        if st and st.get("status") == plugin_runner.STATUS_RUNNING:
            out = {
                "plugin_id":  st.get("plugin_id"),
                "started_at": st.get("started_at"),
            }
            args_dict = st.get("args") or {}
            if isinstance(args_dict, dict):
                _enrich_with_config(out, args_dict.get("name") or args_dict.get("model"))
            return out
        # Fallback: direct-script training detection via recent train.csv mtime.
        return _detect_csv_based_training()
    heartbeat_mod.set_training_provider(_heartbeat_training)
    heartbeat_mod.start()

    def _app_sync_reload():
        lifecycle.restart(app.config)
    app_sync_mod.set_reload_hook(_app_sync_reload)
    if not MINIMAL:
        app_sync_mod.start()

    # Eager-load the pytorch backend OFF the main thread so app.run() starts
    # serving immediately. Only fires when settings say `always`; in the
    # default `on_demand` mode the brain loads when the user actually clicks
    # Generate, and idle-watcher unloads it after inactivity.
    if (not MINIMAL
            and settings_mod.get().get("pytorch_load_mode") == "always"
            and app.config.get("DEFAULT_MODEL") is not None
            and app.config.get("DEFAULT_STEP")  is not None):
        def _eager_load():
            try:
                app.config["PYTORCH_PENDING"] = True
                n, st = app.config["DEFAULT_MODEL"], app.config["DEFAULT_STEP"]
                brain, n2, st2 = _load_pytorch_brain(n, st, threads)
                app.config["BRAIN"] = brain
                app.config["BRAIN_MODEL"] = n2
                app.config["BRAIN_STEP"]  = int(st2)
                app.config["DEFAULT_MODEL"] = n2
                app.config["DEFAULT_STEP"]  = int(st2)
                app.config["BRAIN_LAST_USED"] = time.time()
                app.config["BRAIN_LAST_ERROR"] = None
                logmod.ok("backends", f"pytorch eager-loaded: {n2} step {st2} ({brain.n_params:,} params)")
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                app.config["BRAIN_LAST_ERROR"] = msg
                cur = app.config.get("DEFAULT_MODEL")
                if isinstance(e, RuntimeError) and "PyTorch inference is not enabled" in str(e):
                    logmod.warn("backends", f"pytorch backend skipped for {cur}: non-vanilla architecture (use C engine)")
                else:
                    logmod.error("backends", f"pytorch eager load failed: {msg}")
            finally:
                app.config["PYTORCH_PENDING"] = False
        threading.Thread(target=_eager_load, name="pytorch-eager-load", daemon=True).start()

    print(f"http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True,
            request_handler=_QuietWSGIRequestHandler)


class _QuietWSGIRequestHandler(WSGIRequestHandler):
    # Python 3.14 + werkzeug threaded dev server has a race where socketserver
    # closes the connection in one thread while the handler is still reading
    # from it in another, surfacing as OSError(EBADF) from recv_into. Browser
    # preconnect/keep-alive churn triggers this constantly. Werkzeug already
    # swallows ConnectionError/socket.timeout in handle(); EBADF is not in
    # either bucket, so we extend the same idea narrowly.
    def handle(self):
        try:
            super().handle()
        except OSError as e:
            if e.errno == errno.EBADF:
                return
            raise


if __name__ == "__main__":
    main()
