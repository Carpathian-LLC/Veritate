# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sleep controller (IDEA 20 T3): when the box is idle, consolidate the model's
#   own recorded exchanges (data/experience/) into its weights with a short
#   low-LR run, launched through the one trainer via trainer_runner. Disabled
#   until `sleep_enabled` is on and `sleep_model` names a model.
# - dose scales with use: steps = clamp(sleep_min_steps,
#   exchanges * sleep_steps_per_exchange, sleep_max_steps). No new exchanges
#   since the last sleep -> no run. Idle signal is the experience log mtime;
#   a running trainer always blocks sleep.
# - sleep runs checkpoint every `sleep_ckpt_every` steps so waking early still
#   keeps most of the consolidation. When a sleep run ends (completed or woken),
#   intermediate sleep checkpoints are deleted and only the run's final survives;
#   older sleep finals are thinned to `sleep_keep_finals`. Checkpoints not
#   created by sleep are never touched.
# - recipe is read from the model's own config.json training_args and only the
#   sleep levers are overridden, so the controller is size- and shape-agnostic.
# - status() feeds the Generation-tab sleep panel: state, countdown to sleep,
#   pending exchanges, and (while sleeping) step / eta / last checkpoint.
# - wake() is explicit (button / route). tick() is the watcher entry point,
#   called every WATCH_EVERY_S from the app daemon thread.
# veritate_mri/training/sleep.py
# ------------------------------------------------------------------------------------
# Imports:

import glob
import json
import os
import threading
import time

from readers.paths import EXPERIENCE_ROOT, MODELS_ROOT, SLEEP_ROOT
from runtime import logs as logmod
from runtime import settings as settings_mod

from training import trainer_runner

# ------------------------------------------------------------------------------------
# Constants

WATCH_EVERY_S = 60
STATE_PATH = os.path.join(SLEEP_ROOT, "state.json")
PLUGIN_ID = "native/trainer"
# sleep levers forced onto the model's own recipe; everything else resumes as trained
SLEEP_OVERRIDES = {
    "warmup_steps": 0, "lr_schedule": "wsd", "wsd_decay_frac": 0.1,
    "wsd_decay_kind": "sqrt", "loss_mask": "assistant", "resume": None,
    "name": None, "corpus": None, "total_steps": None, "ckpt_every": None,
    "eval_every": None, "base_lr": None, "min_lr": None, "description": None,
}

_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions


def _load_state():
    if not os.path.isfile(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(st):
    os.makedirs(SLEEP_ROOT, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE_PATH)


def _experience_files():
    return sorted(glob.glob(os.path.join(EXPERIENCE_ROOT, "*.jsonl")))


def _idle_s():
    files = _experience_files()
    if not files:
        return None
    return time.time() - max(os.path.getmtime(f) for f in files)


def _pending_exchanges(since_ts):
    """Exchanges recorded after the last sleep: line count of experience files
    modified since. Approximate on the boundary file; sleep-or-not decisions
    re-count exactly at corpus build time."""
    n = 0
    for f in _experience_files():
        if os.path.getmtime(f) <= since_ts:
            continue
        try:
            with open(f, "rb") as fh:
                n += sum(1 for _ in fh)
        except OSError:
            continue
    return n


def dose_steps(exchanges, cfg):
    """Usage-scaled sleep length: more conversation since the last sleep means a
    longer consolidation, clamped to [sleep_min_steps, sleep_max_steps]."""
    raw = exchanges * int(cfg["sleep_steps_per_exchange"])
    return max(int(cfg["sleep_min_steps"]), min(raw, int(cfg["sleep_max_steps"])))


def _model_step(model):
    cfg_path = os.path.join(MODELS_ROOT, model, "config.json")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return int(json.load(f).get("step") or 0)
    except (OSError, ValueError):
        return None


def launch_args(model, steps, cfg):
    """Sleep recipe = the model's own training_args with only the sleep levers
    overridden. Returns None when the model has no readable recipe."""
    cfg_path = os.path.join(MODELS_ROOT, model, "config.json")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            base = json.load(f).get("training_args") or {}
    except (OSError, ValueError):
        return None
    if not base:
        return None
    args = {k: v for k, v in base.items() if k not in SLEEP_OVERRIDES}
    args.update({k: v for k, v in SLEEP_OVERRIDES.items() if v is not None})
    lr = float(cfg["sleep_lr"])
    args.update({
        "name": model, "resume": model, "corpus": cfg["sleep_corpus"],
        "total_steps": int(steps), "ckpt_every": int(cfg["sleep_ckpt_every"]),
        "eval_every": int(cfg["sleep_ckpt_every"]), "base_lr": lr, "min_lr": lr,
        "description": f"sleep consolidation: {steps} steps over own experience "
                       f"({cfg['sleep_corpus']}), constant lr {lr:g}",
    })
    return args


def _train_progress(model):
    """(absolute_step, elapsed_s_this_run) from the model's train.csv tail;
    (None, None) when unreadable. Columns: step,split,loss,lr,...,elapsed_s,..."""
    path = os.path.join(MODELS_ROOT, model, "train.csv")
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            rows = [r for r in f.read().decode("utf-8", "replace").splitlines() if r]
    except OSError:
        return None, None
    for row in reversed(rows):
        parts = row.split(",")
        if len(parts) >= 7 and parts[1] == "train":
            try:
                return int(parts[0]), float(parts[6])
            except ValueError:
                continue
    return None, None


def prune(model, start_step, end_step, keep_finals, finals):
    """Delete intermediate checkpoints a finished sleep run left behind
    (start_step < N < end_step), then thin recorded sleep finals beyond
    keep_finals. Only steps recorded as sleep-created are ever deleted; the
    newest checkpoint always survives. Returns the surviving finals list."""
    ckpt_dir = os.path.join(MODELS_ROOT, model, "checkpoints")
    removed = []
    for path in glob.glob(os.path.join(ckpt_dir, "step_*.pt")):
        try:
            n = int(os.path.basename(path)[5:-3])
        except ValueError:
            continue
        if start_step < n < end_step:
            try:
                os.remove(path)
                removed.append(n)
            except OSError:
                pass
    finals = sorted(set(finals) | {end_step})
    while len(finals) > max(1, int(keep_finals)):
        victim = finals.pop(0)
        try:
            os.remove(os.path.join(ckpt_dir, f"step_{victim}.pt"))
            removed.append(victim)
        except OSError:
            pass
    if removed:
        logmod.ok("sleep", f"pruned {len(removed)} sleep checkpoint(s) of {model}: {sorted(removed)}")
    return finals


def status():
    """Sleep panel payload for the Generation tab."""
    cfg = settings_mod.get()
    st = _load_state()
    run = trainer_runner.state() or {}
    sleeping = bool(st.get("sleeping")) and run.get("status") == "running"
    out = {
        "enabled": bool(cfg.get("sleep_enabled")),
        "model": cfg.get("sleep_model") or "",
        "state": "sleeping" if sleeping else "awake",
        "last_sleep_ts": st.get("last_sleep_ts"),
        "finals": st.get("finals") or [],
    }
    idle = _idle_s()
    out["idle_s"] = None if idle is None else round(idle, 1)
    out["pending_exchanges"] = _pending_exchanges(float(st.get("last_sleep_ts") or 0))
    if sleeping:
        model = (st.get("run") or {}).get("model") or out["model"]
        total = (st.get("run") or {}).get("steps")
        step, elapsed = _train_progress(model)
        start = (st.get("run") or {}).get("start_step") or 0
        sps = elapsed / (step - start) if (step and elapsed and step > start) else None
        ck = max(1, int(cfg["sleep_ckpt_every"]))
        last_ck = (step // ck) * ck if step else start
        out["run"] = {
            "model": model, "step": step, "total_steps": total,
            "eta_s": round((total - step) * sps, 1) if (total and step is not None and sps) else None,
            "last_ckpt_step": last_ck if last_ck > start else start,
        }
    elif out["enabled"] and idle is not None:
        out["sleeps_in_s"] = max(0.0, round(int(cfg["sleep_idle_min"]) * 60 - idle, 1))
    return out


def maybe_sleep(force_idle=False):
    """Gate chain, then launch. Returns a short reason string for logs/tests.
    force_idle (the /sleep/now route) skips only the idle-time gate."""
    cfg = settings_mod.get()
    if not cfg.get("sleep_enabled"):
        return "disabled"
    model = (cfg.get("sleep_model") or "").strip()
    if not model:
        return "no sleep_model configured"
    if trainer_runner.is_running():
        return "trainer busy"
    idle = _idle_s()
    if idle is None:
        return "no experience yet"
    if not force_idle and idle < int(cfg["sleep_idle_min"]) * 60:
        return f"awake: last exchange {idle / 60:.1f} min ago"
    with _LOCK:
        st = _load_state()
        if st.get("sleeping"):
            return "already sleeping"
        from tools.build_experience_corpus import build
        n, tb, _vb = build(days=int(cfg["sleep_days"]))
        if n < int(cfg["sleep_min_exchanges"]):
            return f"too little new experience ({n} exchanges)"
        steps = dose_steps(n, cfg)
        start_step = _model_step(model)
        if start_step is None:
            return f"model {model} has no config"
        args = launch_args(model, start_step + steps, cfg)
        if args is None:
            return f"model {model} has no training_args recipe"
        res = trainer_runner.start(PLUGIN_ID, args)
        ok = isinstance(res, dict) and res.get("ok", True)
        if not ok:
            return f"launch failed: {res}"
        st.update({"sleeping": True,
                   "run": {"model": model, "start_step": start_step,
                           "steps": start_step + steps, "started_ts": time.time()}})
        _save_state(st)
        logmod.ok("sleep", f"{model} sleeping: {n} exchanges ({tb}B) -> {steps} steps")
        return f"sleeping: {steps} steps over {n} exchanges"


def wake():
    """Stop an in-flight sleep run. Serving resumes from the newest sleep
    checkpoint; finalize() reconciles checkpoints on the next tick."""
    with _LOCK:
        st = _load_state()
        if not st.get("sleeping"):
            return {"ok": True, "state": "awake", "note": "was not sleeping"}
        trainer_runner.stop()
        logmod.ok("sleep", f"woken by user: {(st.get('run') or {}).get('model')}")
        return {"ok": True, "state": "waking"}


def finalize():
    """Reconcile after a sleep run ends: record the sleep, prune checkpoints."""
    with _LOCK:
        st = _load_state()
        run = st.get("run") or {}
        model = run.get("model")
        if not model:
            st.pop("sleeping", None)
            _save_state(st)
            return
        end_step = _model_step(model) or run.get("start_step") or 0
        cfg = settings_mod.get()
        finals = st.get("finals") or []
        if end_step > (run.get("start_step") or 0):
            finals = prune(model, run.get("start_step") or 0, end_step,
                           cfg["sleep_keep_finals"], finals)
            st["last_sleep_ts"] = time.time()
        st.update({"sleeping": False, "finals": finals, "run": {}})
        _save_state(st)
        logmod.ok("sleep", f"{model} awake at step {end_step}")


def tick():
    """One watcher pass; safe to call every WATCH_EVERY_S."""
    st = _load_state()
    if st.get("sleeping"):
        run = trainer_runner.state() or {}
        if run.get("status") != "running":
            finalize()
        return "sleeping"
    return maybe_sleep()


def watcher():
    """Daemon loop for app startup."""
    while True:
        time.sleep(WATCH_EVERY_S)
        try:
            tick()
        except Exception as e:
            logmod.error("sleep", f"watcher: {e}")
