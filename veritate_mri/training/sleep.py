# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sleep controller (IDEA 20 T3): when the box is idle, an enrolled model
#   consolidates its OWN recorded exchanges (data/experience/) into its weights
#   with a short low-LR run, launched through the one trainer via trainer_runner.
#   Disabled until `sleep_enabled` is on and `sleep_models` enrolls at least one
#   model. Enrolled models take turns on the one trainer: each idle window the
#   model with the most pending own-exchanges above sleep_min_exchanges sleeps;
#   the next model waits for finalize().
# - own-conversations-only: experience records are attributed to a model dir via
#   _resolve() (exact dir name, or a uniquely-owned artifact basename for old
#   records); records owned by nobody are consolidated by nobody. The corpus
#   build feeds the builder a per-model filtered view of the log.
# - dose scales with use: steps = clamp(sleep_min_steps,
#   exchanges * sleep_steps_per_exchange, sleep_max_steps). No new exchanges
#   since the model's last sleep -> no run. Idle signal is the experience log
#   mtime (box-wide: serving is one engine); a running trainer always blocks.
# - sleep runs checkpoint every `sleep_ckpt_every` steps so waking early still
#   keeps most of the consolidation. When a sleep run ends (completed or woken),
#   intermediate sleep checkpoints are deleted and only the run's final survives;
#   older sleep finals are thinned to `sleep_keep_finals` per model. Checkpoints
#   not created by sleep are never touched.
# - recipe is read from the model's own config.json training_args and only the
#   sleep levers are overridden, so the controller is size- and shape-agnostic.
# - per-model state (last sleep, in-flight run, finals, cooldown) lives keyed by
#   model in data/sleep/state.json; history events carry the model name.
# - status() feeds the Generation-tab sleep panel: one row per enrolled model
#   (state, pending, last slept, next eligibility) plus the global run/history.
# - wake(model) is explicit (button / route). tick() is the watcher entry point,
#   called every WATCH_EVERY_S from the app daemon thread.
# veritate_mri/training/sleep.py
# ------------------------------------------------------------------------------------
# Imports:

import glob
import json
import os
import threading
import time
from bisect import bisect_right

from readers import bin as binr
from readers.paths import EXPERIENCE_ROOT, MODELS_ROOT, SLEEP_ROOT
from runtime import logs as logmod
from runtime import serving
from runtime import settings as settings_mod

from training import trainer_runner

# ------------------------------------------------------------------------------------
# Constants

WATCH_EVERY_S = 60
# while a sleep child sits suspended the watcher polls fast enough that the
# resume lands close to sleep_resume_s rather than a full watch period later.
WATCH_PAUSED_S = 2
STATE_PATH = os.path.join(SLEEP_ROOT, "state.json")
HISTORY_PATH = os.path.join(SLEEP_ROOT, "history.jsonl")
# per-model filtered view of the experience log fed to the corpus builder
FILTER_ROOT = os.path.join(SLEEP_ROOT, "filtered")
PLUGIN_ID = "native/trainer"
# sleep levers forced onto the model's own recipe; everything else resumes as trained
SLEEP_OVERRIDES = {
    "warmup_steps": 0, "lr_schedule": "wsd", "wsd_decay_frac": 0.1,
    "wsd_decay_kind": "sqrt", "loss_mask": "assistant", "resume": None,
    "name": None, "corpus": None, "total_steps": None, "ckpt_every": None,
    "eval_every": None, "base_lr": None, "min_lr": None, "description": None,
    # a pretrain recipe logs every 10-20 steps; a sleep dose can be shorter than
    # that, and an inherited interval writes no train.csv row at all, so the run
    # looks like nothing happened. Sleep is short by construction: log every step.
    "log_every": 1,
    # the checkpoint dump suite generates text in eager PyTorch to trend a
    # research run across many checkpoints. A sleep run's trend is its own val
    # loss, and every checkpoint but the last is deleted when the run ends, so
    # the suite is measuring artifacts it is about to throw away. It costs ~137 s
    # per checkpoint on a Mac Studio and stalled a whole sleep step on cardinal.
    "hooks": "off",
}
# save() stamps bookkeeping into training_args that are not trainer flags; the
# trainer's unknown-flag gate refuses a launch that forwards them (measured on
# cardinal 2026-08-20 — sleep could not launch anywhere until these were dropped)
SAVE_BOOKKEEPING = ("corpus_bytes", "corpus_sha256", "output_dir")
# after a sleep that gained no steps (failed launch or instant stop), hold off
# this long instead of letting the 60 s watcher retry-storm the trainer
FAIL_COOLDOWN_S = 3600

_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions


def enrolled(cfg):
    """Models opted into sleep, in settings order."""
    return [m.strip() for m in (cfg.get("sleep_models") or [])
            if isinstance(m, str) and m.strip()]


def _load_state():
    """data/sleep/state.json in the per-model shape {"models": {name: {...}}}."""
    st = {}
    if os.path.isfile(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                st = json.load(f)
            if not isinstance(st, dict):
                st = {}
        except (OSError, ValueError):
            st = {}
    if isinstance(st.get("models"), dict):
        return st
    # user-data compat: state written by the single-sleeper controller was flat;
    # move its bookkeeping (last_sleep_ts, finals, cooldown, in-flight run) under
    # the model it belonged to so past sleeps stay attributed
    owner = (st.get("run") or {}).get("model")
    if not owner:
        names = enrolled(settings_mod.get())
        owner = names[0] if len(names) == 1 else None
    ms = {k: st[k] for k in ("sleeping", "run", "last_sleep_ts",
                             "finals", "cooldown_until") if k in st}
    return {"models": {owner: ms}} if owner and ms else {"models": {}}


def _save_state(st):
    os.makedirs(SLEEP_ROOT, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE_PATH)


def _sleeper(st):
    """Name of the model whose sleep run is in flight, or None."""
    return next((n for n, ms in st["models"].items() if ms.get("sleeping")), None)


def _log_event(kind, **fields):
    """Append one line to the sleep history the Generation-tab panel reviews.
    State changes only (fell asleep / woken / awake), never watcher ticks."""
    try:
        os.makedirs(SLEEP_ROOT, exist_ok=True)
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "event": kind, **fields}) + "\n")
    except OSError:
        pass


def history(limit=12):
    """Most-recent-first tail of the sleep event log."""
    try:
        with open(HISTORY_PATH, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            rows = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    out = []
    for row in reversed(rows):
        try:
            ev = json.loads(row)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("event"):
            out.append(ev)
        if len(out) >= limit:
            break
    return out


_ACT_CACHE = {"key": None, "hours": None}


def activity(days=7):
    """Usage ledger: exchanges per local hour-of-day over the last `days` of
    experience. The panel shows it so a human can judge whether the sleep
    cycle actually lands in the quiet hours. Cached on file mtimes — status()
    is polled every 15 s and must not re-read the whole log each time."""
    files = _experience_files()
    key = (days, tuple((f, os.path.getmtime(f)) for f in files))
    if _ACT_CACHE["key"] == key:
        return _ACT_CACHE["hours"]
    cutoff = time.time() - days * 86400
    hours = [0] * 24
    for f in files:
        if os.path.getmtime(f) < cutoff:
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        ts = float(json.loads(line).get("ts") or 0)
                    except (ValueError, AttributeError):
                        continue
                    if ts >= cutoff:
                        hours[time.localtime(ts).tm_hour] += 1
        except OSError:
            continue
    _ACT_CACHE.update(key=key, hours=hours)
    return hours


def _experience_files():
    return sorted(glob.glob(os.path.join(EXPERIENCE_ROOT, "*.jsonl")))


def _idle_s():
    files = _experience_files()
    if not files:
        return None
    return time.time() - max(os.path.getmtime(f) for f in files)


def _model_dirs():
    try:
        return sorted(d for d in os.listdir(MODELS_ROOT)
                      if os.path.isdir(os.path.join(MODELS_ROOT, d)))
    except OSError:
        return []


def _owner_map(dirs):
    """Artifact basename -> owning models. user-data compat: experience records
    from before 2026-08-23 name the serving artifact ("veritate.bin" from the C
    engine, "step_N.pt" from pytorch) rather than the model dir; a basename only
    one model owns still attributes, one that several share attributes to
    nobody."""
    owners = {}
    for m in dirs:
        arts = (glob.glob(os.path.join(MODELS_ROOT, m, "*.bin"))
                + glob.glob(os.path.join(MODELS_ROOT, m, "checkpoints", "step_*.pt")))
        for path in arts:
            owners.setdefault(os.path.basename(path), set()).add(m)
    return owners


def _resolve(rec_model, dirs, owners):
    """Model dir a record's model field belongs to; "" = owned by nobody."""
    if not rec_model:
        return ""
    if rec_model in dirs:
        return rec_model
    own = owners.get(rec_model)
    return next(iter(own)) if own and len(own) == 1 else ""


_EXP_CACHE = {"key": None, "by_model": None}


def _exchange_ts():
    """Sorted exchange timestamps per resolved model over the whole experience
    log; unresolvable records land under "". Cached on file mtimes + the model
    list, like activity()."""
    files = _experience_files()
    dirs = _model_dirs()
    key = (tuple((f, os.path.getmtime(f)) for f in files), tuple(dirs))
    if _EXP_CACHE["key"] == key:
        return _EXP_CACHE["by_model"]
    owners = _owner_map(dirs)
    by_model = {}
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        ts = float(rec.get("ts") or 0)
                        m = str(rec.get("model") or "")
                    except (ValueError, AttributeError):
                        continue
                    by_model.setdefault(_resolve(m, dirs, owners), []).append(ts)
        except OSError:
            continue
    for v in by_model.values():
        v.sort()
    _EXP_CACHE.update(key=key, by_model=by_model)
    return by_model


def _pending(model, since_ts):
    """Exchanges the model itself served after its last sleep."""
    ts = _exchange_ts().get(model) or []
    return len(ts) - bisect_right(ts, since_ts)


def _recipe(model):
    """The model's own training_args, the recipe a sleep run inherits. None when
    there is nothing readable to inherit."""
    cfg_path = os.path.join(MODELS_ROOT, model, "config.json")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            base = json.load(f).get("training_args") or {}
    except (OSError, ValueError):
        return None
    return base or None


def _draw_window(base):
    """Contiguous bytes the trainer draws per sample under this recipe. A bin
    smaller than one draw crashes the child, so both bins must clear it."""
    return int(base.get("seq") or 1024) * int(base.get("n_chunks") or 1) + 2


def _fit_batch(base, cfg, train_bytes, ms=None):
    """Sleep batch sized to two bounds, whichever is tighter.

    The DATA bound: one step draws batch * (seq * n_chunks) bytes, and a pretrain
    recipe's batch is set against a corpus thousands of times larger than a night
    of conversation, so inheriting it re-reads the whole log dozens of times a
    step (cardinal 2026-08-24: batch 48 over 17 kB is 27 min a step against 3 min
    at the 4 the log can fill).

    The BOX bound: the model's last sleep on this machine measured what a step of
    a known batch cost, so the next one scales that to sleep_step_seconds. A box
    that talks enough for the data bound to reach the recipe's batch would
    otherwise go straight back to 27-minute steps. First sleep has no measurement
    and takes the data bound alone, which is the conservative direction.

    An explicit sleep_batch_size overrides both."""
    asked = int(cfg.get("sleep_batch_size", 0) or 0)
    if asked:
        return asked
    fit = min(int(base.get("batch_size") or 1), int(train_bytes) // _draw_window(base))
    step_s = float((ms or {}).get("step_s") or 0)
    prev = int((ms or {}).get("step_batch") or 0)
    if step_s > 0 and prev > 0:
        fit = min(fit, int(prev * float(cfg["sleep_step_seconds"]) / step_s))
    return max(1, fit)


def _fit_ckpt_every(cfg, ms):
    """Checkpoints per sleep run, in steps, bounded by wall clock. sleep_ckpt_every
    is a step count, and a step is seconds on a training box and minutes on a weak
    one: cardinal 2026-08-24 waking at step 12 lost everything past step 6 because
    25 steps is 69 minutes there. Once a model has slept here, the interval is
    whatever fits sleep_ckpt_seconds, capped by the setting so a fast box is
    unaffected."""
    every = max(1, int(cfg["sleep_ckpt_every"]))
    step_s = float((ms or {}).get("step_s") or 0)
    if step_s <= 0:
        return every
    return max(1, min(every, int(float(cfg["sleep_ckpt_seconds"]) / step_s)))


def _fit_eval_iters(base, batch, val_bytes):
    """Sleep eval passes sized to the val bin. Each pass draws batch * (seq *
    n_chunks) bytes, so a recipe's count re-measures a night's val split dozens
    of times for one number: cardinal 2026-08-24 spent 475 s, three times a
    training step, reading 5.4 kB sixty-four times. Never below one pass."""
    window = _draw_window(base) * max(1, int(batch))
    return max(1, min(int(base.get("eval_iters") or 1), int(val_bytes) // window))


def _build_own_corpus(model, cfg, min_val_bytes=0):
    """Build the consolidation bins from the model's own exchanges only: write a
    per-model filtered view of the experience log and run the corpus builder
    over it. The builder reads a module-global root and has no source or model
    parameter yet; repointing that root at the filtered view is the seam — when
    the builder grows a filter/extraction mode, this function (plus the reserved
    sleep_use_extraction setting) is the one place to switch."""
    import tools.build_experience_corpus as bec
    files = _experience_files()
    days = int(cfg["sleep_days"])
    if days:
        files = files[-days:]
    dirs = _model_dirs()
    owners = _owner_map(dirs)
    os.makedirs(FILTER_ROOT, exist_ok=True)
    for stale in glob.glob(os.path.join(FILTER_ROOT, "*.jsonl")):
        os.remove(stale)
    for path in files:
        kept = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        m = str(json.loads(line).get("model") or "")
                    except (ValueError, AttributeError):
                        continue
                    if _resolve(m, dirs, owners) == model:
                        kept.append(line if line.endswith("\n") else line + "\n")
        except OSError:
            continue
        if kept:
            with open(os.path.join(FILTER_ROOT, os.path.basename(path)),
                      "w", encoding="utf-8") as out:
                out.writelines(kept)
    prev = bec.EXPERIENCE_ROOT
    bec.EXPERIENCE_ROOT = FILTER_ROOT
    try:
        return bec.build(days=None, min_val_bytes=min_val_bytes)
    finally:
        bec.EXPERIENCE_ROOT = prev


def dose_steps(exchanges, cfg):
    """Usage-scaled sleep length: more conversation since the model's last sleep
    means a longer consolidation, clamped to [sleep_min_steps, sleep_max_steps]."""
    raw = exchanges * int(cfg["sleep_steps_per_exchange"])
    return max(int(cfg["sleep_min_steps"]), min(raw, int(cfg["sleep_max_steps"])))


def _model_step(model):
    """Step the trainer will actually resume from: the LATEST checkpoint on
    disk, not config.json's "step" (a forked model's config can say 0 while
    checkpoints sit at 3000 — dosing from config would end the sleep
    instantly). None when there is no checkpoint to resume: real
    consolidation is impossible without a .pt (cardinal finding)."""
    steps = []
    for path in glob.glob(os.path.join(MODELS_ROOT, model, "checkpoints", "step_*.pt")):
        try:
            steps.append(int(os.path.basename(path)[5:-3]))
        except ValueError:
            continue
    return max(steps) if steps else None


def launch_args(model, steps, cfg, train_bytes, val_bytes, ms=None):
    """Sleep recipe = the model's own training_args with only the sleep levers
    overridden. Returns None when the model has no readable recipe."""
    base = _recipe(model)
    if base is None:
        return None
    args = {k: v for k, v in base.items()
            if k not in SLEEP_OVERRIDES and k not in SAVE_BOOKKEEPING}
    args.update({k: v for k, v in SLEEP_OVERRIDES.items() if v is not None})
    lr = float(cfg["sleep_lr"])
    args.update({
        "name": model, "resume": model, "corpus": cfg["sleep_corpus"],
        "total_steps": int(steps), "ckpt_every": _fit_ckpt_every(cfg, ms),
        "eval_every": _fit_ckpt_every(cfg, ms), "base_lr": lr, "min_lr": lr,
        "description": f"sleep consolidation: {steps} steps over own experience "
                       f"({cfg['sleep_corpus']}), constant lr {lr:g}",
        # run modifiers, stripped before argv (trainer_runner._build_argv)
        "_cpu_budget": cpu_budget(cfg),
        "_nice": int(cfg.get("sleep_nice", 10)),
    })
    args["batch_size"] = _fit_batch(base, cfg, train_bytes, ms)
    args["eval_iters"] = _fit_eval_iters(base, args["batch_size"], val_bytes)
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


def _val_trend(model, start_step):
    """(first, last) val loss this run logged above start_step, or (None, None)
    when it logged fewer than two.

    Within one run the comparison is sound: every val row is scored on the same
    bin. ACROSS runs it is not, because each sleep rebuilds its bins from the
    experience log as it stands at launch, so run N and run N+1 measure different
    held-out conversation and their numbers are not comparable."""
    path = os.path.join(MODELS_ROOT, model, "train.csv")
    try:
        with open(path, encoding="utf-8") as f:
            rows = [r.split(",") for r in f.read().splitlines() if r]
    except OSError:
        return None, None
    vals = []
    for parts in rows:
        if len(parts) < 3 or parts[1] != "val":
            continue
        try:
            if int(parts[0]) > start_step:
                vals.append(float(parts[2]))
        except ValueError:
            continue
    return (vals[0], vals[-1]) if len(vals) >= 2 else (None, None)


def regressed(model, start_step, cfg):
    """(held, first, last): whether this run's consolidation made the model worse
    on its own held-out conversation by more than sleep_val_tolerance.

    The gate is a guardrail against collapse, not a quality optimizer: publishing
    a self-trained checkpoint feeds the served model its own next training set,
    so a run that walks val the wrong way must not reach serving. A run that
    logged fewer than two val rows cannot be judged and is not held."""
    first, last = _val_trend(model, start_step)
    if first is None or first <= 0:
        return False, first, last
    return last > first * (1.0 + float(cfg["sleep_val_tolerance"])), first, last


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
    """Sleep panel payload for the Generation tab: one row per enrolled model
    plus the global run / history / activity ledger."""
    cfg = settings_mod.get()
    st = _load_state()
    run_state = trainer_runner.state() or {}
    sleeper = _sleeper(st)
    sleeping = sleeper is not None and run_state.get("status") == "running"
    idle = _idle_s()
    now = time.time()
    out = {
        "enabled": bool(cfg.get("sleep_enabled")),
        "state": "sleeping" if sleeping else "awake",
        # a sleeping model parked mid-run because a request is being served
        "suspended": bool(sleeping and suspended()),
        "idle_s": None if idle is None else round(idle, 1),
        "models": [],
        "history": history(),
        "activity_by_hour": activity(),
        "activity_days": 7,
    }
    idle_left = None
    if out["enabled"] and idle is not None:
        idle_left = max(0.0, int(cfg["sleep_idle_min"]) * 60 - idle)
    for name in enrolled(cfg):
        ms = st["models"].get(name) or {}
        row = {
            "name": name,
            "state": "sleeping" if (sleeping and name == sleeper) else "awake",
            "pending_exchanges": _pending(name, float(ms.get("last_sleep_ts") or 0)),
            "last_sleep_ts": ms.get("last_sleep_ts"),
            "finals": ms.get("finals") or [],
        }
        cool = float(ms.get("cooldown_until") or 0)
        if cool > now:
            row["cooldown_s"] = round(cool - now, 1)
        if row["state"] == "awake" and not sleeping and idle_left is not None:
            row["sleeps_in_s"] = round(max(idle_left, cool - now if cool > now else 0.0), 1)
        out["models"].append(row)
    if sleeping:
        run = (st["models"].get(sleeper) or {}).get("run") or {}
        total = run.get("steps")
        step, elapsed = _train_progress(sleeper)
        start = run.get("start_step") or 0
        sps = elapsed / (step - start) if (step and elapsed and step > start) else None
        ck = max(1, int(cfg["sleep_ckpt_every"]))
        last_ck = (step // ck) * ck if step else start
        out["run"] = {
            "model": sleeper, "step": step, "total_steps": total,
            "eta_s": round((total - step) * sps, 1) if (total and step is not None and sps) else None,
            "last_ckpt_step": last_ck if last_ck > start else start,
        }
    return out


def _launch(st, model, cfg):
    """Build the model's own corpus and launch its sleep run. Returns
    (launched, reason); the caller holds _LOCK."""
    start_step = _model_step(model)
    if start_step is None:
        return False, f"model {model} has no checkpoint to resume"
    base = _recipe(model)
    if base is None:
        return False, f"model {model} has no training_args recipe"
    # both bins must clear one draw; build to that floor before gating on it, or
    # the val split starves on a small night (cardinal: 183 B val vs 4098 B draw)
    window = _draw_window(base)
    n, tb, vb = _build_own_corpus(model, cfg, min_val_bytes=window)
    if n < int(cfg["sleep_min_exchanges"]):
        return False, f"too little new experience ({n} exchanges)"
    if tb < window or vb < window:
        return False, (f"experience bins too small for draw window ({tb}/{vb} B < {window} B): "
                       f"{model} needs about {2 * window} B of its own conversation")
    steps = dose_steps(n, cfg)
    args = launch_args(model, start_step + steps, cfg, tb, vb,
                       st["models"].get(model))
    res = trainer_runner.start(PLUGIN_ID, args)
    if not (isinstance(res, dict) and res.get("ok", True)):
        return False, f"launch failed: {res}"
    ms = st["models"].setdefault(model, {})
    ms.update({"sleeping": True,
               "run": {"model": model, "start_step": start_step,
                       "steps": start_step + steps, "batch": args["batch_size"],
                       "started_ts": time.time()}})
    _save_state(st)
    _log_event("sleep", model=model, exchanges=n, bytes=tb, steps=steps,
               start_step=start_step, target_step=start_step + steps)
    logmod.ok("sleep", f"{model} sleeping: {n} exchanges ({tb}B) -> {steps} steps")
    return True, f"sleeping: {model} {steps} steps over {n} exchanges"


_PAUSE = {"suspended": False, "warned": False}


def _child_proc():
    """psutil handle for the running trainer child, or None when there is no
    child or psutil is unavailable."""
    pid = trainer_runner.pid()
    if not pid:
        return None
    try:
        import psutil
        return psutil.Process(pid)
    except ImportError:
        if not _PAUSE["warned"]:
            logmod.warn("sleep", "psutil missing: sleep cannot yield to serving")
            _PAUSE["warned"] = True
        return None
    except Exception:
        return None


def yield_to_serving():
    """Suspend the sleep child so an in-flight request owns the box. Idempotent;
    returns True when a child is suspended as a result of this call or already
    was. Called from the serving path, so it must never raise."""
    if _PAUSE["suspended"]:
        return True
    # Gate on the master switch before touching disk: this runs on every
    # generation, and an install that never sleeps must not pay a state read for it.
    cfg = settings_mod.get()
    if not (cfg.get("sleep_enabled") and cfg.get("sleep_preempt", True)):
        return False
    if not _sleeper(_load_state()):
        return False
    proc = _child_proc()
    if proc is None:
        return False
    try:
        proc.suspend()
    except Exception as e:
        logmod.warn("sleep", f"suspend failed: {type(e).__name__}: {e}")
        return False
    _PAUSE["suspended"] = True
    logmod.info("sleep", "sleep child suspended for a served request")
    return True


def resume_if_quiet():
    """Resume a suspended sleep child once serving has been quiet for
    sleep_resume_s. Returns True when the child is running afterwards."""
    if not _PAUSE["suspended"]:
        return True
    idle = serving.idle_s()
    if idle is None or idle < float(settings_mod.get().get("sleep_resume_s", 5)):
        return False
    proc = _child_proc()
    if proc is None:
        _PAUSE["suspended"] = False   # child is gone; nothing left to resume
        return True
    try:
        proc.resume()
    except Exception as e:
        logmod.warn("sleep", f"resume failed: {type(e).__name__}: {e}")
        return False
    _PAUSE["suspended"] = False
    logmod.info("sleep", f"sleep child resumed after {idle:.1f}s quiet")
    return True


def suspended():
    """True while the sleep child is parked for serving."""
    return _PAUSE["suspended"]


def unpark_orphan():
    """Resume a child left suspended by a previous process. Module state resets on
    restart, so nothing would otherwise know the child is stopped and it would sit
    parked forever holding its memory. Best effort: no child, or a child already
    running, is a no-op."""
    proc = _child_proc()
    if proc is None:
        return False
    try:
        if proc.status() != "stopped":
            return False
        proc.resume()
    except Exception:
        return False
    logmod.ok("sleep", "resumed a sleep child left suspended by a previous run")
    return True


def cpu_budget(cfg):
    """Cores the sleep child may use: physical cores less sleep_reserve_cores,
    floored at 1 so the run still progresses on a 1-2 core box."""
    from routes._common import auto_thread_count
    return max(1, auto_thread_count() - max(0, int(cfg.get("sleep_reserve_cores", 1))))


def maybe_sleep(force_idle=False, model=None):
    """Gate chain, then launch one sleeper. Returns a short reason string for
    logs/tests. force_idle (the /sleep/now route) skips only the idle-time
    gate. model pins the sleeper (route parameter); otherwise the enrolled
    model with the most pending own-exchanges above the minimum goes first."""
    cfg = settings_mod.get()
    if not cfg.get("sleep_enabled"):
        return "disabled"
    names = enrolled(cfg)
    if not names:
        return "no models enrolled in sleep_models"
    if model is not None:
        if model not in names:
            return f"{model} is not enrolled in sleep_models"
        names = [model]
    if trainer_runner.is_running():
        return "trainer busy"
    idle = _idle_s()
    if idle is None:
        return "no experience yet"
    if not force_idle and idle < int(cfg["sleep_idle_min"]) * 60:
        return f"awake: last exchange {idle / 60:.1f} min ago"
    with _LOCK:
        st = _load_state()
        asleep = _sleeper(st)
        if asleep:
            return f"already sleeping: {asleep}"
        now = time.time()
        ranked = []
        for name in names:
            ms = st["models"].get(name) or {}
            cool = float(ms.get("cooldown_until") or 0)
            if now < cool:
                if model is not None:
                    return (f"cooling down after a failed sleep "
                            f"({(cool - now) / 60:.0f} min left)")
                continue
            ranked.append((_pending(name, float(ms.get("last_sleep_ts") or 0)), name))
        ranked.sort(reverse=True)
        if model is None:
            ranked = [(p, n) for p, n in ranked if p >= int(cfg["sleep_min_exchanges"])]
        if not ranked:
            return "no enrolled model has enough new experience"
        reason = ""
        for _, name in ranked:
            launched, reason = _launch(st, name, cfg)
            if launched:
                break
        return reason


def wake(model):
    """Stop the model's in-flight sleep run. Serving resumes from the newest
    sleep checkpoint; finalize() reconciles checkpoints on the next tick."""
    with _LOCK:
        st = _load_state()
        if not (st["models"].get(model) or {}).get("sleeping"):
            return {"ok": True, "state": "awake", "model": model, "note": "was not sleeping"}
        trainer_runner.stop()
        _log_event("wake", model=model)
        logmod.ok("sleep", f"woken by user: {model}")
        return {"ok": True, "state": "waking", "model": model}


_PUBLISH_HOOK = {"fn": None}


def set_publish_hook(fn):
    """Register a callable invoked with a model dir name after sleep replaces
    that model's serving binary. The engine reads a bin into memory and closes
    the file, so the swap itself needs no lock, but a live subprocess still
    holds the pre-sleep weights until something reloads it. app.py registers the
    reload; without a hook the new weights serve from the next engine start."""
    _PUBLISH_HOOK["fn"] = fn


def publish(model):
    """Re-export the model's newest checkpoint over its serving binary, so the
    box talks to what it just consolidated. Without this a sleep run improves a
    .pt nobody serves.

    Only a model that already serves a bin gets one: a PyTorch-only model must
    not grow an engine artifact because it slept. The dtype is read back off the
    bin in place, so an int8 box stays int8. The export writes a sibling temp and
    renames it into place, so a failed export leaves the previous weights
    serving and the box never holds a half-written model.

    Returns the export result, or None when there is nothing to publish."""
    if not binr.exists(model):
        return None
    step = _model_step(model)
    if step is None:
        return None
    from training import export as export_mod
    res = export_mod.export_checkpoint(model, step, dtype=binr.weight_dtype(model))
    logmod.ok("sleep", f"{model}: serving step {step} ({res['bytes']} B, {res['dtype']})")
    fn = _PUBLISH_HOOK["fn"]
    if fn is not None:
        try:
            fn(model)
        except Exception as e:
            # the new weights are on disk and will serve from the next engine
            # start; only this reload was lost
            logmod.warn("sleep", f"{model}: engine reload failed: {type(e).__name__}: {e}")
    return res


def finalize():
    """Reconcile after a sleep run ends: record the sleep, prune checkpoints."""
    with _LOCK:
        st = _load_state()
        cfg = settings_mod.get()
        for model, ms in st["models"].items():
            if not ms.get("sleeping"):
                continue
            run = ms.get("run") or {}
            start = run.get("start_step") or 0
            end_step = _model_step(model) or start
            finals = ms.get("finals") or []
            if end_step > start:
                finals = prune(model, start, end_step, cfg["sleep_keep_finals"], finals)
                ms["last_sleep_ts"] = time.time()
                ms.pop("cooldown_until", None)
                _, elapsed = _train_progress(model)
                if elapsed:
                    # what a step of run["batch"] actually cost on THIS box; the
                    # next launch sizes its step from it. Kept in sleep state, not
                    # read back off train.csv, because a model dir travels between
                    # machines and its rows carry the throughput of whichever one
                    # wrote them.
                    ms["step_s"] = round(elapsed / (end_step - start), 1)
                    ms["step_batch"] = run.get("batch")
            else:
                # no checkpoint survived. The cooldown exists to stop the watcher
                # retry-storming a launch that cannot work (cardinal: 3 failed
                # launches in 11 minutes); a run that trained and was woken
                # between checkpoints is not that, and must not be punished for it
                logged, _ = _train_progress(model)
                lost = max(0, (logged or start) - start)
                if not lost:
                    ms["cooldown_until"] = time.time() + FAIL_COOLDOWN_S
            ms.update({"sleeping": False, "finals": finals, "run": {}})
            if end_step > start:
                held, v0, v1 = regressed(model, start, cfg)
                ms["val"] = v1
                published = None
                if held:
                    logmod.warn("sleep", f"{model}: val {v0:.4f} -> {v1:.4f} over its own "
                                         f"conversation; holding step {end_step} back from serving")
                else:
                    try:
                        published = publish(model)
                    except Exception as e:
                        # the consolidation itself is already safe on disk; a failed
                        # re-export costs the box only the newest weights in serving
                        logmod.error("sleep", f"{model}: publish failed, still serving the "
                                              f"previous weights: {type(e).__name__}: {e}")
                _log_event("awake", model=model, end_step=end_step,
                           steps_gained=end_step - start, finals=finals,
                           served=bool(published), held=held, val_first=v0, val_last=v1)
                logmod.ok("sleep", f"{model} awake at step {end_step}")
            elif lost:
                _log_event("lost", model=model, end_step=end_step, steps_lost=lost)
                logmod.warn("sleep", f"{model} woken between checkpoints: {lost} steps "
                                     f"trained but not saved")
            else:
                _log_event("failed", model=model, end_step=end_step,
                           cooldown_s=FAIL_COOLDOWN_S)
                logmod.error("sleep", f"{model} sleep gained no steps; cooling down "
                                      f"{FAIL_COOLDOWN_S // 60} min")
        _save_state(st)


def tick():
    """One watcher pass; safe to call every WATCH_EVERY_S."""
    st = _load_state()
    if _sleeper(st):
        if (trainer_runner.state() or {}).get("status") != "running":
            _PAUSE["suspended"] = False
            finalize()
            return "finalized"
        resume_if_quiet()
        return "suspended" if suspended() else "sleeping"
    return maybe_sleep()


def watcher():
    """Daemon loop for app startup. The resume check runs on the short poll and
    the full pass every WATCH_EVERY_S: a suspend can land at any point inside a
    watch period, so a loop that picked its interval up front would leave a
    parked run stopped for up to a full period after the box went quiet."""
    serving.on_began(yield_to_serving)
    unpark_orphan()
    last_tick = 0.0
    while True:
        time.sleep(WATCH_PAUSED_S)
        try:
            if suspended():
                resume_if_quiet()
            if time.monotonic() - last_tick >= WATCH_EVERY_S:
                last_tick = time.monotonic()
                tick()
        except Exception as e:
            logmod.error("sleep", f"watcher: {e}")
