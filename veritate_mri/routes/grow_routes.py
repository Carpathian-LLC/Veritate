# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Function-preserving model growth from the Training tab (IDEA 21). POST
#   /models/grow validates everything up front (source checkpoint, unused name,
#   target reachable per training.grow.validate_growth, supported trunk,
#   optional target_seq >= source and a slot-stride multiple), then
#   runs training.grow.grow_checkpoint in a background thread: loading a
#   multi-GB checkpoint plus the transform takes a minute or more. GET
#   /models/grow/status is the poll target; GET /models/grow/options?source=X
#   enumerates the trainer_sizes.json targets reachable from X so the
#   reachability rules live server-side only.
# - On success the new model dir is exactly what the continue-training flow
#   resumes: checkpoints/step_0.pt plus a config.json cloned from the source
#   with name/shape/size/description/training_args updated (fresh budget,
#   warmup > 0 because the grown checkpoint carries no optimizer state, corpus
#   pins cleared like fork so the form's corpus choice wins).
# - Torch import stays inside the worker so cold pages never pay it.
# veritate_mri/routes/grow_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil
import threading

from flask import request
from readers import checkpoints, models, paths, trainers
from readers import config as cfg_reader
from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

# Trunks the growth tool maps exactly. Everything else (MoE/PKM/Monarch FFNs,
# looped depth, delta/pinned state rules) is refused before any work.
SUPPORTED_TRUNKS      = {"", "dense", "hybrid", "recurrent"}
SUPPORTED_STATE_RULES = {"", "gla"}

# The grown model is a fresh training root: same budget/warmup defaults as fork.
GROW_STEP          = 0
DEFAULT_GROW_BUDGET = 5000
DEFAULT_GROW_WARMUP = 200
DESCRIPTION_TMPL = ("Grown from {source}@{step} (function-preserving): "
                    "{src_shape} -> {dst_shape}, ~{params}. "
                    "Continue-train with warmup_steps > 0 (fresh optimizer).")

_LOCK  = threading.Lock()
_STATE = {"running": False, "source": None, "name": None, "phase": "idle",
          "error": None, "result": None}

# ------------------------------------------------------------------------------------
# Functions


def _shape_str(s):
    base = f"{s['layers']}L/{s['hidden']}h/{s['ffn']}f/{s['heads']}heads"
    return base + (f"/{s['seq']}seq" if "seq" in s else "")


def _patch_stride(trunk):
    """Slot stride the target seq must divide by; 1 for non-patched trunks."""
    if trunk != "hybrid":
        return 1
    from veritate_core.model_patched import PATCH_STRIDE
    return PATCH_STRIDE


def _params_str(n):
    return f"{n / 1e6:.1f}M params"


def estimate_params(shape, seq, trunk):
    """Exact unique-parameter count for a supported trunk at `shape`.

    Derived from the model classes (verified against parameters() on the hybrid
    toy): attention Block = 2H norms + 4H^2 attn + 2HF ffn; RecurrentBlock adds
    H^2 gate, 12H conv (CONV_KERNEL=4 depthwise over 3H), KH+K a_proj, H/K
    o_norm. Embeddings: 256H tied (counted once) + seq*H positions + H n_out;
    the patched trunk adds slots*H slot positions and N_LOCAL_ENC+N_LOCAL_DEC
    local blocks around `layers` global ones.
    """
    h, f, k, layers = shape["hidden"], shape["ffn"], shape["heads"], shape["layers"]
    block = 2 * h + 4 * h * h + 2 * h * f
    rec   = block + h * h + 12 * h + k * h + k + h // k
    emb   = 256 * h + seq * h + h
    if trunk == "hybrid":
        from veritate_core.model_patched import N_LOCAL_DEC, N_LOCAL_ENC, PATCH_STRIDE
        return emb + (seq // PATCH_STRIDE) * h + (N_LOCAL_ENC + N_LOCAL_DEC) * block + layers * rec
    if trunk == "recurrent":
        return emb + layers * rec
    return emb + layers * block


def _source_context(name):
    """(config, shape, seq, trunk, latest_step) for a growable source, or raises
    ValueError with the reason it cannot grow."""
    if not name or not models.exists(name):
        raise ValueError(f"model not found: {name}")
    cfg = cfg_reader.load(name)
    if not cfg:
        raise ValueError(f"{name} has no config.json")
    shape_cfg = cfg.get("shape") or {}
    shape = {f: int(shape_cfg.get(f) or 0) for f in ("layers", "hidden", "ffn", "heads")}
    if not all(shape.values()):
        raise ValueError(f"{name}'s config.json shape is incomplete: {shape_cfg}")
    seq = int(shape_cfg.get("seq") or 0)
    if seq <= 0:
        raise ValueError(f"{name}'s config.json shape has no seq")
    ta = cfg.get("training_args") or {}
    trunk = (ta.get("trunk") or "").strip()
    if trunk not in SUPPORTED_TRUNKS:
        raise ValueError(f"growth does not support trunk {trunk!r} "
                         "(dense FFN + gla state rule only)")
    state_rule = (ta.get("state_rule") or "").strip()
    if state_rule not in SUPPORTED_STATE_RULES:
        raise ValueError(f"growth does not support state_rule {state_rule!r}")
    latest = checkpoints.latest_step(name)
    if latest is None:
        raise ValueError(f"{name} has no checkpoints")
    return cfg, shape, seq, trunk, latest


def _resolve_target(target_size):
    """target_size is a trainer_sizes.json key or an explicit shape dict.
    Returns (shape dict, size key or None)."""
    if isinstance(target_size, str):
        sizes = trainers.load_native_sizes()
        entry = sizes.get(target_size)
        if entry is None:
            raise ValueError(f"unknown target_size {target_size!r}; not in trainer_sizes.json")
        return {f: int(entry[f]) for f in ("layers", "hidden", "ffn", "heads")}, target_size
    if isinstance(target_size, dict):
        try:
            shape = {f: int(target_size[f]) for f in ("layers", "hidden", "ffn", "heads")}
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError("explicit target_size needs integer layers/hidden/ffn/heads") from e
        sizes = trainers.load_native_sizes()
        key = next((k for k, v in sizes.items()
                    if all(int(v[f]) == shape[f] for f in shape)), None)
        return shape, key
    raise ValueError("target_size must be a size key or a {layers,hidden,ffn,heads} object")


def _write_grown_config(src_cfg, source, src_step, name, target, size_key, n_params,
                        src_shape):
    """config.json for the grown model, cloned from the source so the continue
    flow resumes it with zero special-casing (mirrors training/fork.py)."""
    cfg = dict(src_cfg)
    desc = DESCRIPTION_TMPL.format(source=source, step=src_step,
                                   src_shape=_shape_str(src_shape),
                                   dst_shape=_shape_str(target),
                                   params=_params_str(n_params))
    shape = dict(cfg.get("shape") or {})
    shape.update(target)
    shape["params"] = n_params
    cfg["name"]  = name
    cfg["step"]  = GROW_STEP
    cfg["shape"] = shape
    cfg["description"]    = desc
    cfg["n_params_total"] = n_params
    cfg["grown_from"] = {"source": source, "step": src_step}
    ta = dict(cfg.get("training_args") or {})
    ta.update(target)
    ta["output_dir"]   = paths.model_dir(name)
    ta["resume"]       = True
    ta["total_steps"]  = GROW_STEP + DEFAULT_GROW_BUDGET
    ta["warmup_steps"] = DEFAULT_GROW_WARMUP
    ta["description"]  = desc
    # Clear the source's pinned corpus paths (same reason as fork): the trainer
    # resolves the form's --corpus stem only when both are empty.
    ta["corpus_bin"] = ""
    ta["val_bin"]    = ""
    if size_key:
        ta["size"] = size_key
    cfg["training_args"] = ta
    cfg_path = paths.config_path(name)
    tmp = cfg_path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, cfg_path)


def _run_grow_job(source, src_step, name, target, size_key, src_cfg, src_shape,
                  trunk):
    new_dir = paths.model_dir(name)
    try:
        with _LOCK:
            _STATE.update(phase="growing")
        from training import grow as grow_mod
        result = grow_mod.grow_checkpoint(
            paths.checkpoint_path(source, src_step),
            paths.checkpoint_path(name, GROW_STEP),
            layers=target["layers"], hidden=target["hidden"],
            ffn=target["ffn"], heads=target["heads"], seq=target["seq"],
            src_heads=src_shape["heads"], out_step=GROW_STEP)
        with _LOCK:
            _STATE.update(phase="writing config")
        n_params = estimate_params(target, target["seq"], trunk)
        _write_grown_config(src_cfg, source, src_step, name, target, size_key,
                            n_params, src_shape)
        with _LOCK:
            _STATE.update(running=False, phase="done", error=None,
                          result={"name": name, "target": target, "size": size_key,
                                  "params_before": result["params_before"],
                                  "params_after":  result["params_after"],
                                  "n_params_total": n_params})
        logmod.ok("grow", f"grew {source}@{src_step} -> {name} "
                          f"({_shape_str(src_shape)} -> {_shape_str(target)})")
    except Exception as e:
        shutil.rmtree(new_dir, ignore_errors=True)
        msg = str(e).strip() or type(e).__name__
        with _LOCK:
            _STATE.update(running=False, phase="error", error=msg, result=None)
        logmod.error("grow", f"grow {source} -> {name} failed: {msg}")


def register(app):
    @app.route("/models/grow/options")
    def models_grow_options():
        """trainer_sizes.json targets reachable from ?source, with exact param
        counts, so the client never re-implements the reachability rules."""
        from training import grow as grow_mod
        source = (request.args.get("source") or "").strip()
        try:
            _, shape, seq, trunk, latest = _source_context(source)
        except ValueError as e:
            return ({"ok": False, "error": str(e)}, 400)
        # seq is orthogonal to the size keys: offer 1x/2x/4x of the source
        # context, with per-choice param counts so the client renders exact
        # before/after numbers without knowing the formula.
        seq_choices = [seq, seq * 2, seq * 4]
        targets = []
        for key, entry in trainers.load_native_sizes().items():
            tgt = {f: int(entry[f]) for f in ("layers", "hidden", "ffn", "heads")}
            if not any(tgt[f] > shape[f] for f in tgt):
                continue
            try:
                grow_mod.validate_growth(shape, tgt)
            except ValueError:
                continue
            targets.append(dict(tgt, size=key, params=estimate_params(tgt, seq, trunk),
                                params_seq={str(c): estimate_params(tgt, c, trunk)
                                            for c in seq_choices}))
        targets.sort(key=lambda t: t["params"])
        return {"ok": True, "source": source, "step": latest,
                "steps": checkpoints.list_steps(source),
                "shape": shape, "params": estimate_params(shape, seq, trunk),
                "seq": seq, "seq_choices": seq_choices,
                "source_params_seq": {str(c): estimate_params(shape, c, trunk)
                                      for c in seq_choices},
                "targets": targets}

    @app.route("/models/grow", methods=["POST"])
    def models_grow():
        """Start a growth job: {source, step?, target_size, name}."""
        from training import grow as grow_mod
        body = request.get_json(silent=True) or {}
        source = (body.get("source") or "").strip()
        name   = (body.get("name") or "").strip()
        try:
            src_cfg, src_shape, seq, trunk, latest = _source_context(source)
            step = latest if body.get("step") is None else int(body.get("step"))
            if step not in (checkpoints.list_steps(source) or []):
                raise ValueError(f"{source} has no checkpoint at step {step}")
            if not models.is_valid_name(name):
                raise ValueError(f"new name {name!r} is invalid: lowercase letters, "
                                 "digits, underscores; no leading/trailing underscore")
            # A bare directory collides too (makedirs only guards the leaf), so
            # check the dir itself, not just models.exists (dir + config.json).
            if models.exists(name) or os.path.isdir(paths.model_dir(name)):
                raise ValueError(f"a model named {name!r} already exists; pick another")
            if body.get("target_size") is None:
                # seq-only growth: keep the source shape.
                target, size_key = dict(src_shape), None
            else:
                target, size_key = _resolve_target(body.get("target_size"))
            target["seq"] = seq if body.get("target_seq") is None \
                else int(body.get("target_seq"))
            src_full = dict(src_shape, seq=seq)
            grow_mod.validate_growth(src_full, target,
                                     seq_multiple=_patch_stride(trunk))
            if not any(target[f] > src_full[f] for f in target):
                raise ValueError("target shape equals the source; nothing to grow")
        except (TypeError, ValueError) as e:
            return ({"ok": False, "error": str(e)}, 400)
        with _LOCK:
            if _STATE["running"]:
                return ({"ok": False,
                         "error": f"a growth job is already running "
                                  f"({_STATE['source']} -> {_STATE['name']})"}, 409)
            try:
                os.makedirs(paths.checkpoints_dir(name), exist_ok=False)
            except OSError as e:
                return ({"ok": False, "error": f"could not create model dir: {e}"}, 400)
            _STATE.update(running=True, source=source, name=name,
                          phase="starting", error=None, result=None)
        threading.Thread(target=_run_grow_job,
                         args=(source, step, name, target, size_key, src_cfg,
                               src_full, trunk),
                         name="grow:job", daemon=True).start()
        return {"ok": True, "source": source, "step": step, "name": name,
                "target": target, "size": size_key}

    @app.route("/models/grow/status")
    def models_grow_status():
        with _LOCK:
            return dict(_STATE, ok=True)
