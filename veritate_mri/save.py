# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the unified save function. every trainer calls save() at every checkpoint.
# - one path to disk: validates name + description, writes the .pt, runs the full
#   dump suite (probe, lens, classroom, grades, concepts, surprise, quant_kl,
#   generation), emits status into the in-memory log.
# - per-step CSV append is append_train_row(). cheap, every step.
# - imported by the trainer process (PyTorch already in memory), not by the MRI
#   server. MRI consumes the resulting files via veritate_mri/readers/.
# veritate_mri/save.py
# ------------------------------------------------------------------------------------
# Imports:

import csv
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import logs as logmod
from readers import paths, models, config as cfg_reader

# ------------------------------------------------------------------------------------
# Constants

CSV_HEADER = ["step", "split", "loss", "lr", "grad_norm", "tok_per_s", "wall_s", "seed"]

NAME_SEP = "_"

SHA256_CHUNK = 1024 * 1024

# canonical dump filenames inside hooks/step_<N>/. dump_* in
# training/checkpoint_probe.py emit prefixed names; we rename to these.
RENAME_MAP_TEMPLATE = {
    "probe_step_{step}.json":     "probe.json",
    "lens_step_{step}.npz":       "lens.npz",
    "classroom_step_{step}.json": "classroom.json",
    "grades_step_{step}.json":    "grades.json",
    "concepts_step_{step}.json":  "concepts.json",
    "surprise_step_{step}.json":  "surprise.json",
    "quant_kl_step_{step}.json":  "quant_kl.json",
    "step_{step}.json":           "generation.json",
}

# ------------------------------------------------------------------------------------
# Functions

def _validate_name(name):
    if not models.is_valid_name(name):
        raise ValueError(
            f"invalid model name: {name!r}. expected <name>_<param>_<precision>_<version> "
            "per documentation/training/model_naming.md"
        )


def compose_name(corpus, size, precision, version):
    leaf = corpus.rsplit(":", 1)[-1] if ":" in corpus else corpus
    return NAME_SEP.join([leaf, size, precision, version])


def require_description(desc):
    if not isinstance(desc, str) or not desc.strip():
        raise ValueError("description required (ROE rule 6)")
    return desc.strip()


def resolve_corpus(stem):
    from readers import corpus as corpus_reader
    train, val = corpus_reader.resolve_paths(stem)
    if train is None:
        raise FileNotFoundError(f"corpus stem not found (shared or bundled): {stem!r}")
    return train, val


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(SHA256_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_corpus(stem):
    train, val = resolve_corpus(stem)
    out = {"stem": stem, "train_sha256": sha256_file(train), "train_bytes": os.path.getsize(train)}
    if val is not None:
        out["val_sha256"] = sha256_file(val)
        out["val_bytes"]  = os.path.getsize(val)
    return out


def _auto_description(name, args):
    parts = [name]
    shape = args.get("shape") if isinstance(args, dict) else None
    if isinstance(shape, dict):
        parts.append(f"layers={shape.get('layers','?')} hidden={shape.get('hidden','?')} ffn={shape.get('ffn','?')} heads={shape.get('heads','?')} seq={shape.get('seq','?')}")
    elif isinstance(args, dict) and args.get("layers"):
        parts.append(f"layers={args.get('layers')} hidden={args.get('hidden')} ffn={args.get('ffn')} heads={args.get('heads')} seq={args.get('seq')}")
    if isinstance(args, dict):
        if args.get("training"):    parts.append(str(args["training"]))
        if args.get("from_model"):  parts.append(f"warm-start from {args['from_model']}")
        c = args.get("corpus_train") or args.get("corpus")
        if c:                       parts.append(f"corpus={os.path.basename(str(c))}")
        if args.get("total_steps"): parts.append(f"total_steps={args['total_steps']}")
        if args.get("base_lr"):     parts.append(f"base_lr={args['base_lr']}")
        if args.get("seed") is not None: parts.append(f"seed={args['seed']}")
    return " | ".join(parts)


def _validate_description(name, args):
    if isinstance(args, dict):
        for k in ("description", "stage_description"):
            v = args.get(k)
            if isinstance(v, str) and v.strip(): return v.strip()
    cfg = cfg_reader.load(name) or {}
    for k in ("description", "stage_description"):
        v = cfg.get(k)
        if isinstance(v, str) and v.strip(): return v.strip()
    ta = cfg.get("training_args") or {}
    v = ta.get("description") if isinstance(ta, dict) else None
    if isinstance(v, str) and v.strip(): return v.strip()
    auto = _auto_description(name, args)
    if auto:
        if isinstance(args, dict): args.setdefault("description", auto)
        return auto
    raise ValueError(
        f"missing description for model {name!r}. ROE rule 6: provide --description, "
        "set 'description' in args, or pass a populated args dict so the auto-description can build one."
    )


def _ensure_config(name, args):
    cfg_path = paths.config_path(name)
    if os.path.isfile(cfg_path):
        return
    if not isinstance(args, dict):
        raise ValueError(f"no config.json at {cfg_path} and no args dict provided to bootstrap one.")
    os.makedirs(paths.model_dir(name), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=2, ensure_ascii=False)
    logmod.info("save", f"wrote new config.json for {name}")


def _rename_dumps(step_dir, step):
    for src_tmpl, dst in RENAME_MAP_TEMPLATE.items():
        src = os.path.join(step_dir, src_tmpl.format(step=step))
        dst_path = os.path.join(step_dir, dst)
        if os.path.isfile(src):
            os.replace(src, dst_path)


def append_train_row(name, step, split, loss, lr=None, grad_norm=None,
                     tok_per_s=None, wall_s=None, seed=None):
    """Append one row to models/<name>/train.csv. Writes header if file is new."""
    _validate_name(name)
    p = paths.train_csv_path(name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    new_file = not os.path.isfile(p)
    with open(p, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow([
            int(step), str(split),
            "" if loss      is None else f"{float(loss):.6f}",
            "" if lr        is None else f"{float(lr):.6e}",
            "" if grad_norm is None else f"{float(grad_norm):.6f}",
            "" if tok_per_s is None else f"{float(tok_per_s):.2f}",
            "" if wall_s    is None else f"{float(wall_s):.3f}",
            "" if seed      is None else int(seed),
        ])


def save(model, name, step, *, optimizer=None, args=None, prompt=None,
         dump_set=None):
    """Unified save: torch checkpoint + full dump suite + status logs.

    Args:
        model:     the trainer's PyTorch model (already on its device).
        name:      model dir name; must validate per model_naming.md.
        step:      training step number.
        optimizer: optional optimizer state to embed in the .pt.
        args:      dict of training args; must contain a non-empty description
                   (ROE rule 6) if config.json doesn't already exist.
        prompt:    canonical probe prompt for dump_probe / dump_generation /
                   dump_surprise / dump_quant_kl. defaults to PROBE_PROMPT.
        dump_set:  optional iterable of dump names to skip ("probe", "lens",
                   "classroom", ...). default: run all.

    Returns: the absolute path of the .pt file written.
    """
    import torch
    from checkpoint_probe import (
        dump_probe, dump_classroom, dump_grades, dump_concepts,
        dump_surprise, dump_quant_kl, dump_generation, PROBE_PROMPT,
    )

    _validate_name(name)
    _ensure_config(name, args)
    _validate_description(name, args)

    if prompt is None:
        prompt = PROBE_PROMPT

    t0 = time.time()
    logmod.info("save", f"start: {name} step {step}")

    ckpt_dir  = paths.checkpoints_dir(name)
    ckpt_path = paths.checkpoint_path(name, step)
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {"model": model.state_dict(), "step": int(step)}
    if optimizer is not None: state["optimizer"] = optimizer.state_dict()
    if args is not None:      state["args"]      = args
    torch.save(state, ckpt_path)
    logmod.info("save", f"wrote checkpoint: {ckpt_path}")

    step_dir = paths.hook_step_dir(name, step)
    os.makedirs(step_dir, exist_ok=True)

    corpus_stem = None
    if isinstance(args, dict) and isinstance(args.get("corpus"), str) and args["corpus"].strip():
        corpus_stem = args["corpus"].strip()
    else:
        cfg = cfg_reader.load(name) or {}
        ta  = cfg.get("training_args") or {}
        if isinstance(ta, dict) and isinstance(ta.get("corpus"), str) and ta["corpus"].strip():
            corpus_stem = ta["corpus"].strip()
    if corpus_stem and ":" in corpus_stem:
        corpus_stem = corpus_stem.rsplit(":", 1)[-1]
    corpus_path = paths.corpus_train_path(corpus_stem) if corpus_stem else None

    skip = set(dump_set or [])
    if "generation" not in skip:
        if not corpus_stem:
            logmod.error("save", (
                f"generation dump skipped: model {name!r} has no corpus stem in "
                f"training_args.corpus. set --corpus <stem> in your trainer so the "
                f"memory fingerprint probe can index the right data."
            ))
            skip.add("generation")
        elif not (corpus_path and os.path.isfile(corpus_path)):
            logmod.error("save", (
                f"generation dump skipped: corpus stem {corpus_stem!r} resolves to "
                f"{corpus_path!r} which does not exist. expected a prepped bin at "
                f"plugins/corpus/{corpus_stem}_train.bin."
            ))
            skip.add("generation")
    dumps = [
        ("probe",      lambda: dump_probe     (model, prompt, step_dir, step)),
        ("classroom",  lambda: dump_classroom (model,         step_dir, step)),
        ("grades",     lambda: dump_grades    (model,         step_dir, step)),
        ("concepts",   lambda: dump_concepts  (model,         step_dir, step)),
        ("surprise",   lambda: dump_surprise  (model, prompt, step_dir, step)),
        ("quant_kl",   lambda: dump_quant_kl  (model, prompt, step_dir, step)),
        ("generation", lambda: dump_generation(model, prompt, step_dir, step, corpus_path=corpus_path)),
    ]
    cuda_avail = torch.cuda.is_available()
    for label, fn in dumps:
        if label in skip:
            logmod.info("save", f"skip dump: {label}")
            continue
        try:
            fn()
        except Exception as e:
            logmod.error("save", f"dump {label} failed: {e}")
        finally:
            if cuda_avail:
                torch.cuda.empty_cache()
    _rename_dumps(step_dir, step)

    logmod.ok("save", f"done: {name} step {step} ({time.time() - t0:.1f}s)")
    return ckpt_path
