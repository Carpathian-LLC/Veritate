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
# veritate_mri/training/save.py
# ------------------------------------------------------------------------------------
# Imports:

import csv
import hashlib
import json
import os
import sys
import time

# Plugin subprocesses import this module before veritate_mri/ is on sys.path.
# Make the package root importable so `from runtime/readers/...` work regardless
# of the entry point that loaded us.
_MRI_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _MRI_ROOT not in sys.path:
    sys.path.insert(0, _MRI_ROOT)

from readers import paths, models, config as cfg_reader
from runtime import logs as logmod

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
    "math_step_{step}.json":      "math.json",
    "grammar_step_{step}.json":   "grammar.json",
    "reasoning_step_{step}.json": "reasoning.json",
    "concepts_step_{step}.json":  "concepts.json",
    "surprise_step_{step}.json":  "surprise.json",
    "quant_kl_step_{step}.json":  "quant_kl.json",
    "writing_health_step_{step}.json": "writing_health.json",
    "reading_comprehension_step_{step}.json": "reading_comprehension.json",
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


def compose_name(*args, **kwargs):
    """Compose the canonical model dir name.

    Preferred form:    compose_name(user_name, size)
                       -> "<slug>_<size>"           (e.g. "chatty_otter_85m")
    Legacy form:       compose_name(corpus, size, precision, version)
                       -> "<corpus>_<size>_<precision>_<version>"

    The user_name is slugified (lowercased, non-alnum -> '_'). The legacy form
    is kept so older plugins / pruning.py keep working unchanged.
    """
    from readers import models as _models  # local import: avoid circular at module load
    if "name" in kwargs or len(args) == 2:
        name = kwargs.get("name", args[0] if args else "")
        size = kwargs.get("size", args[1] if len(args) > 1 else "")
        slug = _models.slugify_user_name(name)
        if not slug:
            raise ValueError("compose_name: empty user name after slugify")
        if not isinstance(size, str) or not size.strip():
            raise ValueError("compose_name: size required (e.g. '85m', '1b')")
        return NAME_SEP.join([slug, size.strip()])
    # Legacy 4-arg form.
    corpus, size, precision, version = args
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
    """Compute the sha256 of `path`, caching the result next to the file as
    `<path>.sha256`. The cache is keyed on the source file's mtime+size; if
    either changes, the cache is invalidated and the hash is recomputed.

    Cache file format (one line):
        <hex_digest> <mtime_ns> <size_bytes>

    Saves ~78 seconds per training run on a 200 GB byte-level corpus.
    """
    cache_path = path + ".sha256"
    try:
        st = os.stat(path)
    except OSError:
        st = None
    if st is not None and os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                line = f.read().strip()
            parts = line.split()
            if len(parts) == 3:
                cached_digest, cached_mtime_ns, cached_size = parts
                if (int(cached_mtime_ns) == st.st_mtime_ns and
                    int(cached_size)     == st.st_size and
                    len(cached_digest)   == 64):
                    return cached_digest
        except (OSError, ValueError):
            pass  # fall through to recompute

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(SHA256_CHUNK), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if st is not None:
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(f"{digest} {st.st_mtime_ns} {st.st_size}\n")
        except OSError:
            pass  # cache write failure is non-fatal
    return digest


def hash_corpus(stem):
    train, val = resolve_corpus(stem)
    out = {"stem": stem, "train_sha256": sha256_file(train), "train_bytes": os.path.getsize(train)}
    if val is not None:
        out["val_sha256"] = sha256_file(val)
        out["val_bytes"]  = os.path.getsize(val)
    return out


def _auto_description(name, args):
    """Build a one-line spec string from the training args. This is what the
    user used to read off the model name (e.g. fineweb_edu_85m_bf16_v1_sparse);
    it now lives in the description, leaving the name free for a human label."""
    parts = [name]
    if isinstance(args, dict):
        spec = []
        c = args.get("corpus_train") or args.get("corpus")
        if c:                          spec.append(f"corpus={os.path.basename(str(c))}")
        if args.get("size"):           spec.append(f"size={args['size']}")
        if args.get("precision"):      spec.append(f"precision={args['precision']}")
        if args.get("version"):        spec.append(f"version={args['version']}")
        if args.get("variant"):        spec.append(f"variant={args['variant']}")
        if spec: parts.append(" ".join(spec))
        shape = args.get("shape") if isinstance(args.get("shape"), dict) else None
        if shape:
            parts.append(f"layers={shape.get('layers','?')} hidden={shape.get('hidden','?')} ffn={shape.get('ffn','?')} heads={shape.get('heads','?')} seq={shape.get('seq','?')}")
        elif args.get("layers"):
            parts.append(f"layers={args.get('layers')} hidden={args.get('hidden')} ffn={args.get('ffn')} heads={args.get('heads')} seq={args.get('seq')}")
        if args.get("training"):    parts.append(str(args["training"]))
        if args.get("from_model"):  parts.append(f"warm-start from {args['from_model']}")
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


def truncate_train_csv_at(name, resume_step):
    """Remove rows where step > resume_step from models/<name>/train.csv.

    Called by plugin trainers immediately after loading a checkpoint, BEFORE
    the training loop starts. Resuming from step_<N>.pt means everything that
    was logged after step N never made it to disk — those rows are stale and
    will be retrained, producing different loss values. Leaving them in place
    creates duplicate step numbers that confuse the dashboard's loss curve and
    training-health widget.

    Returns the number of rows removed (0 if no truncation was needed). The
    header row is always preserved if the file existed."""
    _validate_name(name)
    p = paths.train_csv_path(name)
    if not os.path.isfile(p):
        return 0
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except OSError:
        return 0
    if not rows:
        return 0
    header, body = rows[0], rows[1:]
    kept   = []
    dropped = 0
    for r in body:
        if not r: continue
        try:
            s = int(r[0])
        except (ValueError, IndexError):
            kept.append(r)
            continue
        if s > int(resume_step):
            dropped += 1
        else:
            kept.append(r)
    if dropped == 0:
        return 0
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in kept:
            w.writerow(r)
    os.replace(tmp, p)
    logmod.info("save", f"truncated {dropped} stale row(s) from train.csv after resume from step {resume_step}")
    return dropped


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
    from .checkpoint_probe import (
        dump_probe, dump_classroom, dump_grades, dump_concepts,
        dump_math, dump_grammar, dump_reasoning,
        dump_surprise, dump_quant_kl, dump_generation,
        dump_writing_health, dump_reading_comprehension, PROBE_PROMPT,
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
    # Atomic write: serialize to a sibling .tmp first, then os.replace into
    # place. If the process dies mid-write (SIGKILL, OOM, Windows kill, power
    # loss), the partial file is named step_N.pt.tmp and is invisible to the
    # `step_*.pt` parser in every plugin's latest_checkpoint_step. Without
    # this, a partial step_N.pt would be picked as the resume target and crash
    # torch.load on next start — or worse, silently misload.
    tmp_path = ckpt_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_path)
    logmod.info("save", f"wrote checkpoint: {ckpt_path}")

    # The dump suite walks a canonical Veritate. Non-canonical models (MoE,
    # multimind sidecars) expose a hook_spec() that returns a canonical-shaped
    # adapter; canonical models return self. Plain nn.Modules that haven't
    # opted in fall back to themselves and may fail mid-dump — that's fine,
    # the per-dump try/except below logs and continues.
    view = model.hook_spec() if hasattr(model, "hook_spec") else model

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
        ("probe",      lambda: dump_probe     (view, prompt, step_dir, step)),
        ("classroom",  lambda: dump_classroom (view,         step_dir, step)),
        ("grades",     lambda: dump_grades    (view,         step_dir, step)),
        ("reading_comprehension", lambda: dump_reading_comprehension(view, step_dir, step)),
        ("math",       lambda: dump_math      (view,         step_dir, step)),
        ("grammar",    lambda: dump_grammar   (view,         step_dir, step)),
        ("reasoning",  lambda: dump_reasoning (view,         step_dir, step)),
        ("concepts",   lambda: dump_concepts  (view,         step_dir, step)),
        ("surprise",   lambda: dump_surprise  (view, prompt, step_dir, step)),
        ("quant_kl",   lambda: dump_quant_kl  (view, prompt, step_dir, step)),
        ("writing_health", lambda: dump_writing_health(view,    step_dir, step, corpus_path=corpus_path)),
        ("generation", lambda: dump_generation(view, prompt, step_dir, step, corpus_path=corpus_path)),
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
