# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - pytorch / c-engine model resolution and load helpers shared across route
#   modules and app.main().
# veritate_mri/routes/_brain.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from inference.backends.pytorch import Brain, load_memory
from readers import bin as binr, checkpoints, engine, models, paths, train_csv
from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def resolve_pytorch_model(name):
    if name == "auto":
        candidates = []
        for n in models.list_models():
            if checkpoints.list_steps(n):
                candidates.append((train_csv.file_stat(n).st_mtime if train_csv.file_stat(n) else 0, n))
        if not candidates:
            logmod.warn("backends", "no models with checkpoints under models/. pass --model <name> explicitly.")
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    if not models.exists(name):
        logmod.warn("backends", f"model not found: models/{name}")
        return None
    return name


def load_pytorch_brain(name, step, threads):
    """Load Brain for name at step. On non-vanilla failure, scan other models
    by recency and load the first vanilla one. Returns (brain, name, step)
    or raises the original RuntimeError if nothing vanilla can be loaded."""
    def _try(n, s):
        ck = checkpoints.path_for(n, s)
        mp = os.path.join(paths.model_dir(n), "neuron_memory.json")
        return Brain(ck, threads=threads, memory=load_memory(mp))

    try:
        return (_try(name, step), name, int(step))
    except RuntimeError as e:
        if "PyTorch inference is not enabled" not in str(e):
            raise
        original_exc = e
        original_name = name

    candidates = []
    for n in models.list_models():
        if n == original_name:
            continue
        if not checkpoints.list_steps(n):
            continue
        st = train_csv.file_stat(n)
        candidates.append((st.st_mtime if st else 0, n))
    candidates.sort(reverse=True)
    for _, n in candidates:
        s = checkpoints.latest_step(n)
        if s is None:
            continue
        try:
            brain = _try(n, s)
            logmod.warn("backends", f"pytorch: '{original_name}' is non-vanilla; auto-fell-back to '{n}' step {s}")
            return (brain, n, int(s))
        except RuntimeError as e2:
            if "PyTorch inference is not enabled" in str(e2):
                continue
            raise
    raise original_exc


def resolve_c_model_bin(name):
    if name and os.path.isfile(name): return name
    if name and models.exists(name) and binr.exists(name):
        return paths.bin_path(name)
    candidates = []
    for n in models.list_models():
        if not binr.exists(n): continue
        bp = paths.bin_path(n)
        try: st = os.stat(bp)
        except OSError: continue
        candidates.append((st.st_mtime, bp))
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else None


def resolve_c_engine_exe(explicit):
    if explicit and os.path.isfile(explicit): return explicit
    for e in engine.engines():
        ap = os.path.abspath(e.get("path") or "")
        if os.path.isfile(ap): return ap
    return None
