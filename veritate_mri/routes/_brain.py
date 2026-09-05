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

# NOTE: `inference.backends.pytorch` imports torch (~1.5 GB on Apple Silicon).
# Deferred to `load_pytorch_brain` so dashboard startup, settings, and other
# routes that only touch resolve_*/list helpers do not pay the torch tax. In
# minimal mode the brain never loads, so torch never imports.
from readers import bin as binr
from readers import checkpoints, engine, models, paths, train_csv
from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

AUTO_MODEL = "auto"
NEURON_MEMORY_NAME = "neuron_memory.json"
NON_VANILLA_MARKER = "PyTorch inference is not enabled"

# ------------------------------------------------------------------------------------
# Functions

def neuron_memory_path(name):
    return os.path.join(paths.model_dir(name), NEURON_MEMORY_NAME)


def resolve_pytorch_model(name):
    if name == AUTO_MODEL:
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
    """Load Brain for name at step. A step pruned out from under a cached
    default resolves to the newest on disk. On non-vanilla failure, scan other
    models by recency and load the first vanilla one. Returns (brain, name,
    step) or raises the original RuntimeError if nothing vanilla can be loaded."""
    # Lazy torch import: paying the ~1.5 GB tax only when a backend actually
    # spins up, not at every dashboard import.
    from inference.backends.pytorch import Brain, load_memory

    def _try(n, s):
        ck = checkpoints.path_for(n, s)
        return Brain(ck, threads=threads, memory=load_memory(neuron_memory_path(n)))

    live = checkpoints.resolve_step(name, step)
    if live != step:
        logmod.warn("backends", f"pytorch: {name} step {step} is gone; loading step {live}")
        step = live

    try:
        return (_try(name, step), name, int(step))
    except RuntimeError as e:
        if NON_VANILLA_MARKER not in str(e):
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
            if NON_VANILLA_MARKER in str(e2):
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
        ap = os.path.abspath(e["path"])
        if os.path.isfile(ap): return ap
    return None
