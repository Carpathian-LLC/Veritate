# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Trainer-agnostic mixed-corpus data loader. Single entry point: build a draw
#   callable that returns (toks, tgts) torch tensors, sampling per-sample from
#   N corpora with weights derived from size (default) or user-specified.
# - CLI spec format on `--corpus`:
#     "stem"                           -> single corpus (delegates to caller)
#     "stem1+stem2+stem3"              -> mix, size-proportional weights
#     "stem1:0.5,stem2:0.3,stem3:0.2"  -> mix, explicit weights (normalized to 1)
# - The actual file resolution is the caller's job (passed in as resolver_fn),
#   so this module stays decoupled from the platform's corpus layout.
# - Per-sample sampling (not per-batch) for maximum mix diversity inside each
#   gradient step.
# veritate_core/plugin/multicorpus.py
# ------------------------------------------------------------------------------------
# Imports

import os

import numpy as np
import torch

# ------------------------------------------------------------------------------------
# Constants

MIX_SEP    = "+"   # additive mix, weights derived from size
WEIGHT_SEP = ":"   # explicit weight per stem
LIST_SEP   = ","   # separates weighted stems in the explicit form

# ------------------------------------------------------------------------------------
# Functions

def is_mixed_spec(spec):
    return MIX_SEP in spec or LIST_SEP in spec


def parse_spec(spec):
    """Return list of (stem, weight_or_None). When weight is None, caller fills
    it from corpus size. Single-stem input returns [(spec, None)]."""
    spec = spec.strip()
    if LIST_SEP in spec:
        parts = [p.strip() for p in spec.split(LIST_SEP) if p.strip()]
        out = []
        for p in parts:
            if WEIGHT_SEP in p:
                stem, w = p.split(WEIGHT_SEP, 1)
                out.append((stem.strip(), float(w)))
            else:
                out.append((p.strip(), None))
        return out
    if MIX_SEP in spec:
        return [(s.strip(), None) for s in spec.split(MIX_SEP) if s.strip()]
    return [(spec, None)]


def resolve_and_weight(spec, resolver_fn):
    """Return list of (train_path, val_path, weight) summing to 1.

    resolver_fn(stem) -> (train_path, val_path). Caller-provided so this module
    is platform-agnostic.

    Weights:
      - explicit (WEIGHT_SEP form): used as given, then normalized to 1.
      - implicit (MIX_SEP form or single stem): size-proportional from train
        file bytes.
    """
    parsed = parse_spec(spec)
    resolved = []
    for stem, w in parsed:
        train, val = resolver_fn(stem)
        if train is None:
            raise FileNotFoundError(f"corpus stem not found: {stem!r}")
        resolved.append((train, val, w))

    if len(resolved) == 1:
        return [(resolved[0][0], resolved[0][1], 1.0)]

    have_explicit = any(w is not None for _, _, w in resolved)
    if have_explicit:
        if any(w is None for _, _, w in resolved):
            raise ValueError("mixed spec must use weights for all stems or none")
        weights = np.array([w for _, _, w in resolved], dtype=np.float64)
    else:
        weights = np.array([os.path.getsize(p) for p, _, _ in resolved], dtype=np.float64)

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("corpus weights sum to zero")
    weights /= total
    return [(p, v, float(w)) for (p, v, _), w in zip(resolved, weights)]


def make_mixed_loader(paths_with_weights, batch_size, seq, seed):
    """Build a draw callable that returns (toks, tgts) torch int64 tensors of
    shape [batch_size, seq]. Each sample is independently drawn from one of N
    corpora with the supplied weights.

    paths_with_weights: list of (train_path, val_path_unused, weight)
    """
    if not paths_with_weights:
        raise ValueError("no corpora provided")
    arrays = [np.memmap(p, dtype=np.uint8, mode="r") for p, _, _ in paths_with_weights]
    sizes  = [len(a) for a in arrays]
    weights = np.array([w for _, _, w in paths_with_weights], dtype=np.float64)
    for n in sizes:
        if n < seq + 2:
            raise ValueError(f"corpus too small for seq: {n} < {seq + 2}")
    rng = np.random.RandomState(seed)
    n_corp = len(arrays)
    total_bytes = sum(sizes)

    def draw():
        which = rng.choice(n_corp, size=batch_size, p=weights)
        toks = np.empty((batch_size, seq), dtype=np.int64)
        tgts = np.empty((batch_size, seq), dtype=np.int64)
        for b in range(batch_size):
            i = int(which[b])
            arr = arrays[i]
            s = int(rng.randint(0, sizes[i] - seq - 1))
            toks[b] = arr[s:s + seq]
            tgts[b] = arr[s + 1:s + 1 + seq]
        # CRITICAL: torch.tensor(np) copies, decoupling from the prefetcher's
        # buffer reuse pattern; torch.from_numpy(np) would alias and corrupt
        # under the Prefetcher thread.
        return torch.tensor(toks), torch.tensor(tgts)

    return draw, total_bytes


def format_mix_summary(paths_with_weights):
    """One-line human-readable mix summary for logging."""
    parts = []
    for p, _, w in paths_with_weights:
        stem = os.path.basename(p).rsplit("_train.bin", 1)[0]
        parts.append(f"{stem}:{w:.3f}")
    return ", ".join(parts)
