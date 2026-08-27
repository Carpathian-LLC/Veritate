# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - weight fusion for consolidation (IDEA 20 T3 m1, the "merge fuse" step of the spec:
#   theta <- alpha*theta_ft + (1-alpha)*theta_prev). Interpolating a consolidated model
#   back toward its pre-sleep weights recovers base capability while keeping most of
#   what the run bound.
# - why it is the speed lever, not just a forgetting control: consolidation runs at
#   lr 5e-6 ONLY to avoid damaging the base, and that low rate is what forces ~129
#   epochs (13.8 h on cardinal at 260 B/s against the feature's own 10-20 min/night
#   design target). Bounding the damage after the fact instead lets the run use a rate
#   two orders of magnitude higher and bind in a few epochs.
# - alpha=1.0 is the tuned model, alpha=0.0 is the base. Non-float tensors, shape
#   mismatches and keys absent from the base take the tuned side unchanged: a fused
#   checkpoint must stay loadable by the same resume path.
# - CLI entry point: tools/fuse_checkpoints.py
# veritate_mri/training/fuse.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import torch
from readers import paths

# ------------------------------------------------------------------------------------
# Constants

BASE_PREFIX = "base."

# ------------------------------------------------------------------------------------
# Functions


def _strip_base(sd):
    """QAT-wrapped checkpoints nest the trunk under `base.`; resume strips it, so
    fusion has to compare the same key space on both sides."""
    if any(k.startswith(BASE_PREFIX) for k in sd):
        return {k[len(BASE_PREFIX):]: v for k, v in sd.items() if k.startswith(BASE_PREFIX)}
    return sd


def fuse_states(base_sd, tuned_sd, alpha):
    """theta <- alpha*tuned + (1-alpha)*base, elementwise over float tensors.

    Interpolation runs in float32 and casts back, so a bf16/fp16 checkpoint does not
    lose the update to rounding at small alpha."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    base_sd, tuned_sd = _strip_base(base_sd), _strip_base(tuned_sd)
    out, fused, passed = {}, 0, 0
    for k, tv in tuned_sd.items():
        bv = base_sd.get(k)
        if (bv is None or not torch.is_tensor(tv) or not torch.is_floating_point(tv)
                or bv.shape != tv.shape):
            out[k] = tv
            passed += 1
            continue
        merged = bv.to(torch.float32) * (1.0 - alpha) + tv.to(torch.float32) * alpha
        out[k] = merged.to(tv.dtype)
        fused += 1
    return out, {"fused": fused, "passed_through": passed}


def fuse(name, base_step, tuned_step, alpha, out_step=None, device="cpu"):
    """Write a fused checkpoint for `name`. Returns (out_step, stats).

    The optimizer state is deliberately NOT carried: it belongs to the tuned
    trajectory and would be wrong for the fused weights."""
    base = torch.load(paths.checkpoint_path(name, base_step), map_location=device,
                      weights_only=False)
    tuned = torch.load(paths.checkpoint_path(name, tuned_step), map_location=device,
                       weights_only=False)
    merged, stats = fuse_states(base["model"], tuned["model"], alpha)
    out_step = int(tuned_step if out_step is None else out_step)
    path = paths.checkpoint_path(name, out_step)
    tmp = path + ".tmp"
    torch.save({"model": merged, "step": out_step}, tmp)
    os.replace(tmp, path)
    return out_step, stats
