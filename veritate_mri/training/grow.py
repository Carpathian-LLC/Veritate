# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Function-preserving FFN widening (Net2Net) on a saved state_dict, plus the
#   val-flatten detector that decides WHEN to widen.
# - Widening rule: new ff.up rows get fresh init, new ff.down columns are exactly
#   zero. Output is bit-identical at the widen step, and the new up rows still
#   receive gradient because d(loss)/d(down_col) is nonzero.
# - The detector is the point. Growth measured 3.6x steps-to-target but cost
#   1.66x total compute because the small stage ran ~950 steps past saturation
#   (successes.md 2026-07-25). Growth only nets compute if each stage stops the
#   moment its val curve flattens, so that decision cannot be a hand-picked
#   step count.
# veritate_mri/training/grow.py
# ------------------------------------------------------------------------------------
# Imports:

import torch

# ------------------------------------------------------------------------------------
# Constants

UP_SUFFIX      = ".ff.up.weight"
DOWN_SUFFIX    = ".ff.down.weight"
FLAT_WINDOW    = 4
FLAT_REL_GAIN  = 0.005
INIT_STD       = 0.02

# ------------------------------------------------------------------------------------
# Functions


def ffn_widths(state_dict):
    """Per-layer FFN width, keyed by layer index, for layers that carry an up/down pair."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith(UP_SUFFIX):
            out[int(k.split(".")[1])] = int(v.shape[0])
    return out


def widen_state_dict(state_dict, target_ffn, generator=None):
    """Widen every dense FFN to target_ffn, preserving the function exactly.

    New up rows are freshly initialized; new down columns are zero, so the layer
    output is unchanged at the widen step. Layers already at or above target, and
    layers with no up/down pair (product-key memory, MoE), are left alone.
    """
    sd = dict(state_dict)
    widened = []
    for layer, width in sorted(ffn_widths(sd).items()):
        if width >= target_ffn:
            continue
        up_key, down_key = f"blocks.{layer}{UP_SUFFIX}", f"blocks.{layer}{DOWN_SUFFIX}"
        if down_key not in sd:
            continue
        up, down = sd[up_key], sd[down_key]
        extra = target_ffn - width
        new_up = torch.empty(extra, up.shape[1], dtype=up.dtype, device=up.device)
        new_up.normal_(mean=0.0, std=INIT_STD, generator=generator)
        sd[up_key] = torch.cat([up, new_up], dim=0).contiguous()
        new_down = torch.zeros(down.shape[0], extra, dtype=down.dtype, device=down.device)
        sd[down_key] = torch.cat([down, new_down], dim=1).contiguous()
        widened.append(layer)
    return sd, widened


def widen_model(model, target_ffn, generator=None):
    """Widen a LIVE model's dense FFNs in place, preserving the function.

    Swaps the up/down Parameters rather than rebuilding the model, so a training
    loop can grow mid-run and only has to rebuild the optimizer afterwards.
    Returns the layer indices that grew.
    """
    import torch.nn as nn

    widened = []
    for i, blk in enumerate(model.blocks):
        ff = blk.ff
        if getattr(ff, "probe_weights", None) is None or ff.probe_weights() is None:
            continue
        up, down = ff.up, ff.down
        cur = int(up.weight.shape[0])
        if cur >= target_ffn:
            continue
        extra = target_ffn - cur
        new_rows = torch.empty(extra, up.weight.shape[1],
                               dtype=up.weight.dtype, device=up.weight.device)
        new_rows.normal_(mean=0.0, std=INIT_STD, generator=generator)
        up.weight = nn.Parameter(torch.cat([up.weight.data, new_rows], dim=0).contiguous())
        up.out_features = target_ffn
        new_cols = torch.zeros(down.weight.shape[0], extra,
                               dtype=down.weight.dtype, device=down.weight.device)
        down.weight = nn.Parameter(torch.cat([down.weight.data, new_cols], dim=1).contiguous())
        down.in_features = target_ffn
        widened.append(i)
    model.ffn = target_ffn
    model.ffn_per_layer = [target_ffn] * model.layers
    return widened


def val_has_flattened(val_losses, window=FLAT_WINDOW, rel_gain=FLAT_REL_GAIN):
    """True once the last `window` validations have stopped buying real improvement.

    Compares the best val in the trailing window against the best before it; a
    relative gain under rel_gain means the stage is saturated and every further
    step is compute spent at the CHEAP shape for nothing.
    """
    if len(val_losses) < window * 2:
        return False
    # Compare against the IMMEDIATELY preceding window, not all history: early
    # large gains would otherwise mask a curve that has since gone flat.
    prior = min(val_losses[-2 * window:-window])
    recent = min(val_losses[-window:])
    if prior <= 0:
        return False
    return (prior - recent) / prior < rel_gain


def stage_compute(params, steps):
    """Relative compute for a stage: params x steps. Growth only wins on this sum."""
    return float(params) * float(steps)
