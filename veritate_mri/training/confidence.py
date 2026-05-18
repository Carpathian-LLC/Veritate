# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Calibrated confidence score + its four components (margin, entropy, lens
#   consistency, residual stability). Single owning module: the dumper and the
#   live Brain both call into here so a frame's `confidence` reads the same way
#   regardless of source (rule 23: training-time and inference-time emit the
#   same fields).
# - Calibration weights live in models/<name>/confidence_weights.json. When the
#   file is absent the formula falls back to the engine's main.c constants so
#   the score remains comparable across runs without calibration.
# veritate_mri/training/confidence.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import math
import os

import torch

# ------------------------------------------------------------------------------------
# Constants

WEIGHTS_FILENAME      = "confidence_weights.json"
WEIGHTS_FALLBACK      = (0.5, 0.5, 0.5, 0.5, -1.0)  # (w_M, w_E, w_L, w_S, b) matches engine/src/main.c
ROUND_DIGITS          = 4
PEARSON_EPS           = 1e-12
SIGMA_EPS             = 1e-6

# ------------------------------------------------------------------------------------
# Functions

def load_weights(out_dir):
    """Load calibrated weights from `<out_dir>/confidence_weights.json` if present.
    Returns (w_M, w_E, w_L, w_S, b, loaded)."""
    path = os.path.join(out_dir, WEIGHTS_FILENAME)
    if not os.path.isfile(path):
        return (*WEIGHTS_FALLBACK, False)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cw = json.load(f)
        return (float(cw["w_M"]), float(cw["w_E"]),
                float(cw["w_L"]), float(cw["w_S"]),
                float(cw["b"]), True)
    except (OSError, KeyError, ValueError):
        return (*WEIGHTS_FALLBACK, False)


def _sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def compute_components(last_logits, probs, nxt, lens_argmax, res_stack, embed_row, vocab):
    """Compute the four confidence components for a single byte.

    Args:
        last_logits:  1-D tensor, raw logits over vocab for the last position.
        probs:        1-D tensor, softmax of last_logits (caller-provided to avoid recompute).
        nxt:          int, the sampled byte.
        lens_argmax:  list[int] of length layers, argmax byte from each layer's lens.
        res_stack:    (layers, hidden) tensor of post-block residuals at the last position.
        embed_row:    (hidden,) tensor, embed_w[nxt] for the sampled byte.
        vocab:        int, vocab size (for entropy normalization).

    Returns:
        dict with keys margin, entropy_score, lens_consistency, residual_stab.
    """
    top1, _ = torch.topk(last_logits, 2)
    sigma = float(last_logits.std(unbiased=False).item())
    margin = float((top1[0] - top1[1]).item()) / sigma if sigma > SIGMA_EPS else 0.0

    H = float(-(probs * (probs + PEARSON_EPS).log()).sum().item())
    entropy_score = 1.0 - (H / math.log(vocab))
    entropy_score = max(0.0, min(1.0, entropy_score))

    layers = res_stack.shape[0]
    lens_consistency = sum(1 for am in lens_argmax if am == nxt) / float(layers)

    # residual_stab: mean Pearson r between consecutive layers' (residual * embed_row)
    # after mean-centering. Captures whether the layer-by-layer build-up of the
    # sampled byte's residual signal is smooth (high r) or thrashing (low r).
    vec   = res_stack.float() * embed_row.float()
    vec_c = vec - vec.mean(dim=1, keepdim=True)
    norms = vec_c.pow(2).sum(dim=1).sqrt()
    num   = (vec_c[:-1] * vec_c[1:]).sum(dim=1)
    den   = (norms[:-1] * norms[1:]).clamp(min=PEARSON_EPS)
    r_pair = (num / den).clamp(-1.0, 1.0)
    residual_stab = float(r_pair.mean().item()) if r_pair.numel() > 0 else 0.0

    return {
        "margin":           margin,
        "entropy_score":    entropy_score,
        "lens_consistency": lens_consistency,
        "residual_stab":    residual_stab,
    }


def score(components, weights=None):
    """Combine four components into a [0, 1] confidence using calibrated weights.
    Falls back to the engine's main.c formula when weights is None or not loaded."""
    m, e, l, s = (components["margin"], components["entropy_score"],
                  components["lens_consistency"], components["residual_stab"])
    if weights is not None:
        w_M, w_E, w_L, w_S, b, loaded = weights
        if loaded:
            z = w_M * m + w_E * e + w_L * l + w_S * s + b
            return _sigmoid(z)
    z = 0.5 * (m + e + l + s) - 1.0
    return _sigmoid(z)


def frame_fields(last_logits, probs, nxt, lens_argmax, res_stack, embed_row, vocab, weights=None):
    """One-shot helper for trainer dump + live Brain. Returns the five rounded
    fields that go straight into the TFRM frame: margin, entropy, lens_consistency,
    residual_stab, confidence."""
    c = compute_components(last_logits, probs, nxt, lens_argmax, res_stack, embed_row, vocab)
    out = {
        "margin":           round(c["margin"],           ROUND_DIGITS),
        "entropy":          round(c["entropy_score"],    ROUND_DIGITS),
        "lens_consistency": round(c["lens_consistency"], ROUND_DIGITS),
        "residual_stab":    round(c["residual_stab"],    ROUND_DIGITS),
        "confidence":       round(score(c, weights),     ROUND_DIGITS),
    }
    return out
