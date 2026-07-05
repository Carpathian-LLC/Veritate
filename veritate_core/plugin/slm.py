# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Selective language modeling (RHO-1 class). A frozen reference model scores
#   each token; the student trains only on the tokens where its own loss most
#   exceeds the reference (the surprising fraction), masking the rest out of
#   the loss. Reference forward is no-grad; the mask is detached so gradients
#   flow only through the student's kept tokens.
# veritate_core/plugin/slm.py
# ------------------------------------------------------------------------------------
# Imports:

import glob
import os
import re

import torch
import torch.nn.functional as F

from veritate_core import model as _model_mod

# ------------------------------------------------------------------------------------
# Constants

KEEP_FRAC_DEFAULT = 0.6
SHAPE_KEYS = ("hidden", "layers", "ffn", "heads", "seq")

# ------------------------------------------------------------------------------------
# Functions


def latest_checkpoint(model_dir):
    steps = []
    for p in glob.glob(os.path.join(model_dir, "checkpoints", "step_*.pt")):
        m = re.search(r"step_(\d+)\.pt$", p)
        if m:
            steps.append((int(m.group(1)), p))
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {model_dir}")
    return max(steps)[1]


def load_reference(model_dir, device):
    ckpt = torch.load(latest_checkpoint(model_dir), map_location="cpu", weights_only=False)
    a = ckpt["args"]
    ref = _model_mod.Veritate(vocab=_model_mod.VOCAB_BYTE_LEVEL,
                              **{k: a[k] for k in SHAPE_KEYS})
    ref.load_state_dict(ckpt["model"], strict=True)
    ref.eval().requires_grad_(False)
    return ref.to(device)


def selective_loss(ref, tokens, targets, logits, keep_frac=KEEP_FRAC_DEFAULT):
    V = logits.size(-1)
    flat_t = targets.reshape(-1)
    ce_s = F.cross_entropy(logits.reshape(-1, V), flat_t, ignore_index=-1, reduction="none")
    with torch.no_grad():
        ref_logits, _ = ref(tokens)
        ce_r = F.cross_entropy(ref_logits.reshape(-1, V), flat_t, ignore_index=-1, reduction="none")
        excess = ce_s.detach() - ce_r
        k = max(1, int(keep_frac * excess.numel()))
        thresh = excess.kthvalue(excess.numel() - k + 1).values
        mask = (excess >= thresh).float()
    return (ce_s * mask).sum() / mask.sum().clamp_min(1.0)
