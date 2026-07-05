# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Shared optimizer builder for trainers. Muon (Newton-Schulz orthogonalized momentum,
#   torch native) on 2D hidden weights, AdamW on embeddings/norms/1D params. RMS-matched
#   lr adjustment so a single AdamW-scale schedule drives both groups. Wrapper exposes
#   one optimizer surface (step, zero_grad, param_groups, state_dict) so trainers and
#   save() never branch on optimizer kind.
# veritate_core/plugin/optim.py
# ------------------------------------------------------------------------------------
# Imports:

import torch

# ------------------------------------------------------------------------------------
# Constants

MUON_ADJUST_LR = "match_rms_adamw"
MUON_MOMENTUM = 0.95
ADAMW_EPS = 1e-6
EMB_NAME_TAG = "emb"

# ------------------------------------------------------------------------------------
# Functions


class MuonAdamW:
    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self):
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def build_muon(model, args):
    hidden, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hidden if p.ndim == 2 and EMB_NAME_TAG not in name else rest).append(p)
    muon = torch.optim.Muon(
        hidden, lr=args.base_lr, weight_decay=args.weight_decay,
        momentum=MUON_MOMENTUM, adjust_lr_fn=MUON_ADJUST_LR)
    adamw = torch.optim.AdamW(
        rest, lr=args.base_lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2), eps=ADAMW_EPS)
    return MuonAdamW(muon, adamw)
