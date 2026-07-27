# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Shared optimizer builder for trainers. Muon (Newton-Schulz orthogonalized momentum)
#   on 2D hidden weights, AdamW on embeddings/norms/1D params. RMS-matched lr adjustment
#   so a single AdamW-scale schedule drives both groups. Wrapper exposes one optimizer
#   surface (step, zero_grad, param_groups, state_dict) so trainers and save() never
#   branch on optimizer kind.
# - Platform-robustness: Muon uses torch.optim.Muon when the installed torch exposes it,
#   otherwise a vendored fallback with identical math (so training never crashes on an
#   older torch). 8-bit AdamW uses bitsandbytes when it is importable AND on CUDA,
#   otherwise falls back to torch AdamW. Both fallbacks are silent-safe and logged.
# veritate_core/plugin/optim.py
# ------------------------------------------------------------------------------------
# Imports:

import math

import torch

# ------------------------------------------------------------------------------------
# Constants

MUON_ADJUST_LR = "match_rms_adamw"
MUON_MOMENTUM = 0.95
MUON_NESTEROV = True
MUON_NS_COEFFICIENTS = (3.4445, -4.775, 2.0315)
MUON_NS_STEPS = 5
MUON_EPS = 1e-7
ADAMW_EPS = 1e-6
EMB_NAME_TAG = "emb"

# ------------------------------------------------------------------------------------
# Vendored Muon fallback (used only when torch.optim.Muon is unavailable)
#
# Faithful reimplementation of torch.optim.Muon: quintic Newton-Schulz
# orthogonalization of the (Nesterov) momentum update, with the "match_rms_adamw"
# learning-rate adjustment. Kept behaviorally identical so runs are portable across
# torch versions.


def _zeropower_via_newtonschulz(grad, ns_coefficients, ns_steps, eps):
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        ortho_grad = ortho_grad.T
    ortho_grad = ortho_grad / ortho_grad.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if transposed:
        ortho_grad = ortho_grad.T
    return ortho_grad


def _adjust_lr(lr, adjust_lr_fn, param_shape):
    A, B = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class _VendoredMuon(torch.optim.Optimizer):
    """Drop-in stand-in for torch.optim.Muon for torch builds that lack it."""

    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=MUON_MOMENTUM,
                 nesterov=MUON_NESTEROV, ns_coefficients=MUON_NS_COEFFICIENTS,
                 eps=MUON_EPS, ns_steps=MUON_NS_STEPS, adjust_lr_fn=None):
        defaults = {
            "lr": lr, "weight_decay": weight_decay, "momentum": momentum, "nesterov": nesterov,
            "ns_coefficients": ns_coefficients, "eps": eps, "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf
                update = _zeropower_via_newtonschulz(
                    update, ns_coefficients, ns_steps, eps)
                adjusted_lr = _adjust_lr(lr, adjust_lr_fn, p.shape)
                p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-adjusted_lr)
        return loss


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


def _muon_cls():
    """Native torch.optim.Muon when present, else the vendored fallback."""
    native = getattr(torch.optim, "Muon", None)
    if native is not None:
        return native, "torch.optim.Muon"
    from runtime import logs as logmod
    logmod.warn("optim", "torch.optim.Muon unavailable; using vendored Muon fallback")
    return _VendoredMuon, "vendored"


def _build_adamw(params, args):
    """AdamW group. Honors use_8bit_adam via bitsandbytes when it is usable
    (importable + CUDA); otherwise falls back to torch AdamW."""
    lr = args.base_lr
    kwargs = {
        "lr": lr, "weight_decay": args.weight_decay,
        "betas": (args.beta1, args.beta2), "eps": ADAMW_EPS}
    if getattr(args, "use_8bit_adam", False):
        from runtime import logs as logmod
        if torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, **kwargs)
            except Exception as e:  # import or init failure -> graceful fallback
                logmod.warn("optim", "8-bit AdamW requested but bitsandbytes unusable "
                            f"({e}); falling back to torch AdamW")
        else:
            logmod.warn("optim", "8-bit AdamW requested but no CUDA device; "
                        "falling back to torch AdamW (8-bit optimizers require CUDA)")
    return torch.optim.AdamW(params, **kwargs)


def build_muon(model, args):
    hidden, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hidden if p.ndim == 2 and EMB_NAME_TAG not in name else rest).append(p)
    muon_cls, _ = _muon_cls()
    muon = muon_cls(
        hidden, lr=args.base_lr, weight_decay=args.weight_decay,
        momentum=MUON_MOMENTUM, adjust_lr_fn=MUON_ADJUST_LR)
    adamw = _build_adamw(rest, args)
    return MuonAdamW(muon, adamw)
