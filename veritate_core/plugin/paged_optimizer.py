# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Decoupled-AdamW whose optimizer state (exp_avg, exp_avg_sq) lives in mmap-backed
#   files on disk instead of RAM. On unified-memory hosts the Adam moment buffers are
#   the largest single bucket (8 B/param at fp32); paging them to NVMe drops resident
#   optimizer memory toward zero so a model whose full Adam state exceeds the unified
#   pool still trains. Weights and grads stay resident: paging the optimizer alone
#   does not make a model fit if weights+grads already exceed the budget.
# - The OS pages the MAP_SHARED files: hot moment pages stay in the page cache, cold
#   pages spill to disk under pressure. Each step touches the full state, so step time
#   is bound by NVMe bandwidth at scale. This is the speed/size trade the planner's
#   TIER_PAGE rung buys; bench.run measures the real tok/s so the cost is visible.
# - Update math is the standard fp32 decoupled-AdamW step, identical to
#   torch.optim.AdamW; tests/plugin_contract/test_paged_optimizer.py pins parity.
# - state_dict() carries only step counts + the state directory, never the moment
#   buffers, so checkpoints stay tiny and resume rebinds the on-disk files in place.
# veritate_core/plugin/paged_optimizer.py
# ------------------------------------------------------------------------------------
# Imports

import math
import os
import shutil
import tempfile

import torch

# ------------------------------------------------------------------------------------
# Constants

MOMENT_DTYPE   = torch.float32
MOMENT_BYTES   = torch.finfo(MOMENT_DTYPE).bits // 8
COMPUTE_DTYPE  = torch.float32
HOST_DEVICE    = "cpu"
DEFAULT_BETAS  = (0.9, 0.95)
DEFAULT_EPS    = 1e-6
DEFAULT_WD     = 0.0
M_PREFIX       = "exp_avg_"
V_PREFIX       = "exp_avg_sq_"
STATE_SUFFIX   = ".bin"
SCRATCH_PREFIX = "veritate_paged_opt_"

# ------------------------------------------------------------------------------------
# Functions


def _file_backed(path, numel):
    numel = int(numel)
    fresh = not (os.path.exists(path) and os.path.getsize(path) == numel * MOMENT_BYTES)
    t = torch.from_file(path, shared=True, size=numel, dtype=MOMENT_DTYPE)
    if fresh:
        t.zero_()
    return t


class PagedAdamW(torch.optim.Optimizer):
    """AdamW with exp_avg / exp_avg_sq stored in mmap-backed files under `state_dir`
    so the OS pages them out of RAM. Pass an explicit `state_dir` (kept across resume)
    for a real run; omit it for a throwaway run and a temp dir is created and removed
    on close()."""

    def __init__(self, params, lr, betas=DEFAULT_BETAS, eps=DEFAULT_EPS,
                 weight_decay=DEFAULT_WD, state_dir=None):
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        if state_dir is None:
            self._state_dir = tempfile.mkdtemp(prefix=SCRATCH_PREFIX)
            self._owns_dir  = True
        else:
            os.makedirs(state_dir, exist_ok=True)
            self._state_dir = state_dir
            self._owns_dir  = False
        self._bind_state()

    def _bind_state(self):
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                st["step"] = st.get("step", 0)
                st["exp_avg"]    = _file_backed(self._path(M_PREFIX, idx), p.numel())
                st["exp_avg_sq"] = _file_backed(self._path(V_PREFIX, idx), p.numel())
                idx += 1

    def _path(self, prefix, idx):
        return os.path.join(self._state_dir, f"{prefix}{idx}{STATE_SUFFIX}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd  = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                st["step"] += 1
                t = st["step"]
                grad = p.grad.detach().to(HOST_DEVICE, dtype=COMPUTE_DTYPE).reshape(-1)
                m = st["exp_avg"]
                v = st["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t
                denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)
                update = (m / denom).mul_(lr / bias1)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update.reshape(p.shape).to(p.device, dtype=p.dtype), alpha=-1.0)
        return loss

    def state_dict(self):
        return {
            "state_dir": self._state_dir,
            "steps": [self.state[p]["step"]
                      for group in self.param_groups for p in group["params"]],
            "param_groups": [{k: g[k] for k in ("lr", "betas", "eps", "weight_decay")}
                             for g in self.param_groups],
        }

    def load_state_dict(self, state):
        steps = state.get("steps", [])
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if idx < len(steps):
                    self.state[p]["step"] = steps[idx]
                idx += 1
        saved_dir = state.get("state_dir")
        if saved_dir and os.path.isdir(saved_dir):
            self._state_dir = saved_dir
            self._owns_dir  = False
            self._bind_state()

    def close(self):
        if self._owns_dir and os.path.isdir(self._state_dir):
            shutil.rmtree(self._state_dir, ignore_errors=True)
            self._owns_dir = False

    def __del__(self):
        self.close()
