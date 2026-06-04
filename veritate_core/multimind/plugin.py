# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - MultiMind plugin. Bundles: (a) gate-bias hook from a frozen probe,
#   (b) per-region LoRA sleep adapter manager, (c) persistence helpers.
# - Talks to any multimind-compatible model class via the contract surface
#   (`set_gate_bias_provider`, `region_names`, `gate_g`, `blocks`).
#   See documentation/multimind/contract.md.
# veritate_core/multimind/plugin.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import torch
import torch.nn as nn

# ------------------------------------------------------------------------------------
# Constants

ADAPTER_FILENAME_FMT = "multimind_adapter_{region}.pt"
CONTRACT_METHODS     = ("set_gate_bias_provider",)
CONTRACT_ATTRS       = ("region_names", "blocks")
LORA_INIT_STD        = 0.02

# ------------------------------------------------------------------------------------
# Functions


def _validate(model):
    for m in CONTRACT_METHODS:
        if not callable(getattr(model, m, None)):
            raise TypeError(f"model is not multimind-compatible: missing method {m!r}")
    for a in CONTRACT_ATTRS:
        if not hasattr(model, a):
            raise TypeError(f"model is not multimind-compatible: missing attribute {a!r}")


class _LoRA(nn.Module):
    def __init__(self, hidden, rank):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(hidden, rank))
        self.B = nn.Parameter(torch.randn(rank, hidden) * LORA_INIT_STD)

    def forward(self, x):
        return (x @ self.A) @ self.B


class MultiMindPlugin:
    def __init__(self):
        self.probe   = None
        self.loras   = None
        self._orig_forwards = None

    def attach(self, model, probe_path):
        _validate(model)
        probe = torch.load(probe_path, map_location="cpu", weights_only=False)
        if hasattr(probe, "eval"):
            probe.eval()
            for p in probe.parameters():
                p.requires_grad_(False)
        self.probe = probe
        n_experts = len(model.region_names)

        def provider(tokens):
            with torch.no_grad():
                v = probe(tokens) if not hasattr(probe, "valence") else probe.valence(tokens)
            if v.dim() == 1:
                v = v.unsqueeze(-1)
            g = getattr(model, "gate_g", None)
            if g is not None:
                return g.unsqueeze(0) * v
            return v.expand(tokens.shape[0], n_experts).clone()

        model.set_gate_bias_provider(provider)

    def detach(self, model):
        _validate(model)
        model.set_gate_bias_provider(None)
        self.probe = None

    def sleep(self, model, buffer, lr, steps, rank, save_dir=None):
        _validate(model)
        self._install_loras(model, rank)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in self.loras.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(self.loras.parameters(), lr=lr, weight_decay=0.0)
        model.train()
        for _ in range(steps):
            for item in buffer:
                inp, tgt = item[0], item[1]
                _, loss = model(inp, targets=tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        for p in model.parameters():
            p.requires_grad_(True)
        if save_dir is not None:
            self._save(model, save_dir)

    def wake(self, model, save_dir, rank):
        _validate(model)
        self._install_loras(model, rank)
        for i, name in enumerate(model.region_names):
            path = os.path.join(save_dir, ADAPTER_FILENAME_FMT.format(region=name))
            if os.path.isfile(path):
                blob = torch.load(path, map_location="cpu", weights_only=True)
                self.loras[i].A.data.copy_(blob["A"])
                self.loras[i].B.data.copy_(blob["B"])

    def _install_loras(self, model, rank):
        if self.loras is not None:
            return
        device = next(model.parameters()).device
        hidden = model.blocks[0].n1.weight.shape[0]
        self.loras = nn.ModuleList([_LoRA(hidden, rank) for _ in model.region_names]).to(device)
        n_experts = len(model.region_names)
        self._orig_forwards = []
        for L, blk in enumerate(model.blocks):
            orig = blk.forward
            lora_pool = self.loras
            def _patched(x, gate_bias=None, _orig=orig, _pool=lora_pool, _n=n_experts):
                y = _orig(x, gate_bias=gate_bias)
                delta = sum(lora(y) for lora in _pool) / float(_n)
                return y + delta
            blk.forward = _patched
            self._orig_forwards.append(orig)

    def _save(self, model, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for i, name in enumerate(model.region_names):
            path = os.path.join(save_dir, ADAPTER_FILENAME_FMT.format(region=name))
            torch.save({"A": self.loras[i].A.detach().cpu(),
                        "B": self.loras[i].B.detach().cpu()}, path)
