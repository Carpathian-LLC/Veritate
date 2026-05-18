# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Kangaroo self-speculative draft. Reuses the target model's first
#   exit_layer_frac × L blocks (frozen) plus a small trainable adapter MLP
#   to bridge the early-layer residual to a distribution close to the
#   target's final output. Model-agnostic via the cross-model contract
#   (embed, run_blocks, ensure_context, project_byte0); never inspects
#   model attributes (preflight rule 11a).
# - Source: arXiv 2404.18911 (Kangaroo: Lossless Self-Speculative Decoding
#   via Double Early Exiting). Published 2.04x speedup on Llama-7B with
#   88.7% fewer added params than Medusa-1.
# veritate_mri/decode/kangaroo.py
# ------------------------------------------------------------------------------------
# Imports:

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------
# Constants

# ------------------------------------------------------------------------------------
# Functions


class KangarooAdapter(nn.Module):
    """A small MLP that bridges the residual at layer L/4 to a representation
    the LM head can read. Two linear layers + a residual + a final norm.
    Total params ≈ 2 * hidden^2 + 2*hidden — for 800M's hidden=1536 that's
    ~4.7M params (0.59% of target).
    """

    def __init__(self, hidden: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden    = hidden
        self.expansion = expansion
        self.fc1 = nn.Linear(hidden, hidden * expansion, bias=False)
        self.fc2 = nn.Linear(hidden * expansion, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Two-layer MLP with GELU. Output is residual-added + normed so it
        # produces "trunk-final-shape" hidden states.
        h = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return self.norm(x + h)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class KangarooDraft:
    """Wraps a FROZEN target model + a trained Kangaroo adapter. Forwards
    through the target's first L_exit blocks, applies the adapter, projects
    through the target's project_byte0 contract."""

    def __init__(self, target_model, adapter: KangarooAdapter,
                 exit_layer_frac: float = 0.25):
        self.model   = target_model
        self.adapter = adapter
        self.exit_layer = max(1, int(round(target_model.layers * exit_layer_frac)))

    @torch.no_grad()
    def draft_one(self, tokens: torch.Tensor) -> torch.Tensor:
        m = self.model
        m.ensure_context(tokens.size(1))
        x = m.embed(tokens)
        for L in range(self.exit_layer):
            x = m.run_block(x, L)
        last = x[:, -1, :]
        bridged = self.adapter(last)
        return m.project_byte0(bridged.unsqueeze(1))[:, 0, :]

    def param_count(self) -> int:
        return self.adapter.param_count()
