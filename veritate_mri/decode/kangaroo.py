# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Kangaroo self-speculative draft. Reuses the target model's first
#   exit_layer_frac × L blocks (frozen) plus a small trainable adapter MLP
#   to bridge the early-layer residual to a distribution close to the
#   target's final output. The byte-0 projection is delegated to the
#   target via byte0_projector() — this module is MODEL-AGNOSTIC and works
#   with any byte-level variant (per preflight rule 11a).
# - Source: arXiv 2404.18911 ("Kangaroo: Lossless Self-Speculative Decoding
#   via Double Early Exiting"). Published 2.04× speedup on Llama-7B with
#   88.7% fewer added params than Medusa-1.
# - This module ONLY defines the architecture. Training lives in a separate
#   dispatch. Inference wiring as a stream_fast mode lands once a trained
#   adapter ckpt exists.
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
    """Wraps a FROZEN target model + a trained Kangaroo adapter. Acts as a
    "small drafter" for speculative decoding: forwards through the target's
    first L_exit blocks, applies the adapter, then projects through the
    target's byte-0 head via byte0_projector().

    The math: the adapter learns p_adapter(byte_t | x_<t) ≈ p_target(byte_t | x_<t)
    using only the first L_exit blocks. If the adapter is good, the
    speculator's drafts are accepted at high rate during verify.

    Model-agnostic. The target's specific output projection (canonical
    lm_head, MTP-routed lm_head, future variants) is handled by
    byte0_projector(); this class never inspects model attributes itself.
    Also model-agnostic on the forward path: it delegates to the target's
    `embed(tokens)` if available, otherwise builds the input via
    `tok_emb + pos_emb`. Each block is called via a model-supplied
    `forward_block(block, x)` callable if defined, else heuristically.
    """

    def __init__(self, target_model, adapter: KangarooAdapter,
                 exit_layer_frac: float = 0.25):
        from . import byte0_projector
        self.model   = target_model
        self.adapter = adapter
        # Exit after this many blocks (so layer indices 0..exit_layer-1 run).
        self.exit_layer = max(1, int(round(target_model.layers * exit_layer_frac)))
        self._project = byte0_projector(target_model)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the model's embedding step. Prefers the model's own
        `embed()` method (which knows about RoPE caches / pos_emb / etc.)
        if present. Falls back to tok_emb + pos_emb for plain transformers."""
        m = self.model
        if hasattr(m, "embed") and callable(m.embed):
            return m.embed(tokens)
        # Generic fallback: tok_emb + (optional) pos_emb
        x = m.tok_emb(tokens)
        if hasattr(m, "pos_emb"):
            T = tokens.size(1)
            positions = torch.arange(T, device=tokens.device).unsqueeze(0)
            x = x + m.pos_emb(positions)
        return x

    def _forward_block(self, block, x: torch.Tensor) -> torch.Tensor:
        """Call a block. Model variants whose blocks take extra args (e.g.
        RoPE cos/sin) expose those args via attributes on the model. The
        model class is the source of truth on its own block signature."""
        # If the model exposes a callable for invoking blocks, defer to it
        runner = getattr(self.model, "run_block", None)
        if callable(runner):
            return runner(block, x)
        # Generic two-paths: try (x), then (x, rope_cos, rope_sin) if the model
        # has those buffers
        try:
            return block(x)
        except TypeError:
            if hasattr(self.model, "rope_cos") and hasattr(self.model, "rope_sin"):
                return block(x, self.model.rope_cos, self.model.rope_sin)
            raise

    @torch.no_grad()
    def draft_one(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward through the first L_exit blocks + adapter + project.
        tokens: [B, T] long. Returns logits at the last position: [B, vocab]."""
        # Defensive: extend RoPE cache if model needs it
        if hasattr(self.model, "rope_cos") and hasattr(self.model, "extend_rope"):
            if tokens.size(1) > self.model.rope_cos.size(0):
                self.model.extend_rope(tokens.size(1))

        x = self._embed(tokens)
        for L in range(self.exit_layer):
            x = self._forward_block(self.model.blocks[L], x)

        # Adapter on the last position only — cheaper than running over the
        # full sequence.
        last = x[:, -1, :]                # [B, hidden]
        bridged = self.adapter(last)      # [B, hidden]
        logits = self._project(bridged)   # [B, vocab]
        return logits

    def param_count(self) -> int:
        return self.adapter.param_count()
