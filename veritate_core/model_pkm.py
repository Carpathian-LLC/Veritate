# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Product-key memory FFN (Lample 2019 / Meta memory-layers class). Replaces the
#   dense FFN of a block with a weighted top-k read over sub_keys^2 learned value
#   slots. Two sqrt-sized sub-key searches locate the top-k slots without scoring
#   all sub_keys^2 candidates, so per-token read cost is O(sub_keys*key_dim +
#   top_k*hidden) while capacity is O(sub_keys^2 * hidden).
# - Decouples capacity from bytes-streamed: the search matrices are small and
#   cache-resident, only top_k value rows move per token. Targets bandwidth-bound
#   edge CPUs, where decode cost is bytes read, not arithmetic.
# - Drop-in for FFN/MoEFFN: same (hidden, ffn) leading signature, exposes
#   _last_l1 so post_l1_sum() over a mixed trunk stays uniform. ffn is accepted
#   for slot compatibility and does not size the layer; capacity comes from
#   sub_keys and hidden.
# - Fixed shapes throughout and F.embedding for the gather, per the MPS rules.
# veritate_core/model_pkm.py
# ------------------------------------------------------------------------------------
# Imports:

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ACT_DEFAULT, REG_DEFAULT, QuantLinear, RMSNorm

# ------------------------------------------------------------------------------------
# Constants

PKM_SUB_KEYS  = 256
PKM_TOP_K     = 32
PKM_HEADS     = 4
PKM_KEY_DIM   = 128
PKM_INIT_STD  = 0.02
PKM_GATE_TOPK      = "topk"
PKM_GATE_THRESHOLD = "threshold"
PKM_GATES          = (PKM_GATE_TOPK, PKM_GATE_THRESHOLD)
PKM_GATE_TAU       = 0.5
PKM_THETA_INIT     = 0.0

# ------------------------------------------------------------------------------------
# Functions


class ProductKeyMemory(nn.Module):
    def __init__(self, hidden, ffn, activation=ACT_DEFAULT, capture_l1=False,
                 reg_mode=REG_DEFAULT, sub_keys=PKM_SUB_KEYS, top_k=PKM_TOP_K,
                 heads=PKM_HEADS, key_dim=PKM_KEY_DIM, gate=PKM_GATE_TOPK,
                 sparse_values=False):
        super().__init__()
        if key_dim % 2:
            raise ValueError(f"key_dim must be even, got {key_dim}")
        if top_k > sub_keys:
            raise ValueError(f"top_k {top_k} exceeds sub_keys {sub_keys}")
        if gate not in PKM_GATES:
            raise ValueError(f"unknown gate: {gate!r}; expected one of {PKM_GATES}")
        half = key_dim // 2
        self.query      = QuantLinear(hidden, heads * key_dim, bias=False)
        self.qnorm      = RMSNorm(key_dim)
        self.sub_key    = nn.Parameter(torch.randn(2, heads, sub_keys, half) * PKM_INIT_STD)
        self.values     = nn.Embedding(sub_keys * sub_keys, hidden)
        self.hidden     = hidden
        self.ffn        = ffn
        self.activation = activation
        self.reg_mode   = reg_mode
        self.sub_keys   = sub_keys
        self.top_k      = top_k
        self.heads      = heads
        self.key_dim    = key_dim
        self.half       = half
        self.gate       = gate
        self.sparse_values = bool(sparse_values)
        self.capture_l1 = bool(capture_l1)
        self._last_l1   = None
        self._last_fired = None
        # Firing threshold on the standardized candidate score, so it is scale
        # free and theta=0 starts by keeping about half the candidates.
        self.theta = nn.Parameter(torch.full((heads,), PKM_THETA_INIT)) \
            if gate == PKM_GATE_THRESHOLD else None

    def forward(self, x):
        b, t, _ = x.shape
        q = self.qnorm(self.query(x).view(b, t, self.heads, self.key_dim))
        s1 = torch.einsum("bthd,hmd->bthm", q[..., :self.half], self.sub_key[0])
        s2 = torch.einsum("bthd,hmd->bthm", q[..., self.half:], self.sub_key[1])
        v1, i1 = s1.topk(self.top_k, dim=-1)
        v2, i2 = s2.topk(self.top_k, dim=-1)
        cand   = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)
        slots  = (i1.unsqueeze(-1) * self.sub_keys + i2.unsqueeze(-2)).flatten(-2)
        score, pos = cand.topk(self.top_k, dim=-1)
        weight = F.softmax(score, dim=-1)
        if self.gate == PKM_GATE_THRESHOLD:
            weight = weight * self._fire(score)
            weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-6)
        rows   = F.embedding(slots.gather(-1, pos), self.values.weight,
                             sparse=self.sparse_values)
        if self.capture_l1:
            self._last_l1 = weight.abs().mean()
        return (rows * weight.unsqueeze(-1)).sum(dim=-2).sum(dim=-2)

    def _fire(self, score):
        """Hard on/off per candidate, straight-through so theta still learns.
        Standardizing per token keeps the threshold scale free as scores drift."""
        z = (score - score.mean(dim=-1, keepdim=True)) / (score.std(dim=-1, keepdim=True) + 1e-6)
        margin = z - self.theta.view(1, 1, -1, 1)
        hard = (margin > 0).to(score.dtype)
        soft = torch.sigmoid(margin / PKM_GATE_TAU)
        self._last_fired = hard.detach().sum(dim=-1).mean()
        return hard + soft - soft.detach()

    def probe_module(self):
        """Addressing signal stands in for neuron activations: the query is what
        selects which slots fire, and it is the only per-token dense vector here."""
        return self.query

    def probe_weights(self):
        return None

    def read_bytes_per_token(self, weight_bytes):
        """Bytes streamed per token: the two sub-key searches plus top_k value rows."""
        search = 2 * self.heads * self.sub_keys * self.half
        gather = self.heads * self.top_k * self.hidden
        query  = self.hidden * self.heads * self.key_dim
        return (search + gather + query) * weight_bytes

    def capacity_params(self):
        """Total learned parameters held by the layer, most of which never stream."""
        return self.sub_keys * self.sub_keys * self.hidden
