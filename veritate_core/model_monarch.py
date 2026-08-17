# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Monarch-factored FFN. Replaces each dense matmul with two block-diagonal
#   factors over transposed axes, so a features-wide layer costs
#   features*(s1+s2) parameters and MACs instead of features^2. Activation
#   shapes are unchanged, which is the whole point: the saving is arithmetic,
#   not a narrower layer.
# - Only pays above hidden ~700 on a bandwidth-poor CPU; below that the permute
#   and bmm overhead exceeds the FLOP cut (ideas.md IDEA 14, measured).
# - Drop-in for FFN/MoEFFN/ProductKeyMemory: same (hidden, ffn) leading
#   signature and the same _last_l1 / probe contract.
# veritate_core/model_monarch.py
# ------------------------------------------------------------------------------------
# Imports:

import torch
import torch.nn as nn

from .model import _ACT_FNS, ACT_DEFAULT, REG_DEFAULT, REG_MODES, group_penalty

# ------------------------------------------------------------------------------------
# Constants

MONARCH_INIT_STD = 0.02

# ------------------------------------------------------------------------------------
# Functions


def square_factors(n):
    """Most-square (s1, s2) with s1*s2 == n. Parameter cost is n*(s1+s2), so the
    closer to square the cheaper."""
    s1 = int(n ** 0.5)
    while s1 > 1 and n % s1:
        s1 -= 1
    if s1 < 2:
        raise ValueError(f"cannot factor {n} for a Monarch layer; use a composite width")
    return s1, n // s1


class MonarchLinear(nn.Module):
    """Square features-to-features map as two block-diagonal factors."""

    def __init__(self, features):
        super().__init__()
        s1, s2 = square_factors(features)
        self.features = features
        self.s1, self.s2 = s1, s2
        self.w1 = nn.Parameter(torch.randn(s1, s2, s2) * MONARCH_INIT_STD)
        self.w2 = nn.Parameter(torch.randn(s2, s1, s1) * MONARCH_INIT_STD)

    def forward(self, x):
        lead, s1, s2 = x.shape[:-1], self.s1, self.s2
        z = x.reshape(-1, s1, s2).permute(1, 0, 2)
        z = torch.bmm(z, self.w1)
        z = z.permute(2, 1, 0).contiguous()
        z = torch.bmm(z, self.w2)
        return z.permute(1, 2, 0).reshape(*lead, self.features)

    def param_count(self):
        return self.features * (self.s1 + self.s2)


class MonarchFFN(nn.Module):
    """FFN whose up and down projections are Monarch-factored. ffn must be a
    whole multiple of hidden; each multiple is one independent factored map, so
    the layer keeps the dense FFN's width and activation shape."""

    def __init__(self, hidden, ffn, activation=ACT_DEFAULT, capture_l1=False,
                 reg_mode=REG_DEFAULT):
        super().__init__()
        if activation not in _ACT_FNS:
            raise ValueError(f"unknown activation: {activation!r}")
        if reg_mode not in REG_MODES:
            raise ValueError(f"unknown reg_mode: {reg_mode!r}")
        if ffn % hidden:
            raise ValueError(f"monarch ffn {ffn} must be a multiple of hidden {hidden}")
        self.hidden     = hidden
        self.ffn        = ffn
        self.mult       = ffn // hidden
        self.up         = nn.ModuleList([MonarchLinear(hidden) for _ in range(self.mult)])
        self.down       = nn.ModuleList([MonarchLinear(hidden) for _ in range(self.mult)])
        self.activation = activation
        self._act_fn    = _ACT_FNS[activation]
        self.capture_l1 = bool(capture_l1)
        self.reg_mode   = reg_mode
        self._last_l1   = None

    def forward(self, x):
        post = torch.cat([self._act_fn(m(x)) for m in self.up], dim=-1)
        if self.capture_l1:
            self._last_l1 = group_penalty(post) if self.reg_mode == "group" else post.abs().mean()
        parts = post.split(self.hidden, dim=-1)
        out = self.down[0](parts[0])
        for m, p in zip(self.down[1:], parts[1:], strict=True):
            out = out + m(p)
        return out

    def probe_module(self):
        """First up factor stands in for the neuron activation vector; the full
        post-activation is the concatenation across factors."""
        return self.up[0]

    def probe_weights(self):
        """No single (in, out) matmul pair exists once both sides are factored."""
        return

    def param_count(self):
        return sum(m.param_count() for m in list(self.up) + list(self.down))
