# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - DeepSeekMoE fine-grained feed-forward network. Drop-in for the canonical FFN
#   inside the hybrid trunk's global blocks: 8 routed experts each at 1/4 the dense
#   ffn width + 1 always-on shared expert at 1/4 width, top-2 routed.
# - Routing is capacity-based DENSE dispatch (GShard/Switch form), grouped
#   per-sequence: build a one-hot dispatch mask, scatter tokens into a fixed
#   [batch, experts, capacity, hidden] tensor via einsum, run a batched expert
#   GEMM, einsum-combine back. Every tensor shape is static across steps given a
#   fixed (batch, seq): MPS kernel-cache rule 24c. No sort, no bool advanced
#   indexing; one-hot via integer comparison, position via cumsum.
# - Load balance: aux-loss-free per-expert routing bias b_i (a buffer, not a
#   Parameter) added to the top-k SELECTION affinity only, nudged +-gamma each
#   training forward by over/under-load (DeepSeek-V3 2412.19437). Small
#   sequence-wise balance aux-loss (alpha, DeepSeekMoE 2401.06066) as a backstop,
#   exposed via _last_aux for the trainer to add.
# - Exposes .up/.down (the shared expert projections) so the dump/hook suite that
#   walks blk.ff.up / blk.ff.down.weight does not crash; _last_l1 mirrors the FFN
#   contract. Experts use QuantLinear so the qat flag rides along.
# veritate_core/model_moe.py
# ------------------------------------------------------------------------------------
# Imports:

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import _ACT_FNS, ACT_DEFAULT, FFN, REG_DEFAULT, QuantLinear

# ------------------------------------------------------------------------------------
# Constants

MOE_N_ROUTED        = 8
MOE_TOP_K           = 2
MOE_EXPERT_DIV      = 4
MOE_CAPACITY_FACTOR = 1.25
MOE_AUX_ALPHA       = 0.01
MOE_BIAS_GAMMA      = 1e-3
MOE_SELECT_INF      = 1e9
MOE_EPS             = 1e-9

# ------------------------------------------------------------------------------------
# Functions


class MoEFFN(nn.Module):
    def __init__(self, hidden, ffn, activation=ACT_DEFAULT, capture_l1=False,
                 reg_mode=REG_DEFAULT, num_experts=MOE_N_ROUTED, top_k=MOE_TOP_K,
                 capacity_factor=MOE_CAPACITY_FACTOR):
        super().__init__()
        expert_ffn  = ffn // MOE_EXPERT_DIV
        self.up     = QuantLinear(hidden, expert_ffn, bias=False)
        self.down   = QuantLinear(expert_ffn, hidden, bias=False)
        self.experts = nn.ModuleList([
            FFN(hidden, expert_ffn, activation=activation, capture_l1=False,
                reg_mode=reg_mode)
            for _ in range(num_experts)
        ])
        self.gate   = nn.Linear(hidden, num_experts, bias=False)
        self.register_buffer("route_bias", torch.zeros(num_experts))
        self._act_fn         = _ACT_FNS[activation]
        self.num_experts     = num_experts
        self.top_k           = top_k
        self.capacity_factor = capacity_factor
        self.capture_l1      = bool(capture_l1)
        self.reg_mode        = reg_mode
        self._last_l1        = None
        self._last_aux       = None
        self._last_share     = None
        self.qat             = False

    def _capacity(self, seq_len):
        return max(1, math.ceil(self.capacity_factor * seq_len * self.top_k / self.num_experts))

    def _route(self, x, capacity):
        _B, T, _ = x.shape
        E, K = self.num_experts, self.top_k
        logits = F.linear(x, self.gate.weight.float())
        probs  = F.softmax(logits, dim=-1)
        score  = probs + self.route_bias.float()
        idx_range = torch.arange(E, device=x.device)
        mask = torch.zeros_like(score)
        work = score
        for _ in range(K):
            sel = work.argmax(dim=-1, keepdim=True)
            onehot = (sel == idx_range).to(score.dtype)
            mask = mask + onehot
            work = work - onehot * MOE_SELECT_INF
        gates = probs * mask
        gates = gates / gates.sum(-1, keepdim=True).clamp_min(MOE_EPS)

        pos  = mask.cumsum(dim=1) - 1.0
        keep = mask * (pos < capacity).to(mask.dtype)
        pos_idx = pos.clamp(0, capacity - 1).long()
        pos_oh  = (pos_idx.unsqueeze(-1) == torch.arange(capacity, device=x.device)).to(mask.dtype)
        dispatch = keep.unsqueeze(-1) * pos_oh
        combine  = (gates * keep).unsqueeze(-1) * pos_oh

        f = mask.sum(dim=1) / (K * T)
        p = probs.mean(dim=1)
        self._last_aux   = MOE_AUX_ALPHA * E * (f.detach() * p).sum(-1).mean()
        self._last_share = f.detach().mean(0)
        if self.training:
            with torch.no_grad():
                load = mask.sum(dim=(0, 1))
                self.route_bias += MOE_BIAS_GAMMA * torch.sign(load.mean() - load)
        return dispatch, combine

    def forward(self, x):
        _B, T, _H = x.shape
        shared = self.down(self._act_fn(self.up(x)))
        capacity = self._capacity(T)
        with torch.autocast(device_type=x.device.type, enabled=False):
            dispatch, combine = self._route(x.float(), capacity)
        dispatched = torch.einsum("btec,bth->bech", dispatch.to(x.dtype), x)
        expert_out = torch.stack(
            [self.experts[e](dispatched[:, e]) for e in range(self.num_experts)], dim=1)
        routed = torch.einsum("btec,bech->bth", combine.to(expert_out.dtype), expert_out)
        out = shared + routed
        if self.capture_l1:
            self._last_l1 = out.abs().mean()
        return out
