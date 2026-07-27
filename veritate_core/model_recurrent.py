# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Constant-state byte trunk (gated linear attention class: RWKV/GLA/Mamba-2
#   lineage). Attention is replaced by a chunkwise linear recurrence with a
#   learned scalar-per-head decay: state is one [d_k, d_v] matrix per head,
#   fixed size at any context length. Decode cost per byte is O(1) in position
#   (no KV cache), which is the entire point at conversation lengths.
# - Chunkwise-parallel training: sequential scan over CHUNK-byte chunks, exact
#   (not approximate), log-space decays so nothing explodes, fixed shapes on MPS.
# - state_rule opts the state update in: "gla" (default, uniform per-head decay,
#   bit-identical to the pre-mechanism model), "delta" (Gated DeltaNet error-
#   correcting overwrite, chunkwise WY inverse), "pinned" (gla + a fixed set of
#   decay-exempt memory slots). Default path is untouched so baselines are unaffected.
# - WY inverse is block-recursive (8x8 nilpotent base, halving combine): same exact
#   inverse, no large intermediates. Whole-chunk nilpotent squaring overflows fp32
#   when beta saturates and keys align (see developer_documentation/training/
#   m1delta_divergence_analysis.md).
# - Mixer module is named attn with a combined qkv QuantLinear + proj so the
#   dump/hook suite and QAT walk the same names as the canonical Block.
# - State carry: forward(x, state=None, return_state=False) seeds the
#   recurrence from a prior window's state dict ({"s", "conv"[, "pin"]}) and returns
#   the final one; exact across windows (chunk recurrence + conv tail both carried)
#   when T is a CHUNK multiple. Default call is bit-identical to the stateless path.
#   VeritateRecurrent.forward_streaming(tokens, states) threads per-block states;
#   positions stay window-local (pos_emb indexed 0..T-1 per window). Consumed by
#   inference streaming and by the trainer's state_carry=chunks chunk carry.
# - Research variant: trains + checkpoints via save(); no Brain/load branch,
#   not .bin-exportable (v12 candidate: decode is matvec + rank-1 state update).
# veritate_core/model_recurrent.py
# ------------------------------------------------------------------------------------
# Imports:

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import qat as _qat
from .model import ACT_DEFAULT, FFN, REG_DEFAULT, VOCAB_BYTE_LEVEL, QuantLinear, RMSNorm

# ------------------------------------------------------------------------------------
# Constants

CHUNK          = 64
CONV_KERNEL    = 4
DECAY_MIN_BIAS = -6.9
DECAY_MAX_BIAS = -2.2
STATE_RULES        = ("gla", "delta", "pinned")
STATE_RULE_DEFAULT = "gla"
DELTA_INV_BLOCK    = 8
DELTA_INV_ITERS    = math.ceil(math.log2(DELTA_INV_BLOCK))
PIN_SLOTS          = 4
PIN_KEY_STD        = 0.02
PIN_EPS            = 1e-6

# ------------------------------------------------------------------------------------
# Functions


class RecurrentMixer(nn.Module):
    def __init__(self, hidden, heads, state_rule=STATE_RULE_DEFAULT):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        if state_rule not in STATE_RULES:
            raise ValueError(f"state_rule must be one of {STATE_RULES}, got {state_rule}")
        self.h          = heads
        self.d          = hidden // heads
        self.state_rule = state_rule
        self.qkv  = QuantLinear(hidden, 3 * hidden, bias=False)
        self.proj = QuantLinear(hidden, hidden,     bias=False)
        self.gate = QuantLinear(hidden, hidden,     bias=False)
        self.conv = nn.Conv1d(3 * hidden, 3 * hidden, CONV_KERNEL,
                              groups=3 * hidden, bias=False)
        self.a_proj = nn.Linear(hidden, heads, bias=True)
        nn.init.zeros_(self.a_proj.weight)
        with torch.no_grad():
            self.a_proj.bias.copy_(torch.linspace(DECAY_MIN_BIAS, DECAY_MAX_BIAS, heads))
        self.o_norm = RMSNorm(self.d)
        if state_rule == "delta":
            self.b_proj = nn.Linear(hidden, heads, bias=True)
            nn.init.zeros_(self.b_proj.weight)
            nn.init.zeros_(self.b_proj.bias)
        elif state_rule == "pinned":
            self.pin_key  = nn.Parameter(torch.randn(heads, PIN_SLOTS, self.d) * PIN_KEY_STD)
            self.sal_proj = nn.Linear(hidden, heads, bias=True)
            nn.init.zeros_(self.sal_proj.weight)
            nn.init.zeros_(self.sal_proj.bias)
        self.qat = False

    def _project(self, x, conv_tail=None):
        B, T, C = x.shape
        H, D = self.h, self.d
        pad = (CHUNK - T % CHUNK) % CHUNK
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            T = T + pad
        qkv = self.qkv(x).transpose(1, 2)
        tail = qkv[:, :, 1 - CONV_KERNEL:]
        pre = F.pad(qkv, (CONV_KERNEL - 1, 0)) if conv_tail is None \
            else torch.cat([conv_tail, qkv], dim=2)
        qkv = self.conv(pre).transpose(1, 2)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2) * (D ** -0.5)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        la = -F.softplus(self.a_proj(x)).transpose(1, 2)
        return q, k, v, la, x, B, T, C, H, D, pad, tail

    def _tri_inverse(self, neg_m, eye):
        n = neg_m.size(-1)
        if n <= DELTA_INV_BLOCK:
            term = neg_m
            inv = eye + term
            for _ in range(DELTA_INV_ITERS - 1):
                term = term @ term
                inv = inv @ (eye + term)
            return inv
        h = n // 2
        t1 = self._tri_inverse(neg_m[..., :h, :h], eye[:h, :h])
        t2 = self._tri_inverse(neg_m[..., h:, h:], eye[h:, h:])
        off = t2 @ neg_m[..., h:, :h] @ t1
        top = torch.cat([t1, torch.zeros_like(t1)], dim=-1)
        return torch.cat([top, torch.cat([off, t2], dim=-1)], dim=-2)

    def _pinned_step(self, qc, kc, vc, sc, o, pin_val, scale):
        pk = self.pin_key.transpose(-1, -2)
        o = o + torch.softmax((qc @ pk) * scale, dim=-1) @ pin_val
        wp = torch.softmax((kc @ pk) * scale, dim=-1)
        sw = sc.unsqueeze(-1) * wp
        num = sw.transpose(-1, -2) @ vc
        den = sw.sum(-2).unsqueeze(-1)
        pin_val = pin_val + (den / (den + 1.0)) * (num / (den + PIN_EPS) - pin_val)
        return o, pin_val

    def _forward_delta(self, x, state=None, return_state=False):
        s0 = state["s"] if state is not None else None
        q, k, v, la, x, B, T, C, H, D, pad, tail = self._project(
            x, state["conv"] if state is not None else None)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)
        n_chunks = T // CHUNK
        with torch.autocast(device_type=x.device.type, enabled=False):
            qf = F.normalize(q.float(), dim=-1)
            kf = F.normalize(k.float(), dim=-1)
            vf, laf, betaf = v.float(), la.float(), beta.float()
            eye = torch.eye(CHUNK, device=x.device)
            state = torch.zeros(B, H, D, D, device=x.device) if s0 is None else s0
            outs = []
            for c in range(n_chunks):
                s = c * CHUNK
                qc, kc, vc = qf[:, :, s:s + CHUNK], kf[:, :, s:s + CHUNK], vf[:, :, s:s + CHUNK]
                bcol = betaf[:, :, s:s + CHUNK].unsqueeze(-2)
                cl = laf[:, :, s:s + CHUNK].cumsum(-1)
                A = cl.exp().unsqueeze(-1)
                decay = (cl.unsqueeze(-1) - cl.unsqueeze(-2)).exp()
                M = (decay * bcol * (kc @ kc.transpose(-1, -2))).tril(-1)
                U = self._tri_inverse(-M, eye) @ (vc - A * (kc @ state))
                P = (decay * bcol * (qc @ kc.transpose(-1, -2))).tril(0)
                o = A * (qc @ state) + P @ U
                a_last = cl[:, :, -1:].exp()
                kw = ((a_last / cl.exp()) * betaf[:, :, s:s + CHUNK]).unsqueeze(-1) * kc
                state = a_last.unsqueeze(-1) * state + kw.transpose(-1, -2) @ U
                outs.append(o)
            o = torch.cat(outs, dim=2)
        o = o.to(x.dtype)
        o = self.o_norm(o).transpose(1, 2).reshape(B, T, C)
        o = o * F.silu(self.gate(x))
        o = self.proj(o)
        o = o[:, :T - pad] if pad else o
        if not return_state:
            return o
        return o, {"s": state, "conv": tail}

    def forward(self, x, state=None, return_state=False):
        if (state is not None or return_state) and x.size(1) % CHUNK:
            raise ValueError(f"state carry requires T multiple of {CHUNK}, got {x.size(1)}")
        if self.state_rule == "delta":
            return self._forward_delta(x, state, return_state)
        s0   = state["s"] if state is not None else None
        pin0 = state["pin"] if state is not None and self.state_rule == "pinned" else None
        q, k, v, la, x, B, T, C, H, D, pad, tail = self._project(
            x, state["conv"] if state is not None else None)
        n_chunks = T // CHUNK
        causal = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=x.device).tril()
        state = torch.zeros(B, H, D, D, dtype=x.dtype, device=x.device) if s0 is None else s0
        pinned = self.state_rule == "pinned"
        if pinned:
            pin_val = (torch.zeros(B, H, PIN_SLOTS, D, dtype=x.dtype, device=x.device)
                       if pin0 is None else pin0)
            sal = torch.sigmoid(self.sal_proj(x)).transpose(1, 2)
        outs = []
        for c in range(n_chunks):
            s = c * CHUNK
            qc, kc, vc = q[:, :, s:s + CHUNK], k[:, :, s:s + CHUNK], v[:, :, s:s + CHUNK]
            cl = la[:, :, s:s + CHUNK].cumsum(-1)
            dmat = (cl.unsqueeze(-1) - cl.unsqueeze(-2)).masked_fill(~causal, float("-inf")).exp()
            att = (qc @ kc.transpose(-1, -2)) * dmat
            o = att @ vc
            o = o + (qc * cl.exp().unsqueeze(-1)) @ state
            w = (cl[:, :, -1:] - cl).exp().unsqueeze(-1)
            state = cl[:, :, -1].exp().unsqueeze(-1).unsqueeze(-1) * state \
                + (kc * w).transpose(-1, -2) @ vc
            if pinned:
                o, pin_val = self._pinned_step(qc, kc, vc, sal[:, :, s:s + CHUNK], o,
                                               pin_val, D ** -0.5)
            outs.append(o)
        o = torch.cat(outs, dim=2)
        o = self.o_norm(o).transpose(1, 2).reshape(B, T, C)
        o = o * F.silu(self.gate(x))
        o = self.proj(o)
        o = o[:, :T - pad] if pad else o
        if not return_state:
            return o
        out_state = {"s": state, "conv": tail}
        if pinned:
            out_state["pin"] = pin_val
        return o, out_state


class RecurrentBlock(nn.Module):
    def __init__(self, hidden, ffn, heads, activation=ACT_DEFAULT, capture_l1=False,
                 reg_mode=REG_DEFAULT, state_rule=STATE_RULE_DEFAULT):
        super().__init__()
        self.n1   = RMSNorm(hidden)
        self.attn = RecurrentMixer(hidden, heads, state_rule=state_rule)
        self.n2   = RMSNorm(hidden)
        self.ff   = FFN(hidden, ffn, activation=activation, capture_l1=capture_l1,
                        reg_mode=reg_mode)
        self.qat  = False

    def forward(self, x, state=None, return_state=False):
        a = self.attn(self.n1(x), state=state, return_state=return_state)
        if return_state:
            a, state = a
        x = x + a
        if self.qat: x = _qat.fake_quant_act(x)
        x = x + self.ff(self.n2(x))
        if self.qat: x = _qat.fake_quant_act(x)
        return (x, state) if return_state else x


class VeritateRecurrent(nn.Module):
    def __init__(self, vocab, hidden, layers, ffn, heads, seq,
                 activation=ACT_DEFAULT, capture_l1=False, reg_mode=REG_DEFAULT,
                 state_rule=STATE_RULE_DEFAULT):
        super().__init__()
        if vocab != VOCAB_BYTE_LEVEL:
            raise ValueError(f"vocab must be {VOCAB_BYTE_LEVEL} (byte-level only), got {vocab}")
        self.vocab      = vocab
        self.hidden     = hidden
        self.layers     = layers
        self.ffn        = ffn
        self.ffn_per_layer = [int(ffn)] * layers
        self.heads      = heads
        self.seq        = seq
        self.activation = activation
        self.capture_l1 = bool(capture_l1)
        self.reg_mode   = reg_mode
        self.state_rule = state_rule
        self.qat        = False

        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq, hidden)
        self.blocks  = nn.ModuleList([RecurrentBlock(hidden, ffn, heads,
                                                     activation=activation,
                                                     capture_l1=capture_l1,
                                                     reg_mode=reg_mode,
                                                     state_rule=state_rule)
                                      for _ in range(layers)])
        self.n_out   = RMSNorm(hidden)
        self.lm_head = QuantLinear(hidden, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        for name, p in self.named_parameters():
            if p.dim() >= 2 and "a_proj" not in name:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def post_l1_sum(self):
        if not self.capture_l1:
            return None
        parts = [blk.ff._last_l1 for blk in self.blocks if blk.ff._last_l1 is not None]
        if not parts:
            return None
        return sum(parts)

    def set_qat(self, value):
        return _qat.set_qat(self, value)

    def hook_spec(self):
        return self

    def embed(self, tokens, start_pos=0):
        B, T = tokens.shape
        if start_pos + T > self.seq:
            raise ValueError(f"start_pos+T ({start_pos + T}) exceeds seq {self.seq}")
        pos = torch.arange(start_pos, start_pos + T, device=tokens.device).unsqueeze(0).expand(B, T)
        e = self.tok_emb(tokens) + self.pos_emb(pos)
        if self.qat: e = _qat.fake_quant_act(e)
        return e

    def project_byte0(self, residual):
        return self.lm_head(self.n_out(residual))

    def supports_mtp_decode(self):
        return False

    def supports_streaming(self):
        return True

    def forward_streaming(self, tokens, states=None):
        x = self.embed(tokens)
        out_states = []
        for blk in self.blocks:
            st = states[len(out_states)] if states is not None else None
            x, st = blk(x, state=st, return_state=True)
            out_states.append(st)
        return self.project_byte0(x), out_states

    def forward(self, tokens, targets=None):
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        logits = self.project_byte0(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss
