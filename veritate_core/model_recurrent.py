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
# - Mixer module is named attn with a combined qkv QuantLinear + proj so the
#   dump/hook suite and QAT walk the same names as the canonical Block.
# - Research variant: trains + checkpoints via save(); no Brain/load branch,
#   not .bin-exportable (v12 candidate: decode is matvec + rank-1 state update).
# veritate_core/model_recurrent.py
# ------------------------------------------------------------------------------------
# Imports:

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (VOCAB_BYTE_LEVEL, ACT_DEFAULT, REG_DEFAULT, RMSNorm,
                    QuantLinear, FFN)
from . import qat as _qat

# ------------------------------------------------------------------------------------
# Constants

CHUNK          = 64
CONV_KERNEL    = 4
DECAY_MIN_BIAS = -6.9
DECAY_MAX_BIAS = -2.2

# ------------------------------------------------------------------------------------
# Functions


class RecurrentMixer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.h    = heads
        self.d    = hidden // heads
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
        self.qat = False

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.h, self.d
        pad = (CHUNK - T % CHUNK) % CHUNK
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            T = T + pad
        qkv = self.qkv(x).transpose(1, 2)
        qkv = self.conv(F.pad(qkv, (CONV_KERNEL - 1, 0))).transpose(1, 2)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2) * (D ** -0.5)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        la = -F.softplus(self.a_proj(x)).transpose(1, 2)

        n_chunks = T // CHUNK
        causal = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=x.device).tril()
        state = torch.zeros(B, H, D, D, dtype=x.dtype, device=x.device)
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
            outs.append(o)
        o = torch.cat(outs, dim=2)
        o = self.o_norm(o).transpose(1, 2).reshape(B, T, C)
        o = o * F.silu(self.gate(x))
        o = self.proj(o)
        return o[:, :T - pad] if pad else o


class RecurrentBlock(nn.Module):
    def __init__(self, hidden, ffn, heads, activation=ACT_DEFAULT, capture_l1=False,
                 reg_mode=REG_DEFAULT):
        super().__init__()
        self.n1   = RMSNorm(hidden)
        self.attn = RecurrentMixer(hidden, heads)
        self.n2   = RMSNorm(hidden)
        self.ff   = FFN(hidden, ffn, activation=activation, capture_l1=capture_l1,
                        reg_mode=reg_mode)
        self.qat  = False

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        if self.qat: x = _qat.fake_quant_act(x)
        x = x + self.ff(self.n2(x))
        if self.qat: x = _qat.fake_quant_act(x)
        return x


class VeritateRecurrent(nn.Module):
    def __init__(self, vocab, hidden, layers, ffn, heads, seq,
                 activation=ACT_DEFAULT, capture_l1=False, reg_mode=REG_DEFAULT):
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
        self.qat        = False

        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq, hidden)
        self.blocks  = nn.ModuleList([RecurrentBlock(hidden, ffn, heads,
                                                     activation=activation,
                                                     capture_l1=capture_l1,
                                                     reg_mode=reg_mode)
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
