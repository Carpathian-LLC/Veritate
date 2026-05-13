# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Canonical Veritate model. The single source of truth for byte-level decoder
#   shape across the platform. Plugins train it, the inference Brain loads it,
#   tools diff it. There is no second class.
# - vocab=256 is enforced at construction. Pre-norm RMSNorm + sdpa causal
#   attention + GELU FFN. Combined qkv linear, learned positional embedding,
#   tied LM head.
# - Plugins import via `from veritate_core.plugin import model` and use
#   `model.Veritate(...)`. Inference imports directly from veritate_core.model.
# - QAT support: every module that performs an INT8-relevant op carries a
#   `self.qat` flag (default False). Flip them all at once via
#   `veritate_core.qat.set_qat(model, True)`. State dict layout is unchanged so
#   the exporter and the C engine remain unaware of the flag.
# veritate_core/model.py
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import qat as _qat


VOCAB_BYTE_LEVEL = 256


class RMSNorm(nn.Module):
    def __init__(self, hidden, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps    = eps
        self.qat    = False

    def forward(self, x):
        # Compute the variance reduction in fp32 for numerical stability, but
        # leave the activation in its incoming dtype (bf16 under autocast).
        # Avoids casting the full [B, T, H] tensor up to fp32 and back, which
        # is a memory-bandwidth bottleneck on Apple Silicon's unified memory.
        n   = x.float().pow(2).mean(-1, keepdim=True)
        inv = torch.rsqrt(n + self.eps).to(x.dtype)
        w   = _qat.fake_quant_ln_weight(self.weight) if self.qat else self.weight
        return x * inv * w


class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.qat        = False
        self.quant_mode = _qat.QUANT_MODE_INT8

    def forward(self, x):
        if self.qat:
            return F.linear(_qat.fake_quant_act(x),
                            _qat.fake_quant_weight_mode(self.weight, self.quant_mode),
                            self.bias)
        return super().forward(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.h    = heads
        self.d    = hidden // heads
        self.qkv  = QuantLinear(hidden, 3 * hidden, bias=False)
        self.proj = QuantLinear(hidden, hidden,     bias=False)
        self.qat = False
        self.engine_faithful = False

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qat and self.engine_faithful:
            q = _qat.fake_quant_act(q)
            k = _qat.fake_quant_act(k)
            v = _qat.fake_quant_act(v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.qat and self.engine_faithful:
            out = _qat.fake_quant_act(out)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.up   = QuantLinear(hidden, ffn,    bias=False)
        self.down = QuantLinear(ffn,    hidden, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, hidden, ffn, heads):
        super().__init__()
        self.n1   = RMSNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads)
        self.n2   = RMSNorm(hidden)
        self.ff   = FFN(hidden, ffn)
        self.qat  = False

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        if self.qat: x = _qat.fake_quant_act(x)
        x = x + self.ff(self.n2(x))
        if self.qat: x = _qat.fake_quant_act(x)
        return x


class Veritate(nn.Module):
    def __init__(self, vocab, hidden, layers, ffn, heads, seq):
        super().__init__()
        if vocab != VOCAB_BYTE_LEVEL:
            raise ValueError(f"vocab must be {VOCAB_BYTE_LEVEL} (byte-level only), got {vocab}")
        if isinstance(ffn, (list, tuple)):
            ffn_per_layer = list(ffn)
            if len(ffn_per_layer) != layers:
                raise ValueError(f"ffn list length {len(ffn_per_layer)} does not match layers={layers}")
        else:
            ffn_per_layer = [int(ffn)] * layers
        self.vocab          = vocab
        self.hidden         = hidden
        self.layers         = layers
        self.ffn            = ffn_per_layer[0] if all(f == ffn_per_layer[0] for f in ffn_per_layer) else ffn_per_layer
        self.ffn_per_layer  = ffn_per_layer
        self.heads          = heads
        self.seq            = seq
        self.qat            = False

        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq,   hidden)
        self.blocks  = nn.ModuleList([Block(hidden, f, heads) for f in ffn_per_layer])
        self.n_out   = RMSNorm(hidden)
        self.lm_head = QuantLinear(hidden, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def set_qat(self, value):
        return _qat.set_qat(self, value)

    def hook_spec(self):
        # Canonical model is its own dumper view. Non-canonical models (MoE,
        # workspace, etc.) override this to return an adapter that quacks
        # like a canonical Veritate so the dumper walks one shape.
        return self

    def embed(self, tokens):
        B, T = tokens.shape
        if T > self.seq:
            raise ValueError(f"input length {T} exceeds seq {self.seq}")
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        e = self.tok_emb(tokens) + self.pos_emb(pos)
        if self.qat: e = _qat.fake_quant_act(e)
        return e

    def forward(self, tokens, targets=None):
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        x = self.n_out(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss
