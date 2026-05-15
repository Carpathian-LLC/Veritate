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

ACTIVATIONS    = ("gelu", "relu", "silu")
ACT_DEFAULT    = "gelu"

# Map name -> torch.nn.functional callable. ReLU and SiLU expose post-activation
# sparsity for the L1 penalty; GELU does not (sparsity is undefined for a smooth
# activation, callers must use ReLU when they want l1_lambda > 0).
_ACT_FNS = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
}


class RMSNorm(nn.Module):
    def __init__(self, hidden, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps    = eps
        self.qat    = False
        self._normalized_shape = (hidden,)

    def forward(self, x):
        # F.rms_norm fuses the fp32 reduction + scale into one kernel (~6x
        # speedup vs the unfused manual chain on CUDA; equivalent on CPU/MPS).
        # Weight is cast to x.dtype because the fused dispatcher falls back to
        # scalar when input/weight dtypes mismatch.
        w = _qat.fake_quant_ln_weight(self.weight) if self.qat else self.weight
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        return F.rms_norm(x, self._normalized_shape, w, self.eps)


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
    def __init__(self, hidden, ffn, activation=ACT_DEFAULT, capture_l1=False):
        super().__init__()
        if activation not in _ACT_FNS:
            raise ValueError(f"unknown activation: {activation!r}; expected one of {ACTIVATIONS}")
        self.up         = QuantLinear(hidden, ffn,    bias=False)
        self.down       = QuantLinear(ffn,    hidden, bias=False)
        self.activation = activation
        self._act_fn    = _ACT_FNS[activation]
        self.capture_l1 = bool(capture_l1)
        self._last_l1   = None

    def forward(self, x):
        post = self._act_fn(self.up(x))
        if self.capture_l1:
            self._last_l1 = post.abs().mean()
        return self.down(post)


class Block(nn.Module):
    def __init__(self, hidden, ffn, heads, activation=ACT_DEFAULT, capture_l1=False):
        super().__init__()
        self.n1   = RMSNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads)
        self.n2   = RMSNorm(hidden)
        self.ff   = FFN(hidden, ffn, activation=activation, capture_l1=capture_l1)
        self.qat  = False

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        if self.qat: x = _qat.fake_quant_act(x)
        x = x + self.ff(self.n2(x))
        if self.qat: x = _qat.fake_quant_act(x)
        return x


class Veritate(nn.Module):
    def __init__(self, vocab, hidden, layers, ffn, heads, seq,
                 activation=ACT_DEFAULT, capture_l1=False):
        super().__init__()
        if vocab != VOCAB_BYTE_LEVEL:
            raise ValueError(f"vocab must be {VOCAB_BYTE_LEVEL} (byte-level only), got {vocab}")
        if activation not in _ACT_FNS:
            raise ValueError(f"unknown activation: {activation!r}; expected one of {ACTIVATIONS}")
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
        self.activation     = activation
        self.capture_l1     = bool(capture_l1)
        self.qat            = False

        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq,   hidden)
        self.blocks  = nn.ModuleList([Block(hidden, f, heads,
                                            activation=activation,
                                            capture_l1=capture_l1)
                                      for f in ffn_per_layer])
        self.n_out   = RMSNorm(hidden)
        self.lm_head = QuantLinear(hidden, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def post_l1_sum(self):
        """Sum of per-block post-activation L1 means captured during the last
        forward. Returns None when capture_l1 is off or no forward has run."""
        if not self.capture_l1:
            return None
        parts = [blk.ff._last_l1 for blk in self.blocks if blk.ff._last_l1 is not None]
        if not parts:
            return None
        return sum(parts)

    def set_qat(self, value):
        return _qat.set_qat(self, value)

    def hook_spec(self):
        # Canonical model is its own dumper view. Non-canonical models (MoE,
        # workspace, etc.) override this to return an adapter that quacks
        # like a canonical Veritate so the dumper walks one shape.
        return self

    def embed(self, tokens, start_pos=0):
        B, T = tokens.shape
        if start_pos + T > self.seq:
            raise ValueError(f"start_pos+T ({start_pos + T}) exceeds seq {self.seq}")
        pos = torch.arange(start_pos, start_pos + T, device=tokens.device).unsqueeze(0).expand(B, T)
        e = self.tok_emb(tokens) + self.pos_emb(pos)
        if self.qat: e = _qat.fake_quant_act(e)
        return e

    def ensure_context(self, T):
        if T > self.seq:
            raise ValueError(f"input length {T} exceeds seq {self.seq}")

    def run_blocks(self, x, start_pos=0, exit_after=None):
        n = self.layers if exit_after is None else min(int(exit_after), self.layers)
        for L in range(n):
            x = self.run_block(x, L, start_pos=start_pos)
        return x

    def run_block(self, x, L, start_pos=0):
        return self.blocks[L](x)

    def project_byte0(self, residual):
        return self.lm_head(self.n_out(residual))

    def supports_mtp_decode(self):
        return False

    def kv_cache_patch_attn(self, attn_mod, cache, get_start_pos):
        def fwd(x):
            B, T, C = x.shape
            h, d = attn_mod.h, attn_mod.d
            qkv = attn_mod.qkv(x).view(B, T, 3, h, d).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            cache.append(k, v)
            K, V = cache.view()
            if T == 1:
                out = F.scaled_dot_product_attention(q, K, V, is_causal=False)
            else:
                S = K.size(2)
                abs_start = cache.length - T
                i_idx = torch.arange(T, device=x.device).view(T, 1) + abs_start
                j_idx = torch.arange(S, device=x.device).view(1, S)
                mask = (j_idx <= i_idx)
                attn_mask = torch.zeros(T, S, device=x.device, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(~mask, float("-inf"))
                out = F.scaled_dot_product_attention(q, K, V, attn_mask=attn_mask, is_causal=False)
            out = out.transpose(1, 2).contiguous().view(B, T, C)
            return attn_mod.proj(out)
        return fwd

    def forward(self, tokens, targets=None):
        x = self.embed(tokens)
        x = self.run_blocks(x)
        logits = self.project_byte0(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss
