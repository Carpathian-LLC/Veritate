# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Veritate variant with RoPE rotary positional embeddings and no learned
#   `pos_emb`. State-dict-compatible with canonical Veritate everywhere EXCEPT
#   it lacks `pos_emb.weight`. Used both as a warm-start target for canonical
#   ckpts and as the inference class for any RoPE-only checkpoint without an
#   MTP head.
# - For the variant with an MTP head (the 800M training plugin), see
#   `plugins/veritate_800m/plugin.py::Veritate800M`. This file is the no-MTP
#   sibling; both share the same block API (`attn.qkv`, `attn.proj`, `ff.up`,
#   `ff.down`, `n1`, `n2`) so the MRI's per-block forward hooks attach the
#   same way.
# - extend_rope(new_max_seq) rebuilds the rope cache to accommodate decode
#   beyond the training seq.
# veritate_core/model_rope.py
# ------------------------------------------------------------------------------------
# Imports:

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------
# Constants

VOCAB_BYTE_LEVEL = 256

# ------------------------------------------------------------------------------------
# Functions


class RMSNorm(nn.Module):
    def __init__(self, hidden, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps    = eps

    def forward(self, x):
        n   = x.float().pow(2).mean(-1, keepdim=True)
        inv = torch.rsqrt(n + self.eps).to(x.dtype)
        return x * inv * self.weight


def build_rope_cache(d_head, max_seq, base=10000.0, device=None, dtype=torch.float32):
    if d_head % 2 != 0:
        raise ValueError(f"RoPE requires even head dim, got {d_head}")
    half     = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t        = torch.arange(max_seq, device=device, dtype=torch.float32)
    freqs    = torch.outer(t, inv_freq)
    cos      = freqs.cos().to(dtype)
    sin      = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_t = cos[:T].view(1, 1, T, -1)
    sin_t = sin[:T].view(1, 1, T, -1)
    rx1 = x1 * cos_t - x2 * sin_t
    rx2 = x1 * sin_t + x2 * cos_t
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


class CausalSelfAttentionRoPE(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.h    = heads
        self.d    = hidden // heads
        self.qkv  = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden,     bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos = rope_cos.to(q.dtype)
        sin = rope_sin.to(q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FFN(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.up   = nn.Linear(hidden, ffn,    bias=False)
        self.down = nn.Linear(ffn,    hidden, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, hidden, ffn, heads):
        super().__init__()
        self.n1   = RMSNorm(hidden)
        self.attn = CausalSelfAttentionRoPE(hidden, heads)
        self.n2   = RMSNorm(hidden)
        self.ff   = FFN(hidden, ffn)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.n1(x), rope_cos, rope_sin)
        x = x + self.ff(self.n2(x))
        return x


class VeritateRoPE(nn.Module):
    """Veritate with RoPE positions, no MTP head.

    State-dict layout matches canonical Veritate everywhere except `pos_emb.weight`
    (absent here). Same block API as `Veritate800M` (no MTP), so the MRI's
    per-block forward hooks attach unchanged.
    """

    def __init__(self, vocab=VOCAB_BYTE_LEVEL, hidden=768, layers=12, ffn=3072, heads=12,
                 seq=512, rope_base=10000.0):
        super().__init__()
        if vocab != VOCAB_BYTE_LEVEL:
            raise ValueError(f"vocab must be {VOCAB_BYTE_LEVEL}, got {vocab}")
        self.vocab     = vocab
        self.hidden    = hidden
        self.layers    = layers
        self.ffn       = ffn
        self.heads     = heads
        self.seq       = seq
        self.rope_base = rope_base
        self.d_head    = hidden // heads

        self.tok_emb = nn.Embedding(vocab, hidden)
        self.blocks  = nn.ModuleList([Block(hidden, ffn, heads) for _ in range(layers)])
        self.n_out   = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        cos, sin = build_rope_cache(self.d_head, seq, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def extend_rope(self, new_max_seq):
        device   = self.rope_cos.device
        dtype    = self.rope_cos.dtype
        cos, sin = build_rope_cache(self.d_head, new_max_seq, base=self.rope_base,
                                    device=device, dtype=dtype)
        self.rope_cos = cos
        self.rope_sin = sin

    def embed(self, tokens):
        return self.tok_emb(tokens)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        if T > self.rope_cos.shape[0]:
            self.extend_rope(T)
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin)
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
