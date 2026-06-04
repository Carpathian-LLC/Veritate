# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Multi-Mind (MtM) Veritate variant. 6-expert top-2 MoE FFN per block.
#   Optional `gate_g` adds per-expert sentiment-bias to router logits when
#   `bias_mode=True`; sentiment=None matches a dense forward.
# - Implements the canonical Veritate contract. hook_spec() returns a
#   canonical-shaped FFN adapter (routing-weighted expert combination as a
#   single dense FFN) so the dumper walks one shape.
# veritate_core/model_mtm.py
# ------------------------------------------------------------------------------------
# Imports:

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from . import qat as _qat
except Exception:
    _qat = None

# ------------------------------------------------------------------------------------
# Constants

VOCAB_BYTE_LEVEL = 256
REGION_NAMES     = ("broca", "wernicke", "hippocampus", "prefrontal", "cerebellum", "thalamus")
N_EXPERTS        = len(REGION_NAMES)
TOP_K            = 2
INIT_STD         = 0.02
EPS_ROUTE        = 1e-9

# ------------------------------------------------------------------------------------
# Functions


class _RMSNorm(nn.Module):
    def __init__(self, hidden, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps    = eps

    def forward(self, x):
        n   = x.float().pow(2).mean(-1, keepdim=True)
        inv = torch.rsqrt(n + self.eps).to(x.dtype)
        return x * inv * self.weight


class _MoEFFN(nn.Module):
    def __init__(self, hidden, ffn, n_experts=N_EXPERTS, top_k=TOP_K):
        super().__init__()
        self.hidden, self.ffn, self.n_experts, self.top_k = hidden, ffn, n_experts, top_k
        self.router = nn.Linear(hidden, n_experts, bias=False)
        self.up     = nn.Parameter(torch.randn(n_experts, hidden, ffn) * INIT_STD)
        self.down   = nn.Parameter(torch.randn(n_experts, ffn, hidden) * INIT_STD)
        self._last_gates = None

    def forward(self, x, gate_bias=None):
        B, T, H = x.shape
        flat   = x.reshape(B * T, H)
        logits = self.router(flat)
        if gate_bias is not None:
            logits = logits + gate_bias.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        weights      = F.softmax(logits, dim=-1)
        top_w, top_i = weights.topk(self.top_k, dim=-1)
        top_w        = top_w / (top_w.sum(dim=-1, keepdim=True) + EPS_ROUTE)
        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            idx = top_i[:, k]
            w   = top_w[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                mask = (idx == e)
                if not mask.any():
                    continue
                out[mask] = out[mask] + w[mask] * (F.gelu(flat[mask] @ self.up[e]) @ self.down[e])
        self._last_gates = weights.reshape(B, T, self.n_experts).detach()
        return out.reshape(B, T, H)


class _Attn(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.h    = heads
        self.d    = hidden // heads
        self.qkv  = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden,     bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class _Block(nn.Module):
    def __init__(self, hidden, ffn, heads):
        super().__init__()
        self.n1   = _RMSNorm(hidden)
        self.attn = _Attn(hidden, heads)
        self.n2   = _RMSNorm(hidden)
        self.ff   = _MoEFFN(hidden, ffn)

    def forward(self, x, gate_bias=None):
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x), gate_bias=gate_bias)
        return x


def _ff_view(moe):
    up   = nn.Linear(moe.hidden, moe.ffn,    bias=False)
    down = nn.Linear(moe.ffn,    moe.hidden, bias=False)
    with torch.no_grad():
        w = torch.full((moe.n_experts,), 1.0 / moe.n_experts, device=moe.up.device)
        up.weight.copy_(torch.einsum("e,ehf->fh", w, moe.up))
        down.weight.copy_(torch.einsum("e,efh->hf", w, moe.down))
    view = nn.Module()
    view.up, view.down = up, down
    return view


def _block_view(blk):
    view = nn.Module()
    view.n1, view.attn, view.n2 = blk.n1, blk.attn, blk.n2
    view.ff = _ff_view(blk.ff)
    return view


class _HookAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.tok_emb = model.tok_emb
        self.pos_emb = model.pos_emb
        self.n_out   = model.n_out
        self.lm_head = model.lm_head
        self.blocks  = nn.ModuleList([_block_view(b) for b in model.blocks])


class VeritateMultimind(nn.Module):
    REGION_NAMES = REGION_NAMES

    def __init__(self, vocab=VOCAB_BYTE_LEVEL, hidden=256, layers=4, ffn=512, heads=8,
                 seq=512, bias_mode=False):
        super().__init__()
        if vocab != VOCAB_BYTE_LEVEL:
            raise ValueError(f"vocab must be {VOCAB_BYTE_LEVEL}, got {vocab}")
        self.vocab        = vocab
        self.hidden       = hidden
        self.layers       = layers
        self.ffn          = ffn
        self.heads        = heads
        self.seq          = seq
        self.bias_mode    = bool(bias_mode)
        self.qat          = False
        self.n_experts    = N_EXPERTS
        self.region_names = REGION_NAMES
        self._gate_bias_provider = None

        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq,   hidden)
        self.blocks  = nn.ModuleList([_Block(hidden, ffn, heads) for _ in range(layers)])
        self.n_out   = _RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.gate_g  = nn.Parameter(torch.zeros(N_EXPERTS)) if self.bias_mode else None

        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=INIT_STD)

    def set_gate_bias_provider(self, fn):
        if fn is not None and not callable(fn):
            raise TypeError("gate_bias_provider must be callable or None")
        self._gate_bias_provider = fn

    def embed(self, tokens, start_pos=0):
        B, T = tokens.shape
        if start_pos + T > self.seq:
            raise ValueError(f"start_pos+T ({start_pos + T}) exceeds seq {self.seq}")
        pos = torch.arange(start_pos, start_pos + T, device=tokens.device).unsqueeze(0).expand(B, T)
        return self.tok_emb(tokens) + self.pos_emb(pos)

    def run_block(self, x, L, start_pos=0, gate_bias=None):
        return self.blocks[L](x, gate_bias=gate_bias)

    def run_blocks(self, x, start_pos=0, exit_after=None, gate_bias=None):
        n = self.layers if exit_after is None else min(int(exit_after), self.layers)
        for L in range(n):
            x = self.blocks[L](x, gate_bias=gate_bias)
        return x

    def project_byte0(self, residual):
        return self.lm_head(self.n_out(residual))

    def set_qat(self, value):
        if _qat is not None:
            return _qat.set_qat(self, value)
        self.qat = bool(value)
        return self

    def post_l1_sum(self):
        return None

    def hook_spec(self):
        return _HookAdapter(self)

    def supports_mtp_decode(self):
        return False

    def gate_g_norm(self):
        return float(self.gate_g.detach().norm().item()) if self.gate_g is not None else 0.0

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

    def forward(self, tokens, targets=None, sentiment=None):
        x = self.embed(tokens)
        gate_bias = None
        if sentiment is None and self._gate_bias_provider is not None:
            gb = self._gate_bias_provider(tokens)
            if gb is not None:
                gate_bias = gb.to(device=x.device, dtype=x.dtype)
        elif self.bias_mode and sentiment is not None and self.gate_g is not None:
            s = sentiment if torch.is_tensor(sentiment) else torch.tensor(
                [float(sentiment)] * tokens.shape[0], device=tokens.device, dtype=x.dtype)
            gate_bias = self.gate_g.unsqueeze(0) * s.unsqueeze(-1)
        x = self.run_blocks(x, gate_bias=gate_bias)
        logits = self.project_byte0(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss
