# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Surprise-gated neural memory trunk (Titans MAG class). Canonical dense trunk
#   plus one parallel memory branch at mid-depth: a 2-layer MLP whose weights are
#   FAST WEIGHTS, updated inside the forward pass by the gradient of its own
#   key->value recall loss (the surprise signal), with learned momentum and decay
#   (forgetting). Knowledge is written into memory during use, not by corpus
#   gradient descent. Read happens before write per chunk, so causality holds.
# - The inner gradient is computed in closed form (manual backprop through the
#   tiny MLP, plain matmuls) and stays inside the autograd graph, so the outer
#   optimizer trains the write rule itself (TTT-style). Fixed shapes on MPS.
# - carry_memory()/reset_memory() expose the fast weights across forward calls so
#   an eval harness can feed a document in windows and quiz later: the memory is
#   the persistence mechanism, the context window is not.
# - Research variant: trains + checkpoints via save(); no Brain/load branch, not
#   .bin-exportable.
# veritate_core/model_memory.py
# ------------------------------------------------------------------------------------
# Imports:

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import qat as _qat
from .model import ACT_DEFAULT, REG_DEFAULT, VOCAB_BYTE_LEVEL, Block, QuantLinear, RMSNorm

# ------------------------------------------------------------------------------------
# Constants

MEM_CHUNK      = 64
MEM_DIM_FRAC   = 4
MEM_WIDTH_FRAC = 2
MEM_LAYER_FRAC = 2
INIT_LR        = -2.0
INIT_MOMENTUM  = 1.0
INIT_DECAY     = -4.0

# ------------------------------------------------------------------------------------
# Functions


class NeuralMemory(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        d = hidden // MEM_DIM_FRAC
        m = hidden // MEM_WIDTH_FRAC
        self.d, self.m = d, m
        self.q_proj = nn.Linear(hidden, d, bias=False)
        self.k_proj = nn.Linear(hidden, d, bias=False)
        self.v_proj = nn.Linear(hidden, d, bias=False)
        self.o_proj = nn.Linear(d, hidden, bias=False)
        self.gate   = nn.Linear(hidden, hidden, bias=False)
        self.w1_init = nn.Parameter(torch.randn(d, m) * 0.02)
        self.w2_init = nn.Parameter(torch.randn(m, d) * 0.02)
        self.lr       = nn.Parameter(torch.tensor(INIT_LR))
        self.momentum = nn.Parameter(torch.tensor(INIT_MOMENTUM))
        self.decay    = nn.Parameter(torch.tensor(INIT_DECAY))
        self.state = None

    def reset_memory(self):
        self.state = None

    def carry_memory(self):
        if self.state is not None:
            self.state = tuple(s.detach() for s in self.state)

    def _init_state(self, B, device, dtype):
        w1 = self.w1_init.unsqueeze(0).expand(B, -1, -1).to(dtype)
        w2 = self.w2_init.unsqueeze(0).expand(B, -1, -1).to(dtype)
        return (w1, w2, torch.zeros_like(w1), torch.zeros_like(w2))

    @staticmethod
    def _phi(z):
        return z * torch.sigmoid(1.702 * z)

    def _read(self, w1, w2, q):
        return self._phi(q @ w1) @ w2

    def forward(self, x):
        B, T, _C = x.shape
        pad = (MEM_CHUNK - T % MEM_CHUNK) % MEM_CHUNK
        xp = F.pad(x, (0, 0, 0, pad)) if pad else x
        Tp = T + pad
        # unit-norm keys/queries bound the inner loss surface (TTT/Titans rule);
        # without this the fast-weight update explodes at width >= 320.
        q = F.normalize(self.q_proj(xp), dim=-1)
        k = F.normalize(self.k_proj(xp), dim=-1)
        v = self.v_proj(xp)
        if self.state is None or self.state[0].shape[0] != B:
            self.state = self._init_state(B, x.device, x.dtype)
        w1, w2, s1, s2 = self.state
        eta   = torch.sigmoid(self.lr)
        beta  = torch.sigmoid(self.momentum)
        alpha = torch.sigmoid(self.decay)
        outs = []
        for c in range(Tp // MEM_CHUNK):
            s = c * MEM_CHUNK
            qc, kc, vc = q[:, s:s + MEM_CHUNK], k[:, s:s + MEM_CHUNK], v[:, s:s + MEM_CHUNK]
            outs.append(self._read(w1, w2, qc))
            h_pre = kc @ w1
            h = self._phi(h_pre)
            err = h @ w2 - vc
            g2 = h.transpose(-1, -2) @ err / MEM_CHUNK
            dh = err @ w2.transpose(-1, -2)
            gp = dh * (torch.sigmoid(1.702 * h_pre) * (1 + 1.702 * h_pre * torch.sigmoid(-1.702 * h_pre)))
            g1 = kc.transpose(-1, -2) @ gp / MEM_CHUNK
            s1 = beta * s1 - eta * g1
            s2 = beta * s2 - eta * g2
            w1 = (1 - alpha) * w1 + s1
            w2 = (1 - alpha) * w2 + s2
        self.state = (w1, w2, s1, s2)
        y = torch.cat(outs, dim=1)[:, :T]
        return self.o_proj(y) * torch.sigmoid(self.gate(x))


class VeritateMemory(nn.Module):
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
        self.mem_layer  = layers // MEM_LAYER_FRAC

        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(seq, hidden)
        self.blocks  = nn.ModuleList([Block(hidden, ffn, heads,
                                            activation=activation,
                                            capture_l1=capture_l1,
                                            reg_mode=reg_mode)
                                      for _ in range(layers)])
        self.memory  = NeuralMemory(hidden)
        self.n_out   = RMSNorm(hidden)
        self.lm_head = QuantLinear(hidden, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        for p in self.parameters():
            if p.dim() >= 2:
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

    def reset_memory(self):
        self.memory.reset_memory()

    def carry_memory(self):
        self.memory.carry_memory()

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
        self.reset_memory()
        x = self.embed(tokens)
        for L, blk in enumerate(self.blocks):
            x = blk(x)
            if self.mem_layer == L:
                x = x + self.memory(x)
        logits = self.project_byte0(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def forward_carry(self, tokens, targets=None):
        x = self.embed(tokens)
        for L, blk in enumerate(self.blocks):
            x = blk(x)
            if self.mem_layer == L:
                x = x + self.memory(x)
        logits = self.project_byte0(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss
