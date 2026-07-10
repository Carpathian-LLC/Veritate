# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Boundary-patched byte trunk (SpaceByte-style). Local blocks run on every byte;
#   global blocks run only on patch slots anchored at spacelike boundary bytes
#   (non-alphanumeric, non-UTF-8-continuation), then feed back into the byte stream.
#   Global attention/FFN cost scales with slots (seq/PATCH_STRIDE), not bytes:
#   more parameters per FLOP than the dense trunk at equal per-byte compute.
# - One hidden size everywhere; every block is a canonical model.Block, exposed via
#   self.blocks (enc + global + dec) so hook_spec()/dumps walk one shape. Fixed slot
#   count and fixed tensor shapes throughout (MPS kernel-cache rule).
# - Research variant: trains + checkpoints via save(); no Brain/load branch, not
#   .bin-exportable. Per-layer generation telemetry reconstructs a dense pipeline
#   and is not meaningful for the global blocks.
# - Streaming state carry (global_mixer="recurrent"): forward_streaming(tokens,
#   states) runs one fixed-shape window and carries each global RecurrentBlock's
#   state across windows (local attention stays within-window; positions + slot
#   embeddings window-local). Training forward is untouched; the trainer's
#   state_carry=chunks path consumes forward_streaming per chunk instead.
# veritate_core/model_patched.py
# ------------------------------------------------------------------------------------
# Imports:

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import (VOCAB_BYTE_LEVEL, ACT_DEFAULT, REG_DEFAULT, Block, RMSNorm,
                    QuantLinear)
from .model_recurrent import RecurrentBlock
from .model_moe import MoEFFN
from . import qat as _qat

# ------------------------------------------------------------------------------------
# Constants

PATCH_STRIDE  = 4
N_LOCAL_ENC   = 2
N_LOCAL_DEC   = 2
GLOBAL_MIXER_RECURRENT = "recurrent"
UTF8_CONT_LO  = 0x80
UTF8_CONT_HI  = 0xBF
LOOP_MAX      = 4
LOOP_UNIQUE_DIV = 2
LOOP_INJ_INIT = 0.1


def _boundary_table():
    t = torch.ones(VOCAB_BYTE_LEVEL, dtype=torch.bool)
    for b in range(VOCAB_BYTE_LEVEL):
        c = chr(b)
        if c.isalnum() and b < UTF8_CONT_LO:
            t[b] = False
        if UTF8_CONT_LO <= b <= UTF8_CONT_HI:
            t[b] = False
    return t

# ------------------------------------------------------------------------------------
# Functions


class VeritatePatched(nn.Module):
    def __init__(self, vocab, hidden, layers, ffn, heads, seq,
                 activation=ACT_DEFAULT, capture_l1=False, reg_mode=REG_DEFAULT,
                 global_mixer="attn", global_loops=1, global_ffn="dense",
                 state_rule="gla"):
        super().__init__()
        if vocab != VOCAB_BYTE_LEVEL:
            raise ValueError(f"vocab must be {VOCAB_BYTE_LEVEL} (byte-level only), got {vocab}")
        self.vocab      = vocab
        self.hidden     = hidden
        self.global_loops = global_loops
        self.eval_loops = global_loops
        self.glob_layers = layers // LOOP_UNIQUE_DIV if global_loops > 1 else layers
        self.layers     = N_LOCAL_ENC + self.glob_layers + N_LOCAL_DEC
        self.ffn        = ffn
        self.ffn_per_layer = [int(ffn)] * self.layers
        self.heads      = heads
        self.seq        = seq
        self.slots      = seq // PATCH_STRIDE
        self.activation = activation
        self.capture_l1 = bool(capture_l1)
        self.reg_mode   = reg_mode
        self.qat        = False

        self.global_mixer = global_mixer
        self.global_ffn   = global_ffn
        self.state_rule   = state_rule
        self.tok_emb  = nn.Embedding(vocab, hidden)
        self.pos_emb  = nn.Embedding(seq, hidden)
        self.slot_pos_emb = nn.Embedding(self.slots, hidden)
        glob_cls = RecurrentBlock if global_mixer == GLOBAL_MIXER_RECURRENT else Block
        def mk(cls):
            kw = dict(activation=activation, capture_l1=capture_l1, reg_mode=reg_mode)
            if cls is RecurrentBlock:
                kw["state_rule"] = state_rule
            return cls(hidden, ffn, heads, **kw)
        self.blocks   = nn.ModuleList(
            [mk(Block) for _ in range(N_LOCAL_ENC)]
            + [mk(glob_cls) for _ in range(self.glob_layers)]
            + [mk(Block) for _ in range(N_LOCAL_DEC)])
        if global_loops > 1:
            self.loop_inj = nn.Parameter(torch.full((LOOP_MAX,), LOOP_INJ_INIT))
        self.n_out    = RMSNorm(hidden)
        self.lm_head  = QuantLinear(hidden, vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        # float lookup table + F.embedding: the battle-tested indexed op on MPS
        # (bool advanced indexing showed transient index corruption there).
        self.register_buffer("boundary", _boundary_table().float().unsqueeze(-1), persistent=False)

        if global_ffn == "moe":
            for blk in self.blocks[N_LOCAL_ENC:N_LOCAL_ENC + self.glob_layers]:
                blk.ff = MoEFFN(hidden, ffn, activation=activation,
                                capture_l1=capture_l1, reg_mode=reg_mode)

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

    def moe_aux_sum(self):
        parts = [blk.ff._last_aux for blk in self.blocks
                 if hasattr(blk.ff, "_last_aux") and blk.ff._last_aux is not None]
        return sum(parts) if parts else None

    def moe_expert_share(self):
        return [blk.ff._last_share for blk in self.blocks
                if hasattr(blk.ff, "_last_share") and blk.ff._last_share is not None]

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
        return self.global_mixer == GLOBAL_MIXER_RECURRENT

    def forward_streaming(self, tokens, states=None):
        # inference-only window walk: training forward below stays untouched.
        # states = flat list of global-block states in invocation order (loops
        # included); positions + slot embeddings stay window-local.
        B, T = tokens.shape
        H = self.hidden
        S = min(self.slots, T)
        x = self.embed(tokens)
        for blk in self.blocks[:N_LOCAL_ENC]:
            x = blk(x)

        first = torch.zeros(T, dtype=torch.bool, device=tokens.device)
        first[0] = True
        is_b = (F.embedding(tokens, self.boundary).squeeze(-1) > 0.5) | first
        slot_of = is_b.long().cumsum(1) - 1
        pos_idx = torch.where(is_b, torch.arange(T, device=tokens.device, dtype=torch.int32).expand(B, T),
                              torch.full((B, T), T, device=tokens.device, dtype=torch.int32))
        bpos = pos_idx.sort(1).values[:, :S].long()
        slot_mask = (bpos < T).unsqueeze(-1)
        g = x.gather(1, bpos.clamp_max(T - 1).unsqueeze(-1).expand(B, S, H))
        g = (g + self.slot_pos_emb.weight[:S].unsqueeze(0)) * slot_mask
        glob = self.blocks[N_LOCAL_ENC:N_LOCAL_ENC + self.glob_layers]
        out_states = []
        if self.global_loops > 1:
            g_in = g
            for r in range(self.eval_loops):
                g = g + self.loop_inj[min(r, LOOP_MAX - 1)] * g_in
                for blk in glob:
                    st = states[len(out_states)] if states is not None else None
                    g, st = blk(g, state=st, return_state=True)
                    out_states.append(st)
        else:
            for blk in glob:
                st = states[len(out_states)] if states is not None else None
                g, st = blk(g, state=st, return_state=True)
                out_states.append(st)

        sel = slot_of.clamp(0, S - 1)
        add = g.gather(1, sel.unsqueeze(-1).expand(B, T, H))
        valid = (is_b & (slot_of < S)).unsqueeze(-1)
        x = x + add * valid
        for blk in self.blocks[N_LOCAL_ENC + self.glob_layers:]:
            x = blk(x)
        return self.project_byte0(x), out_states

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        H = self.hidden
        S = min(self.slots, T)
        x = self.embed(tokens)
        for blk in self.blocks[:N_LOCAL_ENC]:
            x = blk(x)

        first = torch.zeros(T, dtype=torch.bool, device=tokens.device)
        first[0] = True
        is_b = (F.embedding(tokens, self.boundary).squeeze(-1) > 0.5) | first
        slot_of = is_b.long().cumsum(1) - 1
        pos_idx = torch.where(is_b, torch.arange(T, device=tokens.device, dtype=torch.int32).expand(B, T),
                              torch.full((B, T), T, device=tokens.device, dtype=torch.int32))
        bpos = pos_idx.sort(1).values[:, :S].long()
        slot_mask = (bpos < T).unsqueeze(-1)
        g = x.gather(1, bpos.clamp_max(T - 1).unsqueeze(-1).expand(B, S, H))
        g = (g + self.slot_pos_emb.weight[:S].unsqueeze(0)) * slot_mask
        glob = self.blocks[N_LOCAL_ENC:N_LOCAL_ENC + self.glob_layers]
        if self.global_loops > 1:
            # weight-tied looped depth: R sampled in training (Huginn-style) so
            # test-time depth scaling emerges; input re-injected each loop.
            R = int(torch.randint(1, LOOP_MAX + 1, (1,)).item()) if self.training else self.eval_loops
            g_in = g
            for r in range(R):
                g = g + self.loop_inj[min(r, LOOP_MAX - 1)] * g_in
                for blk in glob:
                    g = blk(g)
        else:
            for blk in glob:
                g = blk(g)

        sel = slot_of.clamp(0, S - 1)
        add = g.gather(1, sel.unsqueeze(-1).expand(B, T, H))
        valid = (is_b & (slot_of < S)).unsqueeze(-1)
        x = x + add * valid
        for blk in self.blocks[N_LOCAL_ENC + self.glob_layers:]:
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
