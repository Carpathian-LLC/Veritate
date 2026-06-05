# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# tools/coral/merge.py
# ------------------------------------------------------------------------------------
# The Coral Merge algorithm — Polyphase Distill-Merge. Combines two same-shape
# Veritate base models trained on different corpora into a single same-shape
# blended model. Three phases:
#
#   1. Alignment    — Hungarian solver on per-layer FFN intermediate-activation
#                     cross-correlations. Output: per-layer permutation P_ell.
#                     Applied to B by permuting rows of ff.up.weight and columns
#                     of ff.down.weight — symmetry-preserving, output unchanged.
#
#   2. Splice       — Per-matrix scalar coefficients (alpha_M, beta_M) initialised
#                     to 0.5 each. Blended weight M = alpha*M_A + beta*M_B_aligned.
#
#   3. Distill-refine — Two sub-phases on the mixed corpus:
#                     3a: freeze weights, train only scalars (5 percent of budget).
#                     3b: unfreeze weights, keep scalars trainable (95 percent).
#                     Loss = (1-lam)*CE + lam_a*KL(s/T||T_a/T) + lam_b*KL(s/T||T_b/T)
#                     with both originals as frozen teachers.
#
# Output: blend checkpoint at models/<out_name>/checkpoints/step_<refine>.pt,
# train.csv per the canonical contract, coral_meta.json sidecar.
#
# Spec: ~/Documents/GitHub/Agent-Documents/Veritate/coral_merge_spec.md
#
# Usage:
#   python tools/coral/merge.py \
#     --name_a   coral_a_tinystories_30m \
#     --name_b   coral_b_distill_v1_30m \
#     --out_name coral_blend_30m \
#     --corpus   distill_v1_mix_tinystories \
#     --refine_steps 1500
# ------------------------------------------------------------------------------------

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "veritate_mri"))

import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

from veritate_core.model import Veritate                              # noqa: E402
from training import save                                             # noqa: E402
from readers import paths                                             # noqa: E402


# ----------------------------------------------------------------------------
# Args

def parse_args():
    ap = argparse.ArgumentParser(description="Coral Merge — polyphase distill-merge")
    ap.add_argument("--name_a",        required=True)
    ap.add_argument("--name_b",        required=True)
    ap.add_argument("--out_name",      required=True)
    ap.add_argument("--corpus",        required=True, help="mixed corpus stem for distill-refine")
    ap.add_argument("--step_a",        type=int, default=-1, help="-1 = latest checkpoint")
    ap.add_argument("--step_b",        type=int, default=-1)
    ap.add_argument("--align_samples", type=int, default=2048)
    ap.add_argument("--refine_steps",  type=int, default=1500)
    ap.add_argument("--scalar_phase_frac", type=float, default=0.05)
    ap.add_argument("--batch",         type=int, default=16)
    ap.add_argument("--seq",           type=int, default=256)
    ap.add_argument("--lr_scalar",     type=float, default=1e-2,
                    help="lr for the (alpha,beta) scalars in phase 3a")
    ap.add_argument("--lr_weight",     type=float, default=3e-5,
                    help="lr for the merged weights in phase 3b (10x smaller than constituent lr)")
    ap.add_argument("--temperature",   type=float, default=2.0)
    ap.add_argument("--lambda_ce",     type=float, default=0.5)
    ap.add_argument("--lambda_kl",     type=float, default=0.25,
                    help="per-teacher KL weight; total KL contribution = 2 * lambda_kl")
    ap.add_argument("--log_every",     type=int, default=25)
    ap.add_argument("--eval_every",    type=int, default=250)
    ap.add_argument("--eval_iters",    type=int, default=8)
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--device",        default="")
    ap.add_argument("--description",   default="coral merge — polyphase distill-merge")
    return ap.parse_args()


def pick_device(pref: str) -> str:
    if pref: return pref
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# Checkpoint loading

def latest_step(name):
    ckpt_dir = os.path.join(REPO, "models", name, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"no checkpoints dir for: {name}")
    steps = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith("step_") and fn.endswith(".pt"):
            try: steps.append(int(fn[5:-3]))
            except ValueError: pass
    if not steps:
        raise SystemExit(f"no step_*.pt under: {ckpt_dir}")
    return max(steps)


def load_constituent(name, step):
    if step <= 0:
        step = latest_step(name)
    cp_path = os.path.join(REPO, "models", name, "checkpoints", f"step_{step}.pt")
    ckpt = torch.load(cp_path, map_location="cpu")
    shape = dict(ckpt.get("shape") or {})
    args  = ckpt.get("args") or {}
    for k in ("hidden", "layers", "ffn", "heads", "seq"):
        if k not in shape and k in args:
            shape[k] = args[k]
        if k not in shape:
            raise SystemExit(f"checkpoint {cp_path} missing shape key {k!r}")
    model = Veritate(vocab=256,
                     hidden=shape["hidden"], layers=shape["layers"],
                     ffn=shape["ffn"], heads=shape["heads"],
                     seq=int(shape["seq"]))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, shape, step, cp_path


def check_same_shape(s_a, s_b):
    for k in ("hidden", "layers", "ffn", "heads", "seq"):
        if int(s_a[k]) != int(s_b[k]):
            raise SystemExit(f"shape mismatch on {k!r}: A={s_a[k]} B={s_b[k]} — Coral Merge requires identical shapes")


# ----------------------------------------------------------------------------
# Corpus

def make_loader(bin_path, batch, seq, seed):
    arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    n = len(arr)
    if n < seq + 1:
        raise SystemExit(f"corpus too small: {bin_path}")
    rng = np.random.default_rng(seed)

    def draw():
        starts = rng.integers(0, n - seq - 1, size=batch, dtype=np.int64)
        x = np.stack([np.asarray(arr[s:s + seq],         dtype=np.int64) for s in starts])
        y = np.stack([np.asarray(arr[s + 1:s + 1 + seq], dtype=np.int64) for s in starts])
        return torch.from_numpy(x), torch.from_numpy(y)
    return draw, n


# ----------------------------------------------------------------------------
# Phase 1 — Alignment

def collect_ffn_intermediates(model, corpus_loader, n_samples, batch, seq, device):
    """Hook ff.down to capture its INPUT (i.e. post-activation FFN intermediate).
    Returns a list of [n_samples * batch * seq, ffn] tensors, one per layer.
    """
    model = model.to(device)
    captured = [[] for _ in range(model.layers)]

    handles = []
    for li, blk in enumerate(model.blocks):
        def make_hook(idx):
            def hook(_module, inputs, _output):
                act = inputs[0]               # [B, T, F]
                captured[idx].append(act.detach().reshape(-1, act.shape[-1]).cpu())
            return hook
        handles.append(blk.ff.down.register_forward_hook(make_hook(li)))

    n_batches = max(1, n_samples // (batch * seq))
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = corpus_loader()
            x = x.to(device)
            model(x)
    for h in handles: h.remove()

    return [torch.cat(c, dim=0) for c in captured]   # each [N, F]


def hungarian_perm(cross_correlation):
    """Solve max-cost assignment on a square matrix. cross_correlation[i,j] is
    similarity of A's neuron i to B's neuron j. Returns perm such that
    perm[i] = j means B's j-th neuron pairs with A's i-th neuron.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Greedy fallback — N^2*log(N), much worse but works without scipy.
        F_ = cross_correlation.shape[0]
        cc = cross_correlation.clone()
        perm = torch.zeros(F_, dtype=torch.long)
        used = torch.zeros(F_, dtype=torch.bool)
        for i in range(F_):
            j_best = -1
            v_best = -float("inf")
            for j in range(F_):
                if used[j]: continue
                v = float(cc[i, j])
                if v > v_best:
                    v_best = v; j_best = j
            perm[i] = j_best
            used[j_best] = True
        return perm

    row, col = linear_sum_assignment(-cross_correlation.numpy())
    perm = torch.from_numpy(col).long()
    return perm


def align_ffn(model_a, model_b, corpus_loader, n_samples, batch, seq, device, log=print):
    """Returns: list of perms (length L), each shape [FFN], such that applying
    perm to B's layer-ell ff.up rows + ff.down cols brings B into A's FFN basis.
    """
    log("[coral.align] collecting activations from A...")
    acts_a = collect_ffn_intermediates(model_a, corpus_loader, n_samples, batch, seq, device)
    log("[coral.align] collecting activations from B...")
    acts_b = collect_ffn_intermediates(model_b, corpus_loader, n_samples, batch, seq, device)

    perms = []
    for li in range(model_a.layers):
        a = acts_a[li]   # [N, F]
        b = acts_b[li]   # [N, F]
        n = min(a.shape[0], b.shape[0])
        a = a[:n].float()
        b = b[:n].float()
        # Center + normalize for stable correlation.
        a = a - a.mean(dim=0, keepdim=True)
        b = b - b.mean(dim=0, keepdim=True)
        a_n = a / (a.norm(dim=0, keepdim=True) + 1e-8)
        b_n = b / (b.norm(dim=0, keepdim=True) + 1e-8)
        C = a_n.t() @ b_n                # [F, F]
        perm = hungarian_perm(C)
        perms.append(perm)
        log(f"[coral.align] layer {li}  trace_before={float(torch.diag(C).sum()):.3f}  "
            f"trace_after={float(C[torch.arange(C.shape[0]), perm].sum()):.3f}")
    return perms


def apply_perms_to_state_dict(sd_b, perms):
    """Permute B's FFN intermediate dimension in-place to match A's basis.
    For each layer: ff.up.weight rows (dim 0) and ff.down.weight cols (dim 1).
    """
    out = {}
    for k, v in sd_b.items():
        out[k] = v.clone()
    for li, perm in enumerate(perms):
        kup = f"blocks.{li}.ff.up.weight"
        kdn = f"blocks.{li}.ff.down.weight"
        if kup in out and kdn in out:
            out[kup] = out[kup][perm, :]
            out[kdn] = out[kdn][:, perm]
    return out


# ----------------------------------------------------------------------------
# Phase 2 + 3 — Spliced model with learnable scalars + distill-refine

class CoralSpliceLinear(torch.nn.Module):
    """Wraps two same-shape weight tensors with learnable scalars (alpha,beta).
    Effective weight: alpha*W_a + beta*W_b. Bias is treated the same way if
    present. Use this in place of nn.Linear for the merged model.
    """
    def __init__(self, w_a, w_b, b_a=None, b_b=None, init=0.5):
        super().__init__()
        self.W_a = torch.nn.Parameter(w_a.clone())
        self.W_b = torch.nn.Parameter(w_b.clone())
        self.alpha = torch.nn.Parameter(torch.tensor(float(init)))
        self.beta  = torch.nn.Parameter(torch.tensor(float(init)))
        self.has_bias = b_a is not None and b_b is not None
        if self.has_bias:
            self.b_a = torch.nn.Parameter(b_a.clone())
            self.b_b = torch.nn.Parameter(b_b.clone())

    def effective_weight(self):
        return self.alpha * self.W_a + self.beta * self.W_b

    def effective_bias(self):
        if not self.has_bias:
            return None
        return self.alpha * self.b_a + self.beta * self.b_b

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.effective_bias())


class CoralSpliceParam(torch.nn.Module):
    """1D parameter splice — for RMSNorm scales, embeddings, etc."""
    def __init__(self, p_a, p_b, init=0.5):
        super().__init__()
        self.P_a   = torch.nn.Parameter(p_a.clone())
        self.P_b   = torch.nn.Parameter(p_b.clone())
        self.alpha = torch.nn.Parameter(torch.tensor(float(init)))
        self.beta  = torch.nn.Parameter(torch.tensor(float(init)))

    def effective(self):
        return self.alpha * self.P_a + self.beta * self.P_b


def build_blend(model_a, model_b_aligned, shape):
    """Construct a Veritate-compatible model whose weights are spliced from A
    and B-aligned via learnable scalars. We monkey-graft the splice modules
    onto a fresh Veritate skeleton so its forward pass is unmodified.
    """
    blend = Veritate(vocab=256, hidden=shape["hidden"], layers=shape["layers"],
                     ffn=shape["ffn"], heads=shape["heads"], seq=int(shape["seq"]))

    sd_a = model_a.state_dict()
    sd_b = model_b_aligned.state_dict()

    splices = []   # collect for parameter-group construction

    def splice_linear(parent, name):
        a = getattr(parent, name)
        wa = sd_a[_qualified(parent, name) + ".weight"]
        wb = sd_b[_qualified(parent, name) + ".weight"]
        ba = sd_a.get(_qualified(parent, name) + ".bias")
        bb = sd_b.get(_qualified(parent, name) + ".bias")
        spl = CoralSpliceLinear(wa, wb, ba, bb)
        setattr(parent, name, _LinearAdapter(spl, a))
        splices.append(spl)

    def splice_param(parent, name):
        wa = sd_a[_qualified(parent, name) + ".weight"]
        wb = sd_b[_qualified(parent, name) + ".weight"]
        spl = CoralSpliceParam(wa, wb)
        # Inject as a property by replacing the buffer's parameter slot.
        setattr(parent, "_coral_" + name, spl)
        # Replace forward-time read via patching: keep the original module but
        # patch its .weight via a closure-bound forward override.
        _patch_param_forward(parent, name, spl)
        splices.append(spl)

    # Token embedding (tied with lm_head)
    spl = CoralSpliceParam(sd_a["tok_emb.weight"], sd_b["tok_emb.weight"])
    blend.tok_emb = _EmbeddingAdapter(spl, blend.tok_emb)
    # lm_head.weight is tied — we re-tie below
    splices.append(spl)
    blend.lm_head = _LinearTiedAdapter(blend.tok_emb)

    # Pos embedding
    spl_pos = CoralSpliceParam(sd_a["pos_emb.weight"], sd_b["pos_emb.weight"])
    blend.pos_emb = _EmbeddingAdapter(spl_pos, blend.pos_emb)
    splices.append(spl_pos)

    # Per-block
    for li, blk in enumerate(blend.blocks):
        # n1 (RMSNorm scale)
        spl_n1 = CoralSpliceParam(sd_a[f"blocks.{li}.n1.weight"], sd_b[f"blocks.{li}.n1.weight"])
        blk.n1 = _RMSNormAdapter(spl_n1, blk.n1)
        splices.append(spl_n1)
        # attn.qkv
        spl_qkv = _make_splice_linear(sd_a, sd_b, f"blocks.{li}.attn.qkv")
        blk.attn.qkv = spl_qkv
        splices.append(spl_qkv)
        # attn.proj
        spl_op = _make_splice_linear(sd_a, sd_b, f"blocks.{li}.attn.proj")
        blk.attn.proj = spl_op
        splices.append(spl_op)
        # n2
        spl_n2 = CoralSpliceParam(sd_a[f"blocks.{li}.n2.weight"], sd_b[f"blocks.{li}.n2.weight"])
        blk.n2 = _RMSNormAdapter(spl_n2, blk.n2)
        splices.append(spl_n2)
        # ff.up
        spl_up = _make_splice_linear(sd_a, sd_b, f"blocks.{li}.ff.up")
        blk.ff.up = spl_up
        splices.append(spl_up)
        # ff.down
        spl_dn = _make_splice_linear(sd_a, sd_b, f"blocks.{li}.ff.down")
        blk.ff.down = spl_dn
        splices.append(spl_dn)

    # Output RMSNorm
    spl_no = CoralSpliceParam(sd_a["n_out.weight"], sd_b["n_out.weight"])
    blend.n_out = _RMSNormAdapter(spl_no, blend.n_out)
    splices.append(spl_no)

    return blend, splices


def _make_splice_linear(sd_a, sd_b, prefix):
    wa = sd_a[prefix + ".weight"]
    wb = sd_b[prefix + ".weight"]
    ba = sd_a.get(prefix + ".bias")
    bb = sd_b.get(prefix + ".bias")
    return CoralSpliceLinear(wa, wb, ba, bb)


def _qualified(parent, name):
    """Heuristic — we don't actually need this for the linear splice path; the
    callers pass an explicit prefix to _make_splice_linear instead."""
    return name


class _LinearAdapter(torch.nn.Module):
    def __init__(self, splice, _ignore_old):
        super().__init__()
        self.splice = splice
    def forward(self, x):
        return self.splice(x)


class _EmbeddingAdapter(torch.nn.Module):
    def __init__(self, splice, old_emb):
        super().__init__()
        self.splice = splice
        self.num_embeddings  = old_emb.num_embeddings
        self.embedding_dim   = old_emb.embedding_dim
    @property
    def weight(self):
        return self.splice.effective()
    def forward(self, idx):
        return F.embedding(idx, self.splice.effective())


class _RMSNormAdapter(torch.nn.Module):
    def __init__(self, splice, old_rms):
        super().__init__()
        self.splice = splice
        self.eps    = getattr(old_rms, "eps", 1e-6)
    def forward(self, x):
        scale = self.splice.effective()
        # standard RMSNorm
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * scale


class _LinearTiedAdapter(torch.nn.Module):
    """lm_head tied to tok_emb adapter — uses tok_emb's splice as its weight."""
    def __init__(self, tok_emb_adapter):
        super().__init__()
        self.tok_emb_adapter = tok_emb_adapter
    @property
    def weight(self):
        return self.tok_emb_adapter.splice.effective()
    def forward(self, x):
        return F.linear(x, self.tok_emb_adapter.splice.effective())


def _patch_param_forward(parent, name, splice):
    # Reserved hook — not used in the current adapter path.
    return


# ----------------------------------------------------------------------------
# Distill-refine

def freeze_weights_keep_scalars(splices):
    for s in splices:
        for p in s.parameters():
            p.requires_grad = False
        s.alpha.requires_grad = True
        s.beta.requires_grad  = True


def unfreeze_all(splices):
    for s in splices:
        for p in s.parameters():
            p.requires_grad = True


def distill_step(blend, teacher_a, teacher_b, x, y, T, lambda_ce, lambda_kl):
    logits_s, _ = blend(x, targets=None)
    with torch.no_grad():
        logits_a, _ = teacher_a(x, targets=None)
        logits_b, _ = teacher_b(x, targets=None)

    # CE on student
    ce = F.cross_entropy(logits_s.reshape(-1, logits_s.size(-1)), y.reshape(-1))

    # KL between scaled-softmax student and each teacher.
    log_p_s = F.log_softmax(logits_s / T, dim=-1)
    p_a     = F.softmax(logits_a / T,     dim=-1)
    p_b     = F.softmax(logits_b / T,     dim=-1)
    kl_a = F.kl_div(log_p_s, p_a, reduction="batchmean") * (T * T)
    kl_b = F.kl_div(log_p_s, p_b, reduction="batchmean") * (T * T)

    loss = lambda_ce * ce + lambda_kl * kl_a + lambda_kl * kl_b
    return loss, ce.detach(), kl_a.detach(), kl_b.detach()


def lr_at(step, total, warmup, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


# ----------------------------------------------------------------------------
# Main

def main():
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"[coral.merge] loading A={args.name_a} step={args.step_a}", flush=True)
    model_a, shape_a, step_a, cp_a = load_constituent(args.name_a, args.step_a)
    print(f"[coral.merge] loaded A step={step_a} from {cp_a}", flush=True)

    print(f"[coral.merge] loading B={args.name_b} step={args.step_b}", flush=True)
    model_b, shape_b, step_b, cp_b = load_constituent(args.name_b, args.step_b)
    print(f"[coral.merge] loaded B step={step_b} from {cp_b}", flush=True)

    check_same_shape(shape_a, shape_b)
    shape = shape_a

    train_path = paths.corpus_train_path(args.corpus)
    val_path   = paths.corpus_val_path(args.corpus)
    if not os.path.isfile(train_path):
        raise SystemExit(f"no train corpus: {train_path}")
    has_val = os.path.isfile(val_path)
    train_draw, n_train = make_loader(train_path, args.batch, args.seq, seed=args.seed)
    val_draw            = make_loader(val_path,   args.batch, args.seq, seed=args.seed + 1)[0] if has_val else None

    # Phase 1 — alignment
    perms = align_ffn(model_a, model_b, train_draw, args.align_samples,
                      args.batch, args.seq, device)
    perms_hash = abs(hash(tuple(tuple(p.tolist()) for p in perms))) % (10**12)
    print(f"[coral.merge] perms hash={perms_hash}", flush=True)

    # Apply perms to B
    sd_b_aligned = apply_perms_to_state_dict(model_b.state_dict(), perms)
    model_b_aligned = Veritate(vocab=256, hidden=shape["hidden"], layers=shape["layers"],
                               ffn=shape["ffn"], heads=shape["heads"], seq=int(shape["seq"]))
    model_b_aligned.load_state_dict(sd_b_aligned, strict=True)
    model_b_aligned.eval()

    # Phase 2 — build blend with learnable scalars
    print(f"[coral.merge] building spliced blend...", flush=True)
    blend, splices = build_blend(model_a, model_b_aligned, shape)
    blend = blend.to(device)
    model_a = model_a.to(device); model_b_aligned = model_b_aligned.to(device)
    for p in model_a.parameters():        p.requires_grad = False
    for p in model_b_aligned.parameters(): p.requires_grad = False

    n_scalars = sum(2 for _ in splices)
    n_blend_params = sum(p.numel() for p in blend.parameters() if p.requires_grad)
    print(f"[coral.merge] blend params={n_blend_params:,}  scalar pairs={n_scalars}", flush=True)

    # Output dir + config + meta
    out_dir  = os.path.join(REPO, "models", args.out_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    write_blend_config(out_dir, args, shape, n_blend_params, train_path, val_path,
                       cp_a, cp_b, perms_hash)

    # Phase 3a — scalar warmup
    scalar_steps = max(1, int(args.refine_steps * args.scalar_phase_frac))
    weight_steps = args.refine_steps - scalar_steps
    print(f"[coral.merge] phase 3a: {scalar_steps} steps scalars-only", flush=True)
    freeze_weights_keep_scalars(splices)
    scalar_params = [p for s in splices for p in (s.alpha, s.beta)]
    opt = torch.optim.AdamW(scalar_params, lr=args.lr_scalar, betas=(0.9, 0.95), weight_decay=0.0)

    t0 = time.time()
    last_step_time = t0

    for step in range(1, scalar_steps + 1):
        lr = lr_at(step, scalar_steps, max(1, scalar_steps // 10),
                   args.lr_scalar, args.lr_scalar * 0.1)
        for g in opt.param_groups: g["lr"] = lr
        x, y = train_draw()
        x, y = x.to(device), y.to(device)
        loss, ce, kl_a, kl_b = distill_step(blend, model_a, model_b_aligned, x, y,
                                            args.temperature, args.lambda_ce, args.lambda_kl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(scalar_params, 1.0))
        opt.step()

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            tokens = args.log_every * args.batch * args.seq
            tps = tokens / max(1e-6, now - last_step_time)
            save.append_train_row(args.out_name, step, "train",
                                  float(loss.item()), float(lr),
                                  float(gnorm), float(tps), float(now - t0),
                                  int(args.seed))
            print(f"[coral.merge.3a] step {step}/{scalar_steps}  loss={loss.item():.4f}  "
                  f"ce={ce:.3f}  kl_a={kl_a:.3f}  kl_b={kl_b:.3f}  lr={lr:.2e}", flush=True)
            last_step_time = now

    # Phase 3b — full refine
    print(f"[coral.merge] phase 3b: {weight_steps} steps full refine", flush=True)
    unfreeze_all(splices)
    all_params = [p for s in splices for p in s.parameters()]
    opt = torch.optim.AdamW(all_params, lr=args.lr_weight, betas=(0.9, 0.95), weight_decay=0.01)

    base_step = scalar_steps
    for step in range(1, weight_steps + 1):
        lr = lr_at(step, weight_steps, max(1, weight_steps // 20),
                   args.lr_weight, args.lr_weight * 0.1)
        for g in opt.param_groups: g["lr"] = lr
        x, y = train_draw()
        x, y = x.to(device), y.to(device)
        loss, ce, kl_a, kl_b = distill_step(blend, model_a, model_b_aligned, x, y,
                                            args.temperature, args.lambda_ce, args.lambda_kl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(all_params, 1.0))
        opt.step()

        global_step = base_step + step
        if step % args.log_every == 0 or step == 1:
            now = time.time()
            tokens = args.log_every * args.batch * args.seq
            tps = tokens / max(1e-6, now - last_step_time)
            save.append_train_row(args.out_name, global_step, "train",
                                  float(loss.item()), float(lr),
                                  float(gnorm), float(tps), float(now - t0),
                                  int(args.seed))
            print(f"[coral.merge.3b] step {step}/{weight_steps}  loss={loss.item():.4f}  "
                  f"ce={ce:.3f}  kl_a={kl_a:.3f}  kl_b={kl_b:.3f}  lr={lr:.2e}", flush=True)
            last_step_time = now

        if args.eval_every and step % args.eval_every == 0 and val_draw is not None:
            blend.eval()
            losses = []
            with torch.no_grad():
                for _ in range(args.eval_iters):
                    xv, yv = val_draw()
                    xv, yv = xv.to(device), yv.to(device)
                    _logits, vloss_t = blend(xv, targets=yv)
                    losses.append(float(vloss_t.item()))
            vloss = sum(losses) / max(1, len(losses))
            save.append_train_row(args.out_name, global_step, "val",
                                  float(vloss), float(lr),
                                  0.0, 0.0, float(time.time() - t0),
                                  int(args.seed))
            print(f"[coral.merge.3b] step {global_step}  VAL loss={vloss:.4f}", flush=True)
            blend.train()

    # Save blended checkpoint
    final_step = scalar_steps + weight_steps
    # Materialize the blend as a vanilla Veritate state_dict (collapse splices
    # to their effective tensors) so downstream tools can load it as a plain
    # checkpoint without needing the merge.py adapters.
    materialized = materialize_blend_state_dict(blend)
    final_path = os.path.join(ckpt_dir, f"step_{final_step}.pt")
    tmp = final_path + ".tmp"
    torch.save({"model": materialized, "step": final_step,
                "args":  {"name": args.out_name,
                          "parents": [args.name_a, args.name_b],
                          "perms_hash": perms_hash,
                          "scalar_phase_frac": args.scalar_phase_frac,
                          "temperature": args.temperature,
                          "lambda_ce": args.lambda_ce,
                          "lambda_kl": args.lambda_kl,
                          "refine_steps": args.refine_steps,
                          "layers": shape["layers"], "hidden": shape["hidden"],
                          "ffn":    shape["ffn"],    "heads":  shape["heads"],
                          "seq":    int(shape["seq"]), "vocab":  256},
                "shape": shape}, tmp)
    os.replace(tmp, final_path)

    meta = {
        "out_name": args.out_name,
        "parent_a": args.name_a, "step_a": step_a, "ckpt_a": cp_a,
        "parent_b": args.name_b, "step_b": step_b, "ckpt_b": cp_b,
        "corpus": args.corpus,
        "shape": shape,
        "perms_hash": perms_hash,
        "alpha_beta_final": {
            f"splice_{i}": {"alpha": float(s.alpha.detach().cpu()),
                            "beta":  float(s.beta.detach().cpu())}
            for i, s in enumerate(splices)
        },
        "refine_steps": args.refine_steps,
        "scalar_phase_steps": scalar_steps,
        "weight_phase_steps": weight_steps,
        "temperature": args.temperature,
        "lambda_ce": args.lambda_ce,
        "lambda_kl": args.lambda_kl,
        "wrote_at": int(time.time()),
    }
    with open(os.path.join(out_dir, "coral_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[coral.merge] DONE in {time.time() - t0:.1f}s  ckpt={final_path}", flush=True)


def write_blend_config(out_dir, args, shape, n_params, train_path, val_path, cp_a, cp_b, perms_hash):
    cfg = {
        "name": args.out_name,
        "description": args.description,
        "kind": "trainer",
        "plugin": "coral_merge",
        "vocab": 256,
        "shape": {
            "vocab": 256,
            "hidden": shape["hidden"],
            "layers": shape["layers"],
            "ffn":    shape["ffn"],
            "heads":  shape["heads"],
            "seq":    int(shape["seq"]),
        },
        "training_args": {
            "name":         args.out_name,
            "parent_a":     args.name_a,
            "parent_b":     args.name_b,
            "checkpoint_a": cp_a,
            "checkpoint_b": cp_b,
            "perms_hash":   perms_hash,
            "corpus":       args.corpus,
            "corpus_train": train_path,
            "corpus_val":   val_path,
            "refine_steps": int(args.refine_steps),
            "batch":        int(args.batch),
            "seq":          int(args.seq),
            "lr_scalar":    float(args.lr_scalar),
            "lr_weight":    float(args.lr_weight),
            "temperature":  float(args.temperature),
            "lambda_ce":    float(args.lambda_ce),
            "lambda_kl":    float(args.lambda_kl),
            "seed":         int(args.seed),
        },
        "training": "coral_merge",
        "n_params_total": int(n_params),
        "wrote_at": int(time.time()),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def materialize_blend_state_dict(blend):
    """Walk the blend, collapse every splice into its effective tensor, and
    return a state_dict that loads cleanly into a plain Veritate of the same
    shape. Lets downstream code consume the merged model without the splice
    adapters.
    """
    out = {}
    out["tok_emb.weight"] = blend.tok_emb.splice.effective().detach().clone()
    out["pos_emb.weight"] = blend.pos_emb.splice.effective().detach().clone()
    for li, blk in enumerate(blend.blocks):
        out[f"blocks.{li}.n1.weight"]            = blk.n1.splice.effective().detach().clone()
        out[f"blocks.{li}.attn.qkv.weight"]      = blk.attn.qkv.effective_weight().detach().clone()
        b = blk.attn.qkv.effective_bias()
        if b is not None: out[f"blocks.{li}.attn.qkv.bias"] = b.detach().clone()
        out[f"blocks.{li}.attn.proj.weight"] = blk.attn.proj.effective_weight().detach().clone()
        b = blk.attn.proj.effective_bias()
        if b is not None: out[f"blocks.{li}.attn.proj.bias"] = b.detach().clone()
        out[f"blocks.{li}.n2.weight"]            = blk.n2.splice.effective().detach().clone()
        out[f"blocks.{li}.ff.up.weight"]         = blk.ff.up.effective_weight().detach().clone()
        out[f"blocks.{li}.ff.down.weight"]       = blk.ff.down.effective_weight().detach().clone()
    out["n_out.weight"]   = blend.n_out.splice.effective().detach().clone()
    out["lm_head.weight"] = out["tok_emb.weight"]   # tied
    return out


if __name__ == "__main__":
    main()
