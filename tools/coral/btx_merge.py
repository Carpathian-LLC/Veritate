"""BTX-style merge of two Veritate constituents.

Branch-Train-MiX (Sukhbaatar et al., COLM 2024). For each block: keep both
parents' FFNs as MoE experts, average the attention + norms. Embeddings, output
norm, and the LM head are averaged. A small per-block top-2 soft router learns
which expert(s) each token should use.

Why this should beat the splice: averaging conflicting weights destroys signal
(pre-refine val 3.31 in the splice run). Averaging the OUTPUTS of two
independently-computed expert paths preserves each expert's signal — the router
gets to combine them per-token instead of per-matrix.

Two phases:
  phase 1: router-only — freeze everything except the router; learn routing.
  phase 2: joint refine — unfreeze the experts (+ shared) for a short tune.

The trained model loads as a BtxVeritate (custom class) — it is NOT a vanilla
Veritate checkpoint. Use the eval path at the end of this script (or
btx_eval.py) to score it.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "veritate_mri"))

from veritate_core.model import Veritate, Block, FFN, CausalSelfAttention, RMSNorm
from training import save
from readers import paths


# ----------------------------------------------------------------------------
# Args

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name_a", required=True)
    p.add_argument("--name_b", required=True)
    p.add_argument("--step_a", type=int, default=-1)
    p.add_argument("--step_b", type=int, default=-1)
    p.add_argument("--out_name", required=True)
    p.add_argument("--corpus",   required=True)
    p.add_argument("--router_steps", type=int, default=500,
                   help="router-only training steps")
    p.add_argument("--joint_steps",  type=int, default=1000,
                   help="full-model refine after router warmup")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq",   type=int, default=256)
    p.add_argument("--lr_router", type=float, default=3e-3)
    p.add_argument("--lr_joint",  type=float, default=3e-5)
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--log_every",  type=int, default=25)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=8)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--device",     default="")
    p.add_argument("--description", default="BTX-style merge")
    return p.parse_args()


def pick_device(want=""):
    if want: return torch.device(want)
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Constituent loading

def latest_step(name):
    ckpt_dir = os.path.join(REPO, "models", name, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"no checkpoints dir for {name}: {ckpt_dir}")
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
    ck = torch.load(cp_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    shape = dict(hidden=a["hidden"], layers=a["layers"], ffn=a["ffn"],
                 heads=a["heads"], seq=int(a["seq"]))
    model = Veritate(vocab=256, **shape)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, shape, step, cp_path


def check_same_shape(s_a, s_b):
    for k in ("hidden", "layers", "ffn", "heads", "seq"):
        if int(s_a[k]) != int(s_b[k]):
            raise SystemExit(f"shape mismatch on {k!r}: A={s_a[k]} B={s_b[k]}")


# ----------------------------------------------------------------------------
# BTX block

class MoEFFN(nn.Module):
    """Two FFN experts + a top-2 soft router.

    The router maps the residual stream to a 2-vector of logits, softmax to
    weights, and combines the experts as w0*FFN_a(x) + w1*FFN_b(x). With only
    2 experts top-2 routing IS the dense path (both experts always active),
    but the per-token weights are learned. This is the BTX setup for k=2.
    """
    def __init__(self, ffn_a: FFN, ffn_b: FFN, hidden: int):
        super().__init__()
        self.experts = nn.ModuleList([ffn_a, ffn_b])
        self.router  = nn.Linear(hidden, 2, bias=False)
        nn.init.zeros_(self.router.weight)   # uniform routing at init

    def forward(self, x):
        logits  = self.router(x)                  # [B, T, 2]
        weights = F.softmax(logits, dim=-1)       # [B, T, 2]
        out_a   = self.experts[0](x)              # [B, T, H]
        out_b   = self.experts[1](x)              # [B, T, H]
        return weights[..., 0:1] * out_a + weights[..., 1:2] * out_b


def build_btx_model(model_a, model_b, shape):
    """Construct a Veritate skeleton, average the shared weights from A and B,
    and graft an MoEFFN onto each block holding A's and B's FFN as experts.
    """
    btx = Veritate(vocab=256, **shape)
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    # Average embeddings (lm_head is tied to tok_emb, so just tok_emb).
    btx.tok_emb.weight.data = 0.5 * (sd_a["tok_emb.weight"] + sd_b["tok_emb.weight"])
    btx.pos_emb.weight.data = 0.5 * (sd_a["pos_emb.weight"] + sd_b["pos_emb.weight"])
    btx.n_out.weight.data   = 0.5 * (sd_a["n_out.weight"]   + sd_b["n_out.weight"])
    # lm_head is tied — re-tie after averaging
    btx.lm_head.weight = btx.tok_emb.weight

    # Per-block: average attention + norms; swap ff for MoE
    for li, blk in enumerate(btx.blocks):
        blk.n1.weight.data        = 0.5 * (sd_a[f"blocks.{li}.n1.weight"]
                                           + sd_b[f"blocks.{li}.n1.weight"])
        blk.n2.weight.data        = 0.5 * (sd_a[f"blocks.{li}.n2.weight"]
                                           + sd_b[f"blocks.{li}.n2.weight"])
        blk.attn.qkv.weight.data  = 0.5 * (sd_a[f"blocks.{li}.attn.qkv.weight"]
                                           + sd_b[f"blocks.{li}.attn.qkv.weight"])
        blk.attn.proj.weight.data = 0.5 * (sd_a[f"blocks.{li}.attn.proj.weight"]
                                           + sd_b[f"blocks.{li}.attn.proj.weight"])

        # Build expert FFNs by deep-copying A's and B's ffn modules
        ffn_a = FFN(shape["hidden"], shape["ffn"])
        ffn_b = FFN(shape["hidden"], shape["ffn"])
        ffn_a.up.weight.data   = sd_a[f"blocks.{li}.ff.up.weight"].clone()
        ffn_a.down.weight.data = sd_a[f"blocks.{li}.ff.down.weight"].clone()
        ffn_b.up.weight.data   = sd_b[f"blocks.{li}.ff.up.weight"].clone()
        ffn_b.down.weight.data = sd_b[f"blocks.{li}.ff.down.weight"].clone()

        blk.ff = MoEFFN(ffn_a, ffn_b, shape["hidden"])

    return btx


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
    return draw


# ----------------------------------------------------------------------------
# Training

def lr_at(step, total, warmup, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def eval_btx(btx, val_draw, iters, device):
    btx.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            x, y = val_draw()
            x, y = x.to(device), y.to(device)
            _, loss = btx(x, targets=y)
            losses.append(float(loss))
    btx.train()
    return float(np.mean(losses))


def router_params(btx):
    """All MoEFFN router weights (the only thing trained in phase 1)."""
    out = []
    for blk in btx.blocks:
        if isinstance(blk.ff, MoEFFN):
            out.extend(blk.ff.router.parameters())
    return out


def all_trainable_params(btx):
    return [p for p in btx.parameters() if p.requires_grad]


def freeze_all_but_routers(btx):
    for p in btx.parameters():
        p.requires_grad = False
    for blk in btx.blocks:
        if isinstance(blk.ff, MoEFFN):
            for p in blk.ff.router.parameters():
                p.requires_grad = True


def unfreeze_all(btx):
    for p in btx.parameters():
        p.requires_grad = True


def train_phase(btx, train_draw, val_draw, params, lr_max, lr_min,
                steps, warmup, log_every, eval_every, eval_iters,
                device, name, phase_tag, t0, out_name, seed):
    if steps <= 0: return
    opt = torch.optim.AdamW(params, lr=lr_max, betas=(0.9, 0.95), weight_decay=0.01)
    btx.train()
    last_time = time.time()
    for step in range(1, steps + 1):
        lr = lr_at(step, steps, warmup, lr_max, lr_min)
        for g in opt.param_groups: g["lr"] = lr
        x, y = train_draw()
        x, y = x.to(device), y.to(device)
        _, loss = btx(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()

        if step % log_every == 0 or step == 1:
            now = time.time()
            tokens = log_every * x.size(0) * x.size(1)
            tps = tokens / max(1e-6, now - last_time)
            save.append_train_row(out_name, step, "train",
                                  float(loss.item()), float(lr),
                                  float(gn), float(tps), float(now - t0), int(seed))
            print(f"[btx.{phase_tag}] step {step}/{steps}  loss={loss.item():.4f}  "
                  f"lr={lr:.2e}  gn={gn:.3f}  tok/s={tps:.0f}", flush=True)
            last_time = now

        if eval_every and step % eval_every == 0 and val_draw is not None:
            v = eval_btx(btx, val_draw, eval_iters, device)
            save.append_train_row(out_name, step, "val", float(v), float(lr),
                                  0.0, 0.0, float(time.time() - t0), int(seed))
            print(f"[btx.{phase_tag}] step {step}  VAL loss={v:.4f}", flush=True)


# ----------------------------------------------------------------------------
# Save (custom — model has MoEFFN modules, not vanilla FFN)

def save_btx(btx, out_dir, args, shape, step, parents):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = btx.state_dict()
    payload = {
        "model": state,
        "step":  step,
        "args":  {"name": args.out_name,
                  "parents": parents,
                  "router_steps": args.router_steps,
                  "joint_steps":  args.joint_steps,
                  "layers": shape["layers"], "hidden": shape["hidden"],
                  "ffn":    shape["ffn"],    "heads":  shape["heads"],
                  "seq":    int(shape["seq"]), "vocab":  256},
        "shape": shape,
        "btx":   True,
    }
    final = os.path.join(ckpt_dir, f"step_{step}.pt")
    tmp = final + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    return final


def write_btx_config(out_dir, args, shape, n_params):
    cfg = {
        "name": args.out_name,
        "description": args.description,
        "kind": "trainer",
        "plugin": "btx_merge",
        "vocab": 256,
        "shape": {"vocab": 256, **{k: int(shape[k]) for k in ("hidden","layers","ffn","heads","seq")}},
        "training_args": vars(args),
        "n_params": int(n_params),
        "btx":      True,
        "wrote_at": int(time.time()),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ----------------------------------------------------------------------------
# Main

def main():
    args = parse_args()
    device = pick_device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"[btx.merge] loading A={args.name_a}", flush=True)
    model_a, shape_a, step_a, cp_a = load_constituent(args.name_a, args.step_a)
    print(f"[btx.merge] loaded A step={step_a}", flush=True)

    print(f"[btx.merge] loading B={args.name_b}", flush=True)
    model_b, shape_b, step_b, cp_b = load_constituent(args.name_b, args.step_b)
    print(f"[btx.merge] loaded B step={step_b}", flush=True)

    check_same_shape(shape_a, shape_b)
    shape = shape_a

    train_path = paths.corpus_train_path(args.corpus)
    val_path   = paths.corpus_val_path(args.corpus)
    if not os.path.isfile(train_path):
        raise SystemExit(f"no train corpus: {train_path}")
    has_val = os.path.isfile(val_path)
    train_draw = make_loader(train_path, args.batch, args.seq, seed=args.seed)
    val_draw   = make_loader(val_path,   args.batch, args.seq, seed=args.seed + 1) if has_val else None

    print(f"[btx.merge] building BTX skeleton (avg attn/norm/emb, MoE FFN)", flush=True)
    btx = build_btx_model(model_a, model_b, shape).to(device)
    n_total  = sum(p.numel() for p in btx.parameters())
    n_router = sum(p.numel() for blk in btx.blocks for p in blk.ff.router.parameters())
    print(f"[btx.merge] params total={n_total:,}  router={n_router:,}", flush=True)

    out_dir = os.path.join(REPO, "models", args.out_name)
    write_btx_config(out_dir, args, shape, n_total)

    # Initial eval — averaged shared + MoE init (uniform routing)
    if val_draw is not None:
        v0 = eval_btx(btx, val_draw, args.eval_iters, device)
        print(f"[btx.merge] initial VAL loss (uniform router, no train) = {v0:.4f}", flush=True)
        save.append_train_row(args.out_name, 0, "val", float(v0), 0.0, 0.0, 0.0, 0, float(0))

    t0 = time.time()

    # Phase 1 — router only
    print(f"[btx.merge] phase 1: router-only {args.router_steps} steps", flush=True)
    freeze_all_but_routers(btx)
    r_params = router_params(btx)
    train_phase(btx, train_draw, val_draw, r_params,
                args.lr_router, args.lr_router * 0.1,
                args.router_steps, max(1, int(args.router_steps * args.warmup_frac)),
                args.log_every, args.eval_every, args.eval_iters,
                device, args.out_name, "p1.router", t0, args.out_name, args.seed)

    # Phase 2 — joint refine
    print(f"[btx.merge] phase 2: joint refine {args.joint_steps} steps", flush=True)
    unfreeze_all(btx)
    j_params = all_trainable_params(btx)
    train_phase(btx, train_draw, val_draw, j_params,
                args.lr_joint, args.lr_joint * 0.1,
                args.joint_steps, max(1, int(args.joint_steps * args.warmup_frac)),
                args.log_every, args.eval_every, args.eval_iters,
                device, args.out_name, "p2.joint", t0, args.out_name, args.seed)

    final_step = args.router_steps + args.joint_steps
    if val_draw is not None:
        vf = eval_btx(btx, val_draw, max(16, args.eval_iters), device)
        print(f"[btx.merge] final VAL loss = {vf:.4f}", flush=True)
        save.append_train_row(args.out_name, final_step, "val", float(vf), 0.0,
                              0.0, 0.0, float(time.time() - t0), int(args.seed))

    final_path = save_btx(btx, out_dir, args, shape, final_step,
                          parents=[args.name_a, args.name_b])
    print(f"[btx.merge] DONE in {time.time() - t0:.1f}s  ckpt={final_path}", flush=True)


if __name__ == "__main__":
    main()
