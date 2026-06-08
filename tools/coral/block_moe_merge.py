"""Full-block MoE merge — each transformer block is one of: A's whole block or
B's whole block, picked per-token by a learned router.

Contrast with btx_merge.py: that does FFN-only MoE and AVERAGES attention,
which is destructive for from-scratch parents that don't share a basis. Here
we never average — we route at the block level so each expert sees its own
basis at every step. Embeddings and the final LM head are still averaged (no
clean alternative without doubling vocab parameters).

Modes:
  - dense (top-2): out = w0 * A_block(x) + w1 * B_block(x). Both experts run.
  - sparse (top-1): pick whichever expert the router prefers per token.

Sparse adds a load-balance loss so the router doesn't collapse to a single
expert. Dense doesn't need it (always uses both).

Two phases (same as btx_merge):
  phase 1: router-only — freeze experts; learn routing.
  phase 2: joint refine — unfreeze for short tune.
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
    p.add_argument("--mode", choices=["dense", "sparse"], default="dense",
                   help="dense=top-2 (both experts run); sparse=top-1 (one expert)")
    p.add_argument("--router_steps", type=int, default=500)
    p.add_argument("--joint_steps",  type=int, default=2500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq",   type=int, default=256)
    p.add_argument("--lr_router", type=float, default=3e-3)
    p.add_argument("--lr_joint",  type=float, default=1e-4)
    p.add_argument("--lambda_lb", type=float, default=0.01,
                   help="load-balance coefficient (sparse only)")
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--log_every",  type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=8)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--device",     default="")
    p.add_argument("--description", default="Block-MoE merge")
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
    cp = os.path.join(REPO, "models", name, "checkpoints", f"step_{step}.pt")
    ck = torch.load(cp, map_location="cpu", weights_only=False)
    a = ck["args"]
    shape = dict(hidden=a["hidden"], layers=a["layers"], ffn=a["ffn"],
                 heads=a["heads"], seq=int(a["seq"]))
    model = Veritate(vocab=256, **shape)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, shape, step, cp


def check_same_shape(s_a, s_b):
    for k in ("hidden", "layers", "ffn", "heads", "seq"):
        if int(s_a[k]) != int(s_b[k]):
            raise SystemExit(f"shape mismatch on {k!r}: A={s_a[k]} B={s_b[k]}")


# ----------------------------------------------------------------------------
# Block-MoE

class MoEBlock(nn.Module):
    """Holds two complete blocks (from A and B) plus a per-token router."""
    def __init__(self, block_a: Block, block_b: Block, hidden: int, mode: str):
        super().__init__()
        self.experts = nn.ModuleList([block_a, block_b])
        self.router  = nn.Linear(hidden, 2, bias=False)
        nn.init.zeros_(self.router.weight)   # uniform routing at init
        self.mode = mode
        self.last_router_probs = None   # for load-balance loss

    def forward(self, x):
        # Mean-pool over token sequence dim for routing decisions? Or per-token?
        # Per-token is most flexible. Router operates on the residual stream.
        logits = self.router(x)                  # [B, T, 2]
        probs  = F.softmax(logits, dim=-1)
        self.last_router_probs = probs

        if self.mode == "dense":
            out_a = self.experts[0](x)
            out_b = self.experts[1](x)
            return probs[..., 0:1] * out_a + probs[..., 1:2] * out_b

        # sparse: pick top-1 per token; masked write back.
        # Both experts still run on the full input (we can't dynamically slice
        # by mask in vectorized form easily), but the OUTPUT only uses one
        # expert per token. Active forward compute is 2x (we pay for both), but
        # gradient flows only to the selected one. This is the standard
        # straight-through MoE pattern when bsz is small.
        choice = probs.argmax(dim=-1, keepdim=True)         # [B, T, 1] in {0,1}
        mask_a = (choice == 0).to(x.dtype)                  # [B, T, 1]
        mask_b = 1.0 - mask_a
        out_a = self.experts[0](x)
        out_b = self.experts[1](x)
        # Multiply by router prob so gradient flows back to router; the chosen
        # expert dominates because the other's mask is 0.
        return mask_a * probs[..., 0:1] * out_a + mask_b * probs[..., 1:2] * out_b


def build_block_moe(model_a, model_b, shape, mode):
    """Construct a Veritate skeleton, then replace each block with a MoEBlock
    that holds A's and B's whole block as experts. Embeddings and lm_head are
    averaged (no clean alternative). Output norm averaged.
    """
    moe = Veritate(vocab=256, **shape)
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()

    # Average embeddings (lm_head tied to tok_emb), n_out
    moe.tok_emb.weight.data = 0.5 * (sd_a["tok_emb.weight"] + sd_b["tok_emb.weight"])
    moe.pos_emb.weight.data = 0.5 * (sd_a["pos_emb.weight"] + sd_b["pos_emb.weight"])
    moe.n_out.weight.data   = 0.5 * (sd_a["n_out.weight"]   + sd_b["n_out.weight"])
    moe.lm_head.weight      = moe.tok_emb.weight

    # Replace each block with a MoEBlock containing A's and B's full block.
    new_blocks = nn.ModuleList()
    for li in range(shape["layers"]):
        # Build two fresh Block instances and load A/B weights into them.
        blk_a = Block(shape["hidden"], shape["ffn"], shape["heads"])
        blk_b = Block(shape["hidden"], shape["ffn"], shape["heads"])
        for src_sd, dst_blk in [(sd_a, blk_a), (sd_b, blk_b)]:
            dst_blk.n1.weight.data        = src_sd[f"blocks.{li}.n1.weight"].clone()
            dst_blk.n2.weight.data        = src_sd[f"blocks.{li}.n2.weight"].clone()
            dst_blk.attn.qkv.weight.data  = src_sd[f"blocks.{li}.attn.qkv.weight"].clone()
            dst_blk.attn.proj.weight.data = src_sd[f"blocks.{li}.attn.proj.weight"].clone()
            dst_blk.ff.up.weight.data     = src_sd[f"blocks.{li}.ff.up.weight"].clone()
            dst_blk.ff.down.weight.data   = src_sd[f"blocks.{li}.ff.down.weight"].clone()
        new_blocks.append(MoEBlock(blk_a, blk_b, shape["hidden"], mode))

    moe.blocks = new_blocks
    return moe


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


def load_balance_loss(moe_model):
    """Switch-style: encourage uniform expert utilization. Per layer:
    fraction_chosen[i] * mean_prob[i] summed, scaled by num_experts.
    Lower is more uniform.
    """
    n = 2
    losses = []
    for blk in moe_model.blocks:
        if not isinstance(blk, MoEBlock): continue
        probs = blk.last_router_probs           # [B, T, 2]
        if probs is None: continue
        mean_prob = probs.mean(dim=(0, 1))      # [2]
        choice    = probs.argmax(dim=-1)        # [B, T]
        fraction  = torch.zeros(n, device=probs.device)
        for i in range(n):
            fraction[i] = (choice == i).float().mean()
        losses.append(n * (fraction * mean_prob).sum())
    return sum(losses) / max(1, len(losses))


def eval_moe(model, val_draw, iters, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            x, y = val_draw()
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            losses.append(float(loss))
    model.train()
    return float(np.mean(losses))


def router_params(model):
    out = []
    for blk in model.blocks:
        if isinstance(blk, MoEBlock):
            out.extend(blk.router.parameters())
    return out


def freeze_all_but_routers(model):
    for p in model.parameters(): p.requires_grad = False
    for blk in model.blocks:
        if isinstance(blk, MoEBlock):
            for p in blk.router.parameters(): p.requires_grad = True


def unfreeze_all(model):
    for p in model.parameters(): p.requires_grad = True


def train_phase(model, train_draw, val_draw, params, lr_max, lr_min,
                steps, warmup, log_every, eval_every, eval_iters,
                device, args, t0, phase_tag, sparse_lb):
    if steps <= 0: return
    opt = torch.optim.AdamW(params, lr=lr_max, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    last_t = time.time()
    for step in range(1, steps + 1):
        lr = lr_at(step, steps, warmup, lr_max, lr_min)
        for g in opt.param_groups: g["lr"] = lr
        x, y = train_draw()
        x, y = x.to(device), y.to(device)
        _, ce = model(x, targets=y)
        loss = ce
        if sparse_lb > 0:
            lb = load_balance_loss(model)
            loss = ce + sparse_lb * lb

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()

        if step % log_every == 0 or step == 1:
            now = time.time()
            tokens = log_every * x.size(0) * x.size(1)
            tps = tokens / max(1e-6, now - last_t)
            save.append_train_row(args.out_name, step, "train",
                                  float(loss.item()), float(lr),
                                  float(gn), float(tps), float(now - t0), int(args.seed))
            extra = f"  lb={lb.item():.3f}" if sparse_lb > 0 else ""
            print(f"[bmoe.{phase_tag}] step {step}/{steps}  loss={loss.item():.4f}  "
                  f"ce={ce.item():.4f}{extra}  lr={lr:.2e}  gn={gn:.3f}  tok/s={tps:.0f}",
                  flush=True)
            last_t = now

        if eval_every and step % eval_every == 0 and val_draw is not None:
            v = eval_moe(model, val_draw, eval_iters, device)
            save.append_train_row(args.out_name, step, "val", float(v), float(lr),
                                  0.0, 0.0, float(time.time() - t0), int(args.seed))
            print(f"[bmoe.{phase_tag}] step {step}  VAL loss={v:.4f}", flush=True)


# ----------------------------------------------------------------------------
# Save

def save_moe(model, out_dir, args, shape, step, parents):
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "step":  step,
        "args":  {"name": args.out_name,
                  "parents": parents,
                  "mode": args.mode,
                  "router_steps": args.router_steps,
                  "joint_steps":  args.joint_steps,
                  "layers": shape["layers"], "hidden": shape["hidden"],
                  "ffn":    shape["ffn"],    "heads":  shape["heads"],
                  "seq":    int(shape["seq"]), "vocab":  256},
        "shape": shape,
        "block_moe": True,
    }
    final = os.path.join(ckpt_dir, f"step_{step}.pt")
    tmp = final + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, final)
    return final


def write_block_moe_config(out_dir, args, shape, n_params):
    cfg = {
        "name": args.out_name,
        "description": args.description,
        "kind": "trainer",
        "plugin": "block_moe_merge",
        "vocab": 256,
        "shape": {"vocab": 256, **{k: int(shape[k]) for k in ("hidden","layers","ffn","heads","seq")}},
        "training_args": vars(args),
        "n_params": int(n_params),
        "block_moe": True,
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

    print(f"[bmoe] loading A={args.name_a}", flush=True)
    model_a, shape_a, step_a, cp_a = load_constituent(args.name_a, args.step_a)
    print(f"[bmoe] loaded A step={step_a}", flush=True)

    print(f"[bmoe] loading B={args.name_b}", flush=True)
    model_b, shape_b, step_b, cp_b = load_constituent(args.name_b, args.step_b)
    print(f"[bmoe] loaded B step={step_b}", flush=True)

    check_same_shape(shape_a, shape_b)
    shape = shape_a

    train_path = paths.corpus_train_path(args.corpus)
    val_path   = paths.corpus_val_path(args.corpus)
    if not os.path.isfile(train_path):
        raise SystemExit(f"no train corpus: {train_path}")
    has_val = os.path.isfile(val_path)
    train_draw = make_loader(train_path, args.batch, args.seq, seed=args.seed)
    val_draw   = make_loader(val_path,   args.batch, args.seq, seed=args.seed + 1) if has_val else None

    print(f"[bmoe] building block-MoE skeleton (mode={args.mode})", flush=True)
    model = build_block_moe(model_a, model_b, shape, mode=args.mode).to(device)
    n_total  = sum(p.numel() for p in model.parameters())
    n_router = sum(p.numel() for blk in model.blocks for p in blk.router.parameters())
    print(f"[bmoe] params total={n_total:,}  router={n_router:,}", flush=True)

    out_dir = os.path.join(REPO, "models", args.out_name)
    write_block_moe_config(out_dir, args, shape, n_total)

    if val_draw is not None:
        v0 = eval_moe(model, val_draw, args.eval_iters, device)
        print(f"[bmoe] initial VAL loss (uniform router) = {v0:.4f}", flush=True)
        save.append_train_row(args.out_name, 0, "val", float(v0), 0.0, 0.0, 0.0, 0.0, 0)

    t0 = time.time()
    sparse_lb = args.lambda_lb if args.mode == "sparse" else 0.0

    print(f"[bmoe] phase 1: router-only {args.router_steps} steps", flush=True)
    freeze_all_but_routers(model)
    r_params = router_params(model)
    train_phase(model, train_draw, val_draw, r_params,
                args.lr_router, args.lr_router * 0.1,
                args.router_steps, max(1, int(args.router_steps * args.warmup_frac)),
                args.log_every, args.eval_every, args.eval_iters,
                device, args, t0, "p1.router", sparse_lb)

    print(f"[bmoe] phase 2: joint refine {args.joint_steps} steps", flush=True)
    unfreeze_all(model)
    j_params = [p for p in model.parameters() if p.requires_grad]
    train_phase(model, train_draw, val_draw, j_params,
                args.lr_joint, args.lr_joint * 0.1,
                args.joint_steps, max(1, int(args.joint_steps * args.warmup_frac)),
                args.log_every, args.eval_every, args.eval_iters,
                device, args, t0, "p2.joint", sparse_lb)

    final_step = args.router_steps + args.joint_steps
    if val_draw is not None:
        vf = eval_moe(model, val_draw, max(16, args.eval_iters), device)
        print(f"[bmoe] final VAL loss = {vf:.4f}", flush=True)
        save.append_train_row(args.out_name, final_step, "val", float(vf), 0.0,
                              0.0, 0.0, float(time.time() - t0), int(args.seed))

    final_path = save_moe(model, out_dir, args, shape, final_step,
                          parents=[args.name_a, args.name_b])
    print(f"[bmoe] DONE in {time.time() - t0:.1f}s  ckpt={final_path}", flush=True)


if __name__ == "__main__":
    main()
