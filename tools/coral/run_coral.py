# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# tools/coral/run_coral.py
# ------------------------------------------------------------------------------------
# Vanilla base-model trainer for the Coral Merge experiment. No adapters, no QAT.
# Writes the canonical train.csv (step,split,loss,lr,grad_norm,tok_per_s,wall_s,seed)
# and saves checkpoints at models/<name>/checkpoints/step_<N>.pt — same shape the
# Coral Lab dashboard polls. Entire experiment is removable by deleting tools/coral/.
#
# Usage:
#   python tools/coral/run_coral.py \
#     --name coral_a_tinystories_30m \
#     --corpus tinystories \
#     --size 30m \
#     --total_steps 6000
#
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

from veritate_core.model import Veritate                              # noqa: E402
from training import save                                             # noqa: E402
from readers import paths                                             # noqa: E402

SIZE_PRESETS = {
    "30m":  {"layers": 10, "hidden": 512, "ffn": 2048, "heads": 8},
    "50m":  {"layers": 10, "hidden": 640, "ffn": 2560, "heads": 10},
}


def parse_args():
    ap = argparse.ArgumentParser(description="Coral vanilla trainer")
    ap.add_argument("--name",        required=True, help="model name; output goes to models/<name>/")
    ap.add_argument("--corpus",      required=True, help="corpus stem (resolves to plugins/corpus/<stem>_{train,val}.bin)")
    ap.add_argument("--size",        default="30m", choices=list(SIZE_PRESETS.keys()))
    ap.add_argument("--total_steps", type=int,   default=6000)
    ap.add_argument("--batch",       type=int,   default=16)
    ap.add_argument("--seq",         type=int,   default=256)
    ap.add_argument("--base_lr",     type=float, default=3e-4)
    ap.add_argument("--min_lr",      type=float, default=3e-5)
    ap.add_argument("--warmup",      type=int,   default=200)
    ap.add_argument("--weight_decay",type=float, default=0.01)
    ap.add_argument("--grad_clip",   type=float, default=1.0)
    ap.add_argument("--ckpt_every",  type=int,   default=500)
    ap.add_argument("--log_every",   type=int,   default=25)
    ap.add_argument("--eval_every",  type=int,   default=250)
    ap.add_argument("--eval_iters",  type=int,   default=8)
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--device",      default="",  help='"cpu" | "mps" | "cuda" | "" = auto')
    ap.add_argument("--description", default="coral constituent")
    return ap.parse_args()


def pick_device(pref: str) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step, total, warmup, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    p = min(max(p, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))


def make_loader(bin_path, batch, seq, seed):
    arr = np.memmap(bin_path, dtype=np.uint8, mode="r")
    n = len(arr)
    if n < seq + 1:
        raise SystemExit(f"corpus too small: {bin_path} ({n} bytes, need >= {seq + 1})")
    rng = np.random.default_rng(seed)

    def draw():
        starts = rng.integers(0, n - seq - 1, size=batch, dtype=np.int64)
        x = np.stack([np.asarray(arr[s:s + seq],     dtype=np.int64) for s in starts])
        y = np.stack([np.asarray(arr[s + 1:s + 1 + seq], dtype=np.int64) for s in starts])
        return torch.from_numpy(x), torch.from_numpy(y)

    return draw, n


def evaluate(model, val_draw, n_iters, device):
    if val_draw is None:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_iters):
            x, y = val_draw()
            x, y = x.to(device), y.to(device)
            _logits, loss = model(x, targets=y)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def write_config(out_dir, args, shape, n_params, train_path, val_path):
    cfg = {
        "name": args.name,
        "description": args.description,
        "kind": "trainer",
        "plugin": "coral_run",
        "vocab": 256,
        "shape": {
            "vocab": 256,
            "hidden": shape["hidden"],
            "layers": shape["layers"],
            "ffn":    shape["ffn"],
            "heads":  shape["heads"],
            "seq":    int(args.seq),
        },
        "training_args": {
            "name":         args.name,
            "corpus":       args.corpus,
            "corpus_train": train_path,
            "corpus_val":   val_path,
            "size":         args.size,
            "steps":        int(args.total_steps),
            "batch":        int(args.batch),
            "seq":          int(args.seq),
            "lr":           float(args.base_lr),
            "min_lr":       float(args.min_lr),
            "warmup":       int(args.warmup),
            "wd":           float(args.weight_decay),
            "clip":         float(args.grad_clip),
            "ckpt_every":   int(args.ckpt_every),
            "log_every":    int(args.log_every),
            "eval_every":   int(args.eval_every),
            "eval_iters":   int(args.eval_iters),
            "seed":         int(args.seed),
            "device":       args.device or None,
        },
        "training": "vanilla_fp32",
        "n_params_total": int(n_params),
        "wrote_at": int(time.time()),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def main():
    args = parse_args()
    if args.name in {"", None}:
        raise SystemExit("--name required")

    out_dir  = paths.model_dir(args.name) if hasattr(paths, "model_dir") else os.path.join(REPO, "models", args.name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_path = paths.corpus_train_path(args.corpus)
    val_path   = paths.corpus_val_path(args.corpus)
    if not os.path.isfile(train_path):
        raise SystemExit(f"no train corpus: {train_path}")
    has_val = os.path.isfile(val_path)

    device = pick_device(args.device)
    print(f"[coral] device={device} size={args.size} name={args.name} corpus={args.corpus}", flush=True)

    shape = SIZE_PRESETS[args.size]
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    model = Veritate(vocab=256, hidden=shape["hidden"], layers=shape["layers"],
                     ffn=shape["ffn"], heads=shape["heads"], seq=int(args.seq)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[coral] n_params={n_params:,}", flush=True)

    write_config(out_dir, args, shape, n_params, train_path, val_path)

    train_draw, n_train = make_loader(train_path, args.batch, args.seq, seed=args.seed)
    val_draw            = make_loader(val_path,   args.batch, args.seq, seed=args.seed + 1)[0] if has_val else None
    print(f"[coral] train bytes={n_train:,}  val={'yes' if has_val else 'no'}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.base_lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)

    t0 = time.time()
    last_log = t0
    last_step_time = t0

    for step in range(1, args.total_steps + 1):
        lr = lr_at(step, args.total_steps, args.warmup, args.base_lr, args.min_lr)
        for g in opt.param_groups: g["lr"] = lr

        x, y = train_draw()
        x, y = x.to(device), y.to(device)
        _logits, loss = model(x, targets=y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        opt.step()

        now = time.time()
        if step % args.log_every == 0 or step == 1:
            dt = now - last_step_time
            tokens = args.log_every * args.batch * args.seq
            tps = tokens / max(1e-6, dt)
            save.append_train_row(args.name, step, "train",
                                  float(loss.item()), float(lr),
                                  float(gnorm), float(tps), float(now - t0),
                                  int(args.seed))
            print(f"[coral] step {step}/{args.total_steps}  loss={loss.item():.4f}  lr={lr:.2e}  "
                  f"gn={gnorm:.3f}  tok/s={tps:.0f}  wall={now - t0:.1f}s", flush=True)
            last_step_time = now

        if args.eval_every and step % args.eval_every == 0 and val_draw is not None:
            vloss = evaluate(model, val_draw, args.eval_iters, device)
            if vloss is not None:
                save.append_train_row(args.name, step, "val",
                                      float(vloss), float(lr),
                                      0.0, 0.0, float(time.time() - t0),
                                      int(args.seed))
                print(f"[coral] step {step}  VAL loss={vloss:.4f}", flush=True)

        if args.ckpt_every and step % args.ckpt_every == 0:
            ckpt = {
                "model":  model.state_dict(),
                "step":   step,
                "args":   {**vars(args),
                           "layers": shape["layers"], "hidden": shape["hidden"],
                           "ffn":    shape["ffn"],    "heads":  shape["heads"],
                           "seq":    int(args.seq),   "vocab":  256},
                "shape":  shape,
            }
            cp = os.path.join(ckpt_dir, f"step_{step}.pt")
            tmp = cp + ".tmp"
            torch.save(ckpt, tmp)
            os.replace(tmp, cp)
            print(f"[coral] saved checkpoint {cp}", flush=True)

    # Always emit a final checkpoint.
    final = args.total_steps
    final_path = os.path.join(ckpt_dir, f"step_{final}.pt")
    if not os.path.isfile(final_path):
        torch.save({"model": model.state_dict(), "step": final,
                    "args": {**vars(args),
                             "layers": shape["layers"], "hidden": shape["hidden"],
                             "ffn":    shape["ffn"],    "heads":  shape["heads"],
                             "seq":    int(args.seq),   "vocab":  256},
                    "shape": shape}, final_path)
        print(f"[coral] saved final checkpoint {final_path}", flush=True)

    print(f"[coral] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
