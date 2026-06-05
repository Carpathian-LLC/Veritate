"""Eval a checkpoint's mean CE on a corpus val split.
Used by the Coral diagnostics pass — answers: "what does model X score on corpus Y?"
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "veritate_mri"))

from veritate_core.model import Veritate
from readers import paths


def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():        return torch.device("cuda")
    return torch.device("cpu")


def eval_ckpt(ckpt_path, corpus, iters, batch, seq, seed):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = Veritate(vocab=256, hidden=a["hidden"], layers=a["layers"],
                     ffn=a["ffn"], heads=a["heads"], seq=int(a["seq"]))
    model.load_state_dict(ck["model"], strict=True)
    dev = pick_device()
    model.to(dev).eval()

    val_path = paths.corpus_val_path(corpus)
    if not os.path.isfile(val_path):
        raise SystemExit(f"no val for corpus={corpus}: {val_path}")
    arr = np.memmap(val_path, dtype=np.uint8, mode="r")
    n = len(arr)
    rng = np.random.default_rng(seed)
    losses = []
    with torch.no_grad():
        for _ in range(iters):
            starts = rng.integers(0, n - seq - 1, size=batch, dtype=np.int64)
            x = np.stack([np.asarray(arr[s:s + seq],         dtype=np.int64) for s in starts])
            y = np.stack([np.asarray(arr[s + 1:s + 1 + seq], dtype=np.int64) for s in starts])
            x = torch.from_numpy(x).to(dev)
            y = torch.from_numpy(y).to(dev)
            logits, loss = model(x, targets=y)
            losses.append(float(loss))
    return float(np.mean(losses)), float(np.std(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--iters", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()
    mean, std = eval_ckpt(args.ckpt, args.corpus, args.iters, args.batch, args.seq, args.seed)
    print(f"ckpt={os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))} corpus={args.corpus} "
          f"val_loss={mean:.4f} std={std:.4f} iters={args.iters}")


if __name__ == "__main__":
    main()
