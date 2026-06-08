"""Eval a BTX checkpoint on a corpus's val split.
Mirror of eval_on.py but loads via btx_merge.build_btx_model so the MoEFFN
modules are constructed correctly before load_state_dict.
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

# Bring in MoEFFN + build_btx_model so the skeleton matches the saved keys.
sys.path.insert(0, HERE)
from btx_merge import MoEFFN, build_btx_model, load_constituent  # noqa: E402


def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():        return torch.device("cuda")
    return torch.device("cpu")


def eval_btx(ckpt_path, corpus, iters, batch, seq, seed):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    shape = dict(hidden=a["hidden"], layers=a["layers"], ffn=a["ffn"],
                 heads=a["heads"], seq=int(a["seq"]))
    parents = a.get("parents") or []
    if len(parents) != 2:
        raise SystemExit(f"BTX ckpt missing parents in args: {a}")

    # Build the skeleton by re-loading the two constituents (cheap on CPU).
    model_a, _, _, _ = load_constituent(parents[0], -1)
    model_b, _, _, _ = load_constituent(parents[1], -1)
    btx = build_btx_model(model_a, model_b, shape)
    btx.load_state_dict(ck["model"], strict=True)

    dev = pick_device()
    btx.to(dev).eval()

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
            _, loss = btx(x, targets=y)
            losses.append(float(loss))
    return float(np.mean(losses)), float(np.std(losses))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--iters", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq",   type=int, default=256)
    p.add_argument("--seed",  type=int, default=12345)
    a = p.parse_args()
    mean, std = eval_btx(a.ckpt, a.corpus, a.iters, a.batch, a.seq, a.seed)
    name = os.path.basename(os.path.dirname(os.path.dirname(a.ckpt)))
    print(f"ckpt={name} corpus={a.corpus} val_loss={mean:.4f} std={std:.4f} iters={a.iters}")


if __name__ == "__main__":
    main()
