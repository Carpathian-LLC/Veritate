"""Pull val_loss + param counts for every Coral experiment into one table.
Reads each model's latest checkpoint, evals on all three corpora, prints
a markdown-style scoreboard.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "veritate_mri"))
sys.path.insert(0, HERE)

from veritate_core.model import Veritate
from readers import paths
from btx_merge       import build_btx_model, load_constituent as load_btx_parent
from block_moe_merge import build_block_moe


def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():        return torch.device("cuda")
    return torch.device("cpu")


def latest_ckpt(name):
    d = os.path.join(REPO, "models", name, "checkpoints")
    if not os.path.isdir(d): return None, None
    steps = []
    for fn in os.listdir(d):
        if fn.startswith("step_") and fn.endswith(".pt"):
            try: steps.append(int(fn[5:-3]))
            except ValueError: pass
    if not steps: return None, None
    s = max(steps)
    return os.path.join(d, f"step_{s}.pt"), s


def build_for(name, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    shape = dict(hidden=a["hidden"], layers=a["layers"], ffn=a["ffn"],
                 heads=a["heads"], seq=int(a["seq"]))
    if ck.get("btx"):
        parents = a["parents"]
        ma, _, _, _ = load_btx_parent(parents[0], -1)
        mb, _, _, _ = load_btx_parent(parents[1], -1)
        model = build_btx_model(ma, mb, shape)
    elif ck.get("block_moe"):
        parents = a["parents"]
        ma, _, _, _ = load_btx_parent(parents[0], -1)
        mb, _, _, _ = load_btx_parent(parents[1], -1)
        model = build_block_moe(ma, mb, shape, mode=a.get("mode", "dense"))
    else:
        model = Veritate(vocab=256, **shape)
    model.load_state_dict(ck["model"], strict=True)
    return model, shape


def eval_on(model, corpus, iters=32, batch=16, seq=256, seed=12345, device=None):
    val_path = paths.corpus_val_path(corpus)
    if not os.path.isfile(val_path): return None
    arr = np.memmap(val_path, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(seed)
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(iters):
            starts = rng.integers(0, len(arr) - seq - 1, size=batch, dtype=np.int64)
            x = np.stack([np.asarray(arr[s:s + seq],         dtype=np.int64) for s in starts])
            y = np.stack([np.asarray(arr[s + 1:s + 1 + seq], dtype=np.int64) for s in starts])
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            _, loss = model(x, targets=y)
            losses.append(float(loss))
    return float(np.mean(losses))


MODELS = [
    "coral_a_tinystories_30m",
    "coral_b_distill_v1_30m",
    "coral_baseline_50m",
    "coral_scratch_30m_mix",
    "coral_blend_30m",
    "coral_blend_30m_kl0",
    "coral_btx_30m",
    "coral_btx_30m_long",
    "coral_blockmoe_30m",
    "coral_blockmoe_30m_sparse",
]
CORPORA = ["distill_v1_mix_tinystories", "tinystories", "distill_v1"]


def main():
    device = pick_device()
    rows = []
    for name in MODELS:
        ckpt, step = latest_ckpt(name)
        if ckpt is None:
            continue
        try:
            model, shape = build_for(name, ckpt)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue
        model = model.to(device)
        n_total = sum(p.numel() for p in model.parameters())
        row = {"name": name, "step": step, "params": n_total}
        for c in CORPORA:
            try:
                row[c] = eval_on(model, c, device=device)
            except Exception as e:
                row[c] = None
        rows.append(row)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # Print table
    headers = ["model", "params", "step", "mix", "tinystories", "distill_v1"]
    widths  = [30, 12, 6, 8, 12, 12]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        vals = [r["name"], f"{r['params']:,}", str(r["step"]),
                f"{r['distill_v1_mix_tinystories']:.4f}" if r.get('distill_v1_mix_tinystories') is not None else "-",
                f"{r['tinystories']:.4f}"                if r.get('tinystories')                is not None else "-",
                f"{r['distill_v1']:.4f}"                 if r.get('distill_v1')                 is not None else "-"]
        print(" | ".join(v.ljust(w) for v, w in zip(vals, widths)))


if __name__ == "__main__":
    main()
