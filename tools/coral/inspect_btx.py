"""Inspect what a trained BTX model learned: per-layer router preferences
and per-corpus routing splits. Tells us whether the router is making
meaningful decisions or just collapsing to a near-uniform mix.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "veritate_mri"))

sys.path.insert(0, HERE)
from btx_merge import build_btx_model, load_constituent  # noqa: E402
from readers import paths


def collect_probs(btx, val_path, batches, batch_size, seq, seed):
    val = np.memmap(val_path, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(seed)
    all_probs = [[] for _ in range(btx.layers)]
    with torch.no_grad():
        for _ in range(batches):
            starts = rng.integers(0, len(val) - seq - 1, size=batch_size)
            x = np.stack([np.asarray(val[s:s + seq], dtype=np.int64) for s in starts])
            x = torch.from_numpy(x)
            h = btx.embed(x)
            for li, blk in enumerate(btx.blocks):
                h = h + blk.attn(blk.n1(h))
                ff_in = blk.n2(h)
                probs = F.softmax(blk.ff.router(ff_in), dim=-1)
                all_probs[li].append(probs)
                h = h + blk.ff(ff_in)
    return [torch.cat(p, dim=0) for p in all_probs]


def fmt_row(li, probs, label):
    pa = probs[..., 0].flatten()
    pref_a = (probs.argmax(-1) == 0).float().mean()
    return (f"L{li:>2} {label:>14} | mean(A)={pa.mean():.3f}  std={pa.std():.3f}  "
            f"prefer_A_tokens={pref_a:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]
    shape = dict(hidden=args["hidden"], layers=args["layers"], ffn=args["ffn"],
                 heads=args["heads"], seq=int(args["seq"]))
    print(f"BTX ckpt: {a.ckpt}")
    print(f"parents : {args['parents']}")
    print(f"shape   : {shape}")

    ma, _, _, _ = load_constituent(args["parents"][0], -1)
    mb, _, _, _ = load_constituent(args["parents"][1], -1)
    btx = build_btx_model(ma, mb, shape)
    btx.load_state_dict(ck["model"], strict=True)
    btx.eval()

    for corpus, label in [("distill_v1_mix_tinystories", "MIX"),
                          ("tinystories",                "A_native"),
                          ("distill_v1",                 "B_native")]:
        val_path = paths.corpus_val_path(corpus)
        if not os.path.isfile(val_path):
            print(f"(skip — no val for {corpus})")
            continue
        probs = collect_probs(btx, val_path, batches=4, batch_size=16,
                              seq=shape["seq"], seed=0)
        print(f"\nCorpus: {corpus}")
        for li, P in enumerate(probs):
            print(fmt_row(li, P, label))


if __name__ == "__main__":
    main()
