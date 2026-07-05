#!/usr/bin/env python3
# E7 test-time R sweep: same trained looped checkpoint, eval_loops in {1,2,4,8},
# identical val batches for every arm (paired). Dense e1muon on the same batches
# as anchor. Claim under test: val improves with R past the training mean (2.5).
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from veritate_core.model_patched import VeritatePatched
from veritate_core.model import Veritate

VAL_BIN = os.path.join(ROOT, "trainers", "corpus", "fineweb_edu_val.bin")
E7_CKPT = os.path.join(ROOT, "models", "e7looped_10m_qat", "checkpoints", "step_12000.pt")
E1_CKPT = os.path.join(ROOT, "models", "e1muon_10m_qat", "checkpoints", "step_12000.pt")
SEED = 0
N_BATCHES = 64
BS = 32
SEQ = 512
SWEEP = [1, 2, 3, 4, 6, 8]


def draw_batches(device):
    arr = np.memmap(VAL_BIN, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(arr) - SEQ - 1, size=N_BATCHES * BS)
    batches = []
    for i in range(N_BATCHES):
        s = starts[i * BS:(i + 1) * BS]
        win = np.stack([arr[j:j + SEQ + 1] for j in s]).astype(np.int64)
        t = torch.from_numpy(win).to(device)
        batches.append((t[:, :-1], t[:, 1:]))
    return batches


@torch.no_grad()
def mean_ce(model, batches):
    tot, n = 0.0, 0
    for tokens, targets in batches:
        logits, _ = model(tokens)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        tot += ce.item() * targets.numel()
        n += targets.numel()
    return tot / n


def load(cls, ckpt, device, **kw):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = blob.get("args") or {}
    model = cls(vocab=256, hidden=cfg["hidden"], layers=cfg["layers"],
                ffn=cfg["ffn"], heads=cfg["heads"], seq=cfg["seq"], **kw)
    model.load_state_dict(blob["model"], strict=False)
    return model.to(device).eval()


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    batches = draw_batches(device)
    out = {"seed": SEED, "n_batches": N_BATCHES, "bs": BS, "seq": SEQ, "sweep": {}}

    e7 = load(VeritatePatched, E7_CKPT, device, global_loops=4)
    for r in SWEEP:
        e7.eval_loops = r
        ce = mean_ce(e7, batches)
        out["sweep"][str(r)] = round(ce, 6)
        print(f"R={r}: val CE {ce:.4f}", flush=True)
    del e7

    dense = load(Veritate, E1_CKPT, device)
    out["dense_e1muon"] = round(mean_ce(dense, batches), 6)
    print(f"dense e1muon (same batches): {out['dense_e1muon']:.4f}", flush=True)

    path = os.path.join(ROOT, "SMOKE_RESULTS", "e7_loop_sweep_stats.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote " + path, flush=True)


if __name__ == "__main__":
    main()
