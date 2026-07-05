# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - E4 knowledge-injection eval: feed invented facts once through a memory-trunk
#   model (windowed, memory carried), then 8k distractor bytes, then quiz each
#   fact. Falsifier: memory-ON fails to beat memory-OFF on fact-span bpb win
#   rate by >=10 points, or exact-match recall lift < +10 points on a trained
#   model. Untrained models are a plumbing check only (expect ~0 lift).
# - Wall-clock: ~2 min CPU on a 10M model.
# SMOKE_RESULTS/e4_knowledge_injection_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veritate_core.model_memory import VeritateMemory

# ------------------------------------------------------------------------------------
# Constants

N_FACTS        = 12
DISTRACTOR_LEN = 8192
WINDOW         = 512
SEED           = 0
STATS_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "e4_knowledge_injection_stats.json")
DISTRACTOR_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "trainers", "corpus", "fineweb_edu_val.bin")
ENTITIES   = ("zorblat", "kemmen", "vashtor", "quillim", "drenpa", "moxel",
              "tarvin", "yelqua", "brindok", "sulfeth", "gorwim", "plaxen")
ATTRIBUTES = ("fyxel", "morvane", "teppish", "quolt", "zindra", "haxop",
              "welfrim", "ostein", "cruvex", "damlin", "yorvath", "snibbet")

# ------------------------------------------------------------------------------------
# Functions


def facts_and_probes():
    facts, probes = [], []
    for e, a in zip(ENTITIES[:N_FACTS], ATTRIBUTES[:N_FACTS]):
        facts.append(f"The secret name of {e} is {a}. ")
        probes.append((f"The secret name of {e} is ", a))
    return facts, probes


def feed_windows(model, data, device):
    for i in range(0, len(data), WINDOW):
        w = data[i:i + WINDOW]
        x = torch.tensor([list(w)], dtype=torch.long, device=device)
        model.forward_carry(x)
        model.carry_memory()


def span_bpb(model, prompt, gold, device):
    full = prompt + gold
    x = torch.tensor([list(full.encode())], dtype=torch.long, device=device)
    logits, _ = model.forward_carry(x)
    p0 = len(prompt.encode())
    lp = torch.log_softmax(logits[0, p0 - 1:len(full.encode()) - 1].float(), -1)
    gold_ids = torch.tensor(list(gold.encode()), device=device)
    nll = -lp[torch.arange(len(gold_ids)), gold_ids].mean()
    return float(nll / math.log(2))


def exact_match(model, prompt, gold, device):
    ids = list(prompt.encode())
    out = []
    for _ in range(len(gold.encode())):
        x = torch.tensor([ids + out], dtype=torch.long, device=device)
        logits, _ = model.forward_carry(x)
        out.append(int(logits[0, -1].argmax()))
    return bytes(out) == gold.encode()


def run_arm(model, doc, distractor, probes, device, carry):
    model.reset_memory()
    if carry:
        feed_windows(model, doc, device)
        feed_windows(model, distractor, device)
    bpbs, ems = [], []
    for prompt, gold in probes:
        model.carry_memory()
        saved = model.memory.state
        bpbs.append(span_bpb(model, prompt, gold, device))
        model.memory.state = saved
        ems.append(exact_match(model, prompt, gold, device))
        model.memory.state = saved
    return bpbs, ems


def main():
    errors = []
    torch.manual_seed(SEED)
    device = "cpu"
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt:
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = blob.get("args") or blob.get("config") or {}
        model = VeritateMemory(vocab=256, hidden=cfg["hidden"], layers=cfg["layers"],
                               ffn=cfg["ffn"], heads=cfg["heads"], seq=cfg["seq"])
        model.load_state_dict(blob["model"], strict=False)
    else:
        model = VeritateMemory(vocab=256, hidden=64, layers=4, ffn=256, heads=2, seq=WINDOW)
    model.eval()
    torch.set_grad_enabled(True)

    facts, probes = facts_and_probes()
    doc = "".join(facts).encode()
    arr = np.memmap(DISTRACTOR_BIN, dtype=np.uint8, mode="r")
    distractor = bytes(arr[:DISTRACTOR_LEN])

    on_bpb, on_em = run_arm(model, doc, distractor, probes, device, carry=True)
    off_bpb, off_em = run_arm(model, doc, distractor, probes, device, carry=False)

    wins = sum(1 for a, b in zip(on_bpb, off_bpb) if a < b)
    stats = {
        "errors": errors,
        "checkpoint": ckpt or "untrained",
        "n_facts": N_FACTS,
        "distractor_bytes": DISTRACTOR_LEN,
        "mean_bpb_memory_on": round(sum(on_bpb) / len(on_bpb), 4),
        "mean_bpb_memory_off": round(sum(off_bpb) / len(off_bpb), 4),
        "bpb_win_rate_on_vs_off": round(wins / N_FACTS, 3),
        "exact_match_on": sum(on_em),
        "exact_match_off": sum(off_em),
        "recall_lift_points": round(100 * (sum(on_em) - sum(off_em)) / N_FACTS, 1),
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
