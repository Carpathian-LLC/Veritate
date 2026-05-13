# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - diagnostic sampler. loads a MEGA checkpoint, runs N diverse prompts through
#   top-p sampling, prints each completion. used to classify failure modes
#   (partial-word cul-de-sac, repetition loop, semantic drift, gibberish).
# - read-only on the checkpoint. safe to run alongside training as long as the
#   chosen step file is not the one currently being written by save().
# veritate_mri/tools/sample_diverse.py
# ------------------------------------------------------------------------------------
# Imports

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "plugins", "veritate_mega"))
sys.path.insert(0, os.path.join(REPO, "plugins", "multimind_m3"))

from veritate_core.plugin import qat as qat_helpers
from veritate_mri.addons import build_chain, list_addons

# ------------------------------------------------------------------------------------
# Constants

MODELS_DIR        = os.path.join(REPO, "models")
CKPT_PREFIX       = "step_"
CKPT_SUFFIX       = ".pt"
CONFIG_FILENAME   = "config.json"
CHECKPOINTS_SUB   = "checkpoints"

DEFAULT_PROMPTS = [
    "Once upon a time, there was a little girl who",
    "The cat sat on the",
    "Tom and his mother went to the",
    "Hello, my name is",
    "In a small house at the edge of the forest,",
    "She picked up the red ball and",
    "The dragon flew over the",
    "Sara loved to draw pictures of",
    "It was a sunny day. Ben was",
    "Mommy said, \"Don't",
    "The little boy was very",
    "On Sunday morning, the family",
    "Lily found a tiny",
    "He looked up at the sky and saw",
    "After breakfast, they went outside to",
    "The teacher smiled and said,",
    "When the rain stopped, the",
    "A big brown bear walked through the",
    "Anna held her baby brother and",
    "The end of the story was",
]

DEFAULT_MAX_NEW    = 120
DEFAULT_TEMP       = 0.8
DEFAULT_TOP_P      = 0.95
DEFAULT_SEED       = 0
DEFAULT_DEVICE     = "cuda"


# ------------------------------------------------------------------------------------
# Functions

def find_latest_step(ckpt_dir):
    steps = []
    for fn in os.listdir(ckpt_dir):
        if fn.startswith(CKPT_PREFIX) and fn.endswith(CKPT_SUFFIX):
            try:
                steps.append(int(fn[len(CKPT_PREFIX):-len(CKPT_SUFFIX)]))
            except ValueError:
                continue
    if not steps:
        raise FileNotFoundError("no checkpoints under: " + ckpt_dir)
    return max(steps)


def build_model_from_config(cfg, device):
    shape  = cfg["shape"]
    plugin = cfg.get("plugin", "")
    if plugin == "multimind_mega":
        from mega_model import Veritate, MegaModel, VOCAB_BYTE_LEVEL
        base = Veritate(vocab=VOCAB_BYTE_LEVEL, hidden=shape["hidden"], layers=shape["layers"],
                        ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"])
        moe = cfg["mega"]
        model = MegaModel(base, n_experts=moe["n_experts"], router_topk=moe["router_topk"],
                          quant_mode=cfg["quant_mode"], router_aux_loss_w=moe["router_aux_loss"],
                          label_smoothing=0.0)
    elif plugin == "multimind_m3":
        from m3_model import Veritate, HoloModel, VOCAB_BYTE_LEVEL
        base = Veritate(vocab=VOCAB_BYTE_LEVEL, hidden=shape["hidden"], layers=shape["layers"],
                        ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"])
        m3 = cfg["m3"]
        model = HoloModel(base, rank=m3["rank"], alpha=m3["alpha"],
                          inject_layer=m3["inject_layer"], label_smoothing=0.0)
    else:
        raise ValueError("unsupported plugin in config: " + repr(plugin))
    if cfg.get("training") == "qat":
        qat_helpers.set_qat(model, True)
    model.to(device)
    model.eval()
    return model


def load_state(model, ckpt_path, device):
    # Load to CPU first so we never spike VRAM with the unused optimizer state
    # that the trainer wrote to the .pt. For a 1B mega checkpoint, the
    # optimizer dict (AdamW m + v fp32) is ~8 GB — bigger than the model
    # itself. Dropping it immediately keeps resident RAM near the model size.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["model"]
    del ckpt  # frees ckpt["optimizer"] + ckpt["args"]; ~8 GB on 1B
    if any(k.startswith("base.") for k in sd):
        model.load_state_dict(sd)
    else:
        target = getattr(model, "base", model)
        target.load_state_dict(sd, strict=False)
    del sd  # frees the duplicate state_dict copy now that values are in the model


def top_p_filter(logits, top_p):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > top_p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_idx, sorted_logits)
    return out


@torch.no_grad()
def sample_one(model, prompt, max_new, temperature, top_p, max_ctx, device, gen, chain):
    prompt_bytes = list(prompt.encode("utf-8", errors="replace"))
    tokens = list(prompt_bytes)
    out_bytes = []
    chain.reset()
    chain.observe_bytes(prompt_bytes)
    for _ in range(max_new):
        ctx = tokens[-max_ctx:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _, _ = model(x, state=None, targets=None)
        last = logits[0, -1, :].float()
        last = chain.bias_logits(last)
        last = last / max(1e-6, temperature)
        last = top_p_filter(last, top_p)
        probs = F.softmax(last, dim=-1)
        nb = int(torch.multinomial(probs, num_samples=1, generator=gen).item())
        tokens.append(nb)
        out_bytes.append(nb)
        chain.observe(nb)
    return bytes(out_bytes).decode("utf-8", errors="replace")


def run(args):
    cfg_path = os.path.join(MODELS_DIR, args.model, CONFIG_FILENAME)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ckpt_dir = os.path.join(MODELS_DIR, args.model, CHECKPOINTS_SUB)
    step = args.step if args.step > 0 else find_latest_step(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, CKPT_PREFIX + str(step) + CKPT_SUFFIX)

    device = args.device
    print("model:    " + args.model, flush=True)
    print("step:     " + str(step) + " (" + ckpt_path + ")", flush=True)
    print("device:   " + device, flush=True)
    print("temp:     " + str(args.temperature) + "  top_p: " + str(args.top_p), flush=True)
    print("seed:     " + str(args.seed) + "  max_new: " + str(args.max_new), flush=True)

    t0 = time.time()
    model = build_model_from_config(cfg, device)
    load_state(model, ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print("loaded:   " + str(round(n_params / 1e6)) + "M params in " + format(time.time() - t0, ".1f") + "s", flush=True)

    max_ctx = cfg["shape"]["seq"]
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    chain = build_chain(args.addons)
    print("addons:   " + (", ".join(args.addons) if args.addons else "none"), flush=True)

    print("", flush=True)
    print("=" * 80, flush=True)
    for i, prompt in enumerate(args.prompts):
        ts = time.time()
        completion = sample_one(model, prompt, args.max_new, args.temperature, args.top_p, max_ctx, device, gen, chain)
        dt = time.time() - ts
        print("[" + str(i + 1).rjust(2) + "] (" + format(dt, ".1f") + "s) " + repr(prompt), flush=True)
        print("     " + repr(completion), flush=True)
        print("", flush=True)
    print("=" * 80, flush=True)
    print("total:    " + format(time.time() - t0, ".1f") + "s", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="diverse-prompt sampler for failure-mode classification")
    ap.add_argument("--model",       type=str, required=True)
    ap.add_argument("--step",        type=int, default=0, help="0 = latest")
    ap.add_argument("--max_new",     type=int, default=DEFAULT_MAX_NEW)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--top_p",       type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    ap.add_argument("--device",      type=str, default=DEFAULT_DEVICE)
    ap.add_argument("--prompts_file", type=str, default="", help="optional: path to a text file with one prompt per line")
    ap.add_argument("--addons",       type=str, default="", help="comma-separated list of addon ids (see veritate_mri/addons/). example: slot_table")
    ap.add_argument("--list_addons",  action="store_true", help="print available addons and exit")
    args = ap.parse_args()
    if args.list_addons:
        for a in list_addons():
            m = a["manifest"]
            print(a["id"] + "  -  " + m.get("name", "") + "  -  " + m.get("description", ""))
        sys.exit(0)
    args.addons = [s.strip() for s in args.addons.split(",") if s.strip()]
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            args.prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
    else:
        args.prompts = list(DEFAULT_PROMPTS)
    return args


if __name__ == "__main__":
    run(parse_args())
