# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Context-grounded (RAG) SFT: continue-trains a saved checkpoint on the grounded
#   corpus (context + question + answer-from-context) so the model copies facts out
#   of its context window. Writes a NEW model dir; the source is untouched.
# - Full-sequence LM loss. The source checkpoint's args carry through, so the
#   canonical factory reloads the right variant on the next load.
# - Save discipline (rule 21): checkpoints through save.save(), CSV rows through
#   save.append_train_row().
# - Spawned as a subprocess by veritate_mri/routes/rag_routes.py; stdout is the
#   job log the RAG panel tails.
# veritate_mri/training/rag_sft.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import sys

import torch

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MRI_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_REPO     = os.path.normpath(os.path.join(_MRI_ROOT, ".."))
for _p in (_REPO, _MRI_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from readers import checkpoints  # noqa: E402

from training import save as save_mod  # noqa: E402
from training.native_trainer import make_loader  # noqa: E402
from veritate_core.load import load_from_state_dict  # noqa: E402
from veritate_core.plugin import hardware  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_CORPUS = "grounded_v1"
DEFAULT_STEPS  = 1500
DEFAULT_SEQ    = 512
DEFAULT_BATCH  = 8
DEFAULT_LR     = 5e-5
DEFAULT_SEED   = 1234
GRAD_CLIP      = 1.0
EVAL_EVERY     = 200
EVAL_ITERS     = 32
CKPT_EVERY     = 500
BETAS          = (0.9, 0.95)
ADAM_EPS       = 1e-8
MAP_CPU        = "cpu"

# ------------------------------------------------------------------------------------
# Functions

def load_source(name):
    step = checkpoints.latest_step(name)
    if step is None:
        raise FileNotFoundError(f"no checkpoint found for source model {name!r}")
    blob = torch.load(checkpoints.path_for(name, step), map_location=MAP_CPU, weights_only=True)
    src_args = dict(blob.get("args", {}))
    return load_from_state_dict(blob["model"], src_args, strict_canonical=False), src_args


def save_checkpoint(model, src_args, opt, name, step, corpus, description):
    ckpt_args = dict(src_args)
    ckpt_args["description"]      = description
    ckpt_args["corpus"]           = corpus
    ckpt_args["grounded_sft_from"] = src_args.get("name", "")
    return save_mod.save(model, name, step, optimizer=opt, args=ckpt_args)


def eval_val_loss(model, draw, device, iters):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for _ in range(iters):
            x, y = draw()
            _, loss = model(x.to(device), y.to(device))
            if torch.isfinite(loss):
                total += float(loss)
                n += 1
    model.train()
    return total / n if n else None


def train(source, name, corpus, steps, seq, batch, lr, seed):
    device = torch.device(hardware.pick_device())
    torch.manual_seed(seed)

    model, src_args = load_source(source)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=BETAS, eps=ADAM_EPS)

    train_bin, val_bin = save_mod.resolve_corpus(corpus)
    seq = min(seq, model.seq)
    train_draw, n_train = make_loader(train_bin, seq, batch, seed)
    val_draw = make_loader(val_bin, seq, batch, seed + 1)[0] if val_bin and os.path.isfile(val_bin) else None
    print(f"[rag_sft] device={device} corpus={corpus} train_bytes={n_train:,} steps={steps}", flush=True)

    description = f"grounded SFT of {source} on {corpus}"
    final_val = None
    for step in range(1, steps + 1):
        x, y = train_draw()
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            save_mod.append_train_row(name, step, "train", float(loss.detach()), seed=seed)
            if val_draw is not None:
                final_val = eval_val_loss(model, val_draw, device, EVAL_ITERS)
                save_mod.append_train_row(name, step, "val", final_val, seed=seed)
            print(f"[rag_sft] step {step} train={float(loss.detach()):.4f} val={final_val}", flush=True)
        if step % CKPT_EVERY == 0 or step == steps:
            path = save_checkpoint(model, src_args, opt, name, step, corpus, description)
            print(f"[rag_sft] checkpoint + hooks: {path}", flush=True)

    return {"name": name, "source": source, "steps": steps, "final_val": final_val}


def main():
    ap = argparse.ArgumentParser(description="Context-grounded (RAG) SFT of a saved checkpoint.")
    ap.add_argument("--source", type=str,   required=True)
    ap.add_argument("--name",   type=str,   required=True)
    ap.add_argument("--corpus", type=str,   default=DEFAULT_CORPUS)
    ap.add_argument("--steps",  type=int,   default=DEFAULT_STEPS)
    ap.add_argument("--seq",    type=int,   default=DEFAULT_SEQ)
    ap.add_argument("--batch",  type=int,   default=DEFAULT_BATCH)
    ap.add_argument("--lr",     type=float, default=DEFAULT_LR)
    ap.add_argument("--seed",   type=int,   default=DEFAULT_SEED)
    args = ap.parse_args()
    print(json.dumps(train(args.source, args.name, args.corpus, args.steps,
                           args.seq, args.batch, args.lr, args.seed)), flush=True)


if __name__ == "__main__":
    main()
