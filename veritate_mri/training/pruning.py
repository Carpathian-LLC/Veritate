# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - measure FFN unit activity per layer on a held-out corpus sample.
# - apply a per-layer width prune to the canonical Veritate state_dict.
# - the prune keeps the top-N most-active units in each layer; selection score
#   is the post-GELU activation magnitude across the sampled windows. units
#   with the largest aggregate magnitude survive; the rest are dropped.
# - the resulting state_dict has reduced ffn dimensions per layer; load it
#   with the canonical Veritate class which now accepts per-layer ffn lists.
# veritate_mri/training/pruning.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys
import time

_MRI_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _MRI_ROOT not in sys.path:
    sys.path.insert(0, _MRI_ROOT)

from readers import paths

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_SAMPLES   = 64
DEFAULT_THRESHOLD = 1e-3
ACTIVITY_KEEP_MIN = 0.05  # never recommend pruning below 5% width
DEFAULT_HEAD_DIM  = 64    # per-head width the canonical trunk uses; seeds the head-count guess
DEFAULT_VERSION   = "v1"


# ------------------------------------------------------------------------------------
# Functions

def resolve_corpus_mix(spec):
    """Resolve a stored corpus spec (single stem or weighted mixture like
    "a:0.6,b:0.4") to the [(train_path, val_path, weight)] the mixed loader
    consumes. Reuses the canonical multicorpus parser and platform resolver so
    pruning scores against the exact training distribution."""
    from readers import corpus as corpus_reader

    from veritate_core.plugin import multicorpus
    return multicorpus.resolve_and_weight(spec, corpus_reader.resolve_paths)


def measure_activity(model, corpus_mix, n_samples=DEFAULT_SAMPLES,
                     seq_len=None, threshold=DEFAULT_THRESHOLD, seed=0):
    """For each FFN layer, compute the per-unit activity score over n_samples
    random windows drawn from the corpus mix. Returns a list of dicts with the
    per-layer breakdown plus a per-unit score tensor the prune routine consumes.

    score[L][n] = max post-activation magnitude that unit n produced across all
                  sampled positions. a unit is 'alive' if its score > threshold.
    """
    import torch

    from veritate_core.plugin import multicorpus

    seq_len = int(seq_len or model.seq)
    draw, _ = multicorpus.make_mixed_loader(corpus_mix, 1, seq_len, int(seed))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    layers = model.layers
    cap = [None] * layers
    handles = []
    for L, blk in enumerate(model.blocks):
        # Variants with no (in, out) matmul pair hold no prunable FFN units at
        # all (product-key memory addresses slots, it has no width to trim), so
        # they are left out of the plan rather than scored on a stand-in tensor.
        if blk.ff.probe_weights() is None:
            continue
        # Score with the layer's own activation, not a hardcoded GELU, so ReLU
        # and SiLU models report faithful post-activation magnitudes.
        act_fn = blk.ff._act_fn
        def _hook(_m, _i, o, L=L, act_fn=act_fn):
            post = act_fn(o).abs()
            mag  = post.amax(dim=tuple(range(post.dim() - 1)))
            cap[L] = mag if cap[L] is None else torch.maximum(cap[L], mag)
        handles.append(blk.ff.up.register_forward_hook(_hook))

    if not handles:
        raise ValueError(
            "no prunable FFN layers in this model: width pruning needs dense FFN "
            "up/down matrices. Product-key-memory and Mixture-of-Experts trunks "
            "have no per-unit width to trim.")

    try:
        with torch.no_grad():
            for _ in range(int(n_samples)):
                toks, _ = draw()
                model(toks.to(device))
    finally:
        for h in handles: h.remove()
        if was_training: model.train()

    out = []
    scores = []
    for L in range(layers):
        s = cap[L].detach().float().cpu()
        scores.append(s)
        alive = int((s > threshold).sum().item())
        total = int(s.numel())
        out.append({
            "layer":      L,
            "alive":      alive,
            "total":      total,
            "alive_frac": alive / total if total else 0.0,
            "score_max":  float(s.max().item()) if total else 0.0,
            "score_mean": float(s.mean().item()) if total else 0.0,
        })
    return {"per_layer": out, "scores": scores, "n_samples": int(n_samples),
            "seq_len": seq_len, "threshold": float(threshold)}


def recommend_plan(report, _layers):
    """Take the per-layer alive_frac and bucket each layer into one of four
    actions: prune hard (25%), prune (50%), trim (70%), keep (100%). The cutoffs
    are conservative: never recommend dropping below 25% of the original width."""
    plan = {}
    for e in report["per_layer"]:
        L = e["layer"]
        f = e["alive_frac"]
        if f < 0.20:
            keep = 0.25
        elif f < 0.40:
            keep = 0.50
        elif f < 0.60:
            keep = 0.70
        else:
            keep = 1.00
        plan[str(L)] = round(keep, 3)
    return plan



def apply_plan(model, scores, plan):
    """Build a pruned state_dict from `model` using per-unit `scores` and a
    per-layer keep-fraction `plan`. Returns (new_state_dict, new_ffn_per_layer).

    For each layer L:
      keep_frac = plan[L] (clamped to [ACTIVITY_KEEP_MIN, 1.0])
      target_n  = max(1, round(orig_ffn * keep_frac))
      keep_idx  = topk(scores[L], target_n) by score
      W_up_new   = W_up[keep_idx, :]
      W_down_new = W_down[:, keep_idx]
    All other parameters are copied unchanged.
    """
    import torch

    sd = {k: v.clone() for k, v in model.state_dict().items()}
    new_ffn = []

    for L in range(model.layers):
        if model.blocks[L].ff.probe_weights() is None:
            new_ffn.append(model.ffn_per_layer[L])
            continue
        orig_ffn = model.ffn_per_layer[L]
        keep_frac = float(plan.get(str(L), plan.get(L, 1.0)))
        keep_frac = max(ACTIVITY_KEEP_MIN, min(1.0, keep_frac))
        target_n  = max(1, round(orig_ffn * keep_frac))
        if target_n >= orig_ffn:
            new_ffn.append(orig_ffn)
            continue

        s = scores[L]
        _, idx = torch.topk(s, target_n)
        idx, _ = idx.sort()

        up_key   = f"blocks.{L}.ff.up.weight"
        down_key = f"blocks.{L}.ff.down.weight"
        sd[up_key]   = sd[up_key][idx, :].contiguous()
        sd[down_key] = sd[down_key][:, idx].contiguous()
        new_ffn.append(target_n)

    return sd, new_ffn


# ------------------------------------------------------------------------------------
# Plugin generator

PLUGIN_SCRIPT = '''# ------------------------------------------------------------------------------------
# Auto-generated by the Pruning Report.
# Source model: {src_name} step {src_step}
# Generated:    {generated_at}
# ------------------------------------------------------------------------------------

import argparse
import json
import os
import sys
import time

import torch

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from veritate_core.plugin import save, paths, model as vmodel
from training import pruning


def parse_args():
    ap = argparse.ArgumentParser(description="Apply a pruning plan to a saved checkpoint.")
    ap.add_argument("--description", required=True, type=str)
    ap.add_argument("--source",   default="{src_name}")
    ap.add_argument("--step",     default={src_step}, type=int)
    ap.add_argument("--corpus",   default="{corpus_stem}")
    ap.add_argument("--samples",  default=64, type=int,
                    help="number of random windows to sample for activity scoring")
    ap.add_argument("--threshold",default=1e-3, type=float)
    ap.add_argument("--out_version", default="v1")
    return ap.parse_args()


def main():
    args = parse_args()
    save.require_description(args.description)

    plan_path = os.path.join(HERE, "plan.json")
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)["per_layer"]

    src_ckpt = paths.checkpoint_path(args.source, args.step)
    print(f"loading source checkpoint: {{src_ckpt}}", flush=True)
    s = torch.load(src_ckpt, map_location="cpu", weights_only=True)
    cfg = dict(s.get("args", {{}}))
    sd = s["model"]
    del s  # drops optimizer state (~8 GB on 1B) before model construction
    layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    ffn_per_layer = [sd[f"blocks.{{L}}.ff.up.weight"].shape[0] for L in range(layers)]
    vocab, hidden = sd["tok_emb.weight"].shape
    seq           = sd["pos_emb.weight"].shape[0]
    heads         = int(cfg.get("heads") or 0)
    if heads <= 0 or hidden % heads != 0:
        target = max(1, hidden // {head_dim})
        for h in sorted({{d for d in range(1, hidden + 1) if hidden % d == 0}},
                        key=lambda d: (abs(d - target), -d)):
            heads = h; break

    base = vmodel.Veritate(vocab=vocab, hidden=hidden, layers=layers,
                            ffn=ffn_per_layer if len(set(ffn_per_layer)) > 1 else ffn_per_layer[0],
                            heads=heads, seq=seq)
    base.load_state_dict(sd, strict=True)
    base.eval()

    corpus_mix = pruning.resolve_corpus_mix(args.corpus)
    print(f"measuring activity on {{args.samples}} sampled windows of {{args.corpus}}", flush=True)
    t0 = time.time()
    report = pruning.measure_activity(base, corpus_mix,
                                       n_samples=args.samples,
                                       threshold=args.threshold)
    print(f"  done in {{time.time() - t0:.1f}}s", flush=True)
    for e in report["per_layer"]:
        print(f"  layer {{e['layer']:2d}}: {{e['alive']:5d}}/{{e['total']:5d}} "
              f"alive ({{e['alive_frac']*100:5.1f}}%)", flush=True)

    pruned_sd, new_ffn = pruning.apply_plan(base, report["scores"], plan)
    print(f"new ffn per layer: {{new_ffn}}", flush=True)

    pruned = vmodel.Veritate(vocab=vocab, hidden=hidden, layers=layers,
                              ffn=new_ffn if len(set(new_ffn)) > 1 else new_ffn[0],
                              heads=heads, seq=seq)
    pruned.load_state_dict(pruned_sd, strict=True)
    pruned.eval()

    new_params = sum(p.numel() for p in pruned.parameters())
    size_tag = f"{{max(1, round(new_params / 1e6))}}m"
    out_name = save.compose_name(args.corpus, size_tag, "bf16", args.out_version)
    new_args = dict(cfg)
    new_args["description"] = args.description
    new_args["pruned_from"] = f"{{args.source}}/step_{{args.step}}"
    new_args["plan"]        = plan
    new_args["ffn"]         = new_ffn if len(set(new_ffn)) > 1 else new_ffn[0]
    new_args["vocab"]       = vocab
    new_args["hidden"]      = hidden
    new_args["layers"]      = layers
    new_args["heads"]       = heads
    new_args["seq"]         = seq
    new_args["corpus"]      = args.corpus

    print(f"writing pruned checkpoint as model {{out_name}} step 0", flush=True)
    save.save(pruned, out_name, 0, args=new_args)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
'''


def generate_plugin(src_name, src_step, plan, report, corpus_stem, plugins_dir=None):
    """Write a one-shot plugin folder under trainers/ that applies the given plan
    to the source checkpoint. Returns (plugin_dir, plugin_id)."""
    plugins_dir = plugins_dir or paths.PLUGINS_ROOT
    plugin_id = f"prune_{src_name}_step{src_step}"
    plugin_dir = os.path.join(plugins_dir, plugin_id)
    os.makedirs(plugin_dir, exist_ok=True)

    manifest = {
        "name":        f"Prune: {src_name} step {src_step}",
        "description": f"Auto-generated. Applies a width-prune plan to {src_name} step {src_step}.",
        "kind":        "trainer",
        "flow":        "prune",
        "defaults": {
            "out_version": DEFAULT_VERSION,
            "samples":     DEFAULT_SAMPLES,
            "threshold":   DEFAULT_THRESHOLD,
            "version":     DEFAULT_VERSION,
        }
    }
    with open(os.path.join(plugin_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    plan_blob = {
        "source":      {"model": src_name, "step": int(src_step)},
        "corpus_stem": corpus_stem,
        "per_layer":   plan,
        "report":      [{
            "layer":      e["layer"],
            "alive":      e["alive"],
            "total":      e["total"],
            "alive_frac": round(e["alive_frac"], 4),
            "score_max":  round(e["score_max"],  6),
        } for e in report["per_layer"]],
        "generated_at": int(time.time()),
    }
    with open(os.path.join(plugin_dir, "plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan_blob, f, indent=2)

    script = PLUGIN_SCRIPT.format(
        src_name=src_name, src_step=int(src_step),
        corpus_stem=corpus_stem, head_dim=DEFAULT_HEAD_DIM,
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(os.path.join(plugin_dir, "trainer.py"), "w", encoding="utf-8") as f:
        f.write(script)

    return plugin_dir, plugin_id
