# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Function-preserving model growth (Net2Net / LLaMA-Pro family, IDEA 21).
#   Two consumers: the trainer's mid-run FFN widen (widen_model / val_has_flattened,
#   unchanged) and the standalone checkpoint-to-checkpoint tool (grow_checkpoint +
#   CLI: python -m training.grow src.pt --layers N --hidden H --ffn F --heads K
#   --out dst.pt). The tool is pure state-dict surgery: it never builds a model,
#   so it runs on CPU in checkpoint-sized memory.
# - Growth rules, all exact at fp precision:
#   hidden: duplicate stream channels with write scale sqrt(H'/(H*n)) so RMSNorm's
#   mean-square (and its eps) is preserved exactly; n1/n2 copies divide by the
#   write scale, reads split evenly, n_out copies scale by H/H' which the tied
#   lm_head/tok_emb forces. heads with head_dim fixed: new head slices copy an
#   existing head's input-side rows (deterministic copies, not random: no RNG,
#   and the zeroed output column is what silences them) and zero their attn.proj
#   columns. head_dim growth (hidden up, heads fixed): within-head q duplicates
#   scaled sqrt(d'/d)/n (also absorbs the 1/sqrt(d') attention scale), k copies
#   unscaled, v copies scaled sqrt(d'/(d*n)) so o_norm's mean-square is exact;
#   conv kernels and gate rows follow their channel. ffn: duplicate up rows,
#   split down columns. layers: identity global blocks (copy of the last global
#   block with attn.proj and ff.down zeroed) appended to the global span.
#   seq: EXTEND the learned tables, never interpolate (interpolation changes the
#   function at every position). pos_emb/slot_pos_emb keep rows 0..old-1
#   bit-identical; new rows copy the LAST learned row - rows past the old domain
#   are never read for old-domain inputs, so any init preserves the function,
#   and copy-of-last keeps embedding norms in-distribution at the positions the
#   continue run trains (zeros would drop the positional signal entirely there).
#   Patched-trunk caveat: slots = seq/PATCH_STRIDE grows with seq, so exactness
#   holds for inputs whose boundary-byte count <= the OLD slot count; inputs
#   beyond that were silently slot-truncated by the source model and gain live
#   slots (a strict capacity gain, not a preservation break on the trained
#   domain). No other tensor is seq-tied: boundary table is vocab-indexed and
#   non-persistent, GLA CHUNK padding and the conv ring are window-local, sdpa
#   builds its causal mask per call.
# - Duplicated channels start exactly symmetric and their gradients are too;
#   Net2Net breaks the tie with noise, this tool deliberately does not (exactness
#   is the contract). Data-order and optimizer asymmetry separate them slowly.
# - The grown checkpoint carries NO optimizer state: moment shapes cannot be
#   grown meaningfully, so continuation re-warms the optimizer from scratch
#   (use warmup_steps > 0 on the continue run).
# - Mid-run FFN widening rule (trainer path): new ff.up rows get fresh init, new
#   ff.down columns are exactly zero. Output is bit-identical at the widen step,
#   and the new up rows still receive gradient because d(loss)/d(down_col) is
#   nonzero.
# - The val-flatten detector is the point of the mid-run path. Growth measured
#   3.6x steps-to-target but cost 1.66x total compute because the small stage ran
#   ~950 steps past saturation (successes.md 2026-07-25). Growth only nets
#   compute if each stage stops the moment its val curve flattens, so that
#   decision cannot be a hand-picked step count.
# veritate_mri/training/grow.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import math
import os
import re

import torch

# ------------------------------------------------------------------------------------
# Constants

UP_SUFFIX      = ".ff.up.weight"
DOWN_SUFFIX    = ".ff.down.weight"
FLAT_WINDOW    = 4
FLAT_REL_GAIN  = 0.005
INIT_STD       = 0.02

SHAPE_FIELDS   = ("layers", "hidden", "ffn", "heads")
# Stream-width tensors: [., hidden] embeddings plus the tied head.
STREAM_EMB_KEYS = ("tok_emb.weight", "pos_emb.weight", "slot_pos_emb.weight",
                   "lm_head.weight")
# Keys that mark an unsupported variant; growth refuses rather than guessing.
REJECT_PATTERNS = (
    ("loop_inj", "looped trunk (weight-tied loop depth)"),
    (".attn.b_proj.", "state_rule=delta (WY inverse bookkeeping)"),
    (".attn.pin_key", "state_rule=pinned (decay-exempt slots)"),
)
# Identity-block rule: zero the tensors that WRITE into the residual stream.
IDENTITY_ZERO_SUFFIXES = ("attn.proj.weight", "ff.down.weight")

# ------------------------------------------------------------------------------------
# Functions


def ffn_widths(state_dict):
    """Per-layer FFN width, keyed by layer index, for layers that carry an up/down pair."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith(UP_SUFFIX):
            out[int(k.split(".")[1])] = int(v.shape[0])
    return out


def widen_state_dict(state_dict, target_ffn, generator=None):
    """Widen every dense FFN to target_ffn, preserving the function exactly.

    New up rows are freshly initialized; new down columns are zero, so the layer
    output is unchanged at the widen step. Layers already at or above target, and
    layers with no up/down pair (product-key memory, MoE), are left alone.
    """
    sd = dict(state_dict)
    widened = []
    for layer, width in sorted(ffn_widths(sd).items()):
        if width >= target_ffn:
            continue
        up_key, down_key = f"blocks.{layer}{UP_SUFFIX}", f"blocks.{layer}{DOWN_SUFFIX}"
        if down_key not in sd:
            continue
        up, down = sd[up_key], sd[down_key]
        extra = target_ffn - width
        new_up = torch.empty(extra, up.shape[1], dtype=up.dtype, device=up.device)
        new_up.normal_(mean=0.0, std=INIT_STD, generator=generator)
        sd[up_key] = torch.cat([up, new_up], dim=0).contiguous()
        new_down = torch.zeros(down.shape[0], extra, dtype=down.dtype, device=down.device)
        sd[down_key] = torch.cat([down, new_down], dim=1).contiguous()
        widened.append(layer)
    return sd, widened


def widen_model(model, target_ffn, generator=None):
    """Widen a LIVE model's dense FFNs in place, preserving the function.

    Swaps the up/down Parameters rather than rebuilding the model, so a training
    loop can grow mid-run and only has to rebuild the optimizer afterwards.
    Returns the layer indices that grew.
    """
    import torch.nn as nn

    widened = []
    for i, blk in enumerate(model.blocks):
        ff = blk.ff
        if getattr(ff, "probe_weights", None) is None or ff.probe_weights() is None:
            continue
        up, down = ff.up, ff.down
        cur = int(up.weight.shape[0])
        if cur >= target_ffn:
            continue
        extra = target_ffn - cur
        new_rows = torch.empty(extra, up.weight.shape[1],
                               dtype=up.weight.dtype, device=up.weight.device)
        new_rows.normal_(mean=0.0, std=INIT_STD, generator=generator)
        up.weight = nn.Parameter(torch.cat([up.weight.data, new_rows], dim=0).contiguous())
        up.out_features = target_ffn
        new_cols = torch.zeros(down.weight.shape[0], extra,
                               dtype=down.weight.dtype, device=down.weight.device)
        down.weight = nn.Parameter(torch.cat([down.weight.data, new_cols], dim=1).contiguous())
        down.in_features = target_ffn
        widened.append(i)
    model.ffn = target_ffn
    model.ffn_per_layer = [target_ffn] * model.layers
    return widened


def val_has_flattened(val_losses, window=FLAT_WINDOW, rel_gain=FLAT_REL_GAIN):
    """True once the last `window` validations have stopped buying real improvement.

    Compares the best val in the trailing window against the best before it; a
    relative gain under rel_gain means the stage is saturated and every further
    step is compute spent at the CHEAP shape for nothing.
    """
    if len(val_losses) < window * 2:
        return False
    # Compare against the IMMEDIATELY preceding window, not all history: early
    # large gains would otherwise mask a curve that has since gone flat.
    prior = min(val_losses[-2 * window:-window])
    recent = min(val_losses[-window:])
    if prior <= 0:
        return False
    return (prior - recent) / prior < rel_gain


def stage_compute(params, steps):
    """Relative compute for a stage: params x steps. Growth only wins on this sum."""
    return float(params) * float(steps)


def _dup_map(old, new):
    """Duplication map old -> new: source index per grown slot, plus per-source
    multiplicity. First `old` slots are identity; extra slots wrap the low indices."""
    src = [i if i < old else (i - old) % old for i in range(new)]
    counts = [0] * old
    for s in src:
        counts[s] += 1
    return src, counts


def _scaled_select(t, dim, src, scale=None):
    """index_select along dim by src, then multiply per-slot by scale."""
    out = torch.index_select(t, dim, torch.as_tensor(src, dtype=torch.long, device=t.device))
    if scale is not None:
        s = torch.as_tensor(scale, dtype=t.dtype, device=t.device)
        shape = [1] * out.dim()
        shape[dim] = len(scale)
        out = out * s.view(shape)
    return out.contiguous()


def _block_indices(sd):
    return sorted({int(m.group(1)) for m in
                   (re.match(r"blocks\.(\d+)\.", k) for k in sd) if m})


def _recurrent_indices(sd):
    return [i for i in _block_indices(sd) if f"blocks.{i}.attn.gate.weight" in sd]


def checkpoint_shape_sd(sd, src_heads=0):
    """Read (layers, hidden, ffn, heads) out of a model state_dict.

    `layers` is the constructor's number: the recurrent (global) span for a
    hybrid/patched trunk, the full block count for a pure trunk. `heads` comes
    from a_proj when a recurrent mixer exists, else from src_heads (qkv packs
    q/k/v, so a pure-attention trunk cannot reveal its head count).
    """
    for pattern, why in REJECT_PATTERNS:
        hits = [k for k in sd if pattern in k]
        if hits:
            raise ValueError(f"growth does not support {why} (found {hits[0]})")
    idxs = _block_indices(sd)
    if not idxs or "tok_emb.weight" not in sd:
        raise ValueError("not a Veritate model state_dict: no blocks.* / tok_emb keys")
    for i in idxs:
        if f"blocks.{i}.ff.up.weight" not in sd:
            raise ValueError(f"block {i} has no dense ff.up/ff.down pair "
                             "(MoE/PKM/Monarch FFNs are not growable)")
    rec = _recurrent_indices(sd)
    if rec and rec != list(range(rec[0], rec[0] + len(rec))):
        raise ValueError(f"recurrent blocks are not contiguous: {rec}")
    heads = int(sd[f"blocks.{rec[0]}.attn.a_proj.weight"].shape[0]) if rec else int(src_heads)
    return {
        "layers": len(rec) if rec else len(idxs),
        "hidden": int(sd["tok_emb.weight"].shape[1]),
        "ffn":    int(sd[f"blocks.{idxs[0]}.ff.up.weight"].shape[0]),
        "heads":  heads,
        "seq":    int(sd["pos_emb.weight"].shape[0]),
    }


def seq_stride_sd(sd):
    """Patch stride the checkpoint's own tables imply: pos rows / slot rows for
    the patched trunk, 1 (no constraint) for pure trunks. Derived from the
    weights so it cannot drift from the model constants."""
    if "slot_pos_emb.weight" not in sd:
        return 1
    seq, slots = int(sd["pos_emb.weight"].shape[0]), int(sd["slot_pos_emb.weight"].shape[0])
    if slots <= 0 or seq % slots:
        raise ValueError(f"pos_emb rows {seq} not a multiple of slot_pos_emb rows {slots}")
    return seq // slots


def _width_plan(hidden, heads, hidden_new, heads_new):
    """Index/scale vectors for growing hidden and/or heads, derived once.

    Stream channels: n_i copies of channel i are written at c_i = sqrt(H'/(H*n_i))
    so sum(c^2 per source) = H'/H and the grown RMSNorm mean-square equals the
    source's exactly, eps included. Reads of the (exactly reproduced) post-norm
    copies split evenly. The tied lm_head forces n_out copies to scale by H/H'.
    Head interior (per head, d -> d'): q copies at sqrt(d'/d)/n (sum over copies
    = sqrt(d'/d), cancelling the 1/sqrt(d') attention scale back to 1/sqrt(d)),
    k copies unscaled (state rows duplicate whole), v copies at sqrt(d'/(d*n))
    so o_norm's mean-square is exact. New heads (h >= K) copy source head rows
    verbatim; their proj columns are zeroed by the col vectors.
    """
    d, d_new = hidden // heads, hidden_new // heads_new
    ssrc, scnt = _dup_map(hidden, hidden_new)
    write = [math.sqrt(hidden_new / (hidden * scnt[s])) for s in ssrc]
    plan = {
        "ssrc": ssrc,
        "write": write,
        "read":  [1.0 / scnt[s] for s in ssrc],
        "norm":  [1.0 / w for w in write],
        "n_out": [hidden / hidden_new] * hidden_new,
        "hsrc":  [h if h < heads else (h - heads) % heads for h in range(heads_new)],
    }
    msrc, mcnt = _dup_map(d, d_new)
    rho = math.sqrt(d_new / d)
    cv = [math.sqrt(d_new / (d * mcnt[m])) for m in msrc]
    plan["o_norm_src"], plan["o_norm_scale"] = msrc, [1.0 / c for c in cv]
    isec, sq, sv = [], [], []
    col_rec, col_loc = [], []
    for h in range(heads_new):
        for j, m in enumerate(msrc):
            isec.append(plan["hsrc"][h] * d + m)
            sq.append(rho / mcnt[m])
            sv.append(cv[j])
            if h >= heads:
                col_rec.append(0.0)
                col_loc.append(0.0)
            else:
                col_rec.append(1.0 / mcnt[m])
                col_loc.append(1.0 / (mcnt[m] * cv[j]))
    plan.update(isec=isec, sq=sq, sv=sv, col_rec=col_rec, col_loc=col_loc)
    return plan


def _grow_width_sd(sd, hidden_new, heads_new, shape):
    """Grow hidden/heads across the whole state_dict, function preserved."""
    hidden, heads = shape["hidden"], shape["heads"]
    p = _width_plan(hidden, heads, hidden_new, heads_new)
    rec = set(_recurrent_indices(sd))
    out = {}
    for k, t in sd.items():
        if k in STREAM_EMB_KEYS:
            out[k] = _scaled_select(t, 1, p["ssrc"], p["write"])
        elif k == "n_out.weight":
            out[k] = _scaled_select(t, 0, p["ssrc"], p["n_out"])
        elif k.endswith((".n1.weight", ".n2.weight")):
            out[k] = _scaled_select(t, 0, p["ssrc"], p["norm"])
        elif k.endswith(".attn.qkv.weight"):
            q, kk, v = _scaled_select(t, 1, p["ssrc"], p["read"]).split(hidden, dim=0)
            out[k] = torch.cat([_scaled_select(q,  0, p["isec"], p["sq"]),
                                _scaled_select(kk, 0, p["isec"]),
                                _scaled_select(v,  0, p["isec"], p["sv"])], dim=0)
        elif k.endswith(".attn.conv.weight"):
            out[k] = torch.cat([_scaled_select(sec, 0, p["isec"])
                                for sec in t.split(hidden, dim=0)], dim=0)
        elif k.endswith(".attn.gate.weight"):
            out[k] = _scaled_select(_scaled_select(t, 1, p["ssrc"], p["read"]),
                                    0, p["isec"])
        elif k.endswith(".attn.a_proj.weight"):
            out[k] = _scaled_select(_scaled_select(t, 1, p["ssrc"], p["read"]),
                                    0, p["hsrc"])
        elif k.endswith(".attn.a_proj.bias"):
            out[k] = _scaled_select(t, 0, p["hsrc"])
        elif k.endswith(".attn.o_norm.weight"):
            out[k] = _scaled_select(t, 0, p["o_norm_src"], p["o_norm_scale"])
        elif k.endswith(".attn.proj.weight"):
            blk = int(re.match(r"blocks\.(\d+)\.", k).group(1))
            col = p["col_rec"] if blk in rec else p["col_loc"]
            out[k] = _scaled_select(_scaled_select(t, 0, p["ssrc"], p["write"]),
                                    1, p["isec"], col)
        elif k.endswith(UP_SUFFIX):
            out[k] = _scaled_select(t, 1, p["ssrc"], p["read"])
        elif k.endswith(DOWN_SUFFIX):
            out[k] = _scaled_select(t, 0, p["ssrc"], p["write"])
        else:
            out[k] = t
    return out


def _grow_ffn_sd(sd, target_ffn):
    """Widen every dense FFN to target_ffn by duplicate-up / split-down (exact)."""
    out = dict(sd)
    for i in _block_indices(sd):
        up_key, down_key = f"blocks.{i}{UP_SUFFIX}", f"blocks.{i}{DOWN_SUFFIX}"
        width = int(sd[up_key].shape[0])
        if width >= target_ffn:
            continue
        fsrc, fcnt = _dup_map(width, target_ffn)
        out[up_key]   = _scaled_select(sd[up_key],   0, fsrc)
        out[down_key] = _scaled_select(sd[down_key], 1, fsrc,
                                       [1.0 / fcnt[s] for s in fsrc])
    return out


def _grow_depth_sd(sd, layers_new, shape):
    """Insert identity-initialized blocks at the end of the global span.

    New blocks copy the last global block's tensors with attn.proj and ff.down
    zeroed: both residual writes are exactly zero, so the block is the identity.
    The copied input-side weights keep gradients alive (d(loss)/d(zeroed write)
    is nonzero) and start the block on trained features. Streaming states are
    checkpoint-bound and re-form from conversation, so new blocks simply produce
    their own (IDEA 21).
    """
    rec = _recurrent_indices(sd)
    if not rec and "slot_pos_emb.weight" in sd:
        raise ValueError("cannot place identity blocks: patched trunk with attention "
                         "global mixer has no detectable global span")
    idxs = _block_indices(sd)
    insert_at = (rec[-1] + 1) if rec else len(idxs)
    template = rec[-1] if rec else idxs[-1]
    n_new = layers_new - shape["layers"]
    prefix = f"blocks.{template}."
    out = {}
    for k, t in sd.items():
        m = re.match(r"blocks\.(\d+)\.(.*)", k)
        if m and int(m.group(1)) >= insert_at:
            out[f"blocks.{int(m.group(1)) + n_new}.{m.group(2)}"] = t
        else:
            out[k] = t
    for j in range(n_new):
        for k, t in sd.items():
            if k.startswith(prefix):
                suffix = k[len(prefix):]
                out[f"blocks.{insert_at + j}.{suffix}"] = (
                    torch.zeros_like(t) if suffix in IDENTITY_ZERO_SUFFIXES
                    else t.clone())
    return out


def _grow_seq_sd(sd, seq_new, shape):
    """Extend the learned position tables to seq_new: existing rows exactly as
    trained, new rows copies of the last learned row (see file header)."""
    out = dict(sd)
    pe = sd["pos_emb.weight"]
    out["pos_emb.weight"] = torch.cat(
        [pe, pe[-1:].repeat(seq_new - shape["seq"], 1)], dim=0).contiguous()
    if "slot_pos_emb.weight" in sd:
        sp = sd["slot_pos_emb.weight"]
        slots_new = seq_new // seq_stride_sd(sd)
        out["slot_pos_emb.weight"] = torch.cat(
            [sp, sp[-1:].repeat(slots_new - int(sp.shape[0]), 1)], dim=0).contiguous()
    return out


def _stamp_shape(obj, shape):
    """Rewrite layers/hidden/ffn/heads/seq wherever they appear in a nested args
    dict, so the grown checkpoint's embedded args agree with its weights."""
    if isinstance(obj, dict):
        for f in (*SHAPE_FIELDS, "seq"):
            if f in obj and f in shape:
                obj[f] = shape[f]
        for v in obj.values():
            _stamp_shape(v, shape)


def validate_growth(src, target, seq_multiple=1):
    """The tool's shape rules, shared with any caller that pre-validates.

    Growth only: every target axis >= its source. Width growth additionally
    needs a known source head count, divisible hidden/heads on both sides, and
    a non-shrinking head_dim (heads growth therefore requires proportional
    hidden growth in this architecture). seq (validated when both dicts carry
    it) must also be a multiple of seq_multiple - the patched trunk's slot
    stride. Raises ValueError with a user-readable message; returns None when
    the target is reachable.
    """
    fields = SHAPE_FIELDS + (("seq",) if "seq" in src and "seq" in target else ())
    for f in fields:
        if target[f] < src[f]:
            raise ValueError(f"growth only: target {f} {target[f]} < source {src[f]}")
    if "seq" in target and seq_multiple > 1 and target["seq"] % seq_multiple:
        raise ValueError(f"seq must be a multiple of the patch stride "
                         f"{seq_multiple}, got {target['seq']}")
    if target["hidden"] > src["hidden"] or target["heads"] > src["heads"]:
        if src["heads"] <= 0:
            raise ValueError("source head count unknown: no recurrent mixer in the "
                             "checkpoint; pass src_heads")
        if src["hidden"] % src["heads"] or target["hidden"] % target["heads"]:
            raise ValueError("hidden must be divisible by heads (source "
                             f"{src['hidden']}/{src['heads']}, target "
                             f"{target['hidden']}/{target['heads']})")
        if target["hidden"] // target["heads"] < src["hidden"] // src["heads"]:
            raise ValueError(
                "head_dim would shrink (" + str(src["hidden"] // src["heads"])
                + " -> " + str(target["hidden"] // target["heads"]) + "); heads "
                "growth requires proportional hidden growth in this architecture")


def grow_checkpoint(src_path, out_path, *, layers=0, hidden=0, ffn=0, heads=0,
                    seq=0, src_heads=0, out_step=None):
    """Grow a saved checkpoint to a larger shape, preserving the function.

    Any target left at 0 keeps the source dimension. Growth only: every target
    must be >= its source. heads growth requires hidden growth in this
    architecture (head_dim = hidden/heads must not shrink), so hidden must grow
    in proportion; seq growth extends the learned position tables (patched
    trunk: multiple of the slot stride). The output checkpoint carries no
    optimizer state: continuation re-warms the optimizer. out_step overrides the
    recorded step (a grown model starts a fresh training root, typically at
    step 0). Returns a summary dict.
    """
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    sd = {k: v.clone() if isinstance(v, torch.Tensor) else v
          for k, v in ckpt["model"].items()}
    shape = checkpoint_shape_sd(sd, src_heads=src_heads)
    target = {"layers": int(layers) or shape["layers"],
              "hidden": int(hidden) or shape["hidden"],
              "ffn":    int(ffn)    or shape["ffn"],
              "heads":  int(heads)  or shape["heads"],
              "seq":    int(seq)    or shape["seq"]}
    validate_growth(shape, target, seq_multiple=seq_stride_sd(sd))
    if target["hidden"] > shape["hidden"] or target["heads"] > shape["heads"]:
        sd = _grow_width_sd(sd, target["hidden"], target["heads"], shape)
    if target["ffn"] > shape["ffn"]:
        sd = _grow_ffn_sd(sd, target["ffn"])
    if target["seq"] > shape["seq"]:
        sd = _grow_seq_sd(sd, target["seq"], shape)
    if target["layers"] > shape["layers"]:
        sd = _grow_depth_sd(sd, target["layers"], shape)

    params_before = sum(v.numel() for v in ckpt["model"].values())
    params_after  = sum(v.numel() for v in sd.values())
    step = int(ckpt.get("step", 0)) if out_step is None else int(out_step)
    state = {"model": sd, "step": step}
    if isinstance(ckpt.get("args"), dict):
        _stamp_shape(ckpt["args"], target)
        state["args"] = ckpt["args"]
    tmp = out_path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, out_path)
    return {"src": shape, "target": target,
            "params_before": params_before, "params_after": params_after,
            "out": out_path}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Function-preserving checkpoint growth: same logits, larger shape. "
                    "Output carries no optimizer state; the continue run re-warms it.")
    p.add_argument("src", help="source checkpoint .pt")
    p.add_argument("--layers", type=int, default=0, help="target global layers (0 = keep)")
    p.add_argument("--hidden", type=int, default=0, help="target hidden width (0 = keep)")
    p.add_argument("--ffn",    type=int, default=0, help="target FFN width (0 = keep)")
    p.add_argument("--heads",  type=int, default=0, help="target head count (0 = keep)")
    p.add_argument("--seq",    type=int, default=0,
                   help="target context length (0 = keep); extends the learned position tables")
    p.add_argument("--src_heads", type=int, default=0,
                   help="source head count, required only for pure-attention trunks")
    p.add_argument("--out", required=True, help="output checkpoint .pt")
    a = p.parse_args(argv)
    r = grow_checkpoint(a.src, a.out, layers=a.layers, hidden=a.hidden, ffn=a.ffn,
                        heads=a.heads, seq=a.seq, src_heads=a.src_heads)
    print("grow: " + str(r["src"]) + " -> " + str(r["target"])
          + "  params " + str(r["params_before"]) + " -> " + str(r["params_after"])
          + "  wrote " + r["out"])


if __name__ == "__main__":
    main()
