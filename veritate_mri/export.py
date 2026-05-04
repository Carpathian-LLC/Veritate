# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - quantize a PyTorch base Veritate checkpoint to a v3 INT8 veritate.bin.
# - per-tensor symmetric maxabs for matmul weights, fixed scale 64 for RMSNorm.
# - scale_q24 sentinel 0 lets the engine derive RMS-based requant at load time.
# - PTQ on a non-QAT model trades off some accuracy. acceptable as a baseline.
# veritate_mri/export.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from readers import paths

# ------------------------------------------------------------------------------------
# Constants

VERITATE_MODEL_MAGIC   = b"VRTE"
VERITATE_MODEL_VERSION = 9
HEADER_FMT             = "<4sIIIIIII"
LN_FIXED_SCALE         = 64.0
INT8_MAX               = 127
ACT_INT8_SCALE         = 32.0
Q24_SHIFT              = 24
ACT_BOOST_MAX          = 4

SHAPE_KEYS = ("vocab", "hidden", "layers", "ffn", "heads", "seq")

# ------------------------------------------------------------------------------------
# Functions


def quantize_matmul(w):
    arr = np.asarray(w, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs == 0.0:
        return np.zeros(arr.shape, dtype=np.int8), 1
    scale = max_abs / INT8_MAX
    q = np.round(arr / scale).clip(-INT8_MAX, INT8_MAX).astype(np.int8)
    scale_q24 = max(1, int(round(scale * (1 << Q24_SHIFT))))
    return q, scale_q24


def quantize_activation(w):
    arr = np.asarray(w, dtype=np.float32) * ACT_INT8_SCALE
    return np.round(arr).clip(-INT8_MAX, INT8_MAX).astype(np.int8)


def compute_act_boost(tok_w, pos_w):
    tok = np.asarray(tok_w, dtype=np.float32)
    pos = np.asarray(pos_w, dtype=np.float32)
    max_abs = float(max(np.max(np.abs(tok)) if tok.size else 0.0,
                        np.max(np.abs(pos)) if pos.size else 0.0))
    if max_abs <= 0.0:
        return 1
    target = INT8_MAX / (ACT_INT8_SCALE * max_abs)
    boost = 1
    while boost * 2 <= target and boost * 2 <= ACT_BOOST_MAX:
        boost *= 2
    return boost


def quantize_embedding(w, boost):
    arr = np.asarray(w, dtype=np.float32) * (ACT_INT8_SCALE * boost)
    return np.round(arr).clip(-INT8_MAX, INT8_MAX).astype(np.int8)


def quantize_layernorm(w):
    arr = np.asarray(w, dtype=np.float32) * LN_FIXED_SCALE
    return np.round(arr).clip(-INT8_MAX, INT8_MAX).astype(np.int8)


def load_state_dict(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if isinstance(state, dict):
        return state
    raise ValueError(f"unexpected checkpoint structure: {checkpoint_path}")


def shape_from_config(name):
    cfg_path = paths.config_path(name)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"no config.json for model: {name}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    shape = cfg.get("shape") or {}
    out = {}
    for key in SHAPE_KEYS:
        v = shape.get(key)
        if v is None:
            ta = cfg.get("training_args") or {}
            v = ta.get(key)
        if v is None:
            raise ValueError(f"config.json for {name} missing shape field: {key}")
        out[key] = int(v)
    if out["vocab"] != 256:
        raise ValueError(f"engine requires vocab=256, got {out['vocab']}")
    if out["hidden"] % out["heads"] != 0:
        raise ValueError(f"hidden ({out['hidden']}) not divisible by heads ({out['heads']})")
    return out


def fetch(sd, key):
    if key not in sd:
        raise KeyError(f"checkpoint missing key: {key}")
    return sd[key].detach().to(torch.float32).cpu().numpy()


def write_block(f, sd, layer):
    ln1 = quantize_layernorm(fetch(sd, f"blocks.{layer}.n1.weight"))
    qkv, qkv_q24 = quantize_matmul(fetch(sd, f"blocks.{layer}.attn.qkv.weight"))
    opr, opr_q24 = quantize_matmul(fetch(sd, f"blocks.{layer}.attn.proj.weight"))
    ln2 = quantize_layernorm(fetch(sd, f"blocks.{layer}.n2.weight"))
    ffu, ffu_q24 = quantize_matmul(fetch(sd, f"blocks.{layer}.ff.up.weight"))
    ffd, ffd_q24 = quantize_matmul(fetch(sd, f"blocks.{layer}.ff.down.weight"))

    f.write(ln1.tobytes())
    f.write(qkv.tobytes()); f.write(struct.pack("<i", qkv_q24))
    f.write(opr.tobytes()); f.write(struct.pack("<i", opr_q24))
    f.write(ln2.tobytes())
    f.write(ffu.tobytes()); f.write(struct.pack("<i", ffu_q24))
    f.write(ffd.tobytes()); f.write(struct.pack("<i", ffd_q24))


def export_checkpoint(name, step):
    ckpt_path = paths.checkpoint_path(name, step)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    shape = shape_from_config(name)
    sd = load_state_dict(ckpt_path)

    tok_w = fetch(sd, "tok_emb.weight")
    pos_w = fetch(sd, "pos_emb.weight")
    act_boost = compute_act_boost(tok_w, pos_w)
    embed     = quantize_embedding(tok_w, act_boost)
    pos_embed = quantize_embedding(pos_w, act_boost)

    if embed.shape != (shape["vocab"], shape["hidden"]):
        raise ValueError(f"tok_emb shape {embed.shape} != ({shape['vocab']}, {shape['hidden']})")
    if pos_embed.shape != (shape["seq"], shape["hidden"]):
        raise ValueError(f"pos_emb shape {pos_embed.shape} != ({shape['seq']}, {shape['hidden']})")

    out_path = paths.bin_path(name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_out = quantize_layernorm(fetch(sd, "n_out.weight"))
    if n_out.shape != (shape["hidden"],):
        raise ValueError(f"n_out shape {n_out.shape} != ({shape['hidden']},)")

    with open(out_path, "wb") as f:
        f.write(struct.pack(
            HEADER_FMT,
            VERITATE_MODEL_MAGIC,
            VERITATE_MODEL_VERSION,
            shape["vocab"], shape["hidden"], shape["layers"],
            shape["ffn"], shape["heads"], shape["seq"],
        ))
        f.write(struct.pack("<i", act_boost))
        f.write(embed.tobytes())
        f.write(pos_embed.tobytes())
        for layer in range(shape["layers"]):
            write_block(f, sd, layer)
        f.write(n_out.tobytes())

    return {
        "name":      name,
        "step":      int(step),
        "path":      out_path,
        "bytes":     os.path.getsize(out_path),
        "shape":     shape,
        "act_boost": act_boost,
    }
