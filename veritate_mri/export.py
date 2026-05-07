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

VERITATE_MODEL_MAGIC          = b"VRTE"
VERITATE_MODEL_VERSION        = 9
VERITATE_MODEL_VERSION_TERNARY = 10
HEADER_FMT                    = "<4sIIIIIII"
LN_FIXED_SCALE                = 64.0
INT8_MAX                      = 127
ACT_INT8_SCALE                = 32.0
Q24_SHIFT                     = 24
ACT_BOOST_MAX                 = 4
TRITS_PER_BYTE                = 5

SHAPE_KEYS = ("vocab", "hidden", "layers", "ffn", "heads", "seq")

# ------------------------------------------------------------------------------------
# Functions


def quantize_matmul(w):
    # PyTorch nn.Linear weight is [out, in]. The engine's prep_b reads weights as
    # [in, out] = [k, n] row-major (it computes c[r,j] = sum_k a[r,k] * b[p*n+j]).
    # Convert here so the .bin holds bytes in [in, out] order and the engine's
    # load_b -> prep_b path does the right thing without any transpose at load.
    # Without this transpose, prep_b reads PyTorch's [out, in] bytes as [in, out]
    # which scrambles the matmul totally — see HOW_WE_DID_IT.md for the full story.
    arr = np.ascontiguousarray(np.asarray(w, dtype=np.float32).T)
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


# ------------------------------------------------------------------------------------
# Ternary (BitNet b1.58) quantization + 5-trits-per-byte packing.
# Mirrors veritate.qat.fake_quant_weight_ternary EXACTLY: per-tensor mean-abs
# gamma, q in {-1, 0, +1}. Output: 5-trits/byte packed bytes [n, ceil(k/5)] in
# [in, out] = [k, n] layout (transposed from PyTorch's [out, in] like int8 path)
# plus a single int32 gamma_q24 = round(gamma * ACT_INT8_SCALE * 2^24).
# Engine load path: load_b_ternary in v2 reads packed bytes, decodes into a
# {-1,0,+1}-valued INT8 buffer, and runs the existing prep_b — so the engine's
# INT8 hot path runs unchanged on ternary checkpoints. The trit-packed disk
# format gives the 5x density that matters at 5B+; the standalone NEON ternary
# kernel (compiled into v2 but unused at runtime today) is the future fast
# path that exploits that density.

def recover_ternary(w):
    """Recover (gamma_float, q_int_in_{-1,0,+1}) from a ternary-QAT-trained
    fp32 weight tensor. Returns (gamma, q) where q.shape == w.shape."""
    arr = np.asarray(w, dtype=np.float32)
    gamma = max(float(np.abs(arr).mean()), 1e-8)
    q = np.round(arr / gamma).clip(-1.0, 1.0).astype(np.int32)
    return gamma, q


def pack_trits_2d(q):
    """q: int [n, k] in {-1, 0, +1}. Returns uint8 [n, ceil(k/5)] packed
    via base-3: byte = sum_p (trit[p]+1) * 3^p for p in [0..4]."""
    n, k = q.shape
    nbytes = (k + TRITS_PER_BYTE - 1) // TRITS_PER_BYTE
    out = np.zeros((n, nbytes), dtype=np.uint8)
    pad_cols = nbytes * TRITS_PER_BYTE
    if pad_cols > k:
        q_pad = np.zeros((n, pad_cols), dtype=np.int32)
        q_pad[:, :k] = q
    else:
        q_pad = q.astype(np.int32)
    coeffs = np.array([1, 3, 9, 27, 81], dtype=np.int32)            # 3^p for p in [0..4]
    grouped = q_pad.reshape(n, nbytes, TRITS_PER_BYTE) + 1          # {-1,0,+1} -> {0,1,2}
    out_int = (grouped * coeffs).sum(axis=2)                        # [n, nbytes], values 0..242
    if (out_int < 0).any() or (out_int > 242).any():
        bad = (out_int < 0) | (out_int > 242)
        raise ValueError(f"trit pack out of range: {out_int[bad][:5]}")
    out[:] = out_int.astype(np.uint8)
    return out


def quantize_matmul_ternary(w):
    """Ternary equivalent of quantize_matmul. PyTorch [out, in] -> trit-packed
    [in, out] = [k, n] bytes + gamma_q24. Same transpose as INT8 path."""
    arr = np.ascontiguousarray(np.asarray(w, dtype=np.float32).T)   # [in, out]
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8), 1
    gamma, q = recover_ternary(arr)
    # We need pack to be row-major over [n, ceil(k/5)] where rows are output
    # columns of the matmul. Engine's load_b_ternary reads `j` as the slow
    # axis and decodes trits at columns p=0..k-1. So pack_trits_2d with
    # q.T (shape [n=out, k=in]) gives the right byte layout.
    n_out, k_in = arr.shape[1], arr.shape[0]
    q_no = q.T.copy()  # [n_out, k_in]
    packed = pack_trits_2d(q_no)
    # gamma is the per-tensor weight scale. The engine's requant convention
    # uses scale_q24 = scale * 2^24 where scale is the fp value of one int8
    # step. For ternary, one trit step = gamma, so gamma_q24 = gamma * 2^24.
    # No factor of ACT_INT8_SCALE here (that lives on the activation side).
    gamma_q24 = max(1, int(round(gamma * (1 << Q24_SHIFT))))
    return packed, gamma_q24


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


def write_block_ternary(f, sd, layer):
    """v10 ternary block: same layout as INT8 except every matmul tensor is
    trits-packed bytes followed by gamma_q24 (instead of int8 bytes + scale_q24)."""
    ln1 = quantize_layernorm(fetch(sd, f"blocks.{layer}.n1.weight"))
    qkv, qkv_g = quantize_matmul_ternary(fetch(sd, f"blocks.{layer}.attn.qkv.weight"))
    opr, opr_g = quantize_matmul_ternary(fetch(sd, f"blocks.{layer}.attn.proj.weight"))
    ln2 = quantize_layernorm(fetch(sd, f"blocks.{layer}.n2.weight"))
    ffu, ffu_g = quantize_matmul_ternary(fetch(sd, f"blocks.{layer}.ff.up.weight"))
    ffd, ffd_g = quantize_matmul_ternary(fetch(sd, f"blocks.{layer}.ff.down.weight"))

    f.write(ln1.tobytes())
    f.write(qkv.tobytes()); f.write(struct.pack("<i", qkv_g))
    f.write(opr.tobytes()); f.write(struct.pack("<i", opr_g))
    f.write(ln2.tobytes())
    f.write(ffu.tobytes()); f.write(struct.pack("<i", ffu_g))
    f.write(ffd.tobytes()); f.write(struct.pack("<i", ffd_g))


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


def export_checkpoint_ternary(name, step, out_path=None):
    """V2 ternary export. Writes a v10 .bin: VRTE magic + version 10 + shape +
    act_boost + INT8 embeddings + (per layer) ln1 + 4 ternary matmuls + ln2 +
    n_out. Mirrors export_checkpoint's structure but uses trit-packed weights.

    Requires the source checkpoint to be ternary-QAT-trained (i.e. weights
    were trained under fake_quant_weight_ternary). On a non-ternary checkpoint
    the recovery still works — every weight gets snapped to {-1, 0, +1} times
    its mean-abs — but quality will be poor without ternary-aware training.

    Loadable by the v2 engine (`veritate_engine_v2/bin/<os>/<arch>/veritate_v2`).
    The v1 engine refuses v10 (magic version mismatch).
    """
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

    n_out = quantize_layernorm(fetch(sd, "n_out.weight"))
    if n_out.shape != (shape["hidden"],):
        raise ValueError(f"n_out shape {n_out.shape} != ({shape['hidden']},)")

    if out_path is None:
        bin_dir = os.path.dirname(paths.bin_path(name))
        out_path = os.path.join(bin_dir, "veritate_v2.bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(struct.pack(
            HEADER_FMT,
            VERITATE_MODEL_MAGIC,
            VERITATE_MODEL_VERSION_TERNARY,
            shape["vocab"], shape["hidden"], shape["layers"],
            shape["ffn"], shape["heads"], shape["seq"],
        ))
        f.write(struct.pack("<i", act_boost))
        f.write(embed.tobytes())
        f.write(pos_embed.tobytes())
        for layer in range(shape["layers"]):
            write_block_ternary(f, sd, layer)
        f.write(n_out.tobytes())

    return {
        "name":      name,
        "step":      int(step),
        "path":      out_path,
        "bytes":     os.path.getsize(out_path),
        "shape":     shape,
        "act_boost": act_boost,
        "format":    "v10-ternary",
    }
