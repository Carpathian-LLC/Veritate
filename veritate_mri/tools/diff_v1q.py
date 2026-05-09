"""V1q-specific PyTorch ↔ engine layer-by-layer divergence harness.

Adapts veritate_mri/tools/diff.py to:
  - read shape from the VRMR trace header (already parametric there)
  - drive a Veritate model at any shape (V1q is 6L × 512h × 1536ffn × 8h × seq256)
  - compare both fp32 and QAT-mode PyTorch forwards against the C engine

Usage:
  python builder_v1q_diff.py \
      --exe veritate_engine/bin/macos/arm64/veritate \
      --bin models/tinystories_25m_v1q/veritate.bin \
      --checkpoint models/tinystories_25m_v1q/checkpoints/step_1500.pt \
      --prompt "Once upon a time" \
      --pos 15

Goal: find the FIRST layer where C engine residual_pre / residual_post / ffn_post
diverges from PyTorch QAT-mode reference. That layer's hot path holds the bug.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from veritate.model import Veritate
from veritate import qat as vqat

VERITATE_TRACE_MAGIC   = b"VRMR"
VERITATE_TRACE_VERSION = 8
ACTIVATION_INT8_SCALE  = 32.0


def c_trace(exe, model_bin, prompt, out_path):
    env = os.environ.copy()
    env["VERITATE_MODEL_PATH"] = os.path.abspath(model_bin)
    r = subprocess.run([exe, "trace", prompt, out_path], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"engine trace failed:\nstdout:{r.stdout}\nstderr:{r.stderr}")
    return r.stdout


def load_c_trace(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != VERITATE_TRACE_MAGIC:
            raise ValueError(f"bad magic {magic}")
        hdr = struct.unpack("<IIIIIII", f.read(28))
        version, layers, seq, hidden, ffn, heads, real_len = hdr
        if version != VERITATE_TRACE_VERSION:
            raise ValueError(f"version mismatch: {version}")

        residual_pre  = np.frombuffer(f.read(layers*seq*hidden*2), dtype=np.int16).reshape(layers, seq, hidden)
        residual_post = np.frombuffer(f.read(layers*seq*hidden*2), dtype=np.int16).reshape(layers, seq, hidden)
        ffn_neurons   = np.frombuffer(f.read(layers*seq*ffn),      dtype=np.int8).reshape(layers, seq, ffn)
        final_act     = np.frombuffer(f.read(hidden),              dtype=np.int8)
        prompt_bytes  = np.frombuffer(f.read(real_len),            dtype=np.uint8)
        f.read(5 * 8)  # 5 trace_predictions
        has_attention = struct.unpack("<B", f.read(1))[0]
        attention = None
        if has_attention:
            n = layers*heads*seq*seq
            attention = np.frombuffer(f.read(n*4), dtype=np.float32).reshape(layers, heads, seq, seq)

    return {
        "version": version, "layers": layers, "seq": seq, "hidden": hidden,
        "ffn": ffn, "heads": heads, "real_len": real_len,
        "residual_pre":  residual_pre,
        "residual_post": residual_post,
        "ffn_neurons":   ffn_neurons,
        "final_act":     final_act,
        "prompt_bytes":  prompt_bytes,
        "attention":     attention,
    }


def shape_from_config(name):
    cfg_path = os.path.join("models", name, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg["shape"]


def pytorch_trace(checkpoint, shape, prompt, real_len_pad, qat_mode):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sd = state.get("model") if isinstance(state, dict) and "model" in state else state

    m = Veritate(vocab=shape["vocab"], hidden=shape["hidden"], layers=shape["layers"],
                 ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"])
    m.load_state_dict(sd, strict=False)
    m.eval()
    if qat_mode:
        vqat.set_qat(m, True)

    prompt_bytes = prompt.encode("utf-8")
    real_len = len(prompt_bytes)
    tokens = list(prompt_bytes) + [0] * (real_len_pad - real_len)
    ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    L = shape["layers"]
    cap_in   = [None] * L
    cap_out  = [None] * L
    cap_ffn  = [None] * L      # ff.up output (pre-GELU)

    def pre_hook(i, dst):
        def h(_m, inp): dst[i] = inp[0].detach()
        return h
    def out_hook(i, dst):
        def h(_m, _i, out): dst[i] = out.detach()
        return h

    for i, blk in enumerate(m.blocks):
        blk.register_forward_pre_hook(pre_hook(i, cap_in))
        blk.register_forward_hook(out_hook(i, cap_out))
        blk.ff.up.register_forward_hook(out_hook(i, cap_ffn))

    with torch.no_grad():
        logits, _ = m(ids)

    res_pre  = np.stack([cap_in[i][0].cpu().numpy()  for i in range(L)])
    res_post = np.stack([cap_out[i][0].cpu().numpy() for i in range(L)])
    ffn_pre  = np.stack([cap_ffn[i][0].cpu().numpy() for i in range(L)])
    ffn_post = np.array([F.gelu(torch.from_numpy(ffn_pre[i])).cpu().numpy() for i in range(L)])

    return {
        "real_len": real_len,
        "logits":   logits[0].cpu().numpy(),
        "residual_pre":  res_pre,
        "residual_post": res_post,
        "ffn_pre":       ffn_pre,
        "ffn_post":      ffn_post,
    }


def cos_dist(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0 if (na > 1e-12) != (nb > 1e-12) else 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def rms(a, b):
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    if len(a) == 0: return 0.0
    return float(np.sqrt(np.mean((a - b) ** 2)))


def diff_layers(c, py, pos, scale, label):
    layers = c["layers"]
    print(f"\n=== diff vs PyTorch {label} at position {pos} (real_len={c['real_len']}) ===")
    print(f"{'L':>2} {'stage':>15} {'cos':>10} {'rms_fp':>10} {'c_norm':>10} {'py_norm':>10} {'max_abs_dfp':>12}")
    for L in range(layers):
        c_rpre = c["residual_pre"][L][pos].astype(np.float64) / scale
        py_rpre = py["residual_pre"][L][pos].astype(np.float64)
        d = c_rpre - py_rpre
        print(f"{L:>2} {'residual_pre':>15} {cos_dist(c_rpre, py_rpre):10.6f} {rms(c_rpre, py_rpre):10.6f} "
              f"{np.linalg.norm(c_rpre):10.4f} {np.linalg.norm(py_rpre):10.4f} {np.max(np.abs(d)):12.4f}")

        c_rpost = c["residual_post"][L][pos].astype(np.float64) / scale
        py_rpost = py["residual_post"][L][pos].astype(np.float64)
        d = c_rpost - py_rpost
        print(f"{L:>2} {'residual_post':>15} {cos_dist(c_rpost, py_rpost):10.6f} {rms(c_rpost, py_rpost):10.6f} "
              f"{np.linalg.norm(c_rpost):10.4f} {np.linalg.norm(py_rpost):10.4f} {np.max(np.abs(d)):12.4f}")

        c_ffn = c["ffn_neurons"][L][pos].astype(np.float64) / scale
        py_ffn = py["ffn_post"][L][pos].astype(np.float64)
        d = c_ffn - py_ffn
        print(f"{L:>2} {'ffn_post':>15} {cos_dist(c_ffn, py_ffn):10.6f} {rms(c_ffn, py_ffn):10.6f} "
              f"{np.linalg.norm(c_ffn):10.4f} {np.linalg.norm(py_ffn):10.4f} {np.max(np.abs(d)):12.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe",        required=True)
    ap.add_argument("--bin",        required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shape-from-config", default=None,
                    help="model name to look up shape from models/<name>/config.json (overrides --hidden etc.)")
    ap.add_argument("--hidden",     type=int, default=None)
    ap.add_argument("--layers",     type=int, default=None)
    ap.add_argument("--ffn",        type=int, default=None)
    ap.add_argument("--heads",      type=int, default=None)
    ap.add_argument("--seq",        type=int, default=256)
    ap.add_argument("--prompt",     default="Once upon a time")
    ap.add_argument("--out",        default="/tmp/veritate_v1q.vrmr")
    ap.add_argument("--scale",      type=float, default=ACTIVATION_INT8_SCALE)
    ap.add_argument("--pos",        type=int, default=None,
                    help="position to compare (default: real_len-1)")
    ap.add_argument("--also-fp32",  action="store_true",
                    help="also run a fp32 PyTorch comparison (no fake-quant)")
    args = ap.parse_args()

    if args.shape_from_config:
        shape = shape_from_config(args.shape_from_config)
    else:
        shape = {"vocab": 256, "hidden": args.hidden, "layers": args.layers,
                 "ffn": args.ffn, "heads": args.heads, "seq": args.seq}

    print(f"# shape: {shape}")
    print(f"# C engine trace -> {args.out}")
    out = c_trace(args.exe, args.bin, args.prompt, args.out)
    print(out.strip())

    c = load_c_trace(args.out)
    print(f"# C trace: layers={c['layers']} hidden={c['hidden']} ffn={c['ffn']} heads={c['heads']} real_len={c['real_len']}")
    pos = args.pos if args.pos is not None else c["real_len"] - 1

    print(f"# PyTorch trace (QAT mode) from {args.checkpoint}")
    py_qat = pytorch_trace(args.checkpoint, shape, args.prompt, args.seq, qat_mode=True)
    diff_layers(c, py_qat, pos=pos, scale=args.scale, label="QAT-mode")

    if args.also_fp32:
        print(f"\n# PyTorch trace (fp32 mode)")
        py_fp32 = pytorch_trace(args.checkpoint, shape, args.prompt, args.seq, qat_mode=False)
        diff_layers(c, py_fp32, pos=pos, scale=args.scale, label="fp32")

    # Compare top-5 logits
    py = py_qat
    py_logits_last = py["logits"][pos]
    top_py = np.argsort(-py_logits_last)[:5]
    print(f"\n# PyTorch QAT top-5 at pos {pos}: ",
          [(int(t), repr(chr(int(t))), float(py_logits_last[t])) for t in top_py])

    return 0


if __name__ == "__main__":
    sys.exit(main())
