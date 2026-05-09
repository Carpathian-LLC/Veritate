# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - per-layer C-vs-PyTorch divergence. loads the VRMR trace from veritate.exe trace
#   and runs the same prompt through the pytorch checkpoint with hooks.
# - emits per-layer cosine distance / rms / max-abs at residual_pre, residual_post,
#   ffn_neurons (post-gelu).
# ------------------------------------------------------------------------------------

import argparse
import os
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))
from veritate.model import Veritate

VERITATE_TRACE_MAGIC   = b"VRMR"
VERITATE_TRACE_VERSION = 8

# matches engine/src/veritate.h
V_VOCAB  = 256
V_SEQ    = 256
V_HIDDEN = 768
V_HEADS  = 12
V_FFN    = 3072
V_LAYERS = 12
HEAD_DIM = V_HIDDEN // V_HEADS

ACTIVATION_INT8_SCALE = 32.0


def c_trace(exe, model_bin, prompt, out_path):
    env = os.environ.copy()
    env["VERITATE_MODEL_PATH"] = model_bin
    r = subprocess.run([exe, "trace", prompt, out_path], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"veritate.exe trace failed: {r.stderr}")
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
        # 5 trace_predictions @ 8 bytes each
        top_preds_raw = f.read(5 * 8)
        has_attention = struct.unpack("<B", f.read(1))[0]
        attention = None
        if has_attention:
            n = layers*heads*seq*seq
            attention = np.frombuffer(f.read(n*4), dtype=np.float32).reshape(layers, heads, seq, seq)
        has_lens = struct.unpack("<B", f.read(1))[0]
        lens_logits = None
        if has_lens:
            n = layers*seq*V_VOCAB
            lens_logits = np.frombuffer(f.read(n*4), dtype=np.int32).reshape(layers, seq, V_VOCAB)

    return {
        "real_len": real_len,
        "residual_pre":  residual_pre,
        "residual_post": residual_post,
        "ffn_neurons":   ffn_neurons,
        "final_act":     final_act,
        "prompt_bytes":  prompt_bytes,
        "attention":     attention,
        "lens_logits":   lens_logits,
    }


def pytorch_trace(checkpoint, prompt, real_len_pad=V_SEQ):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = dict(state.get("args", {}))
    sd = state["model"]
    del state  # drops optimizer state (~8 GB on 1B) before model construction
    model = Veritate(
        vocab=cfg.get("vocab", 256), hidden=cfg.get("hidden", 768),
        layers=cfg.get("layers", 12), ffn=cfg.get("ffn", 3072),
        heads=cfg.get("heads", 12), seq=cfg.get("seq", 256),
    )
    model.load_state_dict(sd, strict=False)
    del sd
    model.eval()

    prompt_bytes = prompt.encode("utf-8")
    real_len = len(prompt_bytes)
    tokens = list(prompt_bytes) + [0] * (real_len_pad - real_len)
    ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    cap_in   = [None] * model.layers
    cap_out  = [None] * model.layers
    cap_ffn  = [None] * model.layers   # pre-GELU
    cap_qkv  = [None] * model.layers
    cap_h_n1 = [None] * model.layers   # output of n1 (RMSNorm before attention)
    cap_h_n2 = [None] * model.layers   # output of n2 (RMSNorm before FFN)

    def pre_hook(L, dst):
        def h(_m, inp): dst[L] = inp[0].detach()
        return h
    def out_hook(L, dst):
        def h(_m, _i, out): dst[L] = out.detach()
        return h

    for L, blk in enumerate(model.blocks):
        blk.register_forward_pre_hook(pre_hook(L, cap_in))
        blk.register_forward_hook(out_hook(L, cap_out))
        blk.ff.up.register_forward_hook(out_hook(L, cap_ffn))
        blk.attn.qkv.register_forward_hook(out_hook(L, cap_qkv))
        blk.n1.register_forward_hook(out_hook(L, cap_h_n1))
        blk.n2.register_forward_hook(out_hook(L, cap_h_n2))

    with torch.no_grad():
        logits, _ = model(ids)

    # collect arrays per-layer at all positions [seq, ...]
    res_in   = np.stack([cap_in[L][0].cpu().numpy()   for L in range(model.layers)])  # [L, seq, hidden]
    res_out  = np.stack([cap_out[L][0].cpu().numpy()  for L in range(model.layers)])  # [L, seq, hidden]
    ffn_pre  = np.stack([cap_ffn[L][0].cpu().numpy()  for L in range(model.layers)])  # [L, seq, ffn]
    ffn_post = np.array([F.gelu(torch.from_numpy(ffn_pre[L])).cpu().numpy() for L in range(model.layers)])
    qkv      = np.stack([cap_qkv[L][0].cpu().numpy()  for L in range(model.layers)])
    h_n1     = np.stack([cap_h_n1[L][0].cpu().numpy() for L in range(model.layers)])
    h_n2     = np.stack([cap_h_n2[L][0].cpu().numpy() for L in range(model.layers)])

    return {
        "model": model,
        "real_len": real_len,
        "tokens": tokens,
        "logits": logits[0].cpu().numpy(),  # [seq, vocab]
        "residual_pre":  res_in,
        "residual_post": res_out,
        "ffn_pre":       ffn_pre,
        "ffn_post":      ffn_post,
        "qkv":           qkv,
        "h_n1":          h_n1,
        "h_n2":          h_n2,
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


def diff_traces(c, py, pos=None, scale=ACTIVATION_INT8_SCALE):
    """C ints are at activation scale 32 (so fp = int / 32). Compare per-layer."""
    real_len = c["real_len"]
    if pos is None:
        pos = real_len - 1   # last real position
    print(f"# diff at position {pos} (real_len={real_len})")
    print(f"# {'layer':>6} {'stage':>14} {'cos_dist':>12} {'rms_fp':>12} {'c_norm':>10} {'py_norm':>10} {'max_abs_diff_fp':>16}")

    for L in range(V_LAYERS):
        # residual_pre: C int16 -> fp by /scale; PyTorch fp32
        c_rpre = c["residual_pre"][L][pos].astype(np.float64) / scale
        py_rpre = py["residual_pre"][L][pos].astype(np.float64)
        diff = c_rpre - py_rpre
        print(f"  L{L:02d}    residual_pre  {cos_dist(c_rpre, py_rpre):12.6f} {rms(c_rpre, py_rpre):12.6f} "
              f"{np.linalg.norm(c_rpre):10.4f} {np.linalg.norm(py_rpre):10.4f} {np.max(np.abs(diff)):16.4f}")

        c_rpost = c["residual_post"][L][pos].astype(np.float64) / scale
        py_rpost = py["residual_post"][L][pos].astype(np.float64)
        diff = c_rpost - py_rpost
        print(f"  L{L:02d}   residual_post  {cos_dist(c_rpost, py_rpost):12.6f} {rms(c_rpost, py_rpost):12.6f} "
              f"{np.linalg.norm(c_rpost):10.4f} {np.linalg.norm(py_rpost):10.4f} {np.max(np.abs(diff)):16.4f}")

        # ffn_neurons: C int8 post-GELU -> fp by /scale; PyTorch fp32 post-GELU
        c_ffn = c["ffn_neurons"][L][pos].astype(np.float64) / scale
        py_ffn = py["ffn_post"][L][pos].astype(np.float64)
        diff = c_ffn - py_ffn
        print(f"  L{L:02d}    ffn_post     {cos_dist(c_ffn, py_ffn):12.6f} {rms(c_ffn, py_ffn):12.6f} "
              f"{np.linalg.norm(c_ffn):10.4f} {np.linalg.norm(py_ffn):10.4f} {np.max(np.abs(diff)):16.4f}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(os.environ.get("LOCALAPPDATA", ""), "veritate", "veritate.exe"))
    ap.add_argument("--bin",       default="models/tinystories-80m-fp32/veritate.bin")
    ap.add_argument("--checkpoint",default="models/tinystories-80m-fp32/checkpoints/step_45000.pt")
    ap.add_argument("--prompt",    default="Once upon a time")
    ap.add_argument("--out",       default="c_trace.bin")
    ap.add_argument("--scale",     type=float, default=ACTIVATION_INT8_SCALE)
    ap.add_argument("--pos",       type=int, default=None)
    args = ap.parse_args()

    print(f"# C trace -> {args.out}")
    out = c_trace(args.exe, args.bin, args.prompt, args.out)
    print(out.strip())

    print(f"# pytorch trace from {args.checkpoint}")
    py = pytorch_trace(args.checkpoint, args.prompt)

    print(f"# loading C trace")
    c = load_c_trace(args.out)
    assert c["real_len"] == py["real_len"], f"real_len mismatch: c={c['real_len']} py={py['real_len']}"

    diff_traces(c, py, pos=args.pos, scale=args.scale)


if __name__ == "__main__":
    main()
