# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - S63 recipe exporter: produces a compact .s63 artifact from a trained
#   canonical-Veritate checkpoint. Recipe is "salient-top50 binary base +
#   5% FP16 outlier protection" applied to every transformer-block Linear
#   weight (attn.qkv, attn.proj, ff.up, ff.down). Embeddings, RMSNorm
#   gains, lm_head, and any non-target Linear stay fp16.
# - Per-Linear payload:
#       outlier_mask     packed bits, one per weight, bit=1 means "FP16 kept"
#       outlier_values   contiguous fp16 array, length = popcount(mask)
#       binary_mask      packed bits, one per non-outlier, bit=1 means kept
#       binary_alpha     single fp16 scalar (mean(|w|) over kept-binary set)
#       binary_signs     packed bits, one per kept-binary weight
#   Total: ~1.275 bits per weight effective on the recipe's published recipe.
# - Loader reconstructs the float weight tensor; downstream the brain just
#   loads it into a regular nn.Linear. Lossy by design; matches S63 numbers
#   on the val slice (smoke verifies this).
# - This is NOT the C engine .bin format. The C engine needs v12 layout work
#   before it can consume this. The .s63 artifact is the Python-deployable
#   reference: lets a user ship a sub-2-bit model and run it through the
#   PyTorch backend at S63's measured quality.
# veritate_mri/tools/s63_export.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import io
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from veritate.model import Veritate, VOCAB_BYTE_LEVEL  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

MAGIC                = b"S63\x00"
FORMAT_VERSION       = 1
TARGET_TAGS          = ("attn.qkv", "attn.proj", "ff.up", "ff.down")
DEFAULT_OUTLIER_FRAC = 0.05    # S63 recipe
DEFAULT_SALIENT_FRAC = 0.50    # top-50% kept binary, bottom-50% zeroed

# ------------------------------------------------------------------------------------
# Functions


def _is_target_linear(name):
    return any(name.endswith(t) for t in TARGET_TAGS) and name.startswith("blocks.")


def _pack_bits(bool_tensor):
    """Flat boolean tensor -> packed uint8 array (bit 0 = element 0).
    Returns (uint8 numpy array, length-in-bits)."""
    flat = bool_tensor.flatten().to(torch.bool).numpy()
    n = flat.size
    out = np.packbits(flat, bitorder="little")
    return out, n


def _unpack_bits(packed, n_bits):
    """Inverse of _pack_bits."""
    return np.unpackbits(packed, count=n_bits, bitorder="little").astype(bool)


def _quantize_weight(W, outlier_frac=DEFAULT_OUTLIER_FRAC,
                     salient_frac=DEFAULT_SALIENT_FRAC):
    """S63 recipe per single weight tensor [out, in].
    Returns dict with packed payload + a reconstructed float tensor (for
    sanity / smoke comparison)."""
    abs_w = W.abs()
    flat_abs = abs_w.flatten()
    numel = flat_abs.numel()

    out_keep = max(1, int(round(numel * outlier_frac)))
    outlier_thr = torch.topk(flat_abs, out_keep, largest=True).values[-1]
    outlier_mask = abs_w >= outlier_thr

    # Match the S63 awq_outlier_ptq smoke EXACTLY. The base quantizer there
    # picks top-salient_frac of the FULL |W| (including outlier positions),
    # then outlier_protect overwrites the outlier positions with the original
    # W. The binary alpha is computed over the top-salient set (including
    # outliers), not over non-outliers.
    salient_keep = max(1, int(round(numel * salient_frac)))
    salient_thr = torch.topk(flat_abs, salient_keep, largest=True).values[-1]
    full_salient_mask = abs_w >= salient_thr
    salient_w_full = W * full_salient_mask.to(W.dtype)
    n_salient_full = int(full_salient_mask.sum().item())
    alpha = (salient_w_full.abs().sum() / max(1, n_salient_full)).item()

    # The stored "salient" set excludes the outlier positions (those are
    # restored separately). What we serialize as binary signs is the
    # (full_salient & ~outlier) set.
    salient_mask = full_salient_mask & (~outlier_mask)
    n_salient = int(salient_mask.sum().item())

    # Reconstruct expected float (for the smoke)
    expected = torch.zeros_like(W)
    expected = torch.where(outlier_mask, W, expected)
    expected = torch.where(salient_mask, torch.sign(W) * alpha, expected)

    return {
        "shape":         tuple(W.shape),
        "outlier_mask":  outlier_mask,
        "outlier_values": W[outlier_mask].to(torch.float16),
        "salient_mask":  salient_mask,
        "salient_signs": (W[salient_mask] > 0),
        "alpha_fp16":    float(alpha),
        "n_total":       numel,
        "n_outlier":     int(outlier_mask.sum().item()),
        "n_salient":     n_salient,
        "expected":      expected,
    }


def _bytes_per_weight(entry):
    """Effective bits per weight for one quantized layer."""
    n = entry["n_total"]
    bits = (
        entry["n_total"]            # outlier mask: 1 bit/weight
        + entry["n_outlier"] * 16   # outlier values: fp16
        + (entry["n_total"] - entry["n_outlier"])  # salient mask: 1 bit/non-outlier
        + entry["n_salient"]        # salient signs: 1 bit/salient
        + 16                        # alpha
    )
    return bits / n


def quantize_state_dict(sd, outlier_frac=DEFAULT_OUTLIER_FRAC,
                         salient_frac=DEFAULT_SALIENT_FRAC):
    """Walk a canonical-Veritate state dict; quantize all target weights.
    Returns (quant_payload, untouched_fp16, stats)."""
    quant = {}
    raw = {}
    stats = {"per_layer_bpw": [], "total_target_params": 0,
             "total_quant_bits": 0, "total_outliers": 0}
    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if name.endswith(".weight") and _is_target_linear(name.rstrip(".weight")):
            entry = _quantize_weight(tensor.detach().cpu().float(),
                                      outlier_frac=outlier_frac,
                                      salient_frac=salient_frac)
            quant[name] = entry
            stats["per_layer_bpw"].append({"name": name, "bpw": _bytes_per_weight(entry),
                                             "n": entry["n_total"]})
            stats["total_target_params"] += entry["n_total"]
            stats["total_outliers"] += entry["n_outlier"]
        else:
            raw[name] = tensor.detach().cpu().to(torch.float16)
    if stats["total_target_params"]:
        weighted = sum(e["bpw"] * e["n"] for e in stats["per_layer_bpw"])
        stats["target_bpw"] = weighted / stats["total_target_params"]
    else:
        stats["target_bpw"] = 0.0
    return quant, raw, stats


def write_s63(out_path, cfg, quant, raw, stats, recipe):
    """Serialize. Self-contained binary file.
    Header:
       MAGIC (4)
       FORMAT_VERSION (uint32 LE)
       header_len (uint32 LE)
       header_json (utf-8)
    Payload sections appended sequentially per the header's "sections" list.
    """
    payload = io.BytesIO()
    sections = []

    def _emit(name, arr_bytes):
        offset = payload.tell()
        payload.write(arr_bytes)
        sections.append({"name": name, "offset": offset, "nbytes": len(arr_bytes)})

    for name, entry in quant.items():
        packed_outlier, _ = _pack_bits(entry["outlier_mask"])
        packed_salient, _ = _pack_bits(entry["salient_mask"])
        packed_signs, _   = _pack_bits(entry["salient_signs"])
        _emit(name + ":outlier_mask",   packed_outlier.tobytes())
        _emit(name + ":outlier_values", entry["outlier_values"].numpy().tobytes())
        _emit(name + ":salient_mask",   packed_salient.tobytes())
        _emit(name + ":salient_signs",  packed_signs.tobytes())
        _emit(name + ":alpha",          struct.pack("<e", entry["alpha_fp16"]))
    for name, tensor in raw.items():
        _emit(name + ":fp16", tensor.numpy().tobytes())

    header = {
        "format":           "s63",
        "version":          FORMAT_VERSION,
        "recipe":           recipe,
        "cfg":              cfg,
        "quant_layers":     [{"name": n, "shape": list(e["shape"]),
                               "n_outlier": e["n_outlier"],
                               "n_salient": e["n_salient"],
                               "n_total":   e["n_total"]}
                              for n, e in quant.items()],
        "fp16_tensors":     [{"name": n, "shape": list(raw[n].shape)} for n in raw],
        "sections":         sections,
        "stats":            {k: v for k, v in stats.items()
                              if k not in ("per_layer_bpw",)},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload.getvalue())
    return os.path.getsize(out_path)


def read_s63(path):
    """Inverse of write_s63. Returns (cfg, state_dict) where state_dict has
    every tensor reconstructed to fp32 (so any model class can load it
    via load_state_dict)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        (version,)    = struct.unpack("<I", f.read(4))
        (header_len,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(header_len).decode("utf-8"))
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported version: {version}")
        payload_start = f.tell()
        f.seek(0, os.SEEK_END)
        payload_end = f.tell()
        f.seek(payload_start)
        payload = f.read(payload_end - payload_start)

    section_by_name = {s["name"]: (s["offset"], s["nbytes"]) for s in header["sections"]}

    def _slice(name):
        off, nb = section_by_name[name]
        return payload[off:off + nb]

    sd = {}
    for layer in header["quant_layers"]:
        name = layer["name"]
        shape = tuple(layer["shape"])
        n_total = layer["n_total"]
        n_out = layer["n_outlier"]
        n_sal = layer["n_salient"]

        outlier_mask_packed = np.frombuffer(_slice(name + ":outlier_mask"), dtype=np.uint8)
        outlier_mask = _unpack_bits(outlier_mask_packed, n_total).reshape(shape)
        outlier_values = np.frombuffer(_slice(name + ":outlier_values"),
                                        dtype=np.float16).astype(np.float32)
        salient_mask_packed = np.frombuffer(_slice(name + ":salient_mask"), dtype=np.uint8)
        salient_mask = _unpack_bits(salient_mask_packed, n_total).reshape(shape)
        signs_packed = np.frombuffer(_slice(name + ":salient_signs"), dtype=np.uint8)
        salient_signs = _unpack_bits(signs_packed, n_sal)
        alpha = struct.unpack("<e", _slice(name + ":alpha"))[0]

        W = np.zeros(shape, dtype=np.float32)
        W[outlier_mask] = outlier_values
        signed_alpha = np.where(salient_signs, alpha, -alpha).astype(np.float32)
        W[salient_mask] = signed_alpha
        sd[name] = torch.from_numpy(W)

    for fp in header["fp16_tensors"]:
        name = fp["name"]
        shape = tuple(fp["shape"])
        arr = np.frombuffer(_slice(name + ":fp16"), dtype=np.float16).astype(np.float32)
        sd[name] = torch.from_numpy(arr.reshape(shape))

    return header["cfg"], sd, header


def export_canonical_veritate_ckpt(ckpt_path, out_path,
                                    outlier_frac=DEFAULT_OUTLIER_FRAC,
                                    salient_frac=DEFAULT_SALIENT_FRAC):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config")
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    if cfg is None or not isinstance(sd, dict):
        raise ValueError(f"bad ckpt format: {ckpt_path}")
    # Guard: only canonical Veritate (has pos_emb, no rope/mtp)
    if "pos_emb.weight" not in sd:
        raise ValueError("s63 exporter currently only supports canonical Veritate "
                         "(needs pos_emb.weight). RoPE/MTP variants unsupported.")
    if any("rope" in k.lower() for k in sd) or any(k.startswith("mtp.") for k in sd):
        raise ValueError("s63 exporter currently only supports canonical Veritate "
                         "(no rope or mtp keys). Use the .pt directly for other variants.")
    quant, raw, stats = quantize_state_dict(sd, outlier_frac=outlier_frac,
                                              salient_frac=salient_frac)
    nbytes = write_s63(out_path, cfg, quant, raw, stats, recipe={
        "outlier_frac": outlier_frac,
        "salient_frac": salient_frac,
    })
    return {
        "path":             out_path,
        "bytes":            nbytes,
        "target_bpw":       stats["target_bpw"],
        "target_params":    stats["total_target_params"],
        "outlier_params":   stats["total_outliers"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="S63 recipe exporter")
    ap.add_argument("--in_ckpt",       required=True, help="Source .pt checkpoint")
    ap.add_argument("--out_s63",       required=True, help="Destination .s63 path")
    ap.add_argument("--outlier_frac",  type=float, default=DEFAULT_OUTLIER_FRAC)
    ap.add_argument("--salient_frac",  type=float, default=DEFAULT_SALIENT_FRAC)
    args = ap.parse_args()
    info = export_canonical_veritate_ckpt(args.in_ckpt, args.out_s63,
                                            outlier_frac=args.outlier_frac,
                                            salient_frac=args.salient_frac)
    print(json.dumps(info, indent=2))
