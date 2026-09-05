# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - What an image model is doing, written at every checkpoint into
#   hooks/step_<N>/image/. The text probes (grammar, reading, chat health) mean nothing
#   for a model whose output is pixels; this is their replacement, and the Models tab
#   reads it to show a picture model forming.
# - Same seeds at every checkpoint, so the sample grid is the SAME draw evolving over
#   training rather than a new roll each time -- that is what makes the timeline legible.
# - Five things, each a question a person asks of a picture model:
#     samples.png   what does it draw from nothing (unconditional) and from captions
#     fill.png      given half a real picture, can it complete it: original / masked / filled
#     metrics.json  how accurate the fill is per plane (coarse structure vs fine detail),
#                   how loss depends on how much is hidden, how many codes it uses
#                   (collapse shows as a handful), how focused attention is per layer
#     attention.png where one centre cell looks, per layer -- how it thinks
#     recon.png     the codec's own reconstruction of the same pictures: the ceiling the
#                   model cannot beat, so a bad picture is attributed to the right stage
# - Attention is recovered from the qkv projection with a forward hook; the model's
#   fused attention exposes no weights, and a probe must not change the model.
# veritate_core/plugin/image_probe.py
# ------------------------------------------------------------------------------------
# Imports:

import io
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from veritate_core.plugin import image_codec, image_grid, image_sample

# ------------------------------------------------------------------------------------
# Constants

IMAGE_DIR       = "image"
N_SAMPLES       = 8
N_FILL          = 4
FILL_RATIO      = 0.5
RATIOS          = (0.25, 0.5, 0.75, 1.0)
SAMPLE_PASSES   = 8
EVAL_BATCHES    = 2
EVAL_BATCH      = 8
SEED            = 1234
THUMB           = 96          # px per tile in the grids
GAP             = 4
BG              = (14, 16, 22)
MASK_GREY       = (90, 90, 90)
FILES           = ("samples.png", "fill.png", "recon.png", "attention.png", "metrics.json")

# ------------------------------------------------------------------------------------
# Functions


def _tile(frame_u8):
    """uint8 [H, W, 3] -> PIL thumbnail."""
    img = Image.fromarray(np.asarray(frame_u8), "RGB")
    img.thumbnail((THUMB, THUMB))
    return img


def _grid(tiles, cols):
    """Tiles into one PNG, row-major, `cols` wide."""
    if not tiles:
        return None
    rows = math.ceil(len(tiles) / cols)
    w = cols * THUMB + (cols + 1) * GAP
    h = rows * THUMB + (rows + 1) * GAP
    canvas = Image.new("RGB", (w, h), BG)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = GAP + c * (THUMB + GAP) + (THUMB - tile.size[0]) // 2
        y = GAP + r * (THUMB + GAP) + (THUMB - tile.size[1]) // 2
        canvas.paste(tile, (x, y))
    return canvas


def _png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _decode(codec, codes_u8, h, w):
    return codec.decode(codec.from_bytes(bytes(np.asarray(codes_u8, dtype=np.uint8).tolist()), h, w)).cpu().numpy()


def _val_records(val_path, seq, code_bytes, n, seed):
    """n val windows (tokens with the image unmasked) and their captions."""
    draw, count = image_grid.make_record_loader(val_path, seq, n, code_bytes, image_codec.MASK_BYTE, seed)
    tokens, targets = draw()
    first = seq - code_bytes
    # the loader masked some positions; restore them from the targets so the record is whole
    image = tokens[:, first:].clone()
    tg = targets[:, first:]
    keep = tg != image_grid.IGNORE_INDEX
    image[keep] = tg[keep]
    tokens[:, first:] = image
    captions = []
    for b in range(tokens.shape[0]):
        prefix = bytes(int(x) for x in tokens[b, :first].tolist())
        cap = prefix.rsplit(image_grid.RECORD_SEP, 1)[-1].lstrip(bytes([image_grid.PAD_BYTE]))
        captions.append(cap.decode("utf-8", "replace").strip())
    return tokens, captions, count


@torch.no_grad()
def masked_metrics(model, tokens, first, code_bytes, planes, device, seed):
    """Fill accuracy per plane and loss per hidden-fraction, one batch."""
    rng = np.random.RandomState(seed)
    per_plane_hits = np.zeros(planes)
    per_plane_n = np.zeros(planes)
    by_ratio = {}
    cell = code_bytes // planes
    for ratio in RATIOS:
        toks = tokens.clone()
        n_mask = max(1, round(ratio * code_bytes))
        tg = torch.full_like(toks, image_grid.IGNORE_INDEX)
        for b in range(toks.shape[0]):
            where = rng.choice(code_bytes, size=n_mask, replace=False) + first
            tg[b, where] = toks[b, where]
            toks[b, where] = image_codec.MASK_BYTE
        out = model(toks.to(device), tg.to(device))
        logits, loss = (out[0], out[1]) if isinstance(out, (tuple, list)) else (out, None)
        by_ratio[str(ratio)] = float(loss) if loss is not None else None
        pred = logits[:, first:, :].argmax(-1).cpu()
        truth = tg[:, first:]
        masked = truth != image_grid.IGNORE_INDEX
        if ratio == FILL_RATIO:
            hit = (pred == truth) & masked
            for p in range(planes):
                sl = slice(p * cell, (p + 1) * cell)
                per_plane_hits[p] += float(hit[:, sl].sum())
                per_plane_n[p] += float(masked[:, sl].sum())
    acc = [float(h / n) if n else None for h, n in zip(per_plane_hits, per_plane_n, strict=True)]
    return {"fill_accuracy_per_plane": acc,
            "fill_accuracy": float(per_plane_hits.sum() / max(1.0, per_plane_n.sum())),
            "loss_by_hidden_fraction": by_ratio}


@torch.no_grad()
def attention_probe(model, tokens, first, code_bytes, gh, gw, device):
    """Per-layer attention entropy over image positions and the centre cell's map."""
    captured = []
    hooks = []
    for blk in getattr(model, "blocks", []):
        attn = getattr(blk, "attn", None)
        if attn is None or not hasattr(attn, "qkv"):
            continue
        hooks.append(attn.qkv.register_forward_hook(lambda _m, _i, out, a=attn: captured.append((a, out.detach()))))
    if not hooks:
        return {"entropy_per_layer": [], "map": None}
    try:
        model(tokens[:1].to(device))
    finally:
        for h in hooks:
            h.remove()
    entropies, maps = [], []
    centre = first + (gh // 2) * gw + (gw // 2)            # plane 0, centre cell
    for attn, qkv in captured:
        B, T, _ = qkv.shape
        q, k = qkv.view(B, T, 3, attn.h, attn.d).permute(2, 0, 3, 1, 4)[:2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.d)
        probs = F.softmax(scores.float(), dim=-1)[0]                 # [heads, T, T]
        img = probs[:, first:, first:]                                # image -> image
        ent = -(img * (img + 1e-9).log()).sum(-1).mean()
        entropies.append(float(ent / math.log(code_bytes)))          # 0 focused .. 1 uniform
        cell_map = probs[:, centre, first:first + gh * gw].mean(0)   # over plane-0 cells
        maps.append(cell_map.reshape(gh, gw).cpu().numpy())
    return {"entropy_per_layer": entropies, "map": maps}


def _heatmap_tiles(maps):
    tiles = []
    for m in maps or []:
        norm = (m - m.min()) / max(1e-9, float(m.max() - m.min()))
        rgb = np.stack([norm * 255, norm * 120 + 20, (1 - norm) * 200], axis=-1).astype(np.uint8)
        tiles.append(Image.fromarray(rgb, "RGB").resize((THUMB, THUMB), Image.NEAREST))
    return tiles


@torch.no_grad()
def dump(model, codec, geometry, name, step, val_path, device, out_dir=None, captions_from_val=True):
    """Write the image probe for one checkpoint. Returns the metrics dict."""
    from readers import paths
    out_dir = out_dir or os.path.join(paths.hook_step_dir(name, step), IMAGE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    h, w, seq, code_bytes = geometry["height"], geometry["width"], geometry["seq"], geometry["code_bytes"]
    planes, patch = codec.planes, codec.patch
    gh, gw = h // patch, w // patch
    first = seq - code_bytes
    was_training = model.training
    model.eval()
    codec_dev = next(codec.parameters()).device
    metrics = {"step": int(step), "height": h, "width": w, "code_bytes": code_bytes, "planes": planes}

    # 1. samples from nothing, same seeds every checkpoint
    tiles, used = [], np.zeros(image_codec.CODEBOOK_ENTRIES, dtype=np.int64)
    for i in range(N_SAMPLES):
        win = image_sample.build_window(seq, code_bytes)
        codes = image_sample.fill(model, win, first, passes=SAMPLE_PASSES, seed=SEED + i, device=device)
        used += np.bincount(codes, minlength=image_codec.CODEBOOK_ENTRIES)[:image_codec.CODEBOOK_ENTRIES]
        tiles.append(_tile(_decode(codec, codes, h, w)))
    metrics["codes_used"] = int((used > 0).sum())
    metrics["codes_used_fraction"] = float((used > 0).mean())
    p = used / max(1, used.sum())
    metrics["code_entropy_bits"] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    # 2. fill test on real pictures + the codec's own ceiling
    fill_rows, recon_tiles, val_count = [], [], 0
    if val_path and os.path.isfile(val_path):
        try:
            tokens, captions, val_count = _val_records(val_path, seq, code_bytes, N_FILL, SEED)
            if captions_from_val:
                for i, cap in enumerate(captions[:N_SAMPLES]):
                    if not cap:
                        continue
                    win = image_sample.build_window(seq, code_bytes, cap.encode("utf-8"))
                    codes = image_sample.fill(model, win, first, passes=SAMPLE_PASSES,
                                              seed=SEED + 100 + i, device=device)
                    tiles.append(_tile(_decode(codec, codes, h, w)))
                metrics["caption_samples"] = [c for c in captions[:N_SAMPLES] if c]
            rng = np.random.RandomState(SEED)
            for b in range(tokens.shape[0]):
                original = tokens[b, first:].numpy().astype(np.uint8)
                keep_cells = rng.uniform(size=(gh, gw)) >= FILL_RATIO
                keep = image_sample.cells_to_positions(keep_cells, planes)
                win = image_sample.build_window(seq, code_bytes, captions[b].encode("utf-8"), original, keep)
                filled = image_sample.fill(model, win, first, keep, passes=SAMPLE_PASSES, seed=SEED + b, device=device)
                orig_img = _decode(codec, original, h, w)
                masked_img = orig_img.copy()
                for gy in range(gh):
                    for gx in range(gw):
                        if not keep_cells[gy, gx]:
                            masked_img[gy * patch:(gy + 1) * patch, gx * patch:(gx + 1) * patch] = MASK_GREY
                fill_rows += [_tile(orig_img), _tile(masked_img), _tile(_decode(codec, filled, h, w))]
                recon_tiles.append(_tile(orig_img))
            metrics.update(masked_metrics(model, tokens, first, code_bytes, planes, device, SEED))
            att = attention_probe(model, tokens, first, code_bytes, gh, gw, device)
            metrics["attention_entropy_per_layer"] = att["entropy_per_layer"]
            att_tiles = _heatmap_tiles(att["map"])
            if att_tiles:
                _grid(att_tiles, min(8, len(att_tiles))).save(os.path.join(out_dir, "attention.png"))
        except (ValueError, RuntimeError) as e:
            metrics["fill_error"] = type(e).__name__ + ": " + str(e)
    metrics["val_records"] = int(val_count)

    _grid(tiles, N_SAMPLES).save(os.path.join(out_dir, "samples.png"))
    if fill_rows:
        _grid(fill_rows, 3).save(os.path.join(out_dir, "fill.png"))
        _grid(recon_tiles, N_FILL).save(os.path.join(out_dir, "recon.png"))
    metrics["seconds"] = round(time.time() - started, 2)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=1)
    if was_training:
        model.train()
    codec.to(codec_dev)
    return metrics


def read(name):
    """Every step's metrics for a model, oldest first, plus which files exist."""
    from readers import paths
    root = paths.hooks_dir(name)
    out = []
    if not os.path.isdir(root):
        return out
    for entry in sorted(os.listdir(root), key=lambda e: int(e.split("_")[-1]) if e.split("_")[-1].isdigit() else -1):
        d = os.path.join(root, entry, IMAGE_DIR)
        mp = os.path.join(d, "metrics.json")
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp, encoding="utf-8") as handle:
                m = json.load(handle)
        except (OSError, ValueError):
            continue
        m["files"] = [f for f in FILES if os.path.isfile(os.path.join(d, f))]
        out.append(m)
    return out
