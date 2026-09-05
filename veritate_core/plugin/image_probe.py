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
# - Each file answers one question a person asks of a picture model:
#     samples.png     what does it draw from nothing (and from captions)
#     passes.png      how does one picture form over the decode passes
#     fill.png        given half a real picture, can it complete it: original/masked/filled
#     layers.png      what would it draw if it stopped at layer 1, 2, ... L (logit lens)
#     confidence.png  the filled picture beside how sure it was, cell by cell
#     cell_loss.png   where in the frame it struggles, averaged over held-out pictures
#     nearest.png     the training picture closest to each sample: copying or inventing
#     attention.png   where one centre cell looks, per layer
#     recon.png       the codec's own reconstruction: the ceiling the model cannot beat
#     formation.png   in which decode pass each cell was decided: the order a picture forms
#     planes.png      the first sample rendered from 1, 2, ... all planes: coarse to fine,
#                     what each residual plane adds
#     metrics.json    fill accuracy per plane, loss by how much is hidden, codes in use,
#                     attention focus per layer and per head, how far attention reaches per
#                     layer (in cells), agreement and accuracy per layer (where the decision
#                     forms), residual norm per layer, calibration bins and expected
#                     calibration error, novelty per sample, the pass each cell committed in
#                     and its mean per plane, and what forms first: sharpness of the samples
#                     against the codec's own reconstructions and how well their colours
#                     match the held-out pictures
# - Everything is recovered with forward hooks and no_grad; a probe never changes the
#   model. The extra cost is one hooked forward, one confidence forward, a traced sample
#   and a hamming pass over a sample of the training bin.
# veritate_core/plugin/image_probe.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import io
import itertools
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
THUMB           = 192         # px per tile in the grids: shown at ~128 css px, crisp on a 2x display
GAP             = 4
BG              = (14, 16, 22)
MASK_GREY       = (90, 90, 90)
CALIBRATION_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0001)
COMMIT_AGREEMENT = 0.9        # a layer whose picture matches the final one this much has decided
NOVELTY_RECORDS  = 16384      # training pictures compared against, a fixed random sample
LAYER_COLS       = 8
COLOUR_BINS      = 4          # per channel, for the colour-match histogram
FILES           = ("samples.png", "passes.png", "fill.png", "layers.png", "confidence.png",
                   "cell_loss.png", "nearest.png", "recon.png", "attention.png", "formation.png",
                   "planes.png", "metrics.json")

# ------------------------------------------------------------------------------------
# Functions


def _tile(frame_u8):
    img = Image.fromarray(np.asarray(frame_u8, dtype=np.uint8), "RGB")
    return img.resize((THUMB, THUMB), Image.BILINEAR)


def _grid(tiles, cols):
    cols = max(1, min(cols, len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    out = Image.new("RGB", (cols * THUMB + (cols + 1) * GAP, rows * THUMB + (rows + 1) * GAP), BG)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        out.paste(tile, (GAP + c * (THUMB + GAP), GAP + r * (THUMB + GAP)))
    return out


def _png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _decode(codec, codes_u8, h, w, planes=None):
    codes = codec.from_bytes(bytes(np.asarray(codes_u8, dtype=np.uint8).tolist()), h, w)
    return codec.decode(codes, planes=planes).cpu().numpy()


def _formation_order(trace, code_bytes):
    """The decode pass in which each code position was committed, from a fill() trace."""
    order = np.zeros(code_bytes, dtype=np.int64)
    still_unknown = np.ones(code_bytes, dtype=bool)
    for entry in trace:
        newly = still_unknown & ~np.asarray(entry["unknown"], dtype=bool)
        order[newly] = int(entry["pass"])
        still_unknown = np.asarray(entry["unknown"], dtype=bool).copy()
    order[still_unknown] = int(trace[-1]["pass"]) if trace else 0
    return order


def _sharpness(frames):
    """Mean absolute Laplacian of the grey picture: how much fine detail it holds."""
    vals = []
    for f in frames:
        g = np.asarray(f, dtype=np.float32).mean(-1)
        lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        vals.append(float(np.abs(lap).mean()))
    return float(np.mean(vals)) if vals else None


def _colour_hist(frames):
    h = np.zeros((COLOUR_BINS,) * 3, dtype=np.float64)
    for f in frames:
        idx = (np.asarray(f, dtype=np.int64) * COLOUR_BINS) // 256
        flat = idx[..., 0] * COLOUR_BINS * COLOUR_BINS + idx[..., 1] * COLOUR_BINS + idx[..., 2]
        h += np.bincount(flat.ravel(), minlength=COLOUR_BINS ** 3).reshape(h.shape)
    return h / max(1.0, h.sum())


def _colour_match(frames, reference):
    """1 minus half the L1 distance between colour histograms: 1 is the same palette."""
    if not frames or not reference:
        return None
    return float(1.0 - 0.5 * np.abs(_colour_hist(frames) - _colour_hist(reference)).sum())


def _grid_distances(gh, gw):
    """[cell, cell] Euclidean distances between grid cells, in cells."""
    ys, xs = np.mgrid[0:gh, 0:gw]
    pts = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
    return torch.from_numpy(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1))


def _grey_cells(frame, cells, patch):
    """Paint MASK_GREY over the cells (gh x gw bool, True = grey)."""
    out = frame.copy()
    gh, gw = cells.shape
    for gy in range(gh):
        for gx in range(gw):
            if cells[gy, gx]:
                out[gy * patch:(gy + 1) * patch, gx * patch:(gx + 1) * patch] = MASK_GREY
    return out


def _heatmap(m, size=THUMB):
    norm = (m - m.min()) / max(1e-9, float(m.max() - m.min()))
    rgb = np.stack([norm * 255, norm * 120 + 20, (1 - norm) * 200], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").resize((size, size), Image.NEAREST)


def _heatmap_tiles(maps):
    return [_heatmap(m) for m in (maps or [])]


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


def _mask_batch(tokens, first, masks):
    """Tokens with the given code positions hidden, and targets that name them."""
    toks = tokens.clone()
    tg = torch.full_like(toks, image_grid.IGNORE_INDEX)
    for b in range(toks.shape[0]):
        where = np.flatnonzero(masks[b]) + first
        tg[b, where] = toks[b, where]
        toks[b, where] = image_codec.MASK_BYTE
    return toks, tg


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
    """Per-layer (and per-head) attention entropy over image positions, and the centre
    cell's map per layer."""
    captured = []
    hooks = []
    for blk in getattr(model, "blocks", []):
        attn = getattr(blk, "attn", None)
        if attn is None or not hasattr(attn, "qkv"):
            continue
        hooks.append(attn.qkv.register_forward_hook(lambda _m, _i, out, a=attn: captured.append((a, out.detach()))))
    if not hooks:
        return {"entropy_per_layer": [], "entropy_per_head": [], "distance_per_layer": [], "map": None}
    try:
        model(tokens[:1].to(device))
    finally:
        for h in hooks:
            h.remove()
    entropies, per_head, maps, distances = [], [], [], []
    cell = gh * gw
    centre = first + (gh // 2) * gw + (gw // 2)            # plane 0, centre cell
    grid_dist = None
    for attn, qkv in captured:
        B, T, _ = qkv.shape
        q, k = qkv.view(B, T, 3, attn.h, attn.d).permute(2, 0, 3, 1, 4)[:2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.d)
        probs = F.softmax(scores.float(), dim=-1)[0]                 # [heads, T, T]
        img = probs[:, first:, first:]                                # image -> image
        ent_head = -(img * (img + 1e-9).log()).sum(-1).mean(-1) / math.log(code_bytes)
        per_head.append([float(x) for x in ent_head.cpu().tolist()])
        entropies.append(float(ent_head.mean()))                     # 0 focused .. 1 uniform
        cell_map = probs[:, centre, first:first + cell].mean(0)      # over plane-0 cells
        maps.append(cell_map.reshape(gh, gw).cpu().numpy())
        # how far a plane-0 cell looks, in cells: attention-weighted grid distance to the
        # plane-0 cells it attends, over heads and cells. Local heads score ~1, global ~gh/2.
        if grid_dist is None:
            grid_dist = _grid_distances(gh, gw).to(probs.device)
        p0 = probs[:, first:first + cell, first:first + cell]
        reach = (p0 * grid_dist).sum(-1) / p0.sum(-1).clamp_min(1e-9)
        distances.append(float(reach.mean()))
    return {"entropy_per_layer": entropies, "entropy_per_head": per_head,
            "distance_per_layer": distances, "map": maps}


@torch.no_grad()
def depth_probe(model, codec, tokens, mask, first, h, w, device):
    """The logit lens for a picture: the residual after every block, projected through
    the model's own head, decoded. Shows at which layer the answer forms. Returns
    (tiles, agreement with the final layer per layer, accuracy per layer, residual norm
    per layer)."""
    blocks = getattr(model, "blocks", None)
    if blocks is None or not hasattr(model, "project_byte0"):
        return [], [], [], []
    residuals = []
    hooks = [blk.register_forward_hook(lambda _m, _i, out: residuals.append(out.detach())) for blk in blocks]
    toks, _tg = _mask_batch(tokens[:1], first, mask[None])
    try:
        out = model(toks.to(device))
    finally:
        for hk in hooks:
            hk.remove()
    final = (out[0] if isinstance(out, (tuple, list)) else out)[0, first:, :].float()
    final[:, image_codec.MASK_BYTE] = float("-inf")
    final_pred = final.argmax(-1).cpu()
    truth = tokens[0, first:]
    masked = torch.from_numpy(np.asarray(mask, dtype=bool))
    tiles, agree, acc, norms = [], [], [], []
    for res in residuals:
        logits = model.project_byte0(res)[0, first:, :].float()
        logits[:, image_codec.MASK_BYTE] = float("-inf")
        pred = logits.argmax(-1).cpu()
        codes = truth.clone()
        codes[masked] = pred[masked]
        tiles.append(_tile(_decode(codec, codes.numpy().astype(np.uint8), h, w)))
        agree.append(float((pred[masked] == final_pred[masked]).float().mean()))
        acc.append(float((pred[masked] == truth[masked]).float().mean()))
        norms.append(float(res[0, first:, :].float().norm(dim=-1).mean()))
    return tiles, agree, acc, norms


@torch.no_grad()
def confidence_probe(model, tokens, masks, first, code_bytes, planes, gh, gw, device):
    """One forward on the fill-test pictures: how sure the model is on the hidden cells,
    whether that confidence is earned (calibration), and where in the frame it loses."""
    toks, tg = _mask_batch(tokens, first, masks)
    out = model(toks.to(device))
    logits = (out[0] if isinstance(out, (tuple, list)) else out)[:, first:, :].float()
    logits[..., image_codec.MASK_BYTE] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(-1)
    truth = tg[:, first:].to(device)
    masked = truth != image_grid.IGNORE_INDEX
    correct = (pred == truth) & masked
    bins = []
    ece = 0.0
    n_masked = max(1, int(masked.sum()))
    for lo, hi in itertools.pairwise(CALIBRATION_BINS):
        sel = masked & (conf >= lo) & (conf < hi)
        n = int(sel.sum())
        if n:
            acc = float(correct[sel].float().sum() / n)
            mean_conf = float(conf[sel].mean())
            ece += abs(acc - mean_conf) * n / n_masked
        else:
            acc, mean_conf = None, None
        bins.append({"lo": lo, "hi": min(hi, 1.0), "n": n, "accuracy": acc, "confidence": mean_conf})
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), truth.reshape(-1),
                           ignore_index=image_grid.IGNORE_INDEX, reduction="none").reshape(truth.shape)
    cell = code_bytes // planes
    loss_map = np.zeros(cell)
    count_map = np.zeros(cell)
    for p in range(planes):
        sl = slice(p * cell, (p + 1) * cell)
        m = masked[:, sl].float()
        loss_map += (loss[:, sl] * m).sum(0).cpu().numpy()
        count_map += m.sum(0).cpu().numpy()
    loss_map = (loss_map / np.maximum(1.0, count_map)).reshape(gh, gw)
    ys, xs = np.mgrid[0:gh, 0:gw]
    inner = (ys >= gh // 4) & (ys < gh - gh // 4) & (xs >= gw // 4) & (xs < gw - gw // 4)
    centre = float(loss_map[inner].mean()) if inner.any() else float("nan")
    edge = float(loss_map[~inner].mean()) if (~inner).any() else float("nan")
    conf_map = conf[0, :cell].cpu().numpy().copy()
    conf_map[~masked[0, :cell].cpu().numpy()] = 1.0          # known cells: certain by construction
    return {"mean_confidence": float(conf[masked].mean()),
            "calibration": bins, "expected_calibration_error": float(ece),
            "centre_loss": centre, "edge_loss": edge,
            "centre_edge_loss_ratio": float(centre / edge) if edge and edge == edge else None,
            "loss_map": loss_map, "confidence_map": conf_map.reshape(gh, gw)}


def novelty_probe(train_path, samples, code_bytes, seed, max_records=NOVELTY_RECORDS):
    """For each sample, the nearest training picture by cell-wise hamming distance and
    the fraction of cells that differ (0 = a copy, 1 = nothing in common)."""
    if not train_path or not os.path.isfile(train_path) or not samples:
        return None
    ends = image_grid.code_block_ends(train_path)
    ends = ends[ends >= code_bytes]
    if ends.size == 0:
        return None
    rng = np.random.RandomState(seed)
    pick = ends if ends.size <= max_records else np.sort(rng.choice(ends, size=max_records, replace=False))
    arr = np.memmap(train_path, dtype=np.uint8, mode="r")
    idx = pick[:, None].astype(np.int64) - code_bytes + np.arange(code_bytes)[None, :]
    records = np.asarray(arr)[idx]
    out = []
    for codes in samples:
        dist = (records != np.asarray(codes, dtype=np.uint8)[None, :]).sum(1)
        j = int(dist.argmin())
        out.append({"novelty": float(dist[j]) / code_bytes, "nearest": records[j].copy()})
    return out


@torch.no_grad()
def dump(model, codec, geometry, name, step, val_path, device, out_dir=None, captions_from_val=True,
         train_path=None):
    """Write the image probe for one checkpoint. Returns the metrics dict."""
    from readers import paths
    out_dir = out_dir or os.path.join(paths.hook_step_dir(name, step), IMAGE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    h, w, seq, code_bytes = geometry["height"], geometry["width"], geometry["seq"], geometry["code_bytes"]
    planes, patch = codec.planes, codec.patch
    gh, gw = h // patch, w // patch
    cell = gh * gw
    first = seq - code_bytes
    was_training = model.training
    model.eval()
    codec_dev = next(codec.parameters()).device
    metrics = {"step": int(step), "height": h, "width": w, "code_bytes": code_bytes, "planes": planes,
               "layers": int(getattr(model, "layers", 0) or len(getattr(model, "blocks", []))),
               "thumb": THUMB, "gap": GAP}           # the tile grid of every png, so a viewer can crop

    # 1. samples from nothing, same seeds every checkpoint; the first one traced pass by pass
    tiles, sample_codes, sample_frames = [], [], []
    used = np.zeros(image_codec.CODEBOOK_ENTRIES, dtype=np.int64)
    for i in range(N_SAMPLES):
        win = image_sample.build_window(seq, code_bytes)
        trace = [] if i == 0 else None
        codes = image_sample.fill(model, win, first, passes=SAMPLE_PASSES, seed=SEED + i, device=device, trace=trace)
        used += np.bincount(codes, minlength=image_codec.CODEBOOK_ENTRIES)[:image_codec.CODEBOOK_ENTRIES]
        frame = _decode(codec, codes, h, w)
        tiles.append(_tile(frame))
        sample_codes.append(codes)
        sample_frames.append(frame)
        if trace:
            pass_tiles = []
            for entry in trace:
                shown = entry["codes"].copy()
                shown[entry["unknown"]] = 0                          # placeholder under the grey
                grey = entry["unknown"][:cell].reshape(gh, gw)       # a cell is unknown while plane 0 is
                pass_tiles.append(_tile(_grey_cells(_decode(codec, shown, h, w), grey, patch)))
            _grid(pass_tiles, len(pass_tiles)).save(os.path.join(out_dir, "passes.png"))
            metrics["pass_committed"] = [e["committed"] for e in trace]
            metrics["pass_confidence"] = [e["confidence"] for e in trace]
            # the order the picture formed: which pass decided each cell (plane 0), and how
            # early each plane commits on average -- structure should go first, detail last
            order = _formation_order(trace, code_bytes)
            _heatmap(order[:cell].reshape(gh, gw).astype(np.float64), THUMB * 2).save(
                os.path.join(out_dir, "formation.png"))
            metrics["commit_pass_map"] = [int(v) for v in order[:cell]]
            metrics["commit_pass_per_plane"] = [float(order[p * cell:(p + 1) * cell].mean()) for p in range(planes)]
            metrics["formation_passes"] = len(trace)
        if i == 0:
            # coarse to fine: the same codes rendered from 1, 2, ... all planes
            _grid([_tile(_decode(codec, codes, h, w, planes=k)) for k in range(1, planes + 1)], planes).save(
                os.path.join(out_dir, "planes.png"))
    metrics["codes_used"] = int((used > 0).sum())
    metrics["codes_used_fraction"] = float((used > 0).mean())
    # the same-seed samples' codes, so the tab can see which cells changed since the last
    # checkpoint (churn) -- what is still being learned, region by region
    metrics["sample_codes_b64"] = base64.b64encode(np.stack(sample_codes).astype(np.uint8).tobytes()).decode("ascii")
    p = used / max(1, used.sum())
    metrics["code_entropy_bits"] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())

    # 2. the fill test on real pictures, and everything measured on that same batch
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
            masks, filled_frames, originals = [], [], []
            for b in range(tokens.shape[0]):
                original = tokens[b, first:].numpy().astype(np.uint8)
                keep_cells = rng.uniform(size=(gh, gw)) >= FILL_RATIO
                keep = image_sample.cells_to_positions(keep_cells, planes)
                masks.append(~np.asarray(keep, dtype=bool))
                win = image_sample.build_window(seq, code_bytes, captions[b].encode("utf-8"), original, keep)
                filled = image_sample.fill(model, win, first, keep, passes=SAMPLE_PASSES, seed=SEED + b, device=device)
                orig_img = _decode(codec, original, h, w)
                filled_img = _decode(codec, filled, h, w)
                fill_rows += [_tile(orig_img), _tile(_grey_cells(orig_img, ~keep_cells, patch)), _tile(filled_img)]
                recon_tiles.append(_tile(orig_img))
                filled_frames.append(filled_img)
                originals.append(orig_img)
            metrics.update(masked_metrics(model, tokens, first, code_bytes, planes, device, SEED))
            # what forms first: detail (sharpness against the codec's own reconstructions,
            # which is the most detail a sample can have) and palette (colour histogram match)
            metrics["sample_sharpness"] = _sharpness(sample_frames)
            metrics["heldout_sharpness"] = _sharpness(originals)
            metrics["detail_ratio"] = (float(metrics["sample_sharpness"] / metrics["heldout_sharpness"])
                                       if metrics["heldout_sharpness"] else None)
            metrics["colour_match"] = _colour_match(sample_frames, originals)

            att = attention_probe(model, tokens, first, code_bytes, gh, gw, device)
            metrics["attention_entropy_per_layer"] = att["entropy_per_layer"]
            metrics["attention_entropy_per_head"] = att["entropy_per_head"]
            metrics["attention_distance_per_layer"] = att["distance_per_layer"]
            att_tiles = _heatmap_tiles(att["map"])
            if att_tiles:
                _grid(att_tiles, min(LAYER_COLS, len(att_tiles))).save(os.path.join(out_dir, "attention.png"))

            layer_tiles, agree, acc, norms = depth_probe(model, codec, tokens, masks[0], first, h, w, device)
            if layer_tiles:
                _grid(layer_tiles, min(LAYER_COLS, len(layer_tiles))).save(os.path.join(out_dir, "layers.png"))
                metrics["lens_agreement_per_layer"] = agree
                metrics["lens_accuracy_per_layer"] = acc
                metrics["residual_norm_per_layer"] = norms
                metrics["commit_layer"] = next((i + 1 for i, a in enumerate(agree) if a >= COMMIT_AGREEMENT), None)

            conf = confidence_probe(model, tokens, np.stack(masks), first, code_bytes, planes, gh, gw, device)
            _grid([_tile(originals[0]), _tile(_grey_cells(originals[0], masks[0][:cell].reshape(gh, gw), patch)),
                   _tile(filled_frames[0]), _heatmap(conf["confidence_map"])], 4).save(
                os.path.join(out_dir, "confidence.png"))
            _heatmap(conf["loss_map"], THUMB * 2).save(os.path.join(out_dir, "cell_loss.png"))
            for k in ("mean_confidence", "calibration", "expected_calibration_error",
                      "centre_loss", "edge_loss", "centre_edge_loss_ratio"):
                metrics[k] = conf[k]
            metrics["loss_map"] = [round(float(v), 4) for v in conf["loss_map"].ravel()]
            metrics["confidence_map"] = [round(float(v), 4) for v in conf["confidence_map"].ravel()]
            metrics["grid"] = [int(gh), int(gw)]
        except (ValueError, RuntimeError) as e:
            metrics["fill_error"] = type(e).__name__ + ": " + str(e)
    metrics["val_records"] = int(val_count)

    # 3. copying or inventing: the nearest training picture to each sample
    try:
        near = novelty_probe(train_path, sample_codes, code_bytes, SEED)
    except (ValueError, OSError) as e:
        near = None
        metrics["novelty_error"] = type(e).__name__ + ": " + str(e)
    if near:
        _grid([_tile(_decode(codec, n["nearest"], h, w)) for n in near], N_SAMPLES).save(
            os.path.join(out_dir, "nearest.png"))
        metrics["novelty_per_sample"] = [n["novelty"] for n in near]
        metrics["novelty_mean"] = float(np.mean([n["novelty"] for n in near]))

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
