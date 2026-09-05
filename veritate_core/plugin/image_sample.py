# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Generation for image models (IDEA 24). One mechanism, every mode: the model was
#   trained to fill masked cells of an encoded picture given the bytes before it, so
#   text-to-image is "all cells masked, caption in front", image-to-image is "encode the
#   source, mask some cells, fill", inpainting masks a region, expanding shrinks the
#   source into the canvas and masks the border, unconditional masks everything with
#   no caption. No second model, no second objective.
# - Decoding is MaskGIT's parallel refinement: every masked position is predicted at
#   once, the most confident commit, the rest stay masked, and the number still masked
#   follows the cosine schedule the trainer sampled from -- so generation walks the
#   training distribution backwards. Passes are a knob (F2 says 4 should match causal
#   AR; more than 8 fails the CPU budget). One forward per pass, not one per byte.
# - The window is built exactly as training saw it: pad, RECORD_SEP, caption, image, so
#   a caption is preceded by the separator the corpus put there. MASK_BYTE can never be
#   emitted: its logit is -inf. Cells are kept or masked across ALL planes at once, so
#   kept content stays coherent (plane-major layout: index = p*gh*gw + gy*gw + gx).
# - Output resolution is the model's training frame. "Expand" therefore means: the
#   source occupies the centre and the model paints the margin, at the fixed frame.
# veritate_core/plugin/image_sample.py
# ------------------------------------------------------------------------------------
# Imports:

import io
import json
import math

import numpy as np
import torch
from PIL import Image

from veritate_core.plugin import image_codec, image_grid
from veritate_core.plugin.image_decode import RGB

# ------------------------------------------------------------------------------------
# Constants

MODES            = ("text", "variation", "inpaint", "expand", "unconditional")
SOURCE_MODES     = ("variation", "inpaint", "expand")
DEFAULT_PASSES   = 8
MAX_PASSES       = 64
DEFAULT_STRENGTH = 0.6      # variation: share of cells regenerated
DEFAULT_EXPAND   = 0.6      # expand: side of the inner box the source occupies
RGB_MAX          = 255.0
TRAINING_IMAGE   = "image"

# ------------------------------------------------------------------------------------
# Functions


def load_image_model(name, step=None):
    """(model, codec, geometry, step) for a model dir whose config says training: image.
    The model is built bidirectional from the config shape and loaded strictly: a
    checkpoint that does not fit the shape is an error, never a silent partial load."""
    from readers import checkpoints, paths

    from veritate_core.model import Veritate
    with open(paths.config_path(name), encoding="utf-8") as handle:
        cfg = json.load(handle)
    if cfg.get("training") != TRAINING_IMAGE:
        raise ValueError(name + " is not an image model (training=" + str(cfg.get("training") or "text") + ")")
    ta, shape = cfg["training_args"], cfg["shape"]
    step = int(step) if step else checkpoints.latest_step(name)
    if not step:
        raise ValueError("no checkpoint for " + name)
    model = Veritate(vocab=int(shape["vocab"]), hidden=int(shape["hidden"]), layers=int(shape["layers"]),
                     ffn=int(shape["ffn"]), heads=int(shape["heads"]), seq=int(shape["seq"]), causal=False)
    ckpt = torch.load(paths.checkpoint_path(name, step), map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    if any(k.startswith("base.") for k in sd):
        sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
    model.load_state_dict(sd, strict=True)
    model.eval()
    codec = image_codec.load(paths.codec_path(ta["codec"]))
    codec.eval()
    geometry = {"height": int(ta["height"]), "width": int(ta["width"]), "seq": int(shape["seq"]),
                "code_bytes": int(ta["image_code_bytes"]), "patch": codec.patch, "planes": codec.planes,
                "out_scale": getattr(codec, "out_scale", 1)}          # decoded pictures are this many x the frame
    return model, codec, geometry, step


def fit_frame(image, height, width):
    """PIL image -> float tensor [3, H, W] in [0, 1], cover-scaled and centre-cropped,
    the same framing the corpus builder used."""
    img = image.convert("RGB")
    src_w, src_h = img.size
    scale = max(width / src_w, height / src_h)
    img = img.resize((max(width, int(src_w * scale + 0.5)), max(height, int(src_h * scale + 0.5))))
    left, top = (img.size[0] - width) // 2, (img.size[1] - height) // 2
    img = img.crop((left, top, left + width, top + height))
    arr = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(height, width, RGB).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).float().div_(RGB_MAX)


def encode_frame(codec, frame):
    """[3, H, W] float -> uint8 codes [code_bytes], plane-major."""
    with torch.no_grad():
        codes = codec.encode(frame.unsqueeze(0))[0]
    return np.frombuffer(codec.to_bytes(codes), dtype=np.uint8).copy()


def rect_cells(gh, gw, rect):
    """Cells whose extent overlaps a rectangle given as fractions (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in rect)
    inside = np.zeros((gh, gw), dtype=bool)
    for gy in range(gh):
        for gx in range(gw):
            cx0, cx1 = gx / gw, (gx + 1) / gw
            cy0, cy1 = gy / gh, (gy + 1) / gh
            inside[gy, gx] = cx1 > x0 and cx0 < x1 and cy1 > y0 and cy0 < y1
    return inside


def inner_cells(gh, gw, factor):
    """Cells fully inside the centred box whose side is `factor` of the frame."""
    factor = min(max(float(factor), 0.05), 1.0)
    margin = (1.0 - factor) / 2.0
    inside = np.zeros((gh, gw), dtype=bool)
    for gy in range(gh):
        for gx in range(gw):
            inside[gy, gx] = (gx / gw >= margin and (gx + 1) / gw <= 1.0 - margin
                              and gy / gh >= margin and (gy + 1) / gh <= 1.0 - margin)
    return inside


def cells_to_positions(cell_mask, planes):
    """bool [gh, gw] -> bool [planes*gh*gw] in the plane-major order of the byte string."""
    return np.tile(cell_mask.reshape(-1), int(planes))


def build_window(seq, code_bytes, caption=b"", codes=None, keep=None):
    """int64 [seq]: pad, RECORD_SEP, caption, then the image slice -- MASK_BYTE wherever
    `keep` is False (or everywhere when no codes). A caption longer than its budget is
    cut from the front, never the image."""
    first = seq - code_bytes
    window = np.full(seq, image_grid.PAD_BYTE, dtype=np.int64)
    prefix = (image_grid.RECORD_SEP + caption) if caption else b""
    prefix = prefix[-first:] if first else b""
    if prefix:
        window[first - len(prefix):first] = np.frombuffer(prefix, dtype=np.uint8)
    if codes is None:
        window[first:] = image_codec.MASK_BYTE
        return window
    codes = np.asarray(codes, dtype=np.int64)
    if codes.shape != (code_bytes,):
        raise ValueError("codes must have " + str(code_bytes) + " bytes, got " + str(codes.shape))
    keep = np.ones(code_bytes, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    window[first:] = np.where(keep, codes, image_codec.MASK_BYTE)
    return window


@torch.no_grad()
def fill(model, window, first_code, keep=None, passes=DEFAULT_PASSES, temperature=1.0, seed=0,
         device="cpu", trace=None):
    """Parallel masked decode. Returns uint8 codes [code_bytes]. A `trace` list receives
    one entry per pass -- the codes so far, which positions are still unknown, how many
    were committed and their mean confidence -- so a caller can show the picture forming."""
    code_bytes = len(window) - first_code
    keep = np.zeros(code_bytes, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    tokens = torch.from_numpy(np.asarray(window, dtype=np.int64).copy()).unsqueeze(0).to(device)
    unknown = torch.from_numpy(~keep).to(device)
    n0 = int(unknown.sum())
    if n0 == 0:
        return tokens[0, first_code:].to(torch.uint8).cpu().numpy()
    passes = max(1, min(int(passes), MAX_PASSES))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    for t in range(1, passes + 1):
        out = model(tokens)
        logits = (out[0] if isinstance(out, (tuple, list)) else out)[0, first_code:, :].float()
        logits[:, image_codec.MASK_BYTE] = float("-inf")
        if temperature > 0:
            probs = torch.softmax(logits / float(temperature), dim=-1)
            sampled = torch.multinomial(probs.cpu(), 1, generator=gen).squeeze(1).to(device)
        else:
            probs = torch.softmax(logits, dim=-1)
            sampled = logits.argmax(dim=-1)
        conf = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
        conf = conf.masked_fill(~unknown, float("inf"))
        remain = 0 if t == passes else math.floor(n0 * math.cos(math.pi / 2.0 * t / passes))
        remain = min(remain, int(unknown.sum()) - 1) if remain else 0
        commit = unknown.clone()
        if remain > 0:
            commit[torch.topk(-conf, remain).indices] = False
        image = tokens[0, first_code:]
        image[commit] = sampled[commit]
        unknown = unknown & ~commit
        if trace is not None:
            committed = conf[commit]
            trace.append({"pass": t,
                          "codes": tokens[0, first_code:].to(torch.uint8).cpu().numpy().copy(),
                          "unknown": unknown.cpu().numpy().copy(),
                          "committed": int(commit.sum()),
                          "confidence": float(committed.mean()) if committed.numel() else None})
        if not bool(unknown.any()):
            break
    return tokens[0, first_code:].to(torch.uint8).cpu().numpy()


def decode_png(codec, codes, height, width):
    """uint8 codes -> PNG bytes at the model's frame."""
    frame = codec.decode(codec.from_bytes(bytes(np.asarray(codes, dtype=np.uint8).tolist()), height, width))
    image = Image.fromarray(frame.cpu().numpy(), "RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def generate(model, codec, geometry, mode="text", caption=b"", source=None, strength=DEFAULT_STRENGTH,
             rect=None, expand=DEFAULT_EXPAND, passes=DEFAULT_PASSES, temperature=1.0, seed=0,
             device="cpu"):
    """Every mode through one fill. `source` is PIL image bytes (any format PIL reads).
    Returns (png_bytes, info)."""
    codes, info = generate_codes(model, codec, geometry, mode=mode, caption=caption, source=source,
                                 strength=strength, rect=rect, expand=expand, passes=passes,
                                 temperature=temperature, seed=seed, device=device)
    return decode_png(codec, codes, geometry["height"], geometry["width"]), info


def generate_codes(model, codec, geometry, mode="text", caption=b"", source=None, strength=DEFAULT_STRENGTH,
                   rect=None, expand=DEFAULT_EXPAND, passes=DEFAULT_PASSES, temperature=1.0, seed=0,
                   device="cpu"):
    """generate() before the decode: the uint8 codes and the info dict. What a probe that
    compares two generations (with and without the words) needs."""
    if mode not in MODES:
        raise ValueError("unknown mode: " + str(mode) + " (valid: " + ", ".join(MODES) + ")")
    h, w, seq, code_bytes = geometry["height"], geometry["width"], geometry["seq"], geometry["code_bytes"]
    gh, gw, planes = h // codec.patch, w // codec.patch, codec.planes
    if mode in SOURCE_MODES and source is None:
        raise ValueError("mode " + mode + " needs a source image")
    caption = caption if isinstance(caption, bytes) else str(caption or "").encode("utf-8")
    if mode == "unconditional":
        caption = b""

    codes, keep = None, None
    if mode in SOURCE_MODES:
        image = Image.open(io.BytesIO(source)) if isinstance(source, bytes | bytearray) else source
        if mode == "expand":
            frame = fit_frame(image, max(codec.patch, int(h * expand)) // codec.patch * codec.patch,
                              max(codec.patch, int(w * expand)) // codec.patch * codec.patch)
            canvas = frame.mean(dim=(1, 2), keepdim=True).expand(RGB, h, w).clone()
            top, left = (h - frame.shape[1]) // 2, (w - frame.shape[2]) // 2
            canvas[:, top:top + frame.shape[1], left:left + frame.shape[2]] = frame
            codes = encode_frame(codec, canvas)
            keep = cells_to_positions(inner_cells(gh, gw, expand), planes)
        else:
            codes = encode_frame(codec, fit_frame(image, h, w))
            if mode == "inpaint":
                if not rect:
                    raise ValueError("inpaint needs rect (x0, y0, x1, y1) as fractions of the frame")
                keep = cells_to_positions(~rect_cells(gh, gw, rect), planes)
            else:
                rng = np.random.RandomState(int(seed))
                strength = min(max(float(strength), 0.0), 1.0)
                cells = rng.uniform(size=(gh, gw)) >= strength
                keep = cells_to_positions(cells, planes)
    window = build_window(seq, code_bytes, caption, codes, keep)
    out = fill(model, window, seq - code_bytes, keep, passes=passes, temperature=temperature,
               seed=seed, device=device)
    info = {"mode": mode, "height": h, "width": w, "code_bytes": code_bytes, "passes": passes,
            "regenerated": code_bytes if keep is None else int((~keep).sum()),
            "caption_bytes": len(caption)}
    return out, info
