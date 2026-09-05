# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - F0 apparatus for IDEA 24 (CPU-native image generation). Measures decode latency
#   and PEAK ACTIVATION BYTES at a target output resolution for three decoder
#   structures at random weights. No training, no corpus, no checkpoint, no model
#   load: shapes are all that drive the numbers this falsifier needs.
# - Two quantities decide a decoder and both are reported: analytic FLOPs, which
#   dominate, and peak activation bytes, which govern cache residency and whether the
#   decoder fits a small device. IDEA 24's rule is that no tensor whose extent is the
#   output resolution may be materialized; `conv_full` breaks it on purpose so the
#   claim is measured against something rather than asserted.
# - Peak bytes are observed through a dispatch mode that sees every aten output, so
#   the number is independent of what the decoders claim about themselves. That mode
#   distorts timing, so latency and peak are measured in separate passes.
# - Arms: `coord` (per-pixel MLP over cache-sized tiles), `patch` (per-cell features
#   plus their 3x3 neighbourhood expanded to a patch of pixels), `conv_full`
#   (conventional upsampling stack, the control).
# - `PatchDecoder` is shared: F0's byte arm feeds it a code embedding, the codec feeds
#   it a dequantized latent. `render()` writes uint8 band by band and is what the
#   bench and serving call; `forward()` returns float and is what the codec trains
#   through. Both walk the same bands, so the working-set claim covers both.
# veritate_core/plugin/image_decode.py
# ------------------------------------------------------------------------------------
# Imports:

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

# ------------------------------------------------------------------------------------
# Constants

ARMS = ("coord", "patch", "conv_full")

# Defaults are the aggressive row of IDEA 24's budget table. Every one is an argument
# on bench(); the route passes whatever the sweep is exploring.
DEFAULT_LATENT_CH   = 32
DEFAULT_MLP_WIDTH   = 48
DEFAULT_TILE        = 64      # output tile edge; keeps the coord arm's working set in L2
DEFAULT_PATCH       = 20      # 1920x1080 -> a 96x54 grid; dictionary is 307 KB, L2-resident
DEFAULT_CODE_EMB    = 16
DEFAULT_PATCH_HIDDEN = 64
DEFAULT_BAND        = 4       # grid rows decoded per band; bounds the patch arm working set
DEFAULT_CONV_CH     = (128, 64, 32, 16)
DEFAULT_GRID_DIV    = 20       # latent grid for the coord and conv arms
DEFAULT_WARMUP      = 1
DEFAULT_REPS        = 3
VOCAB_BYTE_LEVEL    = 256
RGB                 = 3
CODE_NEIGHBOURHOOD  = 9        # 3x3 including self

# ------------------------------------------------------------------------------------
# Functions


class _PeakBytes(TorchDispatchMode):
    """Largest single tensor produced by any op inside the block, in bytes."""

    def __init__(self):
        super().__init__()
        self.peak = 0

    def _note(self, obj):
        if isinstance(obj, torch.Tensor):
            self.peak = max(self.peak, obj.nelement() * obj.element_size())
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                self._note(item)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._note(out)
        return out


class CoordDecoder(nn.Module):
    """Per-pixel MLP evaluated over output tiles. Weights stay resident; no tensor
    whose extent is the output resolution is ever built."""

    def __init__(self, latent_ch, width, tile):
        super().__init__()
        self.tile = tile
        self.fc1 = nn.Linear(latent_ch + 2, width)
        self.fc2 = nn.Linear(width, width)
        self.out = nn.Linear(width, RGB)

    def flops(self, height, width_px):
        per_px = 2 * (self.fc1.in_features * self.fc1.out_features
                      + self.fc2.in_features * self.fc2.out_features
                      + self.out.in_features * self.out.out_features)
        return per_px * height * width_px

    def render(self, latent, height, width_px):
        canvas = torch.empty((height, width_px, RGB), dtype=torch.uint8, device=latent.device)
        for y0 in range(0, height, self.tile):
            y1 = min(y0 + self.tile, height)
            ys = (torch.arange(y0, y1, device=latent.device, dtype=latent.dtype) + 0.5) / height * 2.0 - 1.0
            for x0 in range(0, width_px, self.tile):
                x1 = min(x0 + self.tile, width_px)
                xs = (torch.arange(x0, x1, device=latent.device, dtype=latent.dtype) + 0.5) / width_px * 2.0 - 1.0
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
                feat = F.grid_sample(latent, grid, mode="bilinear", align_corners=False)
                feat = feat.squeeze(0).permute(1, 2, 0).reshape(-1, latent.shape[1])
                coords = torch.stack((gx, gy), dim=-1).reshape(-1, 2)
                h = torch.cat((feat, coords), dim=1)
                h = F.gelu(self.fc1(h))
                h = F.gelu(self.fc2(h))
                rgb = torch.sigmoid(self.out(h)).mul(255.0).to(torch.uint8)
                canvas[y0:y1, x0:x1] = rgb.reshape(y1 - y0, x1 - x0, RGB)
        return canvas


class PatchDecoder(nn.Module):
    """Each grid cell's feature vector, with its 3x3 neighbourhood, expands to one patch
    of pixels. Decoded in bands of grid rows so the working set is flat in frame area."""

    def __init__(self, feat_dim, patch, hidden, band):
        super().__init__()
        self.patch = patch
        self.band = band
        self.fc1 = nn.Linear(CODE_NEIGHBOURHOOD * feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, patch * patch * RGB)

    def flops(self, height, width_px):
        patches = (height // self.patch) * (width_px // self.patch)
        per_patch = 2 * (self.fc1.in_features * self.fc1.out_features
                         + self.fc2.in_features * self.fc2.out_features)
        return per_patch * patches

    def _padded(self, features):
        return F.pad(features.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="replicate")

    def _band(self, padded, r0, r1, gw):
        neigh = F.unfold(padded[:, :, r0:r1 + 2, :], kernel_size=3).squeeze(0).t()
        tiles = self.fc2(F.gelu(self.fc1(neigh))).reshape(r1 - r0, gw, self.patch, self.patch, RGB)
        return tiles.permute(0, 2, 1, 3, 4).reshape((r1 - r0) * self.patch, gw * self.patch, RGB)

    def forward(self, features):
        """Float frame in [0, 1]. The path the codec trains through."""
        gh, gw, _ = features.shape
        padded = self._padded(features)
        bands = [self._band(padded, r0, min(r0 + self.band, gh), gw)
                 for r0 in range(0, gh, self.band)]
        return torch.sigmoid(torch.cat(bands, dim=0))

    def render(self, features):
        """uint8 frame written band by band, so no float tensor spans the frame."""
        gh, gw, _ = features.shape
        canvas = torch.empty((gh * self.patch, gw * self.patch, RGB),
                             dtype=torch.uint8, device=features.device)
        padded = self._padded(features)
        for r0 in range(0, gh, self.band):
            r1 = min(r0 + self.band, gh)
            band = torch.sigmoid(self._band(padded, r0, r1, gw))
            canvas[r0 * self.patch:r1 * self.patch] = band.mul(255.0).to(torch.uint8)
        return canvas


class BytePatchDecoder(nn.Module):
    """F0's `patch` arm: one byte per grid cell, looked up into a learned feature table
    and expanded by the shared PatchDecoder."""

    def __init__(self, patch, emb, hidden, band):
        super().__init__()
        self.code_emb = nn.Embedding(VOCAB_BYTE_LEVEL, emb)
        self.patches = PatchDecoder(emb, patch, hidden, band)

    def flops(self, height, width_px):
        return self.patches.flops(height, width_px)

    def render(self, codes):
        return self.patches.render(self.code_emb(codes))


class ConvDecoder(nn.Module):
    """Control. A conventional upsampling stack that materializes full-resolution
    feature maps, which is the structure IDEA 24 claims is impossible on a
    single-channel DDR4 box. Present to be measured, not to be shipped."""

    def __init__(self, latent_ch, channels):
        super().__init__()
        self.channels = tuple(channels)
        chans = (latent_ch, *self.channels)
        self.convs = nn.ModuleList([nn.Conv2d(chans[i], chans[i + 1], 3, padding=1)
                                    for i in range(len(self.channels))])
        self.out = nn.Conv2d(self.channels[-1], RGB, 3, padding=1)

    def flops(self, height, width_px):
        total = 0
        stages = len(self.channels)
        for i, conv in enumerate(self.convs):
            scale = 1 << (stages - 1 - i)
            pixels = (height // scale) * (width_px // scale)
            total += 2 * 9 * conv.in_channels * conv.out_channels * pixels
        total += 2 * 9 * self.out.in_channels * RGB * height * width_px
        return total

    def render(self, latent, height, width_px):
        stages = len(self.channels)
        h = latent
        for i, conv in enumerate(self.convs):
            scale = 1 << (stages - 1 - i)
            h = F.interpolate(h, size=(height // scale, width_px // scale), mode="nearest")
            h = F.gelu(conv(h))
        rgb = torch.sigmoid(self.out(h)).squeeze(0).permute(1, 2, 0)
        return rgb.mul(255.0).to(torch.uint8)


def _build(arm, latent_ch, mlp_width, tile, patch, code_emb, patch_hidden, band, conv_ch):
    if arm == "coord":
        return CoordDecoder(latent_ch, mlp_width, tile)
    if arm == "patch":
        return BytePatchDecoder(patch, code_emb, patch_hidden, band)
    return ConvDecoder(latent_ch, conv_ch)


def _inputs(arm, height, width_px, latent_ch, grid_div, patch, device):
    if arm == "patch":
        codes = torch.randint(0, VOCAB_BYTE_LEVEL, (height // patch, width_px // patch), device=device)
        return (codes,)
    latent = torch.randn((1, latent_ch, height // grid_div, width_px // grid_div), device=device)
    return (latent, height, width_px)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench(height, width, arms=ARMS, latent_ch=DEFAULT_LATENT_CH, mlp_width=DEFAULT_MLP_WIDTH,
          tile=DEFAULT_TILE, patch=DEFAULT_PATCH, code_emb=DEFAULT_CODE_EMB,
          patch_hidden=DEFAULT_PATCH_HIDDEN, band=DEFAULT_BAND, conv_ch=DEFAULT_CONV_CH,
          grid_div=DEFAULT_GRID_DIV,
          warmup=DEFAULT_WARMUP, reps=DEFAULT_REPS, device=None, seed=0):
    """Decode one frame per arm at random weights; report latency, achieved GF/s and
    peak activation bytes. Latency and peak are separate passes because the dispatch
    mode that observes peak distorts the clock."""
    for arm in arms:
        if arm not in ARMS:
            raise ValueError("unknown arm: " + str(arm) + " (valid: " + ", ".join(ARMS) + ")")
    if height % patch or width % patch:
        raise ValueError("patch " + str(patch) + " does not divide " + str(height) + "x" + str(width))
    if height % grid_div or width % grid_div:
        raise ValueError("grid_div " + str(grid_div) + " does not divide " + str(height) + "x" + str(width))
    scale = 1 << (len(conv_ch) - 1)
    if height % scale or width % scale:
        raise ValueError("conv_ch depth needs " + str(height) + "x" + str(width) + " divisible by " + str(scale))

    dev = torch.device(device) if device else torch.device("cpu")
    report = {
        "height": height, "width": width, "device": dev.type,
        "threads": torch.get_num_threads(),
        "output_bytes": height * width * RGB,
        "arms": {},
    }
    for arm in arms:
        torch.manual_seed(seed)
        model = _build(arm, latent_ch, mlp_width, tile, patch, code_emb,
                       patch_hidden, band, conv_ch).to(dev).eval()
        args = _inputs(arm, height, width, latent_ch, grid_div, patch, dev)
        with torch.no_grad():
            for _ in range(warmup):
                model.render(*args)
            _sync(dev)
            timings = []
            for _ in range(reps):
                t0 = time.perf_counter()
                model.render(*args)
                _sync(dev)
                timings.append((time.perf_counter() - t0) * 1000.0)
            peak = _PeakBytes()
            with peak:
                model.render(*args)
        timings.sort()
        ms = timings[len(timings) // 2]
        flops = model.flops(height, width)
        report["arms"][arm] = {
            "ms_p50":                ms,
            "ms_all":                timings,
            "gflop":                 flops / 1e9,
            "achieved_gflops":       flops / (ms / 1000.0) / 1e9,
            "peak_activation_bytes": peak.peak,
            "params":                sum(p.nelement() for p in model.parameters()),
        }
    return report
