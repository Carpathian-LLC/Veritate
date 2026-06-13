# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - byte-level codec for numeric price series (stocks, crypto, any OHLCV).
# - one bar -> 3 scale-free, quantized features: return, range, relative volume.
#   each feature is one printable char; a bar is 3 chars at a fixed stride; an
#   instrument is its bars concatenated and terminated by a newline.
# - shared contract: the corpus builder encodes offline, the predict page encodes
#   a live window and decodes the model's continuation. both import this file so
#   the on-disk format and the live format never diverge.
# - normalization uses STRICTLY TRAILING windows (no lookahead): feature at bar t
#   is normalized by stats over bars before t only.
# veritate_mri/tools/series_codec.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ------------------------------------------------------------------------------------
# Constants

FEAT_WINDOW = 20
RET_BINS = 33
RNG_BINS = 16
VOL_BINS = 16
RET_Z_CLIP = 4.0
RNG_RATIO_CLIP = 4.0
VOL_RATIO_CLIP = 6.0

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"
BAR_STRIDE = 3
SEQ_SEP = "\n"
RET_CENTER = RET_BINS // 2
_ALPHA_U8 = np.frombuffer(ALPHABET.encode("ascii"), dtype=np.uint8)
_INDEX = {c: i for i, c in enumerate(ALPHABET)}

# ------------------------------------------------------------------------------------
# Functions

def _trailing(x, w, reducer):
    out = np.full(len(x), np.nan)
    if len(x) > w:
        sw = sliding_window_view(x, w)
        out[w:] = reducer(sw, axis=1)[:-1]
    return out


def compute_features(o, h, l, c, v, adj):
    """ascending-time OHLCV arrays -> (ret_z, rng_ratio, vol_ratio) for valid bars."""
    ret = adj[1:] / adj[:-1] - 1.0
    rng = (h[1:] - l[1:]) / c[:-1]
    vv = v[1:]
    w = FEAT_WINDOW
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_z = ret / _trailing(ret, w, np.std)
        rng_ratio = rng / _trailing(rng, w, np.mean)
        vol_ratio = vv / _trailing(vv, w, np.mean)
    valid = np.isfinite(ret_z) & np.isfinite(rng_ratio) & np.isfinite(vol_ratio)
    return ret_z[valid], rng_ratio[valid], vol_ratio[valid]


def _bucket_vec(v, lo, hi, nbins):
    t = np.clip((np.asarray(v, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return (t * (nbins - 1) + 0.5).astype(np.intp)


def encode_sequence(ret_z, rng_ratio, vol_ratio):
    rb = _bucket_vec(ret_z, -RET_Z_CLIP, RET_Z_CLIP, RET_BINS)
    gb = _bucket_vec(rng_ratio, 0.0, RNG_RATIO_CLIP, RNG_BINS)
    vb = _bucket_vec(vol_ratio, 0.0, VOL_RATIO_CLIP, VOL_BINS)
    out = np.empty((rb.size, BAR_STRIDE), dtype=np.uint8)
    out[:, 0] = _ALPHA_U8[rb]
    out[:, 1] = _ALPHA_U8[gb]
    out[:, 2] = _ALPHA_U8[vb]
    return out.tobytes().decode("ascii")


def _bucket(value, lo, hi, nbins):
    t = (value - lo) / (hi - lo)
    t = min(max(t, 0.0), 1.0)
    return int(t * (nbins - 1) + 0.5)


def encode_bar(ret_z, rng_ratio, vol_ratio):
    rb = _bucket(ret_z, -RET_Z_CLIP, RET_Z_CLIP, RET_BINS)
    gb = _bucket(rng_ratio, 0.0, RNG_RATIO_CLIP, RNG_BINS)
    vb = _bucket(vol_ratio, 0.0, VOL_RATIO_CLIP, VOL_BINS)
    return ALPHABET[rb] + ALPHABET[gb] + ALPHABET[vb]


def decode_ret_bucket(ch):
    i = _INDEX.get(ch, -1)
    return i if 0 <= i < RET_BINS else None


def ret_bucket_sign(bucket):
    if bucket is None:
        return 0
    return (bucket > RET_CENTER) - (bucket < RET_CENTER)
