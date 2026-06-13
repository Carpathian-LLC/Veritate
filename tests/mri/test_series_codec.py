# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the price-series byte codec (veritate_mri/tools/series_codec.py).
#   Covers bar format, return-bucket sign, sequence stride, clipping, and the
#   no-lookahead guarantee that feature[t] never reads bars after t.
# tests/mri/test_series_codec.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "veritate_mri", "tools"))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import series_codec as sc

# ------------------------------------------------------------------------------------
# Fixtures

def _series(n, seed):
    rng = np.random.default_rng(seed)
    c = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.012, n))
    o = c * (1.0 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1.0 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1.0 - np.abs(rng.normal(0, 0.004, n)))
    v = rng.uniform(1e6, 5e6, n)
    return o, h, l, c, v, c.copy()

# ------------------------------------------------------------------------------------
# Functions

def test_encode_bar_is_three_alphabet_chars():
    """encode_bar returns exactly 3 chars, all in the alphabet."""
    bar = sc.encode_bar(0.0, 1.0, 1.0)
    assert len(bar) == sc.BAR_STRIDE
    assert all(ch in sc.ALPHABET for ch in bar)


def test_return_bucket_sign_tracks_direction():
    """Center return encodes flat; strong up/down encode to +/- sign buckets."""
    assert sc.ret_bucket_sign(sc.decode_ret_bucket(sc.encode_bar(0.0, 1.0, 1.0)[0])) == 0
    assert sc.ret_bucket_sign(sc.decode_ret_bucket(sc.encode_bar(3.0, 1.0, 1.0)[0])) == 1
    assert sc.ret_bucket_sign(sc.decode_ret_bucket(sc.encode_bar(-3.0, 1.0, 1.0)[0])) == -1


def test_bucket_clips_extremes_in_range():
    """Values past the clip bounds map to the end buckets, never out of range."""
    hi = sc.encode_bar(1e9, 1e9, 1e9)
    lo = sc.encode_bar(-1e9, -1e9, -1e9)
    for ch in hi + lo:
        assert ch in sc.ALPHABET


def test_sequence_stride_and_length():
    """encode_sequence yields exactly stride*n chars, each return slot decodable."""
    o, h, l, c, v, adj = _series(400, seed=1)
    rz, gr, vr = sc.compute_features(o, h, l, c, v, adj)
    seq = sc.encode_sequence(rz, gr, vr)
    assert len(seq) == sc.BAR_STRIDE * len(rz)
    for i in range(0, len(seq), sc.BAR_STRIDE):
        assert sc.decode_ret_bucket(seq[i]) is not None


def test_encode_sequence_matches_scalar():
    """Vectorized encode_sequence is byte-identical to per-bar encode_bar."""
    o, h, l, c, v, adj = _series(300, seed=3)
    rz, gr, vr = sc.compute_features(o, h, l, c, v, adj)
    seq = sc.encode_sequence(rz, gr, vr)
    scalar = "".join(sc.encode_bar(rz[i], gr[i], vr[i]) for i in range(len(rz)))
    assert seq == scalar


def test_no_lookahead():
    """Truncating the series leaves earlier features bitwise unchanged (trailing-only)."""
    o, h, l, c, v, adj = _series(500, seed=2)
    full = sc.compute_features(o, h, l, c, v, adj)
    k = 300
    trunc = sc.compute_features(o[:k], h[:k], l[:k], c[:k], v[:k], adj[:k])
    m = len(trunc[0])
    assert m > 0
    for fa, ta in zip(full, trunc):
        assert np.array_equal(fa[:m], ta)
