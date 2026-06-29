# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract tests for the price-series byte codec
#   (extensions/canonical/market/server/series_codec.py). Covers bar format,
#   return-bucket sign, sequence stride, clipping, the realized-vol, session,
#   buy-pressure, trade-activity, funding and sentiment channels, the no-lookahead
#   guarantee, and vectorized/scalar identity.
# extensions/canonical/market/tests/test_series_codec.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.normpath(os.path.join(HERE, "..", "server"))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

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


def _taker(v, seed):
    """Synthetic taker-buy volume and trade count aligned to a volume array."""
    rng = np.random.default_rng(seed)
    tb = v * rng.uniform(0.2, 0.8, len(v))
    ntr = rng.integers(50, 5000, len(v)).astype(np.float64)
    return tb, ntr


def _ts_ns(n, start_ms=1_600_000_000_000, step_ms=3_600_000):
    """Monotonic epoch-ns at one bar per hour from a fixed start."""
    ms = start_ms + step_ms * np.arange(n, dtype=np.int64)
    return ms * 1_000_000

# ------------------------------------------------------------------------------------
# Functions

def test_encode_bar_is_stride_alphabet_chars():
    """encode_bar returns exactly BAR_STRIDE chars, all in the alphabet."""
    bar = sc.encode_bar(0.0, 1.0, 1.0, 1.0, 1, 1, 1)
    assert len(bar) == sc.BAR_STRIDE
    assert all(ch in sc.ALPHABET for ch in bar)


def test_bar_stride_is_nine():
    """The format carries nine channels per bar."""
    assert sc.BAR_STRIDE == 9


def test_return_bucket_sign_tracks_direction():
    """Center return encodes flat; strong up/down encode to +/- sign buckets."""
    assert sc.ret_bucket_sign(sc.decode_ret_bucket(sc.encode_bar(0.0, 1.0, 1.0, 1.0, 0, 0, 0)[0])) == 0
    assert sc.ret_bucket_sign(sc.decode_ret_bucket(sc.encode_bar(3.0, 1.0, 1.0, 1.0, 0, 0, 0)[0])) == 1
    assert sc.ret_bucket_sign(sc.decode_ret_bucket(sc.encode_bar(-3.0, 1.0, 1.0, 1.0, 0, 0, 0)[0])) == -1


def test_bucket_clips_extremes_in_range():
    """Values past the clip bounds map to the end buckets, never out of range."""
    hi = sc.encode_bar(1e9, 1e9, 1e9, 1e9, 999, 999, 999)
    lo = sc.encode_bar(-1e9, -1e9, -1e9, -1e9, -5, -5, -5)
    for ch in hi + lo:
        assert ch in sc.ALPHABET


def test_sequence_stride_and_length():
    """encode_sequence yields exactly stride*n chars, each return slot decodable."""
    o, h, l, c, v, adj = _series(400, seed=1)
    feats = sc.compute_features(o, h, l, c, v, adj)
    seq = sc.encode_sequence(*feats)
    assert len(seq) == sc.BAR_STRIDE * len(feats[0])
    for i in range(0, len(seq), sc.BAR_STRIDE):
        assert sc.decode_ret_bucket(seq[i]) is not None


def test_encode_sequence_matches_scalar():
    """Vectorized encode_sequence is byte-identical to per-bar encode_bar (all 9 channels)."""
    o, h, l, c, v, adj = _series(300, seed=3)
    ts = _ts_ns(len(c))
    tb, ntr = _taker(v, seed=4)
    fund = np.linspace(-0.001, 0.001, len(c))
    fng = np.linspace(5.0, 95.0, len(c))
    f = sc.compute_features(o, h, l, c, v, adj, ts, tb, ntr, fund, fng)
    seq = sc.encode_sequence(*f)
    scalar = "".join(
        sc.encode_bar(f[0][i], f[1][i], f[2][i], f[3][i], int(f[4][i]), int(f[5][i]),
                      int(f[6][i]), int(f[7][i]), int(f[8][i]))
        for i in range(len(f[0]))
    )
    assert seq == scalar


def test_stride_prefix_emits_first_n_channels():
    """encode_sequence(stride=k) is the first k chars of each full bar (channels are an
    additive prefix), so a model trained on an older stride is served its exact format."""
    o, h, l, c, v, adj = _series(300, seed=3)
    ts = _ts_ns(len(c)); tb, ntr = _taker(v, seed=4)
    feats = sc.compute_features(o, h, l, c, v, adj, ts, tb, ntr)
    full = sc.encode_sequence(*feats)
    n = len(feats[0])
    for k in (3, 5, sc.BAR_STRIDE):
        sub = sc.encode_sequence(*feats, stride=k)
        assert len(sub) == k * n
        for i in range(n):
            assert sub[i * k:(i + 1) * k] == full[i * sc.BAR_STRIDE:i * sc.BAR_STRIDE + k]


def test_scalar_matches_vector_at_reduced_stride():
    """Reduced-stride vector and scalar encoders agree (a model served at its own stride)."""
    o, h, l, c, v, adj = _series(300, seed=3)
    ts = _ts_ns(len(c)); tb, ntr = _taker(v, seed=4)
    f = sc.compute_features(o, h, l, c, v, adj, ts, tb, ntr)
    seq = sc.encode_sequence(*f, stride=5)
    scalar = "".join(
        sc.encode_bar(f[0][i], f[1][i], f[2][i], f[3][i], int(f[4][i]), int(f[5][i]), int(f[6][i]), stride=5)
        for i in range(len(f[0]))
    )
    assert seq == scalar


def test_no_lookahead():
    """Truncating the series leaves earlier features bitwise unchanged (trailing-only)."""
    o, h, l, c, v, adj = _series(500, seed=2)
    ts = _ts_ns(len(c))
    tb, ntr = _taker(v, seed=6)
    full = sc.compute_features(o, h, l, c, v, adj, ts, tb, ntr)
    k = 300
    trunc = sc.compute_features(o[:k], h[:k], l[:k], c[:k], v[:k], adj[:k], ts[:k], tb[:k], ntr[:k])
    m = len(trunc[0])
    assert m > 0
    for fa, ta in zip(full, trunc):
        assert np.array_equal(fa[:m], ta)


def test_realized_vol_channel_no_lookahead():
    """The realized-vol channel byte at each bar is unchanged when later bars are dropped."""
    o, h, l, c, v, adj = _series(500, seed=7)
    full = sc.encode_sequence(*sc.compute_features(o, h, l, c, v, adj))
    k = 320
    feats = sc.compute_features(o[:k], h[:k], l[:k], c[:k], v[:k], adj[:k])
    trunc = sc.encode_sequence(*feats)
    m = len(feats[0])
    rv_full = full[3::sc.BAR_STRIDE][:m]
    rv_trunc = trunc[3::sc.BAR_STRIDE]
    assert m > 0
    assert rv_full == rv_trunc


def test_realized_vol_ratio_spikes_on_regime_jump():
    """The realized-vol ratio (trailing std / its longer trailing mean) spikes when
    volatility jumps to a new regime, exceeding 1.0 right after the jump."""
    rng = np.random.default_rng(11)
    calm = rng.normal(0, 0.002, 400)
    wild = rng.normal(0, 0.02, 400)
    c = 100.0 * np.cumprod(1.0 + np.concatenate([calm, wild]))
    o = c * (1.0 + rng.normal(0, 0.001, len(c)))
    h = np.maximum(o, c) * 1.001
    l = np.minimum(o, c) * 0.999
    v = np.full(len(c), 1e6)
    yr = sc.compute_features(o, h, l, c, v, c.copy())[3]
    assert len(yr) > 0
    assert yr.max() > 2.0


def test_session_byte_tracks_hour_of_day():
    """Session bin equals hour-of-day + 1 (UTC) for the encoded bar."""
    n = 400
    o, h, l, c, v, adj = _series(n, seed=5)
    ts = _ts_ns(n)
    ss = sc.compute_features(o, h, l, c, v, adj, ts)[4]
    sec = ts[1:] // sc.NS_PER_SEC
    hour = (sec % sc.SECS_PER_DAY) // sc.SECS_PER_HOUR
    expect = (hour + 1)[-len(ss):]
    assert np.array_equal(ss, expect)
    assert ss.min() >= 1 and ss.max() <= sc.HOURS_PER_DAY


def test_buy_pressure_channel_tracks_taker_share():
    """A constant taker-buy share encodes to the matching buy-pressure bin (1..BP_LEVELS)."""
    o, h, l, c, v, adj = _series(300, seed=5)
    for frac in (0.25, 0.75):
        seq = sc.encode_sequence(*sc.compute_features(o, h, l, c, v, adj, None, v * frac, None))
        want = sc.ALPHABET[sc._bucket(frac, 0.0, 1.0, sc.BP_LEVELS) + 1]
        assert set(seq[5::sc.BAR_STRIDE]) == {want}


def test_trade_activity_channel_flat_for_constant_count():
    """A constant trade count gives a ratio of 1.0 -> the same trade-activity bin everywhere."""
    o, h, l, c, v, adj = _series(300, seed=8)
    ntr = np.full(len(c), 1234.0)
    seq = sc.encode_sequence(*sc.compute_features(o, h, l, c, v, adj, None, None, ntr))
    want = sc.ALPHABET[sc._bucket(1.0, 0.0, sc.TR_RATIO_CLIP, sc.TR_LEVELS) + 1]
    assert set(seq[6::sc.BAR_STRIDE]) == {want}


def test_taker_channels_fall_back_to_constant_without_inputs():
    """No taker volume / trade count -> every buy-pressure and trade-activity byte is bin 0."""
    o, h, l, c, v, adj = _series(300, seed=9)
    seq = sc.encode_sequence(*sc.compute_features(o, h, l, c, v, adj))
    none_ch = sc.ALPHABET[sc.CHAN_NONE]
    assert set(seq[5::sc.BAR_STRIDE]) == {none_ch}
    assert set(seq[6::sc.BAR_STRIDE]) == {none_ch}


def test_session_falls_back_to_constant_without_timestamps():
    """No timestamp -> every session byte is the SESSION_NONE fallback char."""
    o, h, l, c, v, adj = _series(300, seed=9)
    seq = sc.encode_sequence(*sc.compute_features(o, h, l, c, v, adj))
    sess_chars = set(seq[4::sc.BAR_STRIDE])
    assert sess_chars == {sc.ALPHABET[sc.SESSION_NONE]}


def test_funding_channel_tracks_signed_rate():
    """A constant signed funding rate encodes to the matching centered bin (1..FND_LEVELS)."""
    o, h, l, c, v, adj = _series(300, seed=5)
    for rate in (-0.0008, 0.0008):
        fund = np.full(len(c), rate)
        seq = sc.encode_sequence(*sc.compute_features(o, h, l, c, v, adj, None, None, None, fund, None))
        want = sc.ALPHABET[sc._bucket(rate, -sc.FND_CLIP, sc.FND_CLIP, sc.FND_LEVELS) + 1]
        assert set(seq[7::sc.BAR_STRIDE]) == {want}


def test_sentiment_channel_tracks_index():
    """A constant fear-greed value encodes to the matching bin (1..FG_LEVELS)."""
    o, h, l, c, v, adj = _series(300, seed=5)
    for val in (20.0, 80.0):
        fng = np.full(len(c), val)
        seq = sc.encode_sequence(*sc.compute_features(o, h, l, c, v, adj, None, None, None, None, fng))
        want = sc.ALPHABET[sc._bucket(val, sc.FG_LO, sc.FG_HI, sc.FG_LEVELS) + 1]
        assert set(seq[8::sc.BAR_STRIDE]) == {want}


def test_funding_sentiment_fall_back_to_constant_without_inputs():
    """No funding / sentiment -> every funding and sentiment byte is bin 0."""
    o, h, l, c, v, adj = _series(300, seed=9)
    seq = sc.encode_sequence(*sc.compute_features(o, h, l, c, v, adj))
    none_ch = sc.ALPHABET[sc.CHAN_NONE]
    assert set(seq[7::sc.BAR_STRIDE]) == {none_ch}
    assert set(seq[8::sc.BAR_STRIDE]) == {none_ch}


def _legacy_ret_rng_vol(o, h, l, c, v, adj):
    """Pre-change 3-channel feature math, recomputed independently as the regression baseline."""
    from numpy.lib.stride_tricks import sliding_window_view
    w = sc.FEAT_WINDOW

    def trail(x, reducer):
        out = np.full(len(x), np.nan)
        if len(x) > w:
            sw = sliding_window_view(x, w)
            out[w:] = reducer(sw, axis=1)[:-1]
        return out

    ret = adj[1:] / adj[:-1] - 1.0
    rng = (h[1:] - l[1:]) / c[:-1]
    vv = v[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rz = ret / trail(ret, np.std)
        gr = rng / trail(rng, np.mean)
        vr = vv / trail(vv, np.mean)
    valid = np.isfinite(rz) & np.isfinite(gr) & np.isfinite(vr)
    return rz[valid], gr[valid], vr[valid]


def test_first_three_channels_regression_on_fixed_seed():
    """Return/range/volume bytes are unchanged vs the pre-change codec on a fixed seed.

    The realized-vol channel adds warmup, so the new valid window is a suffix of the
    legacy one; compare the overlapping tail."""
    o, h, l, c, v, adj = _series(300, seed=3)
    rz, gr, vr, _, _, _, _, _, _ = sc.compute_features(o, h, l, c, v, adj)
    lrz, lgr, lvr = _legacy_ret_rng_vol(o, h, l, c, v, adj)
    m = len(rz)
    assert m > 0 and m <= len(lrz)
    z = (np.ones(m), np.zeros(m, dtype=np.intp), np.zeros(m, dtype=np.intp), np.zeros(m, dtype=np.intp))
    new = sc.encode_sequence(rz, gr, vr, *z)
    leg = sc.encode_sequence(lrz[-m:], lgr[-m:], lvr[-m:], *z)
    for slot in range(3):
        assert new[slot::sc.BAR_STRIDE] == leg[slot::sc.BAR_STRIDE]
