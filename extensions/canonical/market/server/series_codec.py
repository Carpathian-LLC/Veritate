# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - byte-level codec for numeric price series (stocks, crypto, any OHLCV).
# - one bar -> 9 scale-free / categorical features: return, range, relative
#   volume, realized-vol level, session (time-of-day), buy pressure (taker-buy
#   share of volume), trade activity (trade count vs trailing mean), perp funding
#   regime (signed 8h funding rate), market sentiment (fear-greed 0..100). each
#   feature is one printable char; a bar is BAR_STRIDE chars at a fixed stride; an
#   instrument is its bars concatenated and terminated by a newline.
# - channels are an additive prefix: a model serves at the stride it was trained
#   on, so adding a channel never breaks an older model (it just reads fewer chars).
# - shared contract: the corpus builder encodes offline, the predict page encodes
#   a live window and decodes the model's continuation. both import this file so
#   the on-disk format and the live format never diverge.
# - normalization uses STRICTLY TRAILING windows (no lookahead): feature at bar t
#   is normalized by stats over bars before t only.
# - session, buy pressure, trade activity, funding, and sentiment each degrade to
#   a fixed constant byte (bin 0) when their input is absent, so the stride stays
#   fixed across sources that do or do not carry timestamps / taker volume / trade
#   counts / funding / sentiment.
# extensions/canonical/market/server/series_codec.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ------------------------------------------------------------------------------------
# Constants

FEAT_WINDOW = 20
RV_WINDOW = 20
RV_REF_WINDOW = 100
RET_BINS = 33
RNG_BINS = 16
VOL_BINS = 16
RV_BINS = 16
RET_Z_CLIP = 4.0
RNG_RATIO_CLIP = 4.0
VOL_RATIO_CLIP = 6.0
RV_RATIO_CLIP = 4.0

HOURS_PER_DAY = 24
NS_PER_SEC = 1_000_000_000
SECS_PER_HOUR = 3600
SECS_PER_DAY = 86400
SESSION_NONE = 0
SESSION_BINS = HOURS_PER_DAY + 1

# buy pressure (taker-buy / volume in [0,1]) and trade activity (count vs trailing mean):
# bin 0 is reserved for "input absent", levels 1..N carry the bucketed value.
CHAN_NONE = 0
BP_LEVELS = 16
BP_BINS = BP_LEVELS + 1
TR_LEVELS = 16
TR_BINS = TR_LEVELS + 1
TR_RATIO_CLIP = 6.0

# funding regime (signed perp 8h funding rate) and market sentiment (fear-greed 0..100):
# bin 0 = input absent, levels 1..N carry the bucketed value (funding is signed and centered).
FND_LEVELS = 16
FND_BINS = FND_LEVELS + 1
FND_CLIP = 0.0015
FG_LEVELS = 16
FG_BINS = FG_LEVELS + 1
FG_LO = 0.0
FG_HI = 100.0

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"
BAR_STRIDE = 9
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


def _session_from_ns(ts_ns):
    """Epoch-ns per bar -> hour-of-day bin in 1..HOURS_PER_DAY (UTC). None -> all SESSION_NONE."""
    if ts_ns is None:
        return None
    sec = np.asarray(ts_ns, dtype=np.int64) // NS_PER_SEC
    hour = (sec % SECS_PER_DAY) // SECS_PER_HOUR
    return (hour + 1).astype(np.intp)


def _buy_pressure(tb, v, valid):
    """taker-buy share of volume (in [0,1]) -> bin 1..BP_LEVELS; absent / non-finite -> bin 0."""
    if tb is None:
        return np.zeros(int(valid.sum()), dtype=np.intp)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (np.asarray(tb, dtype=np.float64)[1:] / np.asarray(v, dtype=np.float64)[1:])[valid]
    bad = ~np.isfinite(frac)
    b = _bucket_vec(np.where(bad, 0.0, frac), 0.0, 1.0, BP_LEVELS) + 1
    b[bad] = CHAN_NONE
    return b.astype(np.intp)


def _trade_activity(ntr, valid):
    """trade count vs its trailing mean -> bin 1..TR_LEVELS; absent / non-finite -> bin 0."""
    if ntr is None:
        return np.zeros(int(valid.sum()), dtype=np.intp)
    nt = np.asarray(ntr, dtype=np.float64)[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (nt / _trailing(nt, FEAT_WINDOW, np.mean))[valid]
    bad = ~np.isfinite(ratio)
    b = _bucket_vec(np.where(bad, 0.0, ratio), 0.0, TR_RATIO_CLIP, TR_LEVELS) + 1
    b[bad] = CHAN_NONE
    return b.astype(np.intp)


def _funding_regime(fund, valid):
    """signed perp 8h funding rate -> centered bin 1..FND_LEVELS; absent / non-finite -> bin 0."""
    if fund is None:
        return np.zeros(int(valid.sum()), dtype=np.intp)
    f = np.asarray(fund, dtype=np.float64)[1:][valid]
    bad = ~np.isfinite(f)
    b = _bucket_vec(np.where(bad, 0.0, f), -FND_CLIP, FND_CLIP, FND_LEVELS) + 1
    b[bad] = CHAN_NONE
    return b.astype(np.intp)


def _sentiment(fng, valid):
    """fear-greed index 0..100 -> bin 1..FG_LEVELS; absent / non-finite -> bin 0."""
    if fng is None:
        return np.zeros(int(valid.sum()), dtype=np.intp)
    s = np.asarray(fng, dtype=np.float64)[1:][valid]
    bad = ~np.isfinite(s)
    b = _bucket_vec(np.where(bad, FG_LO, s), FG_LO, FG_HI, FG_LEVELS) + 1
    b[bad] = CHAN_NONE
    return b.astype(np.intp)


def compute_features(o, h, l, c, v, adj, ts_ns=None, tb=None, ntr=None, fund=None, fng=None):
    """ascending-time OHLCV (+ optional epoch-ns, taker-buy volume, trade count, funding rate,
    sentiment) arrays -> (ret_z, rng_ratio, vol_ratio, rv_ratio, session, buy_pressure,
    trade_activity, funding_regime, sentiment) for valid bars."""
    ret = adj[1:] / adj[:-1] - 1.0
    rng = (h[1:] - l[1:]) / c[:-1]
    vv = v[1:]
    w = FEAT_WINDOW
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_z = ret / _trailing(ret, w, np.std)
        rng_ratio = rng / _trailing(rng, w, np.mean)
        vol_ratio = vv / _trailing(vv, w, np.mean)
        rv = _trailing(ret, RV_WINDOW, np.std)
        rv_ratio = rv / _trailing(rv, RV_REF_WINDOW, np.mean)
    valid = np.isfinite(ret_z) & np.isfinite(rng_ratio) & np.isfinite(vol_ratio) & np.isfinite(rv_ratio)
    sess = _session_from_ns(None if ts_ns is None else ts_ns[1:])
    sess = np.full(valid.sum(), SESSION_NONE, dtype=np.intp) if sess is None else sess[valid]
    bp = _buy_pressure(tb, v, valid)
    tr = _trade_activity(ntr, valid)
    fr = _funding_regime(fund, valid)
    st = _sentiment(fng, valid)
    return ret_z[valid], rng_ratio[valid], vol_ratio[valid], rv_ratio[valid], sess, bp, tr, fr, st


def _bucket_vec(v, lo, hi, nbins):
    t = np.clip((np.asarray(v, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return (t * (nbins - 1) + 0.5).astype(np.intp)


def encode_sequence(ret_z, rng_ratio, vol_ratio, rv_ratio, session, buy_pressure, trade_activity,
                    funding=None, sentiment=None, stride=BAR_STRIDE):
    """Emit the first `stride` channels per bar (channels are an additive prefix, so
    stride < BAR_STRIDE reproduces an older codec exactly). Lets one model be served at
    the stride it was trained on without re-encoding the whole format. funding / sentiment
    default to the absent byte (bin 0) so a 7-channel caller still encodes a valid prefix."""
    rb = _bucket_vec(ret_z, -RET_Z_CLIP, RET_Z_CLIP, RET_BINS)
    gb = _bucket_vec(rng_ratio, 0.0, RNG_RATIO_CLIP, RNG_BINS)
    vb = _bucket_vec(vol_ratio, 0.0, VOL_RATIO_CLIP, VOL_BINS)
    yb = _bucket_vec(rv_ratio, 0.0, RV_RATIO_CLIP, RV_BINS)
    sb = np.clip(np.asarray(session, dtype=np.intp), 0, SESSION_BINS - 1)
    pb = np.clip(np.asarray(buy_pressure, dtype=np.intp), 0, BP_BINS - 1)
    nb = np.clip(np.asarray(trade_activity, dtype=np.intp), 0, TR_BINS - 1)
    fb = np.zeros(rb.size, dtype=np.intp) if funding is None else np.clip(np.asarray(funding, dtype=np.intp), 0, FND_BINS - 1)
    eb = np.zeros(rb.size, dtype=np.intp) if sentiment is None else np.clip(np.asarray(sentiment, dtype=np.intp), 0, FG_BINS - 1)
    out = np.empty((rb.size, BAR_STRIDE), dtype=np.uint8)
    out[:, 0] = _ALPHA_U8[rb]
    out[:, 1] = _ALPHA_U8[gb]
    out[:, 2] = _ALPHA_U8[vb]
    out[:, 3] = _ALPHA_U8[yb]
    out[:, 4] = _ALPHA_U8[sb]
    out[:, 5] = _ALPHA_U8[pb]
    out[:, 6] = _ALPHA_U8[nb]
    out[:, 7] = _ALPHA_U8[fb]
    out[:, 8] = _ALPHA_U8[eb]
    return out[:, :stride].tobytes().decode("ascii")


def _bucket(value, lo, hi, nbins):
    t = (value - lo) / (hi - lo)
    t = min(max(t, 0.0), 1.0)
    return int(t * (nbins - 1) + 0.5)


def encode_bar(ret_z, rng_ratio, vol_ratio, rv_ratio, session, buy_pressure, trade_activity,
               funding=CHAN_NONE, sentiment=CHAN_NONE, stride=BAR_STRIDE):
    rb = _bucket(ret_z, -RET_Z_CLIP, RET_Z_CLIP, RET_BINS)
    gb = _bucket(rng_ratio, 0.0, RNG_RATIO_CLIP, RNG_BINS)
    vb = _bucket(vol_ratio, 0.0, VOL_RATIO_CLIP, VOL_BINS)
    yb = _bucket(rv_ratio, 0.0, RV_RATIO_CLIP, RV_BINS)
    sb = min(max(int(session), 0), SESSION_BINS - 1)
    pb = min(max(int(buy_pressure), 0), BP_BINS - 1)
    nb = min(max(int(trade_activity), 0), TR_BINS - 1)
    fb = min(max(int(funding), 0), FND_BINS - 1)
    eb = min(max(int(sentiment), 0), FG_BINS - 1)
    chars = (ALPHABET[rb] + ALPHABET[gb] + ALPHABET[vb] + ALPHABET[yb] + ALPHABET[sb]
             + ALPHABET[pb] + ALPHABET[nb] + ALPHABET[fb] + ALPHABET[eb])
    return chars[:stride]


def decode_ret_bucket(ch):
    i = _INDEX.get(ch, -1)
    return i if 0 <= i < RET_BINS else None


def ret_bucket_sign(bucket):
    if bucket is None:
        return 0
    return (bucket > RET_CENTER) - (bucket < RET_CENTER)
