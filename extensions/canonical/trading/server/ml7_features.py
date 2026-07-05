# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The ONE feature builder for the ml7 strategy: the audited 28-feature daily panel
#   extracted verbatim from SMOKE_RESULTS/daily_ml_panel_smoke.py so the frozen model
#   and the live scorer consume byte-identical features (parity test:
#   tests/test_ml7_features.py). Editing any formula here breaks the trained model's
#   contract; retrain (ml7_trader.py --train) after any change.
# extensions/canonical/trading/server/ml7_features.py
# ------------------------------------------------------------------------------------
# Imports:

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Constants

PRICE_COLS = ["time", "open", "high", "low", "close", "volume"]
US_SWITCH = 1e14                  # raw epoch above this is microseconds, else ms
MS_PER_DAY = 86400000
MIN_DAY_MINUTES = 1000            # 1m-resampled day with fewer bars = data gap, masked

RET_WINDOWS = [1, 3, 7, 14, 30, 90]
VOL_SHORT = 7
VOL_LONG = 30
EXP_SHORT = 7
EXP_LONG = 30
RSI_WIN = 14
HI_LO_WINDOWS = [30, 90]
FUND_WIN = 7
BETA_WIN = 90

HORIZON = 7
LABEL = "y_sign_%d" % HORIZON
BTC_PAIR = "BTCUSDT"

FEATS = (["retz_%d" % h for h in RET_WINDOWS]
         + ["vol_7", "vol_30", "vol_ratio", "range_exp", "volume_exp", "rsi_14"]
         + ["dist_hi_%d" % w for w in HI_LO_WINDOWS]
         + ["dist_lo_%d" % w for w in HI_LO_WINDOWS]
         + ["fund_lvl", "fund_chg7", "xs_rank_7", "xs_rank_30", "residz_7"]
         + ["dow_%d" % d for d in range(7)])

# ------------------------------------------------------------------------------------
# Functions

def daily_from_minute_csv(path):
    """1m OHLCV CSV (mixed ms/us epochs) -> daily bars keyed by UTC day, with the
    per-day minute count in nmin (gap masking happens in compute_features)."""
    df = pd.read_csv(path, usecols=PRICE_COLS)
    t = df["time"].to_numpy(np.int64)
    t = np.where(t > US_SWITCH, t // 1000, t)
    df["day"] = t // MS_PER_DAY
    g = df.groupby("day").agg(open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last"),
                              volume=("volume", "sum"), nmin=("close", "size"))
    g.index = pd.to_datetime(g.index, unit="D")
    return g


def funding_daily_from_csv(path):
    df = pd.read_csv(path)
    t = df["time"].to_numpy(np.int64)
    t = np.where(t > US_SWITCH, t // 1000, t)
    day = pd.to_datetime(t // MS_PER_DAY, unit="D")
    return df.groupby(day)["funding"].sum()


def compute_features(daily, fund, btc_r1, btc_ret7):
    """Per-pair daily feature frame. Verbatim from daily_ml_panel_smoke.compute_features
    (h=7 label only): the parity contract with the frozen model."""
    idx = pd.date_range(daily.index[0], daily.index[-1], freq="D")
    d = daily.reindex(idx)
    bad = d["nmin"].isna() | (d["nmin"] < MIN_DAY_MINUTES)
    d.loc[bad, ["open", "high", "low", "close", "volume"]] = np.nan
    c = d["close"]
    logc = np.log(c)
    r1 = logc.diff()
    f = pd.DataFrame(index=idx)
    f["r1_simple"] = np.exp(r1) - 1.0
    f["n_bad_days"] = bad.astype(int)
    vol_s = r1.rolling(VOL_SHORT).std()
    vol_l = r1.rolling(VOL_LONG).std()
    f["vol_7"], f["vol_30"] = vol_s, vol_l
    f["vol_ratio"] = vol_s / vol_l
    for h in RET_WINDOWS:
        ret = logc - logc.shift(h)
        f["ret_%d" % h] = ret
        f["retz_%d" % h] = ret / (vol_l * np.sqrt(h))
    rng = (d["high"] - d["low"]) / c
    f["range_exp"] = rng.rolling(EXP_SHORT).mean() / rng.rolling(EXP_LONG).mean()
    f["volume_exp"] = d["volume"].rolling(EXP_SHORT).mean() / d["volume"].rolling(EXP_LONG).mean()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(RSI_WIN).mean()
    dn = (-delta.clip(upper=0)).rolling(RSI_WIN).mean()
    f["rsi_14"] = 100.0 * up / (up + dn)
    for w in HI_LO_WINDOWS:
        f["dist_hi_%d" % w] = c / c.rolling(w).max() - 1.0
        f["dist_lo_%d" % w] = c / c.rolling(w).min() - 1.0
    fr = fund.reindex(idx)
    f["fund_daily"] = fr.fillna(0.0)
    lvl = fr.rolling(FUND_WIN).mean()
    f["fund_lvl"] = lvl.fillna(0.0)
    f["fund_chg7"] = (lvl - lvl.shift(FUND_WIN)).fillna(0.0)
    br1 = btc_r1.reindex(idx)
    beta = r1.rolling(BETA_WIN).cov(br1) / br1.rolling(BETA_WIN).var()
    f["residz_7"] = (f["ret_7"] - beta * btc_ret7.reindex(idx)) / (vol_l * np.sqrt(7))
    dow = idx.dayofweek
    for dd in range(7):
        f["dow_%d" % dd] = (dow == dd).astype(float)
    fwd = logc.shift(-HORIZON) - logc
    f[LABEL] = (fwd > 0).astype(float).where(fwd.notna())
    return f


def btc_series(btc_daily):
    """BTC daily log-return + 7d log-return on BTC's own calendar (unmasked close,
    matching the smoke's main())."""
    idx = pd.date_range(btc_daily.index[0], btc_daily.index[-1], freq="D")
    logc = np.log(btc_daily["close"].reindex(idx))
    return logc.diff(), logc - logc.shift(HORIZON)


def build_panel(daily_by_pair, fund_by_pair):
    """Pooled panel over all pairs with the cross-sectional ranks, filtered to rows
    where every feature is present. Returns (panel, rets_wide, fund_wide)."""
    btc_r1, btc_ret7 = btc_series(daily_by_pair[BTC_PAIR])
    frames, rets_w, fund_w = [], {}, {}
    for pair in sorted(daily_by_pair):
        f = compute_features(daily_by_pair[pair],
                             fund_by_pair.get(pair, pd.Series(dtype=float)),
                             btc_r1, btc_ret7)
        rets_w[pair] = f["r1_simple"]
        fund_w[pair] = f["fund_daily"]
        f = f.drop(columns=["r1_simple", "fund_daily", "n_bad_days"])
        f["pair"] = pair
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel["xs_rank_7"] = panel.groupby("date")["ret_7"].rank(pct=True) - 0.5
    panel["xs_rank_30"] = panel.groupby("date")["ret_30"].rank(pct=True) - 0.5
    panel = panel[panel[FEATS].notna().all(axis=1)]
    return panel, pd.DataFrame(rets_w), pd.DataFrame(fund_w)
