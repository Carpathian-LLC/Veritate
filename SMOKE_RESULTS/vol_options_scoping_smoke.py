# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - daily ATM straddle proxy (long + short) gated on realized-vol expansion signal, VRP mult x
#   spread-cost x gate-threshold sweep; falsifier: at published VRP (IV=RV*1.1-1.3) and real IBIT
#   spreads (>=300bp of premium round trip), gated long-straddle expectancy <= 0 across test years
#   -> options monetization of the magnitude signal is dead for US retail.
# - wall-clock estimate ~10min (2x 1m CSV load dominates). CPU only.
# SMOKE_RESULTS/vol_options_scoping_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ------------------------------------------------------------------------------------
# Constants

DATA_DIR = Path("/Users/mirach-00-usc1/Development/Veritate/extensions/installed/market/data/crypto_of")
OUT_PATH = Path("/Users/mirach-00-usc1/Development/Veritate/SMOKE_RESULTS/vol_options_scoping_stats.json")
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

US_EPOCH_CUTOFF = 1e14
MS_PER_DAY = 86_400_000
MIN_BARS_PER_DAY = 1200
BARS_PER_DAY = 1440

RV_LONG_D = 30
RV_SHORT_D = 5
MED_WIN_D = 20
PCT_WIN_D = 252
STRADDLE_COEF = float(np.sqrt(2.0 / np.pi))
ANNUAL_DAYS = 365

VRP_MULTS = [1.0, 1.1, 1.2, 1.3]
COST_FRACS = [0.005, 0.01, 0.02, 0.03, 0.06]
LONG_THRS = [0.0, 0.7, 0.8, 0.9]
SHORT_THRS = [1.0, 0.3, 0.2, 0.1]
IV_REFS = ["rv30", "rv5"]
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]

IBIT_IV30_PCT = 40.1
BP = 1e4

# ------------------------------------------------------------------------------------
# Functions


def load_daily(symbol):
    df = pd.read_csv(DATA_DIR / f"{symbol}.csv", usecols=["time", "high", "low", "close"])
    t = df["time"].to_numpy(dtype=np.int64)
    t = np.where(t > US_EPOCH_CUTOFF, t // 1000, t)
    day = t // MS_PER_DAY
    close = df["close"].to_numpy()
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    same_day = np.empty(len(day), dtype=bool)
    same_day[0] = False
    same_day[1:] = day[1:] == day[:-1]
    g = pd.DataFrame({
        "day": day,
        "high": df["high"].to_numpy(),
        "low": df["low"].to_numpy(),
        "close": close,
        "r": np.where(same_day, log_ret, np.nan),
    }).groupby("day")
    daily = pd.DataFrame({
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "rv1m": g["r"].std() * np.sqrt(BARS_PER_DAY),
        "bars": g["close"].size(),
    })
    daily = daily[daily["bars"] >= MIN_BARS_PER_DAY]
    daily.index = pd.to_datetime(daily.index * MS_PER_DAY, unit="ms")
    return daily


def trailing_percentile(x, win):
    out = np.full(len(x), np.nan)
    for i in range(win, len(x)):
        w = x[i - win:i]
        out[i] = np.mean(w < x[i])
    return out


def build_frame(daily):
    d = daily.copy()
    d["ret"] = np.log(d["close"]).diff()
    d["rng"] = (d["high"] - d["low"]) / d["close"]
    d["rv30"] = d["ret"].rolling(RV_LONG_D).std()
    d["rv5"] = d["ret"].rolling(RV_SHORT_D).std()
    med = lambda s: s.rolling(MED_WIN_D).median().shift(1)
    comp = 0.5 * (d["rv1m"] / med(d["rv1m"]) + d["rng"] / med(d["rng"]))
    d["signal"] = trailing_percentile(comp.to_numpy(), PCT_WIN_D)
    d["abs_next"] = (d["close"].shift(-1) / d["close"] - 1.0).abs()
    d["year"] = d.index.year
    return d.dropna(subset=["signal", "abs_next", "rv30", "rv5"])


def cell_stats(net_bp):
    n = len(net_bp)
    if n == 0:
        return [0, None, None]
    return [n, round(float(np.mean(net_bp > 0)), 3), round(float(np.mean(net_bp)), 2)]


def sweep(d):
    rows = {}
    test = d[d["year"].isin(TEST_YEARS)]
    for iv_ref in IV_REFS:
        prem = STRADDLE_COEF * test[iv_ref].to_numpy()
        move = test["abs_next"].to_numpy()
        sig = test["signal"].to_numpy()
        years = test["year"].to_numpy()
        for mult in VRP_MULTS:
            for c in COST_FRACS:
                long_net = (move - prem * mult * (1.0 + c)) * BP
                short_net = (prem * mult * (1.0 - c) - move) * BP
                for thr in LONG_THRS:
                    m = sig >= thr
                    key = f"{iv_ref}|long|vrp{mult}|cost{int(c * BP)}bp|thr{thr}"
                    rows[key] = {
                        "pooled": cell_stats(long_net[m]),
                        "years": {int(y): cell_stats(long_net[m & (years == y)]) for y in TEST_YEARS},
                    }
                for thr in SHORT_THRS:
                    m = sig <= thr
                    key = f"{iv_ref}|short|vrp{mult}|cost{int(c * BP)}bp|thr{thr}"
                    rows[key] = {
                        "pooled": cell_stats(short_net[m]) + [round(float(np.min(short_net[m])), 1) if m.any() else None],
                        "years": {int(y): cell_stats(short_net[m & (years == y)]) for y in TEST_YEARS},
                    }
    return rows


def signal_validity(d):
    out = {}
    for y in TEST_YEARS:
        sub = d[d["year"] == y]
        if len(sub) > RV_LONG_D:
            rho, p = spearmanr(sub["signal"], sub["abs_next"])
            rho_lvl, _ = spearmanr(sub["rv5"], sub["abs_next"])
            out[int(y)] = {"spearman_expansion": round(float(rho), 3), "p": round(float(p), 5),
                           "spearman_rv5_level": round(float(rho_lvl), 3), "n": len(sub)}
    return out


def main():
    t0 = time.time()
    stats = {
        "errors": [],
        "config": {
            "vrp_mults": VRP_MULTS, "cost_fracs_of_premium": COST_FRACS,
            "long_thrs": LONG_THRS, "short_thrs": SHORT_THRS, "iv_refs": IV_REFS,
            "signal": "trailing-252d pct of mean(rv1m/med20, range/med20), day t close -> t+1 straddle",
            "premium": "sqrt(2/pi) * sigma_daily(iv_ref) * vrp_mult, payoff |close-to-close ret t+1|",
            "cost": "total transaction cost as fraction of premium (entry+exit)",
            "units": "net per trade in bp of underlying",
        },
        "symbols": {},
    }
    for sym in SYMBOLS:
        daily = load_daily(sym)
        d = build_frame(daily)
        rv30_ann = float(d["rv30"].iloc[-1] * np.sqrt(ANNUAL_DAYS) * 100)
        stats["symbols"][sym] = {
            "days": len(d), "start": str(d.index[0].date()), "end": str(d.index[-1].date()),
            "rv30_ann_pct_end": round(rv30_ann, 1),
            "ibit_iv30_pct_2026_07": IBIT_IV30_PCT if sym == "BTCUSDT" else None,
            "signal_spearman_by_year": signal_validity(d),
            "cells": sweep(d),
        }
    stats["wall_clock_s"] = round(time.time() - t0, 1)
    OUT_PATH.write_text(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    try:
        s = main()
        print(f"done in {s['wall_clock_s']}s, {sum(len(v['cells']) for v in s['symbols'].values())} cells")
        sys.exit(0)
    except Exception:
        OUT_PATH.write_text(json.dumps({"errors": [traceback.format_exc()]}, indent=1))
        print(traceback.format_exc())
        sys.exit(1)
