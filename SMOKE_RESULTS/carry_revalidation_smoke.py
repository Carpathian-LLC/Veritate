# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - re-validates delta-neutral carry (long spot + short perp, collect funding) on 40
#   Binance USDT funding CSVs 2020-2026, weekly top-K rotation by trailing 7d mean
#   funding (positive-only gate), half capital as margin buffer, 4-leg rotation fees.
#   Falsifier: 2026 net carry on capital < T-bill (~4.5%/yr) -> carry not currently
#   worth venue risk. Also pulls CURRENT Hyperliquid funding (live + 30d realized)
#   to quantify honest $/day per $10k in the present regime.
# - wall-clock: ~2-4 min (CSV backtest seconds; HL API a few dozen paginated calls).
# SMOKE_RESULTS/carry_revalidation_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------------------------------
# Constants

REPO = Path("/Users/mirach-00-usc1/Development/Veritate")
FUNDING_DIR = REPO / "extensions/installed/market/data/funding"
STATS_PATH = REPO / "SMOKE_RESULTS/carry_revalidation_stats.json"

MS_PER_HOUR = 3_600_000
GRID_HOURS = 8
EVENTS_PER_DAY = 3
TRAIL_EVENTS = 21            # 7d trailing window
REBAL_EVENTS = 21            # weekly rotation
WORST_WINDOW_EVENTS = 90     # 30d
K_LIST = [1, 3, 5]
FEE_TIERS_BP = [2.0, 5.0, 10.0]
MARGIN_FRACTION = 0.5        # half capital held as margin buffer
LEGS_PER_SIDE = 2            # enter or exit = spot leg + perp leg
STATIC_PAIRS = ["BTCUSDT", "ETHUSDT"]
YEAR_DAYS = 365.0
CAPITAL = 10_000.0
TBILL_YR = 0.045

HL_API = "https://api.hyperliquid.xyz/info"
HL_TIMEOUT_S = 20
HL_PAGE_MAX = 500
HL_HIST_DAYS = 30
HL_HOURS_PER_YEAR = 24 * 365
HL_CORE_COINS = ["BTC", "ETH", "SOL", "DOGE"]
HL_TOP_ALTS = 10
HL_MIN_DAY_VOL_USD = 5e6
HL_MIN_HIST_HOURS = 480      # >= 20d listed history to trust a 30d realized mean
HL_MAKER_BP = 1.0
HL_TAKER_BP = 3.5

# ------------------------------------------------------------------------------------
# Functions


def load_funding_grid():
    frames = {}
    for csv in sorted(FUNDING_DIR.glob("*.csv")):
        df = pd.read_csv(csv)
        ts = (df["time"] // MS_PER_HOUR) * MS_PER_HOUR
        s = pd.Series(df["funding"].values, index=pd.to_datetime(ts, unit="ms"))
        s = s[~s.index.duplicated(keep="last")]
        frames[csv.stem] = s
    grid = pd.DataFrame(frames).sort_index()
    grid = grid[grid.index == grid.index.round(f"{GRID_HOURS}h")]
    return grid


def backtest_rotation(grid, k, fee_bp):
    fee = fee_bp / 1e4
    slot_frac = MARGIN_FRACTION / k
    vals = grid.fillna(0.0).values
    trail = grid.rolling(TRAIL_EVENTS, min_periods=TRAIL_EVENTS).mean().values
    n = len(grid)
    ret = np.zeros(n)
    held = []
    legs_per_rebal = []
    for i in range(TRAIL_EVENTS, n, REBAL_EVENTS):
        row = trail[i - 1]
        order = np.argsort(-np.nan_to_num(row, nan=-np.inf))
        pick = [j for j in order[:k] if np.isfinite(row[j]) and row[j] > 0]
        enters = [p for p in pick if p not in held]
        exits = [h for h in held if h not in pick]
        n_legs = (len(enters) + len(exits)) * LEGS_PER_SIDE
        legs_per_rebal.append(n_legs)
        ret[i] -= n_legs * fee * slot_frac
        held = pick
        for t in range(i, min(i + REBAL_EVENTS, n)):
            ret[t] += sum(vals[t][p] for p in held) * slot_frac
    series = pd.Series(ret, index=grid.index)
    return series, float(np.mean(legs_per_rebal)) if legs_per_rebal else 0.0


def backtest_static(grid, pair, fee_bp):
    fee = fee_bp / 1e4
    s = grid[pair].fillna(0.0) * MARGIN_FRACTION
    s.iloc[0] -= LEGS_PER_SIDE * fee * MARGIN_FRACTION
    return s


def summarize(series):
    out = {}
    for yr, chunk in series.groupby(series.index.year):
        days = (chunk.index[-1] - chunk.index[0]).days + 1
        raw = float(chunk.sum())
        out[str(yr)] = {
            "return_on_capital_pct": round(raw * 100, 3),
            "annualized_pct": round(raw * (YEAR_DAYS / max(days, 1)) * 100, 3),
            "days_covered": days,
        }
    roll = series.rolling(WORST_WINDOW_EVENTS).sum()
    worst = float(roll.min())
    out["worst_30d_pct"] = round(worst * 100, 3)
    out["worst_30d_end"] = str(roll.idxmin().date())
    return out


def hl_post(body):
    r = requests.post(HL_API, json=body, timeout=HL_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def hl_live_snapshot():
    meta, ctxs = hl_post({"type": "metaAndAssetCtxs"})
    rows = []
    for asset, ctx in zip(meta["universe"], ctxs):
        if asset.get("isDelisted"):
            continue
        rows.append({
            "coin": asset["name"],
            "funding_1h": float(ctx["funding"]),
            "ann_pct": float(ctx["funding"]) * HL_HOURS_PER_YEAR * 100,
            "day_vol_usd": float(ctx["dayNtlVlm"]),
            "open_interest_usd": float(ctx["openInterest"]) * float(ctx["markPx"]),
        })
    return rows


def hl_funding_history(coin, start_ms, end_ms):
    rates = []
    cursor = start_ms
    while cursor < end_ms:
        batch = hl_post({"type": "fundingHistory", "coin": coin, "startTime": cursor})
        if not batch:
            break
        rates.extend(float(b["fundingRate"]) for b in batch)
        cursor = batch[-1]["time"] + 1
        if len(batch) < HL_PAGE_MAX:
            break
    return rates


def main():
    stats = {"errors": []}
    t0 = time.time()

    grid = load_funding_grid()
    stats["data"] = {
        "pairs": int(grid.shape[1]),
        "events": int(grid.shape[0]),
        "start": str(grid.index[0]),
        "end": str(grid.index[-1]),
    }

    bt = {}
    for k in K_LIST:
        for fee_bp in FEE_TIERS_BP:
            series, avg_legs = backtest_rotation(grid, k, fee_bp)
            entry = summarize(series)
            entry["avg_legs_per_week"] = round(avg_legs, 2)
            bt[f"k{k}_fee{fee_bp:g}bp"] = entry
    for pair in STATIC_PAIRS:
        for fee_bp in FEE_TIERS_BP[:1]:
            bt[f"static_{pair}_fee{fee_bp:g}bp"] = summarize(backtest_static(grid, pair, fee_bp))
    stats["backtest_yield_on_capital"] = bt

    hl = {}
    try:
        live = hl_live_snapshot()
        live_sorted = sorted(live, key=lambda r: -r["ann_pct"])
        liquid_alts = [r for r in live_sorted
                       if r["coin"] not in HL_CORE_COINS and r["day_vol_usd"] >= HL_MIN_DAY_VOL_USD]
        top_alts = [r["coin"] for r in liquid_alts[:HL_TOP_ALTS]]
        hl["live_snapshot_count"] = len(live)
        hl["live_top10_by_funding"] = [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
            for r in live_sorted[:10]
        ]
        hl["live_majors"] = [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
            for r in live if r["coin"] in HL_CORE_COINS
        ]

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - HL_HIST_DAYS * 24 * MS_PER_HOUR
        realized = {}
        for coin in HL_CORE_COINS + top_alts:
            rates = hl_funding_history(coin, start_ms, now_ms)
            if rates:
                mean_1h = float(np.mean(rates))
                realized[coin] = {
                    "n_hours": len(rates),
                    "mean_1h": mean_1h,
                    "ann_30d_realized_pct": round(mean_1h * HL_HOURS_PER_YEAR * 100, 3),
                }
        hl["realized_30d"] = realized

        eligible = [(c, v) for c, v in realized.items() if v["n_hours"] >= HL_MIN_HIST_HOURS]
        ranked = sorted(eligible, key=lambda kv: -kv[1]["ann_30d_realized_pct"])
        top3 = ranked[:3]
        basket_ann = float(np.mean([v["ann_30d_realized_pct"] for _, v in top3])) / 100
        avg_legs = bt[f"k3_fee{FEE_TIERS_BP[0]:g}bp"]["avg_legs_per_week"]
        slot_frac = MARGIN_FRACTION / 3
        proj = {}
        for label, fee_bp in [("maker", HL_MAKER_BP), ("taker", HL_TAKER_BP)]:
            drag_wk = avg_legs * (fee_bp / 1e4) * slot_frac
            net_yr = MARGIN_FRACTION * basket_ann - drag_wk * (YEAR_DAYS / 7)
            proj[label] = {
                "gross_yr_on_capital_pct": round(MARGIN_FRACTION * basket_ann * 100, 3),
                "net_yr_on_capital_pct": round(net_yr * 100, 3),
                "usd_per_day_per_10k": round(CAPITAL * net_yr / YEAR_DAYS, 2),
            }
        hl["top3_basket"] = {"coins": [c for c, _ in top3],
                            "basket_ann_on_notional_pct": round(basket_ann * 100, 3)}
        hl["projection_10k_top3_weekly"] = proj
        hl["tbill_usd_per_day_per_10k"] = round(CAPITAL * TBILL_YR / YEAR_DAYS, 2)
        hl["caveats"] = [
            "live snapshot rates pinned at HL baseline (0.00125%/h = 10.95%/yr) when premium ~0; 30d realized is the anchor, not the snapshot",
            "delta-neutral needs a spot leg; small alts (LIT/FARTCOIN) may lack a liquid spot market on the same venue -> cross-venue execution + basis risk not priced here",
            "projection assumes 30d realized funding persists; funding mean-reverts and top-of-book alts decay fastest",
        ]
    except Exception as e:
        stats["errors"].append(f"hyperliquid: {type(e).__name__}: {e}")
    stats["hyperliquid_current"] = hl

    stats["wall_clock_s"] = round(time.time() - t0, 1)
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
