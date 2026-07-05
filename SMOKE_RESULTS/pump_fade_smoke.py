# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests FADING (shorting) scanner-detected pumps on the shortable perp universe:
#   top OKX USDT perps by 24h volume mapped to Binance UM perps, hourly klines
#   2023-2026 from data.binance.vision. Detection replicates
#   extensions/canonical/market_intel/server/scanner.py exactly on hourly bars:
#   z=(x-mean)/pop-std over 14-bar trailing baseline (>=5 samples else 0, round 2dp),
#   price_z>=2.5 on 1-bar return AND vol_z>=3.0 on hourly quote volume, 6h cooldown,
#   $50k 24h-vol floor; thresholds pre-registered by the shipped scanner, untuned.
#   Event study: fwd rets 1/4/12/24/48/72h per year + tercile/meme splits, bootstrap.
#   Strategy: short at first close AFTER detection (1-bar lag), exit +24h or +48h
#   (chosen on 2023-2024, tested 2025-2026), 1/10 slots cap 10 concurrent, +15%
#   stop on hourly highs (gap fills at open), 4.5bp taker/side, realized funding
#   credited to the short. Null: exposure-matched random entries per symbol-year.
#   Falsifier: not net positive 2025 AND 2026 at full costs, or Sharpe < 1.0 on
#   deployed capital, or a single event > 30% of book -> radar stays a radar.
# - wall-clock estimate ~6 min with parsed cache, ~15 min cold (zips pre-downloaded).
# SMOKE_RESULTS/pump_fade_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import glob
import io
import json
import os
import time
import traceback
import zipfile

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Constants

REPO = "/Users/mirach-00-usc1/Development/Veritate"
CACHE = ("/private/tmp/claude-501/-Users-mirach-00-usc1-Development-Veritate/"
         "10655374-b645-4bca-a6e1-0208baad89ab/scratchpad/pumpfade")
KLINES_DIR = os.path.join(CACHE, "klines")
FUNDING_DIR = os.path.join(CACHE, "funding")
META_DIR = os.path.join(CACHE, "meta")
PARSED_DIR = os.path.join(CACHE, "parsed")
STATS_PATH = os.path.join(REPO, "SMOKE_RESULTS/pump_fade_stats.json")

US_SWITCH = 10 ** 14
H_S = 3600
BAR_S = 3600

BASELINE_N = 14
BASELINE_MIN = 5
PRICE_Z_MIN = 2.5
VOL_Z_MIN = 3.0
COOLDOWN_S = 6 * H_S
MIN_VOL_USD = 50000.0
VOL24_BARS = 24

STUDY_HORIZONS = [1, 4, 12, 24, 48, 72]
YEARS = [2023, 2024, 2025, 2026]
TRAIN_YEARS = [2023, 2024]
TEST_YEARS = [2025, 2026]
BOOT_N = 10000
BOOT_SEED = 7

FEE_SIDE = 4.5e-4
STOP_ADV = 0.15
SLOTS = 10
SLOT_FRAC = 1.0 / SLOTS
HORIZON_CANDS = [24, 48]
LIQ_FLOOR_USD = 1e6
NULL_SEEDS = 200
NULL_SEED0 = 11
DAYS_YR = 365.0
EV_BOOK_LIMIT = 0.30

# ------------------------------------------------------------------------------------
# Functions

def load_symbol(binance_sym):
    """Hourly bars for one Binance UM perp from monthly zips -> npz cache.
    Returns dict of arrays ts(s), o, h, l, c, qv sorted unique, or None."""
    os.makedirs(PARSED_DIR, exist_ok=True)
    npz = os.path.join(PARSED_DIR, binance_sym + ".npz")
    if os.path.exists(npz):
        d = np.load(npz)
        return {k: d[k] for k in ("ts", "o", "h", "l", "c", "qv")}
    zips = sorted(glob.glob(os.path.join(KLINES_DIR, binance_sym + "-1h-*.zip")))
    if not zips:
        return None
    frames = []
    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as z:
                raw = z.read(z.namelist()[0])
        except (zipfile.BadZipFile, IndexError, OSError):
            continue
        head = raw[:16].decode(errors="replace")
        df = pd.read_csv(io.BytesIO(raw), header=0 if head.startswith("open_time") else None,
                         usecols=[0, 1, 2, 3, 4, 7],
                         names=["t", "o", "h", "l", "c", "qv"], skip_blank_lines=True)
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    t = df["t"].to_numpy(np.int64)
    t = np.where(t > US_SWITCH, t // 1000, t) // 1000
    order = np.argsort(t)
    t = t[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    out = {"ts": t[keep]}
    for k in ("o", "h", "l", "c", "qv"):
        out[k] = df[k].to_numpy(np.float64)[order][keep]
    np.savez_compressed(npz, **out)
    return out


def load_funding(binance_sym):
    """Funding events (ts_s, rate) from monthly zips, sorted. Empty arrays if none."""
    zips = sorted(glob.glob(os.path.join(FUNDING_DIR, binance_sym + "-fundingRate-*.zip")))
    ts, rt = [], []
    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as z:
                raw = z.read(z.namelist()[0])
        except (zipfile.BadZipFile, IndexError, OSError):
            continue
        head = raw[:16].decode(errors="replace")
        df = pd.read_csv(io.BytesIO(raw), header=0 if head.startswith("calc_time") else None,
                         usecols=[0, 2], names=["t", "r"])
        ts.append(df["t"].to_numpy(np.int64))
        rt.append(df["r"].to_numpy(np.float64))
    if not ts:
        return np.array([], np.int64), np.array([], np.float64)
    t = np.concatenate(ts)
    t = np.where(t > US_SWITCH, t // 1000, t) // 1000
    r = np.concatenate(rt)
    order = np.argsort(t)
    return t[order], r[order]


def zscore(x, hist):
    """Scanner zscore: (x-mean)/pop-std, 0 when baseline short or flat."""
    if len(hist) < BASELINE_MIN:
        return 0.0
    m = hist.mean()
    var = ((hist - m) ** 2).mean()
    if var <= 0:
        return 0.0
    return (x - m) / var ** 0.5


def detect(bars):
    """Scanner pump flags on hourly bars: indices + z's + 24h dollar vol at flag."""
    ts, c, qv = bars["ts"], bars["c"], bars["qv"]
    n = len(ts)
    contig = np.concatenate([[False], np.diff(ts) == BAR_S])
    vol24 = np.convolve(qv, np.ones(VOL24_BARS), "full")[:n]
    events = []
    last_flag = -10 ** 12
    for t in range(VOL24_BARS, n):
        if not contig[t - BASELINE_N + 1:t + 1].all():
            continue
        if vol24[t] < MIN_VOL_USD:
            continue
        if ts[t] - last_flag < COOLDOWN_S:
            continue
        px_h = c[t - BASELINE_N:t]
        rets_h = px_h[1:] / px_h[:-1] - 1.0
        r_now = c[t] / c[t - 1] - 1.0
        price_z = round(zscore(r_now, rets_h), 2)
        if price_z < PRICE_Z_MIN:
            continue
        vol_z = round(zscore(qv[t], qv[t - BASELINE_N:t]), 2)
        if vol_z < VOL_Z_MIN:
            continue
        last_flag = ts[t]
        events.append({"t": t, "ts": int(ts[t]), "price_z": price_z, "vol_z": vol_z,
                       "vol24_usd": float(vol24[t]), "r_now": float(r_now)})
    return events


def fwd_returns(bars, t, horizons):
    """Close-to-close forward returns; NaN when target hour missing (gap)."""
    ts, c = bars["ts"], bars["c"]
    out = {}
    for h in horizons:
        j = t + h
        out[h] = float(c[j] / c[t] - 1.0) if j < len(ts) and ts[j] - ts[t] == h * BAR_S else np.nan
    return out


def boot_ci(vals, rng):
    """Bootstrap mean CI + P(mean<=0)."""
    v = np.asarray(vals, np.float64)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    means = rng.choice(v, (BOOT_N, len(v))).mean(axis=1)
    return {"n": int(len(v)), "mean": float(v.mean()),
            "ci_lo": float(np.percentile(means, 2.5)), "ci_hi": float(np.percentile(means, 97.5)),
            "p_mean_le_0": float((means <= 0).mean())}


def sim_trade(bars, fund, t_detect, horizon):
    """One fade: short at close[t+1], exit at +horizon hours or +15% stop.
    Returns per-unit pnl breakdown or None when entry bar missing."""
    ts, o, h, c = bars["ts"], bars["o"], bars["h"], bars["c"]
    e = t_detect + 1
    if e >= len(ts) or ts[e] - ts[t_detect] != BAR_S:
        return None
    entry = c[e]
    target = ts[e] + horizon * BAR_S
    stop_px = entry * (1.0 + STOP_ADV)
    j, exit_px, stopped, gapped = e + 1, None, False, False
    while j < len(ts):
        if h[j] >= stop_px:
            exit_px = max(stop_px, o[j])
            stopped, gapped = True, bool(o[j] > stop_px)
            break
        if ts[j] >= target:
            exit_px = c[j]
            break
        j += 1
    if exit_px is None:
        j = len(ts) - 1
        exit_px = c[j]
    mae = float(h[e + 1:j + 1].max() / entry - 1.0) if j > e else 0.0
    ft, fr = fund
    lo, hi = np.searchsorted(ft, ts[e], "right"), np.searchsorted(ft, ts[j], "right")
    fpnl = float(fr[lo:hi].sum())
    gross = float((entry - exit_px) / entry)
    net = gross - 2 * FEE_SIDE + fpnl
    return {"entry_ts": int(ts[e]), "exit_ts": int(ts[j]), "gross": gross, "net": net,
            "funding": fpnl, "n_fund": int(hi - lo), "stopped": stopped, "gapped": gapped,
            "mae": mae, "hold_h": int((ts[j] - ts[e]) // H_S)}


def run_strategy(panel, funding, events_by_sym, horizon, cap=True):
    """Time-ordered book sim: 1/N slots cap SLOTS, skip when full. Returns trades."""
    evs = [(ev["ts"], sym, ev) for sym, evl in events_by_sym.items() for ev in evl]
    evs.sort()
    open_until, trades, skipped = [], [], 0
    for ets, sym, ev in evs:
        open_until = [u for u in open_until if u > ets]
        if cap and len(open_until) >= SLOTS:
            skipped += 1
            continue
        tr = sim_trade(panel[sym], funding[sym], ev["t"], horizon)
        if tr is None:
            continue
        tr.update({"sym": sym, "vol24_usd": ev["vol24_usd"], "year": ev["year"]})
        open_until.append(tr["exit_ts"])
        trades.append(tr)
    return trades, skipped


def book_metrics(trades):
    """Per-year net on book + deployed, Sharpe on deployed days, maxDD, worst event."""
    out = {}
    for yr in YEARS:
        tt = [t for t in trades if t["year"] == yr]
        if not tt:
            out[str(yr)] = {"n": 0}
            continue
        daily = {}
        for t in tt:
            d = t["entry_ts"] // 86400
            daily[d] = daily.get(d, 0.0) + t["net"] * SLOT_FRAC
        dv = np.array(list(daily.values()))
        eq = np.cumsum(np.concatenate([[0.0], dv]))
        dd = float((eq - np.maximum.accumulate(eq)).min())
        exp_days = len(set(d for t in tt for d in range(t["entry_ts"] // 86400,
                                                        t["exit_ts"] // 86400 + 1)))
        dep = SLOT_FRAC * sum(t["hold_h"] for t in tt) / (exp_days * 24.0) if exp_days else 0.0
        shp = float(dv.mean() / dv.std() * np.sqrt(DAYS_YR)) if len(dv) > 1 and dv.std() > 0 else None
        worst = min(tt, key=lambda t: t["net"])
        out[str(yr)] = {
            "n": len(tt), "net_book_pct": float(dv.sum() * 100),
            "net_deployed_pct": float(np.mean([t["net"] for t in tt]) * 100),
            "sharpe_trade_days": shp, "maxdd_book_pct": float(dd * 100),
            "avg_deployed_frac_trade_days": float(dep),
            "stop_rate": float(np.mean([t["stopped"] for t in tt])),
            "worst_event_net_pct": float(worst["net"] * 100), "worst_event_sym": worst["sym"],
            "worst_event_book_pct": float(worst["net"] * SLOT_FRAC * 100),
            "mean_funding_pct": float(np.mean([t["funding"] for t in tt]) * 100),
            "funding_coverage": float(np.mean([t["n_fund"] > 0 for t in tt]))}
    return out


def random_null(panel, funding, events_by_sym, horizon, rng):
    """Exposure-matched null: same per-symbol-year trade counts, random entries."""
    tot = {yr: [] for yr in YEARS}
    counts = {}
    for sym, evl in events_by_sym.items():
        for ev in evl:
            counts.setdefault((sym, ev["year"]), 0)
            counts[(sym, ev["year"])] += 1
    elig = {}
    for (sym, yr) in counts:
        bars = panel[sym]
        ts = bars["ts"]
        n = len(ts)
        vol24 = np.convolve(bars["qv"], np.ones(VOL24_BARS), "full")[:n]
        yr_arr = ts.astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
        cand = np.where((yr_arr == yr) & (vol24 >= LIQ_FLOOR_USD) &
                        (np.arange(n) >= VOL24_BARS) & (np.arange(n) < n - 2))[0]
        elig[(sym, yr)] = cand
    per_tr = {yr: [] for yr in YEARS}
    for _ in range(NULL_SEEDS):
        yr_net = {yr: [] for yr in YEARS}
        for (sym, yr), k in counts.items():
            cand = elig[(sym, yr)]
            if len(cand) == 0:
                continue
            for t in rng.choice(cand, k):
                tr = sim_trade(panel[sym], funding[sym], int(t), horizon)
                if tr:
                    yr_net[yr].append(tr["net"])
        for yr in YEARS:
            if yr_net[yr]:
                tot[yr].append(sum(yr_net[yr]) * SLOT_FRAC)
                per_tr[yr].append(float(np.mean(yr_net[yr])))
    return {str(yr): {"null_book_mean_pct": float(np.mean(v) * 100),
                      "null_book_p5_pct": float(np.percentile(v, 5) * 100),
                      "null_book_p95_pct": float(np.percentile(v, 95) * 100),
                      "null_trade_mean_pct": float(np.mean(per_tr[yr]) * 100),
                      "null_trade_p5_pct": float(np.percentile(per_tr[yr], 5) * 100),
                      "null_trade_p95_pct": float(np.percentile(per_tr[yr], 95) * 100)}
            for yr, v in tot.items() if v}


def main():
    t0 = time.time()
    stats = {"smoke": "pump_fade", "ts": int(t0), "errors": []}
    rng = np.random.default_rng(BOOT_SEED)

    mapping = json.load(open(os.path.join(META_DIR, "symbol_map.json")))
    meme_set = set(json.load(open(os.path.join(META_DIR, "cg_meme.json"))))
    panel, funding, gaps = {}, {}, {}
    for inst, m in mapping.items():
        bars = load_symbol(m["binance"])
        if bars is None or len(bars["ts"]) < VOL24_BARS + BASELINE_N + 2:
            continue
        sym = inst.replace("-USDT-SWAP", "")
        panel[sym] = bars
        funding[sym] = load_funding(m["binance"])
        span = (bars["ts"][-1] - bars["ts"][0]) // BAR_S + 1
        gaps[sym] = 1.0 - len(bars["ts"]) / span
    stats["universe"] = {
        "okx_usdt_swaps_ranked": len(mapping), "with_binance_history": len(panel),
        "median_gap_frac": float(np.median(list(gaps.values()))),
        "max_gap_frac": float(max(gaps.values())),
        "bars_total": int(sum(len(b["ts"]) for b in panel.values()))}

    # detection (scanner replica, pre-registered thresholds)
    events_by_sym = {}
    for sym, bars in panel.items():
        evs = detect(bars)
        for ev in evs:
            ev["year"] = int(time.gmtime(ev["ts"]).tm_year)
            ev["meme"] = sym.replace("1000", "") in meme_set or sym in meme_set
        evs = [e for e in evs if e["year"] in YEARS]
        if evs:
            events_by_sym[sym] = evs
    all_ev = [(sym, ev) for sym, evl in events_by_sym.items() for ev in evl]
    stats["detection"] = {
        "events_total": len(all_ev),
        "events_per_year": {str(y): sum(1 for _, e in all_ev if e["year"] == y) for y in YEARS},
        "symbols_with_events": len(events_by_sym),
        "median_vol24_usd": float(np.median([e["vol24_usd"] for _, e in all_ev])),
        "share_vol24_below_1m": float(np.mean([e["vol24_usd"] < LIQ_FLOOR_USD for _, e in all_ev])),
        "meme_share": float(np.mean([e["meme"] for _, e in all_ev]))}

    # event study
    study = {}
    fr_all = {}
    for sym, ev in all_ev:
        fr = fwd_returns(panel[sym], ev["t"], STUDY_HORIZONS)
        fr_all[(sym, ev["ts"])] = fr
    vols = np.array([e["vol24_usd"] for _, e in all_ev])
    terc = np.percentile(vols, [33.3, 66.7]) if len(vols) else [0, 0]
    for h in STUDY_HORIZONS:
        hh = {}
        for yr in YEARS:
            vals = [fr_all[(s, e["ts"])][h] for s, e in all_ev if e["year"] == yr]
            hh[str(yr)] = boot_ci(vals, rng)
        hh["all"] = boot_ci([fr_all[(s, e["ts"])][h] for s, e in all_ev], rng)
        hh["terciles"] = {
            name: boot_ci([fr_all[(s, e["ts"])][h] for s, e in all_ev if sel(e["vol24_usd"])], rng)
            for name, sel in [("small", lambda v: v <= terc[0]),
                              ("mid", lambda v: terc[0] < v <= terc[1]),
                              ("large", lambda v: v > terc[1])]}
        hh["meme"] = boot_ci([fr_all[(s, e["ts"])][h] for s, e in all_ev if e["meme"]], rng)
        hh["non_meme"] = boot_ci([fr_all[(s, e["ts"])][h] for s, e in all_ev if not e["meme"]], rng)
        study[f"h{h}"] = hh
    stats["event_study"] = study
    stats["event_study_note"] = ("fwd rets are GROSS from detection close; negative mean = "
                                 "fade edge. terciles by 24h dollar vol at detection.")

    # strategy: tradeable = vol24 >= $1M
    tradeable = {sym: [e for e in evl if e["vol24_usd"] >= LIQ_FLOOR_USD]
                 for sym, evl in events_by_sym.items()}
    tradeable = {s: e for s, e in tradeable.items() if e}
    horizon_pick, train_net = None, {}
    for hcand in HORIZON_CANDS:
        trades, _ = run_strategy(panel, funding, tradeable, hcand)
        tn = sum(t["net"] * SLOT_FRAC for t in trades if t["year"] in TRAIN_YEARS)
        train_net[str(hcand)] = float(tn * 100)
        if horizon_pick is None or tn > train_net[str(horizon_pick)] / 100:
            horizon_pick = hcand
    stats["horizon_selection"] = {"train_net_book_pct": train_net, "picked_h": horizon_pick,
                                  "picked_on": TRAIN_YEARS}

    trades, skipped = run_strategy(panel, funding, tradeable, horizon_pick)
    stats["strategy"] = {"horizon_h": horizon_pick, "fee_bp_side": FEE_SIDE * 1e4,
                         "stop_adv": STOP_ADV, "slots": SLOTS,
                         "liq_floor_usd": LIQ_FLOOR_USD, "skipped_cap_full": skipped,
                         "per_year": book_metrics(trades)}
    mae = np.array([t["mae"] for t in trades])
    stats["squeeze_tail"] = {
        "mae_p50": float(np.percentile(mae, 50)), "mae_p90": float(np.percentile(mae, 90)),
        "mae_p95": float(np.percentile(mae, 95)), "mae_p99": float(np.percentile(mae, 99)),
        "mae_max": float(mae.max()), "stop_rate": float(np.mean([t["stopped"] for t in trades])),
        "gap_fill_rate": float(np.mean([t["gapped"] for t in trades])),
        "worst_event_book_pct": float(min(t["net"] for t in trades) * SLOT_FRAC * 100)}

    stats["null"] = random_null(panel, funding, tradeable, horizon_pick,
                                np.random.default_rng(NULL_SEED0))
    stats["null_note"] = ("null = same per-symbol-year counts as TRADEABLE events, random "
                          "entries, no slot cap; compare per-trade means to strategy "
                          "net_deployed_pct (book columns are not scale-matched: the real "
                          "book skipped cap-full events).")

    # survivor bar
    py = stats["strategy"]["per_year"]
    test_pos = all(py[str(y)].get("net_book_pct", 0) > 0 for y in TEST_YEARS
                   if py[str(y)].get("n", 0) > 0)
    shp_ok = all((py[str(y)].get("sharpe_trade_days") or 0) >= 1.0 for y in TEST_YEARS
                 if py[str(y)].get("n", 0) > 0)
    tail_ok = abs(stats["squeeze_tail"]["worst_event_book_pct"]) < EV_BOOK_LIMIT * 100
    stats["survivor_bar"] = {"test_years_net_positive": test_pos, "sharpe_ge_1": shp_ok,
                             "worst_event_lt_30pct_book": tail_ok,
                             "passed": bool(test_pos and shp_ok and tail_ok)}
    stats["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(stats, open(STATS_PATH, "w"), indent=1)
    print(json.dumps({k: stats[k] for k in ("universe", "detection", "horizon_selection",
                                            "survivor_bar", "wall_clock_s")}, indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        json.dump({"smoke": "pump_fade", "errors": [err]}, open(STATS_PATH, "w"), indent=1)
        print(err)
        raise SystemExit(1)
