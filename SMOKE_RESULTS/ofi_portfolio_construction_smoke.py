# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - portfolio-construction backtest over OFI direction+magnitude forecasts: signal-weighted
#   deadband (trade-to-target / trade-to-band-edge / fee-adaptive band), hysteresis entry-exit,
#   persistence filter; hyperparams selected on prior-year OOS preds only; falsifier: no
#   construction nets positive %/yr at 3.5bp/side in >=3 of 5 test years incl 2025 or 2026.
# - wall-clock estimate ~2.5h (GBM fits 6 walk-forward years x 2 horizons dominate). CPU only.
# SMOKE_RESULTS/ofi_portfolio_construction_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------------------------
# Constants

DATA_DIR = Path("/Users/mirach-00-usc1/Development/Veritate/extensions/installed/market/data/crypto_of")
CACHE_DIR = Path("/private/tmp/claude-501/-Users-mirach-00-usc1-Development-Veritate/10655374-b645-4bca-a6e1-0208baad89ab/scratchpad/crypto_of_cache")
OUT_JSON = Path("/Users/mirach-00-usc1/Development/Veritate/SMOKE_RESULTS/ofi_portfolio_construction_stats.json")

START = pd.Timestamp("2020-01-01")
START_MS = int(START.value // 1_000_000)
PRED_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
HORIZONS = {"15m": ("15min", pd.Timedelta("15min")), "1h": ("1h", pd.Timedelta("1h"))}
HORIZON_ORDER = ["15m", "1h"]

MS_PER_MIN = 60_000
US_TIME_THRESHOLD = 10**14
Z_WIN = 20
OFI_MOM_WIN = 3
RET_WINS = (1, 3, 12)
RV_WIN = 12
RV_LONG_WIN = 48
OFI_CENTER = 0.5
TWO_PI = 2.0 * np.pi
HOURS_PER_DAY = 24.0
DAY64 = np.timedelta64(1, "D")

FEES_BP = [1.0, 2.0, 2.6, 3.5, 5.0]
KEY_FEE_BP = 3.5
BP = 1e4
KEY_FEE_RET = KEY_FEE_BP / BP
PROB_MID = 0.5
EMBARGO_BARS = 2

W_MAX = 1.0
BAND_MIN = 0.02
BAND_MAX = 1.0
K_SWEEP = (5.0, 10.0, 20.0, 40.0, 80.0)
BAND_SWEEP = (0.1, 0.2, 0.3)
ADAPT_BAND_SCALES = (0.5, 1.0, 2.0)
CONF_HI_Q = 0.8
CONF_LO_QS = (0.0, 0.5, 0.6, 0.7)
MAG_GATE_Q = 0.8
PERSIST_QS = ((0.7, 0.7), (0.7, 0.8), (0.8, 0.7), (0.8, 0.8))
CONF_QS = (0.5, 0.6, 0.7, 0.8)
MAG_QS = (0.7, 0.8)

VARIANT_COMBOS = {
    "deadband_target": [{"k": k, "band": b} for k in K_SWEEP for b in BAND_SWEEP],
    "deadband_edge": [{"k": k, "band": b} for k in K_SWEEP for b in BAND_SWEEP],
    "deadband_adaptive": [{"k": k, "band_scale": c} for k in K_SWEEP for c in ADAPT_BAND_SCALES],
    "hysteresis": [{"conf_lo_q": q} for q in CONF_LO_QS],
    "persistence": [{"conf_q": a, "mag_q": b} for a, b in PERSIST_QS],
}

DAYS_PER_YEAR = 365.0
N_PAIRS_CAPITAL = 40.0
PCT = 100.0
RECENT_YEARS = ("2025", "2026")
MIN_SURVIVE_YEARS = 3

SEED = 0
GB_KW = {"random_state": SEED}
WALL_CAP_S = 3 * 3600
CAP_MARGIN_S = 20 * 60
PAIR_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None

FEATS = ["ofi", "ofi_z", "ofi_mom", "ret1_z", "ret3_z", "ret12_z", "range_z",
         "vol_z", "trades_z", "rv", "vol_ratio", "hour_sin", "hour_cos", "dow"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
       "volume": "sum", "trades": "sum", "taker_buy": "sum"}
SUM_COLS = ["volume", "trades", "taker_buy"]

T0 = time.time()

# ------------------------------------------------------------------------------------
# Functions


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def load_pair(pair):
    paths = {h: CACHE_DIR / f"{pair}_{h}.parquet" for h in HORIZONS}
    integ_path = CACHE_DIR / f"{pair}_integrity.json"
    if integ_path.exists() and all(p.exists() for p in paths.values()):
        return {h: pd.read_parquet(p) for h, p in paths.items()}
    df = pd.read_csv(DATA_DIR / f"{pair}.csv", engine="pyarrow")
    t = df["time"].to_numpy()
    df["time"] = np.where(t > US_TIME_THRESHOLD, t // 1000, t)
    df = df[df["time"] >= START_MS]
    deltas = df["time"].diff().dropna()
    gaps = deltas[deltas > MS_PER_MIN]
    integ = {"rows_1m": int(len(df)),
             "n_gaps": int(len(gaps)),
             "missing_minutes": int(((gaps - MS_PER_MIN) // MS_PER_MIN).sum()),
             "max_gap_minutes": float(deltas.max() / MS_PER_MIN) if len(deltas) else 0.0}
    df.index = pd.to_datetime(df["time"], unit="ms").astype("datetime64[ns]")
    out = {}
    for h, (rule, _) in HORIZONS.items():
        bars = df.resample(rule).agg(AGG)
        empty = df["close"].resample(rule).count() == 0
        bars.loc[empty, SUM_COLS] = np.nan
        bars.to_parquet(paths[h])
        out[h] = bars
    integ_path.write_text(json.dumps(integ))
    return out


def zroll(s):
    m = s.rolling(Z_WIN, min_periods=Z_WIN).mean()
    sd = s.rolling(Z_WIN, min_periods=Z_WIN).std()
    z = (s - m) / sd
    return z.where(np.isfinite(z))


def build_features(bars, pair_code):
    c = bars["close"]
    logc = np.log(c)
    vol = bars["volume"]
    ofi = (bars["taker_buy"] / vol - OFI_CENTER).where(vol != 0, 0.0)
    ret1 = logc.diff()
    rv = ret1.rolling(RV_WIN, min_periods=RV_WIN).std()
    rv_long = ret1.rolling(RV_LONG_WIN, min_periods=RV_LONG_WIN).std()
    vr = rv / rv_long
    idx = bars.index
    hour_frac = (idx.hour + idx.minute / 60.0) / HOURS_PER_DAY
    f = pd.DataFrame(index=idx)
    f["ofi"] = ofi
    f["ofi_z"] = zroll(ofi)
    f["ofi_mom"] = ofi.rolling(OFI_MOM_WIN, min_periods=OFI_MOM_WIN).mean()
    for w in RET_WINS:
        f[f"ret{w}_z"] = zroll(ret1.rolling(w, min_periods=w).sum())
    f["range_z"] = zroll((bars["high"] - bars["low"]) / c)
    f["vol_z"] = zroll(vol)
    f["trades_z"] = zroll(bars["trades"])
    f["rv"] = rv
    f["vol_ratio"] = vr.where(np.isfinite(vr))
    f["hour_sin"] = np.sin(TWO_PI * hour_frac)
    f["hour_cos"] = np.cos(TWO_PI * hour_frac)
    f["dow"] = idx.dayofweek.astype(np.float32)
    f["fwd_ret"] = logc.shift(-1) - logc
    f = f.dropna().astype(np.float32)
    f["y_dir"] = (f["fwd_ret"] > 0).astype(np.int8)
    f["y_mag"] = f["fwd_ret"].abs()
    f["pair"] = np.int16(pair_code)
    f["time"] = f.index.astype("datetime64[ns]")
    return f.reset_index(drop=True)


def quantile_basis(p, m):
    conf = np.abs(p - PROB_MID)
    return {"mag_med": float(np.median(m)),
            "conf_q": {str(q): float(np.quantile(conf, q)) for q in CONF_QS},
            "mag_q": {str(q): float(np.quantile(m, q)) for q in MAG_QS}}


def build_pivot(t, pair, fwd, p, m, mask, n_pairs):
    sub = pd.DataFrame({"time": t[mask], "pair": pair[mask], "p": p, "m": m,
                        "f": fwd[mask]})
    wide = sub.pivot(index="time", columns="pair")
    cols = range(n_pairs)
    P = wide["p"].reindex(columns=cols).to_numpy()
    F = wide["f"].reindex(columns=cols).to_numpy()
    return {"P": P, "M": wide["m"].reindex(columns=cols).to_numpy(),
            "F0": np.nan_to_num(F), "V": np.isfinite(P) & np.isfinite(F),
            "times": wide.index.to_numpy(), "thr": quantile_basis(p, m)}


def make_step(variant, prm, thr, n_pairs):
    if variant.startswith("deadband"):
        k = prm["k"]
        mag_med = thr["mag_med"]
        mode = variant.rsplit("_", 1)[1]

        def step(p, m, w, valid):
            tgt = np.clip(k * (p - PROB_MID) * (m / mag_med), -W_MAX, W_MAX)
            band = (np.clip(prm["band_scale"] * KEY_FEE_RET / m, BAND_MIN, BAND_MAX)
                    if mode == "adaptive" else prm["band"])
            dev = tgt - w
            move = valid & (np.abs(dev) > band)
            w2 = w.copy()
            dst = tgt if mode == "target" else tgt - np.sign(dev) * band
            w2[move] = dst[move]
            return w2
        return step
    if variant == "hysteresis":
        conf_hi = thr["conf_q"][str(CONF_HI_Q)]
        conf_lo = thr["conf_q"][str(prm["conf_lo_q"])] if prm["conf_lo_q"] > 0 else 0.0
        mag_thr = thr["mag_q"][str(MAG_GATE_Q)]

        def step(p, m, w, valid):
            conf = np.abs(p - PROB_MID)
            desired = np.sign(p - PROB_MID)
            hold = valid & (w != 0) & (conf >= conf_lo) & (desired == np.sign(w))
            enter = valid & (conf >= conf_hi) & (m >= mag_thr)
            return np.where(hold, w, np.where(enter, desired, 0.0))
        return step
    conf_thr = thr["conf_q"][str(prm["conf_q"])]
    mag_thr = thr["mag_q"][str(prm["mag_q"])]
    prev = [np.zeros(n_pairs)]

    def step(p, m, w, valid):
        conf = np.abs(p - PROB_MID)
        desired = np.sign(p - PROB_MID)
        fire = valid & (conf >= conf_thr) & (m >= mag_thr)
        fdir = np.where(fire, desired, 0.0)
        out = np.where(fire & (fdir == prev[0]), fdir, 0.0)
        prev[0] = fdir
        return out
    return step


def simulate(data, step):
    P, M, F0, V = data["P"], data["M"], data["F0"], data["V"]
    T, n = P.shape
    w = np.zeros(n)
    gross_t = np.zeros(T)
    sides_t = np.zeros(T)
    expo_t = np.zeros(T)
    for t in range(T):
        valid = V[t]
        drop = ~valid & (w != 0)
        if drop.any():
            sides_t[t] += np.abs(w[drop]).sum()
            w = np.where(drop, 0.0, w)
        w2 = step(P[t], M[t], w, valid)
        sides_t[t] += np.abs(w2 - w).sum()
        w = w2
        gross_t[t] = w @ F0[t]
        expo_t[t] = np.abs(w).sum()
    sides_t[T - 1] += np.abs(w).sum()
    return gross_t, sides_t, expo_t


def seg_metrics(gross_bp_sum, sides, n_days, deploy_mean):
    res = {"sides": round(float(sides), 1), "n_days": round(n_days, 1),
           "avg_deploy": round(deploy_mean / N_PAIRS_CAPITAL, 4)}
    if sides <= 0:
        res["gross_bp_per_side"] = 0.0
        res["net_bp_per_side"] = {str(f): 0.0 for f in FEES_BP}
        res["sides_per_day"] = 0.0
        res["pct_yr"] = {str(f): 0.0 for f in FEES_BP}
        return res
    gps = gross_bp_sum / sides
    ann = DAYS_PER_YEAR / max(n_days, 1.0)
    res["gross_bp_per_side"] = round(gps, 4)
    res["net_bp_per_side"] = {str(f): round(gps - f, 4) for f in FEES_BP}
    res["sides_per_day"] = round(sides / max(n_days, 1.0), 2)
    res["pct_yr"] = {str(f): round((gross_bp_sum - f * sides) / BP / N_PAIRS_CAPITAL * ann * PCT, 3)
                     for f in FEES_BP}
    return res


def year_metrics(gross_t, sides_t, expo_t, times):
    T = len(gross_t)
    nd = max(float((times[-1] - times[0]) / DAY64), 1.0)
    res = seg_metrics(BP * gross_t.sum(), sides_t.sum(), nd, float(expo_t.mean()))
    half = T // 2
    res["halves"] = {}
    for key, sl in (("h1", slice(0, half)), ("h2", slice(half, T))):
        seg_times = times[sl]
        seg_nd = max(float((seg_times[-1] - seg_times[0]) / DAY64), 1.0) if len(seg_times) > 1 else 1.0
        res["halves"][key] = seg_metrics(BP * gross_t[sl].sum(), sides_t[sl].sum(), seg_nd,
                                         float(expo_t[sl].mean()))
    return res


def sel_score(gross_t, sides_t, times):
    nd = max(float((times[-1] - times[0]) / DAY64), 1.0)
    return float((BP * gross_t.sum() - KEY_FEE_BP * sides_t.sum())
                 / BP / N_PAIRS_CAPITAL * (DAYS_PER_YEAR / nd) * PCT)


def run_horizon(df, hname, stats):
    hd64 = HORIZONS[hname][1].to_timedelta64()
    t_arr = df["time"].to_numpy()
    pair_arr = df["pair"].to_numpy()
    fwd = df["fwd_ret"].to_numpy(dtype=np.float64)
    y = df["y_dir"].to_numpy()
    ymag = df["y_mag"].to_numpy(dtype=np.float64)
    X_all = df[FEATS].to_numpy(dtype=np.float32)
    n_pairs = int(pair_arr.max()) + 1
    pivots, models, aucs = {}, {}, {}
    for yr in PRED_YEARS:
        y0 = np.datetime64(f"{yr}-01-01")
        y1 = np.datetime64(f"{yr + 1}-01-01")
        tr = t_arr <= (y0 - EMBARGO_BARS * hd64)
        te = (t_arr >= y0) & (t_arr < y1)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        gbc = HistGradientBoostingClassifier(**GB_KW).fit(X_all[tr], y[tr])
        gbr = HistGradientBoostingRegressor(**GB_KW).fit(X_all[tr], ymag[tr])
        p = gbc.predict_proba(X_all[te])[:, 1]
        m = gbr.predict(X_all[te])
        aucs[str(yr)] = round(float(roc_auc_score(y[te], p)), 4)
        pivots[yr] = build_pivot(t_arr, pair_arr, fwd, p, m, te, n_pairs)
        models[yr] = (gbc, gbr)
        log(f"{hname} {yr}: fit+pred train={int(tr.sum())} test={int(te.sum())} auc={aucs[str(yr)]}")
    hstats = {v: {} for v in VARIANT_COMBOS}
    hstats["auc"] = aucs
    for yr in TEST_YEARS:
        if yr not in pivots or (yr - 1) not in pivots:
            continue
        b0 = np.datetime64(f"{yr - 1}-01-01")
        b1 = np.datetime64(f"{yr}-01-01")
        bmask = (t_arr >= b0) & (t_arr < b1)
        gbc, gbr = models[yr]
        test_thr = quantile_basis(gbc.predict_proba(X_all[bmask])[:, 1], gbr.predict(X_all[bmask]))
        sel, test = pivots[yr - 1], pivots[yr]
        for variant, combos in VARIANT_COMBOS.items():
            best_prm, best_score = combos[0], -np.inf
            for prm in combos:
                g, s, _ = simulate(sel, make_step(variant, prm, sel["thr"], n_pairs))
                score = sel_score(g, s, sel["times"])
                if score > best_score:
                    best_prm, best_score = prm, score
            g, s, e = simulate(test, make_step(variant, best_prm, test_thr, n_pairs))
            res = year_metrics(g, s, e, test["times"])
            res["params"] = best_prm
            res["sel_pct_yr"] = round(best_score, 3)
            hstats[variant][str(yr)] = res
        log(f"{hname} {yr}: variants done")
    stats["horizons"][hname] = hstats


def scan_survivors(stats):
    out = []
    for hname, hst in stats["horizons"].items():
        for variant in VARIANT_COMBOS:
            years = hst.get(variant, {})
            for f in FEES_BP:
                key = str(f)
                pos = sorted(yr for yr, c in years.items() if c["pct_yr"][key] > 0)
                if len(pos) < MIN_SURVIVE_YEARS:
                    continue
                halves_pos = sorted(yr for yr in pos if all(
                    years[yr]["halves"][h]["net_bp_per_side"][key] > 0 for h in ("h1", "h2")))
                out.append({"horizon": hname, "variant": variant, "fee_bp": f,
                            "years_net_pos": pos, "years_both_halves_pos": halves_pos,
                            "includes_2025_or_2026": any(yr in RECENT_YEARS for yr in pos)})
    return out


def print_summary(stats):
    for hname, hst in stats["horizons"].items():
        print(f"\nauc {hname}: {hst.get('auc')}")
        for variant in VARIANT_COMBOS:
            years = hst.get(variant, {})
            if not years:
                continue
            print(f"\n== {hname} {variant} ==")
            print("year  params                         sel%yr  sides/d  deploy  gr/side"
                  "    net@1    net@2  net@2.6  net@3.5    net@5  %yr@3.5   h1%yr   h2%yr")
            for yr in sorted(years):
                c = years[yr]
                nb = c["net_bp_per_side"]
                h = c["halves"]
                print(f"{yr}  {json.dumps(c['params']):29s} {c['sel_pct_yr']:7.2f} "
                      f"{c['sides_per_day']:8.1f} {c['avg_deploy']:7.3f} {c['gross_bp_per_side']:8.3f} "
                      f"{nb['1.0']:8.2f} {nb['2.0']:8.2f} {nb['2.6']:8.2f} {nb['3.5']:8.2f} "
                      f"{nb['5.0']:8.2f} {c['pct_yr']['3.5']:8.2f} "
                      f"{h['h1']['pct_yr']['3.5']:7.2f} {h['h2']['pct_yr']['3.5']:7.2f}")
    print(f"\nsurvivors: {json.dumps(stats['survivors'], indent=1)}", flush=True)


def dump(stats):
    OUT_JSON.write_text(json.dumps(stats, indent=1))


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"errors": [], "notes": [], "config": {
        "feats": FEATS, "fees_bp": FEES_BP, "key_fee_bp": KEY_FEE_BP,
        "pred_years": PRED_YEARS, "test_years": TEST_YEARS, "embargo_bars": EMBARGO_BARS,
        "variants": {k: len(v) for k, v in VARIANT_COMBOS.items()},
        "k_sweep": K_SWEEP, "band_sweep": BAND_SWEEP, "adapt_band_scales": ADAPT_BAND_SCALES,
        "conf_hi_q": CONF_HI_Q, "conf_lo_qs": CONF_LO_QS, "mag_gate_q": MAG_GATE_Q,
        "persist_qs": PERSIST_QS,
        "selection": "hyperparams argmax pct_yr@key_fee on prior-year OOS preds (prior model)",
        "thresholds": "gate quantiles + mag scale frozen from deployed model's preds on year Y-1"},
        "pairs": [], "horizons": {}, "survivors": []}
    try:
        pairs = sorted(p.stem for p in DATA_DIR.glob("*.csv"))[:PAIR_LIMIT]
        stats["pairs"] = pairs
        feats = {h: [] for h in HORIZONS}
        for i, pair in enumerate(pairs):
            bars_map = load_pair(pair)
            for h in HORIZONS:
                feats[h].append(build_features(bars_map[h], i))
            log(f"loaded {pair} ({i + 1}/{len(pairs)})")
        dfs = {h: pd.concat(feats[h], ignore_index=True) for h in HORIZONS}
        feats.clear()
        for hname in HORIZON_ORDER:
            if time.time() - T0 > WALL_CAP_S - CAP_MARGIN_S:
                stats["notes"].append(f"{hname} skipped: wall-clock cap")
                continue
            run_horizon(dfs[hname], hname, stats)
            dump(stats)
        stats["survivors"] = scan_survivors(stats)
        dump(stats)
        print_summary(stats)
    except Exception:
        stats["errors"].append(traceback.format_exc())
        dump(stats)
        print(stats["errors"][-1], flush=True)
        sys.exit(1)


main()
