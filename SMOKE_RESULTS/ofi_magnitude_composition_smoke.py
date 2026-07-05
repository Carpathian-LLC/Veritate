# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - composition backtest: order-flow direction signal x predicted-magnitude gating, fee sweep to
#   maker level; falsifier: no (horizon, gate, fee<=3.5bp/side) cell nets positive in >=3 of 5
#   test years and both temporal halves.
# - wall-clock estimate ~2.5h (40x 1m CSV load ~15min, 1h grid ~40min, 15m grid ~90min). CPU only.
# SMOKE_RESULTS/ofi_magnitude_composition_smoke.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------------
# Constants

DATA_DIR = Path("/Users/mirach-00-usc1/Development/Veritate/extensions/installed/market/data/crypto_of")
CACHE_DIR = Path("/private/tmp/claude-501/-Users-mirach-00-usc1-Development-Veritate/10655374-b645-4bca-a6e1-0208baad89ab/scratchpad/crypto_of_cache")
OUT_JSON = Path("/Users/mirach-00-usc1/Development/Veritate/SMOKE_RESULTS/ofi_magnitude_composition_stats.json")

START = pd.Timestamp("2020-01-01")
START_MS = int(START.value // 1_000_000)
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
HORIZONS = {"1h": ("1h", pd.Timedelta("1h")), "15m": ("15min", pd.Timedelta("15min"))}
HORIZON_ORDER = ["1h", "15m"]

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

FEES_BP = [0.0, 1.0, 2.0, 3.5, 5.0, 10.0]
KEY_FEE_BP = 3.5
BP = 1e4
MAG_GATES = {"mag_q80": 0.80, "mag_q90": 0.90}
CONF_GATE_Q = 0.80
PROB_MID = 0.5
EMBARGO_BARS = 2

CS_TOP_N = 8
CS_MIN_PAIRS = 16
BTC_PAIR = "BTCUSDT"
FUNDING_RATE_YR = 0.10
HOURS_PER_YEAR = 8760.0
DAYS_PER_YEAR = 365.0
N_PAIRS_CAPITAL = 40.0
PCT = 100.0

SEED = 0
LOGREG_KW = {"max_iter": 200, "tol": 1e-3}
GB_KW = {"random_state": SEED}
WALL_CAP_S = 4 * 3600
CAP_MARGIN_S = 30 * 60
MIN_SURVIVE_YEARS = 3
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
        return {h: pd.read_parquet(p) for h, p in paths.items()}, json.loads(integ_path.read_text())
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
        integ[f"empty_bins_{h}"] = int(empty.sum())
        bars.to_parquet(paths[h])
        out[h] = bars
    integ_path.write_text(json.dumps(integ))
    return out, integ


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


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy() * 1.0
    rb = pd.Series(b).rank().to_numpy() * 1.0
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def turnover_sides(t, pair, pos, hd64):
    if len(pos) == 0:
        return 0.0
    contig = (pair[1:] == pair[:-1]) & ((t[1:] - t[:-1]) == hd64)
    prev = np.zeros_like(pos)
    prev[1:][contig] = pos[:-1][contig]
    opens = np.abs(pos - prev).sum()
    next_contig = np.concatenate((contig, [False]))
    closes = np.abs(pos[~next_contig]).sum()
    return float(opens + closes)


def cell_stats(t, pair, fwd, pos, hd64):
    held = pos != 0
    n = int(held.sum())
    res = {"n": n, "n_bars": int(len(pos))}
    if n == 0:
        return res
    gross_sum_bp = float(BP * (pos * fwd).sum())
    to = turnover_sides(t, pair, pos, hd64)
    n_days = max(float((t.max() - t.min()) / DAY64), 1.0)
    y_up = fwd > 0
    res["acc"] = round(float(((pos > 0) == y_up)[held].mean()), 4)
    res["gross_bp"] = round(gross_sum_bp / n, 4)
    res["turnover_per_trade"] = round(to / n, 4)
    res["trades_per_day"] = round(n / n_days, 2)
    res["net_bp"] = {str(f): round((gross_sum_bp - f * to) / n, 4) for f in FEES_BP}
    res["pct_yr_at_key_fee"] = round(
        (gross_sum_bp - KEY_FEE_BP * to) / BP / N_PAIRS_CAPITAL * (DAYS_PER_YEAR / n_days) * PCT, 3)
    return res


def cell_with_halves(t, pair, fwd, pos, hd64):
    res = cell_stats(t, pair, fwd, pos, hd64)
    med = pd.Series(t).median()
    res["halves"] = {}
    for key, mask in (("h1", t < med), ("h2", t >= med)):
        res["halves"][key] = cell_stats(t[mask], pair[mask], fwd[mask], pos[mask], hd64)
    return res


def run_horizon(df, hname, stats):
    hd64 = HORIZONS[hname][1].to_timedelta64()
    t_arr = df["time"].to_numpy()
    pair_arr = df["pair"].to_numpy()
    fwd = df["fwd_ret"].to_numpy(dtype=np.float64)
    y = df["y_dir"].to_numpy()
    ymag = df["y_mag"].to_numpy(dtype=np.float64)
    X_all = df[FEATS].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(SEED)
    hstats = {"logreg": {}, "gbm": {}, "magnitude": {}}
    nulls = {}
    preds = {}
    for yr in TEST_YEARS:
        y0 = np.datetime64(f"{yr}-01-01")
        y1 = np.datetime64(f"{yr + 1}-01-01")
        tr_mask = t_arr <= (y0 - EMBARGO_BARS * hd64)
        te_mask = (t_arr >= y0) & (t_arr < y1)
        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            continue
        Xtr, Xte = X_all[tr_mask], X_all[te_mask]
        log(f"{hname} {yr}: fit train={tr_mask.sum()} test={te_mask.sum()}")
        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(**LOGREG_KW).fit(scaler.transform(Xtr), y[tr_mask])
        p_lr = lr.predict_proba(scaler.transform(Xte))[:, 1]
        gbc = HistGradientBoostingClassifier(**GB_KW).fit(Xtr, y[tr_mask])
        p_gb = gbc.predict_proba(Xte)[:, 1]
        gbr = HistGradientBoostingRegressor(**GB_KW).fit(Xtr, ymag[tr_mask])
        m_pred = gbr.predict(Xte)
        gbs = HistGradientBoostingClassifier(**GB_KW).fit(Xtr, rng.permutation(y[tr_mask]))
        nulls[str(yr)] = round(float(roc_auc_score(y[te_mask], gbs.predict_proba(Xte)[:, 1])), 4)
        hstats["magnitude"][str(yr)] = {
            "spearman_pred_vs_realized": round(spearman(m_pred, ymag[te_mask]), 4)}
        gm = {name: m_pred >= np.quantile(m_pred, q) for name, q in MAG_GATES.items()}
        tt, pp, ff = t_arr[te_mask], pair_arr[te_mask], fwd[te_mask]
        for mname, p in (("logreg", p_lr), ("gbm", p_gb)):
            conf = np.abs(p - PROB_MID)
            conf_gate = conf >= np.quantile(conf, CONF_GATE_Q)
            base_pos = np.sign(p - PROB_MID)
            gates = {"none": np.ones(len(p), dtype=bool),
                     "mag_q80": gm["mag_q80"],
                     "mag_q90": gm["mag_q90"],
                     "conf_q80": conf_gate,
                     "mag80_conf80": gm["mag_q80"] & conf_gate}
            for gname, gmask in gates.items():
                pos = np.where(gmask, base_pos, 0.0)
                cell = cell_with_halves(tt, pp, ff, pos, hd64)
                if gname == "none":
                    cell["auc"] = round(float(roc_auc_score(y[te_mask], p)), 4)
                hstats[mname].setdefault(gname, {})[str(yr)] = cell
        preds[yr] = (np.flatnonzero(te_mask), {"logreg": p_lr, "gbm": p_gb})
        log(f"{hname} {yr}: done (null auc {nulls[str(yr)]})")
    stats["horizons"][hname] = hstats
    stats["nulls"][hname] = nulls
    return preds


def cs_arm_stats(W, R, hd64, short_notional):
    if len(W) == 0:
        return {"n_bars": 0}
    Wv = W.to_numpy(dtype=np.float64)
    Rv = np.nan_to_num(R.to_numpy(dtype=np.float64))
    tv = W.index.to_numpy()
    ret_bp = BP * (Wv * Rv).sum(axis=1)
    contig = (tv[1:] - tv[:-1]) == hd64
    prev = np.zeros_like(Wv)
    prev[1:][contig] = Wv[:-1][contig]
    opens = np.abs(Wv - prev).sum()
    next_contig = np.concatenate((contig, [False]))
    closes = np.abs(Wv[~next_contig]).sum()
    n = len(ret_bp)
    if n == 0:
        return {"n_bars": 0}
    to_per_bar = (opens + closes) / n
    n_days = max(float((tv.max() - tv.min()) / DAY64), 1.0)
    fund_bp = BP * FUNDING_RATE_YR * short_notional / HOURS_PER_YEAR
    gross = float(ret_bp.mean())
    res = {"n_bars": n, "gross_bp": round(gross, 4),
           "turnover_per_bar": round(to_per_bar, 4),
           "funding_bp_per_bar": round(fund_bp, 4),
           "net_bp": {str(f): round(gross - f * to_per_bar, 4) for f in FEES_BP},
           "net_bp_funding": {str(f): round(gross - f * to_per_bar - fund_bp, 4) for f in FEES_BP},
           "pct_yr_at_key_fee": round(
               (gross - KEY_FEE_BP * to_per_bar - fund_bp) * n / n_days * DAYS_PER_YEAR / BP * PCT, 3)}
    half = n // 2
    res["halves"] = {}
    for key, sl in (("h1", slice(0, half)), ("h2", slice(half, n))):
        seg = ret_bp[sl]
        res["halves"][key] = {"n_bars": len(seg),
                              "gross_bp": round(float(seg.mean()), 4),
                              "net_bp": {str(f): round(float(seg.mean()) - f * to_per_bar, 4)
                                         for f in FEES_BP}}
    return res


def cross_sectional(df, preds, stats, btc_code):
    hd64 = HORIZONS["1h"][1].to_timedelta64()
    cs = {"long_short": {}, "long_only": {}, "btc_long": {}}
    for yr, (idx, pmap) in preds.items():
        sub = df.iloc[idx]
        btc = sub[sub["pair"] == btc_code]
        if len(btc):
            mean_bp = float(BP * btc["fwd_ret"].astype(np.float64).mean())
            n_days = max(float((btc["time"].max() - btc["time"].min()) / pd.Timedelta(days=1)), 1.0)
            cs["btc_long"][str(yr)] = {
                "n_bars": int(len(btc)), "gross_bp": round(mean_bp, 4),
                "pct_yr": round(mean_bp * len(btc) / n_days * DAYS_PER_YEAR / BP * PCT, 3)}
        for mname, p in pmap.items():
            frame = pd.DataFrame({"time": sub["time"].to_numpy(), "pair": sub["pair"].to_numpy(),
                                  "p": p, "fwd": sub["fwd_ret"].to_numpy(dtype=np.float64)})
            P = frame.pivot(index="time", columns="pair", values="p")
            R = frame.pivot(index="time", columns="pair", values="fwd")
            cnt = P.notna().sum(axis=1)
            ok = cnt >= CS_MIN_PAIRS
            P, R, cnt = P[ok], R[ok], cnt[ok]
            rk = P.rank(axis=1)
            top = rk.ge(cnt - (CS_TOP_N - 1), axis=0)
            bot = rk.le(CS_TOP_N)
            w_ls = (top.astype(float) - bot.astype(float)) / (2 * CS_TOP_N)
            w_lo = top.astype(float) / CS_TOP_N
            cs["long_short"].setdefault(mname, {})[str(yr)] = cs_arm_stats(w_ls, R, hd64, 0.5)
            cs["long_only"].setdefault(mname, {})[str(yr)] = cs_arm_stats(w_lo, R, hd64, 0.0)
        log(f"cross-sectional {yr}: done")
    stats["cross_sectional"] = cs


def scan_survivors(stats):
    out = []

    def qualifies(cell, fee_key):
        if cell.get("net_bp", {}).get(fee_key, -1.0) <= 0:
            return False
        halves = cell.get("halves", {})
        return all(halves.get(k, {}).get("net_bp", {}).get(fee_key, -1.0) > 0 for k in ("h1", "h2"))

    for hname, hst in stats["horizons"].items():
        for mname in ("logreg", "gbm"):
            for gname, years in hst.get(mname, {}).items():
                for f in FEES_BP:
                    key = str(f)
                    wins = sorted(yr for yr, c in years.items() if qualifies(c, key))
                    if f <= KEY_FEE_BP and len(wins) >= MIN_SURVIVE_YEARS:
                        out.append({"kind": "per_pair", "horizon": hname, "model": mname,
                                    "gate": gname, "fee_bp": f, "years": wins})
    for arm in ("long_short", "long_only"):
        for mname, years in stats.get("cross_sectional", {}).get(arm, {}).items():
            for f in FEES_BP:
                key = str(f)
                wins = sorted(yr for yr, c in years.items() if qualifies(c, key))
                if f <= KEY_FEE_BP and len(wins) >= MIN_SURVIVE_YEARS:
                    out.append({"kind": "cross_sectional", "arm": arm, "model": mname,
                                "fee_bp": f, "years": wins})
    return out


def print_summary(stats):
    for hname, hst in stats["horizons"].items():
        for mname in ("logreg", "gbm"):
            for gname, years in hst.get(mname, {}).items():
                print(f"\n== {hname} {mname} {gname} ==", flush=True)
                print("year  n       tpd    auc    acc    gross    net@1    net@2  net@3.5    net@5   net@10  h1@3.5  h2@3.5")
                for yr in sorted(years):
                    c = years[yr]
                    if c.get("n", 0) == 0:
                        continue
                    nb = c["net_bp"]
                    h = c["halves"]
                    print(f"{yr}  {c['n']:7d} {c['trades_per_day']:6.1f} "
                          f"{c.get('auc', float('nan')):6.3f} {c['acc']:6.3f} "
                          f"{c['gross_bp']:8.2f} {nb['1.0']:8.2f} {nb['2.0']:8.2f} "
                          f"{nb['3.5']:8.2f} {nb['5.0']:8.2f} {nb['10.0']:8.2f} "
                          f"{h['h1'].get('net_bp', {}).get('3.5', float('nan')):7.2f} "
                          f"{h['h2'].get('net_bp', {}).get('3.5', float('nan')):7.2f}")
    for arm in ("long_short", "long_only"):
        for mname, years in stats.get("cross_sectional", {}).get(arm, {}).items():
            print(f"\n== CS 1h {arm} {mname} ==")
            print("year  n_bars   to/bar    gross    net@1    net@2  net@3.5    net@5   net@10  pct_yr@3.5")
            for yr in sorted(years):
                c = years[yr]
                if c.get("n_bars", 0) == 0:
                    continue
                nb = c["net_bp"]
                print(f"{yr}  {c['n_bars']:7d} {c['turnover_per_bar']:7.3f} {c['gross_bp']:8.3f} "
                      f"{nb['1.0']:8.3f} {nb['2.0']:8.3f} {nb['3.5']:8.3f} {nb['5.0']:8.3f} "
                      f"{nb['10.0']:8.3f} {c['pct_yr_at_key_fee']:10.2f}")
    print(f"\nBTC always-long: {stats.get('cross_sectional', {}).get('btc_long', {})}")
    print(f"nulls (shuffled-label auc): {stats['nulls']}")
    print(f"survivors: {json.dumps(stats['survivors'], indent=1)}", flush=True)


def dump(stats):
    OUT_JSON.write_text(json.dumps(stats, indent=1))


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"errors": [], "notes": [], "config": {"feats": FEATS, "fees_bp": FEES_BP,
             "test_years": TEST_YEARS, "mag_gates": MAG_GATES, "conf_gate_q": CONF_GATE_Q,
             "embargo_bars": EMBARGO_BARS, "gate_quantiles_within_test_year": True},
             "pairs": [], "data_integrity": {}, "horizons": {}, "nulls": {},
             "cross_sectional": {}, "survivors": []}
    try:
        pairs = sorted(p.stem for p in DATA_DIR.glob("*.csv"))[:PAIR_LIMIT]
        stats["pairs"] = pairs
        feats = {h: [] for h in HORIZONS}
        for i, pair in enumerate(pairs):
            bars_map, integ = load_pair(pair)
            stats["data_integrity"][pair] = integ
            for h in HORIZONS:
                feats[h].append(build_features(bars_map[h], i))
            log(f"loaded {pair} ({i + 1}/{len(pairs)}) rows_1m={integ['rows_1m']} "
                f"missing_min={integ['missing_minutes']} max_gap_min={integ['max_gap_minutes']:.0f}")
        dfs = {h: pd.concat(feats[h], ignore_index=True) for h in HORIZONS}
        feats.clear()
        preds_1h = run_horizon(dfs["1h"], "1h", stats)
        dump(stats)
        cross_sectional(dfs["1h"], preds_1h, stats, pairs.index(BTC_PAIR) if BTC_PAIR in pairs else -1)
        dump(stats)
        if time.time() - T0 < WALL_CAP_S - CAP_MARGIN_S:
            run_horizon(dfs["15m"], "15m", stats)
        else:
            stats["notes"].append("15m skipped: wall-clock cap")
        stats["survivors"] = scan_survivors(stats)
        dump(stats)
        print_summary(stats)
    except Exception:
        stats["errors"].append(traceback.format_exc())
        dump(stats)
        print(stats["errors"][-1], flush=True)
        sys.exit(1)


main()
