# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests pre-registered regime conditioning of 1w XS momentum L/S (40 Binance USDT
#   majors, 7d lookback train-fixed, quintile L/S, realized funding, fees 2.6/3.5/5bp
#   per side): 9 candidate variants evaluated on pre-2024 train only, selection rule
#   fixed a priori (vol_scaled always + top-2 of 8 filter variants by train Sharpe at
#   3.5bp), only selected variants scored on 2024-2026-05 test. Falsifier: no selected
#   variant reaches test Sharpe >= 0.8 with 2024/2025/2026 all positive at 3.5bp/side
#   and maxDD < 40% -> family stays real-but-thin, closed.
# - wall-clock estimate ~2 min (loaders + npz caches reused from costkilled_retest).
# SMOKE_RESULTS/xsmom_regime_filter_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import costkilled_retest_smoke as ck

# ------------------------------------------------------------------------------------
# Constants

STATS_PATH = os.path.join(ck.REPO, "SMOKE_RESULTS/xsmom_regime_filter_stats.json")

LOOKBACK_D = 7
BTC_PAIR = "BTCUSDT"
DISP_SMOOTH_D = 30
FUND_SMOOTH_D = 30
MED_WINDOW_D = 365
MED_MIN_D = 90
BTC_VOL_WINDOW_D = 30
TREND_MA_D = 100
MOMOM_WINDOW_D = 90
VS_VOL_WINDOW_D = 30
VS_MAX_LEV = 2.0
N_TOP_FILTERS = 2
NULL_MIN_SHIFT_W = 4

# ------------------------------------------------------------------------------------
# Functions


def weekly_reb(m, Td):
    weekday = (np.arange(m["d0"], m["d0"] + Td) + ck.EPOCH_THU) % 7
    return weekday == 0


def above_trailing_median(x, above):
    med = pd.Series(x).rolling(MED_WINDOW_D, min_periods=MED_MIN_D).median().to_numpy()
    ok = np.isfinite(x) & np.isfinite(med)
    on = np.zeros(len(x), bool)
    on[ok] = (x[ok] > med[ok]) if above else (x[ok] < med[ok])
    return on


def gate_series(reb, val, default):
    T = len(reb)
    G = np.zeros(T)
    g = 0.0
    for t in range(T):
        G[t] = g
        if reb[t]:
            g = float(val[t]) if np.isfinite(val[t]) else default
    return G


def build_candidates(m, C, R, Fd, reb, train_mask, base_gross, base_net):
    T, N = C.shape
    j_btc = m["pairs"].index(BTC_PAIR)
    cands = {}

    mom = np.full_like(C, np.nan)
    mom[LOOKBACK_D:] = C[LOOKBACK_D:] / C[:-LOOKBACK_D] - 1.0
    n_av = np.isfinite(mom).sum(axis=1)
    rows = n_av >= ck.MIN_UNIVERSE_D
    xs_std = np.full(T, np.nan)
    xs_std[rows] = np.nanstd(mom[rows], axis=1)
    disp = pd.Series(xs_std).rolling(DISP_SMOOTH_D, min_periods=DISP_SMOOTH_D).mean().to_numpy()
    cands["dispersion_high"] = above_trailing_median(disp, True)

    r_btc = pd.Series(R[:, j_btc])
    vol_btc = r_btc.rolling(BTC_VOL_WINDOW_D, min_periods=BTC_VOL_WINDOW_D).std().to_numpy()
    cands["btc_vol_high"] = above_trailing_median(vol_btc, True)
    cands["btc_vol_low"] = above_trailing_median(vol_btc, False)

    ma = pd.Series(C[:, j_btc]).rolling(TREND_MA_D, min_periods=TREND_MA_D).mean().to_numpy()
    ok = np.isfinite(C[:, j_btc]) & np.isfinite(ma)
    up = np.zeros(T, bool)
    up[ok] = C[ok, j_btc] > ma[ok]
    dn = np.zeros(T, bool)
    dn[ok] = C[ok, j_btc] < ma[ok]
    cands["btc_above_100dma"] = up
    cands["btc_below_100dma"] = dn

    momom = pd.Series(base_net).rolling(MOMOM_WINDOW_D, min_periods=MOMOM_WINDOW_D).mean().to_numpy()
    fm = np.zeros(T, bool)
    fm[np.isfinite(momom)] = momom[np.isfinite(momom)] > 0
    cands["factor_mom_90d_pos"] = fm

    pos = (Fd > 0).sum(axis=1).astype(float)
    nz = (Fd != 0).sum(axis=1).astype(float)
    share = np.where(nz > 0, pos / np.maximum(nz, 1.0), np.nan)
    share_sm = pd.Series(share).rolling(FUND_SMOOTH_D, min_periods=FUND_SMOOTH_D).mean().to_numpy()
    cands["funding_share_high"] = above_trailing_median(share_sm, True)
    cands["funding_share_low"] = above_trailing_median(share_sm, False)

    fvol = pd.Series(base_gross).rolling(VS_VOL_WINDOW_D, min_periods=VS_VOL_WINDOW_D).std().to_numpy()
    okv = np.isfinite(fvol) & (fvol > 0)
    target = float(np.nanmean(fvol[train_mask & okv]))
    sval = np.full(T, np.nan)
    sval[okv] = np.clip(target / fvol[okv], 0.0, VS_MAX_LEV)
    return cands, sval, target


def shift_gate_null(HOLD, R, Fd, day_ms, test_mask, reb, on):
    ridx = np.where(reb)[0]
    gvals = on[ridx].astype(float)
    n_reb = len(ridx)
    obs = None
    sharpes, clears, clears_either = [], [], []
    for k in range(n_reb):
        g = np.roll(gvals, k)
        pair = []
        for gv in (g, 1.0 - g):
            val = np.full(len(reb), np.nan)
            val[ridx] = gv
            G = gate_series(reb, val, 0.0)
            net, _ = ck.eval_book(HOLD * G[:, None], R, Fd, ck.TUNE_FEE_BP * ck.BP)
            pf = ck.perf(day_ms[test_mask], net[test_mask])
            pair.append(pf)
        if k == 0:
            obs = pair[0]
            continue
        if min(k, n_reb - k) < NULL_MIN_SHIFT_W:
            continue
        sharpes.append(pair[0]["sharpe"])
        clears.append(ck.survives(pair[0]))
        clears_either.append(ck.survives(pair[0]) or ck.survives(pair[1]))
    sharpes = np.array(sharpes)
    return {"n_shifts": int(len(sharpes)), "observed_sharpe": round(obs["sharpe"], 3),
            "mean_sharpe": round(float(sharpes.mean()), 3), "sd_sharpe": round(float(sharpes.std()), 3),
            "frac_sharpe_ge_observed": round(float((sharpes >= obs["sharpe"]).mean()), 4),
            "frac_bar_clear": round(float(np.mean(clears)), 4),
            "frac_bar_clear_either_complement": round(float(np.mean(clears_either)), 4),
            "note": "circular shifts of the gate at weekly rebalance granularity (min offset %d weeks); either-complement matches the actual best-of-two-signs selection exposure" % NULL_MIN_SHIFT_W}


def eval_variant(HOLD, R, Fd, day_ms, train_mask, test_mask):
    out = {}
    for fee_bp in ck.FEES_BP:
        net, turn = ck.eval_book(HOLD, R, Fd, fee_bp * ck.BP)
        pf = ck.perf(day_ms[test_mask], net[test_mask])
        if fee_bp == ck.TUNE_FEE_BP:
            pf["train"] = ck.perf(day_ms[train_mask], net[train_mask])
            yrs = (day_ms[test_mask][-1] - day_ms[test_mask][0]) / ck.MS_D / ck.DAYS_YR
            pf["turnover_yr"] = float(turn[test_mask].sum() / yrs)
            pf["pct_invested_test"] = float(np.mean(np.abs(HOLD[test_mask]).sum(axis=1) > 0))
            pf["pct_invested_train"] = float(np.mean(np.abs(HOLD[train_mask]).sum(axis=1) > 0))
            pf["survives"] = ck.survives(pf)
        out["fee_%.1fbp" % fee_bp] = pf
    return out


def train_score(HOLD, R, Fd, day_ms, train_mask):
    net, turn = ck.eval_book(HOLD, R, Fd, ck.TUNE_FEE_BP * ck.BP)
    pf = ck.perf(day_ms[train_mask], net[train_mask])
    pf["pct_invested_train"] = float(np.mean(np.abs(HOLD[train_mask]).sum(axis=1) > 0))
    return pf


def main():
    t_start = time.time()
    os.makedirs(ck.CACHE_DIR, exist_ok=True)
    stats = {"smoke": "xsmom_regime_filter", "date": pd.Timestamp.now("UTC").isoformat(), "errors": []}
    try:
        pairs = ck.list_pairs()
        for p in pairs:
            ck.resample_pair(p)
        m = ck.build_matrices(pairs)
        Th = m["hc"].shape[0]
        Td = m["dc"].shape[0]
        _, Fd = ck.load_funding(pairs, m["h0"], Th, m["d0"], Td)
        day_ms = (m["d0"] + np.arange(Td)) * ck.MS_D
        train_mask = day_ms < ck.TRAIN_END_MS
        test_mask = day_ms >= ck.TRAIN_END_MS
        C = m["dc"]
        R = ck.rets(C)
        reb = weekly_reb(m, Td)
        HOLD = ck.mom_weights(C, reb, LOOKBACK_D)
        base_gross = np.nansum(HOLD * np.nan_to_num(R), axis=1)
        base_net, _ = ck.eval_book(HOLD, R, Fd, ck.TUNE_FEE_BP * ck.BP)

        stats["data"] = {"pairs": len(pairs), "days": int(Td),
                         "start": str(pd.Timestamp(day_ms[0], unit="ms").date()),
                         "end": str(pd.Timestamp(day_ms[-1], unit="ms").date()),
                         "train_end": "2024-01-01"}
        stats["protocol"] = {
            "base_rule": "1w rebalance (Monday close), quintile L/S equal-weight gross 1.0, lookback 7d (train-fixed in costkilled_retest), realized 8h funding",
            "candidates_examined_on_train": 1 + 8,
            "candidate_list": ["dispersion_high", "btc_vol_high", "btc_vol_low", "btc_above_100dma",
                               "btc_below_100dma", "factor_mom_90d_pos", "funding_share_high",
                               "funding_share_low", "vol_scaled"],
            "selection_rule": "pre-registered before any test evaluation: vol_scaled always + top-%d of the 8 filter variants by train Sharpe at 3.5bp; only these + baseline scored on test" % N_TOP_FILTERS,
            "thresholds": "all trailing medians %dd (min %dd), smoothing %dd, BTC MA %dd, factor-mom window %dd, vol-scale window %dd cap %.1fx target=train-mean factor vol; fixed a priori, none swept",
            "filters_gate_at_rebalance_only": True}
        stats["protocol"]["thresholds"] = stats["protocol"]["thresholds"] % (
            MED_WINDOW_D, MED_MIN_D, DISP_SMOOTH_D, TREND_MA_D, MOMOM_WINDOW_D, VS_VOL_WINDOW_D, VS_MAX_LEV)

        cands, sval, vs_target = build_candidates(m, C, R, Fd, reb, train_mask, base_gross, base_net)
        books = {}
        for name, on in cands.items():
            G = gate_series(reb, on.astype(float), 0.0)
            books[name] = HOLD * G[:, None]
        S = gate_series(reb, sval, 1.0)
        books["vol_scaled"] = HOLD * S[:, None]

        train_table = {}
        for name, Hv in books.items():
            pf = train_score(Hv, R, Fd, day_ms, train_mask)
            train_table[name] = {"train_sharpe": round(pf["sharpe"], 3), "train_cagr": round(pf["cagr"], 4),
                                 "train_maxdd": round(pf["maxdd"], 3),
                                 "pct_invested_train": round(pf["pct_invested_train"], 3)}
        base_train = train_score(HOLD, R, Fd, day_ms, train_mask)
        train_table["baseline"] = {"train_sharpe": round(base_train["sharpe"], 3),
                                   "train_cagr": round(base_train["cagr"], 4),
                                   "train_maxdd": round(base_train["maxdd"], 3),
                                   "pct_invested_train": round(base_train["pct_invested_train"], 3)}
        stats["train_table"] = train_table
        stats["vol_scale_target_daily_sd"] = round(vs_target, 6)

        filter_rank = sorted(cands, key=lambda n: -train_table[n]["train_sharpe"])
        selected = ["vol_scaled"] + filter_rank[:N_TOP_FILTERS]
        stats["selected"] = selected

        stats["test_results"] = {"baseline": eval_variant(HOLD, R, Fd, day_ms, train_mask, test_mask)}
        for name in selected:
            stats["test_results"][name] = eval_variant(books[name], R, Fd, day_ms, train_mask, test_mask)

        survivors = [n for n in selected
                     if stats["test_results"][n]["fee_%.1fbp" % ck.TUNE_FEE_BP]["survives"]]
        stats["survivors"] = survivors
        top_filter = filter_rank[0]
        stats["null_shift_gate"] = shift_gate_null(HOLD, R, Fd, day_ms, test_mask, reb, cands[top_filter])
        stats["null_shift_gate"]["gate"] = top_filter
        caveats = []
        for n in survivors:
            if n in train_table and train_table[n]["train_sharpe"] < train_table["baseline"]["train_sharpe"]:
                caveats.append("%s does NOT lift train Sharpe (%.3f vs baseline %.3f); its selection was rank-among-filters, not in-sample improvement" % (
                    n, train_table[n]["train_sharpe"], train_table["baseline"]["train_sharpe"]))
        if survivors and stats["null_shift_gate"]["frac_bar_clear_either_complement"] > 0:
            caveats.append("placebo shifted gates clear the full bar %.1f%% of the time alone, %.1f%% under the actual either-complement exposure" % (
                100 * stats["null_shift_gate"]["frac_bar_clear"],
                100 * stats["null_shift_gate"]["frac_bar_clear_either_complement"]))
        both_signs = [f for f in filter_rank[:N_TOP_FILTERS] if f.rsplit("_", 1)[0] == filter_rank[0].rsplit("_", 1)[0]]
        if len(both_signs) == N_TOP_FILTERS:
            caveats.append("selection rule let both signs of one filter fill both slots; test set adjudicated a train coin-flip")
        stats["caveats"] = caveats
        stats["verdict"] = ("SUCCESS-WITH-CAVEATS: pre-registered bar cleared but evidence weak (see caveats)" if survivors and caveats else
                            "SUCCESS: pre-registered regime filter clears the bar" if survivors else
                            "FAILURE: no pre-registered filter clears Sharpe>=0.8 + all-years-positive + maxDD<40% at 3.5bp; family stays real-but-thin")
    except Exception:
        stats["errors"].append(traceback.format_exc(limit=5))
        stats["verdict"] = "INCONCLUSIVE: errors"
    stats["runtime_s"] = round(time.time() - t_start, 1)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=1, default=float)
    print(json.dumps({"verdict": stats["verdict"], "selected": stats.get("selected"),
                      "survivors": stats.get("survivors"), "errors": len(stats["errors"]),
                      "runtime_s": stats["runtime_s"]}, indent=1))
    return 0 if not stats["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
