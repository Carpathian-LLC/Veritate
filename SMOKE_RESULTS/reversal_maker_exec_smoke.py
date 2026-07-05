# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests maker-execution rescue of the 1h XS reversal (fade top-decile 3-bar movers,
#   rules train-fixed in costkilled_retest_smoke.py, test 2024-01..2026-05): leg split,
#   passive limit-fill sim with fill-selection measurement, concentration cap, long-only
#   spot arm, capacity, exploratory 1w-momentum combo. Falsifier: fill selection eats
#   the edge (filled-only gross materially below all-trades gross, net <= 0 at 0-1bp
#   maker) -> passive execution does not rescue reversal; the 1bp wall stands.
# - fill rule holds one limit level over a full-hour window and taker fallback lands on
#   the hour boundary, so any(1m low <= level) == (hourly min low <= level) and "next
#   1m open" == next hourly open: the 1m sim reduces exactly to cached hourly extremes
#   (equivalence re-verified against raw 1m CSVs for sample pairs, see stats).
# - fills book at the limit level always (no price improvement); x>0 variants keep the
#   level but require x bp penetration (queue conservatism) -> x=0 optimistic touch
#   bound, x>0 strictly conservative.
# - wall-clock estimate ~3 min with warm costkilled cache.
# SMOKE_RESULTS/reversal_maker_exec_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib.util
import json
import os
import time
import traceback

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Constants

REPO = "/Users/mirach-00-usc1/Development/Veritate"
CK_PATH = os.path.join(REPO, "SMOKE_RESULTS/costkilled_retest_smoke.py")
STATS_PATH = os.path.join(REPO, "SMOKE_RESULTS/reversal_maker_exec_stats.json")

_spec = importlib.util.spec_from_file_location("ck", CK_PATH)
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

REV_K = 3
X_SWEEP_BP = [0.0, 2.0, 5.0]
MAKER_BP = [0.0, 0.5, 1.0, 2.0]
TAKER_BP = 2.6
SPOT_TAKER_ALT_BP = 10.0
ARMS = ["A", "B"]
CAP_W = 0.10
CAP_X_BP = 2.0
PARTICIPATION = 0.01
CAPACITY_YEAR = 2026
MOM_L = 7
MOM_FEE_BP = 3.5
COMBO_ARM = "A"
COMBO_X_BP = 2.0
COMBO_MAKER_BP = 1.0
COMBO_BAR = 0.8
SURV_SHARPE = 1.0
SURV_MAXDD = 0.40
SURV_YEARS = ["2024", "2025", "2026"]
SURV_MAKER_MAX = 1.0
VERIFY_PAIRS = ["BTCUSDT", "SOLUSDT"]
VERIFY_START = "2026-01-01"
VERIFY_DAYS = 28

# ------------------------------------------------------------------------------------
# Functions


def reversal_book(m):
    C = m["hc"]
    T, N = C.shape
    sig = np.full_like(C, np.nan)
    sig[REV_K:] = C[REV_K:] / C[:-REV_K] - 1.0
    a = np.abs(sig)
    n_avail = np.isfinite(a).sum(axis=1)
    row_ok = n_avail >= ck.MIN_UNIVERSE_H
    q = np.full(T, np.nan)
    q[row_ok] = np.nanquantile(a[row_ok], ck.DECILE_Q, axis=1)
    sel = row_ok[:, None] & np.isfinite(a) & (a >= q[:, None])
    n_sel = sel.sum(axis=1)
    W = np.zeros((T, N))
    src = np.where(sel, -np.sign(sig), 0.0) / np.maximum(n_sel, 1)[:, None]
    W[1:] = np.nan_to_num(src[:-1])
    return sig, sel, n_sel, W


def year_slice(dms, dnet, y):
    msk = pd.to_datetime(np.asarray(dms), unit="ms").year == y
    seg = dnet[msk]
    sd = seg.std()
    return seg, (float(seg.mean() / sd * np.sqrt(ck.DAYS_YR)) if sd > 0 else 0.0)


def leg_stats(m, W, Fh):
    T = W.shape[0]
    R = ck.rets(m["hc"])
    h_ms = (m["h0"] + np.arange(T)) * ck.MS_H
    te_h = h_ms >= ck.TRAIN_END_MS
    yr_h = pd.to_datetime(h_ms, unit="ms").year.to_numpy()
    span_yr = te_h.sum() / 24.0 / ck.DAYS_YR
    out = {}
    for leg, WL in (("full_ls", W), ("long", np.clip(W, 0.0, None)), ("short", np.clip(W, None, 0.0))):
        gross = np.nansum(WL * np.nan_to_num(R), axis=1)
        fund = -np.nansum(WL * Fh, axis=1)
        turn = np.abs(np.diff(WL, axis=0, prepend=np.zeros((1, WL.shape[1])))).sum(axis=1)
        dms, dnet = ck.hourly_to_daily(m["h0"], np.where(te_h, gross + fund, 0.0))
        te_d = dms >= ck.TRAIN_END_MS
        pf = ck.perf(dms[te_d], dnet[te_d])
        yrs = {}
        for y in (2024, 2025, 2026):
            hm = te_h & (yr_h == y)
            seg, sh = year_slice(dms[te_d], dnet[te_d], y)
            yrs[str(y)] = {"ret": round(float(np.prod(1.0 + seg) - 1.0), 4), "sharpe": round(sh, 2),
                           "gross_bp_side": round(float(gross[hm].sum() / max(turn[hm].sum(), 1e-12) / ck.BP), 3)}
        out[leg] = {"sharpe": round(pf["sharpe"], 2), "maxdd": round(pf["maxdd"], 3),
                    "ret_yr_sum": round(float(dnet[te_d].sum() / span_yr), 4), "per_year": yrs,
                    "turnover_yr": round(float(turn[te_h].sum() / span_yr)),
                    "funding_yr": round(float(fund[te_h].sum() / span_yr), 4)}
    return out


def build_trades(m, sig, sel, n_sel, Fh):
    C, O, Hh, Ll = m["hc"], m["ho"], m["hh"], m["hl"]
    T = C.shape[0]
    h_ms = (m["h0"] + np.arange(T)) * ck.MS_H
    tt, jj = np.where(sel)
    keep = (tt + 3 < T)
    tt, jj = tt[keep], jj[keep]
    keep = h_ms[tt + 1] >= ck.TRAIN_END_MS
    tt, jj = tt[keep], jj[keep]
    d = -np.sign(sig[tt, jj])
    ok = (np.isfinite(C[tt, jj]) & np.isfinite(C[tt + 1, jj]) &
          np.isfinite(O[tt + 2, jj]) & np.isfinite(O[tt + 3, jj]) & (d != 0))
    dropped = int((~ok).sum())
    tt, jj, d = tt[ok], jj[ok], d[ok]
    tr = {"t": tt, "j": jj, "d": d, "w": 1.0 / n_sel[tt],
          "c0": C[tt, jj], "c1": C[tt + 1, jj],
          "lo1": Ll[tt + 1, jj], "hi1": Hh[tt + 1, jj],
          "lo2": Ll[tt + 2, jj], "hi2": Hh[tt + 2, jj],
          "o2": O[tt + 2, jj], "o3": O[tt + 3, jj],
          "f1": Fh[tt + 1, jj], "f2": Fh[tt + 2, jj], "f3": Fh[tt + 3, jj]}
    return tr, dropped


def sub(tr, mask):
    return {k: v[mask] for k, v in tr.items()}


def fills(tr, xbp):
    x = xbp * ck.BP
    long = tr["d"] > 0
    with np.errstate(invalid="ignore"):
        fe = np.where(long, tr["lo1"] <= tr["c0"] * (1 - x), tr["hi1"] >= tr["c0"] * (1 + x))
        fx = np.where(long, tr["hi2"] >= tr["c1"] * (1 + x), tr["lo2"] <= tr["c1"] * (1 - x))
    return fe, fx


def variant_arrays(tr, fe, fx, arm, spot):
    n = len(fe)
    entry_taker = np.zeros(n, bool) if arm == "A" else ~fe
    act = fe if arm == "A" else np.ones(n, bool)
    exit_taker = ~fx
    entry_px = np.where(entry_taker, tr["o2"], tr["c0"])
    exit_px = np.where(exit_taker, tr["o3"], tr["c1"])
    gross = tr["d"] * (exit_px / entry_px - 1.0)
    if spot:
        fund = np.zeros(n)
    else:
        fund = -tr["d"] * (np.where(entry_taker, 0.0, tr["f1"]) + tr["f2"] + np.where(exit_taker, tr["f3"], 0.0))
    n_taker = entry_taker.astype(float) + exit_taker.astype(float)
    exit_h = np.where(exit_taker, tr["t"] + 3, tr["t"] + 2)
    start_h = np.where(entry_taker, tr["t"] + 2, tr["t"] + 1)
    return act, gross, fund, n_taker, exit_h, start_h


def daily_series(m, day_grid, act, pnl_w, exit_h):
    d_lo, d_hi = day_grid
    day = (m["h0"] + exit_h) // 24
    dret = np.bincount(day[act] - d_lo, weights=pnl_w[act], minlength=d_hi - d_lo + 1)
    dms = np.arange(d_lo, d_hi + 1) * ck.MS_D
    return dms, dret


def slim(pf, mk=None):
    out = {"cagr": round(pf["cagr"], 4), "sharpe": round(pf["sharpe"], 2), "maxdd": round(pf["maxdd"], 3),
           "per_year": {k: round(v, 4) for k, v in pf["per_year"].items()}}
    if mk is not None and mk <= SURV_MAKER_MAX:
        out["survives"] = bool(all(pf["per_year"].get(y, -1.0) > 0.0 for y in SURV_YEARS)
                               and pf["sharpe"] >= SURV_SHARPE and pf["maxdd"] < SURV_MAXDD)
    return out


def selection_stats(tr, fe, fx, cc, span_yr):
    out = {"gross_yr_all_trades": round(float((cc * tr["w"]).sum() / span_yr), 4),
           "gross_yr_entry_filled": round(float((cc * tr["w"])[fe].sum() / span_yr), 4)}
    for leg, mask in (("long", tr["d"] > 0), ("short", tr["d"] < 0)):
        f = fe & mask
        out[leg] = {"n": int(mask.sum()),
                    "entry_fill_rate": round(float(fe[mask].mean()), 4) if mask.any() else None,
                    "exit_fill_rate_given_entry": round(float(fx[f].mean()), 4) if f.any() else None,
                    "all_cc_gross_bp": round(float(cc[mask].mean() / ck.BP), 2) if mask.any() else None,
                    "filled_cc_gross_bp": round(float(cc[f].mean() / ck.BP), 2) if f.any() else None,
                    "missed_cc_gross_bp": round(float(cc[mask & ~fe].mean() / ck.BP), 2) if (mask & ~fe).any() else None}
    return out


def run_grid(m, day_grid, te_h, tr, arms, makers, spot, taker_bps):
    Th = m["hc"].shape[0]
    span_yr = (day_grid[1] - day_grid[0] + 1) / ck.DAYS_YR
    cc = tr["d"] * (tr["c1"] / tr["c0"] - 1.0)
    out = {}
    for xbp in X_SWEEP_BP:
        fe, fx = fills(tr, xbp)
        xent = {"selection": selection_stats(tr, fe, fx, cc, span_yr)}
        for arm in arms:
            act, gross, fund, n_taker, exit_h, start_h = variant_arrays(tr, fe, fx, arm, spot)
            delta = np.zeros(Th + 1)
            np.add.at(delta, start_h[act], tr["w"][act])
            np.add.at(delta, exit_h[act] + 1, -tr["w"][act])
            occ = np.cumsum(delta)[:Th]
            aent = {"trades_yr": round(float(act.sum() / span_yr)),
                    "turnover_yr": round(float(2.0 * tr["w"][act].sum() / span_yr)),
                    "avg_gross_exposure": round(float(occ[te_h].mean()), 3),
                    "p99_gross_exposure": round(float(np.percentile(occ[te_h], 99)), 3),
                    "exit_drag_bp": round(float((gross - cc)[fe].mean() / ck.BP), 2)}
            if arm == "B":
                aent["fallback_entry_gross_bp"] = round(float(gross[~fe].mean() / ck.BP), 2) if (~fe).any() else None
            for tk in taker_bps:
                for mk in makers:
                    fees = (mk * (2.0 - n_taker) + tk * n_taker) * ck.BP
                    net = gross + fund - fees
                    dms, dret = daily_series(m, day_grid, act, net * tr["w"], exit_h)
                    key = "maker_%gbp" % mk + ("_taker%g" % tk if len(taker_bps) > 1 else "")
                    aent[key] = slim(ck.perf(dms, dret), mk)
            xent["arm_" + arm] = aent
        out["x_%gbp" % xbp] = xent
    return out


def capacity_stats(m, tr):
    t, j, w = tr["t"], tr["j"], tr["w"]
    h_ms = (m["h0"] + t) * ck.MS_H
    m26 = pd.to_datetime(h_ms, unit="ms").year == CAPACITY_YEAR
    dollar_h = m["hv"][t + 1, j] * m["hc"][t + 1, j]
    drow = (m["h0"] + t) // 24 - m["d0"]
    dollar_d = m["dv"][drow, j] * m["dc"][drow, j]
    okd = m26 & np.isfinite(dollar_d)
    okh = m26 & np.isfinite(dollar_h)
    out = {"median_daily_dollar_vol_2026": round(float(np.median(dollar_d[okd]))),
           "participation_cap": PARTICIPATION}
    for tag, ww in (("uncapped", w), ("capped_10pct", np.minimum(w, CAP_W))):
        cap_acct = PARTICIPATION * dollar_h[okh] / ww[okh]
        out["account_cap_usd_" + tag] = {p: round(float(np.percentile(cap_acct, q)))
                                         for p, q in (("p5", 5), ("p25", 25), ("p50", 50))}
    return out


def combo_stats(m, Fd, tr, day_grid):
    fe, fx = fills(tr, COMBO_X_BP)
    act, gross, fund, n_taker, exit_h, _ = variant_arrays(tr, fe, fx, COMBO_ARM, False)
    net = gross + fund - (COMBO_MAKER_BP * (2.0 - n_taker) + TAKER_BP * n_taker) * ck.BP
    rev_dms, rev_dret = daily_series(m, day_grid, act, net * tr["w"], exit_h)
    C = m["dc"]
    Td = C.shape[0]
    day_ms = (m["d0"] + np.arange(Td)) * ck.MS_D
    weekday = (np.arange(m["d0"], m["d0"] + Td) + ck.EPOCH_THU) % 7
    HOLD = ck.mom_weights(C, weekday == 0, MOM_L)
    mom_net, _ = ck.eval_book(HOLD, ck.rets(C), Fd, MOM_FEE_BP * ck.BP)
    common, ia, ib = np.intersect1d(day_ms, rev_dms, return_indices=True)
    te = common >= ck.TRAIN_END_MS
    a = np.nan_to_num(mom_net[ia][te])
    b = rev_dret[ib][te]
    dms = common[te]
    corr = float(np.corrcoef(a, b)[0, 1])
    comb = 0.5 * (a / a.std() + b / b.std())
    pf = ck.perf(dms, comb * 0.01)
    return {"note": "EXPLORATORY: pre-registered 50/50 equal-vol; rev arm %s x=%gbp maker %gbp, mom 1w L=%dd at %gbp taker"
                    % (COMBO_ARM, COMBO_X_BP, COMBO_MAKER_BP, MOM_L, MOM_FEE_BP),
            "corr_daily": round(corr, 3),
            "mom_sharpe": round(ck.perf(dms, a)["sharpe"], 2),
            "rev_sharpe": round(ck.perf(dms, b)["sharpe"], 2),
            "combo_sharpe": round(pf["sharpe"], 2),
            "combo_per_year_sharpe": {y: round(year_slice(dms, comb, y)[1], 2) for y in (2024, 2025, 2026)},
            "clears_0.8": bool(pf["sharpe"] >= COMBO_BAR)}


def verify_1m_equivalence():
    w0 = int(pd.Timestamp(VERIFY_START).value // 10 ** 6)
    w1 = w0 + VERIFY_DAYS * ck.MS_D
    out = {"pairs": VERIFY_PAIRS, "window": VERIFY_START, "hours": 0, "mismatch": 0}
    for p in VERIFY_PAIRS:
        df = pd.read_csv(os.path.join(ck.DATA_DIR, p + ".csv"), usecols=["time", "low", "high"])
        t = df["time"].to_numpy(np.int64)
        t = np.where(t > ck.US_SWITCH, t // 1000, t)
        msk = (t >= w0) & (t < w1)
        hidx = t[msk] // ck.MS_H
        lo = pd.Series(df["low"].to_numpy(float)[msk]).groupby(hidx).min()
        hi = pd.Series(df["high"].to_numpy(float)[msk]).groupby(hidx).max()
        d = np.load(os.path.join(ck.CACHE_DIR, p + ".npz"))
        pos = np.searchsorted(d["hidx"], lo.index.to_numpy())
        out["hours"] += int(len(lo))
        out["mismatch"] += int((d["hl"][pos] != lo.to_numpy()).sum() + (d["hh"][pos] != hi.to_numpy()).sum())
    return out


def collect_survivors(node, path, out):
    if isinstance(node, dict):
        if node.get("survives"):
            out.append(path)
        for k, v in node.items():
            if isinstance(v, dict):
                collect_survivors(v, path + "/" + k, out)


def main():
    t_start = time.time()
    os.makedirs(ck.CACHE_DIR, exist_ok=True)
    stats = {"smoke": "reversal_maker_exec", "date": pd.Timestamp.now("UTC").isoformat(), "errors": []}
    pairs = ck.list_pairs()
    for p in pairs:
        ck.resample_pair(p)
    m = ck.build_matrices(pairs)
    Th = m["hc"].shape[0]
    Td = m["dc"].shape[0]
    Fh, Fd = ck.load_funding(pairs, m["h0"], Th, m["d0"], Td)
    h_ms = (m["h0"] + np.arange(Th)) * ck.MS_H
    te_h = h_ms >= ck.TRAIN_END_MS
    day_grid = (ck.TRAIN_END_MS // ck.MS_D, (m["h0"] + Th - 1) // 24)
    sig, sel, n_sel, W = reversal_book(m)
    tr, dropped = build_trades(m, sig, sel, n_sel, Fh)
    S = np.where(sel, -np.sign(np.nan_to_num(sig)), 0.0)
    ov_num = int(((S[1:] * S[:-1]) > 0)[te_h[1:]].sum())
    ov_den = int(sel[1:][te_h[1:]].sum())
    stats["data"] = {"pairs": len(pairs), "rules": "K=%d-bar top-decile |ret| fade, hold 1 bar, w=1/n_sel (train-fixed)" % REV_K,
                     "test_span": "2024-01-01..2026-05-31", "trades": int(len(tr["t"])),
                     "trades_dropped_gap": dropped,
                     "netting_overlap_frac": round(ov_num / max(ov_den, 1), 4),
                     "fill_model": "limit at signal-bar close, full-hour window, fill at level iff extreme penetrates x bp; taker fallback 2.6bp at next hourly open; funding charged per held hour"}
    sections = [
        ("equivalence_check", verify_1m_equivalence),
        ("leg_decomposition", lambda: leg_stats(m, W, Fh)),
        ("fill_sim", lambda: run_grid(m, day_grid, te_h, tr, ARMS, MAKER_BP, False, [TAKER_BP])),
        ("concentration_cap_10pct", lambda: run_grid(
            m, day_grid, te_h, dict(tr, w=np.minimum(tr["w"], CAP_W)), ARMS, MAKER_BP, False, [TAKER_BP])),
        ("long_only_spot", lambda: run_grid(
            m, day_grid, te_h, sub(tr, tr["d"] > 0), ARMS, [0.0], True, [TAKER_BP, SPOT_TAKER_ALT_BP])),
        ("capacity", lambda: capacity_stats(m, tr)),
        ("combo_exploratory", lambda: combo_stats(m, Fd, tr, day_grid))]
    for name, fn in sections:
        t0 = time.time()
        try:
            stats[name] = fn()
            if isinstance(stats[name], dict):
                stats[name]["runtime_s"] = round(time.time() - t0, 1)
        except Exception:
            stats["errors"].append("%s: %s" % (name, traceback.format_exc(limit=3)))
    if stats.get("concentration_cap_10pct"):
        stats["concentration_cap_10pct"] = {"note": "same grid, w capped at %g" % CAP_W,
                                            "x_%gbp" % CAP_X_BP: stats["concentration_cap_10pct"]["x_%gbp" % CAP_X_BP]}
    survivors = []
    for sec in ("fill_sim", "concentration_cap_10pct", "long_only_spot"):
        if sec in stats:
            collect_survivors(stats[sec], sec, survivors)
    stats["survivors"] = survivors
    stats["verdict"] = ("SUCCESS: passive execution rescues reversal" if survivors else
                        "FAILURE: no passive-executed variant clears the bar") if not stats["errors"] else "INCONCLUSIVE: errors"
    stats["runtime_s"] = round(time.time() - t_start, 1)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=1, default=float)
    print(json.dumps({"verdict": stats["verdict"], "survivors": survivors, "errors": len(stats["errors"]),
                      "runtime_s": stats["runtime_s"]}, indent=1))
    return 0 if not stats["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
