# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - retests four cost-killed families as perps long/short at 2.6/3.5/5bp per side with
#   realized 8h funding: cross-sectional momentum L/S, 1h reversal + daily RSI mirror,
#   time-of-day/DOW seasonality, vol-expansion breakout. All rules fixed on pre-2024
#   train, scored 2024-2026 OOS. Falsifier: no family clears Sharpe>=0.8, maxDD<40%,
#   positive net 2024 AND 2025 AND 2026YTD at 3.5bp/side -> families stay dead.
# - wall-clock estimate ~25 min first run (1m resample of 40 pairs), ~5 min cached.
# SMOKE_RESULTS/costkilled_retest_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import time
import traceback

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------
# Constants

REPO = "/Users/mirach-00-usc1/Development/Veritate"
DATA_DIR = os.path.join(REPO, "extensions/installed/market/data/crypto_of")
FUNDING_DIR = os.path.join(REPO, "extensions/installed/market/data/funding")
CACHE_DIR = ("/private/tmp/claude-501/-Users-mirach-00-usc1-Development-Veritate/"
             "10655374-b645-4bca-a6e1-0208baad89ab/scratchpad/costkilled_cache")
STATS_PATH = os.path.join(REPO, "SMOKE_RESULTS/costkilled_retest_stats.json")

MS_H = 3_600_000
MS_D = 86_400_000
US_SWITCH = 10 ** 14
LISTING_TRIM_MS = 3 * MS_D
MIN_MINUTES_H = 55
MIN_MINUTES_D = 1320
EPOCH_THU = 3

TRAIN_END_MS = int(pd.Timestamp("2024-01-01").value // 10 ** 6)
FEES_BP = [2.6, 3.5, 5.0]
TUNE_FEE_BP = 3.5
BP = 1e-4
DAYS_YR = 365.0

MIN_UNIVERSE_D = 15
MIN_UNIVERSE_H = 20
QUINTILE = 0.2
DECILE_Q = 0.9
MIN_SIDE = 2
MOM_LOOKBACKS = [7, 14, 30]
REV_LOOKBACKS = [1, 3]
RSI_PERIOD = 14
RSI_LO = 25.0
RSI_HI = 75.0
RSI_HOLDS = [1, 3, 5]
SEAS_MAX_RULES = 3
SEAS_MIN_T = 3.0
SEAS_MIN_BASKET = 10
BRK_ATR_N = 24
BRK_KR = [2.0, 3.0]
BRK_KV = [1.5, 2.0]
BRK_HOLDS = [1, 2, 4]
BRK_MIN_TRADES = 300
BRK_SLOT_PCT = 95
N_NULL = 50
NULL_SEED = 7
SURV_SHARPE = 0.8
SURV_MAXDD = 0.40
SURV_YEARS = ["2024", "2025", "2026"]

# ------------------------------------------------------------------------------------
# Functions


def list_pairs():
    return sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))


def resample_pair(pair):
    cache = os.path.join(CACHE_DIR, pair + ".npz")
    if os.path.exists(cache):
        return
    df = pd.read_csv(os.path.join(DATA_DIR, pair + ".csv"),
                     usecols=["time", "open", "high", "low", "close", "volume"])
    t = df["time"].to_numpy(np.int64)
    t = np.where(t > US_SWITCH, t // 1000, t)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    keep = t >= t.min() + LISTING_TRIM_MS
    t, o, h, l, c, v = t[keep], o[keep], h[keep], l[keep], c[keep], v[keep]
    out = {}
    for tag, ms, in (("h", MS_H), ("d", MS_D)):
        idx = t // ms
        uniq, first = np.unique(idx, return_index=True)
        bounds = first
        out[tag + "idx"] = uniq
        out[tag + "o"] = o[first]
        out[tag + "h"] = np.maximum.reduceat(h, bounds)
        out[tag + "l"] = np.minimum.reduceat(l, bounds)
        out[tag + "c"] = c[np.append(first[1:] - 1, len(t) - 1)]
        out[tag + "v"] = np.add.reduceat(v, bounds)
        out[tag + "n"] = np.diff(np.append(first, len(t)))
    np.savez_compressed(cache, **out)


def build_matrices(pairs):
    data = {p: np.load(os.path.join(CACHE_DIR, p + ".npz")) for p in pairs}
    mats = {}
    for tag, min_n in (("h", MIN_MINUTES_H), ("d", MIN_MINUTES_D)):
        i0 = min(int(d[tag + "idx"].min()) for d in data.values())
        i1 = max(int(d[tag + "idx"].max()) for d in data.values())
        T = i1 - i0 + 1
        for field in ("o", "h", "l", "c", "v"):
            M = np.full((T, len(pairs)), np.nan)
            for j, p in enumerate(pairs):
                d = data[p]
                valid = d[tag + "n"] >= min_n
                M[d[tag + "idx"][valid] - i0, j] = d[tag + field][valid]
            mats[tag + field] = M
        mats[tag + "0"] = i0
    mats["pairs"] = pairs
    return mats


def load_funding(pairs, h0, Th, d0, Td):
    Fh = np.zeros((Th, len(pairs)))
    Fd = np.zeros((Td, len(pairs)))
    for j, p in enumerate(pairs):
        path = os.path.join(FUNDING_DIR, p + ".csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        t = df["time"].to_numpy(np.int64)
        t = np.where(t > US_SWITCH, t // 1000, t)
        f = df["funding"].to_numpy(float)
        hi = t // MS_H - h0
        ok = (hi >= 0) & (hi < Th)
        np.add.at(Fh[:, j], hi[ok], f[ok])
        di = t // MS_D - d0
        ok = (di >= 0) & (di < Td)
        np.add.at(Fd[:, j], di[ok], f[ok])
    return Fh, Fd


def rets(C):
    R = np.full_like(C, np.nan)
    R[1:] = C[1:] / C[:-1] - 1.0
    return R


def perf(day_ms, ret):
    ret = np.nan_to_num(np.asarray(ret, float))
    eq = np.cumprod(1.0 + ret)
    yrs = max((day_ms[-1] - day_ms[0]) / MS_D / DAYS_YR, 1e-9)
    cagr = float(eq[-1] ** (1.0 / yrs) - 1.0) if eq[-1] > 0 else -1.0
    sd = ret.std()
    sharpe = float(ret.mean() / sd * np.sqrt(DAYS_YR)) if sd > 0 else 0.0
    maxdd = float(np.max(1.0 - eq / np.maximum.accumulate(eq)))
    years = pd.to_datetime(np.asarray(day_ms), unit="ms").year
    per_year = {str(y): float(np.prod(1.0 + ret[years == y]) - 1.0) for y in np.unique(years)}
    half = len(ret) // 2
    halves = []
    for seg in (ret[:half], ret[half:]):
        s = seg.std()
        halves.append({"ret": float(np.prod(1.0 + seg) - 1.0),
                       "sharpe": float(seg.mean() / s * np.sqrt(DAYS_YR)) if s > 0 else 0.0})
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd, "per_year": per_year,
            "halves": halves, "total_ret": float(eq[-1] - 1.0)}


def survives(pf):
    yrs_ok = all(pf["per_year"].get(y, -1.0) > 0.0 for y in SURV_YEARS)
    return bool(yrs_ok and pf["sharpe"] >= SURV_SHARPE and pf["maxdd"] < SURV_MAXDD)


def hourly_to_daily(h0, net):
    day = (h0 + np.arange(len(net))) // 24
    d0, d1 = day[0], day[-1]
    dret = np.bincount(day - d0, weights=np.nan_to_num(net), minlength=d1 - d0 + 1)
    day_ms = (np.arange(d0, d1 + 1)) * MS_D
    return day_ms, dret


def eval_book(HOLD, R, F, fee):
    gross = np.nansum(HOLD * np.nan_to_num(R), axis=1)
    fund = -np.nansum(HOLD * F, axis=1)
    turn = np.abs(np.diff(HOLD, axis=0, prepend=np.zeros((1, HOLD.shape[1])))).sum(axis=1)
    net = gross + fund - fee * turn
    return net, turn


def mom_weights(C, reb_mask, lookback):
    T, N = C.shape
    mom = np.full_like(C, np.nan)
    mom[lookback:] = C[lookback:] / C[:-lookback] - 1.0
    HOLD = np.zeros((T, N))
    w = np.zeros(N)
    for t in range(T):
        HOLD[t] = w
        if not reb_mask[t]:
            continue
        avail = np.isfinite(mom[t]) & np.isfinite(C[t])
        n_avail = int(avail.sum())
        if n_avail < MIN_UNIVERSE_D:
            w = np.zeros(N)
            continue
        n_side = max(MIN_SIDE, int(n_avail * QUINTILE))
        order = np.argsort(mom[t][avail])
        ids = np.where(avail)[0][order]
        w = np.zeros(N)
        w[ids[-n_side:]] = 0.5 / n_side
        w[ids[:n_side]] = -0.5 / n_side
    return HOLD


def family_momentum(m, Fd, day_ms, train_mask, test_mask, rng):
    C = m["dc"]
    R = rets(C)
    T = C.shape[0]
    weekday = (np.arange(m["d0"], m["d0"] + T) + EPOCH_THU) % 7
    freqs = {"1d": np.ones(T, bool), "1w": weekday == 0}
    out = {"rules_fixed": "quintile L/S equal-weight gross 1.0; lookback per freq picked on train Sharpe at 3.5bp"}
    for fname, reb in freqs.items():
        train_scores = {}
        holds = {}
        for L in MOM_LOOKBACKS:
            HOLD = mom_weights(C, reb, L)
            holds[L] = HOLD
            net, _ = eval_book(HOLD, R, Fd, TUNE_FEE_BP * BP)
            train_scores[L] = perf(day_ms[train_mask], net[train_mask])["sharpe"]
        best_L = max(train_scores, key=train_scores.get)
        HOLD = holds[best_L]
        entry = {"lookback_d": best_L, "train_sharpe_by_lookback": {str(k): round(v, 3) for k, v in train_scores.items()}}
        for fee_bp in FEES_BP:
            net, turn = eval_book(HOLD, R, Fd, fee_bp * BP)
            pf = perf(day_ms[test_mask], net[test_mask])
            if fee_bp == TUNE_FEE_BP:
                pf["train"] = perf(day_ms[train_mask], net[train_mask])
                yrs = (day_ms[test_mask][-1] - day_ms[test_mask][0]) / MS_D / DAYS_YR
                pf["turnover_yr"] = float(turn[test_mask].sum() / yrs)
                pf["survives"] = survives(pf)
                fund_pnl = -np.nansum(HOLD * Fd, axis=1)
                pf["funding_pnl_yr"] = float(fund_pnl[test_mask].sum() / yrs)
            entry["fee_%.1fbp" % fee_bp] = pf
        null_sh = []
        for _ in range(N_NULL):
            HN = np.zeros_like(HOLD)
            w = np.zeros(C.shape[1])
            for t in range(T):
                HN[t] = w
                if reb[t]:
                    avail = np.isfinite(C[t])
                    n_avail = int(avail.sum())
                    if n_avail < MIN_UNIVERSE_D:
                        w = np.zeros(C.shape[1])
                        continue
                    n_side = max(MIN_SIDE, int(n_avail * QUINTILE))
                    ids = rng.permutation(np.where(avail)[0])
                    w = np.zeros(C.shape[1])
                    w[ids[:n_side]] = 0.5 / n_side
                    w[ids[n_side:2 * n_side]] = -0.5 / n_side
            netn, _ = eval_book(HN, R, Fd, TUNE_FEE_BP * BP)
            null_sh.append(perf(day_ms[test_mask], netn[test_mask])["sharpe"])
        entry["null_random_pick"] = {"mean_sharpe": float(np.mean(null_sh)), "sd": float(np.std(null_sh))}
        out[fname] = entry
    return out


def family_reversal_1h(m, Fh, rng):
    C = m["hc"]
    R = rets(C)
    T, N = C.shape
    out = {"rules_fixed": "fade cross-sectional top-decile |ret| movers, equal-weight gross 1.0, hold 1 bar; lookback (1 vs 3 bars) picked on train Sharpe at 3.5bp"}
    books = {}
    for K in REV_LOOKBACKS:
        sig = np.full_like(C, np.nan)
        sig[K:] = C[K:] / C[:-K] - 1.0
        a = np.abs(sig)
        n_avail = np.isfinite(a).sum(axis=1)
        row_ok = n_avail >= MIN_UNIVERSE_H
        q = np.full(T, np.nan)
        q[row_ok] = np.nanquantile(a[row_ok], DECILE_Q, axis=1)
        sel = row_ok[:, None] & np.isfinite(a) & (a >= q[:, None])
        n_sel = sel.sum(axis=1)
        W = np.zeros((T, N))
        src = np.where(sel, -np.sign(sig), 0.0) / np.maximum(n_sel, 1)[:, None]
        W[1:] = np.nan_to_num(src[:-1])
        books[K] = W
    scores = {}
    for K, W in books.items():
        net, turn = eval_book(W, R, Fh, TUNE_FEE_BP * BP)
        dms, dnet = hourly_to_daily(m["h0"], net)
        tr = dms < TRAIN_END_MS
        scores[K] = perf(dms[tr], dnet[tr])["sharpe"]
    best_K = max(scores, key=scores.get)
    out["lookback_bars"] = best_K
    out["train_sharpe_by_lookback"] = {str(k): round(v, 3) for k, v in scores.items()}
    W = books[best_K]
    for fee_bp in FEES_BP:
        net, turn = eval_book(W, R, Fh, fee_bp * BP)
        dms, dnet = hourly_to_daily(m["h0"], net)
        te = dms >= TRAIN_END_MS
        pf = perf(dms[te], dnet[te])
        if fee_bp == TUNE_FEE_BP:
            tr = dms < TRAIN_END_MS
            pf["train"] = perf(dms[tr], dnet[tr])
            hrs_test = (m["h0"] + np.arange(T)) * MS_H >= TRAIN_END_MS
            yrs = hrs_test.sum() / 24.0 / DAYS_YR
            pf["turnover_yr"] = float(turn[hrs_test].sum() / yrs)
            pf["survives"] = survives(pf)
            gross_net, _ = eval_book(W, R, Fh, 0.0)
            _, dgross = hourly_to_daily(m["h0"], gross_net)
            pf["zero_fee_sharpe"] = perf(dms[te], dgross[te])["sharpe"]
            gross_yr = float(dgross[te].sum() / (te.sum() / DAYS_YR))
            pf["zero_fee_ret_yr"] = gross_yr
            pf["breakeven_bp_side"] = float(gross_yr / pf["turnover_yr"] / BP)
        out["fee_%.1fbp" % fee_bp] = pf
    null_sh = []
    active = W != 0
    for _ in range(N_NULL):
        signs = rng.choice([-1.0, 1.0], size=W.shape)
        WN = np.where(active, np.abs(W) * signs, 0.0)
        netn, _ = eval_book(WN, R, Fh, TUNE_FEE_BP * BP)
        dms, dnet = hourly_to_daily(m["h0"], netn)
        te = dms >= TRAIN_END_MS
        null_sh.append(perf(dms[te], dnet[te])["sharpe"])
    out["null_random_sign"] = {"mean_sharpe": float(np.mean(null_sh)), "sd": float(np.std(null_sh))}
    return out


def wilder_rsi(C):
    T, N = C.shape
    rsi = np.full((T, N), np.nan)
    for j in range(N):
        c = C[:, j]
        m = np.isfinite(c)
        cc = c[m]
        if len(cc) < RSI_PERIOD + 2:
            continue
        delta = np.diff(cc)
        gain = pd.Series(np.clip(delta, 0, None)).ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean().to_numpy()
        loss = pd.Series(np.clip(-delta, 0, None)).ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean().to_numpy()
        r = np.where(loss > 0, 100.0 - 100.0 / (1.0 + gain / np.where(loss > 0, loss, 1.0)), 100.0)
        rsi[np.where(m)[0][1:], j] = r
    return rsi


def rsi_book(C, rsi, hold):
    T, N = C.shape
    long_sig = rsi < RSI_LO
    short_sig = rsi > RSI_HI
    long_act = np.zeros((T, N), bool)
    short_act = np.zeros((T, N), bool)
    for s in range(1, hold + 1):
        long_act[s:] |= long_sig[:-s]
        short_act[s:] |= short_sig[:-s]
    unit = long_act.astype(float) - short_act.astype(float)
    n_active = np.abs(unit).sum(axis=1)
    return unit / np.maximum(n_active, 1)[:, None]


def family_rsi(m, Fd, day_ms, train_mask, test_mask):
    C = m["dc"]
    R = rets(C)
    rsi = wilder_rsi(C)
    out = {"rules_fixed": "RSI(14)<25 long / >75 short mirror, enter next close, equal-weight gross 1.0; hold days picked on train Sharpe at 3.5bp"}
    train_scores = {}
    for hold in RSI_HOLDS:
        W = rsi_book(C, rsi, hold)
        net, _ = eval_book(W, R, Fd, TUNE_FEE_BP * BP)
        train_scores[hold] = perf(day_ms[train_mask], net[train_mask])["sharpe"]
    best_h = max(train_scores, key=train_scores.get)
    out["hold_d"] = best_h
    out["train_sharpe_by_hold"] = {str(k): round(v, 3) for k, v in train_scores.items()}
    W = rsi_book(C, rsi, best_h)
    for fee_bp in FEES_BP:
        net, turn = eval_book(W, R, Fd, fee_bp * BP)
        pf = perf(day_ms[test_mask], net[test_mask])
        if fee_bp == TUNE_FEE_BP:
            pf["train"] = perf(day_ms[train_mask], net[train_mask])
            yrs = (day_ms[test_mask][-1] - day_ms[test_mask][0]) / MS_D / DAYS_YR
            pf["turnover_yr"] = float(turn[test_mask].sum() / yrs)
            pf["survives"] = survives(pf)
            for leg, mask_sign in (("long_leg", 1.0), ("short_leg", -1.0)):
                WL = np.where(np.sign(W) == mask_sign, W, 0.0)
                netl, _ = eval_book(WL, R, Fd, fee_bp * BP)
                pf[leg + "_test_ret"] = float(np.prod(1.0 + np.nan_to_num(netl[test_mask])) - 1.0)
        out["fee_%.1fbp" % fee_bp] = pf
    return out


def family_seasonality(m, Fh, Fd, day_ms, train_mask, test_mask, rng):
    Rh = rets(m["hc"])
    Rd = rets(m["dc"])
    Th = Rh.shape[0]
    n_av = np.isfinite(Rh).sum(axis=1)
    bh = np.where(n_av >= SEAS_MIN_BASKET, np.nansum(Rh, axis=1) / np.maximum(n_av, 1), np.nan)
    fbh = Fh.mean(axis=1)
    n_avd = np.isfinite(Rd).sum(axis=1)
    bd = np.where(n_avd >= SEAS_MIN_BASKET, np.nansum(Rd, axis=1) / np.maximum(n_avd, 1), np.nan)
    fbd = Fd.mean(axis=1)
    h_abs = m["h0"] + np.arange(Th)
    hod = h_abs % 24
    h_ms = h_abs * MS_H
    h_train = h_ms < TRAIN_END_MS
    Td = Rd.shape[0]
    weekday = (np.arange(m["d0"], m["d0"] + Td) + EPOCH_THU) % 7
    cands = []
    for h in range(24):
        x = bh[h_train & (hod == h) & np.isfinite(bh)]
        if len(x) > 1:
            t = x.mean() / (x.std() / np.sqrt(len(x)))
            cands.append(("hour", h, float(np.sign(x.mean())), float(t)))
    for wd in range(7):
        x = bd[train_mask & (weekday == wd) & np.isfinite(bd)]
        if len(x) > 1:
            t = x.mean() / (x.std() / np.sqrt(len(x)))
            cands.append(("dow", wd, float(np.sign(x.mean())), float(t)))
    cands.sort(key=lambda c: -abs(c[3]))
    rules = cands[:SEAS_MAX_RULES]
    out = {"rules_fixed": "top-%d train |t| rules from 24 hour-of-day + 7 day-of-week candidates (max-%d pre-registered control); hold basket sign(train mean) during window" % (SEAS_MAX_RULES, SEAS_MAX_RULES),
           "train_candidates_top5": [{"kind": c[0], "idx": c[1], "sign": c[2], "t": round(c[3], 2)} for c in cands[:5]],
           "rules": [{"kind": c[0], "idx": c[1], "sign": c[2], "train_t": round(c[3], 2)} for c in rules],
           "rules_passing_strict_t%.0f" % SEAS_MIN_T: int(sum(abs(c[3]) >= SEAS_MIN_T for c in rules))}

    def rule_daily(c, fee):
        kind, idx, sign, _ = c
        if kind == "hour":
            fire = hod == idx
            r = np.where(fire & np.isfinite(bh), sign * bh - sign * fbh - 2.0 * fee, 0.0)
            return hourly_to_daily(m["h0"], r)
        r = np.where((weekday == idx) & np.isfinite(bd), sign * bd - sign * fbd - 2.0 * fee, 0.0)
        return day_ms, r

    for fee_bp in FEES_BP:
        fee = fee_bp * BP
        per_rule = []
        combo = np.zeros(Td)
        for c in rules:
            dms, dr = rule_daily(c, fee)
            dr_full = np.zeros(Td)
            i0 = int((dms[0] - day_ms[0]) // MS_D)
            seg = dr[max(0, -i0):]
            n = min(len(seg), Td - max(0, i0))
            dr_full[max(0, i0):max(0, i0) + n] = seg[:n]
            combo += dr_full
            per_rule.append(perf(day_ms[test_mask], dr_full[test_mask]))
        if rules:
            combo /= len(rules)
        pf = perf(day_ms[test_mask], combo[test_mask])
        pf["per_rule"] = per_rule
        if fee_bp == TUNE_FEE_BP:
            pf["train"] = perf(day_ms[train_mask], combo[train_mask])
            pf["survives"] = survives(pf) if rules else False
        out["fee_%.1fbp" % fee_bp] = pf
    null_sh = []
    for _ in range(N_NULL):
        h = int(rng.integers(0, 24))
        sign = float(rng.choice([-1.0, 1.0]))
        dms, dr = rule_daily(("hour", h, sign, 0.0), TUNE_FEE_BP * BP)
        te = dms >= TRAIN_END_MS
        null_sh.append(perf(dms[te], dr[te])["sharpe"])
    out["null_random_hour"] = {"mean_sharpe": float(np.mean(null_sh)), "sd": float(np.std(null_sh))}
    return out


def brk_trades(m, Fh, ATR, sig_mask, hold, fee):
    C, O, H, L = m["hc"], m["ho"], m["hh"], m["hl"]
    T = C.shape[0]
    trades = []
    ti, pi = np.where(sig_mask)
    for t, j in zip(ti, pi):
        entry = C[t, j]
        direction = np.sign(C[t, j] - O[t, j])
        stop = entry - direction * ATR[t, j]
        exit_price, exit_t = None, t
        for i in range(1, hold + 1):
            idx = t + i
            if idx >= T or not np.isfinite(C[idx, j]):
                exit_price = C[exit_t, j]
                break
            exit_t = idx
            if direction > 0 and L[idx, j] <= stop:
                exit_price = stop
                break
            if direction < 0 and H[idx, j] >= stop:
                exit_price = stop
                break
            if i == hold:
                exit_price = C[idx, j]
        if exit_price is None or exit_t == t:
            continue
        gross = direction * (exit_price / entry - 1.0)
        fund = -direction * Fh[t + 1:exit_t + 1, j].sum()
        trades.append((t, exit_t, j, gross + fund - 2.0 * fee))
    return trades


def family_breakout(m, Fh, rng):
    C, O, H, L, V = m["hc"], m["ho"], m["hh"], m["hl"], m["hv"]
    T, N = C.shape
    prev_c = np.vstack([np.full((1, N), np.nan), C[:-1]])
    TR = np.fmax(H - L, np.fmax(np.abs(H - prev_c), np.abs(L - prev_c)))
    kern = np.ones(BRK_ATR_N) / BRK_ATR_N
    ATR = np.full((T, N), np.nan)
    VMA = np.full((T, N), np.nan)
    for j in range(N):
        tr = np.nan_to_num(TR[:, j])
        vv = np.nan_to_num(V[:, j])
        cnt = np.convolve(np.isfinite(TR[:, j]).astype(float), kern, mode="full")[:T]
        sm_tr = np.convolve(tr, kern, mode="full")[:T]
        sm_v = np.convolve(vv, kern, mode="full")[:T]
        full = cnt >= 1.0 - 1e-9
        ATR[1:, j] = np.where(full[:-1], sm_tr[:-1], np.nan)
        VMA[1:, j] = np.where(full[:-1], sm_v[:-1], np.nan)
    h_ms = (m["h0"] + np.arange(T)) * MS_H
    out = {"rules_fixed": "enter breakout direction at close when TR>kr*ATR24 and vol>kv*VMA24, hold H bars, stop 1x ATR; (kr,kv,H) picked on train per-trade t-stat at 3.5bp"}
    grid = {}
    for kr in BRK_KR:
        for kv in BRK_KV:
            sig = (np.nan_to_num(TR) > kr * ATR) & (np.nan_to_num(V) > kv * VMA) & np.isfinite(C) & np.isfinite(O) & (C != O)
            for hold in BRK_HOLDS:
                trades = brk_trades(m, Fh, ATR, sig & (h_ms < TRAIN_END_MS)[:, None], hold, TUNE_FEE_BP * BP)
                nets = np.array([tr[3] for tr in trades])
                if len(nets) < BRK_MIN_TRADES:
                    continue
                tstat = nets.mean() / (nets.std() / np.sqrt(len(nets)))
                grid[(kr, kv, hold)] = (float(tstat), len(nets), float(nets.mean()))
    if not grid:
        out["error"] = "no config reached min train trades"
        return out
    best = max(grid, key=lambda k: grid[k][0])
    kr, kv, hold = best
    out["config"] = {"kr": kr, "kv": kv, "hold_bars": hold}
    out["train_edge"] = {"t": round(grid[best][0], 2), "n": grid[best][1], "mean_net_bp": round(grid[best][2] / BP, 2),
                         "negative_on_train": bool(grid[best][0] < 0)}
    out["train_grid_top5"] = [{"cfg": str(k), "t": round(v[0], 2), "n": v[1], "mean_net_bp": round(v[2] / BP, 2)}
                              for k, v in sorted(grid.items(), key=lambda kv_: -kv_[1][0])[:5]]
    sig = (np.nan_to_num(TR) > kr * ATR) & (np.nan_to_num(V) > kv * VMA) & np.isfinite(C) & np.isfinite(O) & (C != O)
    train_trades = brk_trades(m, Fh, ATR, sig & (h_ms < TRAIN_END_MS)[:, None], hold, TUNE_FEE_BP * BP)
    starts = np.array([tr[0] for tr in train_trades])
    ends = np.array([tr[1] for tr in train_trades])
    conc = np.zeros(T)
    np.add.at(conc, starts, 1.0)
    np.add.at(conc, np.minimum(ends + 1, T - 1), -1.0)
    conc = np.cumsum(conc)
    K = max(1, int(np.ceil(np.percentile(conc[conc > 0], BRK_SLOT_PCT))))
    out["capital_slots"] = K
    sig_test = sig & (h_ms >= TRAIN_END_MS)[:, None]
    for fee_bp in FEES_BP:
        trades = brk_trades(m, Fh, ATR, sig_test, hold, fee_bp * BP)
        nets = np.array([tr[3] for tr in trades])
        exit_day = np.array([(m["h0"] + tr[1]) // 24 for tr in trades])
        d0 = int((TRAIN_END_MS // MS_D))
        d1 = int(h_ms[-1] // MS_D)
        dret = np.zeros(d1 - d0 + 1)
        ok = (exit_day >= d0) & (exit_day <= d1)
        np.add.at(dret, exit_day[ok] - d0, nets[ok] / K)
        dms = np.arange(d0, d1 + 1) * MS_D
        pf = perf(dms, dret)
        pf["n_trades"] = int(len(nets))
        pf["mean_net_bp"] = float(nets.mean() / BP) if len(nets) else 0.0
        pf["mean_gross_bp"] = pf["mean_net_bp"] + 2.0 * fee_bp
        pf["t_stat"] = float(nets.mean() / (nets.std() / np.sqrt(len(nets)))) if len(nets) > 1 else 0.0
        if fee_bp == TUNE_FEE_BP:
            pf["survives"] = survives(pf)
        out["fee_%.1fbp" % fee_bp] = pf
    n_test = int(sig_test.sum())
    null_means = []
    valid_next = np.isfinite(C) & (np.roll(np.isfinite(C), -hold, axis=0))
    cand_t, cand_j = np.where(valid_next & (h_ms >= TRAIN_END_MS)[:, None] & (h_ms < h_ms[-1] - hold * MS_H)[:, None])
    for _ in range(N_NULL):
        pick = rng.integers(0, len(cand_t), size=min(n_test, len(cand_t)))
        tt, jj = cand_t[pick], cand_j[pick]
        dirs = rng.choice([-1.0, 1.0], size=len(tt))
        gr = dirs * (C[tt + hold, jj] / C[tt, jj] - 1.0)
        null_means.append(float(np.nanmean(gr) / BP - 2.0 * TUNE_FEE_BP))
    out["null_random_entry"] = {"mean_net_bp": float(np.mean(null_means)), "sd_bp": float(np.std(null_means))}
    return out


def main():
    t_start = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)
    stats = {"smoke": "costkilled_retest", "date": pd.Timestamp.now("UTC").isoformat(), "errors": []}
    rng = np.random.default_rng(NULL_SEED)
    pairs = list_pairs()
    for p in pairs:
        resample_pair(p)
    m = build_matrices(pairs)
    Th, N = m["hc"].shape
    Td = m["dc"].shape[0]
    Fh, Fd = load_funding(pairs, m["h0"], Th, m["d0"], Td)
    day_ms = (m["d0"] + np.arange(Td)) * MS_D
    train_mask = day_ms < TRAIN_END_MS
    test_mask = day_ms >= TRAIN_END_MS
    stats["data"] = {"pairs": N, "days": int(Td), "hours": int(Th),
                     "start": str(pd.Timestamp(day_ms[0], unit="ms").date()),
                     "end": str(pd.Timestamp(day_ms[-1], unit="ms").date()),
                     "train_end": "2024-01-01", "listing_trim_days": LISTING_TRIM_MS // MS_D,
                     "funding_note": "realized 8h funding charged long/credited short; zero before pair funding history starts (2020+)"}
    fams = [("momentum_xs", lambda: family_momentum(m, Fd, day_ms, train_mask, test_mask, rng)),
            ("reversal_1h", lambda: family_reversal_1h(m, Fh, rng)),
            ("rsi_daily", lambda: family_rsi(m, Fd, day_ms, train_mask, test_mask)),
            ("seasonality", lambda: family_seasonality(m, Fh, Fd, day_ms, train_mask, test_mask, rng)),
            ("breakout_vol", lambda: family_breakout(m, Fh, rng))]
    stats["families"] = {}
    for name, fn in fams:
        t0 = time.time()
        try:
            stats["families"][name] = fn()
            stats["families"][name]["runtime_s"] = round(time.time() - t0, 1)
        except Exception:
            stats["errors"].append("%s: %s" % (name, traceback.format_exc(limit=3)))
    survivors = []
    for name, fam in stats["families"].items():
        for key, val in fam.items():
            if isinstance(val, dict) and val.get("survives"):
                survivors.append("%s/%s" % (name, key))
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, dict) and v2.get("survives"):
                        survivors.append("%s/%s/%s" % (name, key, k2))
    stats["survivors"] = survivors
    stats["verdict"] = ("SUCCESS: survivors found" if survivors else
                        "FAILURE: all families dead at true costs") if not stats["errors"] else "INCONCLUSIVE: errors"
    stats["runtime_s"] = round(time.time() - t_start, 1)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=1, default=float)
    print(json.dumps({"verdict": stats["verdict"], "survivors": survivors, "errors": len(stats["errors"]),
                      "runtime_s": stats["runtime_s"]}, indent=1))
    return 0 if not stats["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
