# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - tests daily pairs stat-arb with a real short leg on 40 Binance USDT majors at
#   perps costs (2.6/3.5/5bp per side per leg, realized 8h funding both legs):
#   yearly walk-forward, pairs selected on expanding train window only (Engle-Granger
#   log-price cointegration p<0.05 via OLS+ADF, MacKinnon N=2 CV, AND spread half-life
#   2-30d), z entry |z|>=2 on cross, exit |z|<0.5 / 30d timeout / |z|>4 stop, dollar-
#   neutral legs, 1/10-capital slots cap 10 concurrent; z definition (train-window vs
#   rolling-90d) picked in-sample on the pre-2022 formation window only. Plus daily
#   beta-hedged dispersion variant (7d residual vs BTC, decile L/S). Nulls: shuffled
#   entries same count/duration/pair. Falsifier: selection unstable OOS and/or net
#   negative at 3.5bp in >2/5 years or Sharpe < 0.8 or maxDD >= 40% -> family closed.
# - wall-clock estimate ~4 min with npz cache, ~30 min cold.
# SMOKE_RESULTS/daily_statarb_smoke.py
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
             "10655374-b645-4bca-a6e1-0208baad89ab/scratchpad/pairs_cache")
STATS_PATH = os.path.join(REPO, "SMOKE_RESULTS/daily_statarb_stats.json")

MS_D = 86_400_000
US_SWITCH = 10 ** 14
LISTING_TRIM_MS = 3 * MS_D
MIN_MINUTES_D = 1320
DAYS_YR = 365.0
BP = 1e-4

TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
PICK_SIM_START = "2019-01-01"
PICK_TRAIN_END = "2022-01-01"
FEES_BP = [2.6, 3.5, 5.0]
PICK_FEE_BP = 3.5

MIN_TRAIN_DAYS = 365
ALIVE_WIN_D = 7
FFILL_LIMIT = 5
ADF_MAXLAG = 4
EG_CV5 = (-3.33613, -6.1101, -6.823)
HL_MIN = 2.0
HL_MAX = 30.0

Z_ENTRY = 2.0
Z_EXIT = 0.5
Z_STOP = 4.0
MAX_HOLD_D = 30
MAX_OPEN = 10
SLOT_W = 0.1
ROLL_Z_WIN = 90

DISP_LOOKBACK = 7
DISP_DECILE = 0.1
DISP_BETA_WIN = 365
DISP_BETA_MIN = 180
DISP_MIN_UNIV = 20

N_NULL = 100
NULL_SEED = 7
SURV_SHARPE = 0.8
SURV_MAXDD = 0.40
SURV_MIN_YEARS = 3
SURV_RECENT = ("2025", "2026")

# ------------------------------------------------------------------------------------
# Functions


def list_pairs():
    return sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))


def resample_pair(pair):
    cache = os.path.join(CACHE_DIR, pair + ".npz")
    if os.path.exists(cache):
        return
    df = pd.read_csv(os.path.join(DATA_DIR, pair + ".csv"), usecols=["time", "close"])
    t = df["time"].to_numpy(np.int64)
    t = np.where(t > US_SWITCH, t // 1000, t)
    c = df["close"].to_numpy(float)
    keep = t >= t.min() + LISTING_TRIM_MS
    t, c = t[keep], c[keep]
    idx = t // MS_D
    uniq, first = np.unique(idx, return_index=True)
    out = {"didx": uniq,
           "dc": c[np.append(first[1:] - 1, len(t) - 1)],
           "dn": np.diff(np.append(first, len(t)))}
    np.savez_compressed(cache, **out)


def build_daily(pairs):
    data = {p: np.load(os.path.join(CACHE_DIR, p + ".npz")) for p in pairs}
    d0 = min(int(d["didx"].min()) for d in data.values())
    d1 = max(int(d["didx"].max()) for d in data.values())
    T = d1 - d0 + 1
    C = np.full((T, len(pairs)), np.nan)
    for j, p in enumerate(pairs):
        d = data[p]
        valid = d["dn"] >= MIN_MINUTES_D
        C[d["didx"][valid] - d0, j] = d["dc"][valid]
    day_ms = (np.arange(T) + d0) * MS_D
    return C, day_ms


def load_funding(pairs, d0, Td):
    Fd = np.zeros((Td, len(pairs)))
    for j, p in enumerate(pairs):
        path = os.path.join(FUNDING_DIR, p + ".csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        t = df["time"].to_numpy(np.int64)
        t = np.where(t > US_SWITCH, t // 1000, t)
        f = df["funding"].to_numpy(float)
        di = t // MS_D - d0
        ok = (di >= 0) & (di < Td)
        np.add.at(Fd[:, j], di[ok], f[ok])
    return Fd


def day_ms_of(date_str):
    return int(pd.Timestamp(date_str).value // 10 ** 6)


def ols_t(y, X):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    sigma2 = resid @ resid / (n - p)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    return beta, resid, beta / se


def adf_t(e):
    de = np.diff(e)
    n0 = len(de) - ADF_MAXLAG
    best = None
    for k in range(ADF_MAXLAG + 1):
        y = de[ADF_MAXLAG:]
        cols = [e[ADF_MAXLAG:-1]]
        for i in range(1, k + 1):
            cols.append(de[ADF_MAXLAG - i:-i])
        X = np.column_stack(cols)
        beta, resid, t = ols_t(y, X)
        rss = resid @ resid
        aic = n0 * np.log(rss / n0) + 2 * (k + 1)
        if best is None or aic < best[0]:
            best = (aic, float(t[0]))
    return best[1], n0


def eg_cv5(n):
    b0, b1, b2 = EG_CV5
    return b0 + b1 / n + b2 / n ** 2


def half_life(e):
    de = np.diff(e)
    lag = e[:-1]
    phi = (lag @ de) / (lag @ lag)
    if not -1.0 < phi < 0.0:
        return np.inf
    return -np.log(2.0) / np.log(1.0 + phi)


def select_pairs(LC, train_mask):
    P = LC.shape[1]
    train_idx = np.where(train_mask)[0]
    tail = train_idx[-ALIVE_WIN_D:]
    n_tested = n_coint = 0
    selected = []
    for a in range(P):
        for b in range(a + 1, P):
            la, lb = LC[train_mask, a], LC[train_mask, b]
            joint = ~np.isnan(la) & ~np.isnan(lb)
            if joint.sum() < MIN_TRAIN_DAYS:
                continue
            if np.isnan(LC[tail, a]).all() or np.isnan(LC[tail, b]).all():
                continue
            n_tested += 1
            ya, yb = la[joint], lb[joint]
            X = np.column_stack([np.ones(len(yb)), yb])
            beta, resid, _ = ols_t(ya, X)
            t, n = adf_t(resid)
            if t >= eg_cv5(n):
                continue
            n_coint += 1
            hl = half_life(resid)
            if not HL_MIN <= hl <= HL_MAX:
                continue
            selected.append({"a": a, "b": b, "beta": float(beta[1]),
                             "alpha": float(beta[0]), "sd": float(resid.std()),
                             "adf_t": float(t), "hl": float(hl)})
    return selected, n_tested, n_coint


def z_series(LC, sel, variant):
    s = LC[:, sel["a"]] - sel["beta"] * LC[:, sel["b"]] - sel["alpha"]
    if variant == "train":
        return s / sel["sd"]
    sp = pd.Series(s)
    m = sp.rolling(ROLL_Z_WIN, min_periods=ROLL_Z_WIN).mean()
    sd = sp.rolling(ROLL_Z_WIN, min_periods=ROLL_Z_WIN).std()
    return ((sp - m) / sd).to_numpy()


def simulate(sels, Z, R, Fd, t0, t1):
    pnl = np.zeros(t1 - t0 + 1)
    gross = np.zeros(t1 - t0 + 1)
    open_pos = {}
    trades = []
    traded_notional = 0.0
    for t in range(t0, t1 + 1):
        i = t - t0
        for k, pos in open_pos.items():
            sel = sels[k]
            ra = np.nan_to_num(R[t, sel["a"]])
            rb = np.nan_to_num(R[t, sel["b"]])
            fa, fb = Fd[t, sel["a"]], Fd[t, sel["b"]]
            leg = pos["dir"] * SLOT_W / 2.0
            pnl[i] += leg * (ra - rb) - leg * (fa - fb)
            gross[i] += SLOT_W
        closing = []
        for k, pos in open_pos.items():
            z = Z[k][t]
            age = t - pos["entry_t"]
            reason = None
            if np.isnan(z):
                reason = "data"
            elif abs(z) < Z_EXIT:
                reason = "converged"
            elif abs(z) > Z_STOP:
                reason = "stopped"
            elif age >= MAX_HOLD_D:
                reason = "timeout"
            elif t == t1:
                reason = "year_end"
            if reason:
                closing.append((k, reason))
        for k, reason in closing:
            pos = open_pos.pop(k)
            trades.append({"k": k, "dir": pos["dir"], "entry_t": pos["entry_t"],
                           "exit_t": t, "reason": reason})
        entries = []
        for k in range(len(sels)):
            if k in open_pos:
                continue
            z, zp = Z[k][t], Z[k][t - 1]
            if np.isnan(z) or np.isnan(zp):
                continue
            if abs(z) >= Z_ENTRY and abs(zp) < Z_ENTRY and t < t1:
                entries.append((abs(z), k, -np.sign(z)))
        entries.sort(reverse=True)
        for _, k, direction in entries:
            if len(open_pos) >= MAX_OPEN:
                break
            open_pos[k] = {"dir": direction, "entry_t": t}
    fee_events = np.zeros(t1 - t0 + 1)
    for tr in trades:
        fee_events[tr["entry_t"] - t0] += SLOT_W
        fee_events[tr["exit_t"] - t0] += SLOT_W
        traded_notional += 2.0 * SLOT_W
    return pnl, gross, trades, fee_events, traded_notional


def perf(day_ms, ret):
    ret = np.nan_to_num(np.asarray(ret, float))
    eq = np.cumprod(1.0 + ret)
    yrs = len(ret) / DAYS_YR
    cagr = eq[-1] ** (1.0 / yrs) - 1.0 if yrs > 0 and eq[-1] > 0 else -1.0
    sd = ret.std()
    sharpe = ret.mean() / sd * np.sqrt(DAYS_YR) if sd > 0 else 0.0
    dd = 1.0 - eq / np.maximum.accumulate(eq)
    years = pd.to_datetime(day_ms, unit="ms").year
    by_year = {}
    for y in np.unique(years):
        m = years == y
        by_year[str(y)] = float(np.prod(1.0 + ret[m]) - 1.0)
    half = len(ret) // 2
    halves = []
    for r in (ret[:half], ret[half:]):
        s = r.std()
        halves.append(float(r.mean() / s * np.sqrt(DAYS_YR)) if s > 0 else 0.0)
    return {"cagr": float(cagr), "sharpe": float(sharpe), "maxdd": float(dd.max()),
            "by_year": by_year, "halves_sharpe": halves}


def year_stats(day_ms, ret, gross):
    years = pd.to_datetime(day_ms, unit="ms").year
    out = {}
    for y in np.unique(years):
        m = years == y
        r = ret[m]
        sd = r.std()
        eq = np.cumprod(1.0 + r)
        dd = 1.0 - eq / np.maximum.accumulate(eq)
        out[str(y)] = {"net": float(np.prod(1.0 + r) - 1.0),
                       "sharpe": float(r.mean() / sd * np.sqrt(DAYS_YR)) if sd > 0 else 0.0,
                       "maxdd": float(dd.max()),
                       "deploy": float(gross[m].mean())}
    return out


def null_entries(sels, trades, R, Fd, t0, t1, fee, rng):
    cums = {}
    for tr in trades:
        k = tr["k"]
        if k not in cums:
            sel = sels[k]
            dr = np.nan_to_num(R[:, sel["a"]]) - np.nan_to_num(R[:, sel["b"]])
            df = Fd[:, sel["a"]] - Fd[:, sel["b"]]
            cums[k] = (np.concatenate([[0.0], np.cumsum(dr)]),
                       np.concatenate([[0.0], np.cumsum(df)]))
    outs = []
    for _ in range(N_NULL):
        tot = 0.0
        for tr in trades:
            dur = tr["exit_t"] - tr["entry_t"]
            e = rng.integers(t0, max(t0 + 1, t1 - dur))
            d = rng.choice([-1.0, 1.0])
            cr, cf = cums[tr["k"]]
            leg = d * SLOT_W / 2.0
            tot += leg * (cr[e + dur + 1] - cr[e + 1]) - leg * (cf[e + dur + 1] - cf[e + 1])
            tot -= 2.0 * SLOT_W * fee
        outs.append(tot)
    return outs


def obs_arith(sels, trades, R, Fd, fee):
    tot = 0.0
    for tr in trades:
        sel = sels[tr["k"]]
        sl = slice(tr["entry_t"] + 1, tr["exit_t"] + 1)
        dr = np.nan_to_num(R[sl, sel["a"]]) - np.nan_to_num(R[sl, sel["b"]])
        df = Fd[sl, sel["a"]] - Fd[sl, sel["b"]]
        leg = tr["dir"] * SLOT_W / 2.0
        tot += leg * dr.sum() - leg * df.sum() - 2.0 * SLOT_W * fee
    return tot


def run_statarb(LC, R, Fd, day_ms, variant, windows, fee):
    pnl_all, gross_all, ms_all = [], [], []
    detail = {}
    for y, sels, Z, t0, t1 in windows:
        pnl, gross, trades, fee_ev, notional = simulate(sels, Z, R, Fd, t0, t1)
        net = pnl - fee_ev * fee
        pnl_all.append(net)
        gross_all.append(gross)
        ms_all.append(day_ms[t0:t1 + 1])
        reasons = {}
        for tr in trades:
            reasons[tr["reason"]] = reasons.get(tr["reason"], 0) + 1
        detail[str(y)] = {"n_trades": len(trades), "exits": reasons,
                          "turnover_x": float(notional / ((t1 - t0 + 1) / DAYS_YR)),
                          "trades": trades}
    return (np.concatenate(pnl_all), np.concatenate(gross_all),
            np.concatenate(ms_all), detail)


def dispersion(C, LC, R, Fd, day_ms, btc_j, t0, t1, fees, rng):
    T, P = R.shape
    rdf = pd.DataFrame(R)
    rb = rdf[btc_j]
    cov = rdf.rolling(DISP_BETA_WIN, min_periods=DISP_BETA_MIN).cov(rb)
    var = rb.rolling(DISP_BETA_WIN, min_periods=DISP_BETA_MIN).var()
    beta = cov.div(var, axis=0).to_numpy()
    Cf = pd.DataFrame(C).ffill(limit=FFILL_LIMIT).to_numpy()
    r7 = Cf[DISP_LOOKBACK:] / Cf[:-DISP_LOOKBACK] - 1.0
    r7 = np.vstack([np.full((DISP_LOOKBACK, P), np.nan), r7])
    resid = r7 - beta * r7[:, [btc_j]]
    resid[:, btc_j] = np.nan
    W = np.zeros((T, P))
    for t in range(t0 - 1, t1):
        row = resid[t]
        ok = np.where(~np.isnan(row))[0]
        if len(ok) < DISP_MIN_UNIV:
            continue
        k = max(2, int(round(len(ok) * DISP_DECILE)))
        order = ok[np.argsort(row[ok])]
        W[t, order[:k]] = 1.0 / (2 * k)
        W[t, order[-k:]] = -1.0 / (2 * k)
    Rz = np.nan_to_num(R)
    pnl = (W[t0 - 1:t1] * (Rz[t0:t1 + 1] - Fd[t0:t1 + 1])).sum(axis=1)
    dW = np.abs(np.diff(np.vstack([np.zeros(P), W[t0 - 1:t1]]), axis=0)).sum(axis=1)
    ms = day_ms[t0:t1 + 1]
    yrs = (t1 - t0 + 1) / DAYS_YR
    out = {"turnover_x": float(dW.sum() / yrs),
           "deploy": float(np.abs(W[t0 - 1:t1]).sum(axis=1).mean())}
    for fee_bp in fees:
        net = pnl - dW * fee_bp * BP
        out[f"fee_{fee_bp}bp"] = perf(ms, net)
    nulls = []
    for _ in range(N_NULL):
        Wn = np.zeros((t1 - t0 + 1, P))
        for i, t in enumerate(range(t0 - 1, t1)):
            row = resid[t]
            ok = np.where(~np.isnan(row))[0]
            if len(ok) < DISP_MIN_UNIV:
                continue
            k = max(2, int(round(len(ok) * DISP_DECILE)))
            pick = rng.choice(ok, 2 * k, replace=False)
            Wn[i, pick[:k]] = 1.0 / (2 * k)
            Wn[i, pick[k:]] = -1.0 / (2 * k)
        pn = (Wn * (Rz[t0:t1 + 1] - Fd[t0:t1 + 1])).sum(axis=1)
        dWn = np.abs(np.diff(np.vstack([np.zeros(P), Wn]), axis=0)).sum(axis=1)
        net = pn - dWn * PICK_FEE_BP * BP
        sd = net.std()
        nulls.append(float(net.mean() / sd * np.sqrt(DAYS_YR)) if sd > 0 else 0.0)
    obs = out[f"fee_{PICK_FEE_BP}bp"]["sharpe"]
    out["null_sharpe_mean"] = float(np.mean(nulls))
    out["null_sharpe_sd"] = float(np.std(nulls))
    out["obs_sharpe_pctile"] = float((np.array(nulls) < obs).mean())
    return out


def survives(pf, ystats):
    pos = [y for y, s in ystats.items() if s["net"] > 0]
    recent = any(y in pos for y in SURV_RECENT)
    return (len(pos) >= SURV_MIN_YEARS and recent
            and pf["sharpe"] >= SURV_SHARPE and pf["maxdd"] < SURV_MAXDD)


def main():
    t_start = time.time()
    stats = {"smoke": "daily_statarb", "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
             "errors": []}
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        pairs = list_pairs()
        for p in pairs:
            resample_pair(p)
        C, day_ms = build_daily(pairs)
        T, P = C.shape
        Fd = load_funding(pairs, int(day_ms[0] // MS_D), T)
        Cf = pd.DataFrame(C).ffill(limit=FFILL_LIMIT).to_numpy()
        LC = np.log(Cf)
        R = np.full_like(Cf, np.nan)
        R[1:] = Cf[1:] / Cf[:-1] - 1.0
        btc_j = pairs.index("BTCUSDT")
        last_valid = int(np.where(~np.isnan(C).all(axis=1))[0][-1])
        stats["data"] = {"pairs": P, "days": T,
                         "start": str(pd.to_datetime(day_ms[0], unit="ms").date()),
                         "end": str(pd.to_datetime(day_ms[last_valid], unit="ms").date())}

        # z-variant pick: in-sample on pre-2022 formation window only
        pick_train = day_ms < day_ms_of(PICK_TRAIN_END)
        sels0, nt0, nc0 = select_pairs(LC, pick_train)
        sim0 = np.where(day_ms >= day_ms_of(PICK_SIM_START))[0][0]
        sim1 = int(np.where(pick_train)[0][-1])
        pick = {}
        for variant in ("train", "roll"):
            Z = [z_series(LC, s, variant) for s in sels0]
            pnl, gross, trades, fee_ev, _ = simulate(sels0, Z, R, Fd, sim0, sim1)
            net = pnl - fee_ev * PICK_FEE_BP * BP
            sd = net.std()
            pick[variant] = float(net.mean() / sd * np.sqrt(DAYS_YR)) if sd > 0 else 0.0
        variant = max(pick, key=pick.get)
        stats["zscore_choice"] = {"train_window_sharpe_at_3p5": pick,
                                  "chosen": variant,
                                  "note": "picked in-sample on formation window ending 2021-12-31"}

        # yearly walk-forward selection + windows
        windows, sel_report, prev_set = [], {}, None
        for y in TEST_YEARS:
            train_mask = day_ms < day_ms_of(f"{y}-01-01")
            sels, n_tested, n_coint = select_pairs(LC, train_mask)
            t0 = int(np.where(~train_mask)[0][0])
            t1 = min(int(np.where(day_ms < day_ms_of(f"{y + 1}-01-01"))[0][-1]), last_valid)
            Z = [z_series(LC, s, variant) for s in sels]
            windows.append((y, sels, Z, t0, t1))
            cur = {(s["a"], s["b"]) for s in sels}
            persist = float(len(cur & prev_set) / len(prev_set)) if prev_set else None
            sel_report[str(y)] = {"pairs_tested": n_tested, "coint_pass": n_coint,
                                  "selected": len(sels),
                                  "persist_from_prev": persist,
                                  "median_hl": float(np.median([s["hl"] for s in sels])) if sels else None}
            prev_set = cur
        stats["pair_selection"] = sel_report

        # trade + score per fee
        rng = np.random.default_rng(NULL_SEED)
        statarb = {}
        for fee_bp in FEES_BP:
            fee = fee_bp * BP
            net, gross, ms, detail = run_statarb(LC, R, Fd, day_ms, variant, windows, fee)
            pf = perf(ms, net)
            ys = year_stats(ms, net, gross)
            for y in ys:
                ys[y].update({k: v for k, v in detail[y].items() if k != "trades"})
            entry = {"full": pf, "by_year": ys}
            if fee_bp == PICK_FEE_BP:
                nulls_by_year = {}
                for (y, sels, Z, t0, t1) in windows:
                    trades = detail[str(y)]["trades"]
                    if not trades:
                        nulls_by_year[str(y)] = None
                        continue
                    nl = null_entries(sels, trades, R, Fd, t0, t1, fee, rng)
                    ob = obs_arith(sels, trades, R, Fd, fee)
                    nulls_by_year[str(y)] = {
                        "obs_arith_net": float(ob),
                        "null_mean": float(np.mean(nl)), "null_sd": float(np.std(nl)),
                        "obs_pctile": float((np.array(nl) < ob).mean())}
                entry["null_shuffled_entries"] = nulls_by_year
                entry["survives"] = survives(pf, ys)
            statarb[f"fee_{fee_bp}bp"] = entry
        stats["statarb"] = statarb

        # dispersion variant
        d0 = int(np.where(day_ms >= day_ms_of(f"{TEST_YEARS[0]}-01-01"))[0][0])
        stats["dispersion"] = dispersion(C, LC, R, Fd, day_ms, btc_j, d0, last_valid,
                                         FEES_BP, rng)
        dpf = stats["dispersion"][f"fee_{PICK_FEE_BP}bp"]
        dys = {y: {"net": n} for y, n in dpf["by_year"].items()}
        stats["dispersion"]["survives"] = survives(dpf, dys)

        sa = statarb[f"fee_{PICK_FEE_BP}bp"]
        stats["survivors"] = {"statarb": sa["survives"],
                              "dispersion": stats["dispersion"]["survives"]}
        stats["verdict"] = ("SURVIVOR" if any(stats["survivors"].values())
                            else "no survivor: daily pairs/stat-arb family closed at perps costs")
    except Exception:
        stats["errors"].append(traceback.format_exc())
    stats["runtime_s"] = round(time.time() - t_start, 1)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in stats.items() if k in
                      ("errors", "verdict", "runtime_s", "zscore_choice")}, indent=1))
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
