# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - definitive daily-to-weekly horizon sweep: pooled ML panel (ridge logistic +
#   HistGradientBoosting cls/reg) over 40 crypto majors at 1d/3d/7d label horizons,
#   walk-forward yearly 2022-2026, quintile L/S + long-only portfolios with
#   overlapping tranches, 2.6/3.5/5bp per side + realized funding. Falsifier: ML
#   panel does not beat plain 1w momentum after costs at any daily+ horizon ->
#   feature set exhausted at daily too.
# - wall-clock estimate ~25 min first run (1m->1D resample of 40 pairs), ~10 min cached.
# SMOKE_RESULTS/daily_ml_panel_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------------
# Constants

REPO = "/Users/mirach-00-usc1/Development/Veritate"
DATA_DIR = os.path.join(REPO, "extensions/installed/market/data/crypto_of")
FUNDING_DIR = os.path.join(REPO, "extensions/installed/market/data/funding")
CACHE_DIR = ("/private/tmp/claude-501/-Users-mirach-00-usc1-Development-Veritate/"
             "10655374-b645-4bca-a6e1-0208baad89ab/scratchpad/daily_panel")
STATS_PATH = os.path.join(REPO, "SMOKE_RESULTS", "daily_ml_panel_stats.json")

PRICE_COLS = ["time", "open", "high", "low", "close", "volume"]
US_SWITCH = 1e14
MS_PER_DAY = 86400000
MIN_DAY_MINUTES = 1000
DAYS_PER_YEAR = 365

RET_WINDOWS = [1, 3, 7, 14, 30, 90]
VOL_SHORT = 7
VOL_LONG = 30
EXP_SHORT = 7
EXP_LONG = 30
RSI_WIN = 14
HI_LO_WINDOWS = [30, 90]
FUND_WIN = 7
BETA_WIN = 90
MAG_CLIP = 10.0

HORIZONS = [1, 3, 7]
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
COSTS_BP = [2.6, 3.5, 5.0]
BAR_COST_BP = 3.5
QUANTILE = 0.2
MIN_NAMES = 10
SEED = 0
N_RANDOM_SEEDS = 5

LOGIT_C = 1.0
LOGIT_MAX_ITER = 2000
HGB_MAX_ITER = 200

SURVIVOR_MIN_POS_YEARS = 3
SURVIVOR_MIN_SHARPE = 0.8
HALF_CATASTROPHIC = -0.20

FEATS = (["retz_%d" % h for h in RET_WINDOWS]
         + ["vol_7", "vol_30", "vol_ratio", "range_exp", "volume_exp", "rsi_14"]
         + ["dist_hi_%d" % w for w in HI_LO_WINDOWS]
         + ["dist_lo_%d" % w for w in HI_LO_WINDOWS]
         + ["fund_lvl", "fund_chg7", "xs_rank_7", "xs_rank_30", "residz_7"]
         + ["dow_%d" % d for d in range(7)])

# ------------------------------------------------------------------------------------
# Functions


def load_daily(pair):
    cache = os.path.join(CACHE_DIR, pair + ".parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = pd.read_csv(os.path.join(DATA_DIR, pair + ".csv"), usecols=PRICE_COLS)
    t = df["time"].to_numpy(np.int64)
    t = np.where(t > US_SWITCH, t // 1000, t)
    df["day"] = t // MS_PER_DAY
    g = df.groupby("day").agg(open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last"),
                              volume=("volume", "sum"), nmin=("close", "size"))
    g.index = pd.to_datetime(g.index, unit="D")
    g.to_parquet(cache)
    return g


def load_funding_daily(pair):
    path = os.path.join(FUNDING_DIR, pair + ".csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    t = df["time"].to_numpy(np.int64)
    t = np.where(t > US_SWITCH, t // 1000, t)
    day = pd.to_datetime(t // MS_PER_DAY, unit="D")
    return df.groupby(day)["funding"].sum()


def compute_features(daily, fund, btc_r1, btc_ret7):
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
    for h in HORIZONS:
        fwd = logc.shift(-h) - logc
        f["y_sign_%d" % h] = (fwd > 0).astype(float).where(fwd.notna())
        f["y_mag_%d" % h] = (fwd / (vol_l * np.sqrt(h))).clip(-MAG_CLIP, MAG_CLIP)
    return f


def portfolio(scores, rets, fund, h, cost_bp, long_only, use_funding):
    dates = scores.index
    cols = rets.columns
    tranche = pd.DataFrame(0.0, index=dates, columns=cols)
    for t in dates:
        s = scores.loc[t].dropna()
        n = len(s)
        if n < MIN_NAMES:
            continue
        k = max(1, int(n * QUANTILE))
        top = s.nlargest(k).index
        if long_only:
            tranche.loc[t, top] = 1.0 / (h * k)
        else:
            bot = s.nsmallest(k).index
            tranche.loc[t, top] = 0.5 / (h * k)
            tranche.loc[t, bot] = tranche.loc[t, bot] - 0.5 / (h * k)
    book = tranche.rolling(h, min_periods=1).sum()
    turnover = (book - book.shift(1).fillna(0.0)).abs().sum(axis=1)
    held = book.shift(1).fillna(0.0)
    r = rets.reindex(index=dates, columns=cols).fillna(0.0)
    pnl = (held * r).sum(axis=1) - cost_bp / 1e4 * turnover
    if use_funding:
        fd = fund.reindex(index=dates, columns=cols).fillna(0.0)
        pnl = pnl - (held * fd).sum(axis=1)
    return pnl, turnover


def metrics(pnl, turnover):
    n = len(pnl)
    if n == 0 or pnl.std() == 0:
        return {"net_yr": 0.0, "sharpe": 0.0, "maxdd": 0.0, "turnover_yr": 0.0,
                "h1_net": 0.0, "h2_net": 0.0}
    ann = DAYS_PER_YEAR / n
    cum = pnl.cumsum()
    half = n // 2
    return {"net_yr": round(float(pnl.sum() * ann), 4),
            "sharpe": round(float(pnl.mean() / pnl.std() * np.sqrt(DAYS_PER_YEAR)), 3),
            "maxdd": round(float((cum.cummax() - cum).max()), 4),
            "turnover_yr": round(float(turnover.sum() * ann), 1),
            "h1_net": round(float(pnl.iloc[:half].sum()), 4),
            "h2_net": round(float(pnl.iloc[half:].sum()), 4)}


def fit_score(name, x_tr, y_sign_tr, y_mag_tr, x_te, rng=None):
    if name == "logit":
        sc = StandardScaler().fit(x_tr)
        m = LogisticRegression(C=LOGIT_C, max_iter=LOGIT_MAX_ITER)
        m.fit(sc.transform(x_tr), y_sign_tr)
        return m.predict_proba(sc.transform(x_te))[:, 1]
    if name == "hgb_cls":
        y = y_sign_tr if rng is None else rng.permutation(y_sign_tr)
        m = HistGradientBoostingClassifier(max_iter=HGB_MAX_ITER, random_state=SEED)
        m.fit(x_tr, y)
        return m.predict_proba(x_te)[:, 1]
    m = HistGradientBoostingRegressor(max_iter=HGB_MAX_ITER, random_state=SEED)
    m.fit(x_tr, y_mag_tr)
    return m.predict(x_te)


def main():
    t_start = time.time()
    out = {"smoke": "daily_ml_panel", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "errors": [], "meta": {}, "results": {}, "nulls": {}}
    os.makedirs(CACHE_DIR, exist_ok=True)
    pairs = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".csv"))

    btc_daily = load_daily("BTCUSDT")
    btc_idx = pd.date_range(btc_daily.index[0], btc_daily.index[-1], freq="D")
    btc_logc = np.log(btc_daily["close"].reindex(btc_idx))
    btc_r1 = btc_logc.diff()
    btc_ret7 = btc_logc - btc_logc.shift(7)

    frames, rets_w, fund_w, gap_audit = [], {}, {}, {}
    for pair in pairs:
        try:
            daily = load_daily(pair)
            f = compute_features(daily, load_funding_daily(pair), btc_r1, btc_ret7)
            gap_audit[pair] = {"days": len(f), "bad_days": int(f["n_bad_days"].sum())}
            rets_w[pair] = f["r1_simple"]
            fund_w[pair] = f["fund_daily"]
            f = f.drop(columns=["r1_simple", "fund_daily", "n_bad_days"])
            f["pair"] = pair
            frames.append(f)
        except Exception:
            out["errors"].append("pair %s: %s" % (pair, traceback.format_exc(limit=2)))
    rets = pd.DataFrame(rets_w)
    fund = pd.DataFrame(fund_w)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel["xs_rank_7"] = panel.groupby("date")["ret_7"].rank(pct=True) - 0.5
    panel["xs_rank_30"] = panel.groupby("date")["ret_30"].rank(pct=True) - 0.5
    panel = panel[panel[FEATS].notna().all(axis=1)]
    out["meta"] = {"pairs": len(pairs), "panel_rows": len(panel),
                   "panel_span": [str(panel.index.min().date()), str(panel.index.max().date())],
                   "features": FEATS, "horizons": HORIZONS, "test_years": TEST_YEARS,
                   "costs_bp_per_side": COSTS_BP, "quantile": QUANTILE,
                   "bad_days_total": int(sum(g["bad_days"] for g in gap_audit.values())),
                   "worst_gap_pair": max(gap_audit, key=lambda p: gap_audit[p]["bad_days"]),
                   "notes": ["labels close-to-close, signal at close t, pnl day t+1",
                             "funding features 0-filled pre-2020",
                             "long-only variant charges no funding (spot)",
                             "overlapping 1/h tranches, netted turnover"]}

    models = ["logit", "hgb_cls", "hgb_reg", "mom7"]
    pooled = {}
    rng_null = np.random.default_rng(SEED)
    for h in HORIZONS:
        hk = "h%d" % h
        out["results"][hk] = {}
        out["nulls"][hk] = {"shuffled": {}, "random_rank": {}}
        ys, ym = "y_sign_%d" % h, "y_mag_%d" % h
        for year in TEST_YEARS:
            t0, t1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
            tr = panel[(panel.index < t0 - pd.Timedelta(days=h)) & panel[ys].notna()
                       & panel[ym].notna()]
            te = panel[(panel.index >= t0) & (panel.index < t1)]
            if len(tr) == 0 or len(te) == 0:
                continue
            x_tr, x_te = tr[FEATS].to_numpy(), te[FEATS].to_numpy()
            lab = te[ys].notna()
            scores_by_model = {"mom7": te["ret_7"].to_numpy()}
            for name in ["logit", "hgb_cls", "hgb_reg"]:
                try:
                    scores_by_model[name] = fit_score(name, x_tr, tr[ys].to_numpy(),
                                                      tr[ym].to_numpy(), x_te)
                except Exception:
                    out["errors"].append("%s h%d %d: %s" % (name, h, year,
                                                            traceback.format_exc(limit=2)))
            for name, sc in scores_by_model.items():
                node = out["results"][hk].setdefault(name, {"per_year": {}})
                auc = float(roc_auc_score(te.loc[lab, ys], np.asarray(sc)[lab.to_numpy()])) \
                    if lab.sum() > 0 and te.loc[lab, ys].nunique() > 1 else 0.5
                wide = te.assign(score=sc).reset_index().pivot(index="date", columns="pair",
                                                               values="score")
                yr = {"auc": round(auc, 4), "ls": {}, "long_only": {}}
                for cost in COSTS_BP:
                    pnl, tvr = portfolio(wide, rets, fund, h, cost, False, True)
                    yr["ls"]["%gbp" % cost] = metrics(pnl, tvr)
                    if cost == BAR_COST_BP:
                        pooled.setdefault((hk, name, "ls"), []).append(pnl)
                    pnl_l, tvr_l = portfolio(wide, rets, fund, h, cost, True, False)
                    yr["long_only"]["%gbp" % cost] = metrics(pnl_l, tvr_l)
                    if cost == BAR_COST_BP:
                        pooled.setdefault((hk, name, "long_only"), []).append(pnl_l)
                node["per_year"][str(year)] = yr
            try:
                sc_null = fit_score("hgb_cls", x_tr, tr[ys].to_numpy(), tr[ym].to_numpy(),
                                    x_te, rng=rng_null)
                auc_n = float(roc_auc_score(te.loc[lab, ys], sc_null[lab.to_numpy()])) \
                    if lab.sum() > 0 and te.loc[lab, ys].nunique() > 1 else 0.5
                wide_n = te.assign(score=sc_null).reset_index().pivot(index="date",
                                                                      columns="pair", values="score")
                pnl_n, tvr_n = portfolio(wide_n, rets, fund, h, BAR_COST_BP, False, True)
                out["nulls"][hk]["shuffled"][str(year)] = dict(metrics(pnl_n, tvr_n),
                                                               auc=round(auc_n, 4))
                rr = []
                base_mask = wide_n.notna()
                for sd in range(N_RANDOM_SEEDS):
                    rw = pd.DataFrame(np.random.default_rng(sd).uniform(size=base_mask.shape),
                                      index=wide_n.index, columns=wide_n.columns).where(base_mask)
                    pnl_r, tvr_r = portfolio(rw, rets, fund, h, BAR_COST_BP, False, True)
                    rr.append(metrics(pnl_r, tvr_r))
                out["nulls"][hk]["random_rank"][str(year)] = {
                    "net_yr_mean": round(float(np.mean([m["net_yr"] for m in rr])), 4),
                    "net_yr_std": round(float(np.std([m["net_yr"] for m in rr])), 4),
                    "sharpe_mean": round(float(np.mean([m["sharpe"] for m in rr])), 3),
                    "turnover_yr": round(float(np.mean([m["turnover_yr"] for m in rr])), 1)}
            except Exception:
                out["errors"].append("nulls h%d %d: %s" % (h, year,
                                                           traceback.format_exc(limit=2)))

        for name in models:
            node = out["results"][hk].get(name)
            if node is None:
                continue
            for variant in ["ls", "long_only"]:
                series = pooled.get((hk, name, variant), [])
                if not series:
                    continue
                allp = pd.concat(series)
                node.setdefault("pooled_%gbp" % BAR_COST_BP, {})[variant] = \
                    metrics(allp, pd.Series(0.0, index=allp.index))
            yrs = node["per_year"]
            nets = {y: yrs[y]["ls"]["%gbp" % BAR_COST_BP]["net_yr"] for y in yrs}
            pos_years = [y for y, v in nets.items() if v > 0]
            halves = [min(yrs[y]["ls"]["%gbp" % BAR_COST_BP]["h1_net"],
                          yrs[y]["ls"]["%gbp" % BAR_COST_BP]["h2_net"]) for y in yrs]
            pooled_sharpe = node.get("pooled_%gbp" % BAR_COST_BP, {}).get("ls", {}).get("sharpe", 0.0)
            node["survivor"] = {
                "pos_years": len(pos_years),
                "includes_2025_or_2026": any(y in pos_years for y in ["2025", "2026"]),
                "pooled_sharpe": pooled_sharpe,
                "min_half": round(min(halves), 4) if halves else 0.0,
                "survives": (len(pos_years) >= SURVIVOR_MIN_POS_YEARS
                             and any(y in pos_years for y in ["2025", "2026"])
                             and pooled_sharpe >= SURVIVOR_MIN_SHARPE
                             and (min(halves) if halves else 0.0) > HALF_CATASTROPHIC)}

    for h in HORIZONS:
        hk = "h%d" % h
        base = out["results"][hk].get("mom7", {}).get("survivor", {}).get("pooled_sharpe", 0.0)
        for name in ["logit", "hgb_cls", "hgb_reg"]:
            node = out["results"][hk].get(name)
            if node:
                node["survivor"]["beats_mom7_pooled_sharpe"] = \
                    bool(node["survivor"]["pooled_sharpe"] > base)

    out["meta"]["wall_clock_s"] = round(time.time() - t_start, 1)
    with open(STATS_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    print("done", out["meta"]["wall_clock_s"], "s, errors:", len(out["errors"]))
    return 0 if not out["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
