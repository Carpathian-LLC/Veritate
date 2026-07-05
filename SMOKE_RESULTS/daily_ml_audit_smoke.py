# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - adversarial hardening of daily_ml_panel h7 winners (hgb_cls/hgb_reg LS 3.5bp):
#   3 HGB seeds, 14d purge, quantile 0.15/0.25, +1d entry lag, funding on/off.
#   Falsifier: any arm (esp. entry lag) drops the cell below the survivor bar ->
#   original result is fragile, not deployable.
# - wall-clock estimate ~15-25 min on cached daily parquet panel.
# SMOKE_RESULTS/daily_ml_audit_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib.util
import json
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# ------------------------------------------------------------------------------------
# Constants

REPO = "/Users/mirach-00-usc1/Development/Veritate"
BASE_SMOKE = os.path.join(REPO, "SMOKE_RESULTS", "daily_ml_panel_smoke.py")
STATS_PATH = os.path.join(REPO, "SMOKE_RESULTS", "daily_ml_audit_stats.json")

HORIZON = 7
COST_BP = 3.5
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
MODELS = ["hgb_cls", "hgb_reg"]
SEEDS = [0, 1, 2]
BASE_QUANTILE = 0.2
ALT_QUANTILES = [0.15, 0.25]
BASE_PURGE = 7
ALT_PURGE = 14
ENTRY_LAG = 1
MIN_POS_YEARS = 3
MIN_MEAN_SHARPE = 0.8

# ------------------------------------------------------------------------------------
# Functions


def load_base():
    spec = importlib.util.spec_from_file_location("daily_ml_panel_smoke", BASE_SMOKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_panel(base):
    pairs = sorted(f[:-4] for f in os.listdir(base.DATA_DIR) if f.endswith(".csv"))
    btc_daily = base.load_daily("BTCUSDT")
    btc_idx = pd.date_range(btc_daily.index[0], btc_daily.index[-1], freq="D")
    btc_logc = np.log(btc_daily["close"].reindex(btc_idx))
    btc_r1 = btc_logc.diff()
    btc_ret7 = btc_logc - btc_logc.shift(HORIZON)
    frames, rets_w, fund_w = [], {}, {}
    for pair in pairs:
        daily = base.load_daily(pair)
        f = base.compute_features(daily, base.load_funding_daily(pair), btc_r1, btc_ret7)
        rets_w[pair] = f["r1_simple"]
        fund_w[pair] = f["fund_daily"]
        f = f.drop(columns=["r1_simple", "fund_daily", "n_bad_days"])
        f["pair"] = pair
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel["xs_rank_7"] = panel.groupby("date")["ret_7"].rank(pct=True) - 0.5
    panel["xs_rank_30"] = panel.groupby("date")["ret_30"].rank(pct=True) - 0.5
    panel = panel[panel[base.FEATS].notna().all(axis=1)]
    return panel, pd.DataFrame(rets_w), pd.DataFrame(fund_w)


def fit_scores(base, model, seed, x_tr, y_sign, y_mag, x_te):
    if model == "hgb_cls":
        m = HistGradientBoostingClassifier(max_iter=base.HGB_MAX_ITER, random_state=seed)
        m.fit(x_tr, y_sign)
        return m.predict_proba(x_te)[:, 1]
    m = HistGradientBoostingRegressor(max_iter=base.HGB_MAX_ITER, random_state=seed)
    m.fit(x_tr, y_mag)
    return m.predict(x_te)


def portfolio(base, scores, rets, fund, quantile, lag, use_funding):
    dates = scores.index
    cols = rets.columns
    tranche = pd.DataFrame(0.0, index=dates, columns=cols)
    for t in dates:
        s = scores.loc[t].dropna()
        n = len(s)
        if n < base.MIN_NAMES:
            continue
        k = max(1, int(n * quantile))
        tranche.loc[t, s.nlargest(k).index] = 0.5 / (HORIZON * k)
        bot = s.nsmallest(k).index
        tranche.loc[t, bot] = tranche.loc[t, bot] - 0.5 / (HORIZON * k)
    book = tranche.rolling(HORIZON, min_periods=1).sum().shift(lag).fillna(0.0)
    turnover = (book - book.shift(1).fillna(0.0)).abs().sum(axis=1)
    held = book.shift(1).fillna(0.0)
    r = rets.reindex(index=dates, columns=cols).fillna(0.0)
    pnl = (held * r).sum(axis=1) - COST_BP / 1e4 * turnover
    fpaid = pd.Series(0.0, index=dates)
    if use_funding:
        fd = fund.reindex(index=dates, columns=cols).fillna(0.0)
        fpaid = (held * fd).sum(axis=1)
        pnl = pnl - fpaid
    return pnl, turnover, fpaid


def arm_summary(per_year):
    years = sorted(per_year)
    nets = [per_year[y]["net_yr"] for y in years]
    sharpes = [per_year[y]["sharpe"] for y in years]
    pos = [y for y, v in zip(years, nets) if v > 0]
    return {"pos_years": len(pos),
            "includes_2025_or_2026": any(y in pos for y in ["2025", "2026"]),
            "mean_net_yr": round(float(np.mean(nets)), 4),
            "mean_sharpe": round(float(np.mean(sharpes)), 3),
            "passes": (len(pos) >= MIN_POS_YEARS
                       and any(y in pos for y in ["2025", "2026"])
                       and float(np.mean(sharpes)) >= MIN_MEAN_SHARPE)}


def mean_across_seeds(per_seed):
    years = sorted(next(iter(per_seed.values())))
    return {y: {"net_yr": round(float(np.mean([per_seed[s][y]["net_yr"] for s in per_seed])), 4),
                "sharpe": round(float(np.mean([per_seed[s][y]["sharpe"] for s in per_seed])), 3)}
            for y in years}


def main():
    t_start = time.time()
    out = {"smoke": "daily_ml_audit", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "errors": [], "meta": {}, "results": {}, "verdict": {}}
    base = load_base()
    panel, rets, fund = build_panel(base)
    ys, ym = "y_sign_%d" % HORIZON, "y_mag_%d" % HORIZON
    out["meta"] = {"horizon": HORIZON, "cost_bp_per_side": COST_BP, "seeds": SEEDS,
                   "purges_days": [BASE_PURGE, ALT_PURGE], "entry_lag_days": ENTRY_LAG,
                   "quantiles": [BASE_QUANTILE] + ALT_QUANTILES,
                   "panel_rows": len(panel),
                   "arms": ["baseline", "lag1", "q0.15", "q0.25", "purge14", "funding_off"],
                   "notes": ["baseline = signal t close, enter t close, pnl from t+1",
                             "lag1 = signal t close, enter t+1 close, pnl from t+2",
                             "funding charged on LS book both directions",
                             "FTM funding ends 2025-06-19, MKR 2025-09-08 (0 after)"]}

    res = {m: {"baseline": {}, "lag1": {}, "q0.15": {}, "q0.25": {}, "purge14": {},
               "funding_off": {}, "funding_paid_yr": {}} for m in MODELS}
    res["mom7"] = {"baseline": {}, "lag1": {}}

    for year in TEST_YEARS:
        t0, t1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        te = panel[(panel.index >= t0) & (panel.index < t1)]
        if len(te) == 0:
            continue
        x_te = te[base.FEATS].to_numpy()
        wide_mom = te.assign(score=te["ret_7"].to_numpy()).reset_index() \
            .pivot(index="date", columns="pair", values="score")
        for arm, lag in [("baseline", 0), ("lag1", ENTRY_LAG)]:
            pnl, tvr, _ = portfolio(base, wide_mom, rets, fund, BASE_QUANTILE, lag, True)
            res["mom7"][arm][str(year)] = base.metrics(pnl, tvr)
        for purge in [BASE_PURGE, ALT_PURGE]:
            tr = panel[(panel.index < t0 - pd.Timedelta(days=purge))
                       & panel[ys].notna() & panel[ym].notna()]
            x_tr = tr[base.FEATS].to_numpy()
            y_sign, y_mag = tr[ys].to_numpy(), tr[ym].to_numpy()
            seeds = SEEDS if purge == BASE_PURGE else [SEEDS[0]]
            for model in MODELS:
                for seed in seeds:
                    try:
                        sc = fit_scores(base, model, seed, x_tr, y_sign, y_mag, x_te)
                    except Exception:
                        out["errors"].append("%s s%d p%d %d: %s" % (
                            model, seed, purge, year, traceback.format_exc(limit=2)))
                        continue
                    wide = te.assign(score=sc).reset_index() \
                        .pivot(index="date", columns="pair", values="score")
                    if purge == ALT_PURGE:
                        pnl, tvr, _ = portfolio(base, wide, rets, fund,
                                                BASE_QUANTILE, 0, True)
                        res[model]["purge14"][str(year)] = base.metrics(pnl, tvr)
                        continue
                    sk = "seed%d" % seed
                    pnl, tvr, fpaid = portfolio(base, wide, rets, fund,
                                                BASE_QUANTILE, 0, True)
                    res[model]["baseline"].setdefault(sk, {})[str(year)] = \
                        base.metrics(pnl, tvr)
                    pnl_l, tvr_l, _ = portfolio(base, wide, rets, fund,
                                                BASE_QUANTILE, ENTRY_LAG, True)
                    res[model]["lag1"].setdefault(sk, {})[str(year)] = \
                        base.metrics(pnl_l, tvr_l)
                    if seed == SEEDS[0]:
                        res[model]["funding_paid_yr"][str(year)] = \
                            round(float(fpaid.sum()), 4)
                        for q in ALT_QUANTILES:
                            pnl_q, tvr_q, _ = portfolio(base, wide, rets, fund, q, 0, True)
                            res[model]["q%g" % q][str(year)] = base.metrics(pnl_q, tvr_q)
                        pnl_f, tvr_f, _ = portfolio(base, wide, rets, fund,
                                                    BASE_QUANTILE, 0, False)
                        res[model]["funding_off"][str(year)] = base.metrics(pnl_f, tvr_f)
        print(year, "done", round(time.time() - t_start, 1), "s", flush=True)

    for model in MODELS:
        node = res[model]
        summ = {}
        for arm in ["baseline", "lag1"]:
            node[arm + "_seed_mean"] = mean_across_seeds(node[arm])
            summ[arm] = arm_summary(node[arm + "_seed_mean"])
        for arm in ["q0.15", "q0.25", "purge14", "funding_off"]:
            summ[arm] = arm_summary(node[arm])
        node["arms_summary"] = summ
        hardening = ["lag1", "q0.15", "q0.25", "purge14"]
        node["survives_all_hardening"] = all(summ[a]["passes"] for a in hardening + ["baseline"])
    res["mom7"]["arms_summary"] = {a: arm_summary(res["mom7"][a])
                                   for a in ["baseline", "lag1"]}
    out["results"] = res
    out["verdict"] = {m: ("CONFIRMED" if res[m]["survives_all_hardening"] else
                          ("WEAKENED" if res[m]["arms_summary"]["baseline"]["passes"]
                           else "KILLED")) for m in MODELS}
    out["meta"]["wall_clock_s"] = round(time.time() - t_start, 1)
    with open(STATS_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    print("done", out["meta"]["wall_clock_s"], "s, errors:", len(out["errors"]))
    return 0 if not out["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
