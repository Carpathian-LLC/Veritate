# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The honesty harness. Trains the platform's two models on a purged split and scores
#   them against the baselines they must beat, out-of-sample:
#     VOLATILITY (regression): HistGBDT vs persistence / EWMA / HAR-OLS  -> OOS R^2, MAE
#     DIRECTION  (classifier): HistGBDT vs base-rate / always-majority   -> acc, AUC,
#                              Brier, log-loss, reliability, + cost-aware backtest Sharpe
# - Prints a plain-language verdict. Expectation from the research+data: vol R^2 is
#   solidly positive (forecastable); direction barely clears the base rate and dies
#   after costs. We report it honestly either way.
# - run: python veritate_mri/market/evaluate.py [--horizon 15] [--base 1m] [--pairs 30]
# veritate_mri/market/evaluate.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as md
import dataset as ds
import features as ff
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# ------------------------------------------------------------------------------------
# Constants

LIQUID = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "DOTUSDT", "ATOMUSDT", "NEARUSDT",
    "MATICUSDT", "UNIUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT",
    "OPUSDT", "INJUSDT", "SUIUSDT", "AAVEUSDT", "ALGOUSDT", "XLMUSDT", "VETUSDT",
    "ICPUSDT", "HBARUSDT", "RUNEUSDT", "SANDUSDT", "FTMUSDT", "GALAUSDT",
]
MINUTES_PER_YEAR = 525600.0

# ------------------------------------------------------------------------------------
# Metric helpers

def oos_r2(y, yhat, ybar):
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - ybar) ** 2)
    return 1.0 - sse / sst if sst > 0 else float("nan")


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() >= 30:
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows

# ------------------------------------------------------------------------------------
# Volatility model + baselines

def eval_volatility(d, tr, va):
    f = d["features"]
    i10, i30, i120 = f.index("rv10"), f.index("rv30"), f.index("rv120")
    h = d["horizon"]
    sh = np.sqrt(h)
    X, y = d["X"], d["y_vol"].astype(np.float64)
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    ybar = ytr.mean()

    # GBDT on log-vol (positive, heavy-tailed) -> exp back
    reg = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=300, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0,
    )
    reg.fit(Xtr, np.log(np.clip(ytr, 1e-9, None)))
    gb = np.exp(reg.predict(Xva))

    persist = Xva[:, i30] * sh                                  # last per-bar vol scaled
    ewma = Xva[:, i10] * sh
    # HAR-OLS on (rv10,rv30,rv120) scaled to the horizon
    A = np.column_stack([np.ones(tr.sum()), Xtr[:, i10] * sh, Xtr[:, i30] * sh, Xtr[:, i120] * sh])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    Av = np.column_stack([np.ones(va.sum()), Xva[:, i10] * sh, Xva[:, i30] * sh, Xva[:, i120] * sh])
    har = Av @ coef

    res = {}
    for name, pred in [("GBDT", gb), ("HAR-OLS", har), ("persistence", persist), ("EWMA", ewma)]:
        res[name] = {
            "r2": oos_r2(yva, pred, ybar),
            "mae": float(np.mean(np.abs(yva - pred))),
            "corr": float(np.corrcoef(pred, yva)[0, 1]),
        }
    return res, reg

# ------------------------------------------------------------------------------------
# Direction model + baselines + cost-aware backtest

def eval_direction(d, tr, va, cost_bps=10.0, margin=0.02):
    X, ydir, yret = d["X"], d["y_dir"], d["y_ret"].astype(np.float64)
    h = d["horizon"]

    nz_tr = tr & (ydir != 0)
    nz_va = va & (ydir != 0)
    ytr = (ydir[nz_tr] == 1).astype(int)
    yva = (ydir[nz_va] == 1).astype(int)

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=0,
    )
    clf.fit(X[nz_tr], ytr)
    p = clf.predict_proba(X[nz_va])[:, 1]

    base_rate = float(yva.mean())
    acc = float(((p > 0.5).astype(int) == yva).mean())
    out = {
        "n_val": int(nz_va.sum()),
        "base_rate_up": base_rate,
        "always_majority_acc": max(base_rate, 1 - base_rate),
        "model_acc": acc,
        "edge_vs_majority": acc - max(base_rate, 1 - base_rate),
        "auc": float(roc_auc_score(yva, p)),
        "brier": float(brier_score_loss(yva, p)),
        "logloss": float(log_loss(yva, p, labels=[0, 1])),
        "reliability": reliability(p, yva),
        "p_range": [float(p.min()), float(p.max())],
    }

    # cost-aware backtest on NON-OVERLAPPING val trades (every h bars), all rows
    pf = clf.predict_proba(X[va])[:, 1]
    rva = yret[va]
    order = np.argsort(d["t"][va])
    pf, rva = pf[order], rva[order]
    step = max(1, h)
    pf, rva = pf[::step], rva[::step]
    side = np.where(pf > 0.5 + margin, 1.0, np.where(pf < 0.5 - margin, -1.0, 0.0))
    cost = (cost_bps * 1e-4) * 2.0                              # round-trip
    gross = side * rva
    net = gross - cost * (side != 0)
    traded = side != 0
    n_tr = int(traded.sum())
    trades_per_year = MINUTES_PER_YEAR / h
    if n_tr > 5 and net[traded].std() > 0:
        sharpe_net = float(net[traded].mean() / net[traded].std() * np.sqrt(trades_per_year))
        sharpe_gross = float(gross[traded].mean() / gross[traded].std() * np.sqrt(trades_per_year))
    else:
        sharpe_net = sharpe_gross = float("nan")
    out["backtest"] = {
        "n_trades": n_tr, "frac_traded": float(traded.mean()),
        "gross_mean_bps": float(gross[traded].mean() * 1e4) if n_tr else float("nan"),
        "net_mean_bps": float(net[traded].mean() * 1e4) if n_tr else float("nan"),
        "hit_rate": float((gross[traded] > 0).mean()) if n_tr else float("nan"),
        "sharpe_gross": sharpe_gross, "sharpe_net": sharpe_net,
        "cost_bps_roundtrip": cost * 1e4,
    }
    return out, clf

# ------------------------------------------------------------------------------------
# Main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--base", default="1m")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--max-bars", type=int, default=525600)
    ap.add_argument("--train-stride", type=int, default=3)
    args = ap.parse_args(argv)

    syms = LIQUID[: args.pairs]
    t0 = time.time()
    print(f"building dataset: {len(syms)} pairs, base={args.base}, horizon={args.horizon}, max_bars={args.max_bars}")
    d = ds.build(syms, horizon=args.horizon, base=args.base, max_bars=args.max_bars)
    tr, va = ds.purged_split(d, val_frac=0.2)
    if args.train_stride > 1:                                   # thin overlapping train rows
        idx = np.flatnonzero(tr)[:: args.train_stride]
        tr = np.zeros_like(tr); tr[idx] = True
    print(f"  rows={len(d['t']):,}  train={tr.sum():,}  val={va.sum():,}  pairs_used={len(d['symbols'])}  ({time.time()-t0:.0f}s)")

    vol, _ = eval_volatility(d, tr, va)
    dirr, _ = eval_direction(d, tr, va)

    print("\n" + "=" * 78)
    print(f"VOLATILITY FORECAST (forward {args.horizon}-bar realized vol)  -- higher OOS R2 = better")
    print("=" * 78)
    print(f"{'model':>14} {'OOS_R2':>9} {'MAE':>11} {'corr':>8}")
    for k in ["GBDT", "HAR-OLS", "EWMA", "persistence"]:
        m = vol[k]
        print(f"{k:>14} {m['r2']:>9.4f} {m['mae']:>11.6f} {m['corr']:>8.4f}")
    gb_lift = vol["GBDT"]["r2"] - max(vol["HAR-OLS"]["r2"], vol["EWMA"]["r2"], vol["persistence"]["r2"])
    print(f"  -> GBDT beats best baseline by {gb_lift:+.4f} R2")

    print("\n" + "=" * 78)
    print(f"DIRECTION (up vs down over next {args.horizon} bars, flat band excluded)")
    print("=" * 78)
    print(f"  base rate (P up)      : {dirr['base_rate_up']:.4f}")
    print(f"  always-majority acc   : {dirr['always_majority_acc']:.4f}")
    print(f"  MODEL accuracy        : {dirr['model_acc']:.4f}   (edge vs majority {dirr['edge_vs_majority']:+.4f})")
    print(f"  AUC / Brier / logloss : {dirr['auc']:.4f} / {dirr['brier']:.4f} / {dirr['logloss']:.4f}")
    print(f"  prob range            : {dirr['p_range'][0]:.3f}..{dirr['p_range'][1]:.3f}")
    print(f"  reliability (pred->obs): " + "  ".join(f"{a:.2f}->{b:.2f}(n{c})" for a, b, c in dirr["reliability"]))
    bt = dirr["backtest"]
    print(f"\n  cost-aware backtest (non-overlapping, {bt['cost_bps_roundtrip']:.0f}bps round-trip):")
    print(f"    trades={bt['n_trades']:,}  traded_frac={bt['frac_traded']:.2f}  hit={bt['hit_rate']:.4f}")
    print(f"    gross={bt['gross_mean_bps']:+.2f}bps/trade  net={bt['net_mean_bps']:+.2f}bps/trade")
    print(f"    Sharpe gross={bt['sharpe_gross']:+.2f}  net={bt['sharpe_net']:+.2f}")
    print("=" * 78)
    print(f"total {time.time()-t0:.0f}s")
    return {"volatility": vol, "direction": dirr}


if __name__ == "__main__":
    main()
