# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Per-instrument honest replay for the dashboard. Loads a saved MarketModel and walks
#   one instrument's recent bars, producing the series the UI plots: volatility
#   predicted-vs-actual, directional reliability (calibration), and a cost-aware equity
#   curve on non-overlapping trades. Same leak-free features.compute as training.
# - This is display/diagnostics, not the certification harness (that is evaluate.py with
#   the purged split across the pooled corpus).
# veritate_mri/market/backtest.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as md
import dataset as ds
import features as ff

MINUTES_PER_YEAR = 525600.0

# ------------------------------------------------------------------------------------
# Functions

def _downsample(a, n):
    a = np.asarray(a)
    if len(a) <= n:
        return a
    return a[:: int(np.ceil(len(a) / n))]


def hindcast(mm, df, max_points=320):
    """Walk the window: at each bar record the model's predicted move-size band (where it
    thought price could go over the next H bars) and its directional guess vs reality.
    Powers the 'what the AI guessed across the whole view' overlay."""
    h = mm.horizon
    f = ff.compute(df)
    X = f.to_numpy(dtype=np.float32)
    fr, fv = ds.make_labels(df, h)
    m = np.isfinite(X).all(axis=1) & np.isfinite(fr)
    idx = np.flatnonzero(m)
    if len(idx) < 30:
        return None
    Xv = X[idx]
    vol_pred = np.exp(mm.vol.predict(Xv)).astype(np.float64)
    p_up = mm.iso.predict(mm.clf.predict_proba(Xv)[:, 1]).astype(np.float64)
    price = df["close"].to_numpy(dtype=np.float64)[idx]
    y_ret = fr[idx].astype(np.float64)
    tsec = (md.index_ns(f.index)[idx] // 1_000_000_000).astype(np.int64)

    band = mm.cone_k * vol_pred                       # ~80% expected |move| over next H bars
    lean = np.sign(p_up - 0.5)
    nz = np.abs(y_ret) > (0.33 * vol_pred)
    correct = np.where(~nz, 0, np.where(lean * np.sign(y_ret) > 0, 1, -1)).astype(np.int8)
    coverage = float((np.abs(y_ret) <= band).mean())
    judged = correct != 0
    hit_rate = float((correct[judged] == 1).mean()) if judged.any() else float("nan")

    def ds_(a):
        return _downsample(a, max_points)
    return {
        "horizon": h, "base": mm.base, "n": int(len(idx)),
        "coverage": coverage, "hit_rate": hit_rate, "cone_cov": mm.cone_cov,
        "t": ds_(tsec).tolist(),
        "price": ds_(price).round(6).tolist(),
        "band": ds_(band).round(6).tolist(),            # fraction; hi=price*exp(band), lo=price*exp(-band)
        "p_up": ds_(p_up).round(4).tolist(),
        "mark": ds_(correct).tolist(),                  # 1 right, -1 wrong, 0 flat
    }


def replay(mm, df, cost_bps=10.0, margin=0.02, max_points=360):
    """Walk df with model mm. Returns JSON-able dict of summary + plot series."""
    h = mm.horizon
    base_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(mm.base, 1)
    f = ff.compute(df)
    X = f.to_numpy(dtype=np.float32)
    fr, fv = ds.make_labels(df, h)
    m = np.isfinite(X).all(axis=1) & np.isfinite(fr) & np.isfinite(fv)
    idx = np.flatnonzero(m)
    if len(idx) < 60:
        return None
    Xv = X[idx]
    vol_pred = np.exp(mm.vol.predict(Xv)).astype(np.float64)
    p_up = mm.iso.predict(mm.clf.predict_proba(Xv)[:, 1]).astype(np.float64)
    y_vol = fv[idx].astype(np.float64)
    y_ret = fr[idx].astype(np.float64)
    tsec = (md.index_ns(f.index)[idx] // 1_000_000_000).astype(np.int64)
    price = df["close"].to_numpy(dtype=np.float64)[idx]

    # volatility skill
    vbar = y_vol.mean()
    vol_r2 = float(1.0 - np.sum((y_vol - vol_pred) ** 2) / max(np.sum((y_vol - vbar) ** 2), 1e-18))
    vol_corr = float(np.corrcoef(vol_pred, y_vol)[0, 1])

    # direction skill on non-flat bars (flat band = 0.33 * forecast forward vol, matches training)
    band = 0.33 * vol_pred
    nz = np.abs(y_ret) > band
    up = (y_ret[nz] > 0).astype(int)
    pnz = p_up[nz]
    if nz.sum() > 30:
        base_rate = float(up.mean())
        dir_acc = float(((pnz > 0.5).astype(int) == up).mean())
        try:
            from sklearn.metrics import roc_auc_score
            dir_auc = float(roc_auc_score(up, pnz)) if up.min() != up.max() else float("nan")
        except Exception:
            dir_auc = float("nan")
    else:
        base_rate = dir_acc = dir_auc = float("nan")

    # reliability bins
    rel = []
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        sel = (pnz >= edges[i]) & (pnz < (edges[i + 1] if i < 9 else edges[i + 1] + 1e-9))
        if sel.sum() >= 15:
            rel.append({"pred": float(pnz[sel].mean()), "obs": float(up[sel].mean()), "n": int(sel.sum())})

    # cost-aware equity on non-overlapping trades
    step = max(1, h)
    side = np.where(p_up > 0.5 + margin, 1.0, np.where(p_up < 0.5 - margin, -1.0, 0.0))[::step]
    rstep = y_ret[::step]
    tstep = tsec[::step]
    cost = (cost_bps * 1e-4) * 2.0
    net = side * rstep - cost * (side != 0)
    equity = np.cumsum(net)
    traded = side != 0
    n_tr = int(traded.sum())
    tpy = MINUTES_PER_YEAR / (h * base_min)
    if n_tr > 5 and net[traded].std() > 0:
        sharpe_net = float(net[traded].mean() / net[traded].std() * np.sqrt(tpy))
    else:
        sharpe_net = float("nan")

    return {
        "summary": {
            "n": int(len(idx)), "horizon": h, "base": mm.base,
            "vol_r2": vol_r2, "vol_corr": vol_corr,
            "dir_base_rate": base_rate, "dir_acc": dir_acc, "dir_auc": dir_auc,
            "n_trades": n_tr, "equity_final_bps": float(equity[-1] * 1e4) if len(equity) else 0.0,
            "sharpe_net": sharpe_net, "cost_bps_roundtrip": cost * 1e4,
        },
        "vol_series": {
            "t": _downsample(tsec, max_points).tolist(),
            "pred": _downsample(vol_pred, max_points).round(8).tolist(),
            "actual": _downsample(y_vol, max_points).round(8).tolist(),
        },
        "reliability": rel,
        "equity": {
            "t": _downsample(tstep, max_points).tolist(),
            "v": _downsample(equity * 1e4, max_points).round(3).tolist(),       # in bps
        },
        "price_series": {
            "t": _downsample(tsec, max_points).tolist(),
            "p": _downsample(price, max_points).round(6).tolist(),
        },
    }
