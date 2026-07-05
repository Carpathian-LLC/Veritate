# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - order-flow A/B score: marketof_80m (real taker-buy + trade-activity channels 5-6)
#   vs marketof_noflow_80m (constant fallback byte), common-metric eval on identical
#   newest-val-split bars, 6 pairs x 1m/15m/1h. Per cell: benchmark metric set (same
#   V._bench_metrics code path benchmark() uses), return-byte (channel 0) CE, expected
#   |z| vs realized |z| corr, after-fee selective P/L sweep (2/7/20 bp round trip).
#   Falsifier: deltas < 1% dir-acc / < 0.02 mag-corr are single-seed noise unless
#   sign-consistent across >= 4 pairs and >= 2 horizons.
# - wall-clock ~30-60 min, CPU only.
# SMOKE_RESULTS/marketof_ab_score_smoke.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SERVER = os.path.join(ROOT, "extensions", "canonical", "market", "server")
# SERVER first: its veritate.py must win over the root veritate package
for _p in (os.path.join(ROOT, "veritate_mri"), ROOT, SERVER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as md
import series_codec as sc
import veritate as V

# ------------------------------------------------------------------------------------
# Constants

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "LTCUSDT"]
HORIZONS = ["1m", "15m", "1h"]
SOURCE = "crypto_of"
MODELS = {"flow": "marketof_80m", "noflow": "marketof_noflow_80m"}
FLOW_COLS = ["trades", "taker_buy"]
EXPECT_STRIDE = 7
VAL_RATIO = 0.1
N_EVAL_BARS = 15000
MIN_SCORED = 30
FEES_RT = {"2bp": 0.0002, "7bp": 0.0007, "20bp": 0.0020}
GATES = [0.0, 0.75, 0.90, 0.95]
PROB_FLOOR = 1e-12
STATS_PATH = os.path.join(HERE, "marketof_ab_score_stats.json")

# ------------------------------------------------------------------------------------
# Functions

def _valid_mask(o, h, l, c, v, adj):
    """Validity mask of sc.compute_features (same math) to locate the corpus split."""
    ret = adj[1:] / adj[:-1] - 1.0
    rng = (h[1:] - l[1:]) / c[:-1]
    vv = v[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        rz = ret / sc._trailing(ret, sc.FEAT_WINDOW, np.std)
        gr = rng / sc._trailing(rng, sc.FEAT_WINDOW, np.mean)
        vr = vv / sc._trailing(vv, sc.FEAT_WINDOW, np.mean)
        rv = sc._trailing(ret, sc.RV_WINDOW, np.std)
        yr = rv / sc._trailing(rv, sc.RV_REF_WINDOW, np.mean)
    return np.isfinite(rz) & np.isfinite(gr) & np.isfinite(vr) & np.isfinite(yr)


def _load_pair(sym):
    """Full 1m frame + first val bar (raw ts ns) under the builder's 90/10 split."""
    df = md.load_1m(md.path_for(sym, SOURCE), cols=md.OHLCV_EXT)
    g = lambda k: df[k].to_numpy(np.float64)
    o, h, l, c, v = g("open"), g("high"), g("low"), g("close"), g("volume")
    keep = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(v)
    if not keep.all():
        df = df[keep]
        o, h, l, c, v = o[keep], h[keep], l[keep], c[keep], v[keep]
    pos = np.flatnonzero(_valid_mask(o, h, l, c, v, c))
    r = pos[int(len(pos) * (1.0 - VAL_RATIO))] + 1
    return df, int(md.index_ns(df.index)[r]), len(pos), str(df.index[r])


def _score_cell(model, seq_len, df, stride):
    """Windowed CPU walk (same window scheme as V._score_bytes): per return byte
    collect full-vocab CE, restricted bucket dist stats, realized next-bar return."""
    import torch
    arr, nb = V._encode_df(df, stride)
    c = df["close"].to_numpy(np.float64)
    off = len(c) - nb
    half = max(1, seq_len // 2)
    t = len(arr)
    ret_idx = torch.tensor(V.RET_BYTES, dtype=torch.long)
    ce_full = {}
    rprob = {}
    with torch.no_grad():
        w = 0
        while w < t - 1:
            chunk = arr[w:w + seq_len]
            ids = torch.from_numpy(chunk.astype(np.int64))[None]
            logits = model(ids)[0][0].float()
            start = 0 if w == 0 else half
            js = np.arange(start, min(len(chunk), t - 1 - w))
            js = js[(w + js + 1) % stride == 0]
            if js.size:
                lp = torch.log_softmax(logits[torch.from_numpy(js)], dim=-1)
                tgt = torch.from_numpy(arr[w + js + 1].astype(np.int64))
                ce = (-lp.gather(1, tgt[:, None])[:, 0]).numpy()
                rp = torch.exp(lp[:, ret_idx]).numpy()
                for m, j in enumerate(js):
                    ce_full[w + j + 1] = float(ce[m])
                    rprob[w + j + 1] = rp[m]
            if len(chunk) < seq_len:
                break
            w += half

    cols = {k: [] for k in ("ce_full", "ce_ret", "pred", "actual", "conf", "psign", "eaz", "ret")}
    az = np.abs(V.Z_CENTERS)
    for k in range(1, nb):
        gp = k * stride
        pr = rprob.get(gp)
        if pr is None:
            continue
        ci = k + off
        if ci >= len(c):
            break
        ab = int(V.BYTE2RET[arr[gp]])
        s = pr.sum()
        if ab < 0 or s <= 0:
            continue
        prn = pr / s
        up = float(prn[sc.RET_CENTER + 1:].sum())
        dn = float(prn[:sc.RET_CENTER].sum())
        pu = up / (up + dn) if (up + dn) > 1e-9 else 0.5
        cols["ce_full"].append(ce_full[gp])
        cols["ce_ret"].append(-np.log(max(prn[ab], PROB_FLOOR)))
        cols["pred"].append(int(np.argmax(prn)))
        cols["actual"].append(ab)
        cols["conf"].append(max(pu, 1.0 - pu))
        cols["psign"].append(1 if up >= dn else -1)
        cols["eaz"].append(float((prn * az).sum()))
        cols["ret"].append(float(np.log(c[ci] / c[ci - 1])) if c[ci - 1] > 0 else 0.0)
    if len(cols["actual"]) < MIN_SCORED:
        return None
    return {k: np.asarray(v) for k, v in cols.items()}


def _pl_sweep(psign, ret, conf):
    """Avg net bps/trade after round-trip fee, confidence-quantile gated."""
    nz = ret != 0
    r = (psign * ret)[nz]
    cf = conf[nz]
    out = {}
    for lbl, fee in FEES_RT.items():
        row = {}
        for q in GATES:
            sel = r[cf >= float(np.quantile(cf, q))] if q > 0 else r
            row[str(q)] = {
                "n": int(sel.size),
                "win": round(float((sel > 0).mean()), 4) if sel.size else None,
                "net_bps": round((float(sel.mean()) - fee) * 1e4, 2) if sel.size else None,
            }
        out[lbl] = row
    return out


def _cell_metrics(a):
    m = V._bench_metrics(a["pred"], a["actual"], a["conf"], a["psign"])
    ao = (a["actual"] - sc.RET_CENTER).astype(float)
    mag = None
    if a["eaz"].std() > 0 and np.abs(ao).std() > 0:
        mag = round(float(np.corrcoef(a["eaz"], np.abs(ao))[0, 1]), 4)
    cal = m["calibration"]
    return {
        "n": int(len(a["actual"])),
        "ce_ret_full": round(float(a["ce_full"].mean()), 4),
        "ce_ret_restricted": round(float(a["ce_ret"].mean()), 4),
        "dir_acc": m["directional_accuracy"],
        "hc_prec": m["high_conf_precision"],
        "decisive_rate": m["decisive_rate"],
        "mag_abs_corr": mag,
        "mag_corr_signed": m["magnitude_corr"],
        "calib_gain": round(cal[-1]["precision"] - cal[0]["precision"], 4) if len(cal) > 1 else None,
        "calibration": cal,
        "pl": _pl_sweep(a["psign"], a["ret"], a["conf"]),
    }


def _summary(cells):
    rows = {}
    agg = {"d_dir_acc": [], "d_mag_abs_corr": [], "d_hc_prec": [], "d_ce_ret_full": []}
    for sym in PAIRS:
        for hz in HORIZONS:
            f = cells.get(f"{sym}|{hz}|flow")
            n = cells.get(f"{sym}|{hz}|noflow")
            if not f or not n:
                continue
            sub = lambda ka: round(f[ka] - n[ka], 4) if f[ka] is not None and n[ka] is not None else None
            row = {
                "d_dir_acc": sub("dir_acc"),
                "d_mag_abs_corr": sub("mag_abs_corr"),
                "d_hc_prec": sub("hc_prec"),
                "d_ce_ret_full": round(n["ce_ret_full"] - f["ce_ret_full"], 4),
            }
            rows[f"{sym}|{hz}"] = row
            for k in agg:
                if row[k] is not None:
                    agg[k].append(row[k])
    cons = {k: {"n": len(v), "pos": int(sum(1 for x in v if x > 0)),
                "neg": int(sum(1 for x in v if x < 0)),
                "mean": round(float(np.mean(v)), 4) if v else None} for k, v in agg.items()}
    return {"cells": rows, "consistency": cons}


def _write(stats):
    with open(STATS_PATH, "w") as fh:
        json.dump(stats, fh, indent=1)


def main():
    t0 = time.time()
    stats = {"errors": [], "meta": {"models": MODELS, "pairs": PAIRS, "horizons": HORIZONS,
                                    "n_eval_bars": N_EVAL_BARS, "stride": EXPECT_STRIDE,
                                    "val_ratio": VAL_RATIO, "fees_rt": FEES_RT, "gates": GATES,
                                    "steps": {}},
             "split": {}, "cells": {}, "summary": {}}
    models = {}
    for arm, name in MODELS.items():
        mdl, seq, step, stride = V.load_model(name)
        if mdl is None or int(stride) != EXPECT_STRIDE:
            stats["errors"].append(f"{name}: load failed or stride {stride} != {EXPECT_STRIDE}")
            _write(stats)
            return 1
        models[arm] = (mdl, seq)
        stats["meta"]["steps"][arm] = step
    for sym in PAIRS:
        try:
            df, cut_ns, nvalid, cut_str = _load_pair(sym)
        except Exception as e:
            stats["errors"].append(f"{sym}: load: {e}")
            continue
        stats["split"][sym] = {"first_val_bar": cut_str, "n_valid_1m": nvalid}
        for hz in HORIZONS:
            dfh = df if hz == "1m" else md.resample(df, hz)
            win = dfh[md.index_ns(dfh.index) >= cut_ns]
            if len(win) > N_EVAL_BARS:
                win = win.iloc[-N_EVAL_BARS:]
            for arm in MODELS:
                key = f"{sym}|{hz}|{arm}"
                tc = time.time()
                try:
                    a = _score_cell(models[arm][0], models[arm][1],
                                    win.drop(columns=FLOW_COLS) if arm == "noflow" else win,
                                    EXPECT_STRIDE)
                except Exception as e:
                    stats["errors"].append(f"{key}: score: {e}")
                    continue
                if a is None:
                    stats["errors"].append(f"{key}: fewer than {MIN_SCORED} scored bars")
                    continue
                stats["cells"][key] = _cell_metrics(a)
                print(f"{key} n={stats['cells'][key]['n']} "
                      f"ce={stats['cells'][key]['ce_ret_full']} {time.time() - tc:.0f}s", flush=True)
        del df
    stats["summary"] = _summary(stats["cells"])
    stats["meta"]["wall_s"] = round(time.time() - t0, 1)
    _write(stats)
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
