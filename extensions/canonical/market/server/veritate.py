# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Drives the market page with the VERITATE byte-level model (the on-mission engine).
#   The byte model is the only engine the page serves.
# - The byte model forecasts the next bar's return BUCKET (a z-scored, scale-free class).
#   hindcast() walks an instrument, turns each predicted bucket distribution into a
#   price-space guess: directional lean + expected move (E|z| x trailing sigma), and
#   scores it right/wrong vs the actual next bar. Output matches backtest.hindcast so the
#   same overlay renders it.
# - Inference runs on CPU (no MPS contention with a live training run). Same series_codec
#   the corpus was built with (one format contract).
# extensions/canonical/market/server/veritate.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
for _p in (HERE, _ROOT, os.path.join(_ROOT, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as md
import series_codec as sc

# ------------------------------------------------------------------------------------
# Constants

RET_BYTES = [ord(c) for c in sc.ALPHABET[:sc.RET_BINS]]
Z_CENTERS = (np.arange(sc.RET_BINS) - sc.RET_CENTER) * (sc.RET_Z_CLIP / sc.RET_CENTER)
BENCH_THREADS = 4
ROUND_TRIP_FEE = 0.0020                            # 20 bps per trade (10 bps/side); matches the page FEE
PREMIUM_WINDOW = 96                                # trailing bars defining the live vol premium (matches policy default)
BYTE2RET = np.full(256, -1, dtype=np.int64)        # ascii byte -> return bucket index
for _i, _b in enumerate(RET_BYTES):
    BYTE2RET[_b] = _i

# ------------------------------------------------------------------------------------
# Byte-model scoring (CPU)

def _score_bytes(model, seq_len, arr):
    """{global_pos: probs over RET buckets} for the byte predicted at each position;
    the second half of each window has full context (windowed forward passes)."""
    import torch
    half = max(1, seq_len // 2)
    t = len(arr)
    ret_idx = torch.tensor(RET_BYTES, dtype=torch.long)
    out = {}
    with torch.no_grad():
        w = 0
        while w < t - 1:
            chunk = arr[w:w + seq_len]
            ids = torch.from_numpy(chunk.astype(np.int64))[None]
            probs = torch.softmax(model(ids)[0][0], dim=-1)[:, ret_idx]
            start = 0 if w == 0 else half
            for i in range(start, len(chunk)):
                gp = w + i + 1
                if gp >= t:
                    break
                out[gp] = probs[i].numpy()
            if len(chunk) < seq_len:
                break
            w += half
    return out


def list_models():
    """Byte-level Veritate models that have at least one checkpoint."""
    from readers import checkpoints, models as rmodels
    return [n for n in rmodels.list_models() if checkpoints.list_steps(n)]


def _encode_df(df, stride):
    """OHLCV(+taker-buy, +trade-count, +time) df -> (ascii byte ids, n_valid_bars) at the
    model's codec `stride` (older models emit fewer channels, so they need no retrain)."""
    o = df["open"].to_numpy(np.float64); h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64); c = df["close"].to_numpy(np.float64)
    v = df["volume"].to_numpy(np.float64); ts = md.index_ns(df.index)
    tb = df["taker_buy"].to_numpy(np.float64) if "taker_buy" in df.columns else None
    ntr = df["trades"].to_numpy(np.float64) if "trades" in df.columns else None
    fund = df["funding"].to_numpy(np.float64) if "funding" in df.columns else None
    fng = df["fng"].to_numpy(np.float64) if "fng" in df.columns else None
    feats = sc.compute_features(o, h, l, c, v, c.copy(), ts, tb, ntr, fund, fng)
    arr = np.frombuffer(sc.encode_sequence(*feats, stride=stride).encode("ascii"), dtype=np.uint8)
    return arr, len(feats[0])


def predict_next(model, seq_len, df, stride=sc.BAR_STRIDE):
    """Live next-bar forecast from the byte model: P(up), expected move, lean."""
    import torch
    c = df["close"].to_numpy(np.float64)
    arr, nb = _encode_df(df, stride)
    if nb < 30:
        return None
    chunk = arr[-seq_len:]
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(chunk.astype(np.int64))[None])[0][0], dim=-1)
        pr = probs[-1].numpy()[RET_BYTES]
    s = pr.sum()
    pr = pr / s if s > 0 else pr
    up = float(pr[sc.RET_CENTER + 1:].sum()); dn = float(pr[:sc.RET_CENTER].sum())
    p_up = up / (up + dn) if (up + dn) > 1e-9 else 0.5
    ez = float((pr * np.abs(Z_CENTERS)).sum())
    sig = _roll_std(np.diff(np.log(np.clip(c, 1e-12, None)), prepend=np.log(max(c[0], 1e-12))), sc.FEAT_WINDOW)[-1]
    return {"p_up": p_up, "expected_move": ez * float(sig), "confidence": abs(p_up - 0.5) * 2.0,
            "vol": float(sig), "lean": 1 if up >= dn else -1}


def trailing_premium(df, window=PREMIUM_WINDOW):
    """Trailing mean absolute log-return over `window` closed bars: the prevailing
    vol premium decide() gates a vol-harvest entry against."""
    c = df["close"].to_numpy(np.float64)
    a = np.abs(np.diff(np.log(np.clip(c, 1e-12, None))))[-window:]
    return float(a.mean()) if len(a) else 0.0


def load_model(name):
    """Latest checkpoint of byte model `name` on CPU. Returns (model, seq_len, step, bar_stride).
    bar_stride is the codec stride the model was trained against; unstamped models default to
    the current codec stride (sc.BAR_STRIDE), the stride they were trained with."""
    from readers import checkpoints
    import torch
    from veritate_core.load import load_from_state_dict
    steps = checkpoints.list_steps(name)
    if not steps:
        return None, None, None, None
    step = max(steps)
    torch.set_num_threads(BENCH_THREADS)
    s = torch.load(checkpoints.path_for(name, step), map_location="cpu", weights_only=True)
    cfg = {**(s.get("args") or {}), **(s.get("config") or {})}
    ta = cfg.get("training_args") or {}
    stride = int(cfg.get("bar_stride", ta.get("bar_stride", sc.BAR_STRIDE)))
    model = load_from_state_dict(s["model"], cfg)
    model.eval()
    return model, int(cfg.get("seq", 256)), step, stride

# ------------------------------------------------------------------------------------
# Hindcast in price space

def _downsample(a, n):
    a = np.asarray(a)
    return a if len(a) <= n else a[:: int(np.ceil(len(a) / n))]


def hindcast(model, seq_len, df, base="1m", stride=sc.BAR_STRIDE, max_points=320):
    """Walk df with the byte model; per bar record its predicted next-bar move band +
    direction guess vs reality. Returns the same shape as backtest.hindcast."""
    c = df["close"].to_numpy(np.float64)
    arr, nb = _encode_df(df, stride)
    if nb < 60:
        return None
    off = len(c) - nb                                  # features trim the warmup
    probpos = _score_bytes(model, seq_len, arr)

    lr = np.diff(np.log(np.clip(c, 1e-12, None)), prepend=np.log(max(c[0], 1e-12)))
    sigma = _roll_std(lr, sc.FEAT_WINDOW)              # trailing return std, per raw bar

    t = []; price = []; band = []; p_up = []; mark = []
    for k in range(1, nb):
        pr = probpos.get(k * stride)
        if pr is None:
            continue
        ci = k + off                                   # raw index of this bar
        if ci + 1 >= len(c):
            break
        s = pr.sum()
        if s <= 0:
            continue
        pr = pr / s
        up = float(pr[sc.RET_CENTER + 1:].sum())
        dn = float(pr[:sc.RET_CENTER].sum())
        pu = up / (up + dn) if (up + dn) > 1e-9 else 0.5
        ez = float((pr * np.abs(Z_CENTERS)).sum())     # expected |z|
        mv = ez * float(sigma[ci])                     # expected move (log-return) this bar
        lean = 1 if up >= dn else -1
        real = np.log(c[ci + 1] / c[ci]) if c[ci] > 0 else 0.0
        t.append(int(df.index[ci].value // 1_000_000_000))
        price.append(float(c[ci]))
        band.append(float(mv))
        p_up.append(round(pu, 4))
        mark.append(0 if real == 0 else (1 if lean * np.sign(real) > 0 else -1))

    if len(price) < 30:
        return None
    price = np.array(price); band = np.array(band); mk = np.array(mark)
    judged = mk != 0
    hit = float((mk[judged] == 1).mean()) if judged.any() else None
    return {
        "engine": "veritate", "horizon": 1, "base": base, "n": int(len(price)),
        "hit_rate": hit, "coverage": None, "cone_cov": 0.0,
        "t": _downsample(np.array(t), max_points).tolist(),
        "price": _downsample(price, max_points).round(6).tolist(),
        "band": _downsample(band, max_points).round(6).tolist(),
        "p_up": _downsample(np.array(p_up), max_points).tolist(),
        "mark": _downsample(mk, max_points).tolist(),
    }


def signal_series(model, seq_len, df, base="1m", stride=sc.BAR_STRIDE):
    """Per-bar policy signal over df: aligned arrays the trading policy consumes
    (policy.py). ret_next[i] is the realized log return from bar i to i+1, i.e. the
    outcome of acting at bar i. Same scoring walk as hindcast, no downsampling."""
    c = df["close"].to_numpy(np.float64)
    arr, nb = _encode_df(df, stride)
    if nb < 60:
        return None
    off = len(c) - nb
    probpos = _score_bytes(model, seq_len, arr)
    lr = np.diff(np.log(np.clip(c, 1e-12, None)), prepend=np.log(max(c[0], 1e-12)))
    sigma = _roll_std(lr, sc.FEAT_WINDOW)
    t = []; price = []; p_up = []; conf = []; exp_move = []; vol = []; ret_next = []
    for k in range(1, nb):
        pr = probpos.get(k * stride)
        if pr is None:
            continue
        ci = k + off
        if ci + 1 >= len(c) or ci < 1 or not np.isfinite(sigma[ci]):
            continue
        s = pr.sum()
        if s <= 0:
            continue
        pr = pr / s
        up = float(pr[sc.RET_CENTER + 1:].sum()); dn = float(pr[:sc.RET_CENTER].sum())
        pu = up / (up + dn) if (up + dn) > 1e-9 else 0.5
        ez = float((pr * np.abs(Z_CENTERS)).sum())
        t.append(int(df.index[ci].value // 1_000_000_000))
        price.append(float(c[ci]))
        p_up.append(round(pu, 4))
        conf.append(round(abs(pu - 0.5) * 2.0, 4))
        exp_move.append(ez * float(sigma[ci]))
        vol.append(float(sigma[ci]))
        ret_next.append(float(np.log(c[ci + 1] / c[ci])))
    if len(price) < 30:
        return None
    return {"base": base, "n": len(price), "t": t, "price": price, "p_up": p_up,
            "conf": conf, "exp_move": exp_move, "vol": vol, "ret_next": ret_next}


# ------------------------------------------------------------------------------------
# Benchmark in return-bucket space (predict-page metrics, driven by the byte model)

def _bench_metrics(pred, actual, conf, psign):
    """All directional/calibration/magnitude metrics, computed on the FULL series.
    pred/actual are return-bucket indices (0..RET_BINS-1, center=RET_CENTER); magnitude uses
    the bucket offsets. psign is the per-bar directional sign from prob mass up-vs-down (same
    signal hindcast scores): the argmax bucket sits on the flat center ~90% of 1m bars, so it
    is useless as a direction; prob mass is the real directional call."""
    rc = sc.RET_CENTER
    pred = np.asarray(pred); actual = np.asarray(actual); conf = np.asarray(conf, dtype=float)
    n = len(pred)
    ps = np.sign(np.asarray(psign)); a_s = np.sign(actual - rc)
    dec = ps != 0
    jdir = a_s != 0
    mdec = dec & jdir
    correct = ps == a_s
    up_m = (ps > 0) & jdir; dn_m = (ps < 0) & jdir
    po = (pred - rc).astype(float); ao = (actual - rc).astype(float)

    hc_p = None; hc_n = 0
    if mdec.any():
        cd = conf[mdec]; thr = float(np.quantile(cd, 0.75))
        hm = mdec & (conf >= thr)
        hc_n = int(hm.sum())
        hc_p = float(correct[hm].mean()) if hm.any() else None

    calib = []
    if mdec.any():
        cd = conf[mdec]; cc = correct[mdec]
        edges = np.unique(np.quantile(cd, np.linspace(0, 1, 9)))
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mm = (cd >= lo) & (cd <= hi) if i == len(edges) - 2 else (cd >= lo) & (cd < hi)
            if mm.sum() >= 5:
                calib.append({"conf": float(cd[mm].mean()), "precision": float(cc[mm].mean()),
                              "n": int(mm.sum())})

    mcorr = None
    if n > 2 and po.std() > 0 and ao.std() > 0:
        mcorr = float(np.corrcoef(po, ao)[0, 1])

    return {
        "n": n,
        "precision_decisive": float(correct[mdec].mean()) if mdec.any() else None,
        "high_conf_precision": hc_p, "high_conf_n": hc_n,
        "avg_confidence": float(conf.mean()) if n else None,
        "up_precision": float((a_s[up_m] > 0).mean()) if up_m.any() else None,
        "down_precision": float((a_s[dn_m] < 0).mean()) if dn_m.any() else None,
        "decisive_rate": float(dec.mean()) if n else 0.0,
        "directional_accuracy": float(correct[jdir].mean()) if jdir.any() else None,
        "magnitude_corr": mcorr,
        "magnitude_mae_buckets": float(np.abs(po - ao).mean()) if n else None,
        "calibration": calib,
        "up_rate": float((a_s > 0).mean()) if n else 0.0,
        "down_rate": float((a_s < 0).mean()) if n else 0.0,
        "flat_rate": float((a_s == 0).mean()) if n else 0.0,
    }


def _trade_metrics(calls):
    """Trader-facing stats from the per-bar directional calls. Most fields are gross
    (before fees); net_return subtracts ROUND_TRIP_FEE per trade so the after-fee result
    is honest. Each call's realized return = sign(direction) x actual log-return that bar."""
    if not calls:
        return None
    r = np.array([(1.0 if c["dir"] == "UP" else -1.0) * c["move"] for c in calls], dtype=np.float64)
    gp = float(r[r > 0].sum()); gl = float(-r[r < 0].sum())
    cum = np.cumsum(r); peak = np.maximum.accumulate(cum)
    sd = float(r.std())
    return {
        "n_trades": int(r.size),
        "win_rate": float((r > 0).mean()),
        "profit_factor": (gp / gl) if gl > 0 else None,
        "expectancy": float(r.mean()),
        "sharpe": (float(r.mean()) / sd) if sd > 0 else None,
        "max_drawdown": float((peak - cum).max()),
        "total_return": float(r.sum()),
        "net_return": float(r.sum() - ROUND_TRIP_FEE * r.size),
    }


def benchmark(model, seq_len, df, base="1m", stride=sc.BAR_STRIDE, max_points=360):
    """Walk df with the byte model; per bar capture the predicted vs actual return
    bucket and a directional confidence, then score the full predict-page metric set.
    Chart arrays are downsampled; metrics are computed on every scored bar."""
    c = df["close"].to_numpy(np.float64)
    arr, nb = _encode_df(df, stride)
    if nb < 60:
        return None
    off = len(c) - nb
    probpos = _score_bytes(model, seq_len, arr)

    pred = []; actual = []; conf = []; price = []; t = []; calls = []; psign = []
    for k in range(1, nb):
        gp = k * stride
        pr = probpos.get(gp)
        if pr is None:
            continue
        ab = int(BYTE2RET[arr[gp]])
        if ab < 0:
            continue
        ci = k + off
        if ci >= len(c) or ci < 1:
            break
        s = pr.sum()
        if s <= 0:
            continue
        pr = pr / s
        up = float(pr[sc.RET_CENTER + 1:].sum()); dn = float(pr[:sc.RET_CENTER].sum())
        pu = up / (up + dn) if (up + dn) > 1e-9 else 0.5
        cf = max(pu, 1.0 - pu)
        pred.append(int(np.argmax(pr)))
        actual.append(ab)
        conf.append(cf)
        psign.append(1 if up >= dn else -1)    # prob-mass direction, same call hindcast scores
        px = float(c[ci]); ts = int(df.index[ci].value // 1_000_000_000)
        price.append(px); t.append(ts)
        ret = float(np.log(c[ci] / c[ci - 1])) if c[ci - 1] > 0 else 0.0
        if ret != 0.0:
            right = (pu >= 0.5) == (ret > 0)
            calls.append({"t": ts, "price": round(px, 6), "dir": "UP" if pu >= 0.5 else "DOWN",
                          "conf": round(cf, 4), "move": round(ret, 6), "right": bool(right),
                          "score": cf * abs(ret)})

    if len(pred) < 30:
        return None
    pa = np.asarray(pred); aa = np.asarray(actual)
    m = _bench_metrics(pa, aa, conf, psign)
    rc = sc.RET_CENTER
    edge = np.asarray(psign) * (aa - rc).astype(float)
    equity = np.cumsum(edge)
    ao = (aa - rc).astype(float)
    base_mae = float(np.abs(ao[1:] - ao[:-1]).mean()) if len(ao) > 1 else None
    best = sorted((c for c in calls if c["right"]), key=lambda r: r["score"], reverse=True)[:6]
    worst = sorted((c for c in calls if not c["right"]), key=lambda r: r["score"], reverse=True)[:6]
    for r in best + worst:
        r.pop("score", None)
    return {
        "engine": "veritate", "n": int(len(pred)), "base": base,
        "ret_center": rc, "ret_bins": sc.RET_BINS,
        "metrics": m,
        "trading": _trade_metrics(calls),
        "baseline": {"magnitude_mae_buckets": base_mae},
        "best": best, "worst": worst,
        "equity": _downsample(equity, max_points).round(3).tolist(),
        "actual": _downsample(aa, max_points).tolist(),
        "pred": _downsample(pa, max_points).tolist(),
        "conf": _downsample(np.asarray(conf), max_points).round(4).tolist(),
        "price": _downsample(np.asarray(price), max_points).round(6).tolist(),
        "t": _downsample(np.asarray(t), max_points).tolist(),
    }


def data_report(source, max_rows=60):
    """Cheap per-instrument inventory for the data panel: file size (exact), first/last
    timestamp (head/tail reads), approximate bar count. No full-file scan of huge CSVs."""
    root = os.path.join(md.EXTERNAL_DIR, source)
    if not os.path.isdir(root):
        return {"n": 0, "total_bars": 0, "gb": 0.0, "instruments": []}
    files = sorted(f for f in os.listdir(root) if f.endswith(".csv"))
    total_bytes = 0; total_bars = 0; rows = []
    for fn in files:
        p = os.path.join(root, fn)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        total_bytes += sz
        first = last = None; bars = 0
        try:
            with open(p, "rb") as f:
                header = f.readline()
                line1 = f.readline().decode("ascii", "replace")
                tail = b""
                if sz > 4096:
                    f.seek(max(len(header), sz - 4096))
                    tail = f.read()
                else:
                    f.seek(0); tail = f.read()
            bars = int(sz / max(24, len(line1) or 72))
            first = _stamp(line1.split(",")[0])
            last_lines = [x for x in tail.decode("ascii", "replace").splitlines() if x.count(",") >= 5]
            if last_lines:
                last = _stamp(last_lines[-1].split(",")[0])
        except Exception:
            pass
        total_bars += bars
        rows.append({"name": fn[:-4], "bars": bars, "first": first or "?", "last": last or "?"})
    rows.sort(key=lambda r: r["bars"], reverse=True)
    return {"n": len(files), "total_bars": total_bars,
            "gb": round(total_bytes / 1e9, 2), "instruments": rows[:max_rows]}


def _stamp(tok):
    """A raw 'time'/'date' cell -> 'YYYY-MM-DD' (epoch ms/us/s or ISO string)."""
    tok = (tok or "").strip()
    if not tok:
        return None
    try:
        x = float(tok)
        ns = md.normalize_time(np.array([int(x)], dtype=np.int64))[0]
        return str(ns.date())
    except ValueError:
        return tok[:10]


def _roll_std(x, n):
    """Trailing std over a window of n (causal); warmup uses expanding std."""
    s = np.full(len(x), np.nan)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    csq = np.concatenate([[0.0], np.cumsum(x * x)])
    for i in range(len(x)):
        a = max(0, i - n + 1)
        m = i + 1 - a
        mean = (csum[i + 1] - csum[a]) / m
        var = max((csq[i + 1] - csq[a]) / m - mean * mean, 0.0)
        s[i] = np.sqrt(var)
    return s
