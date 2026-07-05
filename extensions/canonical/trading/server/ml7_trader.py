# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - ML weekly-book forward paper trader (fake money, long/short): the audited h7
#   strategy (SMOKE_RESULTS/daily_ml_audit_smoke.py, verdict CONFIRMED). A frozen
#   3-seed HistGradientBoosting ensemble scores next-7d direction for 40 crypto majors
#   on the exact 28-feature panel in ml7_features.py; the book is 7 daily-staggered
#   tranches, one rebalanced per day after the UTC close (long top-8 / short bottom-8,
#   3.5bp per side on traded delta). Entry lands the tick after the day closes, so the
#   validated expectation is the audit's lag1 arm (~13%/yr, Sharpe ~1.0), not the
#   same-close baseline. Funding is neither credited nor charged (audit: immaterial).
# - Train the frozen ensemble: python .../ml7_trader.py --train (full crypto_of history
#   resampled to daily, bridged to yesterday from OKX 1D bars; ~minutes of fitting,
#   ~25 min of CSV resample). Never trains at import or boot. Retrain quarterly.
# - FAKE money: JSON ledger, no broker, no keys. Prices are OKX public.
# extensions/canonical/trading/server/ml7_trader.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

import data as md
import ml7_features as mf
import news_trader as nt
import xsmom_trader as xt

# ------------------------------------------------------------------------------------
# Constants

ARMS = ("main",)
HIST_DIR = os.path.join(nt.LEDGER_DIR, "ml7_history")
FUND_CACHE_DIR = os.path.join(HIST_DIR, "funding")
MODEL_PATH = os.path.join(nt.LEDGER_DIR, "ml7_model.joblib")
MANIFEST_PATH = os.path.join(nt.LEDGER_DIR, "ml7_model_manifest.json")
# bar=1Dutc: plain 1D is UTC+8-aligned on OKX, which would break UTC-day parity with
# the crypto_of resample.
OKX_CANDLES_UTC = "https://www.okx.com/api/v5/market/candles?instId={}-USDT&bar=1Dutc&limit={}"
OKX_FUNDING = ("https://www.okx.com/api/v5/public/funding-rate-history"
               "?instId={}-USDT-SWAP&limit=100{}")

SEEDS = (0, 1, 2)              # audited ensemble: mean predict_proba across these
HGB_MAX_ITER = 200
TRANCHES = 7                   # overlapping 7-day books, one rebalanced per day
QUANTILE = 0.2                 # top/bottom quintile of scored names (8 of 40)
MIN_NAMES = 10                 # fewer scored names than this: hold, retry next tick
GROSS = 1.0                    # whole-book gross; each tranche holds GROSS/TRANCHES
FEE = 0.00035                  # 3.5bp per side, the audited fee point
MIN_TRADE_USD = 1.0
TICK_SEC = 3600                # mark hourly; one tranche trades per UTC day
HISTORY_CAP = 5000
ROLL_DAYS = 400                # rolling daily cache depth (features need ~120)
FULL_DAY_MINUTES = 1440        # nmin stamp for OKX daily bars (never gap-masked)
FUND_SEED_DAYS = 33            # one OKX funding page covers ~33 days at 8h
FUND_PAGES_MAX = 8
EPOCH = date(1970, 1, 1)       # tranche slot = epoch-day mod TRANCHES

# ------------------------------------------------------------------------------------
# Daily history + funding caches (seeded by --train, extended live from OKX)

def _hist_path(pair):
    return os.path.join(HIST_DIR, pair + ".csv")


def _fund_path(pair):
    return os.path.join(FUND_CACHE_DIR, pair + ".csv")


def _load_hist(pair):
    p = _hist_path(pair)
    if not os.path.isfile(p):
        return None
    # round_trip: the default float parser is 1 LSB off, which would break feature parity
    return pd.read_csv(p, index_col="day", parse_dates=["day"], float_precision="round_trip")


def _save_hist(pair, df):
    os.makedirs(HIST_DIR, exist_ok=True)
    df.tail(ROLL_DAYS).to_csv(_hist_path(pair), index_label="day")


def _load_fund(pair):
    p = _fund_path(pair)
    if not os.path.isfile(p):
        return pd.Series(dtype=float)
    df = pd.read_csv(p, index_col="day", parse_dates=["day"], float_precision="round_trip")
    return df["funding"] if "funding" in df else pd.Series(dtype=float)


def _save_fund(pair, s):
    os.makedirs(FUND_CACHE_DIR, exist_ok=True)
    s.tail(ROLL_DAYS).to_frame("funding").to_csv(_fund_path(pair), index_label="day")


def _okx_daily(stem, n):
    """Last n closed OKX 1D bars as a daily frame matching the cache schema, or None."""
    rows = xt._get(OKX_CANDLES_UTC.format(stem, min(300, n + 1)))
    if not rows:
        return None
    rows = [r for r in rows if r[8] == "1"]
    if not rows:
        return None
    rows.reverse()
    idx = pd.to_datetime([int(r[0]) // mf.MS_PER_DAY for r in rows], unit="D")
    return pd.DataFrame({"open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
                         "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
                         "volume": [float(r[5]) for r in rows],
                         "nmin": [FULL_DAY_MINUTES] * len(rows)}, index=idx)


def _okx_funding_daily(stem, min_day):
    """Daily funding sums (8h rates summed per UTC day) back to min_day, paging the
    OKX history endpoint. Empty series when the pair has no listed swap."""
    min_ms = (min_day - EPOCH).days * mf.MS_PER_DAY
    rows, after = [], ""
    for _ in range(FUND_PAGES_MAX):
        page = xt._get(OKX_FUNDING.format(stem, after))
        if not page:
            break
        rows += page
        oldest = int(page[-1]["fundingTime"])
        if oldest < min_ms or len(page) < 100:
            break
        after = "&after=%d" % oldest
    if not rows:
        return pd.Series(dtype=float)
    t = np.array([int(r["fundingTime"]) for r in rows], dtype=np.int64)
    day = pd.to_datetime(t // mf.MS_PER_DAY, unit="D")
    s = pd.Series([float(r["fundingRate"]) for r in rows], index=day)
    return s.groupby(level=0).sum().sort_index()


def _bridge_hist(df, stem, day):
    """Append OKX daily bars after df's last row through `day` (closed bars only)."""
    missing = (day - df.index[-1].date()).days
    if missing <= 0:
        return df
    bars = _okx_daily(stem, missing + 2)
    if bars is None:
        return df
    bars = bars[(bars.index > df.index[-1]) & (bars.index <= pd.Timestamp(day))]
    return pd.concat([df, bars]) if len(bars) else df


def _bridge_fund(s, stem, day):
    start = (s.index[-1].date() + timedelta(days=1)) if len(s) \
        else day - timedelta(days=FUND_SEED_DAYS)
    if start > day:
        return s
    fresh = _okx_funding_daily(stem, start)
    if not len(fresh):
        return s
    fresh = fresh[(fresh.index >= pd.Timestamp(start)) & (fresh.index <= pd.Timestamp(day))]
    return pd.concat([s, fresh]) if len(fresh) else s

# ------------------------------------------------------------------------------------
# Frozen model (trained by --train, loaded lazily, never at boot)

_MODEL = {"models": None, "manifest": None}


def load_model():
    if _MODEL["models"] is None:
        if not (os.path.isfile(MODEL_PATH) and os.path.isfile(MANIFEST_PATH)):
            return None
        import joblib
        _MODEL["models"] = joblib.load(MODEL_PATH)
        with open(MANIFEST_PATH) as f:
            _MODEL["manifest"] = json.load(f)
    return _MODEL["models"]


def model_state():
    if load_model() is None:
        return "missing - run ml7_trader.py --train"
    return "ready (trained through %s)" % _MODEL["manifest"]["train_end"]

# ------------------------------------------------------------------------------------
# Signal -> tranche targets -> tranche rebalance

def _signal(day):
    """Ensemble p_up per pair at `day`'s close: extend every cache from OKX, rebuild the
    panel, score the mean predict_proba of the seed ensemble. None when the model is
    missing or fewer than MIN_NAMES pairs have complete features at `day`."""
    models = load_model()
    if models is None:
        return None
    daily, fund = {}, {}
    for stem in xt.UNIVERSE:
        pair = stem + nt.QUOTE
        df = _load_hist(pair)
        if df is None:
            continue
        ext = _bridge_hist(df, stem, day)
        if len(ext) > len(df):
            _save_hist(pair, ext)
        daily[pair] = ext
        s = _load_fund(pair)
        fs = _bridge_fund(s, stem, day)
        if len(fs) > len(s):
            _save_fund(pair, fs)
        fund[pair] = fs
    if mf.BTC_PAIR not in daily:
        return None
    panel, _, _ = mf.build_panel(daily, fund)
    rows = panel[panel.index == pd.Timestamp(day)]
    if len(rows) < MIN_NAMES:
        return None
    x = rows[mf.FEATS].to_numpy()
    p = np.mean([m.predict_proba(x)[:, 1] for m in models], axis=0)
    return dict(zip(rows["pair"], p))


def tranche_targets(p_up):
    """Signed weight fractions for one tranche: +w on the k highest-p_up names, -w on
    the k lowest, w = (GROSS/TRANCHES)/2/k (the audited quintile construction)."""
    n = len(p_up)
    if n < MIN_NAMES:
        return {}
    k = max(1, int(n * QUANTILE))
    ranked = sorted(p_up, key=p_up.get)
    w = GROSS / TRANCHES / 2.0 / k
    out = {s: -w for s in ranked[:k]}
    out.update({s: w for s in ranked[-k:]})
    return out


def rebalance_tranche(led, slot, tgt, prices, fee):
    """Rebalance ONLY tranche `slot` to signed weight fractions of current equity;
    the other six tranches ride untouched (the audited turnover model). Fee per side
    on traded notional; aggregate positions carry the sum of all tranches."""
    tranches = led.setdefault("tranches", {})
    old = tranches.get(str(slot), {})
    eq = nt.equity(led, prices)
    new, acts = {}, []
    for sym in set(tgt) | set(old):
        p = prices.get(sym)
        if not p:
            if old.get(sym):
                new[sym] = old[sym]
            continue
        want = tgt.get(sym, 0.0) * eq / p
        dqty = want - old.get(sym, 0.0)
        if abs(dqty * p) < MIN_TRADE_USD:
            if old.get(sym):
                new[sym] = old[sym]
            continue
        led["cash"] -= dqty * p + abs(dqty * p) * fee
        pos = led["positions"].get(sym, 0.0) + dqty
        if abs(pos * p) >= MIN_TRADE_USD:
            led["positions"][sym] = pos
        else:
            led["positions"].pop(sym, None)
        if want:
            new[sym] = want
        acts.append({"sym": sym, "side": "buy" if dqty > 0 else "sell",
                     "qty": round(dqty, 8), "price": p})
    tranches[str(slot)] = new
    return acts


def _last_closed_day():
    return (datetime.now(timezone.utc) - timedelta(days=1)).date()


def tick(arm, ledger_path):
    """One cycle: on a new closed UTC day (and model + data ready), score all pairs and
    rebalance that day's tranche; every cycle, mark to live OKX prices and append an
    _exp_view-compatible history row."""
    led = nt.load_ledger(ledger_path)
    day = _last_closed_day()
    acts = []
    if led.get("day") != str(day):
        sig = _signal(day)
        if sig:
            tgt = tranche_targets(sig)
            slot = (day - EPOCH).days % TRANCHES
            syms = set(tgt) | set(led.setdefault("tranches", {}).get(str(slot), {})) \
                | set(led["positions"])
            last_px = led.setdefault("last_px", {})
            prices = {}
            for s in syms:
                p = xt.mark_price(s[:-len(nt.QUOTE)])
                if p:
                    last_px[s] = p
                prices[s] = p or last_px.get(s)
            acts = rebalance_tranche(led, slot, tgt, prices, FEE)
            led["day"] = str(day)
    last_px = led.setdefault("last_px", {})
    for sym in led["positions"]:
        p = xt.mark_price(sym[:-len(nt.QUOTE)])
        if p:
            last_px[sym] = p
    marks = {s: last_px.get(s) for s in led["positions"]}
    eq = nt.equity(led, marks)
    btc_px = xt.mark_price("BTC")
    led["history"].append({"t": int(time.time()), "equity": round(eq, 2), "market": "crypto",
                           "bench_px": btc_px, "bench_asset": "BTC", "day": led.get("day"),
                           "prices": {s[:-len(nt.QUOTE)]: p for s, p in marks.items() if p},
                           "acts": acts})
    led["history"] = led["history"][-HISTORY_CAP:]
    nt.save_ledger(led, ledger_path)
    return {"equity": eq, "pnl": eq - led["start_cash"], "acts": acts, "day": led.get("day")}

# ------------------------------------------------------------------------------------
# Training (CLI only; writes the frozen ensemble + manifest + seeds the live caches)

def train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    import sklearn
    t0 = time.time()
    day = _last_closed_day()
    src = md.source_dir("crypto_of")
    daily, fund = {}, {}
    for stem in xt.UNIVERSE:
        pair = stem + nt.QUOTE
        d = mf.daily_from_minute_csv(os.path.join(src, pair + ".csv"))
        daily[pair] = _bridge_hist(d, stem, day)
        fpath = os.path.join(md.FUNDING_DIR, pair + ".csv")
        s = mf.funding_daily_from_csv(fpath) if os.path.isfile(fpath) else pd.Series(dtype=float)
        fund[pair] = _bridge_fund(s, stem, day)
        print("%s days=%d (thru %s) funding=%d" % (pair, len(daily[pair]),
              daily[pair].index[-1].date(), len(fund[pair])), flush=True)
    panel, _, _ = mf.build_panel(daily, fund)
    tr = panel[panel[mf.LABEL].notna()]
    x, y = tr[mf.FEATS].to_numpy(), tr[mf.LABEL].to_numpy()
    print("panel=%d train_rows=%d span=%s..%s" % (len(panel), len(tr),
          tr.index.min().date(), tr.index.max().date()), flush=True)
    t_fit = time.time()
    models = [HistGradientBoostingClassifier(max_iter=HGB_MAX_ITER, random_state=s).fit(x, y)
              for s in SEEDS]
    fit_s = time.time() - t_fit
    os.makedirs(nt.LEDGER_DIR, exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    manifest = {"train_end": str(tr.index.max().date()),
                "data_end": str(max(d.index[-1] for d in daily.values()).date()),
                "rows": len(tr), "features": mf.FEATS, "label": mf.LABEL,
                "seeds": list(SEEDS), "hgb_max_iter": HGB_MAX_ITER,
                "sklearn": sklearn.__version__,
                "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fit_seconds": round(fit_s, 1),
                "total_seconds": round(time.time() - t0, 1)}
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=1)
    for pair in daily:
        _save_hist(pair, daily[pair])
        _save_fund(pair, fund[pair])
    _MODEL["models"] = None
    print("saved %s (fit %.1fs, total %.1fs)" % (MODEL_PATH, fit_s, manifest["total_seconds"]),
          flush=True)
    return manifest

# ------------------------------------------------------------------------------------
# Managed threads (dashboard start/stop) + resume + CLI

_RUNS = {}


def label_for(arm):
    return "ml7_" + arm


def start_thread(arm):
    if arm not in ARMS:
        return False
    label = label_for(arm)
    r = _RUNS.get(label)
    if r and r["thread"] and r["thread"].is_alive():
        return False
    path = nt.ledger_for(label)
    led = nt.load_ledger(path)
    led["auto"] = True
    nt.save_ledger(led, path)
    ev = threading.Event()
    cfg = {"label": label, "arm": arm, "fee_bps": round(FEE * 1e4, 1), "interval": TICK_SEC,
           "started": int(time.time())}
    _RUNS[label] = {"thread": None, "stop": ev, "cfg": cfg}

    def _run():
        while not ev.is_set():
            try:
                tick(arm, path)
            except Exception:
                pass
            ev.wait(TICK_SEC)
    t = threading.Thread(target=_run, name=f"ml7-{arm}", daemon=True)
    _RUNS[label]["thread"] = t
    t.start()
    return True


def stop_thread(arm=None):
    for label in ([label_for(arm)] if arm else list(_RUNS)):
        r = _RUNS.get(label)
        if r and r["stop"]:
            r["stop"].set()
        _RUNS.pop(label, None)
        path = nt.ledger_for(label)
        led = nt.load_ledger(path)
        led["auto"] = False
        nt.save_ledger(led, path)
    return True


def resume():
    """Restart arms flagged running in their ledger (server reboot must not gap the record)."""
    return {a: start_thread(a) for a in ARMS if nt.load_ledger(nt.ledger_for(label_for(a))).get("auto")}


def status_all():
    out = {}
    for arm in ARMS:
        r = _RUNS.get(label_for(arm))
        alive = bool(r and r["thread"] and r["thread"].is_alive())
        out[arm] = {"running": alive, **(r["cfg"] if (alive and r) else {})}
    return out


def main():
    p = argparse.ArgumentParser(description="ML weekly-book (7-day ranks) paper trader (fake money).")
    p.add_argument("--train", action="store_true", help="train + freeze the seed ensemble, then exit")
    p.add_argument("--once", action="store_true", help="one tick, then exit")
    a = p.parse_args()
    if a.train:
        train()
        return
    if a.once:
        r = tick("main", nt.ledger_for(label_for("main")))
        led = nt.load_ledger(nt.ledger_for(label_for("main")))
        print(json.dumps({**r, "model": model_state(),
                          "tranches": {k: {s: round(q, 6) for s, q in v.items()}
                                       for k, v in led.get("tranches", {}).items()}}, indent=2))
        return
    start_thread("main")
    print(f"ml7_trader [PAPER] 3-seed HGB ensemble, 7 daily tranches, fee={FEE * 1e4:.1f}bp/side "
          f"tick={TICK_SEC}s model={model_state()} -> {nt.ledger_for(label_for('main'))}", flush=True)
    while True:
        time.sleep(TICK_SEC)


if __name__ == "__main__":
    main()
