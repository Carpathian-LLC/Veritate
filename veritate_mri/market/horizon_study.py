# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Empirical predictability study on our real 1m crypto bars. Answers the question
#   the platform hinges on: WHICH horizon and WHICH target (direction vs volatility)
#   actually carries structure we can forecast — measured, not assumed.
# - Per horizon (1m..1d) and averaged across a representative basket, reports:
#     ret_ac1   : lag-1 autocorr of returns        (direction memory: ~0 = a coin flip)
#     |r|_ac1   : lag-1 autocorr of |returns|       (volatility clustering: the real signal)
#     VR(8),z   : Lo-MacKinlay variance ratio       (>1 trend, <1 reversion, z=significance)
#     up_rate   : unconditional P(up)
#     persist   : hit-rate of "predict last bar's sign" (naive directional baseline)
#     vol_R2    : in-sample R^2 of |r_t| ~ EWMA(|r|) (how forecastable next-bar vol is)
# - Pure measurement, no lookahead beyond the trailing EWMA. Run:
#     python veritate_mri/market/horizon_study.py
# veritate_mri/market/horizon_study.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as md

# ------------------------------------------------------------------------------------
# Constants

HORIZON_ORDER = ["1m", "5m", "15m", "1h", "4h", "1d"]
BASKET = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "DOTUSDT",
    "MATICUSDT", "ATOMUSDT", "NEARUSDT",
]

# ------------------------------------------------------------------------------------
# Metric helpers

def autocorr1(x):
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[:-1] * x[1:]).sum() / d) if d > 0 else 0.0


def variance_ratio(r, q):
    """Lo-MacKinlay VR(q) with homoskedastic z-stat. r = base-period log returns."""
    n = len(r)
    if n < q * 4:
        return np.nan, np.nan
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    if var1 <= 0:
        return np.nan, np.nan
    rq = np.convolve(r, np.ones(q), "valid")          # overlapping q-period returns
    m = q * (n - q + 1) * (1.0 - q / n)
    varq = ((rq - q * mu) ** 2).sum() / m
    vr = varq / (q * var1)
    z = (vr - 1.0) / np.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n))
    return float(vr), float(z)


def vol_r2(absr, halflife=10):
    """In-sample R^2 of |r_t| explained by a trailing EWMA of |r| (forecastable vol)."""
    a = 1.0 - np.exp(np.log(0.5) / halflife)
    ewma = np.empty_like(absr)
    acc = absr[0]
    for i in range(len(absr)):
        ewma[i] = acc
        acc = a * absr[i] + (1 - a) * acc
    pred = np.empty_like(absr)        # forecast of t uses EWMA through t-1 (no lookahead)
    pred[0] = absr[0]
    pred[1:] = ewma[:-1]
    y = absr[1:]
    yh = pred[1:]
    ss_res = ((y - yh) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def measure(close):
    r = np.diff(np.log(close))
    r = r[np.isfinite(r)]
    if len(r) < 200:
        return None
    nz = r[r != 0]
    absr = np.abs(r)
    vr8, z8 = variance_ratio(r, 8)
    sgn = np.sign(r)
    persist = float((sgn[1:] == sgn[:-1])[r[1:] != 0].mean()) if (r[1:] != 0).any() else np.nan
    return {
        "n": len(r),
        "ret_ac1": autocorr1(r),
        "absret_ac1": autocorr1(absr),
        "vr8": vr8, "z8": z8,
        "up_rate": float((nz > 0).mean()) if len(nz) else np.nan,
        "persist": persist,
        "vol_r2": vol_r2(absr),
    }

# ------------------------------------------------------------------------------------
# Study

def run(symbols):
    avail = set(md.list_instruments("crypto"))
    syms = [s for s in symbols if s in avail]
    print(f"basket: {len(syms)}/{len(symbols)} present -> {', '.join(syms)}\n")
    rows = {h: [] for h in HORIZON_ORDER}
    for s in syms:
        try:
            df = md.load(s, "1m", cols=["close", "volume"])
        except Exception as e:
            print(f"  skip {s}: {e}")
            continue
        for h in HORIZON_ORDER:
            bars = df if h == "1m" else md.resample(df, h)
            m = measure(bars["close"].to_numpy(dtype=np.float64))
            if m:
                rows[h].append(m)
        print(f"  loaded {s}: {len(df):,} 1m bars")

    keys = ["ret_ac1", "absret_ac1", "vr8", "z8", "up_rate", "persist", "vol_r2"]
    print("\n" + "=" * 92)
    print("PREDICTABILITY ACROSS HORIZONS  (mean across basket; '|sig' = # pairs with |ret_ac1|>2/sqrt(n))")
    print("=" * 92)
    hdr = f"{'horizon':>8} {'pairs':>6} {'avg_bars':>10} {'ret_ac1':>9} {'|r|_ac1':>9} {'VR(8)':>7} {'z8':>7} {'up_rate':>8} {'persist':>8} {'vol_R2':>7} {'dir|sig':>8}"
    print(hdr)
    print("-" * 92)
    summary = {}
    for h in HORIZON_ORDER:
        rs = rows[h]
        if not rs:
            continue
        def avg(k):
            v = [r[k] for r in rs if r[k] == r[k]]
            return float(np.mean(v)) if v else float("nan")
        nsig = sum(1 for r in rs if abs(r["ret_ac1"]) > 2.0 / np.sqrt(r["n"]))
        summary[h] = {k: avg(k) for k in keys}
        summary[h]["avg_bars"] = float(np.mean([r["n"] for r in rs]))
        summary[h]["dir_sig"] = nsig
        summary[h]["pairs"] = len(rs)
        print(f"{h:>8} {len(rs):>6} {summary[h]['avg_bars']:>10,.0f} "
              f"{avg('ret_ac1'):>9.4f} {avg('absret_ac1'):>9.4f} {avg('vr8'):>7.3f} "
              f"{avg('z8'):>7.2f} {avg('up_rate'):>8.4f} {avg('persist'):>8.4f} "
              f"{avg('vol_r2'):>7.3f} {nsig:>4}/{len(rs):<3}")
    print("=" * 92)
    print("read: ret_ac1~0 + persist~0.50 => direction is ~unpredictable; |r|_ac1 high + vol_R2 high")
    print("      => volatility/magnitude is forecastable. |VR-1| w/ |z|>2 => real trend(>1)/reversion(<1).")
    return summary


if __name__ == "__main__":
    run(BASKET)
