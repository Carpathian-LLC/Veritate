# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The production model bundle the dashboard + live feed load. Per (base, horizon) it
#   holds: a volatility forecaster (the strong signal), an isotonic-CALIBRATED direction
#   classifier (so a "65%" really means ~65%), a split-conformal cone scale (so the band
#   covers its nominal rate), and regime thresholds. Trained on a purged split.
# - predict_latest(df) recomputes features with the SAME features.compute used in
#   training (no train/serve skew) and returns a forecast for the last closed bar.
# - run to train+save all default horizons:  python veritate_mri/market/models.py
# veritate_mri/market/models.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import data as md
import dataset as ds
import features as ff
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# ------------------------------------------------------------------------------------
# Constants

MODEL_DIR = os.path.join(md.ROOT, "models", "market")
DEFAULT_PAIRS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT", "DOTUSDT", "ATOMUSDT", "NEARUSDT",
    "UNIUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "INJUSDT", "SUIUSDT", "AAVEUSDT", "XLMUSDT", "ICPUSDT", "HBARUSDT", "RUNEUSDT",
]
DEFAULT_HORIZONS = [5, 15, 60]
CONE_COVERAGE = 0.80                      # nominal |return| coverage of the conformal band

# ------------------------------------------------------------------------------------
# Training

def _gb_reg():
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, random_state=0)


def _gb_clf():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=0)


def train(symbols, horizon, base="1m", max_bars=500000, train_stride=3, source="crypto"):
    d = ds.build(symbols, horizon=horizon, base=base, max_bars=max_bars, source=source)
    tr, va = ds.purged_split(d, val_frac=0.2)
    t = d["t"]
    # carve a calibration slice = newest 15% of train (time-ordered, before val) for isotonic
    tr_idx = np.flatnonzero(tr)
    tr_idx = tr_idx[np.argsort(t[tr_idx])]
    cut = int(len(tr_idx) * 0.85)
    fit_idx = tr_idx[:cut][:: train_stride]
    cal_idx = tr_idx[cut:]
    va_idx = np.flatnonzero(va)

    X = d["X"]
    # --- volatility ---
    vol = _gb_reg()
    vol.fit(X[fit_idx], np.log(np.clip(d["y_vol"][fit_idx], 1e-9, None)))
    vpred = np.exp(vol.predict(X[va_idx]))
    vy = d["y_vol"][va_idx].astype(np.float64)
    vbar = d["y_vol"][fit_idx].mean()
    vol_r2 = 1.0 - np.sum((vy - vpred) ** 2) / np.sum((vy - vbar) ** 2)

    # --- direction (fit on non-flat, isotonic-calibrate on the calibration slice) ---
    ydir = d["y_dir"]
    fit_nz = fit_idx[ydir[fit_idx] != 0]
    cal_nz = cal_idx[ydir[cal_idx] != 0]
    va_nz = va_idx[ydir[va_idx] != 0]
    clf = _gb_clf()
    clf.fit(X[fit_nz], (ydir[fit_nz] == 1).astype(int))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(clf.predict_proba(X[cal_nz])[:, 1], (ydir[cal_nz] == 1).astype(int))

    praw = clf.predict_proba(X[va_nz])[:, 1]
    p = iso.predict(praw)
    yv = (ydir[va_nz] == 1).astype(int)
    base_rate = float(yv.mean())
    metrics = {
        "vol_r2": float(vol_r2),
        "vol_corr": float(np.corrcoef(vpred, vy)[0, 1]),
        "dir_base_rate": base_rate,
        "dir_majority_acc": max(base_rate, 1 - base_rate),
        "dir_acc": float(((p > 0.5).astype(int) == yv).mean()),
        "dir_auc": float(roc_auc_score(yv, p)),
        "dir_brier": float(brier_score_loss(yv, p)),
        "dir_logloss": float(log_loss(yv, p, labels=[0, 1])),
        "n_val": int(len(va_idx)), "n_train": int(len(fit_idx)),
        "pairs": d["symbols"], "horizon": horizon, "base": base, "source": source,
    }

    # --- conformal cone scale: k s.t. P(|y_ret| <= k * vol_fwd) ~ CONE_COVERAGE on val ---
    ratio = np.abs(d["y_ret"][va_idx].astype(np.float64)) / np.clip(vpred, 1e-9, None)
    cone_k = float(np.quantile(ratio[np.isfinite(ratio)], CONE_COVERAGE))

    # --- regime thresholds from train per-bar vol (feature rv30) ---
    i_rv30 = d["features"].index("rv30")
    rv30_tr = X[fit_idx, i_rv30]
    regime = {"calm": float(np.quantile(rv30_tr, 0.33)), "turbulent": float(np.quantile(rv30_tr, 0.66))}

    return {
        "vol": vol, "clf": clf, "iso": iso, "features": d["features"], "horizon": horizon,
        "base": base, "cone_k": cone_k, "cone_cov": CONE_COVERAGE, "regime": regime,
        "metrics": metrics,
    }


def train_and_save(symbols=None, horizons=None, base="1m", max_bars=500000, train_stride=3, source="crypto"):
    symbols = symbols or DEFAULT_PAIRS
    horizons = horizons or DEFAULT_HORIZONS
    os.makedirs(MODEL_DIR, exist_ok=True)
    summary = {}
    for h in horizons:
        bundle = train(symbols, h, base=base, max_bars=max_bars, train_stride=train_stride, source=source)
        name = f"{base}_h{h}"
        path = os.path.join(MODEL_DIR, name + ".joblib")
        joblib.dump(bundle, path + ".tmp")          # atomic: never expose a half-written model
        os.replace(path + ".tmp", path)
        summary[name] = bundle["metrics"]
        m = bundle["metrics"]
        print(f"[{name}] vol_R2={m['vol_r2']:.3f} dir_acc={m['dir_acc']:.4f} "
              f"(maj {m['dir_majority_acc']:.4f}) auc={m['dir_auc']:.3f} cone_k={bundle['cone_k']:.2f}")
    with open(os.path.join(MODEL_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

# ------------------------------------------------------------------------------------
# Inference

_LOAD_CACHE = {}


class MarketModel:
    def __init__(self, bundle):
        self.__dict__.update(bundle)
        self._fidx = {f: i for i, f in enumerate(self.features)}

    @classmethod
    def load(cls, base, horizon):
        path = os.path.join(MODEL_DIR, f"{base}_h{horizon}.joblib")
        mt = os.path.getmtime(path)
        hit = _LOAD_CACHE.get(path)
        if hit and hit[0] == mt:                    # cache, invalidated when the file is retrained
            return hit[1]
        mm = cls(joblib.load(path))
        _LOAD_CACHE[path] = (mt, mm)
        return mm

    @classmethod
    def available(cls):
        if not os.path.isdir(MODEL_DIR):
            return []
        return sorted(f[:-7] for f in os.listdir(MODEL_DIR) if f.endswith(".joblib"))

    def _regime_label(self, x):
        rv = x[self._fidx["rv30"]]
        if rv <= self.regime["calm"]:
            return "calm"
        return "turbulent" if rv >= self.regime["turbulent"] else "normal"

    def predict_row(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        vol_fwd = float(np.exp(self.vol.predict(x))[0])
        p_up = float(self.iso.predict(self.clf.predict_proba(x)[:, 1])[0])
        conf = abs(p_up - 0.5) * 2.0
        return {"vol_fwd": vol_fwd, "p_up": p_up, "confidence": conf,
                "regime": self._regime_label(x[0]), "horizon": self.horizon, "base": self.base}

    def predict_latest(self, df):
        f = ff.compute(df).dropna()
        if f.empty:
            return None
        return self.predict_row(f.iloc[-1].to_numpy())

    def cone(self, vol_fwd, p_up, levels=(0.5, 0.8, 0.95)):
        """Forward probability cone: per-step bands from forecast vol, slight drift from p_up.
        Returns {level: [(lo, hi) per step 1..H]} in log-return space (multiply price by exp)."""
        H = self.horizon
        step_sd = vol_fwd / np.sqrt(H)
        drift = (p_up - 0.5) * 2.0 * step_sd * 0.5          # weak directional tilt
        from scipy.stats import norm
        out = {}
        for lv in levels:
            z = norm.ppf(0.5 + lv / 2.0) * (self.cone_k / norm.ppf(0.5 + self.cone_cov / 2.0))
            band = []
            for k in range(1, H + 1):
                cum_sd = step_sd * np.sqrt(k)
                mu = drift * k
                band.append((mu - z * cum_sd, mu + z * cum_sd))
            out[lv] = band
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="5,15,60")
    ap.add_argument("--base", default="1m")
    ap.add_argument("--max-bars", type=int, default=500000)
    ap.add_argument("--pairs", type=int, default=28)
    a = ap.parse_args()
    train_and_save(symbols=DEFAULT_PAIRS[: a.pairs],
                   horizons=[int(x) for x in a.horizons.split(",")],
                   base=a.base, max_bars=a.max_bars)
