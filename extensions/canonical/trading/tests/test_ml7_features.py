# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Feature-parity tests for ml7_features: the extracted builder must produce
#   byte-identical values to SMOKE_RESULTS/daily_ml_panel_smoke.py (the audited
#   construction the frozen model was validated on). No network, seeded RNG.
# extensions/canonical/trading/tests/test_ml7_features.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import ml7_features as mf

# ------------------------------------------------------------------------------------
# Constants

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SMOKE_PATH = os.path.join(REPO, "SMOKE_RESULTS", "daily_ml_panel_smoke.py")
N_DAYS = 220
SEED = 7

# ------------------------------------------------------------------------------------
# Fixtures / helpers

def _smoke():
    spec = importlib.util.spec_from_file_location("daily_ml_panel_smoke", SMOKE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _daily(rng, n=N_DAYS, start="2024-01-01", gap_days=()):
    idx = pd.date_range(start, periods=n, freq="D")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    nmin = np.full(n, 1440)
    nmin[list(gap_days)] = 800          # early gaps: exercise masking, leave the tail warm
    return pd.DataFrame({
        "open": np.concatenate([[100.0], close[:-1]]),
        "high": close * (1 + np.abs(rng.normal(0, 0.01, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.01, n))),
        "close": close,
        "volume": np.exp(rng.normal(10, 1, n)),
        "nmin": nmin}, index=idx)


def _fund(rng, idx):
    keep = idx[rng.random(len(idx)) < 0.9]
    return pd.Series(rng.normal(0, 1e-4, len(keep)), index=keep)


def _fixture():
    rng = np.random.default_rng(SEED)
    daily = {"BTCUSDT": _daily(rng), "AAAUSDT": _daily(rng, gap_days=(4, 9)),
             "BBBUSDT": _daily(rng),
             "CCCUSDT": _daily(rng, n=150, start="2024-03-11", gap_days=(2,))}
    fund = {p: _fund(rng, daily[p].index) for p in ["BTCUSDT", "AAAUSDT", "CCCUSDT"]}
    return daily, fund


def _smoke_panel(base, daily, fund):
    """The smoke's own panel assembly (main() lines), run on the fixture frames."""
    btc = daily["BTCUSDT"]
    idx = pd.date_range(btc.index[0], btc.index[-1], freq="D")
    logc = np.log(btc["close"].reindex(idx))
    r1, ret7 = logc.diff(), logc - logc.shift(7)
    frames = []
    for pair in sorted(daily):
        f = base.compute_features(daily[pair], fund.get(pair, pd.Series(dtype=float)), r1, ret7)
        f = f.drop(columns=["r1_simple", "fund_daily", "n_bad_days"])
        f["pair"] = pair
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    panel["xs_rank_7"] = panel.groupby("date")["ret_7"].rank(pct=True) - 0.5
    panel["xs_rank_30"] = panel.groupby("date")["ret_30"].rank(pct=True) - 0.5
    return panel[panel[base.FEATS].notna().all(axis=1)]

# ------------------------------------------------------------------------------------
# Functions

def test_feats_list_matches_smoke():
    """FEATS is exactly the smoke's 28-feature list, same order."""
    assert mf.FEATS == _smoke().FEATS


def test_panel_features_byte_equal_smoke():
    """build_panel produces byte-identical feature values to the smoke's builder."""
    base = _smoke()
    daily, fund = _fixture()
    got, _, _ = mf.build_panel(daily, fund)
    want = _smoke_panel(base, daily, fund)
    assert len(got) == len(want) and len(got) > 0
    assert list(got["pair"]) == list(want["pair"])
    assert np.array_equal(got[mf.FEATS].to_numpy(), want[base.FEATS].to_numpy())


def test_label_matches_smoke():
    """The y_sign_7 label matches the smoke's, NaN tail included."""
    base = _smoke()
    daily, fund = _fixture()
    got, _, _ = mf.build_panel(daily, fund)
    want = _smoke_panel(base, daily, fund)
    assert np.array_equal(got[mf.LABEL].to_numpy(), want["y_sign_7"].to_numpy(),
                          equal_nan=True)


def test_minute_csv_mixed_epoch_units(tmp_path):
    """daily_from_minute_csv groups mixed ms/us epochs onto the right UTC days."""
    day_ms = 20000 * mf.MS_PER_DAY
    rows = ["time,open,high,low,close,volume"]
    for m in range(3):
        rows.append(f"{day_ms + m * 60000},1,2,0.5,1.5,10")
    for m in range(2):
        rows.append(f"{(day_ms + mf.MS_PER_DAY + m * 60000) * 1000},2,3,1.5,2.5,20")
    p = tmp_path / "PAIRUSDT.csv"
    p.write_text("\n".join(rows))
    g = mf.daily_from_minute_csv(str(p))
    assert list(g["nmin"]) == [3, 2] and list(g["close"]) == [1.5, 2.5]
    assert g.index[1] - g.index[0] == pd.Timedelta(days=1)
