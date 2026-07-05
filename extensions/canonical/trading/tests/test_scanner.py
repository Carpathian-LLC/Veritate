# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Deterministic tests for the Market Intel scanner: z-score and anomaly math, pump
#   flag + cooldown, event append/rotation, snapshot baselines. No network: fetchers
#   monkeypatched, all paths redirected to tmp_path.
# extensions/canonical/trading/tests/test_scanner.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import scanner as sc

# ------------------------------------------------------------------------------------
# Fixtures

T0 = 1000000
STEP = 300


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every scanner disk path to tmp_path and clear the in-memory cache."""
    monkeypatch.setattr(sc, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(sc, "SNAP_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(sc, "EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setattr(sc, "STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(sc, "CG_CACHE_PATH", str(tmp_path / "cg_cache.json"))
    monkeypatch.setattr(sc, "_CACHE", None)
    return tmp_path


def _tick(px=100.0, vol=1e6, sym="AAA"):
    return {sym: {"px": px, "open24h": px, "high": px * 1.01, "low": px * 0.99, "vol_usd": vol}}


def _snap_list(n, sym="AAA", jitter=False):
    """n trailing snapshots; jitter adds deterministic wiggle so baselines have variance."""
    out = []
    for i in range(n):
        px = 100.0 * (1 + 0.001 * (i % 3)) if jitter else 100.0
        vol = 1e6 * (1 + 0.01 * (i % 5)) if jitter else 1e6
        out.append({"ts": T0 + i * STEP, "syms": {sym: [px, vol, 0.02]}})
    return out


def _seed_snaps(n, **kw):
    for s in _snap_list(n, **kw):
        sc._save_snap(s)


def _no_cg(monkeypatch, trending=None, meme=None):
    monkeypatch.setattr(sc, "cg_boards", lambda refresh=False: {
        "ts": 0, "trending": trending or [], "meme": meme or []})

# ------------------------------------------------------------------------------------
# Math

def test_zscore_flat_baseline_is_zero():
    """A flat history has zero variance, so the z-score is 0, not a crash."""
    assert sc.zscore(5.0, [1.0] * 10) == 0.0


def test_zscore_short_baseline_is_zero():
    """Fewer than BASELINE_MIN observations returns 0 (untrusted baseline)."""
    assert sc.zscore(9.0, [1.0, 2.0]) == 0.0


def test_zscore_value():
    """z = (x - mean) / std over the given baseline."""
    hist = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    assert sc.zscore(2.0, hist) == pytest.approx(1.0)


def test_anomaly_score_weights():
    """Score combines |price z|, positive vol z, and range expansion over 1x."""
    got = sc.anomaly_score(-2.0, 3.0, 1.5)
    assert got == pytest.approx(sc.W_PRICE * 2.0 + sc.W_VOL * 3.0 + sc.W_RANGE * 0.5)


def test_anomaly_score_ignores_negative_vol_and_compression():
    """Volume below baseline and range compression add nothing to the score."""
    assert sc.anomaly_score(1.0, -2.0, 0.5) == pytest.approx(sc.W_PRICE * 1.0)


def test_pump_flag_requires_both_spikes():
    """Price z alone or volume z alone never flags; both at threshold does."""
    assert not sc.pump_flag(sc.PRICE_Z_MIN, sc.VOL_Z_MIN - 0.1)
    assert not sc.pump_flag(sc.PRICE_Z_MIN - 0.1, sc.VOL_Z_MIN)
    assert sc.pump_flag(sc.PRICE_Z_MIN, sc.VOL_Z_MIN)

# ------------------------------------------------------------------------------------
# Events file

def test_append_event_writes_one_json_line(sandbox):
    """Each flagged event is one parseable JSON line in events.jsonl."""
    sc.append_event({"ts": 1, "sym": "AAA", "score": 9.9})
    with open(sc.EVENTS_PATH) as f:
        lines = f.readlines()
    assert len(lines) == 1 and json.loads(lines[0])["sym"] == "AAA"


def test_append_event_rotates_at_cap(sandbox, monkeypatch):
    """events.jsonl keeps only the newest EVENTS_MAX lines."""
    monkeypatch.setattr(sc, "EVENTS_MAX", 3)
    for i in range(5):
        sc.append_event({"ts": i, "sym": f"S{i}"})
    got = sc.events(limit=10)
    assert len(got) == 3 and got[0]["sym"] == "S4"


def test_events_newest_first(sandbox):
    """events() returns the most recent event first."""
    sc.append_event({"ts": 1, "sym": "OLD"})
    sc.append_event({"ts": 2, "sym": "NEW"})
    assert [e["sym"] for e in sc.events(limit=2)] == ["NEW", "OLD"]

# ------------------------------------------------------------------------------------
# Snapshots

def test_snapshot_rotation(sandbox, monkeypatch):
    """Snapshot files beyond SNAP_KEEP are pruned oldest-first."""
    monkeypatch.setattr(sc, "SNAP_KEEP", 4)
    _seed_snaps(7)
    snaps = sc._snaps()
    assert len(snaps) == 4 and snaps[0]["ts"] == T0 + 3 * STEP

# ------------------------------------------------------------------------------------
# Scoring + pump flag + cooldown

def test_score_rows_flags_spike_and_appends_event(sandbox):
    """A price+volume spike vs a trailing baseline yields both z-spikes, a pump flag,
    and exactly one appended event."""
    snaps = _snap_list(10, jitter=True)
    rows, evs = sc._score_rows(_tick(px=130.0, vol=2e7), snaps, T0 + 10 * STEP, {})
    r = rows[0]
    assert r["pump"] is True and r["price_z"] >= sc.PRICE_Z_MIN and r["vol_z"] >= sc.VOL_Z_MIN
    assert len(evs) == 1 and sc.event_count() == 1


def test_score_rows_cooldown_suppresses_repeat(sandbox):
    """A second flag for the same symbol inside COOLDOWN_S records no second event."""
    snaps = _snap_list(10, jitter=True)
    state = {}
    sc._score_rows(_tick(px=130.0, vol=2e7), snaps, T0 + 10 * STEP, state)
    _, evs2 = sc._score_rows(_tick(px=130.0, vol=2e7), snaps, T0 + 11 * STEP, state)
    assert evs2 == [] and sc.event_count() == 1


def test_score_rows_reflag_after_cooldown(sandbox):
    """The same symbol can flag again once COOLDOWN_S has elapsed."""
    snaps = _snap_list(10, jitter=True)
    state = {}
    sc._score_rows(_tick(px=130.0, vol=2e7), snaps, T0 + 10 * STEP, state)
    _, evs2 = sc._score_rows(_tick(px=130.0, vol=2e7), snaps,
                             T0 + 10 * STEP + sc.COOLDOWN_S, state)
    assert len(evs2) == 1 and sc.event_count() == 2


def test_score_rows_quiet_market_no_flag(sandbox):
    """An in-line move on baseline volume produces no pump flag and no event."""
    snaps = _snap_list(10, jitter=True)
    rows, evs = sc._score_rows(_tick(px=100.0, vol=1e6), snaps, T0 + 10 * STEP, {})
    assert evs == [] and rows[0]["pump"] is False

# ------------------------------------------------------------------------------------
# Scan

def test_scan_ret_1h_from_old_snapshot(sandbox, monkeypatch):
    """ret_1h is computed against the newest snapshot at least RET1H_S old."""
    _seed_snaps(14)
    monkeypatch.setattr(sc, "okx_rows", lambda: _tick(px=110.0))
    _no_cg(monkeypatch)
    monkeypatch.setattr(sc.time, "time", lambda: T0 + 14 * STEP)
    s = sc.scan(refresh=True)
    assert s["rows"][0]["ret_1h"] == pytest.approx(0.10)


def test_scan_tags_meme_and_trending(sandbox, monkeypatch):
    """Rows carry meme/trending flags and display names from the CoinGecko boards."""
    _seed_snaps(6)
    monkeypatch.setattr(sc, "okx_rows", lambda: _tick())
    _no_cg(monkeypatch,
           trending=[{"sym": "AAA", "name": "Aaa Coin", "rank": 7}],
           meme=[{"sym": "AAA", "name": "Aaa Coin", "px": 1.0, "ret_1h": 0, "ret_24h": 0,
                  "ret_7d": 0, "vol_usd": 1, "mcap": 1}])
    monkeypatch.setattr(sc.time, "time", lambda: T0 + 6 * STEP)
    row = sc.scan(refresh=True)["rows"][0]
    assert row["meme"] and row["trending"] and row["name"] == "Aaa Coin"


def test_scan_feed_down_keeps_last_result(sandbox, monkeypatch):
    """An unreachable feed returns the previous scan instead of an empty board."""
    _seed_snaps(6)
    monkeypatch.setattr(sc, "okx_rows", lambda: _tick())
    _no_cg(monkeypatch)
    monkeypatch.setattr(sc.time, "time", lambda: T0 + 6 * STEP)
    first = sc.scan(refresh=True)
    monkeypatch.setattr(sc, "okx_rows", lambda: None)
    assert sc.scan(refresh=True) is first


def test_scan_persists_snapshot(sandbox, monkeypatch):
    """Each scan appends one snapshot carrying price, volume, and range per symbol."""
    monkeypatch.setattr(sc, "okx_rows", lambda: _tick(px=50.0, vol=7e5))
    _no_cg(monkeypatch)
    monkeypatch.setattr(sc.time, "time", lambda: T0)
    sc.scan(refresh=True)
    snaps = sc._snaps()
    assert len(snaps) == 1 and snaps[0]["syms"]["AAA"][0] == 50.0

# ------------------------------------------------------------------------------------
# Watch resume

def test_resume_restarts_stamped_watch(sandbox, monkeypatch):
    """resume() starts the watch thread when state.json says it was on."""
    started = {}
    monkeypatch.setattr(sc, "start_thread", lambda interval, model: started.update(i=interval) or True)
    sc._write_json(sc.STATE_PATH, {"watch": {"on": True, "interval": 60, "model": None}})
    assert sc.resume() is True and started["i"] == 60


def test_resume_noop_when_off(sandbox):
    """resume() does nothing when the watch was stopped."""
    sc._write_json(sc.STATE_PATH, {"watch": {"on": False}})
    assert sc.resume() is False
