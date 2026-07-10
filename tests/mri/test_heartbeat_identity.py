# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for heartbeat device identity: fingerprint binding, legacy
#   grandfathering (no churn), and clone auto-deconfliction. state + hardware
#   key are stubbed so nothing touches the real machine.
# tests/mri/test_heartbeat_identity.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(REPO_ROOT, "veritate_mri") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "veritate_mri"))

from runtime import heartbeat as hb
from runtime import sys_metrics as sm

# ------------------------------------------------------------------------------------
# Constants

KEY_A = "machine-key-A"
KEY_B = "machine-key-B"

# ------------------------------------------------------------------------------------
# Functions

def _setup(monkeypatch, tmp_path, key, name="hb.json"):
    """Point heartbeat state at a temp file and stub the hardware key + caches."""
    monkeypatch.setattr(hb, "STATE_PATH", str(tmp_path / name))
    monkeypatch.setattr(hb, "_STATE_CACHE", None)
    monkeypatch.setattr(hb, "_FINGERPRINT_CACHE", None)
    monkeypatch.setattr(sm, "stable_machine_key", lambda: key)


def _expected_id(fp):
    return hashlib.sha256(("machine|" + fp).encode("utf-8")).hexdigest()[:hb.MACHINE_ID_LEN]


def test_fresh_state_binds_id_to_fingerprint(monkeypatch, tmp_path):
    """Empty state derives a machine_id deterministically from the fingerprint."""
    _setup(monkeypatch, tmp_path, KEY_A)
    mid, host = hb._ensure_identity()
    assert mid == _expected_id(hb._hw_fingerprint())
    assert len(host) == hb.HOST_TOKEN_LEN


def test_legacy_id_is_grandfathered(monkeypatch, tmp_path):
    """An existing id with no fingerprint is kept (no churn) and stamped."""
    _setup(monkeypatch, tmp_path, KEY_A)
    hb._update_state({"machine_id": "6fa286b879b09b4d", "host_token": "ce0c45eae2c3"})
    hb._STATE_CACHE = None
    mid, _ = hb._ensure_identity()
    assert mid == "6fa286b879b09b4d"


def test_legacy_grandfather_stamps_fingerprint(monkeypatch, tmp_path):
    """Grandfathering writes the current fingerprint so future clones mismatch."""
    _setup(monkeypatch, tmp_path, KEY_A)
    hb._update_state({"machine_id": "6fa286b879b09b4d", "host_token": "ce0c45eae2c3"})
    hb._STATE_CACHE = None
    hb._ensure_identity()
    assert json.load(open(hb.STATE_PATH))["machine_fingerprint"] == hb._hw_fingerprint()


def test_copied_clone_regenerates(monkeypatch, tmp_path):
    """State carrying another box's id + fingerprint regenerates on this machine."""
    _setup(monkeypatch, tmp_path, KEY_A)
    hb._update_state({
        "machine_id": "6fa286b879b09b4d",
        "host_token": "ce0c45eae2c3",
        "machine_fingerprint": "deadbeefdeadbeef",
    })
    hb._STATE_CACHE = None
    mid, _ = hb._ensure_identity()
    assert mid == _expected_id(hb._hw_fingerprint())
    assert mid != "6fa286b879b09b4d"


def test_different_machines_get_different_ids(monkeypatch, tmp_path):
    """Two machines with different hardware keys derive different machine_ids."""
    _setup(monkeypatch, tmp_path, KEY_A, name="a.json")
    a, _ = hb._ensure_identity()
    _setup(monkeypatch, tmp_path, KEY_B, name="b.json")
    b, _ = hb._ensure_identity()
    assert a != b


def test_reconcile_identity_stable_when_matched(monkeypatch, tmp_path):
    """Once bound to this machine, repeated reconcile returns the same id."""
    _setup(monkeypatch, tmp_path, KEY_A)
    first = hb.reconcile_identity()
    hb._STATE_CACHE = None
    assert hb.reconcile_identity() == first
