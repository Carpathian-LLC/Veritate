# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - /trainers/tune_defaults writes auto-tune results to the trainer manifest and
#   saved specs. Manifest-merge behavior is covered by update_defaults itself;
#   these cover the new specs-measured persistence and the route's input guard.
# tests/mri/test_tune_defaults.py
# ------------------------------------------------------------------------------------
# Imports

import json
import os
import sys

import pytest

# ------------------------------------------------------------------------------------
# Constants

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MRI_DIR   = os.path.join(REPO_ROOT, "veritate_mri")

MEASURED = {"device": "mps", "max_batch": 192, "mem_ceiling_gb": 19.0,
            "tok_per_s": 11736.0}

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture(scope="module")
def client():
    """Flask test client for the MRI app."""
    if MRI_DIR not in sys.path:
        sys.path.insert(0, MRI_DIR)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from veritate_mri.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_tune_defaults_requires_id(client):
    """POST /trainers/tune_defaults without id returns 400."""
    r = client.post("/trainers/tune_defaults", json={"args": {"batch_size": 4}})
    assert r.status_code == 400


def test_save_measured_writes_specs(tmp_path, monkeypatch):
    """save_measured merges a measured block into the specs file."""
    from runtime import sys_metrics
    monkeypatch.setattr(sys_metrics, "SPECS_PATH", str(tmp_path / "specs.json"))
    sys_metrics.save_specs({"platform": "test"})
    out = sys_metrics.save_measured(MEASURED)
    assert out["measured"]["max_batch"] == 192
    on_disk = json.loads((tmp_path / "specs.json").read_text())
    assert on_disk["platform"] == "test"
    assert on_disk["measured"] == MEASURED


def test_save_measured_rejects_non_dict(tmp_path, monkeypatch):
    """save_measured ignores a non-dict payload."""
    from runtime import sys_metrics
    monkeypatch.setattr(sys_metrics, "SPECS_PATH", str(tmp_path / "specs.json"))
    assert sys_metrics.save_measured("junk") is None
    assert not (tmp_path / "specs.json").exists()


def test_detect_and_save_preserves_measured(tmp_path, monkeypatch):
    """Re-detecting specs keeps the previously measured block."""
    from runtime import sys_metrics
    monkeypatch.setattr(sys_metrics, "SPECS_PATH", str(tmp_path / "specs.json"))
    sys_metrics.save_specs({"platform": "test", "measured": MEASURED})
    out = sys_metrics.detect_and_save()
    assert out["measured"] == MEASURED
