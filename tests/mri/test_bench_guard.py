# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - trainer_runner.start refuses a bench-mode launch for trainers whose manifest
#   does not declare bench. A trainer without --bench would silently drop the
#   flag and start a real training run; the guard makes that impossible.
# tests/mri/test_bench_guard.py
# ------------------------------------------------------------------------------------
# Imports

import os
import sys

# ------------------------------------------------------------------------------------
# Constants

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MRI_DIR   = os.path.join(REPO_ROOT, "veritate_mri")
for p in (MRI_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

FAKE_PLUGIN = {"id": "fake/nobench", "path": "/dev/null", "manifest": {"kind": "trainer"}}

# ------------------------------------------------------------------------------------
# Functions


def test_bench_launch_refused_without_manifest_flag(monkeypatch):
    """start(bench=True) on a manifest without bench returns an error, no launch."""
    from training import trainer_runner as tr
    monkeypatch.setattr(tr.plugins_reader, "scan", lambda: [dict(FAKE_PLUGIN)])
    res = tr.start("fake/nobench", {"bench": True, "size": "10m"})
    assert res["ok"] is False
    assert "bench" in res["error"]
    assert tr.state()["status"] != tr.STATUS_RUNNING


def test_all_shipped_trainers_declare_bench():
    """Every scanned trainer manifest declares bench (full auto-tune coverage)."""
    from readers import trainers as plugins_reader
    missing = [p["id"] for p in plugins_reader.scan()
               if not (p.get("manifest") or {}).get("bench")]
    assert missing == [], f"trainers without bench: {missing}"
