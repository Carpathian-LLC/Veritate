# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Measured on cardinal-01: the checkout sat on `master`, a branch deleted upstream
#   when the repo renamed to `main`. The updater followed it anyway, GitHub served the
#   DEFAULT branch's tarball for the unresolvable ref (archive URLs redirect), and every
#   compare against `master` 404'd, which the caller reads as "up to date". The update
#   button was disabled forever on a box 40 commits behind, and the one pull that did
#   land wrote main's source while reporting "synced master".
# - The branch a box follows must therefore be resolved against the remote, and status()
#   must report it without a network call: the dashboard polls it once a minute per tab.
# tests/mri/test_updater_branch.py
# ------------------------------------------------------------------------------------
# Imports:

import urllib.error
import urllib.request

from training.sync import app_sync

# ------------------------------------------------------------------------------------
# Constants

SHA     = "a" * 40
DEAD    = "master"
CHANNEL = "dev"

# ------------------------------------------------------------------------------------
# Functions


def _http_error(code):
    def _urlopen(req, timeout=None, context=None):
        raise urllib.error.HTTPError(req.full_url, code, "", {}, None)
    return _urlopen


def _no_network(req, timeout=None, context=None):
    raise AssertionError("status() must not call the network")


def test_branch_deleted_upstream_falls_back_to_the_channel(monkeypatch):
    """A checkout on a branch the remote no longer has follows its channel instead."""
    monkeypatch.setattr(app_sync, "_local_git_branch", lambda: DEAD)
    monkeypatch.setattr(app_sync, "_channel_branch", lambda: CHANNEL)
    monkeypatch.setattr(app_sync, "_remote_branch_sha", lambda b: (None, True))
    assert app_sync._tracked_branch() == CHANNEL


def test_local_branch_the_remote_still_has_wins(monkeypatch):
    """A developer testing on a real branch keeps being updated from it."""
    monkeypatch.setattr(app_sync, "_local_git_branch", lambda: "experimental")
    monkeypatch.setattr(app_sync, "_channel_branch", lambda: CHANNEL)
    monkeypatch.setattr(app_sync, "_remote_branch_sha", lambda b: (SHA, False))
    assert app_sync._tracked_branch() == "experimental"


def test_unreachable_remote_keeps_the_local_branch(monkeypatch):
    """Offline is not evidence that a branch was deleted."""
    monkeypatch.setattr(app_sync, "_local_git_branch", lambda: DEAD)
    monkeypatch.setattr(app_sync, "_channel_branch", lambda: CHANNEL)
    monkeypatch.setattr(app_sync, "_remote_branch_sha", lambda b: (None, False))
    assert app_sync._tracked_branch() == DEAD


def test_unresolvable_ref_reads_as_missing(monkeypatch):
    """422 is what the commits endpoint answers for a branch the remote does not have."""
    monkeypatch.setattr(urllib.request, "urlopen", _http_error(422))
    assert app_sync._remote_branch_sha(DEAD) == (None, True)


def test_rate_limit_is_not_a_missing_branch(monkeypatch):
    """A quota answer must not retarget the box onto another branch."""
    monkeypatch.setattr(urllib.request, "urlopen", _http_error(403))
    assert app_sync._remote_branch_sha(CHANNEL) == (None, False)


def test_status_reports_the_resolved_branch_without_network(monkeypatch):
    """The panel shows the branch the last check settled on, not the local guess."""
    monkeypatch.setattr(app_sync, "_state", lambda: {"remote_branch": CHANNEL})
    monkeypatch.setattr(app_sync, "_local_git_branch", lambda: DEAD)
    monkeypatch.setattr(app_sync, "_local_head_sha", lambda: SHA)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    out = app_sync.status()
    assert out["branch"] == CHANNEL
    assert out["tracked_channel"] == "development"
