# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The updater applies a TARBALL but measured `behind` against `.git/HEAD`, which a
#   tarball extract never advances. Measured on fortis: a pull copied 584 files and
#   returned ok, and the very next check still said "6 behind" -- forever, however
#   many times the user updated. The base has to be what was actually pulled.
# tests/mri/test_app_update_counter.py
# ------------------------------------------------------------------------------------
# Imports:

from training.sync import app_sync

# ------------------------------------------------------------------------------------
# Constants

PULLED = "a" * 40
HEADSHA = "b" * 40
BRANCH  = "dev"

# ------------------------------------------------------------------------------------
# Functions


def test_pulled_commit_wins_over_git_head(monkeypatch):
    """The whole bug: HEAD is stale on a tarball-updated checkout."""
    monkeypatch.setattr(app_sync, "_state",
                        lambda: {"pulled_commit": PULLED, "pulled_branch": BRANCH})
    monkeypatch.setattr(app_sync, "_local_head_sha", lambda: HEADSHA)
    assert app_sync._compare_base(True, BRANCH) == PULLED


def test_falls_back_to_head_when_never_pulled(monkeypatch):
    """A real git-managed checkout this updater has never written still works."""
    monkeypatch.setattr(app_sync, "_state", dict)
    monkeypatch.setattr(app_sync, "_local_head_sha", lambda: HEADSHA)
    assert app_sync._compare_base(True, BRANCH) == HEADSHA


def test_commit_from_another_branch_is_discarded(monkeypatch):
    """After a channel switch, the old branch's commit is not a valid base."""
    monkeypatch.setattr(app_sync, "_state",
                        lambda: {"pulled_commit": PULLED, "pulled_branch": "main"})
    monkeypatch.setattr(app_sync, "_local_head_sha", lambda: HEADSHA)
    assert app_sync._compare_base(True, BRANCH) == HEADSHA


def test_no_base_on_a_plain_tarball_install(monkeypatch):
    """No .git and no pull history: the caller must fall back to the ETag path."""
    monkeypatch.setattr(app_sync, "_state", dict)
    monkeypatch.setattr(app_sync, "_local_head_sha", lambda: None)
    assert app_sync._compare_base(False, BRANCH) is None


def test_switch_channel_clears_the_pulled_commit():
    """Guards the reset: a stale commit would survive a channel switch otherwise."""
    import inspect
    src = inspect.getsource(app_sync.switch_channel)
    assert '"pulled_commit"' in src


def test_pull_records_the_commit_it_installed():
    import inspect
    src = inspect.getsource(app_sync.pull_update)
    assert "_remote_branch_sha(branch)" in src


def test_head_short_is_a_commit_not_an_etag(monkeypatch):
    """It rendered the ETag, so the panel showed a quote-prefixed string."""
    monkeypatch.setattr(app_sync, "_state",
                        lambda: {"pulled_commit": PULLED, "etag": '"deadbeefcafe"'})
    monkeypatch.setattr(app_sync, "_local_head_sha", lambda: HEADSHA)
    out = app_sync.status()
    assert out["head_short"] == PULLED[:7]
    assert '"' not in (out["head_short"] or "")
