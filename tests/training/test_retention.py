# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Checkpoint retention. A 200M pretrain at ckpt_every 500 had laid down 158
#   checkpoints / 239 GB by step 79k of 164k; the ladder frees the middle without
#   touching the hooks dumps, which are the artifacts that make a passed step
#   analyzable once its weights are gone.
# - The guards under test are the destructive ones: hooks survive, the newest N
#   survive, a recent file survives even off-ladder, and keep_last 0 is refused.
# tests/training/test_retention.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import time

import pytest
from readers import checkpoints, paths
from training import retention

# ------------------------------------------------------------------------------------
# Constants

MODEL      = "toy_retain"
CKPT_BYTES = b"x" * 64
STEPS      = list(range(500, 10_001, 500))     # 20 checkpoints, milestones at 5000/10000
OLD_MTIME_S = 86_400

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture
def model_root(tmp_path, monkeypatch):
    """A model with 20 checkpoints and a hooks dump per step, all aged past the guard."""
    monkeypatch.setattr(paths, "MODELS_ROOT", str(tmp_path / "models"))
    os.makedirs(paths.checkpoints_dir(MODEL))
    with open(paths.config_path(MODEL), "w") as f:
        f.write("{}")
    old = time.time() - OLD_MTIME_S
    for step in STEPS:
        path = checkpoints.path_for(MODEL, step)
        with open(path, "wb") as f:
            f.write(CKPT_BYTES)
        os.utime(path, (old, old))
        os.makedirs(paths.hook_step_dir(MODEL, step))
        with open(paths.hook_artifact_path(MODEL, step, "probe"), "w") as f:
            f.write("{}")
    return tmp_path


def test_plan_reads_only(model_root):
    """plan() must never delete: it is what the dashboard renders before committing."""
    retention.plan(MODEL, keep_every=5000, keep_last=2)
    assert checkpoints.list_steps(MODEL) == STEPS


def test_ladder_and_newest_survive(model_root):
    p = retention.plan(MODEL, keep_every=5000, keep_last=2)
    assert sorted(r["step"] for r in p["keep"]) == [5000, 9500, 10_000]
    assert len(p["delete"]) == len(STEPS) - 3


def test_prune_removes_exactly_the_planned_steps(model_root):
    p = retention.prune(MODEL, keep_every=5000, keep_last=2)
    assert checkpoints.list_steps(MODEL) == [5000, 9500, 10_000]
    assert p["remaining"] == 3
    assert p["freed_bytes"] == len(CKPT_BYTES) * (len(STEPS) - 3)
    assert p["failed"] == []


def test_hooks_are_never_touched(model_root):
    """The dump suite is the research artifact; pruning weights must not cost it."""
    retention.prune(MODEL, keep_every=5000, keep_last=1)
    kept = sorted(int(paths.HOOK_STEP_RE.match(e).group(1))
                  for e in os.listdir(paths.hooks_dir(MODEL)))
    assert kept == STEPS


def test_recent_checkpoints_survive_even_off_ladder(model_root):
    """A live trainer may still be renaming step_<N>.pt.tmp into place."""
    fresh = 3000
    os.utime(checkpoints.path_for(MODEL, fresh), None)
    p = retention.plan(MODEL, keep_every=5000, keep_last=1)
    assert fresh in [r["step"] for r in p["keep"]]
    assert fresh not in [r["step"] for r in p["delete"]]


def test_min_age_zero_disables_the_recency_guard(model_root):
    fresh = 3000
    os.utime(checkpoints.path_for(MODEL, fresh), None)
    p = retention.plan(MODEL, keep_every=5000, keep_last=1, min_age_s=0)
    assert fresh in [r["step"] for r in p["delete"]]


def test_keep_last_zero_is_refused(model_root):
    """Pruning a model to zero checkpoints destroys it; there is no valid reason."""
    with pytest.raises(retention.RetentionError):
        retention.plan(MODEL, keep_every=5000, keep_last=0)


def test_keep_every_zero_keeps_only_the_newest(model_root):
    p = retention.plan(MODEL, keep_every=0, keep_last=3)
    assert sorted(r["step"] for r in p["keep"]) == [9000, 9500, 10_000]


def test_unknown_model_is_refused(model_root):
    with pytest.raises(retention.RetentionError):
        retention.plan("no_such_model")


def test_invalid_name_is_refused(model_root):
    with pytest.raises(retention.RetentionError):
        retention.plan("../../etc")


def test_every_kept_row_states_why(model_root):
    p = retention.plan(MODEL, keep_every=5000, keep_last=2)
    assert all(r.get("reason") for r in p["keep"])


def test_plan_counts_hooks_it_is_preserving(model_root):
    p = retention.plan(MODEL, keep_every=5000, keep_last=2)
    assert p["hooks_kept"] == len(STEPS)
