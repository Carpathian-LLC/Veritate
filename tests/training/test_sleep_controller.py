# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - pins the sleep controller (training/sleep.py): usage-scaled dose clamping,
#   sleep-checkpoint pruning that never touches non-sleep checkpoints, recipe
#   reuse from the model's own config.json, the maybe_sleep gate chain, and the
#   awake status payload.
# tests/training/test_sleep_controller.py
# ------------------------------------------------------------------------------------
# Imports:

import json

import pytest
from training import sleep

# ------------------------------------------------------------------------------------
# Constants

CFG = {
    "sleep_enabled": True, "sleep_model": "toy", "sleep_idle_min": 20,
    "sleep_days": 3, "sleep_min_exchanges": 8, "sleep_steps_per_exchange": 10,
    "sleep_min_steps": 50, "sleep_max_steps": 500, "sleep_ckpt_every": 25,
    "sleep_keep_finals": 3, "sleep_lr": 5e-06,
    "sleep_corpus": "experience:0.75,mixed_chat:0.25",
}

# ------------------------------------------------------------------------------------
# Functions


def _model(tmp_path, name="toy", step=3000, training_args=None):
    d = tmp_path / "models" / name
    (d / "checkpoints").mkdir(parents=True)
    cfg = {"step": step, "training_args": training_args if training_args is not None else {
        "name": name, "size": "10m", "trunk": "hybrid", "total_steps": step,
        "base_lr": 2e-05, "min_lr": 2e-06, "warmup_steps": 100, "corpus": "hansard:1.0",
        "ckpt_every": 200, "eval_every": 200, "seq": 1024, "batch_size": 48,
    }}
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def test_dose_clamps_to_min_and_max():
    """Dose scales with exchanges and clamps to [min_steps, max_steps]."""
    assert sleep.dose_steps(0, CFG) == 50
    assert sleep.dose_steps(12, CFG) == 120
    assert sleep.dose_steps(10_000, CFG) == 500


def test_prune_deletes_intermediates_keeps_final_and_others(tmp_path, monkeypatch):
    """Pruning removes only sleep intermediates between start and end; the run
    final, pre-sleep checkpoints, and unrelated files survive."""
    d = _model(tmp_path)
    monkeypatch.setattr(sleep, "MODELS_ROOT", str(tmp_path / "models"))
    for n in (3000, 3025, 3050, 3075, 3100):
        (d / "checkpoints" / f"step_{n}.pt").write_bytes(b"x")
    (d / "checkpoints" / "step_2000.pt").write_bytes(b"x")
    finals = sleep.prune("toy", 3000, 3100, keep_finals=3, finals=[])
    left = sorted(p.name for p in (d / "checkpoints").iterdir())
    assert left == ["step_2000.pt", "step_3000.pt", "step_3100.pt"]
    assert finals == [3100]


def test_prune_thins_old_sleep_finals(tmp_path, monkeypatch):
    """Recorded sleep finals beyond keep_finals are deleted oldest-first; the
    newest always survives."""
    d = _model(tmp_path)
    monkeypatch.setattr(sleep, "MODELS_ROOT", str(tmp_path / "models"))
    for n in (3100, 3200, 3300, 3400):
        (d / "checkpoints" / f"step_{n}.pt").write_bytes(b"x")
    finals = sleep.prune("toy", 3300, 3400, keep_finals=2, finals=[3100, 3200, 3300])
    left = sorted(p.name for p in (d / "checkpoints").iterdir())
    assert left == ["step_3300.pt", "step_3400.pt"]
    assert finals == [3300, 3400]


def test_launch_args_reuses_recipe_with_sleep_levers(tmp_path, monkeypatch):
    """The sleep launch reuses the model's own training_args and overrides only
    the sleep levers (constant low LR, no warmup, dose steps, dense ckpts)."""
    _model(tmp_path)
    monkeypatch.setattr(sleep, "MODELS_ROOT", str(tmp_path / "models"))
    args = sleep.launch_args("toy", 3120, CFG)
    assert args["resume"] == "toy" and args["name"] == "toy"
    assert args["total_steps"] == 3120 and args["ckpt_every"] == 25
    assert args["base_lr"] == args["min_lr"] == 5e-06 and args["warmup_steps"] == 0
    assert args["corpus"] == CFG["sleep_corpus"]
    assert args["trunk"] == "hybrid" and args["seq"] == 1024  # recipe preserved
    assert sleep.launch_args("missing", 100, CFG) is None


def test_launch_args_strips_save_bookkeeping(tmp_path, monkeypatch):
    """save() stamps corpus_bytes/corpus_sha256/output_dir into training_args;
    forwarding them trips the trainer's unknown-flag gate (cardinal 2026-08-20),
    so launch_args must drop them."""
    ta = {"size": "10m", "trunk": "hybrid", "seq": 1024, "batch_size": 48,
          "corpus_bytes": 123456, "corpus_sha256": "ab" * 32,
          "output_dir": "/somewhere/models/toy"}
    _model(tmp_path, training_args=ta)
    monkeypatch.setattr(sleep, "MODELS_ROOT", str(tmp_path / "models"))
    args = sleep.launch_args("toy", 3015, CFG)
    for k in sleep.SAVE_BOOKKEEPING:
        assert k not in args
    assert args["trunk"] == "hybrid"  # real recipe keys survive


def test_model_step_reads_latest_checkpoint_not_config(tmp_path, monkeypatch):
    """The resume step is the latest checkpoint on disk; config.json "step" can
    lie on forked models (says 0 or a stale fork step). No checkpoint -> None:
    consolidation is impossible without a .pt to resume."""
    d = _model(tmp_path, step=0)  # config claims step 0
    monkeypatch.setattr(sleep, "MODELS_ROOT", str(tmp_path / "models"))
    assert sleep._model_step("toy") is None  # no checkpoints yet
    for n in (1000, 3000):
        (d / "checkpoints" / f"step_{n}.pt").write_bytes(b"x")
    assert sleep._model_step("toy") == 3000


def test_maybe_sleep_gate_chain(tmp_path, monkeypatch):
    """Disabled, unconfigured, busy-trainer, and no-experience states each stop
    the launch with a distinct reason and never call the trainer."""
    monkeypatch.setattr(sleep, "EXPERIENCE_ROOT", str(tmp_path / "exp"))
    monkeypatch.setattr(sleep, "STATE_PATH", str(tmp_path / "sleep_state.json"))
    launched = []
    monkeypatch.setattr(sleep.trainer_runner, "start", lambda *a: launched.append(a))
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: False)
    cfg = dict(CFG)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: cfg)

    cfg["sleep_enabled"] = False
    assert sleep.maybe_sleep() == "disabled"
    cfg["sleep_enabled"] = True
    cfg["sleep_model"] = ""
    assert "sleep_model" in sleep.maybe_sleep()
    cfg["sleep_model"] = "toy"
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: True)
    assert sleep.maybe_sleep() == "trainer busy"
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: False)
    assert sleep.maybe_sleep() == "no experience yet"
    assert launched == []


def test_maybe_sleep_idle_gate_respects_recent_activity(tmp_path, monkeypatch):
    """A fresh exchange keeps the model awake; force_idle bypasses only the
    timer (and still stops later at the too-little-experience gate)."""
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "20260820.jsonl").write_text("{}\n")
    monkeypatch.setattr(sleep, "EXPERIENCE_ROOT", str(exp))
    monkeypatch.setattr(sleep, "STATE_PATH", str(tmp_path / "sleep_state.json"))
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: False)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: CFG)
    assert sleep.maybe_sleep().startswith("awake:")
    calls = {}

    def fake_build(days=None, **kw):
        calls["days"] = days
        return 2, 100, 10

    import tools.build_experience_corpus as bec
    monkeypatch.setattr(bec, "build", fake_build)
    out = sleep.maybe_sleep(force_idle=True)
    assert "too little" in out and calls["days"] == 3


def test_status_awake_shape(tmp_path, monkeypatch):
    """The awake status payload carries enabled/model/state, pending exchange
    count, and the countdown to sleep."""
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "20260820.jsonl").write_text("{}\n{}\n{}\n")
    monkeypatch.setattr(sleep, "EXPERIENCE_ROOT", str(exp))
    monkeypatch.setattr(sleep, "STATE_PATH", str(tmp_path / "sleep_state.json"))
    monkeypatch.setattr(sleep, "HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: CFG)
    monkeypatch.setattr(sleep.trainer_runner, "state", dict)
    out = sleep.status()
    assert out["state"] == "awake" and out["enabled"] and out["model"] == "toy"
    assert out["pending_exchanges"] == 3
    assert 0 <= out["sleeps_in_s"] <= 20 * 60
    assert out["history"] == []


def test_history_records_state_changes_newest_first(tmp_path, monkeypatch):
    """State changes land in history.jsonl and read back newest-first; garbage
    lines are skipped rather than breaking the panel."""
    monkeypatch.setattr(sleep, "SLEEP_ROOT", str(tmp_path))
    monkeypatch.setattr(sleep, "HISTORY_PATH", str(tmp_path / "history.jsonl"))
    sleep._log_event("sleep", model="toy", exchanges=9, steps=90,
                     start_step=3000, target_step=3090)
    (tmp_path / "history.jsonl").open("a").write("not json\n")
    sleep._log_event("awake", model="toy", end_step=3090, steps_gained=90, finals=[3090])
    evs = sleep.history()
    assert [e["event"] for e in evs] == ["awake", "sleep"]
    assert evs[0]["steps_gained"] == 90 and evs[1]["target_step"] == 3090
    assert all("ts" in e for e in evs)
    assert sleep.history(limit=1) == [evs[0]]


def _armed(tmp_path, monkeypatch, built=(20, 50000, 5000)):
    """Idle experience + real model with checkpoint + fake corpus build: every
    gate before launch passes unless a test breaks one on purpose."""
    import time as _t
    exp = tmp_path / "exp"
    exp.mkdir()
    f = exp / "20260820.jsonl"
    f.write_text("{}\n" * 20)
    old = _t.time() - 40 * 60
    os_utime = __import__("os").utime
    os_utime(str(f), (old, old))
    d = _model(tmp_path)
    (d / "checkpoints" / "step_3000.pt").write_bytes(b"x")
    monkeypatch.setattr(sleep, "EXPERIENCE_ROOT", str(exp))
    monkeypatch.setattr(sleep, "MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(sleep, "STATE_PATH", str(tmp_path / "sleep_state.json"))
    monkeypatch.setattr(sleep, "HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: CFG)
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: False)
    import tools.build_experience_corpus as bec
    monkeypatch.setattr(bec, "build", lambda days=None, **kw: built)


def test_maybe_sleep_gates_small_bins_before_launch(tmp_path, monkeypatch):
    """A bin smaller than one draw window (seq*n_chunks+2) crashes the trainer
    child, so maybe_sleep refuses to launch on it (cardinal: 183 B val vs a
    4098 B window). CFG recipe has no n_chunks -> window = 1024+2."""
    _armed(tmp_path, monkeypatch, built=(20, 50000, 183))
    launched = []
    monkeypatch.setattr(sleep.trainer_runner, "start", lambda *a: launched.append(a) or {"ok": True})
    out = sleep.maybe_sleep()
    assert "too small for draw window" in out and launched == []
    monkeypatch.setattr(__import__("tools.build_experience_corpus", fromlist=["build"]),
                        "build", lambda days=None, **kw: (20, 50000, 5000))
    assert sleep.maybe_sleep().startswith("sleeping:")
    assert len(launched) == 1


def test_failed_sleep_sets_cooldown_and_blocks_retry(tmp_path, monkeypatch):
    """A sleep that gains no steps (failed launch / instant stop) records a
    "failed" event and a cooldown that blocks the watcher's next attempts —
    without it the 60 s tick retry-storms the trainer (cardinal: 3 failed
    launches in 11 minutes)."""
    _armed(tmp_path, monkeypatch)
    monkeypatch.setattr(sleep.trainer_runner, "start", lambda *a: {"ok": True})
    assert sleep.maybe_sleep().startswith("sleeping:")
    sleep.finalize()  # trainer died before any checkpoint: end == start
    st = sleep._load_state()
    assert not st.get("sleeping") and st.get("cooldown_until", 0) > 0
    assert sleep.maybe_sleep().startswith("cooling down")
    evs = sleep.history()
    assert evs[0]["event"] == "failed" and evs[1]["event"] == "sleep"


def test_activity_ledger_buckets_by_local_hour(tmp_path, monkeypatch):
    """The usage ledger buckets experience records into local hour-of-day and
    ignores records older than the window; garbage lines are skipped."""
    import time as _t
    exp = tmp_path / "exp"
    exp.mkdir()
    now = _t.time()
    old = now - 30 * 86400
    lines = [json.dumps({"ts": now}), json.dumps({"ts": now}),
             json.dumps({"ts": old}), "garbage"]
    (exp / "20260820.jsonl").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(sleep, "EXPERIENCE_ROOT", str(exp))
    monkeypatch.setattr(sleep, "_ACT_CACHE", {"key": None, "hours": None})
    hours = sleep.activity(days=7)
    assert len(hours) == 24 and sum(hours) == 2
    assert hours[_t.localtime(now).tm_hour] == 2
    assert sleep.activity(days=7) is hours  # mtime-keyed cache hit


def test_wake_without_sleep_is_noop(tmp_path, monkeypatch):
    """Waking an awake model reports awake and never touches the trainer."""
    monkeypatch.setattr(sleep, "STATE_PATH", str(tmp_path / "sleep_state.json"))
    monkeypatch.setattr(sleep.trainer_runner, "stop",
                        lambda: pytest.fail("stop called while awake"))
    out = sleep.wake()
    assert out["ok"] and out["state"] == "awake"
