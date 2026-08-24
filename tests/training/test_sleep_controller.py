# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - pins the sleep controller (training/sleep.py): usage-scaled dose clamping,
#   sleep-checkpoint pruning that never touches non-sleep checkpoints, recipe
#   reuse from the model's own config.json, the maybe_sleep gate chain, the
#   multi-model status payload, per-model experience isolation (a model never
#   trains on another model's exchanges), turn-taking on the one trainer,
#   per-model cooldowns, and migration from the single-sleeper settings/state.
# tests/training/test_sleep_controller.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import json
import os
import time

import pytest
from training import sleep

# ------------------------------------------------------------------------------------
# Constants

CFG = {
    "sleep_enabled": True, "sleep_models": ["toy"], "sleep_idle_min": 20,
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


def _rec(model, ts=None, prompt=b"user asks something here", output=b"a reply long enough to keep"):
    return json.dumps({"ts": ts if ts is not None else time.time(), "model": model,
                       "prompt_b64": base64.b64encode(prompt).decode("ascii"),
                       "output_b64": base64.b64encode(output).decode("ascii")})


def _roots(tmp_path, monkeypatch):
    monkeypatch.setattr(sleep, "EXPERIENCE_ROOT", str(tmp_path / "exp"))
    monkeypatch.setattr(sleep, "MODELS_ROOT", str(tmp_path / "models"))
    monkeypatch.setattr(sleep, "STATE_PATH", str(tmp_path / "sleep_state.json"))
    monkeypatch.setattr(sleep, "HISTORY_PATH", str(tmp_path / "history.jsonl"))
    monkeypatch.setattr(sleep, "FILTER_ROOT", str(tmp_path / "filtered"))
    monkeypatch.setattr(sleep, "_EXP_CACHE", {"key": None, "by_model": None})
    monkeypatch.setattr(sleep, "_ACT_CACHE", {"key": None, "hours": None})


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
    """Disabled, no-enrollment, unenrolled-model, busy-trainer, and
    no-experience states each stop the launch with a distinct reason and never
    call the trainer."""
    _roots(tmp_path, monkeypatch)
    launched = []
    monkeypatch.setattr(sleep.trainer_runner, "start", lambda *a: launched.append(a))
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: False)
    cfg = dict(CFG)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: cfg)

    cfg["sleep_enabled"] = False
    assert sleep.maybe_sleep() == "disabled"
    cfg["sleep_enabled"] = True
    cfg["sleep_models"] = []
    assert "sleep_models" in sleep.maybe_sleep()
    cfg["sleep_models"] = ["toy"]
    assert "not enrolled" in sleep.maybe_sleep(model="ghost")
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: True)
    assert sleep.maybe_sleep() == "trainer busy"
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: False)
    assert sleep.maybe_sleep() == "no experience yet"
    assert launched == []


def test_maybe_sleep_idle_gate_respects_recent_activity(tmp_path, monkeypatch):
    """A fresh exchange keeps the box awake; force_idle bypasses only the timer
    (and still stops later at the too-little-experience gate). The corpus build
    runs over the per-model filtered view, not the raw experience log."""
    _roots(tmp_path, monkeypatch)
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "20260820.jsonl").write_text(_rec("toy") + "\n" + _rec("toy") + "\n")
    d = _model(tmp_path)
    (d / "checkpoints" / "step_3000.pt").write_bytes(b"x")
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: False)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: CFG)
    assert sleep.maybe_sleep().startswith("awake:")
    calls = {}

    import tools.build_experience_corpus as bec

    def fake_build(days=None, **kw):
        calls["days"] = days
        calls["root"] = bec.EXPERIENCE_ROOT
        return 2, 100, 10

    monkeypatch.setattr(bec, "build", fake_build)
    out = sleep.maybe_sleep(force_idle=True, model="toy")
    assert "too little" in out
    assert calls["days"] is None and calls["root"] == sleep.FILTER_ROOT


def test_status_awake_shape(tmp_path, monkeypatch):
    """The awake status payload carries enabled/state, one row per enrolled
    model with its own pending count and countdown to sleep."""
    _roots(tmp_path, monkeypatch)
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "20260820.jsonl").write_text("\n".join(_rec("toy") for _ in range(3)) + "\n")
    _model(tmp_path)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: CFG)
    monkeypatch.setattr(sleep.trainer_runner, "state", dict)
    out = sleep.status()
    assert out["state"] == "awake" and out["enabled"]
    assert [m["name"] for m in out["models"]] == ["toy"]
    row = out["models"][0]
    assert row["state"] == "awake" and row["pending_exchanges"] == 3
    assert 0 <= row["sleeps_in_s"] <= 20 * 60
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


def _armed(tmp_path, monkeypatch, built=(20, 50000, 5000), models=("toy",),
           pending=(20,)):
    """Idle own-experience + real models with checkpoints + fake corpus build:
    every gate before launch passes unless a test breaks one on purpose."""
    _roots(tmp_path, monkeypatch)
    exp = tmp_path / "exp"
    exp.mkdir()
    old = time.time() - 40 * 60
    lines = []
    for name, n in zip(models, pending, strict=True):
        lines += [_rec(name, ts=old) for _ in range(n)]
    f = exp / "20260820.jsonl"
    f.write_text("\n".join(lines) + "\n")
    os.utime(str(f), (old, old))
    for name in models:
        d = _model(tmp_path, name=name)
        (d / "checkpoints" / "step_3000.pt").write_bytes(b"x")
    cfg = dict(CFG, sleep_models=list(models))
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: cfg)
    monkeypatch.setattr(sleep.trainer_runner, "is_running", lambda: False)
    import tools.build_experience_corpus as bec
    monkeypatch.setattr(bec, "build", lambda days=None, **kw: built)
    return cfg


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
    ms = sleep._load_state()["models"]["toy"]
    assert not ms.get("sleeping") and ms.get("cooldown_until", 0) > 0
    assert sleep.maybe_sleep(model="toy").startswith("cooling down")
    evs = sleep.history()
    assert evs[0]["event"] == "failed" and evs[1]["event"] == "sleep"


def test_activity_ledger_buckets_by_local_hour(tmp_path, monkeypatch):
    """The usage ledger buckets experience records into local hour-of-day and
    ignores records older than the window; garbage lines are skipped."""
    exp = tmp_path / "exp"
    exp.mkdir()
    now = time.time()
    old = now - 30 * 86400
    lines = [json.dumps({"ts": now}), json.dumps({"ts": now}),
             json.dumps({"ts": old}), "garbage"]
    (exp / "20260820.jsonl").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(sleep, "EXPERIENCE_ROOT", str(exp))
    monkeypatch.setattr(sleep, "_ACT_CACHE", {"key": None, "hours": None})
    hours = sleep.activity(days=7)
    assert len(hours) == 24 and sum(hours) == 2
    assert hours[time.localtime(now).tm_hour] == 2
    assert sleep.activity(days=7) is hours  # mtime-keyed cache hit


def test_wake_without_sleep_is_noop(tmp_path, monkeypatch):
    """Waking an awake model reports awake and never touches the trainer."""
    monkeypatch.setattr(sleep, "STATE_PATH", str(tmp_path / "sleep_state.json"))
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: CFG)
    monkeypatch.setattr(sleep.trainer_runner, "stop",
                        lambda: pytest.fail("stop called while awake"))
    out = sleep.wake("toy")
    assert out["ok"] and out["state"] == "awake"


def test_experience_filter_isolates_models(tmp_path, monkeypatch):
    """THE per-model property: a model's consolidation corpus contains only its
    own exchanges — never another model's, never a record whose bin-name model
    field is ambiguous, never a no-model record."""
    _roots(tmp_path, monkeypatch)
    exp = tmp_path / "exp"
    exp.mkdir()
    mine = _rec("alpha", output=b"ALPHA_OWN_REPLY_BYTES")
    other = _rec("beta", output=b"BETA_OWN_REPLY_BYTES")
    ambiguous = _rec("veritate.bin", output=b"AMBIGUOUS_REPLY_BYTES")  # both have bins
    nobody = _rec("", output=b"NO_MODEL_REPLY_BYTES")
    (exp / "20260820.jsonl").write_text("\n".join([mine, other, ambiguous, nobody]) + "\n")
    for name in ("alpha", "beta"):
        d = _model(tmp_path, name=name)
        (d / "veritate.bin").write_bytes(b"x")
    import tools.build_experience_corpus as bec
    monkeypatch.setattr(bec, "CORPUS_ROOT", str(tmp_path / "corpus"))
    n, _tb, _vb = sleep._build_own_corpus("alpha", CFG)
    assert n == 1
    blob = ((tmp_path / "corpus" / "experience_train.bin").read_bytes()
            + (tmp_path / "corpus" / "experience_val.bin").read_bytes())
    assert b"ALPHA_OWN_REPLY_BYTES" in blob
    for foreign in (b"BETA_OWN_REPLY_BYTES", b"AMBIGUOUS_REPLY_BYTES", b"NO_MODEL_REPLY_BYTES"):
        assert foreign not in blob


def test_bin_name_resolves_only_unique_owner(tmp_path, monkeypatch):
    """Old records name serving artifacts, not model dirs: a basename exactly
    one model owns attributes to it; a shared basename (veritate.bin under two
    models) or an unknown name attributes to nobody."""
    monkeypatch.setattr(sleep, "MODELS_ROOT", str(tmp_path / "models"))
    a = _model(tmp_path, name="alpha")
    (a / "veritate.bin").write_bytes(b"x")
    (a / "checkpoints" / "step_3000.pt").write_bytes(b"x")
    dirs = sleep._model_dirs()
    owners = sleep._owner_map(dirs)
    assert sleep._resolve("alpha", dirs, owners) == "alpha"
    assert sleep._resolve("veritate.bin", dirs, owners) == "alpha"   # unique owner
    assert sleep._resolve("step_3000.pt", dirs, owners) == "alpha"
    assert sleep._resolve("unknown_thing", dirs, owners) == ""
    b = _model(tmp_path, name="beta")
    (b / "veritate.bin").write_bytes(b"x")
    dirs = sleep._model_dirs()
    owners = sleep._owner_map(dirs)
    assert sleep._resolve("veritate.bin", dirs, owners) == ""        # now ambiguous


def test_turn_taking_fuller_queue_sleeps_first(tmp_path, monkeypatch):
    """Two enrolled models, one trainer: the model with more pending
    own-exchanges sleeps first; the second sleeps only after finalize."""
    _armed(tmp_path, monkeypatch, models=("quill", "toy"), pending=(12, 20))
    launched = []
    monkeypatch.setattr(sleep.trainer_runner, "start",
                        lambda pid, args: launched.append(args["name"]) or {"ok": True})
    assert sleep.maybe_sleep().startswith("sleeping: toy")
    assert launched == ["toy"]
    assert sleep.maybe_sleep().startswith("already sleeping: toy")
    # toy's run gains steps; finalize ends its turn
    (tmp_path / "models" / "toy" / "checkpoints" / "step_3050.pt").write_bytes(b"x")
    sleep.finalize()
    assert sleep.maybe_sleep().startswith("sleeping: quill")
    assert launched == ["toy", "quill"]


def test_cooldown_is_per_model(tmp_path, monkeypatch):
    """A failed sleep cools down only the model that failed; another enrolled
    model still sleeps in the same window."""
    _armed(tmp_path, monkeypatch, models=("quill", "toy"), pending=(12, 20))
    launched = []
    monkeypatch.setattr(sleep.trainer_runner, "start",
                        lambda pid, args: launched.append(args["name"]) or {"ok": True})
    assert sleep.maybe_sleep().startswith("sleeping: toy")
    sleep.finalize()  # no new checkpoint: toy failed -> cooldown
    assert sleep._load_state()["models"]["toy"].get("cooldown_until", 0) > 0
    assert sleep.maybe_sleep().startswith("sleeping: quill")
    assert launched == ["toy", "quill"]


def test_state_migrates_from_flat_single_sleeper(tmp_path, monkeypatch):
    """A flat pre-per-model state.json keeps its bookkeeping: last sleep,
    finals, and any in-flight run move under the model they belonged to."""
    _roots(tmp_path, monkeypatch)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: CFG)
    flat = {"last_sleep_ts": 123.0, "finals": [3100, 3200],
            "sleeping": True, "run": {"model": "toy", "start_step": 3200, "steps": 3250}}
    (tmp_path / "sleep_state.json").write_text(json.dumps(flat))
    st = sleep._load_state()
    ms = st["models"]["toy"]
    assert ms["last_sleep_ts"] == 123.0 and ms["finals"] == [3100, 3200]
    assert ms["sleeping"] and ms["run"]["start_step"] == 3200
    # no run recorded: the single enrolled model owns the bookkeeping
    (tmp_path / "sleep_state.json").write_text(json.dumps({"last_sleep_ts": 5.0}))
    assert sleep._load_state()["models"]["toy"]["last_sleep_ts"] == 5.0


def test_settings_migrate_sleep_model_to_enrollment(tmp_path, monkeypatch):
    """An install holding the old single sleep_model string comes up with that
    model enrolled in sleep_models, with no user action."""
    from runtime import settings as settings_mod
    live = tmp_path / "mri_settings.json"
    live.write_text(json.dumps({"sleep_enabled": True, "sleep_model": "quill"}))
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", str(live))
    monkeypatch.setattr(settings_mod, "_CACHE", None)
    cfg = settings_mod.get()
    assert cfg["sleep_models"] == ["quill"]
    on_disk = json.loads(live.read_text())
    assert on_disk["sleep_models"] == ["quill"] and "sleep_model" not in on_disk


class _FakeChild:
    """Stand-in for the trainer child: records suspend/resume calls."""

    def __init__(self):
        self.calls = []

    def suspend(self):
        self.calls.append("suspend")

    def resume(self):
        self.calls.append("resume")


def _sleeping_state(tmp_path, monkeypatch, child):
    _roots(tmp_path, monkeypatch)
    st = {"models": {"toy": {"sleeping": True, "run": {"model": "toy", "steps": 10}}}}
    monkeypatch.setattr(sleep, "_load_state", lambda: st)
    monkeypatch.setattr(sleep, "_child_proc", lambda: child)
    monkeypatch.setattr(sleep, "_PAUSE", {"suspended": False, "warned": False})
    return st


def test_yield_to_serving_suspends_the_sleep_child(tmp_path, monkeypatch):
    """A served request parks the in-flight sleep run."""
    child = _FakeChild()
    _sleeping_state(tmp_path, monkeypatch, child)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: {**CFG, "sleep_preempt": True})
    assert sleep.yield_to_serving() is True
    assert child.calls == ["suspend"]
    assert sleep.suspended() is True


def test_yield_to_serving_is_idempotent(tmp_path, monkeypatch):
    """Overlapping requests suspend the child once, not once each."""
    child = _FakeChild()
    _sleeping_state(tmp_path, monkeypatch, child)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: {**CFG, "sleep_preempt": True})
    sleep.yield_to_serving()
    sleep.yield_to_serving()
    assert child.calls == ["suspend"]


def test_preempt_off_leaves_the_child_running(tmp_path, monkeypatch):
    """sleep_preempt False keeps the pre-preemption behavior."""
    child = _FakeChild()
    _sleeping_state(tmp_path, monkeypatch, child)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: {**CFG, "sleep_preempt": False})
    assert sleep.yield_to_serving() is False
    assert child.calls == []


def test_resume_waits_for_the_quiet_window(tmp_path, monkeypatch):
    """A suspended child stays parked until serving has been quiet long enough."""
    child = _FakeChild()
    _sleeping_state(tmp_path, monkeypatch, child)
    monkeypatch.setattr(sleep.settings_mod, "get",
                        lambda: {**CFG, "sleep_preempt": True, "sleep_resume_s": 5})
    sleep.yield_to_serving()
    monkeypatch.setattr(sleep.serving, "idle_s", lambda: 1.0)
    assert sleep.resume_if_quiet() is False
    assert sleep.suspended() is True
    monkeypatch.setattr(sleep.serving, "idle_s", lambda: 9.0)
    assert sleep.resume_if_quiet() is True
    assert child.calls == ["suspend", "resume"]


def test_resume_holds_while_a_request_is_still_streaming(tmp_path, monkeypatch):
    """idle_s None (a live stream) must not resume the child."""
    child = _FakeChild()
    _sleeping_state(tmp_path, monkeypatch, child)
    monkeypatch.setattr(sleep.settings_mod, "get",
                        lambda: {**CFG, "sleep_preempt": True, "sleep_resume_s": 5})
    sleep.yield_to_serving()
    monkeypatch.setattr(sleep.serving, "idle_s", lambda: None)
    assert sleep.resume_if_quiet() is False
    assert child.calls == ["suspend"]


def test_launch_args_carry_cpu_budget_and_nice(tmp_path, monkeypatch):
    """The sleep recipe reserves cores and deprioritizes its child."""
    _roots(tmp_path, monkeypatch)
    _model(tmp_path)
    monkeypatch.setattr(sleep, "cpu_budget", lambda cfg: 7)
    args = sleep.launch_args("toy", 10, {**CFG, "sleep_nice": 10})
    assert args["_cpu_budget"] == 7
    assert args["_nice"] == 10


def test_sleep_batch_size_overrides_the_recipe(tmp_path, monkeypatch):
    """A weak box shrinks the step so it fits between interruptions."""
    _roots(tmp_path, monkeypatch)
    _model(tmp_path)
    monkeypatch.setattr(sleep, "cpu_budget", lambda cfg: 4)
    assert sleep.launch_args("toy", 10, {**CFG, "sleep_batch_size": 0})["batch_size"] == 48
    assert sleep.launch_args("toy", 10, {**CFG, "sleep_batch_size": 4})["batch_size"] == 4


def test_run_modifiers_never_reach_the_trainer_argv():
    """Underscore run modifiers are stripped before argv."""
    from training import trainer_runner
    argv = trainer_runner._build_argv({"path": "t.py"},
                                      {"name": "toy", "_cpu_budget": 7, "_nice": 10})
    assert "--_cpu_budget" not in argv and "7" not in argv
    assert "--name" in argv and "toy" in argv


def test_sleep_logs_every_step(tmp_path, monkeypatch):
    """A short dose must still write train.csv rows; an inherited log_every of 10
    leaves a 4-step sleep looking like nothing happened."""
    _roots(tmp_path, monkeypatch)
    _model(tmp_path, training_args={"name": "toy", "size": "10m", "seq": 1024,
                                    "batch_size": 48, "log_every": 10})
    monkeypatch.setattr(sleep, "cpu_budget", lambda cfg: 4)
    assert sleep.launch_args("toy", 4, CFG)["log_every"] == 1


def test_unpark_resumes_a_child_left_suspended_by_a_previous_process(tmp_path, monkeypatch):
    """Module state resets on restart, so a child stopped by the previous process
    would sit parked forever holding its memory."""
    class _Stopped(_FakeChild):
        def status(self):
            return "stopped"

    child = _Stopped()
    _roots(tmp_path, monkeypatch)
    monkeypatch.setattr(sleep, "_child_proc", lambda: child)
    assert sleep.unpark_orphan() is True
    assert child.calls == ["resume"]


def test_unpark_leaves_a_running_child_alone(tmp_path, monkeypatch):
    """A healthy child must not be touched on startup."""
    class _Running(_FakeChild):
        def status(self):
            return "running"

    child = _Running()
    _roots(tmp_path, monkeypatch)
    monkeypatch.setattr(sleep, "_child_proc", lambda: child)
    assert sleep.unpark_orphan() is False
    assert child.calls == []


def test_yield_does_not_touch_disk_when_sleep_is_off(tmp_path, monkeypatch):
    """The hot path runs on every generation; a box that never sleeps must not
    pay a state read for it."""
    child = _FakeChild()
    _sleeping_state(tmp_path, monkeypatch, child)
    reads = []
    monkeypatch.setattr(sleep, "_load_state", lambda: reads.append(1) or {})
    monkeypatch.setattr(sleep.settings_mod, "get",
                        lambda: {**CFG, "sleep_enabled": False, "sleep_preempt": True})
    assert sleep.yield_to_serving() is False
    assert reads == []
    assert child.calls == []


def test_status_reports_a_parked_run_as_suspended(tmp_path, monkeypatch):
    """The panel must distinguish a run that is training from one parked for a
    request, or a paused run reads as healthy progress."""
    _roots(tmp_path, monkeypatch)
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "20260820.jsonl").write_text(_rec("toy") + "\n")
    _model(tmp_path)
    st = {"models": {"toy": {"sleeping": True, "run": {"model": "toy", "steps": 10}}}}
    monkeypatch.setattr(sleep, "_load_state", lambda: st)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: CFG)
    monkeypatch.setattr(sleep.trainer_runner, "state", lambda: {"status": "running"})
    monkeypatch.setattr(sleep, "_train_progress", lambda m: (5, 10.0))

    monkeypatch.setattr(sleep, "_PAUSE", {"suspended": False, "warned": False})
    assert sleep.status()["suspended"] is False
    monkeypatch.setattr(sleep, "_PAUSE", {"suspended": True, "warned": False})
    assert sleep.status()["suspended"] is True


def test_status_is_never_suspended_while_awake(tmp_path, monkeypatch):
    """suspended is a property of an in-flight run, not a stale flag."""
    _roots(tmp_path, monkeypatch)
    (tmp_path / "exp").mkdir()
    _model(tmp_path)
    monkeypatch.setattr(sleep.settings_mod, "get", lambda: CFG)
    monkeypatch.setattr(sleep.trainer_runner, "state", dict)
    monkeypatch.setattr(sleep, "_PAUSE", {"suspended": True, "warned": False})
    assert sleep.status()["suspended"] is False
