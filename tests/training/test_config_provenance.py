# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - _ensure_config writes config.json only when absent, so a fork plus resume (every
#   capability SFT) left the FORK SOURCE's corpus mix and schedule in the model forever:
#   the checkpoint advanced, the config lied. _sync_run_args records the live run's args
#   instead. model_type is excluded on purpose, so hand-editing a mislaunched run's type
#   (rule 24a) still survives the next checkpoint.
# tests/training/test_config_provenance.py
# ------------------------------------------------------------------------------------
# Imports:

import json

from training.save import RUN_ARG_KEYS, _sync_run_args

# ------------------------------------------------------------------------------------
# Constants

BASE_ARGS = {
    "resume": "", "corpus": "fineweb_edu:0.45,chat_500mb:0.20", "lr_schedule": "wsd",
    "base_lr": 0.0003, "batch_size": 24, "total_steps": 63500, "model_type": "language",
}
SFT_ARGS = {
    "resume": "wren_sft", "corpus": "instruct:0.15,fineweb_edu:0.42", "lr_schedule": "constant",
    "base_lr": 2e-05, "batch_size": 32, "total_steps": 61125, "loss_mask": "assistant",
}

# ------------------------------------------------------------------------------------
# Functions


def _cfg(tmp_path, monkeypatch, training_args):
    from readers import paths
    d = tmp_path / "m"
    d.mkdir()
    p = d / "config.json"
    p.write_text(json.dumps({"name": "m", "training_args": dict(training_args)}), encoding="utf-8")
    monkeypatch.setattr(paths, "config_path", lambda _n: str(p))
    return p


def test_the_resumed_runs_corpus_replaces_the_inherited_one(tmp_path, monkeypatch):
    """The defect: an SFT'd model kept advertising its base's pretrain mix."""
    p = _cfg(tmp_path, monkeypatch, BASE_ARGS)
    _sync_run_args("m", SFT_ARGS)
    ta = json.loads(p.read_text())["training_args"]
    assert ta["corpus"] == "instruct:0.15,fineweb_edu:0.42"


def test_the_schedule_and_lr_are_recorded_from_the_live_run(tmp_path, monkeypatch):
    p = _cfg(tmp_path, monkeypatch, BASE_ARGS)
    _sync_run_args("m", SFT_ARGS)
    ta = json.loads(p.read_text())["training_args"]
    assert (ta["lr_schedule"], ta["base_lr"]) == ("constant", 2e-05)


def test_a_key_only_the_live_run_sets_is_added(tmp_path, monkeypatch):
    """loss_mask is absent from a pretrain config; the SFT must record it."""
    p = _cfg(tmp_path, monkeypatch, BASE_ARGS)
    _sync_run_args("m", SFT_ARGS)
    assert json.loads(p.read_text())["training_args"]["loss_mask"] == "assistant"


def test_model_type_is_never_overwritten(tmp_path, monkeypatch):
    """Rule 24a: correcting a mislaunched run by hand-editing model_type must survive."""
    p = _cfg(tmp_path, monkeypatch, dict(BASE_ARGS, model_type="code"))
    _sync_run_args("m", dict(SFT_ARGS, model_type="language"))
    assert json.loads(p.read_text())["training_args"]["model_type"] == "code"
    assert "model_type" not in RUN_ARG_KEYS


def test_keys_outside_the_run_arg_set_are_left_alone(tmp_path, monkeypatch):
    p = _cfg(tmp_path, monkeypatch, dict(BASE_ARGS, description="hand written"))
    _sync_run_args("m", SFT_ARGS)
    assert json.loads(p.read_text())["training_args"]["description"] == "hand written"


def test_a_missing_config_is_not_created(tmp_path, monkeypatch):
    """_ensure_config owns bootstrapping; this must never write a config of its own."""
    from readers import paths
    p = tmp_path / "gone" / "config.json"
    monkeypatch.setattr(paths, "config_path", lambda _n: str(p))
    _sync_run_args("m", SFT_ARGS)
    assert not p.exists()


def test_a_non_dict_args_is_ignored(tmp_path, monkeypatch):
    p = _cfg(tmp_path, monkeypatch, BASE_ARGS)
    _sync_run_args("m", None)
    assert json.loads(p.read_text())["training_args"]["corpus"] == BASE_ARGS["corpus"]


def test_an_unchanged_run_leaves_the_file_untouched(tmp_path, monkeypatch):
    p = _cfg(tmp_path, monkeypatch, BASE_ARGS)
    before = p.stat().st_mtime_ns
    _sync_run_args("m", {k: BASE_ARGS[k] for k in BASE_ARGS if k in RUN_ARG_KEYS})
    assert p.stat().st_mtime_ns == before


# ------------------------------------------------------------------------------------
# Cadence, optimization shape, seed and memory regime. Measured gap: a resumed SFT
# checkpointed every 250 steps while its config still reported the fork source's 1500.

def test_ckpt_cadence_is_recorded_from_the_live_run(tmp_path, monkeypatch):
    p = _cfg(tmp_path, monkeypatch, dict(BASE_ARGS, ckpt_every=1500))
    _sync_run_args("m", dict(SFT_ARGS, ckpt_every=250))
    assert json.loads(p.read_text())["training_args"]["ckpt_every"] == 250


def test_seed_is_recorded_from_the_live_run(tmp_path, monkeypatch):
    """A rerun on the wrong seed is silently non-comparable."""
    p = _cfg(tmp_path, monkeypatch, dict(BASE_ARGS, seed=0))
    _sync_run_args("m", dict(SFT_ARGS, seed=7))
    assert json.loads(p.read_text())["training_args"]["seed"] == 7


def test_optimization_shape_is_recorded_from_the_live_run(tmp_path, monkeypatch):
    base = dict(BASE_ARGS, weight_decay=0.1, label_smoothing=0.0, grad_clip=1.0,
                beta1=0.9, beta2=0.95)
    p = _cfg(tmp_path, monkeypatch, base)
    _sync_run_args("m", dict(SFT_ARGS, weight_decay=0.01, label_smoothing=0.05,
                             grad_clip=0.5, beta1=0.85, beta2=0.99))
    ta = json.loads(p.read_text())["training_args"]
    assert (ta["weight_decay"], ta["label_smoothing"], ta["grad_clip"],
            ta["beta1"], ta["beta2"]) == (0.01, 0.05, 0.5, 0.85, 0.99)


def test_memory_regime_is_recorded_from_the_live_run(tmp_path, monkeypatch):
    p = _cfg(tmp_path, monkeypatch, dict(BASE_ARGS, use_act_ckpt=True, use_8bit_adam=True))
    _sync_run_args("m", dict(SFT_ARGS, use_act_ckpt=False, use_8bit_adam=False))
    ta = json.loads(p.read_text())["training_args"]
    assert ta["use_act_ckpt"] is False and ta["use_8bit_adam"] is False


def test_eval_cadence_is_recorded_from_the_live_run(tmp_path, monkeypatch):
    p = _cfg(tmp_path, monkeypatch, dict(BASE_ARGS, eval_every=500, eval_iters=16,
                                         log_every=25))
    _sync_run_args("m", dict(SFT_ARGS, eval_every=125, eval_iters=32, log_every=10))
    ta = json.loads(p.read_text())["training_args"]
    assert (ta["eval_every"], ta["eval_iters"], ta["log_every"]) == (125, 32, 10)


def test_hand_owned_keys_stay_out_of_the_synced_set():
    """model_type is rule 24a's hand-edit workflow; description is hand written."""
    assert "model_type" not in RUN_ARG_KEYS
    assert "description" not in RUN_ARG_KEYS
    assert "name" not in RUN_ARG_KEYS
