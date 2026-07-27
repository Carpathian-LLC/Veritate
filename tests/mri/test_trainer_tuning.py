# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - machine-local trainer tuning: round-trip store, update_defaults filtering +
#   coercion (writes the local store, never the manifest), and the scan overlay
#   that prefills the form from a machine's benchmarked settings.
# tests/mri/test_trainer_tuning.py
# ------------------------------------------------------------------------------------
# Imports:



from readers import trainer_tuning, trainers

# ------------------------------------------------------------------------------------
# Functions

def test_save_and_args_for_roundtrip(tmp_path, monkeypatch):
    """save() persists args; args_for() reads them back per plugin_id."""
    monkeypatch.setattr(trainer_tuning, "TUNING_PATH", str(tmp_path / "t.json"))
    assert trainer_tuning.args_for("veritate_10m") == {}
    assert trainer_tuning.save("veritate_10m", {"batch_size": 4}) is True
    assert trainer_tuning.args_for("veritate_10m") == {"batch_size": 4}


def test_save_merges_and_reports_no_change(tmp_path, monkeypatch):
    """save() merges new keys and returns False when nothing changed."""
    monkeypatch.setattr(trainer_tuning, "TUNING_PATH", str(tmp_path / "t.json"))
    trainer_tuning.save("p", {"a": 1})
    assert trainer_tuning.save("p", {"b": 2}) is True
    assert trainer_tuning.args_for("p") == {"a": 1, "b": 2}
    assert trainer_tuning.save("p", {"b": 2}) is False


def test_update_defaults_filters_run_only_keys_and_coerces(tmp_path, monkeypatch):
    """update_defaults keeps only manifest-default keys, coerces to the default's
    type, and writes the local tuning store."""
    monkeypatch.setattr(trainer_tuning, "TUNING_PATH", str(tmp_path / "t.json"))
    fake = {"id": "p", "manifest": {"defaults": {"batch_size": 32, "precision": "bf16", "qat_enabled": True}}}
    monkeypatch.setattr(trainers, "by_id", lambda pid: fake if pid == "p" else None)
    ok = trainers.update_defaults("p", {"batch_size": "4", "precision": "fp32",
                                        "qat_enabled": False, "corpus": "DROP", "name": "DROP"})
    assert ok is True
    stored = trainer_tuning.args_for("p")
    assert stored == {"batch_size": 4, "precision": "fp32", "qat_enabled": False}
    assert isinstance(stored["batch_size"], int) and stored["qat_enabled"] is False


def test_overlay_tuning_replaces_only_default_keys(tmp_path, monkeypatch):
    """_overlay_tuning overlays tuned values onto manifest defaults, ignoring keys
    that are not manifest defaults."""
    monkeypatch.setattr(trainer_tuning, "TUNING_PATH", str(tmp_path / "t.json"))
    trainer_tuning.save("p", {"batch_size": 4, "not_a_default": 9})
    recs = [{"id": "p", "manifest": {"defaults": {"batch_size": 32, "seq": 512}}}]
    trainers._overlay_tuning(recs)
    d = recs[0]["manifest"]["defaults"]
    assert d["batch_size"] == 4
    assert d["seq"] == 512
    assert "not_a_default" not in d
