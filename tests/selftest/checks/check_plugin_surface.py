# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the veritate_core.plugin contract surface that trainer plugins consume.
# - drift here breaks every external trainer.
# tests/selftest/checks/check_plugin_surface.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA           = "plugin_contract"
EXPECTED_ALL   = {"save", "paths", "model", "qat"}
SAVE_FUNCS     = ("save", "append_train_row")
PATHS_FUNCS    = ("model_dir", "config_path", "train_csv_path", "bin_path",
                  "checkpoints_dir", "checkpoint_path", "hooks_dir",
                  "hook_step_dir", "engine_binary_path")
MODEL_NAMES    = ("Veritate", "VOCAB_BYTE_LEVEL")
QAT_NAMES      = ("fake_quant_weight", "fake_quant_act", "fake_quant_ln_weight",
                  "set_qat", "INT8_MAX", "ACT_INT8_SCALE", "LN_FIXED_SCALE")

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """veritate_core.plugin re-exports save / paths / model / qat with the
    documented public names."""
    try:
        import veritate_core.plugin as plug
    except Exception as exc:
        return _status.fail("plugin_surface", f"import failed: {exc}")

    if set(getattr(plug, "__all__", ())) != EXPECTED_ALL:
        return _status.fail("plugin_surface",
                            f"__all__ {set(getattr(plug, '__all__', ()))} != {EXPECTED_ALL}")

    miss = []
    miss += [f"save.{n}" for n in SAVE_FUNCS if not hasattr(plug.save, n)]
    miss += [f"paths.{n}" for n in PATHS_FUNCS if not hasattr(plug.paths, n)]
    miss += [f"model.{n}" for n in MODEL_NAMES if not hasattr(plug.model, n)]
    miss += [f"qat.{n}" for n in QAT_NAMES if not hasattr(plug.qat, n)]
    if miss:
        return _status.fail("plugin_surface", f"missing surface: {miss[:6]}", {"missing": miss})
    return _status.ok("plugin_surface",
                      f"all {len(SAVE_FUNCS) + len(PATHS_FUNCS) + len(MODEL_NAMES) + len(QAT_NAMES)} names present")
