# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate.plugin is the single import surface external plugins are allowed
#   to touch. Drift here breaks every plugin in the veritate-plugins repo.
# - These tests assert what is exported, not what is called. Behavior tests
#   for save / paths / model live with their owning module.
# tests/plugin_contract/test_surface.py
# ------------------------------------------------------------------------------------
# Imports

import pytest

# ------------------------------------------------------------------------------------
# Functions

def test_plugin_module_imports():
    """`import veritate.plugin` succeeds without side effects that raise."""
    import veritate.plugin                     # noqa: F401


def test_plugin_exports_save():
    """veritate.plugin.save is the checkpoint-save module exposed to plugins."""
    from veritate.plugin import save
    assert hasattr(save, "save"),              "save.save() is the canonical checkpoint+hooks entry point"
    assert hasattr(save, "append_train_row"),  "append_train_row is the only sanctioned per-step CSV writer"


def test_plugin_exports_paths():
    """veritate.plugin.paths exposes the path helpers plugins resolve through."""
    from veritate.plugin import paths
    for fn in ("model_dir", "checkpoint_path", "config_path", "bin_path"):
        assert hasattr(paths, fn),             f"paths.{fn} missing -- plugins resolve via this surface"


def test_plugin_exports_model():
    """veritate.plugin.model exposes the Veritate base model + byte-vocab constant."""
    from veritate.plugin import model
    assert hasattr(model, "Veritate"),         "Veritate base class is the canonical training target"
    assert hasattr(model, "VOCAB_BYTE_LEVEL"), "byte-level vocab constant is required for shape config"
    assert model.VOCAB_BYTE_LEVEL == 256,      "byte-level vocab must be 256 (raw bytes)"


def test_plugin_exports_qat():
    """veritate.plugin.qat exposes the QAT helpers ternary/INT8 trainers depend on."""
    from veritate.plugin import qat
    # the qat module is shipped via veritate.plugin even if specific helpers vary;
    # the import itself is the contract here. Helper names checked downstream.
    assert qat is not None


def test_plugin_all_lists_documented_surface():
    """veritate.plugin.__all__ matches the four documented modules and nothing else."""
    import veritate.plugin
    assert set(veritate.plugin.__all__) == {"save", "paths", "model", "qat"}, \
        "adding to __all__ requires updating documentation/plugins/contract.md"


def test_plugins_dir_is_not_imported_via_sys_path_injection():
    """Tests that veritate.plugin does NOT add plugins/ to sys.path. Plugins import
    veritate.plugin; veritate.plugin must not silently reach back into plugins/."""
    import sys
    import veritate.plugin                     # noqa: F401
    bad = [p for p in sys.path if p.endswith("/plugins") or p.endswith("/plugins/")]
    assert not bad, f"plugins/ leaked into sys.path: {bad}"
