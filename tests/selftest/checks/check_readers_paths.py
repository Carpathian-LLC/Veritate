# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - veritate_mri/readers/paths.py is the on-disk source of truth. every path
#   helper must return a string, the engine binary path must point under
#   veritate_engine/, and the model dir must live under models/.
# tests/selftest/checks/check_readers_paths.py
# ------------------------------------------------------------------------------------
# Imports

import os

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA           = "platform"

PATH_FUNCS     = (
    ("model_dir",         ("demo",)),
    ("config_path",       ("demo",)),
    ("train_csv_path",    ("demo",)),
    ("bin_path",          ("demo",)),
    ("checkpoints_dir",   ("demo",)),
    ("checkpoint_path",   ("demo", 100)),
    ("hooks_dir",         ("demo",)),
    ("hook_step_dir",     ("demo", 100)),
    ("hook_artifact_path",("demo", 100, "probe")),
    ("corpus_dir",        ()),
    ("corpus_train_path", ("tinystories",)),
    ("corpus_val_path",   ("tinystories",)),
    ("engine_binary_path",()),
    ("build_script_path", ()),
)

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """every paths.* helper exists, returns a string, and engine paths land
    under veritate_engine/."""
    try:
        from readers import paths
    except Exception as exc:
        return _status.fail("readers_paths", f"import failed: {exc}")

    bad = []
    for name, args in PATH_FUNCS:
        fn = getattr(paths, name, None)
        if fn is None:
            bad.append(f"missing {name}")
            continue
        try:
            val = fn(*args)
        except Exception as exc:
            bad.append(f"{name} raised: {exc}")
            continue
        if not isinstance(val, str) or not val:
            bad.append(f"{name} returned non-string: {val!r}")
    if bad:
        return _status.fail("readers_paths", "; ".join(bad[:4]), {"errors": bad})

    eng = paths.engine_binary_path()
    if "veritate_engine" not in eng:
        return _status.fail("readers_paths", f"engine path off-tree: {eng}")
    mdir = paths.model_dir("demo")
    if os.path.basename(mdir) != "demo":
        return _status.fail("readers_paths", f"model_dir basename wrong: {mdir}")
    return _status.ok("readers_paths", f"{len(PATH_FUNCS)} helpers ok")
