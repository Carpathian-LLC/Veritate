# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - every module under veritate_mri/training/ imports cleanly. these own
#   checkpoint save, export, atlas, pruning, sync, and the trainer runner. an
#   import failure here breaks the dashboard at startup.
# tests/selftest/checks/check_training_modules.py
# ------------------------------------------------------------------------------------
# Imports

from tests.selftest import _status

# ------------------------------------------------------------------------------------
# Constants

AREA            = "platform"
TRAINING_MODS   = (
    "save", "export", "atlas", "pruning", "confidence", "fork",
    "build_runner", "trainer_runner", "train_stream",
    "native_trainer", "checkpoint_probe",
)
SYNC_MODS       = ("trainers_sync", "models_sync", "corpus_sync", "app_sync")

# ------------------------------------------------------------------------------------
# Functions

def run(ctx):
    """import every training/* and training/sync/* module."""
    failures = []
    for name in TRAINING_MODS:
        try:
            __import__("training." + name)
        except Exception as exc:
            failures.append(f"training.{name}: {type(exc).__name__}: {exc}")
    for name in SYNC_MODS:
        try:
            __import__("training.sync." + name)
        except Exception as exc:
            failures.append(f"training.sync.{name}: {type(exc).__name__}: {exc}")
    if failures:
        return _status.fail("training_modules", failures[0], {"errors": failures})
    return _status.ok("training_modules",
                      f"{len(TRAINING_MODS) + len(SYNC_MODS)} training modules imported")
