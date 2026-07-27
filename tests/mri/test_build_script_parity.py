# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Guards against build.bat / build.sh drift on the shared src/ + x86_64 kernel
#   TU set. A kernel TU added to one script but forgotten in the other links
#   fine on one platform and fails with undefined symbols on the other; this
#   test turns that latent, platform-specific breakage into a red CI run.
# tests/mri/test_build_script_parity.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from readers import paths
from training import build_runner

BUILD_DIR = paths.ENGINE_BUILD

# ------------------------------------------------------------------------------------
# Functions

def test_build_scripts_share_the_same_shared_tu_set():
    """build.bat and build.sh must reference the identical set of src/ and
    kernels/x86_64/ .c files. This is the exact invariant whose violation caused
    hybrid_matvec_*_avx2 / hybrid_matmul_*_avx2 to link on Linux/macOS but fail
    on Windows."""
    bat = build_runner._referenced_shared_tus(os.path.join(BUILD_DIR, "build.bat"))
    sh  = build_runner._referenced_shared_tus(os.path.join(BUILD_DIR, "build.sh"))
    assert bat is not None, "build.bat unreadable"
    assert sh  is not None, "build.sh unreadable"

    only_bat = sorted(bat - sh)
    only_sh  = sorted(sh - bat)
    assert not only_sh,  f"build.bat is missing x86_64 TUs that build.sh compiles: {only_sh}"
    assert not only_bat, f"build.sh is missing x86_64 TUs that build.bat compiles: {only_bat}"


def test_referenced_kernels_exist_on_disk():
    """Every x86_64 kernel TU the scripts name must actually exist, so a rename
    that updates the source tree but not the scripts is caught too."""
    bat = build_runner._referenced_shared_tus(os.path.join(BUILD_DIR, "build.bat")) or set()
    engine_root = paths.ENGINE_ROOT
    for rel in sorted(bat):
        assert os.path.isfile(os.path.join(engine_root, rel)), f"referenced TU missing on disk: {rel}"


def test_parity_helper_flags_injected_drift():
    """The parser must actually notice a dropped kernel: proves the guard isn't
    trivially passing by returning empty sets."""
    bat = build_runner._referenced_shared_tus(os.path.join(BUILD_DIR, "build.bat")) or set()
    sh  = build_runner._referenced_shared_tus(os.path.join(BUILD_DIR, "build.sh")) or set()
    victim = "kernels/x86_64/matvec_f32_avx2.c"
    assert victim in bat and victim in sh, "fixture kernel not present in both scripts"
    drifted = bat - {victim}
    assert sorted(sh - drifted) == [victim]
