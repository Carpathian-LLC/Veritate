# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Static contract test for documentation/hooks/contract.md. Verifies the
#   canonical artifact set in save.RENAME_MAP_TEMPLATE matches the dashboard
#   contract, and that any present generation.json carries the required TFRM
#   v7 frame fields. Model-free; runs against existing dumps if any.
# tests/mri/test_hooks_contract.py
# ------------------------------------------------------------------------------------
# Imports:

import glob
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MRI_DIR = os.path.join(REPO_ROOT, "veritate_mri")
if MRI_DIR not in sys.path:
    sys.path.insert(0, MRI_DIR)

from training import save as save_mod
from readers import paths

# ------------------------------------------------------------------------------------
# Constants

CANONICAL_ARTIFACTS = {
    "probe.json",
    "lens.npz",
    "classroom.json",
    "grades.json",
    "math.json",
    "grammar.json",
    "reasoning.json",
    "concepts.json",
    "surprise.json",
    "quant_kl.json",
    "writing_health.json",
    "reading_comprehension.json",
    "generation.json",
}

# minimum TFRM v7 frame fields required on every per-token frame
TFRM_V7_REQUIRED_FIELDS = (
    "kind", "byte", "argmax_byte", "T", "fwd_ms",
    "entropy_bits", "surprise_bits",
)

GENERATION_GLOB = os.path.join(paths.MODELS_ROOT, "*", "hooks", "step_*", "generation.json")
MAX_FRAMES_TO_CHECK = 4

# ------------------------------------------------------------------------------------
# Functions

def _read_first_frame(path):
    with open(path, "rb") as f:
        head = f.read(2 * 1024 * 1024)
    text = head.decode("utf-8", errors="replace")
    i = text.find('"frames"')
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        c = text[k]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[j:k + 1])
                except Exception:
                    return None
    return None


class TestHooksContract(unittest.TestCase):

    def test_rename_map_matches_canonical_set(self):
        rmap = save_mod.RENAME_MAP_TEMPLATE
        canonical = set(rmap.values())
        self.assertEqual(canonical, CANONICAL_ARTIFACTS,
                         "save.RENAME_MAP_TEMPLATE drifted from documented contract")

    def test_existing_generation_dumps_have_required_frame_fields(self):
        gens = glob.glob(GENERATION_GLOB)
        if not gens:
            self.skipTest("no models/*/hooks/step_*/generation.json present")
        for path in gens:
            frame = _read_first_frame(path)
            self.assertIsNotNone(frame, f"could not parse first frame in {path}")
            missing = [f for f in TFRM_V7_REQUIRED_FIELDS if f not in frame]
            self.assertEqual(missing, [], f"{path}: missing TFRM v7 fields: {missing}")


if __name__ == "__main__":
    unittest.main()
