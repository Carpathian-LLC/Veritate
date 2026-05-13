# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Roundtrip test for save(): tiny torch model -> save() -> torch.load() ->
#   verify state_dict keys + tensor norms match. All hook dumps skipped so
#   no real Veritate model is required.
# veritate_mri/tests/test_save_roundtrip.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import shutil
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
MRI_DIR = os.path.dirname(HERE)
if MRI_DIR not in sys.path:
    sys.path.insert(0, MRI_DIR)

import torch
from readers import paths
from training import save as save_mod

# ------------------------------------------------------------------------------------
# Constants

TEST_NAME = "tmproundtrip_1m_fp32_v1"
TEST_STEP = 1
TEST_ARGS = {
    "description": "checkpoint roundtrip test",
    "shape": {"layers": 1, "hidden": 8, "ffn": 16, "heads": 2, "seq": 4, "vocab": 256},
}
SKIP_ALL_DUMPS = ("probe", "lens", "classroom", "grades", "concepts",
                  "surprise", "quant_kl", "generation")
NORM_PLACES = 5

# ------------------------------------------------------------------------------------
# Functions

def _cleanup():
    d = paths.model_dir(TEST_NAME)
    if os.path.isdir(d):
        shutil.rmtree(d)


class TestCheckpointRoundtrip(unittest.TestCase):

    def setUp(self):
        _cleanup()

    def tearDown(self):
        _cleanup()

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
        )
        before = {k: float(v.norm().item()) for k, v in model.state_dict().items()}

        ckpt_path = save_mod.save(
            model, TEST_NAME, TEST_STEP,
            args=TEST_ARGS, dump_set=SKIP_ALL_DUMPS,
        )
        self.assertTrue(os.path.isfile(ckpt_path))

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.assertIn("model", loaded)
        self.assertEqual(loaded.get("step"), TEST_STEP)

        self.assertEqual(set(before.keys()), set(loaded["model"].keys()))
        for k, v in before.items():
            self.assertAlmostEqual(
                v, float(loaded["model"][k].norm().item()), places=NORM_PLACES,
                msg=f"tensor norm changed for {k}",
            )


if __name__ == "__main__":
    unittest.main()
