# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - unit tests for readers/capabilities.py and the trainer-manifest teaches
#   field. covers the tier schema, status transitions enforced by mark(),
#   the legacy fallback for pre-pipeline models, fork preservation, and the
#   teaches() helper. no torch, no network.
# tests/mri/test_capabilities.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import shutil
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
MRI_DIR = os.path.join(REPO_ROOT, "veritate_mri")
if MRI_DIR not in sys.path:
    sys.path.insert(0, MRI_DIR)

from readers import capabilities as caps, paths
from readers import trainers as trainers_reader

# ------------------------------------------------------------------------------------
# Constants

TEST_NAME = "tmpcaps_1m_fp32_v1"

# ------------------------------------------------------------------------------------
# Functions

def _cleanup():
    d = paths.model_dir(TEST_NAME)
    if os.path.isdir(d):
        shutil.rmtree(d)


def _write_config(extra=None):
    d = paths.model_dir(TEST_NAME)
    os.makedirs(d, exist_ok=True)
    cfg = {"name": TEST_NAME, "training_args": {"total_steps": 100}}
    if extra:
        cfg.update(extra)
    with open(paths.config_path(TEST_NAME), "w", encoding="utf-8") as f:
        json.dump(cfg, f)


def _read_config():
    with open(paths.config_path(TEST_NAME), "r", encoding="utf-8") as f:
        return json.load(f)


class TestReadFallback(unittest.TestCase):
    """read() synthesizes a legacy block for missing/legacy models."""

    def setUp(self):
        _cleanup()

    def tearDown(self):
        _cleanup()

    def test_missing_model_returns_legacy_autocomplete_trained(self):
        """read() on a nonexistent model name returns legacy autocomplete=trained."""
        block = caps.read("does_not_exist_85m_fp32_v1")
        self.assertEqual(block["autocomplete"]["status"], caps.STATUS_TRAINED)
        self.assertEqual(block["chat"]["status"],         caps.STATUS_UNTRAINED)
        self.assertEqual(block["agent"]["status"],        caps.STATUS_UNTRAINED)

    def test_missing_block_in_existing_config_is_legacy(self):
        """A config without a capabilities key falls back to legacy autocomplete=trained."""
        _write_config()
        block = caps.read(TEST_NAME)
        self.assertEqual(block["autocomplete"]["status"], caps.STATUS_TRAINED)
        self.assertTrue(block["autocomplete"].get("legacy"))


class TestMarkTransitions(unittest.TestCase):
    """mark() promotes / never regresses tier statuses."""

    def setUp(self):
        _cleanup()
        _write_config()

    def tearDown(self):
        _cleanup()

    def test_in_progress_records_trainer_and_step(self):
        """STATUS_IN_PROGRESS writes trainer + step into the entry."""
        entry = caps.mark(TEST_NAME, "autocomplete", caps.STATUS_IN_PROGRESS,
                          trainer="veritate_85m", step=10, total_steps=100)
        self.assertEqual(entry["status"], caps.STATUS_IN_PROGRESS)
        self.assertEqual(entry["trainer"], "veritate_85m")
        self.assertEqual(entry["step"], 10)

    def test_final_step_promotes_to_trained(self):
        """Hitting total_steps with STATUS_IN_PROGRESS promotes to STATUS_TRAINED."""
        entry = caps.mark(TEST_NAME, "autocomplete", caps.STATUS_IN_PROGRESS,
                          trainer="veritate_85m", step=100, total_steps=100)
        self.assertEqual(entry["status"], caps.STATUS_TRAINED)
        self.assertIn("completed_at", entry)

    def test_failed_never_overwrites_trained(self):
        """A FAILED mark on a previously TRAINED tier leaves it TRAINED."""
        caps.mark(TEST_NAME, "autocomplete", caps.STATUS_TRAINED, trainer="x", step=100)
        entry = caps.mark(TEST_NAME, "autocomplete", caps.STATUS_FAILED, trainer="x", step=50)
        self.assertEqual(entry["status"], caps.STATUS_TRAINED)

    def test_unknown_tier_raises(self):
        """An unknown tier name is a programming error, not a no-op."""
        with self.assertRaises(ValueError):
            caps.mark(TEST_NAME, "bogus", caps.STATUS_TRAINED)

    def test_unknown_status_raises(self):
        """An unknown status name is rejected at the API boundary."""
        with self.assertRaises(ValueError):
            caps.mark(TEST_NAME, "autocomplete", "bogus")

    def test_missing_config_returns_none(self):
        """mark() refuses to bootstrap config.json; that is save._ensure_config's job."""
        _cleanup()
        self.assertIsNone(caps.mark(TEST_NAME, "autocomplete", caps.STATUS_TRAINED))


class TestHighestTrained(unittest.TestCase):
    """highest_trained() walks tiers from highest to lowest."""

    def test_returns_highest_trained_tier(self):
        """agent wins over chat wins over autocomplete when all trained."""
        block = {
            "autocomplete": {"status": caps.STATUS_TRAINED},
            "chat":         {"status": caps.STATUS_TRAINED},
            "agent":        {"status": caps.STATUS_TRAINED},
        }
        self.assertEqual(caps.highest_trained(block), "agent")

    def test_returns_none_when_nothing_trained(self):
        """Empty block returns None, not autocomplete-by-default."""
        block = {t: {"status": caps.STATUS_UNTRAINED} for t in caps.TIERS}
        self.assertIsNone(caps.highest_trained(block))


class TestTrainerTeachesField(unittest.TestCase):
    """trainers.teaches() reads the manifest field with the autocomplete default."""

    def test_pretraining_trainer_defaults_to_autocomplete(self):
        """Missing teaches field falls back to autocomplete (pretraining default)."""
        self.assertEqual(trainers_reader.teaches("veritate_85m"), "autocomplete")

    def test_unknown_plugin_id_returns_default(self):
        """An unknown plugin id falls back to autocomplete, not None or an error."""
        self.assertEqual(trainers_reader.teaches("nope/missing"), "autocomplete")


if __name__ == "__main__":
    unittest.main()
