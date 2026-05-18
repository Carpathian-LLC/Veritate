# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - shared context for selftest checks. paths, repo root, sys.path injection.
# - every check reads from this; no path strings in check bodies.
# tests/selftest/_ctx.py
# ------------------------------------------------------------------------------------
# Imports

import os
import sys
import time

# ------------------------------------------------------------------------------------
# Constants

HERE          = os.path.dirname(os.path.abspath(__file__))
TESTS_ROOT    = os.path.normpath(os.path.join(HERE, ".."))
REPO_ROOT     = os.path.normpath(os.path.join(HERE, "..", ".."))
CHECKS_DIR    = os.path.join(HERE, "checks")
LOGS_DIR      = os.path.join(HERE, "logs")

MODELS_DIR    = os.path.join(REPO_ROOT, "models")
TRAINERS_DIR  = os.path.join(REPO_ROOT, "trainers")
MRI_DIR       = os.path.join(REPO_ROOT, "veritate_mri")
ENGINE_DIR    = os.path.join(REPO_ROOT, "veritate_engine")
VERSIONS_JSON = os.path.join(REPO_ROOT, "versions.json")

CHECK_PREFIX  = "check_"
CHECK_SUFFIX  = ".py"
SUMMARY_NAME  = "summary.json"
RUN_LOG_NAME  = "run.log"
TS_FORMAT     = "%Y%m%d_%H%M%S"

# ------------------------------------------------------------------------------------
# Functions

def install_paths():
    """put repo root and mri root on sys.path so checks can import platform code."""
    for p in (REPO_ROOT, MRI_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


def new_run_id():
    """timestamp string used as the per-run log subdir name."""
    return time.strftime(TS_FORMAT, time.localtime())


def run_log_dir(run_id):
    return os.path.join(LOGS_DIR, run_id)


def check_log_path(run_id, check_name):
    return os.path.join(run_log_dir(run_id), f"{check_name}.log")


def summary_path(run_id):
    return os.path.join(run_log_dir(run_id), SUMMARY_NAME)


def run_log_path(run_id):
    return os.path.join(run_log_dir(run_id), RUN_LOG_NAME)


class Ctx:
    """context handed to every check. carries paths, options, and the run logger."""

    def __init__(self, run_id, log_dir, options):
        self.run_id   = run_id
        self.log_dir  = log_dir
        self.options  = options
        self.repo     = REPO_ROOT
        self.models   = MODELS_DIR
        self.trainers = TRAINERS_DIR
        self.mri      = MRI_DIR
        self.engine   = ENGINE_DIR
