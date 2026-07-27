# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - single owner of test import paths (preflight rule 20). the platform is not an
#   installed package: veritate_mri modules import each other by bare name, so the
#   repo root, veritate_mri/, and veritate_mri/tools/ all go on sys.path once here.
# - test modules import REPO_ROOT from this file rather than recomputing it.
# tests/conftest.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import sys

# ------------------------------------------------------------------------------------
# Constants

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MRI_ROOT  = os.path.join(REPO_ROOT, "veritate_mri")
TOOLS_DIR = os.path.join(MRI_ROOT, "tools")

# ------------------------------------------------------------------------------------
# Functions

for _p in (TOOLS_DIR, MRI_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
