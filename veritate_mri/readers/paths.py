# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - single source of truth for the on-disk layout of every model.
# - every other reader imports from here. no path strings outside this file.
# veritate_mri/readers/paths.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import platform
import re
import sys

# ------------------------------------------------------------------------------------
# Constants

REPO_ROOT       = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MODELS_ROOT     = os.path.join(REPO_ROOT, "models")
PLUGINS_ROOT    = os.path.join(REPO_ROOT, "trainers")
CORPUS_ROOT     = os.path.join(PLUGINS_ROOT, "corpus")
MRI_ROOT        = os.path.join(REPO_ROOT, "veritate_mri")
# Platform-level data lives under veritate_mri/data/. eval/grade/ holds the
# committed grade-level eval corpora; eval/samples/ holds the small smoke
# subsets of MMLU / HellaSwag / IFEval; wiki/ holds dashboard wiki markdown;
# corpus/ holds the Veritate-native corpora the Settings library installs by
# copying into trainers/corpus/.
DATA_ROOT          = os.path.join(MRI_ROOT, "data")
EVAL_ROOT          = os.path.join(DATA_ROOT, "eval")
GRADE_EVAL_ROOT    = os.path.join(EVAL_ROOT, "grade")
EVAL_SAMPLES_ROOT  = os.path.join(EVAL_ROOT, "samples")
WIKI_ROOT          = os.path.join(DATA_ROOT, "wiki")
NATIVE_CORPUS_ROOT = os.path.join(DATA_ROOT, "corpus")
ENGINE_ROOT     = os.path.join(REPO_ROOT, "veritate_engine")
# v1 is the primary (and only built) engine. v2 is an empty scratchpad folder
# reserved for future engine experiments (see documentation/engine/v2.md).
ENGINE_PRIMARY  = os.path.join(ENGINE_ROOT, "v1")
ENGINE_BIN      = os.path.join(ENGINE_PRIMARY, "bin")
ENGINE_BUILD    = os.path.join(ENGINE_PRIMARY, "build")
ENGINE_VERSIONS_JSON = os.path.join(ENGINE_PRIMARY, "engine_versions.json")

OS_WINDOWS = "windows"
OS_LINUX   = "linux"
OS_MACOS   = "macos"

ARCH_X86_64 = "x86_64"
ARCH_ARM64  = "arm64"

BINARY_NAME_BY_OS = {
    OS_WINDOWS: "veritate.exe",
    OS_LINUX:   "veritate",
    OS_MACOS:   "veritate",
}

BUILD_SCRIPT_BY_OS = {
    OS_WINDOWS: "build.bat",
    OS_LINUX:   "build.sh",
    OS_MACOS:   "build.sh",
}

CONFIG_NAME       = "config.json"
TRAIN_CSV_NAME    = "train.csv"
BIN_NAME          = "veritate.bin"
CHECKPOINTS_DIR   = "checkpoints"
HOOKS_DIR         = "hooks"

WIKI_ENTRY_SUFFIX = ".md"

CORPUS_TRAIN_SUFFIX = "_train.bin"
CORPUS_VAL_SUFFIX   = "_val.bin"
GRADE_EVAL_PREFIX   = "grade_"
GRADE_EVAL_SUFFIX   = "_eval.bin"

CHECKPOINT_RE = re.compile(r"^step_(\d+)\.pt$")
HOOK_STEP_RE  = re.compile(r"^step_(\d+)$")

HOOK_ARTIFACTS = {
    "probe":      ("probe.json",      "json"),
    "lens":       ("lens.npz",        "npz"),
    "classroom":  ("classroom.json",  "json"),
    "grades":     ("grades.json",     "json"),
    "math":       ("math.json",       "json"),
    "grammar":    ("grammar.json",    "json"),
    "reasoning":  ("reasoning.json",  "json"),
    "concepts":   ("concepts.json",   "json"),
    "surprise":   ("surprise.json",   "json"),
    "quant_kl":   ("quant_kl.json",   "json"),
    "writing_health": ("writing_health.json", "json"),
    "reading_comprehension": ("reading_comprehension.json", "json"),
    "generation": ("generation.json", "json"),
}

# ------------------------------------------------------------------------------------
# Functions

def model_dir(name):
    return os.path.join(MODELS_ROOT, name)


def corpus_dir():
    return CORPUS_ROOT


def corpus_train_path(stem):
    return os.path.join(CORPUS_ROOT, f"{stem}{CORPUS_TRAIN_SUFFIX}")


def corpus_val_path(stem):
    return os.path.join(CORPUS_ROOT, f"{stem}{CORPUS_VAL_SUFFIX}")


def native_corpus_train_path(stem):
    return os.path.join(NATIVE_CORPUS_ROOT, f"{stem}{CORPUS_TRAIN_SUFFIX}")


def native_corpus_val_path(stem):
    return os.path.join(NATIVE_CORPUS_ROOT, f"{stem}{CORPUS_VAL_SUFFIX}")


def grade_eval_path(level):
    return os.path.join(GRADE_EVAL_ROOT, f"{GRADE_EVAL_PREFIX}{level}{GRADE_EVAL_SUFFIX}")


def bundled_corpus_train_path(bundle_corpus_dir, stem):
    return os.path.join(bundle_corpus_dir, f"{stem}{CORPUS_TRAIN_SUFFIX}")


def bundled_corpus_val_path(bundle_corpus_dir, stem):
    return os.path.join(bundle_corpus_dir, f"{stem}{CORPUS_VAL_SUFFIX}")


def config_path(name):
    return os.path.join(model_dir(name), CONFIG_NAME)


def train_csv_path(name):
    return os.path.join(model_dir(name), TRAIN_CSV_NAME)


def bin_path(name):
    return os.path.join(model_dir(name), BIN_NAME)


def checkpoints_dir(name):
    return os.path.join(model_dir(name), CHECKPOINTS_DIR)


def checkpoint_path(name, step):
    return os.path.join(checkpoints_dir(name), f"step_{int(step)}.pt")


def hooks_dir(name):
    return os.path.join(model_dir(name), HOOKS_DIR)


def hook_step_dir(name, step):
    return os.path.join(hooks_dir(name), f"step_{int(step)}")


def hook_artifact_path(name, step, artifact):
    fname = HOOK_ARTIFACTS[artifact][0]
    return os.path.join(hook_step_dir(name, step), fname)


def wiki_root():
    return WIKI_ROOT


def wiki_category_dir(category):
    return os.path.join(WIKI_ROOT, category)


def wiki_entry_path(category, slug):
    return os.path.join(WIKI_ROOT, category, f"{slug}{WIKI_ENTRY_SUFFIX}")


def current_os():
    p = sys.platform
    if p.startswith("win"):    return OS_WINDOWS
    if p.startswith("darwin"): return OS_MACOS
    return OS_LINUX


def current_arch():
    m = (platform.machine() or "").lower()
    if m in ("arm64", "aarch64"):  return ARCH_ARM64
    if m in ("x86_64", "amd64"):   return ARCH_X86_64
    return m or ARCH_X86_64


def engine_binary_name(os_name=None):
    return BINARY_NAME_BY_OS.get(os_name or current_os(), "veritate")


def engine_binary_path(os_name=None, arch=None):
    o = os_name or current_os()
    a = arch or current_arch()
    return os.path.join(ENGINE_BIN, o, a, engine_binary_name(o))


def build_script_path(os_name=None):
    o = os_name or current_os()
    return os.path.join(ENGINE_BUILD, BUILD_SCRIPT_BY_OS.get(o, "build.sh"))
