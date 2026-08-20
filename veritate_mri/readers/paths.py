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
# Machine-local, not-code, survives app updates: repo-root data/ is in the
# updater's DEFAULT_SKIP_DIRS (veritate_mri/data/ is NOT, since only top-level
# names are matched, which is exactly why platform data belongs there and bulk
# local data belongs here).
LOCAL_DATA_ROOT = os.path.join(REPO_ROOT, "data")
# Corpus .bin home. Moved out of trainers/corpus/ on 2026-07-29 when the trainer
# became platform code and `trainers/` stopped holding any: a directory named
# "trainers" containing only training data was misleading. LEGACY_CORPUS_ROOT is
# still READ so existing installs keep working; downloads and builders write to
# CORPUS_ROOT. Resolution order is CORPUS_ROOT first (see corpus_search_dirs).
CORPUS_ROOT        = os.path.join(LOCAL_DATA_ROOT, "corpus")
LEGACY_CORPUS_ROOT = os.path.join(PLUGINS_ROOT, "corpus")
# Raw Project Gutenberg text cache the chat / agent corpus builders read from.
PG_CACHE_ROOT   = os.path.join(CORPUS_ROOT, "_pg_cache")
# The experience log: one JSONL file per local day recording every serving
# exchange (bytes in/out + params). The substrate for sleep consolidation
# (IDEA 20 T3) — the model trains on its own thought and actions. Bulk local
# data, so it lives under data/ with the corpus, not under veritate_mri/.
EXPERIENCE_ROOT = os.path.join(LOCAL_DATA_ROOT, "experience")
# Serialized streaming-decode states (IDEA 20 T2): one .pt per conversation id
# holding the carried recurrent states + pending window buffer + the checkpoint
# they belong to. A state is only valid for the exact weights that wrote it.
STREAM_STATE_ROOT = os.path.join(LOCAL_DATA_ROOT, "stream_states")
# Sleep controller state (IDEA 20 T3): last-sleep marker, in-flight run record,
# surviving sleep-final checkpoint steps. See training/sleep.py.
SLEEP_ROOT      = os.path.join(LOCAL_DATA_ROOT, "sleep")
MRI_ROOT        = os.path.join(REPO_ROOT, "veritate_mri")
# Scratch tree for corpus builder input / output. Gitignored.
TEMP_ROOT       = os.path.join(REPO_ROOT, "temp")
REQUIREMENTS_PATH = os.path.join(REPO_ROOT, "requirements.txt")
# Platform-level data lives under veritate_mri/data/. eval/grade/ holds the
# committed grade-level eval corpora; eval/samples/ holds the small smoke
# subsets of MMLU / HellaSwag / IFEval; corpus/ holds the Veritate-native
# corpora the Settings library installs by copying into trainers/corpus/;
# synth_jobs/ holds raw teacher synth output per job (samples, cache, state)
# until a corpus .bin is built from it; corpus_mix_profiles.json holds the
# mix planner's intent profiles; authoring/corpus_spec.json holds the
# editable genre, prompt, and quality-gate data the corpus authoring
# pipeline runs on. The dashboard wiki tab serves repo-root documentation.md
# (see documentation_path()), not a file under this tree.
DATA_ROOT          = os.path.join(MRI_ROOT, "data")
EVAL_ROOT          = os.path.join(DATA_ROOT, "eval")
GRADE_EVAL_ROOT    = os.path.join(EVAL_ROOT, "grade")
# Hand-authored passages the grade / comprehension builders read, and the three
# generated smartness-meter axes the checkpoint probe scores against.
GRADE_EVAL_SOURCES_ROOT   = os.path.join(GRADE_EVAL_ROOT, "sources")
GRADE_EVAL_MATH_ROOT      = os.path.join(GRADE_EVAL_ROOT, "math")
GRADE_EVAL_REASONING_ROOT = os.path.join(GRADE_EVAL_ROOT, "reasoning")
GRADE_EVAL_GRAMMAR_ROOT   = os.path.join(GRADE_EVAL_ROOT, "grammar")
EVAL_SAMPLES_ROOT  = os.path.join(EVAL_ROOT, "samples")
# Held-out question/answer sets the RAG corpus builder writes beside its bins.
RAG_EVAL_ROOT      = os.path.join(EVAL_ROOT, "rag")
NATIVE_CORPUS_ROOT = os.path.join(DATA_ROOT, "corpus")
SYNTH_JOBS_ROOT    = os.path.join(DATA_ROOT, "synth_jobs")
AUTHORING_ROOT     = os.path.join(DATA_ROOT, "authoring")
ENGINE_ROOT     = os.path.join(REPO_ROOT, "veritate_engine")
ENGINE_BIN      = os.path.join(ENGINE_ROOT, "bin")
ENGINE_BUILD    = os.path.join(ENGINE_ROOT, "build")
ENGINE_SRC      = os.path.join(ENGINE_ROOT, "src")
ENGINE_KERNELS  = os.path.join(ENGINE_ROOT, "kernels")
ENGINE_SOURCE_SUFFIXES = (".c", ".h", ".S")
# Component version ledger for the whole platform.
VERSIONS_JSON_PATH = os.path.join(REPO_ROOT, "versions.json")
# Single-file platform documentation the dashboard wiki tab serves.
DOCUMENTATION_PATH = os.path.join(REPO_ROOT, "documentation.md")

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
BIN_TERNARY_NAME  = "veritate_ternary.bin"   # ternary export sits beside the int8 bin
CHECKPOINTS_DIR   = "checkpoints"
HOOKS_DIR         = "hooks"

GRADE_SOURCE_PREFIX = "grade_"
GRADE_SOURCE_SUFFIX = "_source.txt"
COMPREHENSION_PREFIX      = "comprehension_"
COMPREHENSION_HARD_PREFIX = "comprehension_hard_"
COMPREHENSION_SUFFIX      = ".json"
JSONL_SUFFIX              = ".jsonl"
RAG_EVAL_SUFFIX           = "_test.json"

MIX_PROFILES_NAME = "corpus_mix_profiles.json"
AUTHORING_SPEC_NAME = "corpus_spec.json"

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
    "chat_health": ("chat_health.json", "json"),
    "reading_comprehension": ("reading_comprehension.json", "json"),
    "generation": ("generation.json", "json"),
}

# ------------------------------------------------------------------------------------
# Functions

def model_dir(name):
    return os.path.join(MODELS_ROOT, name)


def corpus_dir():
    """Where NEW corpus bins are written. Reads may resolve elsewhere; see
    corpus_search_dirs()."""
    return CORPUS_ROOT


def corpus_search_dirs():
    """Every directory a corpus bin may live in, canonical first. Callers that
    list or glob corpora must walk all of them, or a legacy install goes blind
    to its own data."""
    return (CORPUS_ROOT, LEGACY_CORPUS_ROOT)


def _corpus_file_path(filename):
    """An existing file wins wherever it lives, so a legacy install keeps
    resolving and an in-place rebuild overwrites the copy actually in use.
    Falls back to the canonical root, which is the correct target for a stem
    that does not exist yet."""
    for root in corpus_search_dirs():
        candidate = os.path.join(root, filename)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(CORPUS_ROOT, filename)


def corpus_train_path(stem):
    return _corpus_file_path(f"{stem}{CORPUS_TRAIN_SUFFIX}")


def corpus_val_path(stem):
    return _corpus_file_path(f"{stem}{CORPUS_VAL_SUFFIX}")


def native_corpus_train_path(stem):
    return os.path.join(NATIVE_CORPUS_ROOT, f"{stem}{CORPUS_TRAIN_SUFFIX}")


def native_corpus_val_path(stem):
    return os.path.join(NATIVE_CORPUS_ROOT, f"{stem}{CORPUS_VAL_SUFFIX}")


def mix_profiles_path():
    return os.path.join(DATA_ROOT, MIX_PROFILES_NAME)


def authoring_spec_path():
    return os.path.join(AUTHORING_ROOT, AUTHORING_SPEC_NAME)


def synth_jobs_root():
    return SYNTH_JOBS_ROOT


def synth_job_dir(job_id):
    return os.path.join(SYNTH_JOBS_ROOT, job_id)


def grade_eval_path(level):
    return os.path.join(GRADE_EVAL_ROOT, f"{GRADE_EVAL_PREFIX}{level}{GRADE_EVAL_SUFFIX}")


def grade_source_path(level):
    return os.path.join(GRADE_EVAL_SOURCES_ROOT, f"{GRADE_SOURCE_PREFIX}{level}{GRADE_SOURCE_SUFFIX}")


def comprehension_path(level, hard=False):
    prefix = COMPREHENSION_HARD_PREFIX if hard else COMPREHENSION_PREFIX
    return os.path.join(GRADE_EVAL_ROOT, f"{prefix}{level}{COMPREHENSION_SUFFIX}")


def rag_eval_path(stem):
    return os.path.join(RAG_EVAL_ROOT, f"{stem}{RAG_EVAL_SUFFIX}")


def eval_axis_item_path(axis_root, name):
    return os.path.join(axis_root, f"{name}{JSONL_SUFFIX}")


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


def bin_ternary_path(name):
    return os.path.join(model_dir(name), BIN_TERNARY_NAME)


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


def documentation_path():
    return DOCUMENTATION_PATH


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


def engine_binary_dir(os_name=None, arch=None):
    return os.path.join(ENGINE_BIN, os_name or current_os(), arch or current_arch())


def engine_binary_path(os_name=None, arch=None):
    o = os_name or current_os()
    return os.path.join(engine_binary_dir(o, arch), engine_binary_name(o))


def build_script_path(os_name=None):
    o = os_name or current_os()
    return os.path.join(ENGINE_BUILD, BUILD_SCRIPT_BY_OS.get(o, "build.sh"))
