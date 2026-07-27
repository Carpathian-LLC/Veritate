# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - save.save() on a tiny CPU model: the canonical dump filenames land in
#   hooks/step_<N>/, no dump reports a failure, and the model_type gate
#   (rule 24a) skips the language suite for `statistical` only.
# - three module-scoped saves (language / code / statistical) keep the runtime
#   near one dump suite each; the whole module is marked slow.
# tests/training/test_save_dumps.py
# ------------------------------------------------------------------------------------
# Imports:

import io
import os
from contextlib import redirect_stdout

import pytest
import torch
from readers import paths
from training import save

from veritate_core.model import Veritate

# ------------------------------------------------------------------------------------
# Constants

VOCAB  = 256
HIDDEN = 16
LAYERS = 2
FFN    = 32
HEADS  = 2
SEQ    = 64
STEP   = 1

CORPUS_STEM    = "tinycorpus"
CORPUS_TEXT    = b"the quick brown fox jumps over the lazy dog. she said hello to the man there. "
CORPUS_REPEATS = 200

DUMP_FAILED_MARK = "DUMP FAILED:"
PROBE_ARTIFACT   = "probe"

MODEL_TYPE_CODE        = "code"
MODEL_TYPE_STATISTICAL = "statistical"

pytestmark = pytest.mark.slow

# ------------------------------------------------------------------------------------
# Functions

def _language_filenames():
    return {paths.HOOK_ARTIFACTS[d][0] for d in save.LANGUAGE_DUMPS}


def _run_save(root, name, model_type):
    torch.manual_seed(0)
    model = Veritate(VOCAB, HIDDEN, LAYERS, FFN, HEADS, SEQ)
    corpus_root = os.path.join(root, "corpus")
    os.makedirs(corpus_root, exist_ok=True)
    with open(os.path.join(corpus_root, CORPUS_STEM + paths.CORPUS_TRAIN_SUFFIX), "wb") as f:
        f.write(CORPUS_TEXT * CORPUS_REPEATS)
    args = {
        "description": "save dump suite fixture",
        "corpus": CORPUS_STEM,
        "shape": {"vocab": VOCAB, "hidden": HIDDEN, "layers": LAYERS,
                  "ffn": FFN, "heads": HEADS, "seq": SEQ},
    }
    buf = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(paths, "MODELS_ROOT", os.path.join(root, "models"))
        mp.setattr(paths, "CORPUS_ROOT", corpus_root)
        mp.delenv(save.PLUGIN_ID_ENV, raising=False)
        if model_type is None:
            mp.delenv(save.MODEL_TYPE_ENV, raising=False)
        else:
            mp.setenv(save.MODEL_TYPE_ENV, model_type)
        with redirect_stdout(buf):
            ckpt = save.save(model, name, STEP, args=args)
        step_dir = paths.hook_step_dir(name, STEP)
    return {"ckpt": ckpt, "step_dir": step_dir, "written": set(os.listdir(step_dir)),
            "out": buf.getvalue()}


@pytest.fixture(scope="module")
def language_run(tmp_path_factory):
    return _run_save(str(tmp_path_factory.mktemp("language")), "toy_language", None)


@pytest.fixture(scope="module")
def code_run(tmp_path_factory):
    return _run_save(str(tmp_path_factory.mktemp("code")), "toy_code", MODEL_TYPE_CODE)


@pytest.fixture(scope="module")
def statistical_run(tmp_path_factory):
    return _run_save(str(tmp_path_factory.mktemp("statistical")), "toy_statistical",
                     MODEL_TYPE_STATISTICAL)


def test_default_type_writes_every_language_dump(language_run):
    """save() with no model_type writes every LANGUAGE_DUMPS artifact into hooks/step_1/."""
    assert _language_filenames() - language_run["written"] == set()


def test_default_type_reports_no_dump_failure(language_run):
    """No dump prints DUMP FAILED during a default-type save."""
    assert DUMP_FAILED_MARK not in language_run["out"]


def test_save_writes_checkpoint_file(language_run):
    """save() returns the path of a checkpoint that exists on disk."""
    assert os.path.isfile(language_run["ckpt"])


def test_code_type_writes_every_language_dump(code_run):
    """model_type=code keeps the full LANGUAGE_DUMPS set (rule 24a)."""
    assert _language_filenames() - code_run["written"] == set()


def test_code_type_reports_no_dump_failure(code_run):
    """No dump prints DUMP FAILED during a code-type save."""
    assert DUMP_FAILED_MARK not in code_run["out"]


def test_statistical_type_writes_no_language_dump(statistical_run):
    """model_type=statistical emits zero LANGUAGE_DUMPS artifacts."""
    assert _language_filenames() & statistical_run["written"] == set()


def test_statistical_type_still_writes_probe(statistical_run):
    """The architecture probe runs for statistical models despite the language gate."""
    assert paths.HOOK_ARTIFACTS[PROBE_ARTIFACT][0] in statistical_run["written"]


def test_statistical_type_reports_no_dump_failure(statistical_run):
    """The statistical gate skips language dumps rather than failing them."""
    assert DUMP_FAILED_MARK not in statistical_run["out"]
