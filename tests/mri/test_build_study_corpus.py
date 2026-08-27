# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers tools/build_study_corpus.py, the general study-form generator of the
#   tell-it-once loop. build_fact_sft renders {subj,obj} flashcards; this renders the
#   same E4 mechanism (varied forms, content in the assistant turn) for ARBITRARY
#   documents so code and long-form prose consolidate through one path.
# - the load-bearing invariant is the assistant-turn rule: consolidation runs under
#   loss_mask=assistant, so a form whose answer is not the material trains the model on
#   its own prose instead of on the document. That is exactly the defect that made
#   raw-transcript sleep degrade the model (failures.md 2026-08-21 m2).
# tests/mri/test_build_study_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import random

from tools import build_study_corpus as bsc

# ------------------------------------------------------------------------------------
# Constants

PY_SRC = '''"""mod."""


def alpha(x):
    """Return x doubled, padded so the chunk clears MIN_CHUNK."""
    total = x * 2
    return total


class Beta:
    """A class whose body is long enough to survive the minimum-size filter."""

    def method(self):
        return 42
'''

C_SRC = """#include <stdio.h>

int add_values(int a, int b)
{
    int total = a + b;
    return total;
}

static void log_line(const char *msg)
{
    printf("%s\\n", msg);
}
"""

MD_SRC = """# Title

Intro paragraph that is comfortably longer than the minimum chunk size for tests.

## Section One

Body text for section one, also written long enough to clear the minimum size.

## Section Two

Body text for section two, likewise padded out past the minimum chunk length.
"""

# ------------------------------------------------------------------------------------
# Functions


def _answer(form):
    return form.split("<|im_start|>assistant\n")[1].rsplit("<|im_end|>", 1)[0]


def _prompt(form):
    return form.split("<|im_start|>assistant")[0]


def test_python_chunks_by_function_and_class():
    """AST chunking labels each top-level definition by name."""
    labels = [lb for lb, _ in bsc.chunk_document(PY_SRC, "mod.py")]
    assert "mod.py::alpha" in labels
    assert "mod.py::Beta" in labels


def test_c_chunks_by_brace_matching():
    """C chunking recovers whole function bodies including the closing brace."""
    chunks = dict(bsc.chunk_document(C_SRC, "engine.c"))
    assert "engine.c::add_values" in chunks
    assert chunks["engine.c::add_values"].rstrip().endswith("}")


def test_markdown_chunks_by_heading():
    """Each heading owns the text up to the next heading."""
    labels = [lb for lb, _ in bsc.chunk_document(MD_SRC, "doc.md")]
    assert "doc.md: Section One" in labels
    assert "doc.md: Section Two" in labels


def test_unparseable_python_falls_back_to_prose():
    """A syntax error costs one file's structure, not the build."""
    broken = "def (((:\n\n" + ("paragraph text padded well past the minimum chunk size. " * 3)
    chunks = bsc.chunk_document(broken, "broken.py")
    assert chunks and all("::" not in lb for lb, _ in chunks)


def test_oversize_chunk_splits_on_line_boundaries():
    """Splitting never breaks a line, so a code chunk cannot end mid-token."""
    body = "".join(f"    line_{i} = {i}\n" for i in range(200))
    parts = list(bsc._split_oversize("big", body, 400))
    assert len(parts) > 1
    assert all(len(seg) <= 400 or "\n" not in seg for _, seg in parts)
    assert "".join(seg for _, seg in parts) == body


def test_every_form_answers_with_the_material():
    """The assistant turn is always the document's own bytes (or the label, for the
    reverse-direction form) -- never model prose, which is what loss_mask=assistant
    would otherwise train on."""
    text = "def f(x):\n    return x + 1\n" * 6
    forms = bsc.study_forms("m.py::f", text, random.Random(0), per_chunk=8)
    for form in forms:
        ans = _answer(form)
        assert ans in text or ans == "m.py::f", ans


def test_identify_form_trains_the_reverse_direction():
    """At least one form maps body -> label: E4 showed both directions are needed."""
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"
    forms = bsc.study_forms("src.txt::x", text, random.Random(0), per_chunk=8)
    assert any(_answer(f) == "src.txt::x" for f in forms)


def test_infill_answer_is_the_removed_gap():
    """The infill form's answer is exactly the span cut out of the prompt."""
    text = "".join(f"token{i} " for i in range(60))
    forms = bsc.study_forms("f.txt::g", text, random.Random(1), per_chunk=8)
    gaps = [f for f in forms if "<<<gap>>>" in _prompt(f)]
    assert gaps
    pre, post = _prompt(gaps[0]).split("<<<gap>>>")
    assert pre.split(":\n", 1)[1] + _answer(gaps[0]) + post.removesuffix("<|im_end|>\n") == text


def test_build_writes_bins_and_manifest(tmp_path):
    """build() emits train/val bins plus an auditable chunk manifest."""
    (tmp_path / "mod.py").write_text(PY_SRC)
    (tmp_path / "doc.md").write_text(MD_SRC)
    out = tmp_path / "out"
    nf, nc, ne, tb, vb = bsc.build([str(tmp_path)], stem="study", out_dir=str(out), seed=0)
    assert nf == 2 and nc > 0 and ne > 0
    assert tb > 0 and vb > 0
    assert os.path.isfile(out / "study_train.bin")
    assert os.path.isfile(out / "study_val.bin")
    manifest = json.loads((out / "study_chunks.json").read_text())
    assert len(manifest) == nc and all("label" in m for m in manifest)


def test_code_only_filter_skips_prose(tmp_path):
    """--code-only restricts the walk to source extensions."""
    (tmp_path / "mod.py").write_text(PY_SRC)
    (tmp_path / "notes.md").write_text(MD_SRC)
    out = tmp_path / "out"
    nf, _nc, _ne, _tb, _vb = bsc.build([str(tmp_path)], stem="study", out_dir=str(out),
                                       exts=bsc.CODE_EXT, seed=0)
    assert nf == 1


def test_holdout_chunks_never_reach_the_training_bins(tmp_path):
    """The exam's control set must be chunks the run never saw. The train/val split is
    by exchange, so the same chunk lands in both; only a chunk-level holdout is a
    control."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(10):
        (src / f"m{i}.py").write_text(
            f"def fn_{i}(x):\n    '''unique marker zzq{i} padded past the minimum.'''\n"
            f"    return x + {i}\n")
    out = tmp_path / "out"
    _nf, nc, _ne, _tb, _vb = bsc.build([str(src)], stem="s", out_dir=str(out),
                                       seed=0, holdout_frac=0.4)
    exam = json.loads((out / "s_exam.json").read_text())
    assert len(exam["holdout"]) == 4 and len(exam["studied"]) == 6
    assert nc == len(exam["studied"])
    trained = (out / "s_train.bin").read_bytes() + (out / "s_val.bin").read_bytes()
    for chunk in exam["holdout"]:
        assert chunk["label"].encode() not in trained, chunk["label"]
    studied_labels = {c["label"] for c in exam["studied"]}
    assert studied_labels.isdisjoint({c["label"] for c in exam["holdout"]})


def test_limit_caps_the_chunk_budget(tmp_path):
    """A run sized to a fixed budget caps chunks after shuffling, so the sample stays
    representative rather than being the first N files walked."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(20):
        (src / f"m{i}.py").write_text(
            f"def fn_{i}(x):\n    '''padded docstring well past the minimum size.'''\n"
            f"    return x + {i}\n")
    out = tmp_path / "out"
    _nf, nc, _ne, _tb, _vb = bsc.build([str(src)], stem="s", out_dir=str(out),
                                       seed=0, limit=8, holdout_frac=0.25)
    exam = json.loads((out / "s_exam.json").read_text())
    assert len(exam["studied"]) + len(exam["holdout"]) == 8
    assert nc == len(exam["studied"]) == 6
