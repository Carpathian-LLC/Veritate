# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Offline tests for the code corpus builder: filters, dedup, framing, and
#   deterministic assembly from a synthetic staging dir. No network.
# tests/corpus/test_build_code_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os

import build_code_corpus as bcc

# ------------------------------------------------------------------------------------
# Constants

VALID_PY   = "def add(a, b):\n    return a + b\n" * 20
BROKEN_PY  = "def add(a, b:\n    return a +\n" * 20
VALID_JS   = "function add(a, b) {\n  return a + b;\n}\n" * 20
MINIFIED_JS = "!function(e){var t=" + "e+1;" * 600 + "}(x);"
VALID_CSS  = ".card {\n  color: red;\n  margin: 4px;\n}\n" * 20
MINIFIED_CSS = ".a{color:red}" * 200

# ------------------------------------------------------------------------------------
# Functions

def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def test_accept_py_valid():
    """accept_py accepts syntactically valid python."""
    assert bcc.accept_py(VALID_PY)


def test_accept_py_broken():
    """accept_py rejects python that fails ast.parse."""
    assert not bcc.accept_py(BROKEN_PY)


def test_accept_js_minified():
    """accept_js rejects single-line minified bundles."""
    assert not bcc.accept_js(MINIFIED_JS)


def test_accept_css_minified():
    """accept_css rejects newline-free minified css."""
    assert not bcc.accept_css(MINIFIED_CSS)


def test_dedup_exact():
    """Deduper flags a byte-identical document."""
    d = bcc.Deduper()
    assert not d.is_dup(VALID_PY)
    assert d.is_dup(VALID_PY)


def test_dedup_near():
    """Deduper flags a document differing by one word."""
    d = bcc.Deduper()
    assert not d.is_dup(VALID_JS)
    assert d.is_dup(VALID_JS.replace("add", "sum", 1))


def test_wrap_qa_chatml():
    """_wrap_qa emits the ChatML frame with the endoftext separator."""
    frame = bcc._wrap_qa("How do I sort a list?", "Use sorted(xs).")
    assert frame == ("<|im_start|>user\nHow do I sort a list?<|im_end|>\n"
                     "<|im_start|>assistant\nUse sorted(xs).<|im_end|>\n<|endoftext|>\n")


def _write_staging(tmp_path):
    rows = {"py":   [{"text": f"def f{i}(x):\n    return x * {i}\n" + f"# pad {i}\n" * 40,
                      "score": 4} for i in range(300)],
            "js":   [{"text": f"function f{i}(x) {{\n  return x * {i};\n}}\n" * 12,
                      "score": 4} for i in range(300)],
            "html": [{"text": f"<!doctype html>\n<html>\n<body>\n<div>page {i}</div>\n"
                              + f"<p>row {i}</p>\n" * 30 + "</body>\n</html>\n",
                      "score": None} for i in range(300)],
            "css":  [{"text": f".c{i} {{\n  margin: {i}px;\n  color: blue;\n}}\n" * 10,
                      "score": None} for i in range(300)]}
    for fam, docs in rows.items():
        with open(os.path.join(str(tmp_path), fam + ".jsonl"), "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")


def test_build_deterministic(tmp_path):
    """Two builds from the same staging dir and seed produce identical bytes."""
    _write_staging(tmp_path)
    outs = []
    for tag in ("a", "b"):
        tr = str(tmp_path / f"t_{tag}.bin")
        va = str(tmp_path / f"v_{tag}.bin")
        bcc.build("mixed_files", str(tmp_path), tr, va, 1, 0.02, 7, 3)
        outs.append((_read_bytes(tr), _read_bytes(va)))
    assert outs[0] == outs[1]


def test_build_emits_eot_separators(tmp_path):
    """Built train bin separates documents with the endoftext record separator."""
    _write_staging(tmp_path)
    tr = str(tmp_path / "t.bin")
    va = str(tmp_path / "v.bin")
    bcc.build("py", str(tmp_path), tr, va, 1, 0.02, 7, 3)
    assert _read_bytes(tr).count(b"<|endoftext|>") > 1
