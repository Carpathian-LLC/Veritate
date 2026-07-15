# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - zip_bundle corpus install: _extract_zip_bundle pulls <stem>_train.bin +
#   <stem>_val.bin out of one archive and always deletes the zip; the corpus
#   builders (agent, mcp) emit deterministic, correctly framed bytes.
# tests/mri/test_corpus_library_zip.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys
import zipfile

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri"),
           os.path.join(REPO_ROOT, "veritate_mri", "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.sync import corpus_sync
import build_agent_corpus
import build_mcp_corpus

# ------------------------------------------------------------------------------------
# Constants

STEM = "tiny"
TEST_SEED = 7
TEST_TARGET_MB = 1

# ------------------------------------------------------------------------------------
# Functions

def _make_zip(tmp_path, members):
    zp = os.path.join(tmp_path, f"{STEM}.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return zp


def test_extract_zip_bundle_extracts_both_and_deletes_zip(tmp_path):
    """_extract_zip_bundle writes train+val bins from the archive and removes the zip."""
    tmp = str(tmp_path)
    zp = _make_zip(tmp, {f"folder/{STEM}_train.bin": b"TRAIN", f"folder/{STEM}_val.bin": b"VAL"})
    train, val = os.path.join(tmp, f"{STEM}_train.bin"), os.path.join(tmp, f"{STEM}_val.bin")
    err = corpus_sync._extract_zip_bundle(zp, train, val)
    assert err is None
    assert open(train, "rb").read() == b"TRAIN"
    assert open(val, "rb").read() == b"VAL"
    assert not os.path.exists(zp)


def test_extract_zip_bundle_missing_train_member_errors(tmp_path):
    """A zip without <stem>_train.bin returns an error and still deletes the zip."""
    tmp = str(tmp_path)
    zp = _make_zip(tmp, {f"{STEM}_val.bin": b"VAL"})
    err = corpus_sync._extract_zip_bundle(
        zp, os.path.join(tmp, f"{STEM}_train.bin"), os.path.join(tmp, f"{STEM}_val.bin"))
    assert err is not None and "no" in err
    assert not os.path.exists(zp)


def test_install_refuses_coming_soon():
    """install() rejects an entry flagged coming_soon before touching the network."""
    res = corpus_sync.install({"stem": STEM, "format": "zip_bundle",
                               "train_url": "https://example.com/x.zip", "coming_soon": True})
    assert res["ok"] is False and "coming soon" in res["error"]


def test_build_agent_corpus_deterministic(tmp_path):
    """Two agent builds with the same seed produce identical bytes."""
    outs = []
    for tag in ("a", "b"):
        t, v = os.path.join(str(tmp_path), f"{tag}_t.bin"), os.path.join(str(tmp_path), f"{tag}_v.bin")
        build_agent_corpus.build(t, v, TEST_TARGET_MB, 0.02, TEST_SEED)
        outs.append(open(t, "rb").read())
    assert outs[0] == outs[1]


def test_build_agent_corpus_frames_valid(tmp_path):
    """Agent output is ChatML-framed and every tool_call payload is one-line valid JSON."""
    t, v = os.path.join(str(tmp_path), "t.bin"), os.path.join(str(tmp_path), "v.bin")
    build_agent_corpus.build(t, v, TEST_TARGET_MB, 0.02, TEST_SEED)
    text = open(t, "rb").read().decode("utf-8")
    assert text.count("<|im_start|>") and text.count("<|endoftext|>")
    calls = [seg.split("\n</tool_call>")[0] for seg in text.split("<tool_call>\n")[1:]]
    assert calls
    for c in calls:
        parsed = json.loads(c)
        assert set(parsed) == {"name", "arguments"}


def test_build_mcp_corpus_deterministic(tmp_path):
    """Two MCP builds with the same seed produce identical bytes."""
    outs = []
    for tag in ("a", "b"):
        t, v = os.path.join(str(tmp_path), f"{tag}_t.bin"), os.path.join(str(tmp_path), f"{tag}_v.bin")
        build_mcp_corpus.build(t, v, TEST_TARGET_MB, 0.02, TEST_SEED)
        outs.append(open(t, "rb").read())
    assert outs[0] == outs[1]


def test_build_mcp_corpus_protocol_lines_valid(tmp_path):
    """Every MCP transcript line after the arrow prefix parses as JSON-RPC 2.0."""
    t, v = os.path.join(str(tmp_path), "t.bin"), os.path.join(str(tmp_path), "v.bin")
    build_mcp_corpus.build(t, v, TEST_TARGET_MB, 0.02, TEST_SEED)
    lines = [l for l in open(t, "rb").read().decode("utf-8").splitlines()
             if l.startswith(("-> ", "<- "))]
    assert lines
    for l in lines:
        assert json.loads(l[3:])["jsonrpc"] == "2.0"
