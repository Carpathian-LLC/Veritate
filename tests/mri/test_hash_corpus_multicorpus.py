# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - hash_corpus must accept the multicorpus weighted-spec form ("a:0.6,b:0.3") for
#   FRESH runs, not only a single stem. A fresh 300M run regressed here because the
#   whole spec string was treated as one corpus stem and failed to resolve.
# tests/mri/test_hash_corpus_multicorpus.py
# ------------------------------------------------------------------------------------
# Imports

import os

import pytest

# ------------------------------------------------------------------------------------
# Functions

def _stems(repo_root):
    from veritate_mri.readers import paths
    d = paths.corpus_dir()
    if not os.path.isdir(d):
        return []
    return sorted({f[:-len("_train.bin")] for f in os.listdir(d) if f.endswith("_train.bin")})


def test_hash_corpus_single_stem(repo_root):
    """hash_corpus on a single stem returns a 64-char digest + byte count."""
    from veritate_mri.training import save
    stems = _stems(repo_root)
    if not stems:
        pytest.skip("no corpora present")
    out = save.hash_corpus(stems[0])
    assert len(out["train_sha256"]) == 64
    assert out["train_bytes"] > 0


def test_hash_corpus_weighted_mix(repo_root):
    """hash_corpus parses the weighted multicorpus form and aggregates members."""
    from veritate_mri.training import save
    stems = _stems(repo_root)
    if len(stems) < 2:
        pytest.skip("need >=2 corpora to test a mix")
    spec = f"{stems[0]}:0.6,{stems[1]}:0.4"
    out = save.hash_corpus(spec)
    assert len(out["train_sha256"]) == 64
    assert out["train_bytes"] > 0
    assert len(out.get("members", [])) == 2


def test_hash_corpus_plus_form(repo_root):
    """hash_corpus parses the additive '+' multicorpus form."""
    from veritate_mri.training import save
    stems = _stems(repo_root)
    if len(stems) < 2:
        pytest.skip("need >=2 corpora to test a mix")
    out = save.hash_corpus(f"{stems[0]}+{stems[1]}")
    assert len(out.get("members", [])) == 2
