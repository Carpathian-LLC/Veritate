# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Contract test for extension dataset download: download() fetches the catalog's single hosted
#   archive (here a local tar.gz via a file:// url, so no network) and extracts its CSVs into the
#   per-extension data dir. Deterministic, temp roots.
# extensions/tests/test_data.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import sys
import tarfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
for _p in (os.path.join(REPO, "veritate_mri"), REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from extensions import registry
from extensions import data as ext_data

# ------------------------------------------------------------------------------------
# Tests

def test_download_extracts_single_archive(tmp_path, monkeypatch):
    """download() pulls the dataset's single tar.gz from its catalog url and extracts the CSVs into installed/<id>/data/extension_data/<source>."""
    can = str(tmp_path / "canonical")
    inst = str(tmp_path / "installed")
    ext = os.path.join(can, "foo")
    os.makedirs(ext)

    src = tmp_path / "src"
    os.makedirs(src)
    for name in ("a.csv", "b.csv"):
        with open(src / name, "w", encoding="utf-8") as f:
            f.write("1\n")
    archive = str(tmp_path / "stocks.tar.gz")
    with tarfile.open(archive, "w:gz") as t:
        t.add(str(src / "a.csv"), arcname="a.csv")
        t.add(str(src / "b.csv"), arcname="b.csv")

    with open(os.path.join(ext, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"id": "foo"}, f)
    with open(os.path.join(ext, "data_catalog.json"), "w", encoding="utf-8") as f:
        json.dump({"datasets": [{"source": "stocks", "url": "file://" + archive}]}, f)

    monkeypatch.setattr(registry, "CANONICAL_ROOT", can)
    monkeypatch.setattr(registry, "INSTALLED_ROOT", inst)
    monkeypatch.setattr(registry, "DISABLED_PATH", str(tmp_path / "disabled.json"))
    monkeypatch.setattr(registry, "logmod", types.SimpleNamespace(ok=lambda *a: None, error=lambda *a: None))
    monkeypatch.setattr(ext_data, "INSTALLED_ROOT", inst)
    monkeypatch.setattr(ext_data, "logmod", types.SimpleNamespace(ok=lambda *a: None, error=lambda *a: None))

    res = ext_data.download("foo", "stocks")
    assert res["ok"] is True
    assert res["files"] == 2
    assert os.path.isfile(os.path.join(inst, "foo", "data", "extension_data", "stocks", "a.csv"))


def test_download_extracts_zip_archive(tmp_path, monkeypatch):
    """download() also handles a .zip bucket object (the type is detected from the url suffix)."""
    import zipfile
    can = str(tmp_path / "canonical")
    inst = str(tmp_path / "installed")
    ext = os.path.join(can, "foo")
    os.makedirs(ext)
    archive = str(tmp_path / "indices.zip")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("x.csv", "1\n")
    with open(os.path.join(ext, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"id": "foo"}, f)
    with open(os.path.join(ext, "data_catalog.json"), "w", encoding="utf-8") as f:
        json.dump({"datasets": [{"source": "indices", "url": "file://" + archive}]}, f)
    monkeypatch.setattr(registry, "CANONICAL_ROOT", can)
    monkeypatch.setattr(registry, "INSTALLED_ROOT", inst)
    monkeypatch.setattr(registry, "DISABLED_PATH", str(tmp_path / "disabled.json"))
    monkeypatch.setattr(registry, "logmod", types.SimpleNamespace(ok=lambda *a: None, error=lambda *a: None))
    monkeypatch.setattr(ext_data, "INSTALLED_ROOT", inst)
    monkeypatch.setattr(ext_data, "logmod", types.SimpleNamespace(ok=lambda *a: None, error=lambda *a: None))
    res = ext_data.download("foo", "indices")
    assert res["ok"] is True
    assert os.path.isfile(os.path.join(inst, "foo", "data", "extension_data", "indices", "x.csv"))
