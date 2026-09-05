# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The Images flow's discovery: which picture sets and codecs exist. Counts are what
#   the picker shows next to each set, so they must count pictures and captions and
#   nothing else.
# tests/mri/test_readers_images.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
from readers import images, paths

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setattr(paths, "CODEC_ROOT", str(tmp_path / "codecs"))
    return tmp_path


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"x")


def test_no_roots_means_empty_lists_not_errors(home):
    assert images.list_sets() == []
    assert images.list_codecs() == []


def test_sets_report_picture_and_caption_counts(home):
    root = str(home / "images")
    _touch(os.path.join(root, "mine", "a.png"))
    _touch(os.path.join(root, "mine", "a.txt"))
    _touch(os.path.join(root, "mine", "b.JPG"))
    _touch(os.path.join(root, "mine", ".ingest.json"))
    _touch(os.path.join(root, "other", "c.webp"))
    _touch(os.path.join(root, ".hidden", "d.png"))
    assert images.list_sets() == [
        {"name": "mine", "images": 2, "captions": 1},
        {"name": "other", "images": 1, "captions": 0},
    ]


def test_codecs_are_listed_by_name(home):
    root = str(home / "codecs")
    _touch(os.path.join(root, "mine_codec.codec.pt"))
    _touch(os.path.join(root, "notes.txt"))
    assert images.list_codecs() == [{"name": "mine_codec"}]
