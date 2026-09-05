# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The collector's contract: one picture stored once however many paths reach it, a
#   re-run that adds only what is new, and a resolution floor that rejects rather than
#   upscales. Each of these is a corpus defect if it fails silently.
# tests/corpus/test_ingest_images.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
from PIL import Image
from tools import ingest_images

# ------------------------------------------------------------------------------------
# Constants

BIG   = (600, 600)
SMALL = (64, 64)

# ------------------------------------------------------------------------------------
# Functions


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A private data/images root, so a test never touches the real library."""
    monkeypatch.setattr(ingest_images.paths, "IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setattr(ingest_images.paths, "image_set_dir",
                        lambda name: os.path.join(str(tmp_path / "images"), name))
    return tmp_path


def _write(path, size=BIG, color=(10, 120, 200)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _stored(home, set_name="set"):
    d = os.path.join(str(home / "images"), set_name)
    return sorted(n for n in os.listdir(d) if not n.startswith("."))


def test_one_picture_reached_by_two_paths_is_stored_once(home):
    """Photo libraries are full of the same file in several folders."""
    _write(str(home / "a" / "shot.png"))
    _write(str(home / "b" / "copy_of_shot.png"))
    rep = ingest_images.ingest("set", [str(home / "a"), str(home / "b")])
    assert rep["added"] == 1
    assert rep["duplicates"] == 1
    assert len([n for n in _stored(home) if n.endswith(".png")]) == 1


def test_pictures_below_the_floor_are_rejected_not_upscaled(home):
    """A thumbnail blurred up to the training crop teaches the codec the blur."""
    _write(str(home / "src" / "thumb.png"), size=SMALL)
    _write(str(home / "src" / "real.png"), size=BIG, color=(4, 4, 4))
    rep = ingest_images.ingest("set", [str(home / "src")], min_edge=512)
    assert rep["added"] == 1
    assert rep["too_small"] == 1


def test_a_second_run_ingests_only_what_is_new(home):
    """Re-running after adding photos must not re-read the whole library."""
    _write(str(home / "src" / "one.png"), color=(1, 2, 3))
    first = ingest_images.ingest("set", [str(home / "src")])
    _write(str(home / "src" / "two.png"), color=(9, 8, 7))
    second = ingest_images.ingest("set", [str(home / "src")])
    assert first["added"] == 1
    assert second["added"] == 1
    assert second["already_ingested"] == 1
    assert len([n for n in _stored(home) if n.endswith(".png")]) == 2


def test_a_caption_sidecar_is_carried_across(home):
    """Conditioning costs no architecture, so a caption that exists must survive."""
    _write(str(home / "src" / "pic.png"))
    with open(str(home / "src" / "pic.txt"), "w", encoding="utf-8") as handle:
        handle.write("a blue square")
    ingest_images.ingest("set", [str(home / "src")])
    captions = [n for n in _stored(home) if n.endswith(".txt")]
    assert len(captions) == 1
    with open(os.path.join(str(home / "images"), "set", captions[0]), encoding="utf-8") as h:
        assert h.read() == "a blue square"


def test_the_folder_name_becomes_the_caption_when_asked(home):
    """A photo library already encodes captions as folder names."""
    _write(str(home / "src" / "Iceland 2024" / "pic.png"))
    ingest_images.ingest("set", [str(home / "src")], caption_from_folder=True)
    captions = [n for n in _stored(home) if n.endswith(".txt")]
    with open(os.path.join(str(home / "images"), "set", captions[0]), encoding="utf-8") as h:
        assert h.read() == "Iceland 2024"


def test_a_dry_run_writes_nothing(home):
    """The report has to be trustworthy before a library is touched."""
    _write(str(home / "src" / "pic.png"))
    rep = ingest_images.ingest("set", [str(home / "src")], dry_run=True)
    assert rep["added"] == 1
    assert not os.path.isdir(os.path.join(str(home / "images"), "set"))


def test_the_original_is_never_moved(home):
    """This tool reads a person's photo library; it does not rearrange it."""
    src = _write(str(home / "src" / "pic.png"))
    ingest_images.ingest("set", [str(home / "src")])
    assert os.path.isfile(src)


def test_progress_reports_pictures_read(home):
    """A library takes minutes to hash; the caller must be able to show that it is moving."""
    for i in range(3):
        _write(str(home / "src" / f"p{i}.png"), color=(i, i, i))
    seen = []
    ingest_images.ingest("set", [str(home / "src")], progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (3, 3)

