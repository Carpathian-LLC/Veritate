# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The captioning stage against a fake teacher: sidecars land beside the pictures,
#   captioned pictures are skipped unless asked, captions are cleaned and capped, a
#   failure is counted and not fatal, and the picture rides in the provider's shape.
# tests/corpus/test_caption_images.py
# ------------------------------------------------------------------------------------
# Imports:

import os

import pytest
from PIL import Image
from readers import paths
from tools import caption_images

# ------------------------------------------------------------------------------------
# Functions


class FakeClient:
    def __init__(self, reply="This photo shows a red square on a white wall.", fail_on=()):
        self.reply, self.fail_on, self.calls = reply, set(fail_on), []
        self.provider = {"system_message_style": "inline"}

    def complete(self, messages, **kw):
        self.calls.append(messages)
        content = messages[0]["content"]
        if any(len(part.get("image_url", {}).get("url", "")) > 100 and part["image_url"]["url"]
               .startswith("data:image/jpeg;base64,") for part in content if part.get("type") == "image_url"):
            pass
        else:
            raise AssertionError("no image part in the request")
        if len(self.calls) in self.fail_on:
            raise RuntimeError("teacher down")
        return self.reply


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "IMAGES_ROOT", str(tmp_path / "images"))
    set_dir = os.path.join(str(tmp_path / "images"), "set")
    os.makedirs(set_dir)
    for i in range(4):
        Image.new("RGB", (300, 200), (200, 10 * i, 10)).save(os.path.join(set_dir, f"p{i}.png"))
    return set_dir


def test_every_picture_gets_a_sidecar_with_a_cleaned_caption(home):
    client = FakeClient()
    rep = caption_images.caption_set("set", client=client, concurrency=2)
    assert rep["done"] == 4 and rep["failed"] == 0
    with open(os.path.join(home, "p0.txt"), encoding="utf-8") as handle:
        assert handle.read().strip() == "A red square on a white wall."
    assert len(client.calls) == 4


def test_captioned_pictures_are_skipped_unless_overwrite(home):
    with open(os.path.join(home, "p1.txt"), "w", encoding="utf-8") as handle:
        handle.write("hand written\n")
    rep = caption_images.caption_set("set", client=FakeClient())
    assert rep["already_captioned"] == 1 and rep["done"] == 3
    with open(os.path.join(home, "p1.txt"), encoding="utf-8") as handle:
        assert handle.read().strip() == "hand written"
    rep = caption_images.caption_set("set", client=FakeClient(reply="Redone."), overwrite=True)
    assert rep["done"] == 4
    with open(os.path.join(home, "p1.txt"), encoding="utf-8") as handle:
        assert handle.read().strip() == "Redone."


def test_a_failing_picture_is_counted_not_fatal(home):
    rep = caption_images.caption_set("set", client=FakeClient(fail_on={2}), concurrency=1)
    assert rep["done"] == 3 and rep["failed"] == 1
    assert rep["errors"][0]["error"].startswith("RuntimeError")


def test_progress_and_stop_are_honoured(home):
    seen = []
    stops = iter([False, True])
    rep = caption_images.caption_set("set", client=FakeClient(), concurrency=2,
                                     progress=lambda d, t, n, c: seen.append((d, t)),
                                     should_stop=lambda: next(stops))
    assert rep["stopped"] is True
    assert rep["done"] == 2
    assert seen[-1] == (2, 4)


def test_captions_are_capped_and_stripped_of_preamble():
    assert caption_images.clean_caption('"the image shows a dog running on a beach at sunset"', 5) \
        == "A dog running on a"
    assert caption_images.clean_caption("  many   spaces\nhere ", 40) == "Many spaces here"


def test_prompts_come_from_the_style_or_the_custom_text():
    assert "40 words" in caption_images.prompt_for("sentence", None, 40)
    assert caption_images.prompt_for("tags", None, 40).startswith("List the main subjects")
    assert caption_images.prompt_for("custom", "Name the dish.", 40) == "Name the dish."
    with pytest.raises(ValueError, match="needs a prompt"):
        caption_images.prompt_for("custom", "", 40)


def test_anthropic_gets_its_own_image_part_shape():
    msgs = caption_images.vision_messages({"system_message_style": "field"}, "describe", "QUJD")
    part = msgs[0]["content"][0]
    assert part["type"] == "image" and part["source"]["data"] == "QUJD"
    msgs = caption_images.vision_messages({"system_message_style": "inline"}, "describe", "QUJD")
    assert msgs[0]["content"][1]["type"] == "image_url"
