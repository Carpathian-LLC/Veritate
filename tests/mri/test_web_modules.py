# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the dashboard has no test harness of its own, so this pins what can be pinned from
#   the outside: every script and stylesheet index.html loads exists, every web module
#   parses (node --check, skipped without node), the standalone modules touch no
#   index.js symbol, and a concern that moved out of index.js stays out.
# tests/mri/test_web_modules.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import re
import shutil
import subprocess

import pytest
from conftest import REPO_ROOT

# ------------------------------------------------------------------------------------
# Constants

WEB = os.path.join(REPO_ROOT, "veritate_mri", "web")
# self-contained modules: one window.<Name> export, nothing read from index.js
STANDALONE = ("image_mri.js", "generation_images.js", "image_live.js", "tutorial.js", "prune.js",
              "wiki.js")
INDEX_ONLY = ("imgMriState", "_imriSvgLine", "IMRI_SHARED_STYLE", "imgGenPanel", "_imgGenRun",
              "imgLiveState", "_imgLiveParseCsv", "IMG_LIVE_STAGES", "_imgLiveRender")

# ------------------------------------------------------------------------------------
# Functions


def _read(name):
    with open(os.path.join(WEB, name), encoding="utf-8") as handle:
        return handle.read()


def test_every_static_asset_index_html_loads_exists():
    """A missing script or stylesheet fails silently in the browser; catch it here."""
    refs = re.findall(r'(?:src|href)="/static/([^"]+)"', _read("index.html"))
    assert refs
    for ref in refs:
        assert os.path.isfile(os.path.join(WEB, ref)), ref


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("name", sorted(f for f in os.listdir(WEB) if f.endswith(".js")))
def test_every_web_script_parses(name):
    """node --check catches the syntax slip a browser would report only in its console."""
    p = subprocess.run(["node", "--check", os.path.join(WEB, name)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("name", STANDALONE)
def test_standalone_modules_are_iifes_with_one_export(name):
    """A module owns its concern behind one window.<Name>; nothing of index.js leaks in."""
    src = _read(name)
    assert src.splitlines()[1] == f"/* veritate_mri/web/{name} */"
    assert "(function () {" in src and '"use strict";' in src
    assert len(re.findall(r"window\.[A-Z]\w+ = ", src)) <= 1
    for sym in ("_trEsc(", "_trOpenImageContinue(", "learningTimelineName", "trainState.", "$(\""):
        assert sym not in src, sym


def test_the_image_views_stay_out_of_index_js():
    """The image MRI, the image generation layout and the image run live view moved to
    their own modules; index.js reaches them only through their window exports."""
    src = _read("index.js")
    for sym in INDEX_ONLY:
        assert sym not in src, sym
    assert "ImageMri.create(" in src
    assert "ImageLive.create(" in src and "ImageLive.TRAINER_ID" in src


def test_the_generation_tab_has_the_kind_switch_and_the_image_layout():
    """index.html carries the text | images selector and the #imgGenView layout the module fills."""
    html = _read("index.html")
    assert html.count('name="genKind"') == 2
    for el in ("imgGenView", "imgGenModel", "imgGenRun", "imgGenTrace", "imgGenMri"):
        assert f'id="{el}"' in html, el
    css = _read("generation_images.css")
    assert '.tab-body[data-tab="generation"].is-images' in css


def test_the_wiki_tab_has_no_container_the_module_never_fills():
    """Every id in the Wiki tab is written by wiki.js: an empty box is a design defect."""
    html = _read("index.html")
    tab = html[html.index('<div class="tab-body" data-tab="wiki">'):]
    tab = tab[:tab.index("<!-- shared dropdown content")]
    ids = re.findall(r'id="([^"]+)"', tab)
    assert set(ids) == {"wikiFilter", "wikiToc", "wikiArticle", "wikiNavEmpty"}
    module = _read("wiki.js")
    for el in ids:
        assert f'"{el}"' in module, el


def test_index_js_reaches_the_wiki_only_through_its_export():
    """Wiki rendering lives in the module; index.js holds no wiki state or markup."""
    src = _read("index.js")
    assert "window.Wiki.load()" in src
    for sym in ("wikiState", "renderWikiToc", "ensureWikiLoaded", "wikiSubtabs"):
        assert sym not in src, sym


def test_the_fast_mode_picker_offers_only_modes_a_model_can_serve():
    """The multi-byte-head modes left with the retired trainers; the picker must not offer them."""
    html = _read("index.html")
    picker = html[html.index('id="genFastMode"'):]
    picker = picker[:picker.index("</select>")]
    assert 'value="mtp' not in picker and 'value="kv"' in picker and 'value="stream"' in picker
