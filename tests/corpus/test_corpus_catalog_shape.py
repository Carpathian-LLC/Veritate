# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - guards the corpus_catalog.json shape the dashboard modal groups on (family +
#   topic from a fixed vocabulary), the coming-soon install refusal, the
#   _entry_skeleton field contract, and the mix-planner dashboard wiring.
# tests/corpus/test_corpus_catalog_shape.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import re

import pytest
from conftest import REPO_ROOT
from training.sync import corpus_sync

# ------------------------------------------------------------------------------------
# Constants

CATALOG_PATH = os.path.join(
    REPO_ROOT, "veritate_mri", "training", "sync", "corpus_catalog.json"
)

# Must stay in sync with CORPUS_FAMILIES / CORPUS_TOPICS in veritate_mri/web/index.js.
ALLOWED_FAMILIES = {"carpathian", "public"}
ALLOWED_TOPICS = {"chat", "agent", "mcp", "code", "knowledge", "market", "special_sft", "memory"}
CARPATHIAN_TOPICS = ("chat", "agent", "mcp", "code", "market", "special_sft")
PUBLIC_TOPIC = "knowledge"
BANNED_CURATION_WORDS = ("Claude", "Claude-authored", "Claude-curated")
SHA256_HEX_LEN = 64

PROFILES_PATH = os.path.join(REPO_ROOT, "veritate_mri", "data", "corpus_mix_profiles.json")

WEB_DIR  = os.path.join(REPO_ROOT, "veritate_mri", "web")
INDEX_JS = os.path.join(WEB_DIR, "index.js")
INDEX_HTML = os.path.join(WEB_DIR, "index.html")
MIX_ID_RE = re.compile(r'\$\("(corpusMix\w+)"\)')
MIX_SECTION_START = "// ---- Corpus mix planner ----"
MIX_SECTION_END   = "function _corpusCloseLibraryModal"

COMING_SOON_ENTRY = {
    "stem": "coming_soon_probe",
    "label": "Probe",
    "family": "carpathian",
    "topic": "chat",
    "format": "zip_bundle",
    "train_url": "PLACEHOLDER_URL",
    "coming_soon": True,
}

# ------------------------------------------------------------------------------------
# Functions

def _load_catalog():
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _ids(entries):
    return [e.get("stem") or repr(e) for e in entries]


CORPORA = _load_catalog()["corpora"]
ENTRY_PARAM = pytest.mark.parametrize("entry", CORPORA, ids=_ids(CORPORA))
PUBLIC = [e for e in CORPORA if e.get("family") == "public"]


def test_catalog_parses_to_an_object():
    """The shipped catalog parses to a JSON object."""
    assert isinstance(_load_catalog(), dict)


def test_catalog_corpora_is_a_non_empty_list():
    """The catalog's corpora key is a non-empty array."""
    assert len(_load_catalog()["corpora"]) > 0


@ENTRY_PARAM
def test_entry_has_a_stem(entry):
    """Every catalog entry carries a stem."""
    assert entry.get("stem")


@ENTRY_PARAM
def test_entry_family_is_in_the_vocabulary(entry):
    """Every catalog entry's family is one the dashboard modal groups on."""
    assert entry.get("family") in ALLOWED_FAMILIES


@ENTRY_PARAM
def test_entry_topic_is_in_the_vocabulary(entry):
    """Every catalog entry's topic is one the dashboard modal groups on."""
    assert entry.get("topic") in ALLOWED_TOPICS


@pytest.mark.parametrize("topic", CARPATHIAN_TOPICS)
def test_carpathian_family_covers_topic(topic):
    """The carpathian family publishes an entry under each shipped topic."""
    assert topic in {e["topic"] for e in CORPORA if e["family"] == "carpathian"}


@pytest.mark.parametrize("entry", PUBLIC, ids=_ids(PUBLIC))
def test_public_entries_are_knowledge_topic(entry):
    """Every public-family entry sits under the knowledge topic."""
    assert entry["topic"] == PUBLIC_TOPIC


@pytest.fixture
def sft_idk():
    entry = next((e for e in CORPORA if e["stem"] == "sft_idk"), None)
    if entry is None:
        pytest.fail("sft_idk entry missing from catalog")
    return entry


def test_sft_idk_train_url_is_live_https(sft_idk):
    """The sft_idk entry publishes a live https train_url, not a placeholder."""
    assert (sft_idk.get("train_url") or "").startswith("https://")


def test_sft_idk_is_not_coming_soon(sft_idk):
    """The sft_idk entry is published, so coming_soon is false or absent."""
    assert not sft_idk.get("coming_soon")


def test_sft_idk_is_a_zip_bundle(sft_idk):
    """The sft_idk entry declares the zip_bundle format."""
    assert sft_idk.get("format") == "zip_bundle"


def test_sft_idk_is_grouped_under_special_sft(sft_idk):
    """The sft_idk entry is grouped under carpathian/special_sft."""
    assert (sft_idk.get("family"), sft_idk.get("topic")) == ("carpathian", "special_sft")


def test_sft_idk_carries_both_checksums(sft_idk):
    """The sft_idk entry carries a full sha256 for the train and val halves."""
    assert [len(sft_idk.get("sha256_train") or ""), len(sft_idk.get("sha256_val") or "")] == \
        [SHA256_HEX_LEN, SHA256_HEX_LEN]


def test_sft_idk_carries_both_sizes(sft_idk):
    """The sft_idk entry carries a train and a val byte size."""
    assert [bool(sft_idk.get("size_train")), bool(sft_idk.get("size_val"))] == [True, True]


def test_install_refuses_a_coming_soon_entry():
    """corpus_sync.install() refuses a coming_soon entry."""
    assert corpus_sync.install(COMING_SOON_ENTRY)["ok"] is False


def test_install_refusal_names_coming_soon():
    """corpus_sync.install()'s refusal of a coming_soon entry says why."""
    assert "coming soon" in (corpus_sync.install(COMING_SOON_ENTRY).get("error") or "").lower()


def test_entry_skeleton_preserves_family_and_topic():
    """_entry_skeleton keeps the family and topic the dashboard groups on."""
    out = corpus_sync._entry_skeleton(
        {"stem": "example_stem", "label": "Example", "family": "carpathian",
         "topic": "chat", "format": "raw_bytes"})
    assert (out["family"], out["topic"]) == ("carpathian", "chat")


def test_entry_skeleton_defaults_missing_family_topic_to_none():
    """_entry_skeleton fills a legacy entry's absent family and topic with None."""
    out = corpus_sync._entry_skeleton({"stem": "legacy", "label": "Legacy", "format": "raw_bytes"})
    assert (out["family"], out["topic"]) == (None, None)


def test_mix_planner_ids_exist_exactly_once_in_the_markup():
    """Every #corpusMix* id the planner JS looks up appears exactly once in index.html."""
    js, html = _read(INDEX_JS), _read(INDEX_HTML)
    rendered = set(re.findall(r'id="(corpusMix\w+)"', js))
    missing = [i for i in sorted(set(MIX_ID_RE.findall(js)) - rendered) if html.count(f'id="{i}"') != 1]
    assert missing == []


def test_dashboard_fetches_the_profile_list():
    """The dashboard asks the server for the mix profile list."""
    assert "/corpus/mix/profiles" in _read(INDEX_JS)


def test_mix_profile_ids_are_not_baked_into_the_dashboard():
    """No shipped mix profile id is hardcoded in the mix planner JS section."""
    js = _read(INDEX_JS)
    start = js.index(MIX_SECTION_START)
    section = js[start:js.index(MIX_SECTION_END, start)]
    shipped = json.loads(_read(PROFILES_PATH))["profiles"]
    assert [n for n in shipped if f'"{n}"' in section] == []


@ENTRY_PARAM
def test_entry_label_has_no_curation_marketing(entry):
    """No catalog label names who curated the corpus."""
    assert [w for w in BANNED_CURATION_WORDS if w in (entry.get("label") or "")] == []


@ENTRY_PARAM
def test_entry_description_has_no_curation_marketing(entry):
    """No catalog description names who curated the corpus."""
    assert [w for w in BANNED_CURATION_WORDS if w in (entry.get("description") or "")] == []
