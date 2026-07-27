# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Guards the corpus_catalog.json shape the dashboard modal depends on:
#   every entry carries family + topic, both drawn from the fixed vocabularies
#   the JS render code (veritate_mri/web/index.js) groups on. Also verifies the
#   coming-soon sft_idk entry stays present with a placeholder train_url and
#   the coming_soon flag set, and _entry_skeleton preserves the two fields.
#   Also guards the mix planner's dashboard contract: every #corpusMix* id the
#   JS looks up exists exactly once in index.html, and no profile id from
#   corpus_mix_profiles.json is hardcoded in the JS.
# tests/mri/test_corpus_catalog_shape.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import re
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO_ROOT, os.path.join(REPO_ROOT, "veritate_mri")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training.sync import corpus_sync

# ------------------------------------------------------------------------------------
# Constants

CATALOG_PATH = os.path.join(
    REPO_ROOT, "veritate_mri", "training", "sync", "corpus_catalog.json"
)

# These must stay in sync with CORPUS_FAMILIES / CORPUS_TOPICS in
# veritate_mri/web/index.js. Adding a new family/topic on either side without
# updating this set fails the test on purpose.
ALLOWED_FAMILIES = {"carpathian", "public"}
ALLOWED_TOPICS = {"chat", "agent", "mcp", "code", "knowledge", "market", "special_sft"}

PROFILES_PATH = os.path.join(REPO_ROOT, "veritate_mri", "data", "corpus_mix_profiles.json")

WEB_DIR  = os.path.join(REPO_ROOT, "veritate_mri", "web")
INDEX_JS = os.path.join(WEB_DIR, "index.js")
INDEX_HTML = os.path.join(WEB_DIR, "index.html")
MIX_ID_RE = re.compile(r'\$\("(corpusMix\w+)"\)')
MIX_SECTION_START = "// ---- Corpus mix planner ----"
MIX_SECTION_END   = "function _corpusCloseLibraryModal"

# ------------------------------------------------------------------------------------
# Functions

def _load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_catalog_is_valid_json_and_has_corpora():
    """The shipped catalog parses and has a non-empty 'corpora' array."""
    data = _load_catalog()
    assert isinstance(data, dict)
    assert isinstance(data.get("corpora"), list)
    assert len(data["corpora"]) > 0


def test_every_entry_has_stem_family_topic():
    """The dashboard modal groups by family -> topic. Missing either would
    dump entries into the 'Other' catch-all instead of the intended group."""
    data = _load_catalog()
    for entry in data["corpora"]:
        stem = entry.get("stem")
        assert stem, f"catalog entry missing stem: {entry!r}"
        assert entry.get("family"), f"{stem}: missing 'family' field"
        assert entry.get("topic"),  f"{stem}: missing 'topic' field"


def test_family_and_topic_values_are_from_the_fixed_vocabulary():
    """Family/topic values must match the JS render code's known lists;
    anything else lands in 'Other' silently."""
    data = _load_catalog()
    for entry in data["corpora"]:
        stem = entry["stem"]
        assert entry["family"] in ALLOWED_FAMILIES, \
            f"{stem}: family={entry['family']!r} not in {ALLOWED_FAMILIES}"
        assert entry["topic"] in ALLOWED_TOPICS, \
            f"{stem}: topic={entry['topic']!r} not in {ALLOWED_TOPICS}"


def test_carpathian_family_has_all_expected_topics():
    """The Carpathian family should cover every topic we publish so the modal
    renders those sub-groups. If we drop a family from the catalog, adjust
    this test — the user's intent was chat/agent/mcp/code/market/special_sft."""
    data = _load_catalog()
    carpathian_topics = {e["topic"] for e in data["corpora"] if e["family"] == "carpathian"}
    for expected in ("chat", "agent", "mcp", "code", "market", "special_sft"):
        assert expected in carpathian_topics, f"Carpathian family missing topic {expected!r}"


def test_public_family_only_has_knowledge_topics():
    """Public corpora (fineweb_edu, wikitext103, etc.) all sit under the
    knowledge topic. If a public entry lands under chat/agent/code/market
    that's likely a mis-tag rather than a real new category."""
    data = _load_catalog()
    for entry in data["corpora"]:
        if entry["family"] != "public":
            continue
        assert entry["topic"] == "knowledge", \
            f"public entry {entry['stem']!r}: expected topic='knowledge', got {entry['topic']!r}"


def test_sft_idk_entry_is_live_and_valid():
    """The IDK SFT corpus is published (COS-hosted zip_bundle). Verify the
    catalog carries an https train_url, both sha256s, both sizes, and the
    special_sft topic mapping the modal groups it under."""
    data = _load_catalog()
    idk = next((e for e in data["corpora"] if e["stem"] == "sft_idk"), None)
    assert idk is not None, "sft_idk entry missing from catalog"
    assert idk.get("family") == "carpathian"
    assert idk.get("topic") == "special_sft"
    assert idk.get("format") == "zip_bundle"
    train_url = idk.get("train_url") or ""
    assert train_url.startswith("https://"), \
        f"sft_idk train_url should be a live https link, got {train_url!r}"
    assert not idk.get("coming_soon"), "sft_idk is live; coming_soon must be false/absent"
    assert len(idk.get("sha256_train") or "") == 64
    assert len(idk.get("sha256_val")   or "") == 64
    assert idk.get("size_train") and idk.get("size_val")


def test_install_refuses_coming_soon_entries():
    """corpus_sync.install() must refuse any coming_soon entry before touching
    the network. Complements the modal's disabled button as a second guard.
    Uses a synthetic entry so the test survives sft_idk (or any other entry)
    going live."""
    fake = {
        "stem": "coming_soon_probe",
        "label": "Probe",
        "family": "carpathian",
        "topic": "chat",
        "format": "zip_bundle",
        "train_url": "PLACEHOLDER_URL",
        "coming_soon": True,
    }
    res = corpus_sync.install(fake)
    assert res["ok"] is False
    assert "coming soon" in (res.get("error") or "").lower()


def test_entry_skeleton_preserves_family_and_topic():
    """_entry_skeleton normalizes catalog entries for the dashboard; if it
    drops family/topic then the JS grouping falls back to the per-stem table,
    which will silently rot as new stems are added."""
    src = {
        "stem": "example_stem",
        "label": "Example",
        "family": "carpathian",
        "topic": "chat",
        "format": "raw_bytes",
    }
    out = corpus_sync._entry_skeleton(src)
    assert out["family"] == "carpathian"
    assert out["topic"] == "chat"


def test_entry_skeleton_defaults_missing_family_topic_to_none():
    """Legacy entries without family/topic must still parse; _entry_skeleton
    fills the missing fields with None so the JS side can fall through to its
    per-stem defaults."""
    src = {"stem": "legacy", "label": "Legacy", "format": "raw_bytes"}
    out = corpus_sync._entry_skeleton(src)
    assert out["family"] is None
    assert out["topic"] is None


def test_mix_planner_ids_exist_in_the_markup():
    """Every #corpusMix* element the mix planner looks up is present exactly
    once in index.html; a typo would make $() return null and the panel would
    fail silently. Ids the JS creates in its own rendered HTML are excluded."""
    with open(INDEX_JS, "r", encoding="utf-8") as f:
        js = f.read()
    with open(INDEX_HTML, "r", encoding="utf-8") as f:
        html = f.read()
    rendered = set(re.findall(r'id="(corpusMix\w+)"', js))
    for ident in sorted(set(MIX_ID_RE.findall(js)) - rendered):
        assert html.count(f'id="{ident}"') == 1, \
            f"{ident}: expected exactly one element in index.html"


def test_mix_planner_profile_names_are_not_baked_into_the_dashboard():
    """Intent profiles are user-editable data (corpus_mix_profiles.json), so
    the dashboard must fetch them; a profile id copied into index.js would go
    stale the moment the file is retuned."""
    with open(INDEX_JS, "r", encoding="utf-8") as f:
        js = f.read()
    assert "/corpus/mix/profiles" in js, "the dashboard must ask the server for the profile list"
    section = js[js.index(MIX_SECTION_START):js.index(MIX_SECTION_END, js.index(MIX_SECTION_START))]
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        shipped = json.load(f)["profiles"]
    for name in shipped:
        assert f'"{name}"' not in section, f"profile id {name!r} is hardcoded in the mix planner JS"


def test_no_claude_curated_language_in_labels_or_descriptions():
    """Marketing language calling Claude out by name got scrubbed for the
    2026-07-16 modal reorg. Labels and descriptions should describe what the
    corpus IS, not who curated it."""
    data = _load_catalog()
    for entry in data["corpora"]:
        stem = entry["stem"]
        label = entry.get("label") or ""
        desc  = entry.get("description") or ""
        for banned in ("Claude", "Claude-authored", "Claude-curated"):
            assert banned not in label, f"{stem}: label mentions {banned!r}"
            assert banned not in desc,  f"{stem}: description mentions {banned!r}"
