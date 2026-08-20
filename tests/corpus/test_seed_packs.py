# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Tests for readers/seeds.py and every shipped seed pack.
# - Seeds are the ceiling on interview mode: measured 2026-08-20, one seed yields
#   about 150 usable openers before the distinct-5-gram ratio hits the 0.90 floor.
#   So a duplicate seed is not cosmetic -- it is a hole in the ceiling, and
#   test_no_duplicate_seeds_anywhere_in_a_pack is the one that matters most here.
# - The content tests are parametrised over whatever packs are on disk, so a new
#   vertical is held to the same bar the day it lands without editing this file.
# - The availability tests exist because a vertical with no seeds must never be
#   selectable: it would silently fall back to a genre's own thin situations list
#   and produce a corpus about the wrong thing.
# tests/corpus/test_seed_packs.py
# ------------------------------------------------------------------------------------
# Imports:

import pytest
from readers import seeds
from teacher import authoring

# ------------------------------------------------------------------------------------
# Constants

VERTICAL = "conversation"
MIN_SEEDS = 1500
MIN_GROUPS = 20
MIN_SEED_CHARS = 15

SHIPPED = sorted(p["vertical"] for p in seeds.list_packs() if p["available"])

# ------------------------------------------------------------------------------------
# Functions

@pytest.fixture(scope="module")
def pack():
    p = seeds.load_pack(VERTICAL)
    assert p is not None, "the conversation seed pack must ship"
    return p


@pytest.fixture(scope="module")
def all_seeds(pack):
    return [s for g in pack["groups"] for s in g["seeds"]]


def _seeds_of(vertical):
    return [s for g in seeds.load_pack(vertical)["groups"] for s in g["seeds"]]


def test_at_least_the_conversation_pack_ships():
    """Guards the parametrisation below: an empty SHIPPED list would make every
    per-pack test vanish silently rather than fail."""
    assert VERTICAL in SHIPPED


@pytest.mark.parametrize("vertical", SHIPPED)
def test_each_pack_is_large_enough_to_be_useful(vertical):
    """1,500 seeds is roughly 225,000 conversations at the measured yield."""
    assert len(_seeds_of(vertical)) >= MIN_SEEDS


@pytest.mark.parametrize("vertical", SHIPPED)
def test_each_pack_is_split_into_selectable_topics(vertical):
    """One undifferentiated file was the thing the user asked to avoid."""
    groups = seeds.load_pack(vertical)["groups"]
    assert len(groups) >= MIN_GROUPS
    for g in groups:
        assert g["id"] and g["label"] and g["seeds"]


@pytest.mark.parametrize("vertical", SHIPPED)
def test_group_ids_are_unique(vertical):
    """Duplicate ids would make a topic unselectable in the UI."""
    ids = [g["id"] for g in seeds.load_pack(vertical)["groups"]]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("vertical", SHIPPED)
def test_no_duplicate_seeds_anywhere_in_a_pack(vertical):
    """A repeated seed buys no extra openers, so it silently lowers the ceiling
    while looking like added coverage. Checked on the same normalisation the
    opener deduplicator uses, so this catches what that would catch."""
    norm = [authoring.normalize(s) for s in _seeds_of(vertical)]
    seen, dupes = set(), set()
    for s in norm:
        (dupes if s in seen else seen).add(s)
    assert not dupes, f"{vertical}: duplicate seeds {sorted(dupes)[:5]}"


@pytest.mark.parametrize("vertical", SHIPPED)
def test_seeds_are_specific_enough_to_generate_from(vertical):
    """A one-word seed produces the same generic questions every time."""
    too_short = [s for s in _seeds_of(vertical) if len(s) < MIN_SEED_CHARS]
    assert not too_short, f"{vertical}: too vague {too_short[:5]}"


@pytest.mark.parametrize("vertical", SHIPPED)
def test_seeds_are_plain_ascii(vertical):
    """The corpus is byte-level; a stray non-ascii character in a seed becomes a
    stray byte pattern in the training data."""
    bad = [s for s in _seeds_of(vertical) if any(ord(c) > 127 for c in s)]
    assert not bad, f"{vertical}: non-ascii seeds {bad[:3]}"


@pytest.mark.parametrize("vertical", SHIPPED)
def test_seeds_are_subject_phrases_not_sentences(vertical):
    """Seeds are interpolated into 'Subject area: {seed}'. A capitalised, full
    stopped sentence reads as an instruction there and drags the teacher's
    openers toward echoing the seed back verbatim."""
    bad = [s for s in _seeds_of(vertical)
           if s.endswith(".") or s != s.strip() or s[:1].isupper()]
    assert not bad, f"{vertical}: not subject phrases {bad[:5]}"


def test_selecting_topics_narrows_the_seed_list(pack):
    """The whole point of groups: a corpus about pets is not a corpus about grief."""
    first, second = pack["groups"][0], pack["groups"][1]
    picked = seeds.seeds_for(VERTICAL, [first["id"]])
    assert len(picked) == len(first["seeds"])
    both = seeds.seeds_for(VERTICAL, [first["id"], second["id"]])
    assert len(both) == len(first["seeds"]) + len(second["seeds"])


def test_no_topics_means_every_topic(all_seeds):
    """An empty selection is 'all', not 'none' -- the API default must be useful."""
    assert len(seeds.seeds_for(VERTICAL)) == len(all_seeds)


def test_unknown_topic_ids_are_ignored_not_fatal(pack):
    """A stale selection in someone's browser must not break a run."""
    real = pack["groups"][0]["id"]
    got = seeds.seeds_for(VERTICAL, [real, "no_such_group"])
    assert len(got) == len(pack["groups"][0]["seeds"])


def test_a_vertical_with_no_pack_yields_nothing(pack):
    """Refusing here is what stops a run silently using the wrong subjects."""
    assert seeds.seeds_for("no_such_vertical") == []


def test_listing_marks_written_packs_available():
    """Only real packs are selectable, and the count the dashboard sizes a run
    from is the count that is actually on disk."""
    packs = {p["vertical"]: p for p in seeds.list_packs()}
    for vertical in SHIPPED:
        assert packs[vertical]["available"] is True
        assert packs[vertical]["seed_count"] == len(_seeds_of(vertical))


def test_a_planned_but_unwritten_vertical_is_listed_and_disabled(tmp_path, monkeypatch):
    """The roadmap is shown, but a vertical with no file must never be selectable.
    Pointed at an empty directory so this holds however many packs ship."""
    monkeypatch.setattr(seeds, "seeds_dir", lambda: str(tmp_path))
    listed = seeds.list_packs()
    assert len(listed) == len(seeds.PLANNED_VERTICALS)
    for p in listed:
        assert p["available"] is False
        assert p["groups"] == [] and p["seed_count"] == 0


def test_group_counts_in_the_listing_match_the_pack(pack):
    """The dashboard shows these counts and sizes a run from them."""
    listed = {g["id"]: g["count"] for p in seeds.list_packs()
              if p["vertical"] == VERTICAL for g in p["groups"]}
    for g in pack["groups"]:
        assert listed[g["id"]] == len(g["seeds"])


def test_a_malformed_pack_reads_as_absent(tmp_path, monkeypatch):
    """Half-written JSON must not make a vertical look selectable."""
    monkeypatch.setattr(seeds, "seeds_dir", lambda: str(tmp_path))
    (tmp_path / "broken.json").write_text("{not json")
    assert seeds.load_pack("broken") is None
    assert seeds.seeds_for("broken") == []
