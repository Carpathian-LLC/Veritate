# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Guards the Distillation tab's HTML/JS/CSS contract. The dashboard is one big
#   static bundle with no build step, so a renamed id fails silently in the browser
#   and nothing catches it. These tests are the substitute for a compiler.
# - The moved-panel checks matter most: #synthPanel and #authorPanel were lifted out
#   of the Training tab on 2026-08-20, and the failure mode is a duplicate left
#   behind (two elements, same id, the wrong one wired) or a stale CSS rule that
#   hides them again.
# - Pure text inspection of the shipped files; nothing is executed (rule 48).
# tests/mri/test_distillation_tab_wiring.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import re

import pytest
from conftest import REPO_ROOT

# ------------------------------------------------------------------------------------
# Constants

WEB_DIR = os.path.join(REPO_ROOT, "veritate_mri", "web")
INDEX_HTML = os.path.join(WEB_DIR, "index.html")
INDEX_JS = os.path.join(WEB_DIR, "index.js")
INDEX_CSS = os.path.join(WEB_DIR, "index.css")

TAB_ID = "distillation"

# Every id the distillation module reaches for with $("...").
REQUIRED_IDS = (
    "distTeacherName", "distTargetChip", "distTeacherGate", "distContentionNote",
    "distRefreshTarget", "distGotoSettings", "distGateSettings",
    "distConfirmModal", "distConfirmBody", "distConfirmClose", "distConfirmCancel",
    "distConfirmGo",
    "distAuditPanel", "distAuditVerdict", "distAuditChecks", "distAuditDetail",
    "authorProgress", "authorProgressFill",
    "authorProgressLeft", "authorProgressMid", "authorProgressRight",
    "interviewPanel", "interviewGenreList", "interviewJobSelect", "interviewCount",
    "interviewDepth", "interviewConcurrency", "interviewStartBtn", "interviewStopBtn",
    "interviewCostLine", "interviewPlanLine", "interviewStatsRow", "interviewProgress",
    "interviewProgressFill", "interviewProgressLeft", "interviewProgressMid",
    "interviewProgressRight", "interviewBuildRow", "interviewStem", "interviewLabel",
    "interviewBuildBtn", "interviewBuildStatus", "interviewLiveWrap", "interviewLiveOutput",
    "interviewVertical", "interviewVerticalNote", "interviewTopicList",
    "interviewTopicAll", "interviewTopicNone",
)

# Panels that moved out of the Training tab and must now live in this one.
MOVED_PANELS = ("synthPanel", "authorPanel")

# ------------------------------------------------------------------------------------
# Functions

def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


HTML = _read(INDEX_HTML)
JS = _read(INDEX_JS)
CSS = _read(INDEX_CSS)


def _tab_body(name):
    """The markup between this tab-body's opening tag and the next tab-body."""
    start = HTML.index(f'<div class="tab-body" data-tab="{name}">')
    rest = HTML[start + 1:]
    nxt = rest.find('<div class="tab-body" data-tab=')
    return rest if nxt < 0 else rest[:nxt]


def test_the_tab_has_a_nav_entry():
    """Without the nav entry the tab body is unreachable."""
    assert f'<div class="tab" data-tab="{TAB_ID}">' in HTML


def test_the_tab_has_a_body():
    """And without a body the nav entry shows an empty page."""
    assert f'<div class="tab-body" data-tab="{TAB_ID}">' in HTML


def test_activate_tab_accepts_the_name():
    """activateTab silently falls back to generation for names not on its list."""
    valid = re.search(r"const valid = \[([^\]]+)\]", JS).group(1)
    assert f'"{TAB_ID}"' in valid


@pytest.mark.parametrize("el_id", REQUIRED_IDS)
def test_every_id_the_module_uses_exists_exactly_once(el_id):
    """A typo'd id is a no-op in the browser, so each one is pinned here."""
    assert HTML.count(f'id="{el_id}"') == 1, f'{el_id} must appear exactly once in index.html'


@pytest.mark.parametrize("panel", MOVED_PANELS)
def test_moved_panels_are_not_duplicated(panel):
    """The move must not have left a copy behind in the Training tab."""
    assert HTML.count(f'id="{panel}"') == 1


@pytest.mark.parametrize("panel", MOVED_PANELS)
def test_moved_panels_live_in_the_distillation_tab(panel):
    """They have to be inside this tab body, not merely present somewhere."""
    assert f'id="{panel}"' in _tab_body(TAB_ID)


@pytest.mark.parametrize("panel", MOVED_PANELS)
def test_training_active_no_longer_hides_the_panels(panel):
    """The point of the tab is that distillation runs DURING training. A leftover
    `body.training-active #authorPanel { display: none !important }` would undo
    the whole feature while every other test still passed."""
    assert f"body.training-active #{panel}" not in CSS


def test_the_distillation_flow_card_is_not_dimmed_during_training():
    """The Training tab's pointer to this tab must stay clickable mid-run."""
    rule = next(ln for ln in CSS.splitlines()
                if "body.training-active .train-flow-card" in ln and "opacity" in ln)
    assert f'not([data-flow="{TAB_ID}"])' in rule


def test_both_start_buttons_go_through_the_contention_guard():
    """Binding a start button straight to its handler bypasses the confirm."""
    for handler in ("_synthStart", "_authorStart"):
        assert f'addEventListener("click", {handler})' not in JS
    assert JS.count("_distGuardedStart(_synthStart)") == 1
    assert JS.count("_distGuardedStart(_authorStart)") == 1


def test_the_guard_rechecks_before_it_decides():
    """A cached read goes stale: a run can start between opening the tab and
    pressing go, so the guard must fetch fresh status."""
    body = JS[JS.index("function _distGuardedStart"):]
    body = body[:body.index("\nfunction ")]
    assert "_distFetchTarget()" in body


def test_the_mode_choice_is_persisted():
    """Refreshing must not silently drop the user back to the other mode."""
    assert "DIST_MODE_STORE" in JS
    assert "localStorage.setItem(DIST_MODE_STORE" in JS


def test_the_tab_rehydrates_a_running_job_on_activation():
    """The restore used to hang off the training flow picker, which no longer
    selects these flows. If this regresses, a refresh mid-run shows an idle tab
    while the job is still burning teacher calls."""
    body = JS[JS.index("function _distOnTabActivated"):]
    body = body[:body.index("\ndocument.addEventListener")]
    assert "_authorStored()" in body
    assert "_authorPollStart()" in body
    assert "_synthReattach()" in body


def test_progress_is_read_from_the_server_plan():
    """Progress denominators come from the persisted plan, not from a variable
    that a page refresh would clear."""
    body = JS[JS.index("function _authorRenderProgress"):]
    body = body[:body.index("\nfunction ")]
    assert "s.plan" in body
    assert "target_bytes" in body
    assert "total_calls" in body


def test_the_audit_report_is_rendered_when_a_build_returns_one():
    """The acceptance gate is only useful if the user is shown its verdict."""
    assert "_distRenderAudit(d.audit)" in JS


def test_progress_uses_call_counters_not_the_record_count():
    """`completed` on the status route counts RECORDS in samples.jsonl, while the
    plan's denominator counts CALLS. One teacher call yields many records, so
    mixing them reads past 100%. Measured on job 0bc3f270a674: 21 records from
    2 successful calls."""
    body = JS[JS.index("function _authorRenderProgress"):]
    body = body[:body.index("\nfunction ")]
    assert "s.calls_remaining" in body
    assert "s.calls_ok" in body
    assert "Number(s.completed || 0) + Number(s.failed || 0)" not in body


def test_interview_is_a_registered_mode_and_the_default():
    """Two-pass is the method that clears the acceptance gate (344 B median vs
    120 B for dialogue-scripting), so it is what a new user lands on."""
    assert '"interview"' in JS
    assert 'const DIST_DEFAULT_MODE = "interview"' in JS
    assert 'data-mode="interview"' in HTML


def test_the_interview_start_button_goes_through_the_contention_guard():
    """Same rule as the other two: never bind a start handler directly."""
    assert 'addEventListener("click", _ivStart)' not in JS
    assert "_distGuardedStart(_ivStart)" in JS


def test_interview_reuses_the_shared_status_and_build_endpoints():
    """InterviewJob writes the SynthJob on-disk contract precisely so these are
    shared. If it grew its own endpoints, the contract has drifted."""
    assert "TEACHER_SYNTH_STATUS" in JS
    assert "TEACHER_AUTHOR_BUILD" in JS
    assert JS.count("TEACHER_INTERVIEW_START") >= 1


def test_only_dialogue_genres_are_offered_for_interviewing():
    """Standalone-prose genres (jokes, writing, news) have no user turn to ask,
    so interviewing them is meaningless."""
    body = JS[JS.index("function _ivRenderGenres"):]
    body = body[:body.index("\nfunction ")]
    assert 'schema === "turns"' in body


def test_the_interview_job_is_rehydrated_on_tab_activation():
    """A refresh mid-run must land back on the live interview job."""
    body = JS[JS.index("function _distOnTabActivated"):]
    body = body[:body.index("\ndocument.addEventListener")]
    assert "_ivStored()" in body
    assert "_ivPollStart()" in body


def test_the_topic_picker_is_wired_to_the_seed_pack_route():
    """Topics are what stop a corpus being one undifferentiated file."""
    assert "TEACHER_SEED_PACKS" in JS
    assert '"/teacher/seed_packs"' in JS
    assert "_ivSelectedTopics" in JS


def test_the_start_request_sends_the_chosen_vertical_and_topics():
    """Without these the backend falls back to the genre's thin situations list."""
    body = JS[JS.index("function _ivStart"):]
    body = body[:body.index("\nfunction ")]
    assert "vertical: interviewState.vertical" in body
    assert "topics: topics" in body


def test_starting_with_no_topics_is_refused_in_the_client():
    """Cheaper to catch here than after the teacher has been called."""
    body = JS[JS.index("function _ivStart"):]
    body = body[:body.index("\nfunction ")]
    assert "pick at least one topic" in body


def test_concurrency_is_a_fixed_set_of_powers_of_two():
    """The user asked for 2/4/8/.../256 rather than a free-text number."""
    from teacher import providers
    assert providers.CONCURRENCY_CHOICES == (2, 4, 8, 16, 32, 64, 128, 256)
    assert providers.LOCAL_MAX_CONCURRENCY >= 256
    assert '<select id="interviewConcurrency"' in HTML


def test_the_topic_selection_is_persisted():
    """40 topics is too many to re-tick after every refresh."""
    assert "INTERVIEW_TOPIC_STORE" in JS
    assert "INTERVIEW_VERTICAL_STORE" in JS


def test_topic_memory_is_keyed_by_vertical():
    """A group id means nothing outside its own pack, so one flat list would let
    a stale conversation selection decide which code topics are ticked."""
    assert "_ivTopicStore" in JS
    assert "all[interviewState.vertical" in JS


def test_unavailable_verticals_are_disabled_not_hidden():
    """The roadmap is useful; a vertical that would silently use the wrong seeds
    is not. So they render disabled."""
    body = JS[JS.index("function _ivLoadPacks"):]
    body = body[:body.index("\nfunction ")]
    assert "disabled" in body
    assert "p.available" in body


def test_the_cost_line_warns_when_the_request_exceeds_the_seed_ceiling():
    """Asking for more conversations than the seeds support is the failure mode
    that wastes a whole overnight run."""
    body = JS[JS.index("function _ivCost"):]
    body = body[:body.index("\nfunction ")]
    assert "OPENERS_PER_SEED" in JS
    assert "ceiling" in body
