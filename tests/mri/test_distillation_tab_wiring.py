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
    "distViewModal", "distViewTitle", "distViewClose", "distViewRange",
    "distViewPrev", "distViewNext", "distViewJump", "distViewGo", "distViewBody",
    "interviewJobName", "authorJobName", "synthJobName",
    "distAuditPanel", "distAuditVerdict", "distAuditChecks", "distAuditDetail",
    "authorProgress", "authorProgressFill",
    "authorProgressLeft", "authorProgressMid", "authorProgressRight",
    "interviewPanel", "interviewGenreList", "interviewJobSelect", "interviewCount",
    "interviewDepth", "interviewConcurrency", "interviewStartBtn", "interviewStopBtn",
    "interviewCostLine", "interviewStatsRow", "interviewProgress",
    "interviewProgressFill", "interviewProgressLeft", "interviewProgressMid",
    "interviewProgressRight", "interviewBuildRow", "interviewStem", "interviewLabel",
    "interviewBuildBtn", "interviewBuildStatus", "interviewLiveWrap", "interviewLiveOutput",
    "interviewCallsWrap", "interviewCallsOutput", "interviewCallsStats",
    "interviewGateWrap", "interviewGateHits", "interviewGateCount",
    "interviewBannedFold", "interviewBannedList", "interviewBannedSave", "interviewBannedStatus",
    "interviewVertical", "interviewVerticalNote", "interviewTopicList",
    "interviewTopicAll", "interviewTopicNone",
    "distJobsList", "distJobsSummary", "distJobsRefresh",
    "interviewRunBar", "interviewRunWhat", "interviewProgressStep",
    "authorRunBar", "authorRunWhat", "authorProgressStep",
    "synthRunBar", "synthRunWhat", "synthProgressStep",
)

# Every mode collapses the same way while its job runs.
MODES = ("interview", "author", "synth")

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


def _mode_panel(mode):
    """The markup of one mode's workspace, up to the next one."""
    body = _tab_body(TAB_ID)
    starts = sorted(body.index(f'id="{m}Panel"') for m in MODES)
    here = body.index(f'id="{mode}Panel"')
    after = [x for x in starts if x > here]
    return body[here:after[0]] if after else body[here:body.index("corpora on disk")]


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


def test_the_training_menu_has_no_distillation_card():
    """The tab is reached from the nav, not from the training flow menu. A card
    left behind would still route through a flow branch that no longer exists."""
    assert f'data-flow="{TAB_ID}"' not in HTML
    assert f'flow === "{TAB_ID}"' not in JS
    assert f'data-flow="{TAB_ID}"' not in CSS


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


def test_every_poll_refreshes_the_live_call_feed():
    """A run's only visible sign of life used to be a progress bar. The feed is
    what says which call is open, what was sent, and how long it has been
    waiting, so it has to move with the poller."""
    body = JS[JS.index("function _ivPollOnce"):]
    body = body[:body.index("\nfunction ")]
    assert "_ivLoadCalls();" in body
    assert "TEACHER_SYNTH_CALLS" in JS


def test_open_calls_tick_faster_than_the_poller():
    """A two-second poll makes a clock that jumps two seconds at a time. The tick
    updates only the in-flight rows, so nothing else is redrawn."""
    body = JS[JS.index("function _ivCallsTick"):]
    body = body[:body.index("\nfunction ")]
    assert ".dist-call.is-live" in body
    assert "innerHTML" not in body
    assert "interviewState.tickTimer = setInterval(_ivCallsTick, IV_CALL_TICK_MS)" in JS


def test_a_failed_call_keeps_its_row_and_its_latency():
    """How long a call took to fail is what separates a timeout from a refusal,
    so a dead call is a row in the feed, not a gap in it."""
    body = JS[JS.index("function _ivCallRow"):]
    body = body[:body.index("\n// Only the open calls")]
    assert "c.error" in body
    assert "is-failed" in body
    assert ".dist-call.is-failed" in CSS


def test_the_durable_call_counters_come_from_the_status_payload():
    """The live feed dies with the job. Calls made, calls failed and conversations
    kept short have to survive a restart, so they are read from state.json."""
    body = JS[JS.index("function _ivRenderStats"):]
    body = body[:body.index("\n  if (warn)")]
    assert "s.call_stats" in body
    assert "c.salvaged" in body


def test_the_gate_says_which_banned_phrase_it_blocked():
    """A count of rejects does not say which entry in the list to change."""
    body = JS[JS.index("function _ivRenderGateHits"):]
    body = body[:body.index("\nconst ")]
    assert "a.banned_hits" in body
    assert "_ivRenderGateHits(a)" in JS


def test_the_ban_list_is_editable_and_saved_whole():
    """The list is data the run depends on, so it is edited here rather than by
    hand in corpus_spec.json."""
    assert '"/teacher/authoring/banned"' in JS
    assert 'addEventListener("click", _ivSaveBanned)' in JS
    assert 'id="interviewBannedList"' in _mode_panel("interview")


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


def test_every_job_is_listed_with_a_view_rename_and_delete():
    """Jobs accumulate in the destination pickers with no way out of them. The
    list is the only surface that can read, rename or remove one."""
    body = JS[JS.index("function _distJobRow"):]
    body = body[:body.index("\nfunction ")]
    assert '["view", "use", "ren", "del"]' in body
    for act in ("view", "use", "ren", "del"):
        assert f"{act}: _distJob" in JS[JS.index("const DIST_JOB_ACTIONS"):][:200]


def test_a_running_job_can_still_be_read():
    """A run that is writing records is exactly when you want to see them, so
    view survives the swap to the running row; the mutating actions do not."""
    body = JS[JS.index("function _distJobRow"):]
    body = body[:body.index("\nfunction ")]
    assert 'j.running ? ["view"]' in body


def test_deleting_a_job_names_what_it_will_remove():
    """Rule: a destructive action restates its exact target before it runs."""
    body = JS[JS.index("function _distJobDelete"):]
    body = body[:body.index("\nfunction ")]
    assert "confirm(" in body
    assert "synth_jobs/${jobId}" in body


def test_the_list_refreshes_after_a_rename_or_a_delete():
    """A stale row still offers `use` on a directory that no longer exists. Both
    mutations go through one poster, so the reload cannot be forgotten on one."""
    for fn in ("_distJobRename", "_distJobDelete"):
        body = JS[JS.index(f"function {fn}"):]
        body = body[:body.index("\nfunction ")]
        assert "_distJobPost(" in body
    post = JS[JS.index("function _distJobPost"):]
    post = post[:post.index("\n}")]
    assert ".then(_synthLoadJobs)" in post


def test_one_fetch_feeds_every_picker_and_the_list():
    """Three pickers reading three copies of the job list is how they drift out
    of agreement about what exists. `_synthLoadJobs` is the only reader."""
    assert JS.count("fetch(TEACHER_SYNTH_JOBS)") == 1
    loader = JS[JS.index("function _synthLoadJobs"):]
    loader = loader[:loader.index("\n}")]
    assert "_distFillJobPickers()" in loader
    assert "_distRenderJobs()" in loader
    for gone in ("_authorFillJobs", "_ivFillJobs"):
        assert gone not in JS


def test_the_pickers_show_the_label_not_the_hex_id():
    """The whole point of rename: `_distJobName` falls back to the id, and one
    filler drives all three pickers off the same option text."""
    body = JS[JS.index("function _distFillJobPickers"):]
    body = body[:body.index("\n}\n")]
    assert "_distJobName(j)" in body
    ids = re.findall(r'id: "(\w+JobSelect)"', JS)
    assert sorted(ids) == ["authorJobSelect", "interviewJobSelect", "synthJobSelect"]


def test_the_tab_carries_no_inline_control_styling():
    """Every field used to repeat its own background/border/padding inline,
    which is why no two rows lined up. The styling is one CSS rule now."""
    body = _tab_body(TAB_ID)
    assert "background:#0a0c12" not in body.replace(" ", "")
    assert '.tab-body[data-tab="distillation"] select' in CSS


@pytest.mark.parametrize("mode", MODES)
def test_each_mode_marks_its_configuration_blocks(mode):
    """`.dist-config` is what the running state hides. A step without it stays on
    screen mid-run showing controls that no longer affect anything."""
    assert _mode_panel(mode).count("dist-config") >= 3


def test_running_hides_the_configuration_and_nothing_else():
    """The rule is one selector. If it ever grows to hide the progress block or
    the stop button, a run becomes unobservable and unstoppable."""
    rule = next(ln for ln in CSS.splitlines() if ".dist-work.is-running .dist-config" in ln)
    assert "display: none" in rule
    assert ".dist-work.is-running .dist-progress" not in CSS
    assert ".dist-work.is-running .dist-run-bar" not in CSS


@pytest.mark.parametrize("mode", MODES)
def test_the_stop_button_lives_in_the_run_bar(mode):
    """It used to sit in the settings row, which the running state now hides."""
    body = _tab_body(TAB_ID)
    bar = body[body.index(f'id="{mode}RunBar"'):]
    bar = bar[:bar.index("</div>")]
    stop = "synthStopPollBtn" if mode == "synth" else f"{mode}StopBtn"
    assert f'id="{stop}"' in bar


@pytest.mark.parametrize("mode", MODES)
def test_every_status_poll_drives_the_running_state(mode):
    """A poll that forgets this leaves the form collapsed after the job stops, or
    shows the form while it is still burning teacher calls."""
    assert f'_distSetRunning("{mode}"' in JS


def test_the_run_bar_says_what_is_running():
    """Hiding the settings is only safe if the one line that replaces them names
    the destination and the size that were chosen."""
    body = JS[JS.index("function _ivRenderStats"):]
    body = body[:body.index("\nfunction ")]
    assert "_distJobNameById" in body
    assert "p.conversations" in body and "p.depth" in body


def test_a_run_is_visible_from_the_other_mode_cards():
    """Switching mode mid-run must not make the run look like it stopped."""
    assert "dist-mode-live" in HTML
    assert ".dist-mode.has-run .dist-mode-live" in CSS
    body = JS[JS.index("function _distSetRunning"):]
    body = body[:body.index("\nfunction ")]
    assert 'classList.toggle("has-run"' in body


def test_the_list_binds_one_listener_not_one_per_button():
    """The list re-renders on every job change; per-button binding leaks a fresh
    closure per row per render."""
    assert "_distRenderJobs" in JS
    render = JS[JS.index("function _distRenderJobs"):]
    render = render[:render.index("\nfunction ")]
    assert "addEventListener" not in render
    assert 'jobsList.addEventListener("click"' in JS


def test_a_running_row_is_patched_not_rebuilt():
    """Re-rendering the whole list on a 2 s poll throws away scroll position and
    any open confirm. Only the count moves while a job runs."""
    body = JS[JS.index("function _distSyncJobRow"):]
    body = body[:body.index("\nfunction ")]
    assert "textContent" in body
    assert "innerHTML" not in body


def test_a_finished_run_rebuilds_the_row():
    """Patching alone would leave the row showing `running` with no actions after
    the job stops."""
    body = JS[JS.index("function _distSyncJobRow"):]
    body = body[:body.index("\nfunction ")]
    assert "const ended = job.running && !running" in body
    assert "if (ended) { _synthLoadJobs(); return; }" in body


def test_an_unchanged_poll_touches_no_dom():
    """Three modes polling every 2 s would otherwise write six style properties a
    tick for the entire life of a run."""
    body = JS[JS.index("function _distSetRunning"):]
    body = body[:body.index("\nfunction ")]
    assert "distState.running[mode] === running" in body
    assert body.index("return;") < body.index("classList.toggle")


@pytest.mark.parametrize("flow", ("synth", "author"))
def test_the_moved_flows_are_gone_from_the_training_tab(flow):
    """`synth` and `author` became modes of this tab. A flow left in the valid
    list restores from localStorage into a screen with no card and no controls."""
    valid = re.search(r"const TRAIN_VALID_FLOWS = \[([^\]]+)\]", JS).group(1)
    assert f'"{flow}"' not in valid
    assert f'data-flow="{flow}"' not in HTML


# ------------------------------------------------------------------------------------
# Corpus viewer and start-time naming.

def test_the_viewer_pages_instead_of_loading_a_whole_corpus():
    """A finished job is 100 MB of jsonl. The viewer must ask for a window of it,
    never the file: /teacher/synth/samples tails and cannot page at all."""
    body = JS[JS.index("function _distViewLoad"):]
    body = body[:body.index("\nfunction ")]
    assert "TEACHER_SYNTH_BROWSE" in body
    assert "offset=${distView.offset}" in body
    assert "limit=${DIST_VIEW_PAGE}" in body


def test_the_viewer_reads_paging_state_back_from_the_server():
    """Trusting the local offset would drift past the end of a corpus that was
    deleted from or is still being written."""
    body = JS[JS.index("function _distViewLoad"):]
    body = body[:body.index("\nfunction ")]
    assert "distView.total = d.total" in body
    assert "distView.offset = d.offset" in body


def test_the_viewer_escapes_every_record_it_prints():
    """Records are teacher output: unescaped, one of them eventually closes a tag."""
    body = JS[JS.index("function _distViewCard"):]
    body = body[:body.index("\nfunction ")]
    assert "_trEsc(t.text" in body and "_trEsc(t.role" in body
    assert "_trEsc(r.text" in body and "_trEsc(r.id" in body


def test_the_viewer_can_be_dismissed_three_ways():
    """A modal with only an X is a trap on a laptop trackpad."""
    body = JS[JS.index('const vClose = $("distViewClose")'):]
    body = body[:body.index("});", body.index('e.key !== "Escape"'))]
    assert "_distViewClose" in body
    assert 'e.target === vModal' in body
    assert 'e.key !== "Escape"' in body


def test_the_name_box_is_disabled_when_appending_to_an_existing_corpus():
    """The box names a corpus at creation. Left live over an existing job it reads
    as a rename the start route would have to guess the intent of."""
    body = JS[JS.index("function _distSyncNameField"):]
    body = body[:body.index("\nfunction ")]
    assert "box.disabled = true" in body
    assert "box.disabled = false" in body


def test_a_typed_name_is_only_sent_for_a_new_corpus():
    """_distNewName returns nothing while the box is disabled, so appending never
    silently renames the corpus it appends to."""
    body = JS[JS.index("function _distNewName"):]
    body = body[:body.index("\nfunction ")]
    assert "if (!box || box.disabled) return \"\";" in body


@pytest.mark.parametrize("pick", ("interviewJobSelect", "authorJobSelect", "synthJobSelect"))
def test_every_mode_sends_its_start_time_name(pick):
    """A mode that forgets the label mints a hex id the user then has to rename."""
    assert f'_distNewName("{pick}")' in JS


def test_the_steps_explain_themselves_instead_of_wearing_numbers():
    """The numbered badges implied an order the tab does not enforce; each step
    carries a sentence saying what its controls do instead."""
    assert "dist-step-num" not in HTML
    assert "dist-step-num" not in CSS
    assert HTML.count('class="dist-step-sub"') >= 3
    assert ".dist-step-sub" in CSS
