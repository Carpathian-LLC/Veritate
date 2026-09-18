# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the .claude/hooks guards are the enforcement layer for the operating rules that used
#   to depend on an agent remembering them. every guard FAILS OPEN by design, so a broken
#   pattern does not wedge a session - it silently stops enforcing. that is the failure
#   this file exists to catch.
# - exit 2 is a refusal (stderr reaches the agent), exit 0 with a permissionDecision
#   payload routes the decision to the user, bare exit 0 is allow.
# - guards are subprocesses, so these tests need no import of hook internals.
# tests/hooks/test_guards.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import re
import subprocess
import sys

import pytest
from conftest import REPO_ROOT

# ------------------------------------------------------------------------------------
# Constants

HOOKS = os.path.join(REPO_ROOT, ".claude", "hooks")

BLOCK = 2
ALLOW = 0

# Derived, not pinned to this box: the guard keys on the path SHAPE (a memory directory
# inside a .claude tree), so the test stays true on any install.
MEMORY_DIR = os.path.join(os.path.expanduser("~"), ".claude", "projects", "-a-project", "memory")

# ------------------------------------------------------------------------------------
# Functions

def hook(name, payload):
    """Run a guard against a payload. Returns (exit code, stderr or stdout)."""
    p = subprocess.run([sys.executable, os.path.join(HOOKS, name)],
                       input=json.dumps(payload), capture_output=True, text=True,
                       env={**os.environ, "CLAUDE_PROJECT_DIR": REPO_ROOT}, cwd=REPO_ROOT)
    return p.returncode, (p.stderr or p.stdout).strip()


def write(path):
    return hook("guard_write.py", {"tool_input": {"file_path": path}})


def transcript(tmp_path, blocks):
    """A minimal assistant transcript: one JSONL row per content block list."""
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps({"type": "assistant", "message": {"content": b}})
                           for b in blocks))
    return str(p)


EDIT_SOURCE = [{"type": "tool_use", "name": "Edit",
                "input": {"file_path": f"{REPO_ROOT}/veritate_mri/routes/x.py"}}]
RAN_TESTS   = [{"type": "tool_use", "name": "Bash", "input": {"command": "pytest tests/mri -q"}}]


@pytest.mark.parametrize("path", [
    "NOTES.md",
    "docs/architecture.md",
    "veritate_mri/PLAN.md",
])
def test_new_documentation_outside_the_five_files_is_refused(path):
    """Documentation lives in five files at repo root and nowhere else (rule 34)."""
    assert write(os.path.join(REPO_ROOT, path))[0] == BLOCK


@pytest.mark.parametrize("path", [
    "documentation.md",
    "handoff.md",
    "worklog.md",
    "lab/2026-09-02_probe.md",
    ".claude/skills/veritate-code/SKILL.md",
])
def test_sanctioned_documentation_targets_are_allowed(path):
    """The five files, the logs, lab/ and the agent config tree stay writable."""
    assert write(os.path.join(REPO_ROOT, path))[0] == ALLOW


def test_scratch_markdown_outside_the_repo_is_allowed(tmp_path):
    """Rule 34 governs this repo. A scratch note in the session scratchpad is not documentation."""
    assert write(str(tmp_path / "notes.md"))[0] == ALLOW


def test_versions_json_is_refused():
    """Version bumps are the user's call (rule 12)."""
    assert write(os.path.join(REPO_ROOT, "versions.json"))[0] == BLOCK


def test_a_second_trainer_is_refused():
    """There is exactly one trainer; per-size trainers never get created (rule 23)."""
    assert write(os.path.join(REPO_ROOT, "veritate_mri/training/wren2_trainer.py"))[0] == BLOCK


def test_root_level_script_is_refused():
    """A .py at repo root that a human runs by hand is a defect (rule 22)."""
    assert write(os.path.join(REPO_ROOT, "quickcheck.py"))[0] == BLOCK


@pytest.mark.parametrize("path", [
    "veritate_mri/routes/_brain.py",
    "tests/mri/test_something.py",
])
def test_platform_source_and_tests_stay_writable(path):
    """The doc and script guards must not touch ordinary platform work."""
    assert write(os.path.join(REPO_ROOT, path))[0] == ALLOW


def test_memory_files_are_refused():
    """Durable corrections become hooks or skill rules, never memory files."""
    assert write(f"{MEMORY_DIR}/project_something.md")[0] == BLOCK


def test_the_memory_index_stays_writable():
    """MEMORY.md carries the redirect policy, so it is the one exception."""
    assert write(f"{MEMORY_DIR}/MEMORY.md")[0] == ALLOW


@pytest.mark.parametrize("prompt", [
    "why did you do that, I told you not to touch git",
    "this is still broken and its annoying",
    "no. that is wrong, I wanted the other approach",
])
def test_corrections_trigger_the_rule_redirect(prompt):
    """A correction must be converted into enforcement, not into a note."""
    assert "frustration-signal" in hook("frustration_to_rule.py", {"prompt": prompt})[1]


@pytest.mark.parametrize("prompt", [
    "can you add a test for the resolve_step function",
    "what is the current wren2 step",
    "please update documentation.md with the new route",
])
def test_neutral_prompts_stay_quiet(prompt):
    """Ordinary requests must not be read as frustration."""
    assert "frustration-signal" not in hook("frustration_to_rule.py", {"prompt": prompt})[1]


def test_stop_is_refused_when_source_changed_without_verification(tmp_path):
    """Code changed and nothing run afterwards is not a finished turn (rule 33)."""
    t = transcript(tmp_path, [EDIT_SOURCE, [{"type": "text", "text": "Done, the fix is in."}]])
    assert hook("persist.py", {"transcript_path": t, "stop_hook_active": False})[0] == BLOCK


def test_stop_is_allowed_once_the_change_was_run(tmp_path):
    """A verified change ends the turn cleanly."""
    t = transcript(tmp_path, [EDIT_SOURCE, RAN_TESTS, [{"type": "text", "text": "5 passed."}]])
    assert hook("persist.py", {"transcript_path": t, "stop_hook_active": False})[0] == ALLOW


def test_shell_written_source_counts_as_a_change(tmp_path):
    """Edits made through Bash are changes too, and carry the same obligation."""
    blocks = [[{"type": "tool_use", "name": "Bash",
                "input": {"command": "cat > veritate_mri/routes/x.py <<EOF"}}],
              [{"type": "text", "text": "Written."}]]
    t = transcript(tmp_path, blocks)
    assert hook("persist.py", {"transcript_path": t, "stop_hook_active": False})[0] == BLOCK


def test_editing_a_guard_is_held_to_the_verification_gate(tmp_path):
    """The guards are source too: changing one and running nothing is not a finished turn."""
    edit = [{"type": "tool_use", "name": "Edit",
             "input": {"file_path": f"{REPO_ROOT}/.claude/hooks/guard_write.py"}}]
    t = transcript(tmp_path, [edit, [{"type": "text", "text": "Done, the guard is updated."}]])
    assert hook("persist.py", {"transcript_path": t, "stop_hook_active": False})[0] == BLOCK


def test_capitulation_without_evidence_is_refused(tmp_path):
    """Handing the work back with nothing measured is stopping at the easy answer."""
    text = "Unfortunately this is not possible. You could patch it manually."
    t = transcript(tmp_path, [[{"type": "text", "text": text}]])
    assert hook("persist.py", {"transcript_path": t, "stop_hook_active": False})[0] == BLOCK


def test_capitulation_phrasing_with_evidence_is_allowed(tmp_path):
    """A verified result may still offer follow-up work without tripping the gate."""
    text = "12 passed, verified. Let me know if you want the follow-up."
    t = transcript(tmp_path, [RAN_TESTS, [{"type": "text", "text": text}]])
    assert hook("persist.py", {"transcript_path": t, "stop_hook_active": False})[0] == ALLOW


def test_stop_gate_cannot_loop(tmp_path):
    """stop_hook_active short-circuits the gate, so it can never re-block itself."""
    t = transcript(tmp_path, [EDIT_SOURCE])
    assert hook("persist.py", {"transcript_path": t, "stop_hook_active": True})[0] == ALLOW


def test_stop_gate_fails_open_on_a_missing_transcript():
    """A guard that cannot read its input allows the action rather than wedging it."""
    assert hook("persist.py", {"transcript_path": "/nonexistent.jsonl"})[0] == ALLOW


def schedule(prompt, tool="CronCreate"):
    return hook("guard_schedule.py", {"tool_name": tool, "tool_input": {"prompt": prompt}})


def test_croncreate_is_refused_because_it_dies_with_the_session():
    """Session-only scheduling silently drops persistent work; launchd owns that job."""
    code, msg = schedule("Veritate facts cycle: HOURLY cadence, three Sonnet workers.")
    assert code == BLOCK
    assert "launchd" in msg


def test_explicitly_ephemeral_scheduling_is_allowed():
    """A job meant to die with the session passes once it says so."""
    assert schedule("SESSION-ONLY: poll the build every 5 minutes")[0] == ALLOW


def test_scheduling_guard_ignores_other_tools():
    """The guard keys on CronCreate alone and never blocks unrelated tool calls."""
    assert schedule("anything at all", tool="Bash")[0] == ALLOW


def test_croncreate_with_no_prompt_is_still_refused():
    """A missing prompt cannot be read as an opt-out."""
    assert hook("guard_schedule.py", {"tool_name": "CronCreate", "tool_input": {}})[0] == BLOCK


def test_subagents_and_workflows_are_refused():
    """All work happens in the main session, off a todo list (user, 2026-09-08)."""
    for tool in ("Agent", "Workflow"):
        code, msg = hook("guard_agents.py", {"tool_name": tool, "tool_input": {"prompt": "x"}})
        assert code == BLOCK, tool
        assert "todo" in msg


def test_agent_guard_ignores_other_tools():
    """The guard keys on Agent and Workflow alone."""
    assert hook("guard_agents.py", {"tool_name": "Bash", "tool_input": {"command": "ls"}})[0] == ALLOW


FORK = ('curl -s -X POST localhost:8010/models/fork '
        '-d \'{{"source": "wren2_0", "new_name": "{name}"}}\'')


def fleet(cmd):
    return hook("guard_fleet.py", {"tool_name": "Bash", "tool_input": {"command": cmd}})


def suggested(msg):
    """The fleet name a refusal tells you to use instead."""
    hit = re.search(r"use: (wren\d+_\d+)", msg)
    assert hit, msg
    return hit.group(1)


def test_a_scratch_name_never_reaches_a_fork():
    """exp_* is not a run name (user, 2026-09-17). The refusal names the rung to use."""
    code, msg = fleet(FORK.format(name="exp_carpchat_0917"))
    assert code == BLOCK
    assert "not a wren fleet name" in msg
    assert suggested(msg).startswith("wren2_")


def test_the_suggested_rung_is_accepted():
    """Whatever the guard tells you to use has to pass the guard."""
    assert fleet(FORK.format(name=suggested(fleet(FORK.format(name="exp_x_0101"))[1])))[0] == ALLOW


def test_skipping_a_rung_is_refused():
    """The number bumps by one. A gap hides a run that is not written down anywhere."""
    lineage, rung = suggested(fleet(FORK.format(name="exp_x_0101"))[1]).split("_")
    code, msg = fleet(FORK.format(name=f"{lineage}_{int(rung) + 5}"))
    assert code == BLOCK
    assert "does not bump" in msg


def test_a_model_on_disk_is_the_same_run_continuing():
    """Resuming wren2_0 under its own name is not a new rung and is left alone."""
    run = ('curl -s -X POST localhost:8010/trainers/run '
           '-d \'{"id": "veritate", "args": {"name": "wren2_0", "resume": "wren2_0"}}\'')
    assert fleet(run)[0] == ALLOW


def test_a_name_posted_from_a_file_is_still_seen():
    """The launch payload is normally a file: curl -d @payload.json."""
    path = os.path.join(REPO_ROOT, "tests", "hooks", "_fleet_payload.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"id": "veritate", "args": {"name": "exp_carpchat_0917", "resume": "wren2_0"}}, fh)
    try:
        code, msg = fleet(f"curl -s -X POST localhost:8010/trainers/run -d @{path}")
    finally:
        os.remove(path)
    assert code == BLOCK
    assert "exp_carpchat_0917" in msg


def test_fleet_guard_ignores_commands_that_launch_nothing():
    """A name in unrelated json is not a run."""
    assert fleet("echo '{\"name\": \"exp_whatever\"}'")[0] == ALLOW


EDIT_WIKI_CSS = [{"type": "tool_use", "name": "Edit",
                  "input": {"file_path": f"{REPO_ROOT}/veritate_mri/web/wiki.css"}}]
TOOK_A_LOOK   = [{"type": "tool_use", "name": "Bash",
                  "input": {"command": "chrome --headless --screenshot=out.png page.html"}}]


def test_changing_a_dashboard_surface_without_rendering_it_is_refused(tmp_path):
    """A turn that edits veritate_mri/web/ and never renders the page cannot end."""
    code, msg = hook("guard_unlooked_ui.py", {"transcript_path": transcript(tmp_path, [EDIT_WIKI_CSS])})
    assert code == BLOCK
    assert "wiki.css" in msg


def test_a_rendered_surface_is_allowed(tmp_path):
    """A screenshot in the same turn is the evidence the gate asks for."""
    t = transcript(tmp_path, [EDIT_WIKI_CSS, TOOK_A_LOOK])
    assert hook("guard_unlooked_ui.py", {"transcript_path": t})[0] == ALLOW


def test_surface_gate_ignores_non_web_edits(tmp_path):
    """Python and kernels have tests; only the web surface needs a picture."""
    t = transcript(tmp_path, [EDIT_SOURCE])
    assert hook("guard_unlooked_ui.py", {"transcript_path": t})[0] == ALLOW


def test_surface_gate_cannot_loop(tmp_path):
    """stop_hook_active short-circuits so the gate can never wedge a session."""
    t = transcript(tmp_path, [EDIT_WIKI_CSS])
    assert hook("guard_unlooked_ui.py", {"transcript_path": t, "stop_hook_active": True})[0] == ALLOW


def test_every_guard_fails_open_on_unparseable_input():
    """Malformed stdin must never block work in any guard."""
    for name in ("guard_write.py", "guard_schedule.py", "guard_agents.py", "frustration_to_rule.py",
                     "persist.py", "guard_unlooked_ui.py"):
        p = subprocess.run([sys.executable, os.path.join(HOOKS, name)],
                           input="not json", capture_output=True, text=True,
                           env={**os.environ, "CLAUDE_PROJECT_DIR": REPO_ROOT}, cwd=REPO_ROOT)
        assert p.returncode == ALLOW, name
