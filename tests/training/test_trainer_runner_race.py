# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - covers the stop-then-start race in trainer_runner. stop() marks the state stopped the
#   instant it signals the child; the child exits later. On cardinal 2026-09-02 a launch in
#   that window was accepted, the old child's reaper then stamped the NEW run failed and
#   nulled its handle, and a relaunch put two trainers on one model dir. The child is a
#   fake process the test releases by hand; no subprocess, no pid file, no log tailer.
# tests/training/test_trainer_runner_race.py
# ------------------------------------------------------------------------------------
# Imports:

import threading
import time

import pytest
from training import trainer_runner as tr

# ------------------------------------------------------------------------------------
# Constants

PLUGIN = {"id": "toy", "path": "toy.py", "manifest": {}}
WAIT_S = 3.0

# ------------------------------------------------------------------------------------
# Functions


class _FakeProc:
    """A child that exits only when the test releases it."""

    def __init__(self):
        self.pid = 4242
        self.returncode = None
        self._exit = threading.Event()

    def wait(self):
        self._exit.wait()
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def release(self, code):
        self.returncode = code
        self._exit.set()


def _until(pred):
    deadline = time.monotonic() + WAIT_S
    while not pred():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.01)


@pytest.fixture
def runner(monkeypatch, tmp_path):
    """The runner with its OS edges stubbed; yields the list of spawned fake children."""
    procs = []

    def fake_popen(argv, **kw):
        procs.append(_FakeProc())
        return procs[-1]

    monkeypatch.setattr(tr.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(tr, "RUN_LOG_FILE", str(tmp_path / "run.log"))
    monkeypatch.setattr(tr, "_write_pid_file", lambda *a, **k: None)
    monkeypatch.setattr(tr, "_clear_pid_file", lambda: None)
    monkeypatch.setattr(tr, "_tail_run_log", lambda *a, **k: None)
    monkeypatch.setattr(tr.plugins_reader, "scan", lambda: [PLUGIN])
    monkeypatch.setattr(tr.plugins_reader, "update_defaults", lambda pid, args: False)
    from runtime import heartbeat
    monkeypatch.setattr(heartbeat, "record_training_event", lambda *a, **k: None)
    monkeypatch.setattr(tr, "_PROC", None)
    monkeypatch.setattr(tr, "_RECOVERED_PID", None)
    tr._STATE.update(status=tr.STATUS_IDLE, plugin_id=None, args=None,
                     started_at=None, finished_at=None, exit_code=None)
    yield procs
    for p in procs:
        p.release(0)
    _until(lambda: tr._PROC is None)


def test_a_launch_while_the_previous_child_is_exiting_is_refused(runner):
    """stop() flips the status before the child has gone; a second start in that window
    must be refused, or two trainers share one model dir."""
    assert tr.start("toy", {"name": "m"})["ok"]
    _until(lambda: tr.pid() == 4242)
    assert tr.stop()["ok"]
    assert tr.state()["status"] == tr.STATUS_STOPPED
    second = tr.start("toy", {"name": "m"})
    assert second["ok"] is False and "still exiting" in second["error"]
    assert len(runner) == 1


def test_the_trainer_counts_as_running_until_its_child_has_exited(runner):
    """Every gate that asks is_running() (sleep's "trainer busy") must see the child, not
    the status, while it winds down."""
    tr.start("toy", {"name": "m"})
    _until(lambda: tr.pid() == 4242)
    tr.stop()
    assert tr.is_running() is True
    runner[0].release(-15)
    _until(lambda: not tr.is_running())
    assert tr.state()["exit_code"] == -15


def test_a_launch_after_the_child_has_exited_is_accepted(runner):
    """The refusal is about the child, not the stop: once it is gone the next run starts."""
    tr.start("toy", {"name": "m"})
    _until(lambda: tr.pid() == 4242)
    tr.stop()
    runner[0].release(-15)
    _until(lambda: tr._PROC is None)
    assert tr.start("toy", {"name": "m"})["ok"]
    _until(lambda: len(runner) == 2)


def test_a_late_exit_of_a_replaced_run_does_not_stamp_the_new_run(runner):
    """The reaper of a run that has been replaced writes no terminal status: the state
    belongs to the run whose start time it carries."""
    tr._STATE.update(status=tr.STATUS_RUNNING, started_at=2.0, exit_code=None)
    tr._finish_run(1.0, -15)
    assert tr.state()["status"] == tr.STATUS_RUNNING and tr.state()["exit_code"] is None
    tr._finish_run(2.0, 0)
    assert tr.state()["status"] == tr.STATUS_OK
