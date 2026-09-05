# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Stage-by-stage progress of an image run, as one small JSON file the dashboard
#   polls. The image trainer does most of its work before train.csv has a single row:
#   decoding pictures, fitting the codec, encoding the corpus. Without this file a run
#   looks dead for an hour, and the only tell is a CPU meter.
# - Four stages in fixed order: decode, codec, encode, train. A stage is pending,
#   running, done or skipped; the file also carries the run state (running, done,
#   stopped, failed), the device, and whatever the trainer notes (last checkpoint).
# - Written atomically (tmp + replace) and throttled, so a reader never sees a torn
#   file and a 60 img/s decode does not write 60 files a second.
# - StopRequested is the exception a stage raises when the trainer's stop flag is set;
#   it lives here because every stage and the trainer import this module already.
# veritate_core/plugin/image_progress.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import time

# ------------------------------------------------------------------------------------
# Constants

FILE_NAME       = "progress.json"
STAGES          = ("decode", "codec", "encode", "train")
STAGE_PENDING   = "pending"
STAGE_RUNNING   = "running"
STAGE_DONE      = "done"
STAGE_SKIPPED   = "skipped"
RUN_RUNNING     = "running"
RUN_DONE        = "done"
RUN_STOPPED     = "stopped"
RUN_FAILED      = "failed"
WRITE_EVERY_S   = 0.5

# ------------------------------------------------------------------------------------
# Classes


class StopRequested(Exception):
    """Raised inside a stage when the trainer was asked to stop."""


class Progress:
    """One run's progress file. Create it as soon as the model name is known."""

    def __init__(self, model_dir, device, total_steps=None):
        self.path = path(model_dir)
        self.state = {
            "state":   RUN_RUNNING,
            "device":  device,
            "started": time.time(),
            "updated": time.time(),
            "message": "",
            "current": None,
            "stages":  {s: {"state": STAGE_PENDING} for s in STAGES},
            "notes":   {},
        }
        if total_steps is not None:
            self.state["stages"]["train"]["total"] = int(total_steps)
        self._last_write = 0.0
        os.makedirs(model_dir, exist_ok=True)
        self._write(force=True)

    # -- stages ----------------------------------------------------------------------

    def stage(self, name, done, total, message="", **extra):
        """A stage is running with `done` of `total` units finished. Rate and ETA come
        from the stage's own clock, so a fast stage after a slow one is not misread."""
        st = self.state["stages"][name]
        now = time.time()
        entering = st.get("state") != STAGE_RUNNING
        if entering:
            st.update({"state": STAGE_RUNNING, "started": now, "first_done": int(done)})
        st.update({"done": int(done), "total": int(total), "updated": now})
        elapsed = max(1e-6, now - st["started"])
        advanced = int(done) - int(st.get("first_done") or 0)
        rate = advanced / elapsed if advanced > 0 else 0.0
        st["rate"] = round(rate, 2)
        st["eta_s"] = round((int(total) - int(done)) / rate) if rate > 0 and total else None
        st.update(extra)
        self.state["current"] = name
        self.state["message"] = message
        self._write(force=entering or (int(done) >= int(total) and int(total) > 0))

    def done(self, name, message="", **extra):
        st = self.state["stages"][name]
        now = time.time()
        st.update({"state": STAGE_DONE, "updated": now,
                   "seconds": round(now - st.get("started", now), 1)})
        if "total" in st:
            st["done"] = st["total"]
        st.update(extra)
        self.state["message"] = message
        self._write(force=True)

    def skip(self, name, message=""):
        self.state["stages"][name].update({"state": STAGE_SKIPPED, "message": message})
        self.state["message"] = message
        self._write(force=True)

    def note(self, **notes):
        """Facts the dashboard shows beside the stage bar: last checkpoint, val loss."""
        self.state["notes"].update(notes)
        self._write(force=True)

    def end(self, state, message=""):
        self.state.update({"state": state, "message": message, "ended": time.time()})
        self._write(force=True)

    # -- io --------------------------------------------------------------------------

    def _write(self, force=False):
        now = time.time()
        if not force and now - self._last_write < WRITE_EVERY_S:
            return
        self.state["updated"] = now
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.state, handle)
        os.replace(tmp, self.path)
        self._last_write = now


# ------------------------------------------------------------------------------------
# Functions


def path(model_dir):
    return os.path.join(model_dir, FILE_NAME)


def read(model_dir):
    """The progress dict, or None when the run never wrote one."""
    p = path(model_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None
