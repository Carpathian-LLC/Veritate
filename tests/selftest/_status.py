# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - check result model. three states: OK, SKIP, FAIL.
# - skip means a prerequisite was absent (no model, no binary, missing dep), not a bug.
# tests/selftest/_status.py
# ------------------------------------------------------------------------------------
# Imports

# ------------------------------------------------------------------------------------
# Constants

STATUS_OK   = "ok"
STATUS_SKIP = "skip"
STATUS_FAIL = "fail"

STATUS_ORDER = (STATUS_OK, STATUS_SKIP, STATUS_FAIL)

GLYPH = {
    STATUS_OK:   "[ OK ]",
    STATUS_SKIP: "[SKIP]",
    STATUS_FAIL: "[FAIL]",
}

# ------------------------------------------------------------------------------------
# Functions

class Result:
    """outcome of one check. summary is a one-line human string."""

    def __init__(self, name, status, summary, details=None, log_path=None, elapsed=0.0):
        self.name     = name
        self.status   = status
        self.summary  = summary
        self.details  = details or {}
        self.log_path = log_path
        self.elapsed  = elapsed

    def to_dict(self):
        return {
            "name":     self.name,
            "status":   self.status,
            "summary":  self.summary,
            "details":  self.details,
            "log_path": self.log_path,
            "elapsed":  round(self.elapsed, 3),
        }


def ok(name, summary, details=None):
    return Result(name, STATUS_OK, summary, details)


def skip(name, summary, details=None):
    return Result(name, STATUS_SKIP, summary, details)


def fail(name, summary, details=None):
    return Result(name, STATUS_FAIL, summary, details)
