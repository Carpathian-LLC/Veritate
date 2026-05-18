# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - per-check file logger plus stdout / stderr capture.
# - each check gets its own log file under tests/selftest/logs/<run_id>/.
# tests/selftest/_log.py
# ------------------------------------------------------------------------------------
# Imports

import contextlib
import io
import os
import sys
import time
import traceback

# ------------------------------------------------------------------------------------
# Constants

OPEN_MODE  = "a"
ENCODING   = "utf-8"
NEWLINE    = "\n"
TS_FMT     = "%H:%M:%S"
SEP_RULE   = "-" * 84

# ------------------------------------------------------------------------------------
# Functions

def _stamp():
    return time.strftime(TS_FMT, time.localtime())


def write_line(path, line):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, OPEN_MODE, encoding=ENCODING, newline=NEWLINE) as fh:
        fh.write(f"[{_stamp()}] {line}{NEWLINE}")


def write_block(path, header, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, OPEN_MODE, encoding=ENCODING, newline=NEWLINE) as fh:
        fh.write(f"[{_stamp()}] {header}{NEWLINE}")
        fh.write(SEP_RULE + NEWLINE)
        fh.write(body)
        if not body.endswith(NEWLINE):
            fh.write(NEWLINE)
        fh.write(SEP_RULE + NEWLINE)


def dump_exception(path, exc):
    body = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    write_block(path, f"EXCEPTION: {type(exc).__name__}: {exc}", body)


@contextlib.contextmanager
def capture(path):
    """redirect stdout + stderr into the log file for the duration of the block."""
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield
    out = buf_out.getvalue()
    err = buf_err.getvalue()
    if out:
        write_block(path, "stdout", out)
    if err:
        write_block(path, "stderr", err)


def tee(path, line):
    """write to log and to real stdout for live console feedback."""
    write_line(path, line)
    sys.stdout.write(line + NEWLINE)
    sys.stdout.flush()
