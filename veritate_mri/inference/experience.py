# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the experience log: every completed serving exchange is appended here as one
#   JSONL record — the replay substrate for sleep consolidation (IDEA 20 T3: the
#   model trains on its OWN thought and actions, not curated external data).
#   Bytes are base64 (byte-level models; lossless round-trip). One file per
#   local day under data/experience/. Logging must never break serving: every
#   failure degrades to a warn, and a per-day size cap stops runaway growth.
#   VERITATE_EXPERIENCE_LOG=0 disables.
# veritate_mri/inference/experience.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import json
import os
import threading
import time

from readers.paths import EXPERIENCE_ROOT
from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

ENABLED_ENV   = "VERITATE_EXPERIENCE_LOG"
MAX_DAY_BYTES = 512 * 1024 * 1024
_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions


def enabled():
    return os.environ.get(ENABLED_ENV, "1").strip() != "0"


def day_path(ts=None):
    """The JSONL file exchanges land in for the given (default: current) time."""
    day = time.strftime("%Y%m%d", time.localtime(ts if ts is not None else time.time()))
    return os.path.join(EXPERIENCE_ROOT, f"{day}.jsonl")


def append_exchange(model, prompt_bytes, output_bytes, meta=None):
    """Record one completed exchange. Never raises: serving must not depend on
    the log. Empty outputs are skipped — an exchange with no generation carries
    nothing to consolidate."""
    if not enabled() or not output_bytes:
        return False
    try:
        rec = {
            "ts": round(time.time(), 3),
            "model": str(model or ""),
            "prompt_b64": base64.b64encode(bytes(prompt_bytes)).decode("ascii"),
            "output_b64": base64.b64encode(bytes(output_bytes)).decode("ascii"),
        }
        if meta:
            rec["meta"] = meta
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        path = day_path()
        with _LOCK:
            os.makedirs(EXPERIENCE_ROOT, exist_ok=True)
            try:
                if os.path.getsize(path) + len(line) > MAX_DAY_BYTES:
                    logmod.warn("experience", f"day cap reached, dropping exchange: {path}")
                    return False
            except OSError:
                pass  # file does not exist yet
            with open(path, "a", encoding="ascii") as f:
                f.write(line)
        return True
    except Exception as e:
        logmod.warn("experience", f"append failed: {type(e).__name__}: {e}")
        return False


def record_events(gen, model, prompt, meta=None):
    """Wrap an SSE-event generator: pass every event through, accumulate the
    emitted byte stream (token / fast_byte events), and append one exchange
    record when the stream ends — including client disconnects, so partial
    replies still count as experience."""
    out = bytearray()
    try:
        for ev in gen:
            if isinstance(ev, dict) and ev.get("kind") in ("token", "fast_byte"):
                b = ev.get("byte")
                if isinstance(b, int):
                    out.append(b & 0xff)
            yield ev
    finally:
        append_exchange(model, prompt.encode("utf-8") if isinstance(prompt, str) else prompt,
                        bytes(out), meta=meta)
