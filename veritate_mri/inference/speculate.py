# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - speculative prefetch: answers the draft prompt a client is still typing, so the
#   real request flushes what already exists instead of paying prefill.
# - the buffer holds assembled MRI frames, not bare bytes: a speculative turn runs
#   traced, so a flushed reply reaches the dashboard with the telemetry a live one
#   carries. frames come from the make_frame callable the caller injects, keeping
#   frame assembly (and the engine's reused parse buffers) with the route that owns it.
# - at most one job process-wide. a new draft supersedes the old one; a stand-down
#   drops it. cancellation is "do not issue the next chunk", never an abort of a live
#   engine turn, so the pipe always reaches TEND.
# - a real request always wins the engine: take() ends the job on every call, hit or
#   miss. a job left running reclaims the subprocess lock between chunks and starves
#   the request waiting on it.
# - each draft carries an id. take() requires that id plus the same prompt and
#   subprocess; anything else is a miss and the request generates normally.
# veritate_mri/inference/speculate.py
# ------------------------------------------------------------------------------------
# Imports:

import threading
import time

from runtime import logs as logmod

# ------------------------------------------------------------------------------------
# Constants

WIRE_ENCODING = "latin-1"

# A real request owns the engine for a whole generation. Wait that out instead of
# dropping the draft; give up only if the engine never comes free.
BUSY_POLL_S    = 0.05
BUSY_WAIT_MAX_S = 30.0

# Why a job stopped, reported to the dashboard so a stalled rail is explainable.
REASON_RUNNING = "running"
REASON_DONE    = "done"
REASON_BUDGET  = "budget"
REASON_BUSY    = "busy"
REASON_ERROR   = "error"

_LOCK  = threading.Lock()
_JOB   = None
_NEXT_ID = 1
_STATS = {"served": 0, "served_bytes": 0, "discarded": 0, "spent_bytes": 0}

# ------------------------------------------------------------------------------------
# Functions

class _Job:
    def __init__(self, draft_id, sub, prompt, params, budget, chunk, stop_sequences,
                 make_frame, compact):
        self.draft_id       = draft_id
        self.sub            = sub
        self.prompt         = prompt
        self.params         = dict(params)
        self.budget         = int(budget)
        self.chunk          = int(chunk)
        self.stop_sequences = tuple(stop_sequences or ())
        self.make_frame     = make_frame
        self.compact        = bool(compact)
        self.frames         = []
        self.cancelled      = False
        self.finished       = False
        self.reason         = REASON_RUNNING

    def matches(self, sub, prompt, params):
        return self.sub is sub and self.prompt == prompt and self.params == params

    def reply_bytes(self):
        return bytes(f["byte"] for f in self.frames)


def _discard(job):
    if job is None or job.cancelled:
        return
    job.cancelled = True
    _STATS["discarded"] += 1


def _run(job):
    try:
        while not job.cancelled and len(job.frames) < job.budget:
            # The subprocess lock is held for a whole generation, so a held lock means
            # a real request owns the engine. Wait it out rather than abandoning the
            # draft: a request that finishes frees the engine, and a draft that gave up
            # leaves the rail stalled with a half-written reply.
            if job.sub.lock.locked():
                waited = 0.0
                while job.sub.lock.locked() and not job.cancelled and waited < BUSY_WAIT_MAX_S:
                    time.sleep(BUSY_POLL_S)
                    waited += BUSY_POLL_S
                if job.cancelled or job.sub.lock.locked():
                    job.reason = REASON_BUSY
                    return
            want = min(job.chunk, job.budget - len(job.frames))
            p = job.params
            got = []
            last = time.perf_counter()
            for raw in job.sub.stream(job.prompt + job.reply_bytes().decode(WIRE_ENCODING),
                                      p["temperature"], p["top_k"], want,
                                      ablate_layer=p["ablate_layer"], ablate_neuron=p["ablate_neuron"],
                                      addons_csv=p["addons_csv"], rep_window=p["rep_window"],
                                      rep_penalty=p["rep_penalty"], no_repeat_ngram=p["no_repeat_ngram"],
                                      do_trace=True, compact=job.compact,
                                      stop_sequences=job.stop_sequences):
                now = time.perf_counter()
                got.append(job.make_frame(raw, (now - last) * 1000.0))
                last = now
            with _LOCK:
                if job.cancelled:
                    return
                job.frames.extend(got)
                _STATS["spent_bytes"] += len(got)
            # A short turn means the engine ended on a stop sequence: the reply is whole.
            if len(got) < want:
                job.reason = REASON_DONE
                return
        if job.reason == REASON_RUNNING:
            job.reason = REASON_BUDGET
    except Exception as e:
        job.reason = REASON_ERROR
        logmod.warn("speculate", f"job dropped: {type(e).__name__}: {e}")
    finally:
        job.finished = True


def start(sub, prompt, params, budget, chunk, stop_sequences, make_frame, compact=False):
    """Answer `prompt` ahead of the client submitting it. Supersedes any current job;
    re-posting the same draft keeps the running one. Returns the state, including the
    draft_id the client echoes back on submit to claim the reply."""
    global _JOB, _NEXT_ID
    with _LOCK:
        if _JOB is not None and not _JOB.cancelled and _JOB.matches(sub, prompt, params):
            return _status_locked()
        _discard(_JOB)
        _JOB = _Job(_NEXT_ID, sub, prompt, params, budget, chunk, stop_sequences,
                    make_frame, compact)
        _NEXT_ID += 1
        job = _JOB
        out = _status_locked()
    threading.Thread(target=_run, args=(job,), name="speculate", daemon=True).start()
    return out


def stand_down():
    """The draft is stale (the client edited it, or a real request needs the engine).
    Stop after the current chunk and drop the reply."""
    global _JOB
    with _LOCK:
        _discard(_JOB)
        _JOB = None
        return _status_locked()


def take(sub, prompt, draft_id):
    """The frames already generated for exactly this request, or empty on a miss.
    The job ends either way: a reply is flushed once or never."""
    global _JOB
    with _LOCK:
        job = _JOB
        _JOB = None
        if job is None:
            return []
        hit = (not job.cancelled and draft_id and job.draft_id == draft_id
               and job.sub is sub and job.prompt == prompt)
        if not hit:
            _discard(job)
            return []
        job.cancelled = True
        out = list(job.frames)
        _STATS["served"] += 1
        _STATS["served_bytes"] += len(out)
        return out


def _status_locked():
    job = _JOB
    return {
        "speculating": job is not None and not job.cancelled and not job.finished,
        "buffered":    len(job.frames) if job is not None else 0,
        "budget":      job.budget if job is not None else 0,
        "draft_id":    job.draft_id if job is not None else 0,
        "state":       job.reason if job is not None else REASON_DONE,
        "stats":       dict(_STATS),
    }


def status():
    with _LOCK:
        return _status_locked()


def preview():
    """Status plus the reply so far, for the dashboard's live draft view. Read-only:
    watching a job never consumes or cancels it."""
    with _LOCK:
        out = _status_locked()
        job = _JOB
        out["text"] = job.reply_bytes().decode("utf-8", "replace") if job is not None else ""
        return out
