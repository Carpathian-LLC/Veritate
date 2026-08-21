# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Runs interview.py as a background job. Deliberately writes the SAME on-disk
#   contract as SynthJob (samples.jsonl / state.json / errors.jsonl / meta.json),
#   so /teacher/synth/status, /stop, /samples and /authoring/build all work on an
#   interview job with no change. Matching the contract is what buys the whole UI.
# - Every conversation goes through the existing RecordGate before it is kept, so
#   the 121 banned phrases, the em-dash policy, exact/near duplicate detection and
#   the distinct-5-gram variety floor all still apply. The gate takes a genre and
#   a JSON string, which is exactly what a built conversation serialises to.
# - The unit of work is a CONVERSATION, not a teacher call: one conversation costs
#   2*depth-1 sequential calls (answer, follow-up, answer, ...). Progress counts
#   conversations, because that is what the user asked for a number of.
# veritate_mri/teacher/interview_job.py
# ------------------------------------------------------------------------------------
# Imports:

import collections
import functools
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import authoring as authoring_mod
from . import interview
from .client import Client, TeacherError

# ------------------------------------------------------------------------------------
# Constants

SAMPLES_FILE = "samples.jsonl"
STATE_FILE = "state.json"
ERRORS_FILE = "errors.jsonl"
# Pass 1 costs one teacher call per OPENER_BATCH openers and used to live only in
# memory: a stopped or crashed run threw the whole pool away, and the resume paid
# for it again only for the gate to reject the result as near duplicates.
OPENERS_FILE = "openers.jsonl"
STATE_FLUSH_EVERY = 5
# Failure reasons are counted the way synth.py counts them, so the dashboard can
# say *what* went wrong instead of only how many did. Long teacher errors are
# truncated to a key so a hundred variants of one message collapse into one row.
ERROR_SUMMARY_TOP = 6
REASON_KEY_MAX = 120

# The live call feed. Held in memory only: a 10,000 conversation run at depth 3
# makes 50,000 teacher calls, and this is a monitor of what is happening now,
# not a record. CALL_TEXT_MAX is what the panel shows of each side of a call;
# the full text lands in samples.jsonl if the conversation is kept.
CALL_FEED_MAX = 40
CALL_TEXT_MAX = 700
LATENCY_WINDOW = 200
CALL_SEED = "seed"
PCT_MID = 0.50
PCT_TAIL = 0.95

OPENER_SYSTEM = (
    "You write opening lines for conversations: the first thing a person says when they "
    "start talking to someone knowledgeable. Each is a real question or a real situation in "
    "that person's own words. Vary the subject widely, vary the phrasing, vary how formal "
    "they are. Output one per line, nothing else, no numbering, no quotes."
)
OPENER_BATCH = 12
OPENER_MAX_TOKENS = 700
MIN_OPENER_BYTES = 12
MAX_OPENER_BYTES = 400

FAILURE_ABORT_STREAK = 12
# Consecutive opener batches that add nothing new before the pool is called dry.
DRY_ROUNDS = 3
OPENER_ROUND_SLACK = 4

# ------------------------------------------------------------------------------------
# Functions

def _read_jsonl(path):
    """Every line of a jsonl file that parses. A run appends and flushes as it
    goes, so the last line of an interrupted file can be half written."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def clip(text, limit=CALL_TEXT_MAX):
    """Cut display text to a byte ceiling on a character boundary."""
    raw = (text or "").encode("utf-8")
    if len(raw) <= limit:
        return text or ""
    return raw[:limit].decode("utf-8", "ignore") + "..."


def percentile(ordered, q):
    if not ordered:
        return 0
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))])


def conversation_id(genre_id, idx):
    return f"{genre_id}_{idx:05d}"


class CallFeed:
    """What the run is saying to the teacher and hearing back, call by call.

    Keyed by thread because one worker thread owns one conversation at a time,
    which is what pairs a reply with the request that is still open."""

    def __init__(self):
        self._lock = threading.Lock()
        self._recent = collections.deque(maxlen=CALL_FEED_MAX)
        self._latency = collections.deque(maxlen=LATENCY_WINDOW)
        self._inflight = {}
        self._seq = 0
        self._calls = 0
        self._failed = 0
        self._salvaged = 0
        self._reply_bytes = 0
        self._started = time.time()

    def watcher(self, cid):
        """The callable interview.ask() reports both edges of a call to."""
        return functools.partial(self._event, cid)

    def _event(self, cid, phase, kind, text, ms):
        tid = threading.get_ident()
        with self._lock:
            if phase == interview.PHASE_SALVAGE:
                self._salvaged += 1
                return
            if phase == interview.PHASE_START:
                self._seq += 1
                self._inflight[tid] = {
                    "seq": self._seq, "id": cid, "kind": kind,
                    "sent": clip(text), "sent_bytes": len(text.encode("utf-8")),
                    "started": time.time(),
                }
                return
            call = self._inflight.pop(tid, None)
            if call is None:
                return
            if phase == interview.PHASE_FAIL:
                # Kept out of the latency window on purpose: a call that died on
                # a 60 s socket timeout is not a reply time and would drag the
                # percentiles somewhere they cannot be read.
                call["error"] = clip(text)
                call["ms"] = round(ms)
                self._recent.append(call)
                self._failed += 1
                return
            call["got"] = clip(text)
            call["got_bytes"] = len(text.encode("utf-8"))
            call["ms"] = round(ms)
            self._recent.append(call)
            self._latency.append(ms)
            self._calls += 1
            self._reply_bytes += call["got_bytes"]

    def drop(self):
        """Forget this thread's open call. A call that raised never reports a
        reply, and its row would otherwise sit in the panel as in flight until
        the same thread happened to start another."""
        with self._lock:
            self._inflight.pop(threading.get_ident(), None)

    def snapshot(self):
        now = time.time()
        with self._lock:
            recent = list(self._recent)
            inflight = [dict(c, elapsed_ms=round((now - c["started"]) * 1000))
                        for c in sorted(self._inflight.values(), key=lambda c: c["seq"])]
            ordered = sorted(self._latency)
            calls, failed, salvaged = self._calls, self._failed, self._salvaged
            reply_bytes = self._reply_bytes
            elapsed = max(1.0, now - self._started)
        return {
            "calls": recent[::-1],
            "inflight": inflight,
            "stats": {
                "calls": calls,
                "failed": failed,
                "salvaged": salvaged,
                "in_flight": len(inflight),
                "reply_bytes": reply_bytes,
                "per_min": round(calls * 60.0 / elapsed, 1),
                "p50_ms": percentile(ordered, PCT_MID),
                "p95_ms": percentile(ordered, PCT_TAIL),
            },
        }


class InterviewJob:
    """Two-pass conversation generation as a stoppable, resumable job."""

    def __init__(self, job_id, output_dir, spec, genre_ids, n_conversations, depth,
                 provider, model=None, base_url=None, api_key=None,
                 temperature=0.9, max_concurrency=4, gate=None, seed=0, seeds=None):
        self.job_id = job_id
        self.output_dir = output_dir
        self.spec = spec
        self.genre_ids = list(genre_ids)
        self.n_conversations = int(n_conversations)
        self.depth = int(depth)
        self.temperature = float(temperature)
        self.max_concurrency = max(1, int(max_concurrency))
        self.gate = gate if gate is not None else authoring_mod.RecordGate(spec)
        self.seed = int(seed)
        # Subject seeds chosen by the user in the dashboard. These replace the
        # genre's own `situations` list, which is thin (conversation ships 14) and
        # was never meant to carry a whole corpus.
        self.seeds = list(seeds) if seeds else []
        self._client_args = (provider, model, base_url, api_key)
        self._stop = threading.Event()
        # Two tiers. stop() drops the queue but lets conversations already in
        # flight finish and be written -- bounded by one conversation. kill()
        # abandons those too and returns at once, for when the teacher is hung
        # and every in-flight call is sitting on its 60 s socket timeout.
        self._kill = threading.Event()
        self._lock = threading.Lock()
        self._aborted = False
        self._counts = {"completed": 0, "failed": 0, "consec_fail": 0, "last_error": ""}
        self._err_counts = collections.Counter()
        self._remaining = []
        self._opener_shortfall = {}
        self.feed = CallFeed()

    # -- lifecycle ------------------------------------------------------------

    def stop(self):
        self._stop.set()

    def kill(self):
        self._kill.set()
        self._stop.set()

    def _client(self):
        provider, model, base_url, api_key = self._client_args
        return Client(provider, model=model, base_url=base_url, api_key=api_key)

    def _path(self, name):
        return os.path.join(self.output_dir, name)

    # -- pass 1: openers ------------------------------------------------------

    def _openers_for(self, client, genre, want, taken):
        """Ask the teacher for opening lines. Scripting IS the right task here --
        short varied user turns are what a script writer is good at.

        Deduped against `taken` (openers this job already has) as well as against
        this batch, appended to disk as they arrive so an interrupted pass 1 is
        not paid for twice, and given up on after DRY_ROUNDS consecutive batches
        that add nothing new. That last is the real ceiling on this mode: the
        opener pool is bounded by the seeds behind it, so asking for far more
        conversations than the subject pool can carry yields repeats, which
        RecordGate would then reject as near-duplicates after paying for the
        generation. Better to stop early and say so."""
        rng = random.Random(self.seed + (hash(genre["id"]) % 10_000))
        situations = self.seeds or list(genre.get("situations") or ["anything"])
        order = list(range(len(situations)))
        rng.shuffle(order)
        out = []
        seen = {authoring_mod.normalize(t) for t in taken}
        dry, fails = 0, 0
        max_rounds = max(4, (want // OPENER_BATCH) * OPENER_ROUND_SLACK)
        rounds = 0
        store = open(self._path(OPENERS_FILE), "a", encoding="utf-8")  # noqa: SIM115
        try:
            while len(out) < want and rounds < max_rounds and dry < DRY_ROUNDS:
                if self._stop.is_set():
                    break
                # walk the seed list rather than sampling with replacement: every seed
                # is used once before any repeats, which is where the diversity is
                sit = situations[order[rounds % len(order)]]
                rounds += 1
                prompt = (f"Subject area: {sit}\nWrite {OPENER_BATCH} different opening lines. "
                          f"Batch tag {rng.randrange(10**6):06d} (never output this).")
                try:
                    raw = interview.ask(client, [{"role": "user", "content": prompt}],
                                        OPENER_SYSTEM, max(self.temperature, 0.95),
                                        OPENER_MAX_TOKENS, self._stop.is_set,
                                        self.feed.watcher(genre["id"]), CALL_SEED)
                except TeacherError as e:
                    # A teacher that is down fails every batch. Without this the
                    # loop sat here for max_rounds x the client's own retries,
                    # which on a large run is hours of nothing.
                    fails += 1
                    self._counts["last_error"] = f"openers: {type(e).__name__}: {e}"
                    self._err_counts[str(self._counts["last_error"])[:REASON_KEY_MAX]] += 1
                    if fails >= FAILURE_ABORT_STREAK:
                        self._aborted = True
                        break
                    continue
                fails = 0
                added = 0
                for line in (raw or "").splitlines():
                    line = line.strip().strip('"').strip()
                    line = line.lstrip("0123456789.)-\u2022 ").strip()
                    if not MIN_OPENER_BYTES <= len(line.encode("utf-8")) <= MAX_OPENER_BYTES:
                        continue
                    key = authoring_mod.normalize(line)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(line)
                    store.write(json.dumps({"genre": genre["id"], "text": line},
                                           ensure_ascii=False) + "\n")
                    added += 1
                if added:
                    store.flush()
                dry = dry + 1 if added == 0 else 0
        finally:
            store.close()
        return out[:want]

    def _stored_openers(self):
        """Openers this job already paid for and has not answered yet.

        Answered ones are read back out of samples.jsonl by their first user turn:
        re-asking one would cost a full conversation for a record the gate then
        rejects as a near duplicate of the one already on disk."""
        pool = {}
        path = self._path(OPENERS_FILE)
        if not os.path.isfile(path):
            return pool
        used = self._answered_openers()
        seen = set()
        for row in _read_jsonl(path):
            text = row.get("text")
            if not isinstance(text, str):
                continue
            key = authoring_mod.normalize(text)
            if key in used or key in seen:
                continue
            seen.add(key)
            pool.setdefault(str(row.get("genre", "")), []).append(text)
        return pool

    def _answered_openers(self):
        """Normalised first user turn of every conversation already written."""
        used = set()
        for row in _read_jsonl(self._path(SAMPLES_FILE)):
            rec = row.get(authoring_mod.RECORD_KEY)
            if not isinstance(rec, dict):
                continue
            turns = rec.get(authoring_mod.TURNS_KEY) or []
            if turns:
                used.add(authoring_mod.normalize(turns[0].get("text", "")))
        return used

    # -- pass 2: the conversation --------------------------------------------

    def _one(self, client, genre, opener, idx):
        # A stopped run must not start work that was merely queued. The pool
        # holds one future per planned conversation -- ten thousand of them on a
        # big run -- and without this check every queued task still made its
        # teacher calls after Stop was pressed.
        if self._stop.is_set():
            return None, "stopped"
        cid = conversation_id(genre["id"], idx)
        try:
            turns = interview.build_conversation(
                client, opener, self.depth, seed=self.seed + idx,
                temperature=self.temperature, cancel_check=self._stop.is_set,
                watch=self.feed.watcher(cid))
        finally:
            self.feed.drop()
        if not turns:
            return None, "empty conversation"
        record = {
            authoring_mod.GENRE_KEY: genre["id"],
            authoring_mod.VOICE_KEY: "interview",
            authoring_mod.TURNS_KEY: turns,
        }
        # Straight through the existing gate: ban list, em dashes, duplicates,
        # near-duplicates, variety floor. Nothing here gets a free pass.
        kept, why = self.gate({"genre": genre["id"]}, json.dumps(record, ensure_ascii=False))
        if not kept:
            # `why` is the gate's own reason list (ban list, em dash, duplicate,
            # near-duplicate, variety floor). Reporting the bare word "rejected"
            # threw away the only information that says how to fix the corpus.
            reason = ", ".join(str(w) for w in why) if why else "unknown check"
            return None, f"gate: {reason}"
        return kept[0], None

    # -- run ------------------------------------------------------------------

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        client = self._client()
        genres = [authoring_mod.genre_by_id(self.spec, g) for g in self.genre_ids]
        genres = [g for g in genres if g is not None] or [
            authoring_mod.genre_by_id(self.spec, "conversation")]

        per = max(1, self.n_conversations // len(genres))
        stored = self._stored_openers()
        plan = []
        for g in genres:
            have = stored.get(g["id"], [])[:per]
            fresh = self._openers_for(client, g, per - len(have), have) if len(have) < per else []
            self._opener_shortfall[g["id"]] = max(0, per - len(have) - len(fresh))
            for op in have + fresh:
                plan.append((g, op))
        self._remaining = [conversation_id(g["id"], i) for i, (g, _) in enumerate(plan)]
        self._flush()

        samples = open(self._path(SAMPLES_FILE), "a", encoding="utf-8")  # noqa: SIM115
        errors = open(self._path(ERRORS_FILE), "a", encoding="utf-8")    # noqa: SIM115
        # Not a `with` block: its __exit__ is shutdown(wait=True), which drains
        # every queued future before returning. Pressing Stop on a 10,000
        # conversation run would sit there working through the whole backlog.
        ex = ThreadPoolExecutor(max_workers=self.max_concurrency)
        try:
            futures = {
                ex.submit(self._one, self._client(), g, op, i): (i, g)
                for i, (g, op) in enumerate(plan)
            }
            dropped_queue = False
            try:
                for fut in as_completed(futures):
                    # A hard kill abandons results as well as work. Stop does not:
                    # it drops what is still queued and keeps reading the rest, so
                    # a conversation that was in flight is written like any other
                    # instead of being paid for and thrown away.
                    if self._kill.is_set():
                        break
                    if self._stop.is_set() and not dropped_queue:
                        dropped_queue = True
                        for queued in futures:
                            queued.cancel()
                    if fut.cancelled():
                        continue
                    i, g = futures[fut]
                    cid = conversation_id(g["id"], i)
                    try:
                        rec, why = fut.result()
                    except Exception as e:                       # noqa: BLE001
                        rec, why = None, f"{type(e).__name__}: {e}"
                    with self._lock:
                        if cid in self._remaining:
                            self._remaining.remove(cid)
                        if rec is not None:
                            samples.write(json.dumps(
                                {"id": cid, authoring_mod.RECORD_KEY: rec},
                                ensure_ascii=False) + "\n")
                            samples.flush()
                            self._counts["completed"] += 1
                            self._counts["consec_fail"] = 0
                        else:
                            self._counts["failed"] += 1
                            self._counts["consec_fail"] += 1
                            self._counts["last_error"] = why or "rejected"
                            self._err_counts[(why or "rejected")[:REASON_KEY_MAX]] += 1
                            errors.write(json.dumps(
                                {"id": cid, "error": why, "ts": int(time.time())}) + "\n")
                            errors.flush()
                            if self._counts["consec_fail"] >= FAILURE_ABORT_STREAK:
                                self._aborted = True
                                self._stop.set()
                        done = self._counts["completed"] + self._counts["failed"]
                    if done % STATE_FLUSH_EVERY == 0:
                        # flush() alone survives the process dying; fsync is what
                        # survives the machine dying, and bounds what a power cut
                        # can take to STATE_FLUSH_EVERY records.
                        os.fsync(samples.fileno())
                        self._flush()
            finally:
                # cancel_futures drops everything still queued instead of running
                # it. Waiting is skipped only on a hard kill.
                ex.shutdown(wait=not self._kill.is_set(), cancel_futures=True)
        finally:
            os.fsync(samples.fileno())
            samples.close()
            errors.close()
            self._flush()

    def _flush(self):
        payload = {
            "job_id": self.job_id,
            "remaining": list(self._remaining),
            "completed": self._counts["completed"],
            "failed": self._counts["failed"],
            "skipped_dup": 0,
            "last_error": self._counts["last_error"],
            "error_summary": dict(self._err_counts.most_common(ERROR_SUMMARY_TOP)),
            "aborted": self._aborted,
            "opener_shortfall": {k: v for k, v in self._opener_shortfall.items() if v},
            # the live feed is memory on the running job; these are the numbers
            # that have to survive it, so they go in state.json with the rest
            "call_stats": self.feed.snapshot()["stats"],
            "authoring": self.gate.stats(),
        }
        tmp = self._path(STATE_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, self._path(STATE_FILE))
