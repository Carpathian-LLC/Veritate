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
STATE_FLUSH_EVERY = 5

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
        self._lock = threading.Lock()
        self._aborted = False
        self._counts = {"completed": 0, "failed": 0, "consec_fail": 0, "last_error": ""}
        self._remaining = []
        self._opener_shortfall = {}

    # -- lifecycle ------------------------------------------------------------

    def stop(self):
        self._stop.set()

    def _client(self):
        provider, model, base_url, api_key = self._client_args
        return Client(provider, model=model, base_url=base_url, api_key=api_key)

    def _path(self, name):
        return os.path.join(self.output_dir, name)

    # -- pass 1: openers ------------------------------------------------------

    def _openers_for(self, client, genre, want):
        """Ask the teacher for opening lines. Scripting IS the right task here --
        short varied user turns are what a script writer is good at.

        Deduped as they arrive, and the loop gives up after DRY_ROUNDS consecutive
        batches that add nothing new. That is the real ceiling on this mode: the
        opener pool is bounded by the genre's `situations` list (conversation has
        14), so asking for far more conversations than the subject pool can carry
        yields repeats, which RecordGate would then reject as near-duplicates
        after paying for the generation. Better to stop early and say so."""
        rng = random.Random(self.seed + (hash(genre["id"]) % 10_000))
        situations = self.seeds or list(genre.get("situations") or ["anything"])
        order = list(range(len(situations)))
        rng.shuffle(order)
        out, seen, dry = [], set(), 0
        max_rounds = max(4, (want // OPENER_BATCH) * OPENER_ROUND_SLACK)
        rounds = 0
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
                raw = client.complete([{"role": "user", "content": prompt}],
                                      temperature=max(self.temperature, 0.95),
                                      max_tokens=OPENER_MAX_TOKENS, system=OPENER_SYSTEM,
                                      cancel_check=self._stop.is_set)
            except TeacherError:
                continue
            added = 0
            for line in (raw or "").splitlines():
                line = line.strip().strip('"').strip()
                line = line.lstrip("0123456789.)-• ").strip()
                if not MIN_OPENER_BYTES <= len(line.encode("utf-8")) <= MAX_OPENER_BYTES:
                    continue
                key = authoring_mod.normalize(line)
                if key in seen:
                    continue
                seen.add(key)
                out.append(line)
                added += 1
            dry = dry + 1 if added == 0 else 0
        self._opener_shortfall[genre["id"]] = max(0, want - len(out))
        return out[:want]

    # -- pass 2: the conversation --------------------------------------------

    def _one(self, client, genre, opener, idx):
        turns = interview.build_conversation(
            client, opener, self.depth, seed=self.seed + idx,
            temperature=self.temperature, cancel_check=self._stop.is_set)
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
            return None, (why[0] if why else "rejected")
        return kept[0], None

    # -- run ------------------------------------------------------------------

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        client = self._client()
        genres = [authoring_mod.genre_by_id(self.spec, g) for g in self.genre_ids]
        genres = [g for g in genres if g is not None] or [
            authoring_mod.genre_by_id(self.spec, "conversation")]

        per = max(1, self.n_conversations // len(genres))
        plan = []
        for g in genres:
            for op in self._openers_for(client, g, per):
                plan.append((g, op))
        self._remaining = [f"{g['id']}_{i:05d}" for i, (g, _) in enumerate(plan)]
        self._flush()

        samples = open(self._path(SAMPLES_FILE), "a", encoding="utf-8")  # noqa: SIM115
        errors = open(self._path(ERRORS_FILE), "a", encoding="utf-8")    # noqa: SIM115
        try:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
                futures = {
                    ex.submit(self._one, self._client(), g, op, i): (i, g)
                    for i, (g, op) in enumerate(plan)
                }
                for fut in as_completed(futures):
                    if self._stop.is_set():
                        break
                    i, g = futures[fut]
                    cid = f"{g['id']}_{i:05d}"
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
                            errors.write(json.dumps(
                                {"id": cid, "error": why, "ts": int(time.time())}) + "\n")
                            errors.flush()
                            if self._counts["consec_fail"] >= FAILURE_ABORT_STREAK:
                                self._aborted = True
                                self._stop.set()
                        done = self._counts["completed"] + self._counts["failed"]
                    if done % STATE_FLUSH_EVERY == 0:
                        self._flush()
        finally:
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
            "error_summary": {},
            "aborted": self._aborted,
            "opener_shortfall": {k: v for k, v in self._opener_shortfall.items() if v},
            "authoring": self.gate.stats(),
        }
        tmp = self._path(STATE_FILE) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, self._path(STATE_FILE))
