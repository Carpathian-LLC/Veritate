# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - synth job runner. consumes a prompt list, calls the teacher client, applies
#   quality gates, writes accepted samples to jsonl, tracks state for resume,
#   caches by content hash in sqlite. ThreadPoolExecutor concurrency; main
#   thread serializes disk writes.
# veritate_mri/teacher/synth.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

try:
    from runtime import logs as logmod
except ImportError:
    logmod = None

from .client import Client
from .providers import DEFAULT_MAX_CONCURRENCY, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, get_provider
from .quality import (
    DEFAULT_HAMMING_THRESHOLD,
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    is_json_valid,
    is_length_ok,
    is_near_dup,
    simhash64,
    strip_code_fence,
)

# ------------------------------------------------------------------------------------
# Constants

STATE_FLUSH_EVERY = 10
SYNTH_REQUEST_TIMEOUT_S = 180
SYNTH_MAX_RETRIES = 2
CACHE_DB_NAME = "cache.sqlite"
SAMPLES_FILE_NAME = "samples.jsonl"
STATE_FILE_NAME = "state.json"
ERRORS_FILE_NAME = "errors.jsonl"

# Circuit breaker: abort a job that is mostly failing instead of grinding all
# night. Trips on a consecutive-failure streak or a high failure rate after a
# warmup window. Keeps the teacher backlog bounded so stop is immediate.
FAILURE_ABORT_STREAK = 25
FAILURE_RATE_WARMUP = 20
FAILURE_RATE_ABORT = 0.85
ERROR_SUMMARY_TOP = 5
REASON_KEY_MAX = 80

_CACHE_SCHEMA = "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, ts INTEGER)"

# ------------------------------------------------------------------------------------
# Functions

def _log_info(source, msg):
    if logmod is not None:
        logmod.info(source, msg)


def _log_warn(source, msg):
    if logmod is not None:
        logmod.warn(source, msg)


def _log_error(source, msg):
    if logmod is not None:
        logmod.error(source, msg)


def _cache_key(provider_id, model, messages, temperature):
    payload = json.dumps({
        "p": provider_id,
        "m": model,
        "ms": messages,
        "t": temperature,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _norm_reason(reason):
    return (reason or "unknown")[:REASON_KEY_MAX]


def _load_done_ids(samples_path):
    done = set()
    if not os.path.isfile(samples_path):
        return done
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "id" in row:
                    done.add(row["id"])
            except json.JSONDecodeError:
                continue
    return done


class SynthJob:
    def __init__(self, job_id, provider_id, model, prompts, output_dir, **opts):
        self.job_id = job_id
        self.provider_id = provider_id
        self.model = model
        self.prompts = list(prompts)
        self.output_dir = output_dir
        self.opts = dict(opts)
        self.temperature = opts.get("temperature", DEFAULT_TEMPERATURE)
        self.max_tokens = opts.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.max_concurrency = int(opts.get("max_concurrency", DEFAULT_MAX_CONCURRENCY))
        self.timeout_s = int(opts.get("timeout_s", SYNTH_REQUEST_TIMEOUT_S))
        self.max_retries = int(opts.get("max_retries", SYNTH_MAX_RETRIES))
        self._aborted = False
        self.min_chars = int(opts.get("min_chars", DEFAULT_MIN_CHARS))
        self.max_chars = int(opts.get("max_chars", DEFAULT_MAX_CHARS))
        self.hamming_threshold = int(opts.get("hamming_threshold", DEFAULT_HAMMING_THRESHOLD))
        self.format = opts.get("format")
        self.base_url = opts.get("base_url")
        self.api_key = opts.get("api_key")
        self.client_factory = opts.get("client_factory")
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _samples_path(self):
        return os.path.join(self.output_dir, SAMPLES_FILE_NAME)

    def _state_path(self):
        return os.path.join(self.output_dir, STATE_FILE_NAME)

    def _cache_path(self):
        return os.path.join(self.output_dir, CACHE_DB_NAME)

    def _errors_path(self):
        return os.path.join(self.output_dir, ERRORS_FILE_NAME)

    def _open_cache(self):
        conn = sqlite3.connect(self._cache_path(), check_same_thread=False)
        conn.execute(_CACHE_SCHEMA)
        conn.commit()
        return conn

    def _cache_get(self, conn, key):
        with self._cache_lock:
            cur = conn.execute("SELECT response FROM cache WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def _cache_put(self, conn, key, response):
        with self._cache_lock:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, response, ts) VALUES (?, ?, ?)",
                (key, response, int(time.time())),
            )
            conn.commit()

    def _make_client(self):
        if self.client_factory is not None:
            return self.client_factory()
        return Client(self.provider_id, model=self.model, base_url=self.base_url,
                      api_key=self.api_key, timeout_s=self.timeout_s,
                      max_retries=self.max_retries)

    def _unload(self, client):
        try:
            local = get_provider(self.provider_id).get("kind") == "local"
        except ValueError:
            return
        if local and client.unload(self.model):
            _log_info("teacher", f"job {self.job_id} freed {self.model} from server memory")

    def _write_state(self, remaining_ids, counts=None):
        tmp = self._state_path() + ".tmp"
        payload = {"job_id": self.job_id, "remaining": list(remaining_ids)}
        if counts is not None:
            payload.update(counts)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, self._state_path())

    def _append_sample(self, fp, row):
        fp.write(json.dumps(row) + "\n")
        fp.flush()

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        samples_path = self._samples_path()
        done_ids = _load_done_ids(samples_path)
        pending = [p for p in self.prompts if p["id"] not in done_ids]
        conn = self._open_cache()
        client = self._make_client()
        seen_hashes = set()
        remaining = {p["id"] for p in pending}
        workers = max(1, self.max_concurrency)
        c = {"completed": 0, "failed": 0, "skipped": 0, "processed": 0,
             "consec_fail": 0, "last_error": ""}
        err_counts = Counter()

        def worker(prompt):
            pid = prompt["id"]
            messages = prompt.get("messages")
            if not messages:
                content = prompt.get("prompt") or prompt.get("text") or ""
                messages = [{"role": "user", "content": content}]
            system = prompt.get("system")
            send_msgs = list(messages)
            if system is not None:
                send_msgs = [{"role": "system", "content": system}] + send_msgs
            key = _cache_key(self.provider_id, self.model, send_msgs, self.temperature)
            cached = self._cache_get(conn, key)
            if cached is not None:
                return pid, cached, None, True
            try:
                resp = client.complete(messages, temperature=self.temperature,
                                       max_tokens=self.max_tokens, system=system,
                                       cancel_check=self._stop.is_set)
                self._cache_put(conn, key, resp)
                return pid, resp, None, False
            except Exception as e:
                return pid, None, str(e), False

        def flush_state():
            self._write_state(remaining, {
                "completed": c["completed"], "failed": c["failed"],
                "skipped_dup": c["skipped"], "last_error": c["last_error"],
                "error_summary": dict(err_counts.most_common(ERROR_SUMMARY_TOP)),
                "aborted": self._aborted,
            })

        def maybe_abort():
            if self._stop.is_set():
                return
            if c["consec_fail"] >= FAILURE_ABORT_STREAK:
                why = f"{c['consec_fail']} consecutive failures"
            elif c["processed"] >= FAILURE_RATE_WARMUP and \
                    c["failed"] / c["processed"] >= FAILURE_RATE_ABORT:
                why = f"{c['failed']}/{c['processed']} failed"
            else:
                return
            self._aborted = True
            self._stop.set()
            top = err_counts.most_common(1)
            reason = top[0][0] if top else c["last_error"]
            _log_error("teacher", f"job {self.job_id} ABORTED ({why}); dominant error: {reason}. "
                                  f"check teacher provider/model and set concurrency to match the server.")

        def record_failure(efp, pid, reason):
            c["failed"] += 1
            c["processed"] += 1
            c["consec_fail"] += 1
            c["last_error"] = reason
            err_counts[_norm_reason(reason)] += 1
            remaining.discard(pid)
            efp.write(json.dumps({"id": pid, "error": reason, "ts": int(time.time())}) + "\n")
            efp.flush()
            _log_warn("teacher", f"job {self.job_id} prompt {pid} failed: {reason}")
            maybe_abort()

        def record_ok(pid, key, value):
            c[key] += 1
            c["processed"] += 1
            c["consec_fail"] = 0
            remaining.discard(pid)
            if value is not None:
                with self._lock:
                    self._append_sample(samples_fp, value)
            if c["processed"] % STATE_FLUSH_EVERY == 0:
                flush_state()

        samples_fp = open(samples_path, "a", encoding="utf-8")
        errors_fp = open(self._errors_path(), "a", encoding="utf-8")
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            pending_iter = iter(pending)
            inflight = {}

            def submit_next():
                while len(inflight) < workers:
                    p = next(pending_iter, None)
                    if p is None:
                        return
                    inflight[ex.submit(worker, p)] = p

            submit_next()
            while inflight and not self._stop.is_set():
                done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    prompt = inflight.pop(fut)
                    pid = prompt["id"]
                    try:
                        _rid, resp, err, _cached = fut.result()
                    except Exception as e:
                        record_failure(errors_fp, pid, f"crash: {e}")
                        continue
                    if err is not None or resp is None:
                        record_failure(errors_fp, pid, err or "empty response")
                        continue
                    if self.format == "json" and not is_json_valid(strip_code_fence(resp)):
                        record_failure(errors_fp, pid, "invalid json")
                        continue
                    if not is_length_ok(resp, self.min_chars, self.max_chars):
                        record_failure(errors_fp, pid,
                                       f"filtered by quality gate: reply length outside "
                                       f"{self.min_chars}-{self.max_chars} chars (too short or too long)")
                        continue
                    h = simhash64(resp)
                    if is_near_dup(h, seen_hashes, self.hamming_threshold):
                        record_ok(pid, "skipped", None)
                        continue
                    seen_hashes.add(h)
                    record_ok(pid, "completed", {
                        "id": pid, "response": resp, "provider": self.provider_id,
                        "model": self.model, "ts": int(time.time())})
                if not self._stop.is_set():
                    submit_next()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            samples_fp.close()
            errors_fp.close()
        flush_state()
        self._unload(client)
        conn.close()
        return {
            "completed": c["completed"],
            "failed": c["failed"],
            "skipped_dup": c["skipped"],
            "aborted": self._aborted,
            "output_path": samples_path,
        }
