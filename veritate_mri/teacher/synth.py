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
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from runtime import logs as logmod
except ImportError:
    logmod = None

from .client import Client, TeacherError
from .providers import DEFAULT_MAX_CONCURRENCY, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
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
CACHE_DB_NAME = "cache.sqlite"
SAMPLES_FILE_NAME = "samples.jsonl"
STATE_FILE_NAME = "state.json"

_CACHE_SCHEMA = "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, ts INTEGER)"

# ------------------------------------------------------------------------------------
# Functions

def _log_warn(source, msg):
    if logmod is not None:
        logmod.warn(source, msg)


def _cache_key(provider_id, model, messages, temperature):
    payload = json.dumps({
        "p": provider_id,
        "m": model,
        "ms": messages,
        "t": temperature,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        return Client(self.provider_id, model=self.model, base_url=self.base_url, api_key=self.api_key)

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
        completed = 0
        failed = 0
        skipped_dup = 0
        seen_hashes = set()
        remaining = {p["id"] for p in pending}

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
                                       max_tokens=self.max_tokens, system=system)
                self._cache_put(conn, key, resp)
                return pid, resp, None, False
            except (TeacherError, Exception) as e:
                return pid, None, str(e), False

        with open(samples_path, "a", encoding="utf-8") as fp:
            ex = ThreadPoolExecutor(max_workers=max(1, self.max_concurrency))
            try:
                futures = {ex.submit(worker, p): p for p in pending}
                processed = 0
                for fut in as_completed(futures):
                    if self._stop.is_set():
                        break
                    prompt = futures[fut]
                    pid = prompt["id"]
                    try:
                        rid, resp, err, _cached = fut.result()
                    except Exception as e:
                        _log_warn("teacher", f"job {self.job_id} prompt {pid} crash: {e}")
                        failed += 1
                        remaining.discard(pid)
                        processed += 1
                        continue
                    if err is not None or resp is None:
                        _log_warn("teacher", f"job {self.job_id} prompt {pid} failed: {err}")
                        failed += 1
                        remaining.discard(pid)
                        processed += 1
                        continue
                    text = resp
                    if self.format == "json":
                        text_stripped = strip_code_fence(text)
                        if not is_json_valid(text_stripped):
                            _log_warn("teacher", f"job {self.job_id} prompt {pid} invalid json")
                            failed += 1
                            remaining.discard(pid)
                            processed += 1
                            continue
                    if not is_length_ok(text, self.min_chars, self.max_chars):
                        _log_warn("teacher", f"job {self.job_id} prompt {pid} length out of bounds")
                        failed += 1
                        remaining.discard(pid)
                        processed += 1
                        continue
                    h = simhash64(text)
                    if is_near_dup(h, seen_hashes, self.hamming_threshold):
                        skipped_dup += 1
                        remaining.discard(pid)
                        processed += 1
                        continue
                    seen_hashes.add(h)
                    row = {
                        "id": pid,
                        "response": text,
                        "provider": self.provider_id,
                        "model": self.model,
                        "ts": int(time.time()),
                    }
                    with self._lock:
                        self._append_sample(fp, row)
                    completed += 1
                    remaining.discard(pid)
                    processed += 1
                    if processed % STATE_FLUSH_EVERY == 0:
                        self._write_state(remaining, {"completed": completed, "failed": failed, "skipped_dup": skipped_dup})
            finally:
                ex.shutdown(wait=False, cancel_futures=True)
        self._write_state(remaining, {"completed": completed, "failed": failed, "skipped_dup": skipped_dup})
        conn.close()
        return {
            "completed": completed,
            "failed": failed,
            "skipped_dup": skipped_dup,
            "output_path": samples_path,
        }
