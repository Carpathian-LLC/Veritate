# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - teacher model routes. config GET/POST, connection probe, synth job
#   start/status. api key value never leaves the server; responses surface
#   only has_api_key. job handles live in a process-local dict guarded by a
#   lock and queried by reading state.json + samples.jsonl on disk.
# veritate_mri/routes/teacher_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import importlib
import json
import os
import threading
import uuid

from flask import request

from readers.paths import REPO_ROOT
from runtime import settings as settings_mod

# ------------------------------------------------------------------------------------
# Constants

JOB_ID_LEN = 12
SYNTH_JOBS_DIR = "synth_jobs"
TEACHER_API_KEY_ENV = "VERITATE_TEACHER_API_KEY"
SAMPLES_FILE = "samples.jsonl"
TEACHER_PKG = "teacher"
SEEDS_DIR = os.path.join(REPO_ROOT, "veritate_mri", "data", "seeds")
SEED_CATALOG_FILE = "seed_catalog.json"
CATALOG_VERSION = 1

_JOBS = {}
_JOBS_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions

def _teacher_mods():
    return (importlib.import_module(TEACHER_PKG),
            importlib.import_module(TEACHER_PKG + ".synth"),
            importlib.import_module(TEACHER_PKG + ".test_connection"))


def _public_view(s, teacher_mod):
    has_key = bool(os.environ.get(TEACHER_API_KEY_ENV)) or bool(s.get("teacher_api_key"))
    return {
        "providers": teacher_mod.list_providers(),
        "configured": bool(s.get("teacher_provider")),
        "provider": s.get("teacher_provider", ""),
        "model": s.get("teacher_model", ""),
        "base_url": s.get("teacher_base_url", ""),
        "has_api_key": has_key,
        "max_concurrency": int(s.get("teacher_max_concurrency", 0)),
        "max_tokens": int(s.get("teacher_max_tokens", 0)),
        "temperature": float(s.get("teacher_temperature", 0.0)),
    }


def _count_lines(path):
    if not os.path.isfile(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _load_seed_catalog():
    path = os.path.join(SEEDS_DIR, SEED_CATALOG_FILE)
    if not os.path.isfile(path):
        return {"version": CATALOG_VERSION, "seeds": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _seed_entry(seed_id):
    cat = _load_seed_catalog()
    for s in cat.get("seeds", []):
        if s.get("id") == seed_id:
            return s
    return None


def _read_seed_prompts(seed_id):
    entry = _seed_entry(seed_id)
    if entry is None:
        return None
    fname = entry.get("file") or ""
    path = os.path.join(SEEDS_DIR, fname)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            out.append(rec)
    return out


def _read_state_counts(output_dir):
    samples = os.path.join(output_dir, SAMPLES_FILE)
    return {
        "completed": _count_lines(samples),
        "failed": 0,
        "skipped_dup": 0,
        "output_path": samples,
    }


def register(app):
    teacher_mod, synth_mod, test_mod = _teacher_mods()

    @app.route("/teacher", methods=["GET", "POST"])
    def teacher_route():
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            try:
                settings_mod.update(body)
            except ValueError as ve:
                return {"error": str(ve)}, 400
        return _public_view(settings_mod.get(), teacher_mod)

    @app.route("/teacher/test", methods=["POST"])
    def teacher_test_route():
        body = request.get_json(silent=True) or {}
        s = settings_mod.get()
        provider = body.get("provider") or s.get("teacher_provider") or ""
        model = body.get("model") or s.get("teacher_model") or None
        base_url = body.get("base_url") or s.get("teacher_base_url") or None
        api_key = (body.get("api_key")
                   or os.environ.get(TEACHER_API_KEY_ENV)
                   or s.get("teacher_api_key")
                   or None)
        if not provider:
            return {"error": "provider required"}, 400
        return test_mod.test(provider, model=model, base_url=base_url, api_key=api_key)

    @app.route("/teacher/synth/start", methods=["POST"])
    def teacher_synth_start_route():
        body = request.get_json(silent=True) or {}
        prompts = body.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            return {"error": "prompts must be a non-empty list"}, 400
        fmt = body.get("format")
        s = settings_mod.get()
        provider = s.get("teacher_provider") or ""
        model = s.get("teacher_model") or None
        if not provider:
            return {"error": "teacher_provider not configured"}, 400
        job_id = uuid.uuid4().hex[:JOB_ID_LEN]
        out_root = body.get("output_dir") or os.path.join(REPO_ROOT, SYNTH_JOBS_DIR, job_id)
        os.makedirs(out_root, exist_ok=True)
        api_key = os.environ.get(TEACHER_API_KEY_ENV) or s.get("teacher_api_key") or None
        job = synth_mod.SynthJob(
            job_id, provider, model, prompts, out_root,
            format=fmt,
            base_url=s.get("teacher_base_url") or None,
            api_key=api_key,
            temperature=float(s.get("teacher_temperature", 0.7)),
            max_tokens=int(s.get("teacher_max_tokens", 2048)),
            max_concurrency=int(s.get("teacher_max_concurrency", 16)),
        )
        thread = threading.Thread(target=job.run, name=f"teacher-synth-{job_id}", daemon=True)
        with _JOBS_LOCK:
            _JOBS[job_id] = {"job": job, "thread": thread, "output_dir": out_root}
        thread.start()
        return {"job_id": job_id, "output_dir": out_root}

    @app.route("/teacher/seeds", methods=["GET"])
    def teacher_seeds_route():
        cat = _load_seed_catalog()
        seeds = cat.get("seeds", [])
        total = sum(int(s.get("count", 0)) for s in seeds)
        return {"version": cat.get("version", CATALOG_VERSION), "seeds": seeds, "total_count": total}

    @app.route("/teacher/seeds/<seed_id>", methods=["GET"])
    def teacher_seed_detail_route(seed_id):
        prompts = _read_seed_prompts(seed_id)
        if prompts is None:
            return {"error": "unknown seed id"}, 404
        return {"id": seed_id, "count": len(prompts), "prompts": prompts}

    @app.route("/teacher/synth/status", methods=["GET"])
    def teacher_synth_status_route():
        job_id = request.args.get("job_id") or ""
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
        if entry is None:
            return {"error": "unknown job"}, 404
        counts = _read_state_counts(entry["output_dir"])
        return {
            "job_id": job_id,
            "running": entry["thread"].is_alive(),
            "completed": counts["completed"],
            "failed": counts["failed"],
            "skipped_dup": counts["skipped_dup"],
            "output_path": counts["output_path"],
        }
