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
import re
import threading
import uuid

from flask import request

from readers import paths as paths_mod
from readers.paths import REPO_ROOT
from runtime import logs as logmod
from runtime import settings as settings_mod
from tools.jsonl_to_bin import jsonl_to_bin

# ------------------------------------------------------------------------------------
# Constants

LOG_SOURCE = "teacher"
JOB_ID_LEN = 12
SYNTH_JOBS_DIR = "synth_jobs"
TEACHER_API_KEY_ENV = "VERITATE_TEACHER_API_KEY"
SAMPLES_FILE = "samples.jsonl"
STATE_FILE = "state.json"
JOB_META_FILE = "meta.json"
SAMPLES_PREVIEW_DEFAULT = 20
SAMPLES_PREVIEW_MAX = 100
TEACHER_PKG = "teacher"
SEEDS_DIR = os.path.join(REPO_ROOT, "veritate_mri", "data", "seeds")
SEED_CATALOG_FILE = "seed_catalog.json"
CATALOG_VERSION = 1
SYNTH_RESPONSE_KEY = "response"
SYNTH_VAL_RATIO = 0.02
STEM_RE = re.compile(r"^[a-z0-9_]+$")

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
    configs = {pid: {"model": cfg.get("model", ""), "base_url": cfg.get("base_url", ""),
                     "has_key": bool(cfg.get("api_key"))}
               for pid, cfg in (s.get("teacher_configs") or {}).items()}
    return {
        "providers": teacher_mod.list_providers(),
        "configured": bool(s.get("teacher_provider")),
        "provider": s.get("teacher_provider", ""),
        "model": s.get("teacher_model", ""),
        "base_url": s.get("teacher_base_url", ""),
        "has_api_key": has_key,
        "configs": configs,
        "max_concurrency": int(s.get("teacher_max_concurrency", 0)),
        "max_tokens": int(s.get("teacher_max_tokens", 0)),
        "temperature": float(s.get("teacher_temperature", 0.0)),
    }


def _stored_key(s, provider):
    key = (s.get("teacher_configs") or {}).get(provider, {}).get("api_key", "")
    if key:
        return key
    if provider and provider == s.get("teacher_provider"):
        return s.get("teacher_api_key", "")
    return ""


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


def _read_job_meta(output_dir):
    path = os.path.join(output_dir, JOB_META_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_job_meta(output_dir, seeds, categories):
    cur = _read_job_meta(output_dir)
    merged = {
        "seeds": sorted(set(cur.get("seeds", [])) | set(seeds or [])),
        "categories": sorted(set(cur.get("categories", [])) | set(categories or [])),
    }
    tmp = os.path.join(output_dir, JOB_META_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f)
    os.replace(tmp, os.path.join(output_dir, JOB_META_FILE))
    return merged


def _read_recent_samples(output_dir, limit):
    path = os.path.join(output_dir, SAMPLES_FILE)
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            rows.append({"id": rec.get("id", ""), "response": rec.get("response", "")})
    return rows[-limit:]


def _read_state_counts(output_dir):
    samples = os.path.join(output_dir, SAMPLES_FILE)
    failed = 0
    skipped = 0
    state_path = os.path.join(output_dir, STATE_FILE)
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            failed = int(st.get("failed", 0))
            skipped = int(st.get("skipped_dup", 0))
        except (OSError, ValueError):
            pass
    return {
        "completed": _count_lines(samples),
        "failed": failed,
        "skipped_dup": skipped,
        "output_path": samples,
    }


def register(app):
    teacher_mod, synth_mod, test_mod = _teacher_mods()

    @app.route("/teacher", methods=["GET", "POST"])
    def teacher_route():
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            # Per-provider memory: a typed key wins, else the provider's stored
            # key. Switching providers never carries another provider's key.
            prov = (body.get("teacher_provider") or "").strip()
            if prov:
                s = settings_mod.get()
                key = (body.get("teacher_api_key") or "").strip() or _stored_key(s, prov)
                body["teacher_api_key"] = key
                cfgs = dict(s.get("teacher_configs") or {})
                cfgs[prov] = {"api_key": key,
                              "model": (body.get("teacher_model") or "").strip(),
                              "base_url": (body.get("teacher_base_url") or "").strip()}
                body["teacher_configs"] = cfgs
            try:
                settings_mod.update(body)
            except ValueError as ve:
                logmod.warn(LOG_SOURCE, f"config rejected: {ve}")
                return {"error": str(ve)}, 400
            view = _public_view(settings_mod.get(), teacher_mod)
            logmod.info(LOG_SOURCE, f"config saved: provider={view['provider'] or '(none)'} "
                                    f"model={view['model'] or '(default)'} "
                                    f"base_url={view['base_url'] or '(default)'} has_key={view['has_api_key']}")
            return view
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
                   or _stored_key(s, provider)
                   or None)
        if not provider:
            return {"error": "provider required"}, 400
        logmod.info(LOG_SOURCE, f"test connection: provider={provider} "
                                f"model={model or '(default)'} base_url={base_url or '(default)'}")
        res = test_mod.test(provider, model=model, base_url=base_url, api_key=api_key)
        if res.get("ok"):
            logmod.ok(LOG_SOURCE, f"test ok: {provider}/{res.get('model') or model or '(default)'} "
                                  f"{res.get('latency_ms', 0)}ms")
        else:
            logmod.warn(LOG_SOURCE, f"test failed: {provider} -> {res.get('error')}")
        return res

    @app.route("/teacher/models", methods=["POST"])
    def teacher_models_route():
        body = request.get_json(silent=True) or {}
        s = settings_mod.get()
        provider = body.get("provider") or s.get("teacher_provider") or ""
        base_url = body.get("base_url") or s.get("teacher_base_url") or None
        api_key = (body.get("api_key")
                   or os.environ.get(TEACHER_API_KEY_ENV)
                   or _stored_key(s, provider)
                   or None)
        if not provider:
            return {"error": "provider required"}, 400
        return {"models": test_mod.list_models(provider, base_url=base_url, api_key=api_key)}

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
        seed_ids = body.get("seed_ids") or []
        categories = sorted({p.get("category") for p in prompts
                             if isinstance(p, dict) and p.get("category")})
        existing_id = (body.get("job_id") or "").strip()
        if existing_id:
            with _JOBS_LOCK:
                entry = _JOBS.get(existing_id)
            if entry is not None and entry["thread"].is_alive():
                return {"error": "job still running"}, 409
            job_id = existing_id
            out_root = os.path.join(REPO_ROOT, SYNTH_JOBS_DIR, job_id)
        else:
            job_id = uuid.uuid4().hex[:JOB_ID_LEN]
            out_root = body.get("output_dir") or os.path.join(REPO_ROOT, SYNTH_JOBS_DIR, job_id)
        os.makedirs(out_root, exist_ok=True)
        _write_job_meta(out_root, seed_ids, categories)
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
        logmod.info(LOG_SOURCE, f"synth started: job={job_id} prompts={len(prompts)} "
                                f"provider={provider} model={model or '(default)'}")
        return {"job_id": job_id, "output_dir": out_root}

    @app.route("/teacher/synth/jobs", methods=["GET"])
    def teacher_synth_jobs_route():
        root = os.path.join(REPO_ROOT, SYNTH_JOBS_DIR)
        out = []
        if os.path.isdir(root):
            with _JOBS_LOCK:
                running_ids = {jid for jid, e in _JOBS.items() if e["thread"].is_alive()}
            for jid in sorted(os.listdir(root)):
                d = os.path.join(root, jid)
                if not os.path.isdir(d):
                    continue
                counts = _read_state_counts(d)
                meta = _read_job_meta(d)
                out.append({"job_id": jid, "completed": counts["completed"],
                            "categories": meta.get("categories", []),
                            "seeds": meta.get("seeds", []),
                            "running": jid in running_ids})
        return {"jobs": out}

    @app.route("/teacher/synth/build_corpus", methods=["POST"])
    def teacher_synth_build_corpus_route():
        body = request.get_json(silent=True) or {}
        job_id = body.get("job_id") or ""
        stem = (body.get("stem") or "").strip().lower()
        if not stem or not STEM_RE.match(stem):
            return {"error": "stem must be lowercase letters, digits, underscores"}, 400
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
        output_dir = entry["output_dir"] if entry else os.path.join(REPO_ROOT, SYNTH_JOBS_DIR, job_id)
        samples = os.path.join(output_dir, SAMPLES_FILE)
        if not os.path.isfile(samples):
            return {"error": "no samples for job"}, 404
        train_bin = paths_mod.corpus_train_path(stem)
        val_bin = paths_mod.corpus_val_path(stem)
        try:
            stats = jsonl_to_bin(samples, train_bin, trace_key=SYNTH_RESPONSE_KEY,
                                 val_split_ratio=SYNTH_VAL_RATIO, val_bin_path=val_bin)
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            return {"error": str(e)}, 400
        logmod.ok(LOG_SOURCE, f"corpus built: stem={stem} records={stats['n_records']} "
                              f"train={stats['train_bytes']}B val={stats['val_bytes']}B")
        return {"stem": stem, "train_bin": train_bin, "val_bin": val_bin,
                "n_records": stats["n_records"], "n_train": stats["n_train"], "n_val": stats["n_val"]}

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

    @app.route("/teacher/synth/stop", methods=["POST"])
    def teacher_synth_stop_route():
        body = request.get_json(silent=True) or {}
        job_id = body.get("job_id") or ""
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
        if entry is None:
            return {"error": "unknown job"}, 404
        entry["job"].stop()
        logmod.info(LOG_SOURCE, f"synth stop requested: job={job_id}")
        return {"job_id": job_id, "stopping": True}

    @app.route("/teacher/synth/samples", methods=["GET"])
    def teacher_synth_samples_route():
        job_id = request.args.get("job_id") or ""
        try:
            limit = int(request.args.get("limit") or SAMPLES_PREVIEW_DEFAULT)
        except ValueError:
            limit = SAMPLES_PREVIEW_DEFAULT
        limit = max(1, min(limit, SAMPLES_PREVIEW_MAX))
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
        output_dir = entry["output_dir"] if entry else os.path.join(REPO_ROOT, SYNTH_JOBS_DIR, job_id)
        if not job_id or not os.path.isdir(output_dir):
            return {"error": "unknown job"}, 404
        return {"job_id": job_id, "samples": _read_recent_samples(output_dir, limit)}

    @app.route("/teacher/synth/status", methods=["GET"])
    def teacher_synth_status_route():
        job_id = request.args.get("job_id") or ""
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
        if entry is not None:
            output_dir = entry["output_dir"]
            running = entry["thread"].is_alive()
        else:
            output_dir = os.path.join(REPO_ROOT, SYNTH_JOBS_DIR, job_id)
            if not job_id or not os.path.isdir(output_dir):
                return {"error": "unknown job"}, 404
            running = False
        counts = _read_state_counts(output_dir)
        return {
            "job_id": job_id,
            "running": running,
            "completed": counts["completed"],
            "failed": counts["failed"],
            "skipped_dup": counts["skipped_dup"],
            "output_path": counts["output_path"],
        }
