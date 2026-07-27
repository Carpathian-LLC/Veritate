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
# - /teacher/authoring/* runs the self-authored corpus pipeline: spec read/write,
#   start (plans calls from a byte target and gates every record), build (packs
#   ChatML bins, zips them, registers a coming_soon catalog entry). Status, stop,
#   samples, and delete are shared with /teacher/synth/*.
# veritate_mri/routes/teacher_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import importlib
import json
import os
import re
import shutil
import threading
import uuid
import zipfile

from flask import request

from readers import paths as paths_mod
from readers.paths import REPO_ROOT
from runtime import logs as logmod
from runtime import settings as settings_mod
from teacher import authoring as authoring_mod
from tools.build_sft_corpus import build as build_sft_bins
from tools.jsonl_to_bin import jsonl_to_bin
from training.sync.corpus_sync import LOCAL_CATALOG_PATH

# ------------------------------------------------------------------------------------
# Constants

LOG_SOURCE = "teacher"
JOB_ID_LEN = 12
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

AUTHORING_ID_PREFIX = "auth_"
AUTHORING_FAMILIES_DIR = "families"
AUTHORING_DIST_DIR = "dist"
AUTHORING_PURPOSE = "Self-authored Veritate chat and reading corpus."
AUTHORING_LICENSE = "# Veritate Authored Corpus\n\nSelf-authored. For use with Veritate models only.\n"
AUTHORING_FAMILY = "carpathian"
AUTHORING_TOPIC = "chat"
AUTHORING_FORMAT = "zip_bundle"
AUTHORING_TRAINED_MODES = ["chat"]
CATALOG_PLACEHOLDER_URL = "https://api.carpathian.ai/cos/PLACEHOLDER/{stem}.zip"
CATALOG_INDENT = 1
MB = 1024 * 1024
SHA_CHUNK = 1 << 20

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


def _resolve_concurrency(s, provider):
    # Cloud APIs take the user value (or the high default). Local servers are
    # clamped to a safe ceiling so a high global value never floods a single
    # local GPU into an out-of-memory crash, whatever OLLAMA_NUM_PARALLEL is set
    # to. No server-side tuning required.
    providers = importlib.import_module(TEACHER_PKG + ".providers")
    try:
        local = providers.get_provider(provider).get("kind") == "local"
    except ValueError:
        local = False
    saved = int(s.get("teacher_max_concurrency") or 0)
    if local:
        return min(saved, providers.LOCAL_MAX_CONCURRENCY) if saved > 0 else providers.LOCAL_MAX_CONCURRENCY
    return saved if saved > 0 else providers.DEFAULT_MAX_CONCURRENCY


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
            rows.append({"id": rec.get("id", ""),
                         "response": rec.get("response") or _render_record(rec.get(authoring_mod.RECORD_KEY))})
    return rows[-limit:]


def _render_record(rec):
    """Authoring rows carry a record, not a raw response. Show it readably."""
    if not isinstance(rec, dict):
        return ""
    head = f"[{rec.get(authoring_mod.GENRE_KEY, '')} / {rec.get(authoring_mod.VOICE_KEY, '')}]"
    turns = rec.get(authoring_mod.TURNS_KEY)
    if isinstance(turns, list):
        body = "\n".join(f"{t.get(authoring_mod.ROLE_KEY, '')}: {t.get(authoring_mod.TEXT_KEY, '')}"
                         for t in turns)
    else:
        body = rec.get(authoring_mod.TEXT_KEY, "")
    return f"{head}\n{body}"


def _read_state_counts(output_dir):
    samples = os.path.join(output_dir, SAMPLES_FILE)
    failed = 0
    skipped = 0
    last_error = ""
    error_summary = {}
    aborted = False
    authoring = {}
    state_path = os.path.join(output_dir, STATE_FILE)
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            failed = int(st.get("failed", 0))
            skipped = int(st.get("skipped_dup", 0))
            last_error = st.get("last_error", "") or ""
            error_summary = st.get("error_summary", {}) or {}
            aborted = bool(st.get("aborted", False))
            authoring = st.get("authoring", {}) or {}
        except (OSError, ValueError):
            pass
    return {
        "completed": _count_lines(samples),
        "failed": failed,
        "skipped_dup": skipped,
        "last_error": last_error,
        "error_summary": error_summary,
        "aborted": aborted,
        "authoring": authoring,
        "output_path": samples,
    }


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(SHA_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_families(output_dir, spec):
    """Split gated authoring records into per-genre jsonl the ChatML packer eats."""
    fam_dir = os.path.join(output_dir, AUTHORING_FAMILIES_DIR)
    if os.path.isdir(fam_dir):
        shutil.rmtree(fam_dir)
    os.makedirs(fam_dir, exist_ok=True)
    handles, counts = {}, {}
    with open(os.path.join(output_dir, SAMPLES_FILE), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line).get(authoring_mod.RECORD_KEY)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            genre = authoring_mod.genre_by_id(spec, rec.get(authoring_mod.GENRE_KEY, ""))
            if genre is None:
                continue
            gid = genre["id"]
            if gid not in handles:
                handles[gid] = open(os.path.join(fam_dir, f"{gid}.jsonl"), "w", encoding="utf-8")
                counts[gid] = 0
            handles[gid].write(json.dumps(_to_family_row(rec, genre["schema"]),
                                          ensure_ascii=False) + "\n")
            counts[gid] += 1
    for fp in handles.values():
        fp.close()
    return fam_dir, counts


def _to_family_row(rec, schema):
    if schema == authoring_mod.SCHEMA_TURNS:
        return {"turns": [{"role": t[authoring_mod.ROLE_KEY], "content": t[authoring_mod.TEXT_KEY]}
                          for t in rec[authoring_mod.TURNS_KEY]]}
    return {"text": rec[authoring_mod.TEXT_KEY]}


def _zip_bins(dist_dir, stem):
    zip_path = os.path.join(dist_dir, f"{stem}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for suffix in ("_train.bin", "_val.bin"):
            name = f"{stem}{suffix}"
            z.write(os.path.join(dist_dir, name), arcname=name)
    return zip_path


def _register_catalog_entry(stem, label, description, manifest, min_params, max_params):
    """Append a coming_soon zip_bundle entry so the corpus shows in the library."""
    with open(LOCAL_CATALOG_PATH, "r", encoding="utf-8") as f:
        cat = json.load(f)
    entry = {
        "stem": stem,
        "label": label,
        "family": AUTHORING_FAMILY,
        "topic": AUTHORING_TOPIC,
        "description": description,
        "format": AUTHORING_FORMAT,
        "train_url": CATALOG_PLACEHOLDER_URL.format(stem=stem),
        "size_train": manifest["train_bytes"],
        "size_val": manifest["val_bytes"],
        "sha256_train": manifest["train_sha256"],
        "sha256_val": manifest["val_sha256"],
        "trained_modes": list(AUTHORING_TRAINED_MODES),
        "recommended_min_params": min_params,
        "recommended_max_params": max_params,
        "coming_soon": True,
    }
    cat["corpora"] = [c for c in cat.get("corpora", []) if c.get("stem") != stem] + [entry]
    tmp = LOCAL_CATALOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cat, f, indent=CATALOG_INDENT, ensure_ascii=False)
    os.replace(tmp, LOCAL_CATALOG_PATH)
    return entry


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

    @app.route("/teacher/complete", methods=["POST"])
    def teacher_complete_route():
        # One-shot completion from a user-added model: body {prompt, system?, provider?,
        # model?, base_url?, api_key?, max_tokens?, temperature?}. Defaults to the configured
        # teacher. The programmatic surface extensions call to score text with the user's model.
        body = request.get_json(silent=True) or {}
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return {"ok": False, "error": "prompt required"}, 400
        s = settings_mod.get()
        provider = body.get("provider") or s.get("teacher_provider") or ""
        model = body.get("model") or s.get("teacher_model") or None
        base_url = body.get("base_url") or s.get("teacher_base_url") or None
        api_key = (body.get("api_key") or os.environ.get(TEACHER_API_KEY_ENV)
                   or _stored_key(s, provider) or None)
        if not provider:
            return {"ok": False, "error": "no teacher configured"}, 400
        opts = {"base_url": base_url, "api_key": api_key,
                "max_tokens": int(body.get("max_tokens") or s.get("teacher_max_tokens", 2048)),
                "temperature": float(body.get("temperature", s.get("teacher_temperature", 0.7)))}
        if body.get("system"):
            opts["system"] = body["system"]
        try:
            text = teacher_mod.complete(provider, model, [{"role": "user", "content": prompt}], **opts)
        except Exception as e:
            logmod.warn(LOG_SOURCE, f"complete failed: {provider}/{model or '(default)'} -> {type(e).__name__}: {e}")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}, 502
        return {"ok": True, "text": text, "provider": provider, "model": model or s.get("teacher_model", "")}

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
            out_root = paths_mod.synth_job_dir(job_id)
        else:
            job_id = uuid.uuid4().hex[:JOB_ID_LEN]
            out_root = body.get("output_dir") or paths_mod.synth_job_dir(job_id)
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
            max_concurrency=_resolve_concurrency(s, provider),
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
        root = paths_mod.synth_jobs_root()
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

    @app.route("/teacher/synth/delete", methods=["POST"])
    def teacher_synth_delete_route():
        body = request.get_json(silent=True) or {}
        job_id = (body.get("job_id") or "").strip()
        if not job_id:
            return {"error": "job_id required"}, 400
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
            if entry is not None and entry["thread"].is_alive():
                return {"error": "job still running"}, 409
        root = os.path.realpath(paths_mod.synth_jobs_root())
        target = os.path.realpath(os.path.join(root, job_id))
        if os.path.dirname(target) != root or not os.path.isdir(target):
            return {"error": "unknown job"}, 404
        shutil.rmtree(target)
        with _JOBS_LOCK:
            _JOBS.pop(job_id, None)
        logmod.info(LOG_SOURCE, f"synth job deleted: job={job_id}")
        return {"job_id": job_id, "deleted": True}

    @app.route("/teacher/synth/build_corpus", methods=["POST"])
    def teacher_synth_build_corpus_route():
        body = request.get_json(silent=True) or {}
        job_id = body.get("job_id") or ""
        stem = (body.get("stem") or "").strip().lower()
        if not stem or not STEM_RE.match(stem):
            return {"error": "stem must be lowercase letters, digits, underscores"}, 400
        with _JOBS_LOCK:
            entry = _JOBS.get(job_id)
        output_dir = entry["output_dir"] if entry else paths_mod.synth_job_dir(job_id)
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

    @app.route("/teacher/authoring/spec", methods=["GET", "POST"])
    def teacher_authoring_spec_route():
        if request.method == "POST":
            body = request.get_json(silent=True)
            if not isinstance(body, dict) or not body.get("genres") or not body.get("gates"):
                return {"error": "spec must be an object with genres and gates"}, 400
            authoring_mod.save_spec(body)
            logmod.info(LOG_SOURCE, f"authoring spec saved: {len(body['genres'])} genres")
            return authoring_mod.load_spec()
        return authoring_mod.load_spec()

    @app.route("/teacher/authoring/start", methods=["POST"])
    def teacher_authoring_start_route():
        body = request.get_json(silent=True) or {}
        spec = authoring_mod.load_spec()
        genre_ids = [g for g in (body.get("genres") or [])
                     if authoring_mod.genre_by_id(spec, g) is not None]
        if not genre_ids:
            return {"error": "pick at least one genre"}, 400
        s = settings_mod.get()
        provider = s.get("teacher_provider") or ""
        if not provider:
            return {"error": "teacher_provider not configured"}, 400
        target_mb = float(body.get("target_mb") or 0)
        if target_mb <= 0:
            return {"error": "target_mb must be greater than zero"}, 400
        floor = body.get("ngram_distinct_floor")
        if floor is not None:
            spec["gates"]["ngram_distinct_floor"] = float(floor)
        calls = authoring_mod.plan_calls(spec, genre_ids, int(target_mb * MB))
        existing_id = (body.get("job_id") or "").strip()
        if existing_id:
            with _JOBS_LOCK:
                entry = _JOBS.get(existing_id)
            if entry is not None and entry["thread"].is_alive():
                return {"error": "job still running"}, 409
            job_id = existing_id
        else:
            job_id = uuid.uuid4().hex[:JOB_ID_LEN]
        out_root = paths_mod.synth_job_dir(job_id)
        os.makedirs(out_root, exist_ok=True)
        _write_job_meta(out_root, [], genre_ids)
        prompts = authoring_mod.build_prompts(spec, calls, int(spec["gates"]["build_seed"]),
                                              AUTHORING_ID_PREFIX)
        gate = authoring_mod.RecordGate(spec)
        gate.seed_from_file(os.path.join(out_root, SAMPLES_FILE))
        conc = int(body.get("max_concurrency") or 0) or _resolve_concurrency(s, provider)
        job = synth_mod.SynthJob(
            job_id, provider, s.get("teacher_model") or None, prompts, out_root,
            base_url=s.get("teacher_base_url") or None,
            api_key=os.environ.get(TEACHER_API_KEY_ENV) or s.get("teacher_api_key") or None,
            temperature=float(s.get("teacher_temperature", 0.7)),
            max_tokens=int(s.get("teacher_max_tokens", 2048)),
            max_concurrency=conc,
            record_gate=gate,
        )
        thread = threading.Thread(target=job.run, name=f"teacher-authoring-{job_id}", daemon=True)
        with _JOBS_LOCK:
            _JOBS[job_id] = {"job": job, "thread": thread, "output_dir": out_root}
        thread.start()
        logmod.info(LOG_SOURCE, f"authoring started: job={job_id} genres={','.join(genre_ids)} "
                                f"calls={len(prompts)} concurrency={conc}")
        return {"job_id": job_id, "output_dir": out_root, "calls": calls,
                "total_calls": len(prompts), "max_concurrency": conc}

    @app.route("/teacher/authoring/build", methods=["POST"])
    def teacher_authoring_build_route():
        body = request.get_json(silent=True) or {}
        job_id = (body.get("job_id") or "").strip()
        stem = (body.get("stem") or "").strip().lower()
        if not stem or not STEM_RE.match(stem):
            return {"error": "stem must be lowercase letters, digits, underscores"}, 400
        output_dir = paths_mod.synth_job_dir(job_id)
        if not os.path.isfile(os.path.join(output_dir, SAMPLES_FILE)):
            return {"error": "no samples for job"}, 404
        spec = authoring_mod.load_spec()
        fam_dir, counts = _write_families(output_dir, spec)
        if not counts:
            return {"error": "no authored records in job"}, 400
        dist_dir = os.path.join(output_dir, AUTHORING_DIST_DIR)
        manifest = build_sft_bins(
            stem, fam_dir, dist_dir, sorted(f"{g}.jsonl" for g in counts),
            AUTHORING_PURPOSE, AUTHORING_LICENSE,
            int(spec["gates"]["build_seed"]), float(spec["gates"]["build_val_ratio"]),
            corpus_dir=paths_mod.corpus_dir())
        zip_path = _zip_bins(dist_dir, stem)
        entry = _register_catalog_entry(
            stem, body.get("label") or stem, body.get("description") or AUTHORING_PURPOSE,
            manifest, int(body.get("recommended_min_params") or 0),
            body.get("recommended_max_params"))
        logmod.ok(LOG_SOURCE, f"authored corpus built: stem={stem} "
                              f"train={manifest['train_bytes']}B val={manifest['val_bytes']}B "
                              f"zip={zip_path}")
        return {
            "stem": stem, "zip_path": zip_path, "zip_bytes": os.path.getsize(zip_path),
            "zip_sha256": _sha256_file(zip_path),
            "family_counts": counts, "manifest": manifest, "catalog_entry": entry,
            "next_steps": [
                f"Upload {zip_path} to COS. The zip holds {stem}_train.bin and "
                f"{stem}_val.bin at the top level.",
                f"COS returns a link. Paste it into corpus_catalog.json as the train_url "
                f"for stem '{stem}', replacing the PLACEHOLDER value.",
                f"Remove \"coming_soon\": true from that same entry to release the corpus "
                f"in the corpus library.",
            ],
        }

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
        output_dir = entry["output_dir"] if entry else paths_mod.synth_job_dir(job_id)
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
            output_dir = paths_mod.synth_job_dir(job_id)
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
            "last_error": counts["last_error"],
            "error_summary": counts["error_summary"],
            "aborted": counts["aborted"],
            "authoring": counts["authoring"],
            "output_path": counts["output_path"],
        }
