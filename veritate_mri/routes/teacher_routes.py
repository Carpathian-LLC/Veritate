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
#   import (gates an externally-authored jsonl directory into a job the same way
#   as a teacher call), start (plans calls from a byte target and gates every
#   record), build (packs ChatML bins, zips them, registers a coming_soon catalog
#   entry). Status, stop, samples, and delete are shared with /teacher/synth/*.
# veritate_mri/routes/teacher_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import collections
import contextlib
import hashlib
import importlib
import json
import os
import re
import shutil
import socket
import threading
import urllib.parse
import uuid
import zipfile
from collections import Counter

from flask import request
from readers import paths as paths_mod
from readers import seeds as seeds_mod
from runtime import logs as logmod
from runtime import settings as settings_mod
from teacher import authoring as authoring_mod
from teacher import interview_job as interview_mod
from tools.build_sft_corpus import build as build_sft_bins
from tools.corpus_audit import audit_file
from tools.jsonl_to_bin import jsonl_to_bin
from training import trainer_runner
from training.sync.corpus_sync import LOCAL_CATALOG_PATH

from ._common import safe_route as _safe

# ------------------------------------------------------------------------------------
# Constants

LOG_SOURCE = "teacher"
JOB_ID_LEN = 12
TEACHER_API_KEY_ENV = "VERITATE_TEACHER_API_KEY"
SAMPLES_FILE = "samples.jsonl"
STATE_FILE = "state.json"
PLAN_FILE = "plan.json"
JOB_META_FILE = "meta.json"
JOB_LABEL_MAX = 80
LINE_COUNT_CACHE_MAX = 256
SAMPLES_PREVIEW_DEFAULT = 20
SAMPLES_PREVIEW_MAX = 100
# Browsing a finished corpus is a different job from previewing a running one:
# it pages from an offset instead of tailing, and it must stay memory-bounded on
# a 100 MB samples.jsonl, so a page is streamed line by line and never collected
# whole. One runaway record cannot blow up the page either.
BROWSE_PAGE_DEFAULT = 25
BROWSE_PAGE_MAX = 100
BROWSE_TEXT_MAX = 8000
TEACHER_PKG = "teacher"
SEEDS_DIRNAME = "seeds"
SEEDS_DIR = os.path.join(paths_mod.DATA_ROOT, SEEDS_DIRNAME)
SEED_CATALOG_FILE = "seed_catalog.json"
CATALOG_VERSION = 1
SYNTH_RESPONSE_KEY = "response"
SYNTH_VAL_RATIO = 0.02
STEM_RE = re.compile(r"^[a-z0-9_]+$")
PROVIDER_KIND_LOCAL = "local"
INTERVIEW_MAX_DEPTH = 20
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"})
CONTENTION_NONE = "none"
CONTENTION_LOCAL_RUN = "local_run"


AUTHORING_ID_PREFIX = "auth_"
IMPORT_ID_PREFIX = "import_"
IMPORT_STEM_RE = re.compile(r"[^a-z0-9]+")
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

# Sampling knobs the synth/complete paths send to the teacher. Fallbacks come
# from settings.DEFAULTS, so changing a default in one place actually changes
# what these routes send.
SETTING_MAX_TOKENS  = "teacher_max_tokens"
SETTING_TEMPERATURE = "teacher_temperature"

_JOBS = {}
_JOBS_LOCK = threading.Lock()
_LINE_COUNTS = {}
_LINE_COUNTS_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions

def _setting(s, key):
    return s.get(key, settings_mod.DEFAULTS[key])


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
    """Non-blank lines in an append-only file, counted incrementally.

    The Distillation tab polls a running job every couple of seconds and the
    corpora list counts every job on disk, so a full re-read would scale with
    corpus size on every poll. The cache is keyed by (size, mtime_ns); a file
    that only grew is resumed from the byte offset the previous count covered,
    and that resume is taken only when the offset lands on a record boundary."""
    try:
        st = os.stat(path)
    except OSError:
        with _LINE_COUNTS_LOCK:
            _LINE_COUNTS.pop(path, None)
        return 0
    with _LINE_COUNTS_LOCK:
        prev = _LINE_COUNTS.get(path)
    if prev is not None and prev[0] == st.st_size and prev[1] == st.st_mtime_ns:
        return prev[2]
    offset, count = 0, 0
    if prev is not None and prev[0] <= st.st_size and _ends_on_record_boundary(path, prev[0]):
        offset, count = prev[0], prev[2]
    with open(path, "rb") as f:
        f.seek(offset)
        count += sum(1 for line in f if line.strip())
    with _LINE_COUNTS_LOCK:
        if len(_LINE_COUNTS) >= LINE_COUNT_CACHE_MAX:
            _LINE_COUNTS.clear()
        _LINE_COUNTS[path] = (st.st_size, st.st_mtime_ns, count)
    return count


def _ends_on_record_boundary(path, offset):
    """True when byte `offset` starts a fresh line, so a resumed count is exact.

    A writer caught mid-record would otherwise have its half-line counted twice:
    once as a tail here, once as a head on the next pass."""
    if offset == 0:
        return True
    with open(path, "rb") as f:
        f.seek(offset - 1)
        return f.read(1) == b"\n"


def _load_seed_catalog():
    path = os.path.join(SEEDS_DIR, SEED_CATALOG_FILE)
    if not os.path.isfile(path):
        return {"version": CATALOG_VERSION, "seeds": []}
    with open(path, encoding="utf-8") as f:
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
    with open(path, encoding="utf-8") as f:
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
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_job_meta(output_dir, seeds, categories, label=None):
    cur = _read_job_meta(output_dir)
    merged = dict(cur)
    merged["seeds"] = sorted(set(cur.get("seeds", [])) | set(seeds or []))
    merged["categories"] = sorted(set(cur.get("categories", [])) | set(categories or []))
    # A name given at start time is the same field `/teacher/synth/rename`
    # writes, so a corpus can be named once and never carry a hex id. Blank
    # leaves whatever name it already had alone: appending to a named corpus
    # must not silently strip its name.
    label = (label or "").strip()[:JOB_LABEL_MAX]
    if label:
        merged["label"] = label
    return _save_job_meta(output_dir, merged)


def _save_job_meta(output_dir, meta):
    tmp = os.path.join(output_dir, JOB_META_FILE) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(tmp, os.path.join(output_dir, JOB_META_FILE))
    return meta


def _job_dir(job_id):
    """Realpath of a job directory, or None when job_id escapes the jobs root."""
    root = os.path.realpath(paths_mod.synth_jobs_root())
    target = os.path.realpath(os.path.join(root, job_id))
    if os.path.dirname(target) != root or not os.path.isdir(target):
        return None
    return target


def _path_size(path):
    return os.path.getsize(path) if os.path.isfile(path) else 0


def _path_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def _read_recent_samples(output_dir, limit):
    """The last `limit` records, without parsing the ones in front of them.

    The live panel polls this every couple of seconds for six conversations. The
    file it is tailing reaches 100 MB, so decoding every line to throw all but
    the last six away burned the whole corpus through json.loads on every tick.
    A deque keeps only the raw tail; parsing happens once, on those lines."""
    path = os.path.join(output_dir, SAMPLES_FILE)
    if not os.path.isfile(path):
        return []
    tail = collections.deque(maxlen=max(1, limit))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tail.append(line)
    rows = []
    for line in tail:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        # `response` is the flat text the raw-synth and authoring panels print.
        # The live interview transcript needs the turns themselves -- it was
        # JSON.parsing this rendered string and silently getting nothing -- so
        # the structured form rides along beside it.
        row = _browse_row(rec)
        row["response"] = rec.get("response") or _render_record(rec.get(authoring_mod.RECORD_KEY))
        rows.append(row)
    return rows


def _browse_row(rec):
    """One samples.jsonl line as structured data the viewer can render.

    Structured rather than pre-joined text: every corpus on this install is
    conversational, and the viewer styles user and assistant turns differently.
    Raw-synth rows carry a flat `response` instead and fall back to `text`."""
    if not isinstance(rec, dict):
        return {"id": "", "text": ""}
    out = {"id": str(rec.get("id", ""))}
    body = rec.get(authoring_mod.RECORD_KEY)
    if isinstance(body, dict):
        out["genre"] = body.get(authoring_mod.GENRE_KEY, "")
        out["voice"] = body.get(authoring_mod.VOICE_KEY, "")
        turns = body.get(authoring_mod.TURNS_KEY)
        if isinstance(turns, list):
            out["turns"] = [{"role": t.get(authoring_mod.ROLE_KEY, ""),
                             "text": str(t.get(authoring_mod.TEXT_KEY, ""))[:BROWSE_TEXT_MAX]}
                            for t in turns if isinstance(t, dict)]
        else:
            out["text"] = str(body.get(authoring_mod.TEXT_KEY, ""))[:BROWSE_TEXT_MAX]
    else:
        out["text"] = str(rec.get("response") or "")[:BROWSE_TEXT_MAX]
    return out


def _browse_samples(output_dir, offset, limit):
    """A page of records starting at `offset`, counted over non-blank lines.

    Streams and stops as soon as the page is full: reading the whole file to
    slice 25 rows out of it would pull 100 MB into memory per request."""
    path = os.path.join(output_dir, SAMPLES_FILE)
    if not os.path.isfile(path):
        return []
    rows = []
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if seen >= offset + limit:
                break
            if seen >= offset:
                try:
                    rows.append(_browse_row(json.loads(line)))
                except ValueError:
                    rows.append({"id": "", "text": "(unreadable record)"})
            seen += 1
    return rows


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
    shortfall = {}
    calls_ok = 0
    calls_remaining = None
    skipped = 0
    last_error = ""
    error_summary = {}
    aborted = False
    authoring = {}
    state_path = os.path.join(output_dir, STATE_FILE)
    if os.path.isfile(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                st = json.load(f)
            failed = int(st.get("failed", 0))
            calls_ok = int(st.get("completed", 0))
            calls_remaining = len(st.get("remaining") or [])
            skipped = int(st.get("skipped_dup", 0))
            last_error = st.get("last_error", "") or ""
            error_summary = st.get("error_summary", {}) or {}
            aborted = bool(st.get("aborted", False))
            shortfall = st.get("opener_shortfall", {}) or {}
            authoring = st.get("authoring", {}) or {}
        except (OSError, ValueError):
            pass
    return {
        "completed": _count_lines(samples),   # RECORDS kept, not calls
        "calls_ok": calls_ok,
        "calls_failed": failed,
        "calls_remaining": calls_remaining,
        "failed": failed,
        "skipped_dup": skipped,
        "last_error": last_error,
        "error_summary": error_summary,
        "aborted": aborted,
        "authoring": authoring,
        "opener_shortfall": shortfall,
        "plan": _read_plan(output_dir),
        "output_path": samples,
    }


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(SHA_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_stem(filename):
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    return IMPORT_STEM_RE.sub("_", stem).strip("_")


def _import_file(path, stem, spec, gate, out_f):
    """Gate every bare record in one externally-authored jsonl file, append accepted
    rows to samples.jsonl in {"id", "record"} shape, and count outcomes by reason."""
    accepted = 0
    rejects = Counter()
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                rejects[authoring_mod.REJECT_JSON] += 1
                continue
            if not isinstance(rec, dict):
                rejects[authoring_mod.REJECT_SCHEMA] += 1
                continue
            genre_id = rec.get(authoring_mod.GENRE_KEY)
            genre = authoring_mod.genre_by_id(spec, genre_id) if genre_id else None
            if genre is None:
                rejects[f"{authoring_mod.REJECT_UNKNOWN_GENRE}: {genre_id}"] += 1
                continue
            kept, why = gate({"genre": genre_id}, json.dumps(rec, ensure_ascii=False))
            if not kept:
                for reason in why:
                    rejects[reason] += 1
                continue
            rid = f"{IMPORT_ID_PREFIX}{stem}_{line_no:06d}"
            out_f.write(json.dumps({"id": rid, authoring_mod.RECORD_KEY: kept[0]},
                                   ensure_ascii=False) + "\n")
            accepted += 1
    return accepted, dict(rejects.most_common())


def _write_families(output_dir, spec):
    """Split gated authoring records into per-genre jsonl the ChatML packer eats."""
    fam_dir = os.path.join(output_dir, AUTHORING_FAMILIES_DIR)
    if os.path.isdir(fam_dir):
        shutil.rmtree(fam_dir)
    os.makedirs(fam_dir, exist_ok=True)
    handles, counts = {}, {}
    with contextlib.ExitStack() as open_handles, \
         open(os.path.join(output_dir, SAMPLES_FILE), encoding="utf-8") as f:
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
                handles[gid] = open_handles.enter_context(
                    open(os.path.join(fam_dir, f"{gid}.jsonl"), "w", encoding="utf-8"))
                counts[gid] = 0
            handles[gid].write(json.dumps(_to_family_row(rec, genre["schema"]),
                                          ensure_ascii=False) + "\n")
            counts[gid] += 1
    return fam_dir, counts


def _to_family_row(rec, schema):
    if schema == authoring_mod.SCHEMA_TURNS:
        return {"turns": [{"role": t[authoring_mod.ROLE_KEY], "content": t[authoring_mod.TEXT_KEY]}
                          for t in rec[authoring_mod.TURNS_KEY]]}
    return {"text": rec[authoring_mod.TEXT_KEY]}


def _zip_bins(dist_dir, stem):
    zip_path = os.path.join(dist_dir, f"{stem}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for suffix in (paths_mod.CORPUS_TRAIN_SUFFIX, paths_mod.CORPUS_VAL_SUFFIX):
            name = f"{stem}{suffix}"
            z.write(os.path.join(dist_dir, name), arcname=name)
    return zip_path


def _register_catalog_entry(stem, label, description, manifest, min_params, max_params):
    """Append a coming_soon zip_bundle entry so the corpus shows in the library.

    min_params/max_params are accepted but written as-is and are expected to be
    None: size ladders were retired 2026-08-20, so a corpus is matched to a model
    by what it contains, not by how many bytes it is."""
    with open(LOCAL_CATALOG_PATH, encoding="utf-8") as f:
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


def _write_plan(output_dir, plan):
    """Persist the run's plan (target bytes, planned calls, genres) beside the job.

    The job's own state.json only ever counts what has happened. Without the
    denominator written down, a dashboard refresh loses "of how many?" and the
    progress bar silently becomes a spinner."""
    try:
        with open(os.path.join(output_dir, PLAN_FILE), "w", encoding="utf-8") as f:
            json.dump(plan, f)
    except OSError as e:
        logmod.warn(LOG_SOURCE, f"could not write plan for {output_dir}: {e}")


def _read_plan(output_dir):
    try:
        with open(os.path.join(output_dir, PLAN_FILE), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def providers_concurrency_choices():
    providers = importlib.import_module(TEACHER_PKG + ".providers")
    return getattr(providers, "CONCURRENCY_CHOICES", (2, 4, 8, 16))


def _audit_or_none(path):
    """Score a freshly built bin. A gate that 500s on a corner case would block
    the build it is meant to advise, so failure degrades to no report."""
    try:
        return audit_file(path)
    except Exception as e:
        logmod.warn(LOG_SOURCE, f"corpus audit skipped for {path}: {type(e).__name__}: {e}")
        return None


def _target_host(base_url):
    """Hostname the teacher's calls land on. Empty base_url means the provider default."""
    if not base_url:
        return ""
    try:
        return (urllib.parse.urlsplit(base_url).hostname or "").lower()
    except ValueError:
        return ""


def _is_this_machine(host):
    """True when `host` resolves to the box this dashboard runs on."""
    if not host:
        return False
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return host.split(".")[0] == socket.gethostname().split(".")[0].lower()
    except OSError:
        return False


def _active_run():
    """The trainer run on THIS box, or None. Never raises: the guard must not be
    the reason a distillation start fails."""
    try:
        st = trainer_runner.state() or {}
    except Exception:
        return None
    if st.get("status") != "running":
        return None
    args = st.get("args") or {}
    return {
        "plugin_id":  st.get("plugin_id"),
        "name":       args.get("name") or args.get("resume") or args.get("base_ckpt") or "",
        "size":       args.get("size") or "",
        "started_at": st.get("started_at"),
    }


def _target_status(s):
    """Where distillation calls will land, and whether a training run is already
    using that machine. `contention` true is what raises the confirm in the UI.

    A cloud provider never contends. A local provider pointed at another box
    (an ssh tunnel to cardinal, say) cannot be inspected from here, so it
    reports training_active=None rather than guessing a safe-looking false."""
    provider = (s.get("teacher_provider") or "").strip()
    model    = (s.get("teacher_model") or "").strip()
    cfg      = (s.get("teacher_configs") or {}).get(provider) or {}
    base_url = (cfg.get("base_url") or s.get("teacher_base_url") or "").strip()

    providers = importlib.import_module(TEACHER_PKG + ".providers")
    try:
        kind = providers.get_provider(provider).get("kind") if provider else ""
    except ValueError:
        kind = ""

    host     = _target_host(base_url)
    is_local = kind == PROVIDER_KIND_LOCAL
    on_this_box = is_local and _is_this_machine(host)

    run = _active_run() if on_this_box else None
    if not is_local:
        training_active, reason = False, "teacher is a hosted API; nothing on this machine to contend with"
    elif not on_this_box:
        training_active, reason = None, f"teacher runs on {host or 'another machine'}; its training state is not visible from here"
    elif run:
        training_active, reason = True, "a training run is using this machine right now"
    else:
        training_active, reason = False, "teacher runs on this machine and no training run is active"

    return {
        "provider":        provider,
        "model":           model,
        "kind":            kind,
        "base_url":        base_url,
        "host":            host,
        "targets_this_machine": on_this_box,
        "training_active": training_active,
        "run":             run,
        "contention":      bool(run),
        "contention_kind": CONTENTION_LOCAL_RUN if run else CONTENTION_NONE,
        "reason":          reason,
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

    @app.route("/teacher/target_status", methods=["GET"])
    def teacher_target_status_route():
        return _safe(LOG_SOURCE, lambda: _target_status(settings_mod.get()))

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
                "max_tokens": int(body.get("max_tokens") or _setting(s, SETTING_MAX_TOKENS)),
                "temperature": float(body.get("temperature", _setting(s, SETTING_TEMPERATURE)))}
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
        _write_job_meta(out_root, seed_ids, categories, body.get("label"))
        api_key = os.environ.get(TEACHER_API_KEY_ENV) or _stored_key(s, provider) or None
        job = synth_mod.SynthJob(
            job_id, provider, model, prompts, out_root,
            format=fmt,
            base_url=s.get("teacher_base_url") or None,
            api_key=api_key,
            temperature=float(_setting(s, SETTING_TEMPERATURE)),
            max_tokens=int(_setting(s, SETTING_MAX_TOKENS)),
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
                meta = _read_job_meta(d)
                samples = os.path.join(d, SAMPLES_FILE)
                out.append({"job_id": jid, "completed": _count_lines(samples),
                            "label": meta.get("label", ""),
                            "categories": meta.get("categories", []),
                            "seeds": meta.get("seeds", []),
                            "bytes": _path_size(samples),
                            "updated_at": _path_mtime(samples) or _path_mtime(d),
                            "running": jid in running_ids})
        return {"jobs": out}

    @app.route("/teacher/synth/rename", methods=["POST"])
    def teacher_synth_rename_route():
        body = request.get_json(silent=True) or {}
        job_id = (body.get("job_id") or "").strip()
        label = (body.get("label") or "").strip()[:JOB_LABEL_MAX]
        if not job_id:
            return {"error": "job_id required"}, 400
        target = _job_dir(job_id)
        if target is None:
            return {"error": "unknown job"}, 404
        meta = _read_job_meta(target)
        meta["label"] = label
        _save_job_meta(target, meta)
        logmod.info(LOG_SOURCE, f"synth job renamed: job={job_id} label={label!r}")
        return {"job_id": job_id, "label": label}

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
        target = _job_dir(job_id)
        if target is None:
            return {"error": "unknown job"}, 404
        shutil.rmtree(target)
        with _JOBS_LOCK:
            _JOBS.pop(job_id, None)
        with _LINE_COUNTS_LOCK:
            _LINE_COUNTS.pop(os.path.join(target, SAMPLES_FILE), None)
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
                "n_records": stats["n_records"], "n_train": stats["n_train"], "n_val": stats["n_val"],
                "audit": _audit_or_none(train_bin)}

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

    @app.route("/teacher/authoring/import", methods=["POST"])
    def teacher_authoring_import_route():
        body = request.get_json(silent=True) or {}
        source_dir = (body.get("source_dir") or "").strip()
        if not source_dir or not os.path.isdir(source_dir):
            return {"error": "source_dir must be an existing directory"}, 400
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
        samples_path = os.path.join(out_root, SAMPLES_FILE)
        _write_job_meta(out_root, [], [], body.get("label"))
        spec = authoring_mod.load_spec()
        gate = authoring_mod.RecordGate(spec)
        gate.seed_from_file(samples_path)
        files = sorted(f for f in os.listdir(source_dir) if f.endswith(".jsonl"))
        results = []
        accepted_total = 0
        with open(samples_path, "a", encoding="utf-8") as out_f:
            for fname in files:
                accepted, rejected = _import_file(os.path.join(source_dir, fname),
                                                  _import_stem(fname), spec, gate, out_f)
                accepted_total += accepted
                results.append({"file": fname, "accepted": accepted, "rejected": rejected})
        stats = gate.stats()
        logmod.info(LOG_SOURCE, f"authoring import: job={job_id} files={len(files)} "
                                f"accepted={accepted_total} ratio={stats['ngram_ratio']}")
        return {"job_id": job_id, "output_dir": out_root, "files": results,
                "accepted_total": accepted_total, "ngram_ratio": stats["ngram_ratio"],
                "ngram_floor": stats["ngram_floor"], "ngram_below_floor": stats["ngram_below_floor"]}

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
        _write_job_meta(out_root, [], genre_ids, body.get("label"))
        prompts = authoring_mod.build_prompts(spec, calls, int(spec["gates"]["build_seed"]),
                                              AUTHORING_ID_PREFIX)
        gate = authoring_mod.RecordGate(spec)
        gate.seed_from_file(os.path.join(out_root, SAMPLES_FILE))
        conc = int(body.get("max_concurrency") or 0) or _resolve_concurrency(s, provider)
        job = synth_mod.SynthJob(
            job_id, provider, s.get("teacher_model") or None, prompts, out_root,
            base_url=s.get("teacher_base_url") or None,
            api_key=os.environ.get(TEACHER_API_KEY_ENV) or _stored_key(s, provider) or None,
            temperature=float(_setting(s, SETTING_TEMPERATURE)),
            max_tokens=int(_setting(s, SETTING_MAX_TOKENS)),
            max_concurrency=conc,
            record_gate=gate,
        )
        prior = _read_plan(out_root)
        _write_plan(out_root, {
            "target_bytes":  int(prior.get("target_bytes") or 0) + int(target_mb * MB),
            "total_calls":   int(prior.get("total_calls") or 0) + len(prompts),
            "genres":        sorted(set(list(prior.get("genres") or []) + genre_ids)),
            "calls_by_genre": calls,
            "max_concurrency": conc,
        })
        thread = threading.Thread(target=job.run, name=f"teacher-authoring-{job_id}", daemon=True)
        with _JOBS_LOCK:
            _JOBS[job_id] = {"job": job, "thread": thread, "output_dir": out_root}
        thread.start()
        logmod.info(LOG_SOURCE, f"authoring started: job={job_id} genres={','.join(genre_ids)} "
                                f"calls={len(prompts)} concurrency={conc}")
        return {"job_id": job_id, "output_dir": out_root, "calls": calls,
                "total_calls": len(prompts), "max_concurrency": conc}

    @app.route("/teacher/seed_packs", methods=["GET"])
    def teacher_seed_packs_route():
        """Seed packs and their topic groups. Verticals without a pack on disk come
        back available=False so the dashboard can show the roadmap greyed out."""
        return _safe(LOG_SOURCE, lambda: {
            "packs": seeds_mod.list_packs(),
            "concurrency_choices": list(providers_concurrency_choices()),
        })

    @app.route("/teacher/interview/start", methods=["POST"])
    def teacher_interview_start_route():
        """Two-pass generation: the teacher is ASKED questions rather than told to
        script a dialogue. Registers into the same job table as authoring, so
        status / stop / samples / build all work on it unchanged."""
        body = request.get_json(silent=True) or {}
        spec = authoring_mod.load_spec()
        genre_ids = [g for g in (body.get("genres") or ["conversation"])
                     if authoring_mod.genre_by_id(spec, g) is not None]
        if not genre_ids:
            return {"error": "pick at least one genre"}, 400
        s = settings_mod.get()
        provider = s.get("teacher_provider") or ""
        if not provider:
            return {"error": "teacher_provider not configured"}, 400
        n_convs = int(body.get("conversations") or 0)
        if n_convs <= 0:
            return {"error": "conversations must be greater than zero"}, 400
        depth = int(body.get("depth") or 3)
        if not 1 <= depth <= INTERVIEW_MAX_DEPTH:
            return {"error": f"depth must be 1..{INTERVIEW_MAX_DEPTH}"}, 400
        floor = body.get("ngram_distinct_floor")
        if floor is not None:
            spec["gates"]["ngram_distinct_floor"] = float(floor)

        # Seeds decide what the corpus is about. A vertical with no pack on disk
        # is refused here rather than silently falling back to the genre's own
        # thin `situations` list, which would produce a corpus the user did not ask for.
        vertical = (body.get("vertical") or "conversation").strip()
        topics = body.get("topics") or []
        seeds = seeds_mod.seeds_for(vertical, topics)
        if not seeds:
            return {"error": f"no seeds for vertical '{vertical}'"
                             + (f" topics {', '.join(topics)}" if topics else "")}, 400

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
        _write_job_meta(out_root, [], genre_ids, body.get("label"))

        gate = authoring_mod.RecordGate(spec)
        gate.seed_from_file(os.path.join(out_root, SAMPLES_FILE))
        conc = int(body.get("max_concurrency") or 0) or _resolve_concurrency(s, provider)
        job = interview_mod.InterviewJob(
            job_id, out_root, spec, genre_ids, n_convs, depth,
            provider, model=s.get("teacher_model") or None,
            base_url=s.get("teacher_base_url") or None,
            api_key=os.environ.get(TEACHER_API_KEY_ENV) or _stored_key(s, provider) or None,
            temperature=float(_setting(s, SETTING_TEMPERATURE)),
            max_concurrency=conc, gate=gate, seeds=seeds)

        prior = _read_plan(out_root)
        _write_plan(out_root, {
            "total_calls":   int(prior.get("total_calls") or 0) + n_convs,
            "conversations": int(prior.get("conversations") or 0) + n_convs,
            "depth":         depth,
            "genres":        sorted(set(list(prior.get("genres") or []) + genre_ids)),
            "max_concurrency": conc,
            "mode":          "interview",
            "vertical":      vertical,
            "topics":        topics,
            "seed_count":    len(seeds),
        })
        thread = threading.Thread(target=job.run, name=f"teacher-interview-{job_id}",
                                 daemon=True)
        with _JOBS_LOCK:
            _JOBS[job_id] = {"job": job, "thread": thread, "output_dir": out_root}
        thread.start()
        logmod.info(LOG_SOURCE, f"interview started: job={job_id} "
                                f"genres={','.join(genre_ids)} conversations={n_convs} "
                                f"depth={depth} concurrency={conc}")
        return {"job_id": job_id, "output_dir": out_root, "conversations": n_convs,
                "depth": depth, "total_calls": n_convs, "max_concurrency": conc,
                "vertical": vertical, "seed_count": len(seeds)}

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
        # An SFT corpus is not a pretrain corpus: standalone-prose genres (jokes,
        # writing, news) carry no user turn, so training on them teaches the model
        # to write rather than to answer. `genres` selects which families go in.
        want = body.get("genres")
        if want is not None:
            if not isinstance(want, list) or not all(isinstance(g, str) for g in want):
                return {"error": "genres must be a list of genre ids"}, 400
            unknown = [g for g in want if g not in counts]
            if unknown:
                return {"error": f"job has no records for genre(s): {', '.join(sorted(unknown))}. "
                                 f"available: {', '.join(sorted(counts))}"}, 400
            counts = {g: n for g, n in counts.items() if g in want}
            if not counts:
                return {"error": "genres selected nothing"}, 400
        dist_dir = os.path.join(output_dir, AUTHORING_DIST_DIR)
        manifest = build_sft_bins(
            stem, fam_dir, dist_dir, sorted(f"{g}.jsonl" for g in counts),
            AUTHORING_PURPOSE, AUTHORING_LICENSE,
            int(spec["gates"]["build_seed"]), float(spec["gates"]["build_val_ratio"]),
            corpus_dir=paths_mod.corpus_dir())
        train_bin = os.path.join(dist_dir, f"{stem}_train.bin")
        audit = _audit_or_none(train_bin)
        if audit:
            verdict = "passed" if audit["passed"] else "FAILED"
            logmod.info(LOG_SOURCE, f"corpus audit {verdict}: stem={stem} "
                                    f"unique_user={audit['unique_user_ratio']:.3f} "
                                    f"unique_content={audit['unique_content_ratio']:.3f} "
                                    f"median_assistant={audit['median_assistant_bytes']:.0f}B "
                                    f"artifacts_per_1k={audit['artifacts_per_1k']:.1f}")
        zip_path = _zip_bins(dist_dir, stem)
        entry = _register_catalog_entry(
            stem, body.get("label") or stem, body.get("description") or AUTHORING_PURPOSE,
            manifest, None, None)
        logmod.ok(LOG_SOURCE, f"authored corpus built: stem={stem} "
                              f"train={manifest['train_bytes']}B val={manifest['val_bytes']}B "
                              f"zip={zip_path}")
        return {
            "stem": stem, "zip_path": zip_path, "zip_bytes": os.path.getsize(zip_path),
            "zip_sha256": _sha256_file(zip_path),
            "family_counts": counts, "manifest": manifest, "catalog_entry": entry,
            "audit": audit,
            "next_steps": [
                f"Upload {zip_path} to COS. The zip holds {stem}_train.bin and "
                f"{stem}_val.bin at the top level.",
                f"COS returns a link. Paste it into corpus_catalog.json as the train_url "
                f"for stem '{stem}', replacing the PLACEHOLDER value.",
                "Remove \"coming_soon\": true from that same entry to release the corpus "
                "in the corpus library.",
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

    @app.route("/teacher/synth/browse", methods=["GET"])
    def teacher_synth_browse_route():
        """Paged read of any corpus on disk, running or long finished.

        `/teacher/synth/samples` tails the live job and cannot page; the corpora
        list needs to open a year-old job at record 4,000. Resolves through
        `_job_dir`, so nothing outside the jobs root is readable."""
        job_id = (request.args.get("job_id") or "").strip()
        if not job_id:
            return {"error": "job_id required"}, 400
        target = _job_dir(job_id)
        if target is None:
            return {"error": "unknown job"}, 404
        try:
            offset = int(request.args.get("offset") or 0)
        except ValueError:
            offset = 0
        try:
            limit = int(request.args.get("limit") or BROWSE_PAGE_DEFAULT)
        except ValueError:
            limit = BROWSE_PAGE_DEFAULT
        offset = max(0, offset)
        limit = max(1, min(limit, BROWSE_PAGE_MAX))
        meta = _read_job_meta(target)
        return {"job_id": job_id, "label": meta.get("label", ""),
                "offset": offset, "limit": limit,
                "total": _count_lines(os.path.join(target, SAMPLES_FILE)),
                "rows": _browse_samples(target, offset, limit)}

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
            "calls_ok": counts["calls_ok"],
            "calls_failed": counts["calls_failed"],
            "calls_remaining": counts["calls_remaining"],
            "failed": counts["failed"],
            "skipped_dup": counts["skipped_dup"],
            "last_error": counts["last_error"],
            "error_summary": counts["error_summary"],
            "aborted": counts["aborted"],
            "authoring": counts["authoring"],
            "opener_shortfall": counts["opener_shortfall"],
            "plan": counts["plan"],
            "output_path": counts["output_path"],
        }
