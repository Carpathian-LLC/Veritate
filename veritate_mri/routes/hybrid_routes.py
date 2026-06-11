# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - The chat endpoint behind the "/" front door. Resolves a user-selected trained
#   model + engine (pytorch or c), generates a reply, and optionally grounds it
#   with retrieval. With no local model it uses the always-available public
#   Carpathian model (ai_assist, hardcoded public key).
# - Retrieval is local BM25 over a knowledge base the user builds by uploading
#   text files (KB_DIR). No external embedder, no Ollama: the BM25 index is the
#   platform's own pure-Python retriever.
# - Generation reuses the platform paths: load via _brain / _spawn_c_subprocess,
#   decode via brain.stream / _c_engine_stream. No bespoke decode loop, no
#   hardcoded model.
# veritate_mri/routes/hybrid_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from flask import current_app, request

from readers import checkpoints, models, paths

from ._common import auto_thread_count, user_error

# ------------------------------------------------------------------------------------
# Constants

CLOUD_ID       = "cloud"
CLOUD_LABEL    = "Carpathian AI (public)"
TEACHER_ID     = "teacher"
KB_DIR         = os.path.join(paths.DATA_ROOT, "kb")
KB_EXTS        = (".txt", ".md", ".rst", ".text")
KB_CHUNK_BYTES = 1024
KB_CHUNK_OVERLAP = 128
KB_CHUNK_PREVIEW = 480
UPLOAD_MAX_BYTES = 64 * 1024 * 1024
TOP_K          = 3
MAX_NEW        = 256
PROMPT_TMPL    = "context: {ctx}\n<|user|>\n{msg}\n<|assistant|>\n"
PLAIN_TMPL     = "<|user|>\n{msg}\n<|assistant|>\n"
STOP_MARKERS   = ("<|end|>", "<|user|>", "\ncontext:")

_STATE = {"chunks": None, "bm25": None}


# ------------------------------------------------------------------------------------
# Functions

def kb_files():
    if not os.path.isdir(KB_DIR):
        return []
    return [f for f in sorted(os.listdir(KB_DIR)) if os.path.isfile(os.path.join(KB_DIR, f))]


def has_corpus():
    return len(kb_files()) > 0


def _kb_chunks():
    """Cached list of text chunks across every file in KB_DIR. Reuses the
    platform retriever's whitespace-aware chunker."""
    if _STATE["chunks"] is None:
        from inference.agent.tools.retriever import _split_chunks
        out = []
        for fn in kb_files():
            try:
                with open(os.path.join(KB_DIR, fn), "rb") as fh:
                    text = fh.read().decode("utf-8", "replace")
            except OSError:
                continue
            for _off, ch in _split_chunks(text, KB_CHUNK_BYTES, KB_CHUNK_OVERLAP):
                if ch:
                    out.append(ch)
        _STATE["chunks"] = out
    return _STATE["chunks"]


def _kb_index():
    if _STATE["bm25"] is None:
        from inference.agent.tools.retriever import BM25Index
        chunks = _kb_chunks()
        _STATE["bm25"] = BM25Index([(i, c) for i, c in enumerate(chunks)])
    return _STATE["bm25"]


def retrieve(message, k=TOP_K):
    chunks = _kb_chunks()
    if not chunks:
        return [], []
    hits = _kb_index().search(message, k)
    return ([chunks[did][:KB_CHUNK_PREVIEW] for _s, did in hits],
            [round(float(s), 2) for s, did in hits])


def reset_index_cache():
    """Drop the cached corpus + index so the next retrieve rebuilds from disk.
    Called after an upload grows KB_DIR."""
    _STATE["chunks"] = None
    _STATE["bm25"] = None


def _save_kb_upload(f):
    from werkzeug.utils import secure_filename
    name = secure_filename(f.filename) or "upload"
    if not name.lower().endswith(KB_EXTS):
        name += ".txt"
    os.makedirs(KB_DIR, exist_ok=True)
    f.save(os.path.join(KB_DIR, name))
    reset_index_cache()
    return name


def build_prompt(message, facts):
    return PROMPT_TMPL.format(ctx=" ".join(facts), msg=message)


def build_plain_prompt(message):
    return PLAIN_TMPL.format(msg=message)


def collect(events):
    """Join a generation stream into text. Token bytes carry on 'token' /
    'fast_byte' events; 'stop' / 'error' end the stream."""
    out = bytearray()
    for ev in events:
        kind = ev.get("kind")
        if kind in ("token", "fast_byte"):
            b = ev.get("byte")
            if b is not None:
                out.append(int(b))
        elif kind in ("stop", "error"):
            break
    return bytes(out).decode("utf-8", "replace")


def _trim(answer):
    for stop in STOP_MARKERS:
        answer = answer.split(stop)[0]
    return answer.strip()


def is_local_model(name):
    return bool(name) and name != CLOUD_ID and models.exists(name) \
        and checkpoints.latest_step(name) is not None


def _ensure_pytorch(cfg, name):
    from . import _brain
    if cfg.get("BRAIN") is not None and cfg.get("BRAIN_MODEL") == name:
        return cfg["BRAIN"]
    step = checkpoints.latest_step(name)
    threads = int(cfg.get("DEFAULT_THREADS") or auto_thread_count())
    brain, name, step = _brain.load_pytorch_brain(name, step, threads)
    cfg["BRAIN"] = brain
    cfg["BRAIN_MODEL"] = name
    cfg["BRAIN_STEP"] = int(step)
    return brain


def _ensure_c(cfg, name):
    from readers import bin as binr
    from .backends_routes import _spawn_c_subprocess
    if not binr.exists(name):
        raise FileNotFoundError(f"{name} has no veritate.bin; export it or use the pytorch engine")
    exe = paths.engine_binary_path()
    if not os.path.isfile(exe):
        raise FileNotFoundError("c engine not built; build it from the platform Settings")
    model_bin = os.path.abspath(paths.bin_path(name))
    if cfg.get("C_SUBPROCESS") is not None and os.path.abspath(cfg.get("C_MODEL") or "") == model_bin:
        return
    old = cfg.get("C_SUBPROCESS")
    if old is not None:
        try: old.close()
        except Exception: pass
    if not _spawn_c_subprocess(cfg, exe, model_bin):
        raise RuntimeError("c engine failed to spawn")


def _generate_local(cfg, backend, prompt):
    if backend == "c":
        from .backends_routes import _c_engine_stream
        events = _c_engine_stream(cfg, prompt, MAX_NEW)
    else:
        events = cfg["BRAIN"].stream(prompt, max_new=MAX_NEW)
    return _trim(collect(events))


def _cloud_answer(message):
    """The always-available public model. Backed by the hardcoded public
    Carpathian key in settings DEFAULTS (overridable per-user), via ai_assist."""
    from runtime import ai_assist
    res = ai_assist.chat(message)
    if not res.get("ok"):
        return ({"ok": False, "error": f"{CLOUD_LABEL} unavailable: {res.get('error', '')}".strip()}, 503)
    return {"ok": True, "answer": (res.get("answer") or "").strip(), "model": CLOUD_LABEL,
            "backend": "cloud", "confident": False, "sources": []}


def teacher_view():
    """The configured teacher model, if any, surfaced for the chat dropdown.
    Provider + model name only; the key never leaves the server."""
    from runtime import settings as settings_mod
    s = settings_mod.get()
    provider = (s.get("teacher_provider") or "").strip()
    model = (s.get("teacher_model") or "").strip()
    if provider and model:
        return {"configured": True, "label": f"{provider}: {model}"}
    return {"configured": False, "label": ""}


def _teacher_answer(message):
    """The user-configured teacher provider/model (any of the known providers),
    using its own key from settings or VERITATE_TEACHER_API_KEY."""
    from runtime import settings as settings_mod
    from teacher import client as teacher_client, providers as teacher_providers
    s = settings_mod.get()
    provider = (s.get("teacher_provider") or "").strip()
    model = (s.get("teacher_model") or "").strip()
    if not provider or not model:
        return ({"ok": False, "error": "no teacher model configured; set one in the platform Settings"}, 503)
    api_key = teacher_providers.resolve_api_key(provider, s.get("teacher_api_key"))
    prov = teacher_providers.get_provider(provider)
    if prov and prov.get("requires_key") and not api_key:
        return ({"ok": False, "error": f"teacher {provider} needs an API key (set it in Settings)"}, 503)
    try:
        answer = teacher_client.complete(provider, model,
                                         [{"role": "user", "content": message}],
                                         api_key=api_key, base_url=s.get("teacher_base_url") or None)
    except teacher_client.TeacherError as e:
        return ({"ok": False, "error": user_error(e, "teacher chat")}, 503)
    return {"ok": True, "answer": (answer or "").strip(), "model": f"{provider}: {model}",
            "backend": "teacher", "confident": False, "sources": []}


def register(app):
    @app.route("/hybrid/health")
    def hybrid_health():
        return {"ok": True, "has_corpus": has_corpus(), "n_files": len(kb_files()),
                "teacher": teacher_view()}

    @app.route("/hybrid/kb/upload", methods=["POST"])
    def hybrid_kb_upload():
        f = request.files.get("file")
        if f is None or not f.filename:
            return ({"ok": False, "error": "no file uploaded"}, 400)
        if request.content_length and request.content_length > UPLOAD_MAX_BYTES:
            return ({"ok": False, "error": f"file too large; cap is "
                     f"{UPLOAD_MAX_BYTES // (1024 * 1024)} MB"}, 413)
        try:
            name = _save_kb_upload(f)
        except OSError as e:
            return ({"ok": False, "error": user_error(e, "save upload")}, 500)
        return {"ok": True, "filename": name, "n_files": len(kb_files()),
                "n_chunks": len(_kb_chunks())}

    @app.route("/hybrid/chat", methods=["POST"])
    def hybrid_chat():
        cfg = current_app.config
        body = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip()
        if not message:
            return ({"ok": False, "error": "empty message"}, 400)
        model = (body.get("model") or "").strip()
        backend = (body.get("backend") or "pytorch").strip().lower()
        use_rag = bool(body.get("use_rag"))
        k = int(body.get("k") or TOP_K)

        if model == TEACHER_ID:
            return _teacher_answer(message)
        if not is_local_model(model):
            return _cloud_answer(message)

        facts, sources = [], []
        if use_rag and has_corpus():
            facts, scores = retrieve(message, k)
            sources = [{"text": t, "score": s} for t, s in zip(facts, scores)]

        prompt = build_prompt(message, facts) if facts else build_plain_prompt(message)
        try:
            if backend == "c":
                _ensure_c(cfg, model)
            else:
                _ensure_pytorch(cfg, model)
            answer = _generate_local(cfg, backend, prompt)
        except (FileNotFoundError, RuntimeError) as e:
            return ({"ok": False, "error": user_error(e)}, 503)
        except Exception as e:
            return ({"ok": False, "error": user_error(e)}, 500)
        return {"ok": True, "answer": answer, "model": model, "backend": backend,
                "confident": bool(facts), "sources": sources}
