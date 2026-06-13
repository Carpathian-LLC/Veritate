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
TEACHER_PREFIX = "teacher:"
CLOUD_GROUP    = "Cloud"
ENUM_TIMEOUT_S = 3
KB_DIR         = os.path.join(paths.DATA_ROOT, "kb")
PLATFORM_KB_DIR = os.path.join(paths.DATA_ROOT, "platform_kb")
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

CHAT_SYSTEM    = ("You are Veritate, a concise, helpful assistant. Use the conversation so far and any "
                  "provided facts to answer the user directly. No emoji, no emdash.")
SUMMARY_SYSTEM = ("Condense the conversation into a brief note preserving facts, names, decisions, and "
                  "the user's goal. Plain text, no preamble.")
HISTORY_MAX_TURNS       = 60
CTX_KEEP_TAIL_TURNS     = 6
SUMMARY_MAX_CHARS       = 1500

# Context window the conversation memory is measured against, per model. Local
# Veritate models are byte-level, so the window is exactly their trained seq
# (chars == bytes). Cloud/teacher windows are not exposed by the OpenAI-compatible
# API, so these are nominal per-provider model-family sizes in tokens; the value a
# host actually serves (e.g. an Ollama num_ctx) may be lower. CONTEXT_MEMORY_FRACTION
# reserves room for the system preamble, retrieved facts, the question, and the reply.
CHARS_PER_TOKEN         = 4
DEFAULT_CONTEXT_TOKENS  = 8192
DEFAULT_LOCAL_SEQ       = 1024
MIN_CONTEXT_CHARS       = 256
CONTEXT_MEMORY_FRACTION = 0.6
CONTEXT_TOKENS_BY_PROVIDER = {
    "carpathian": 32768, "openai": 128000, "anthropic": 200000, "gemini": 1000000,
    "xai": 131072, "deepseek": 65536, "mistral": 32000, "groq": 32768,
    "openrouter": 32768, "ollama": 8192, "lm_studio": 8192, "llama_cpp": 8192,
}

VERITATE_DOCS_ID = "veritate_docs"   # chat-page-only pick: public model grounded on the platform KB
KB_SCOPES        = ("all", "platform", "user")
TRAIN_LOG_SOURCES = ("plugin", "save")   # ring-buffer sources shown on the Training tab; plugin:<id> by prefix
TRAIN_LOG_TAIL    = 15

_STATE = {}   # scope -> {"chunks": [...], "bm25": index}; built lazily, cleared by reset_index_cache


# ------------------------------------------------------------------------------------
# Functions

def _dir_files(d):
    if not os.path.isdir(d):
        return []
    return [f for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))]


def kb_files():
    """User-uploaded knowledge base files (KB_DIR only). Drives the upload UI's
    file count; the retrieval corpus is wider (see _corpus_files)."""
    return _dir_files(KB_DIR)


def _corpus_files(scope="all"):
    """(dir, filename) for the retrievable docs in `scope`: `platform` is the
    shipped Veritate KB, `user` is the uploads, `all` is both. The platform KB
    ships in the repo, so chat can answer platform questions on a fresh install
    with no upload."""
    dirs = {"platform": (PLATFORM_KB_DIR,), "user": (KB_DIR,)}.get(scope, (PLATFORM_KB_DIR, KB_DIR))
    return [(d, fn) for d in dirs for fn in _dir_files(d)]


def has_corpus():
    return len(_corpus_files()) > 0


def _kb_chunks(scope="all"):
    """Per-scope cached text chunks. Reuses the platform retriever's
    whitespace-aware chunker."""
    st = _STATE.get(scope)
    if st is None:
        from inference.agent.tools.retriever import _split_chunks
        out = []
        for d, fn in _corpus_files(scope):
            try:
                with open(os.path.join(d, fn), "rb") as fh:
                    text = fh.read().decode("utf-8", "replace")
            except OSError:
                continue
            for _off, ch in _split_chunks(text, KB_CHUNK_BYTES, KB_CHUNK_OVERLAP):
                if ch:
                    out.append(ch)
        st = _STATE[scope] = {"chunks": out, "bm25": None}
    return st["chunks"]


def _kb_index(scope="all"):
    _kb_chunks(scope)
    st = _STATE[scope]
    if st["bm25"] is None:
        from inference.agent.tools.retriever import BM25Index
        st["bm25"] = BM25Index([(i, c) for i, c in enumerate(st["chunks"])])
    return st["bm25"]


def retrieve(message, k=TOP_K, scope="all"):
    chunks = _kb_chunks(scope)
    if not chunks:
        return [], []
    hits = _kb_index(scope).search(message, k)
    return ([chunks[did][:KB_CHUNK_PREVIEW] for _s, did in hits],
            [round(float(s), 2) for s, did in hits])


def training_log_lines():
    """Recent Training-tab log lines (trainer subprocess output, run control, and
    checkpoint saves) from the in-memory ring, so the platform-docs assistant can
    answer questions about a run. Tail only; the ring is ephemeral."""
    from runtime import logs as logmod
    lines = [f"[{e['level']}] {e['source']}: {e['msg']}" for e in logmod.snapshot()
             if e["source"] in TRAIN_LOG_SOURCES or e["source"].startswith("plugin:")]
    return lines[-TRAIN_LOG_TAIL:]


def reset_index_cache():
    """Drop the cached corpus + index so the next retrieve rebuilds from disk.
    Called after an upload grows KB_DIR."""
    _STATE.clear()


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


class ChatUnavailable(Exception):
    """An expected, user-safe chat failure (bad key, unreachable host, nothing
    configured). hybrid_chat maps it to a 503."""


def _provider_model_names(pid, prov, cfg):
    """Selectable model names for one configured teacher provider. Local-kind
    providers (ollama, lm_studio, llama_cpp) are enumerated live so every
    installed sub-model is offered; a dead host falls back to the configured
    model. API providers offer the model the user saved (or the provider
    default). Enumeration uses a short timeout so a down host can't stall the
    picker."""
    from teacher import client as teacher_client, providers as teacher_providers
    configured = (cfg.get("model") or "").strip()
    if prov["kind"] == "local":
        try:
            c = teacher_client.Client(pid, base_url=cfg.get("base_url") or None,
                                      api_key=cfg.get("api_key") or None,
                                      timeout_s=ENUM_TIMEOUT_S, max_retries=0)
            names = c.list_models() or []
        except teacher_client.TeacherError:
            names = []
        if names:
            return names
        return [configured] if configured else []
    if configured:
        return [configured]
    default = teacher_providers.default_model_for(pid)
    return [default] if default else []


def remote_models():
    """Every selectable non-local model for the chat picker: the public
    Carpathian model plus, per teacher provider configured in Settings, its
    installed/saved models. Each entry carries an id the chat route can route on
    (`cloud`, or `teacher:<provider>:<model>`); keys never leave the server."""
    from runtime import settings as settings_mod
    from teacher import providers as teacher_providers
    s = settings_mod.get()
    out = [{"id": CLOUD_ID, "label": CLOUD_LABEL, "group": CLOUD_GROUP}]
    for pid, cfg in (s.get("teacher_configs") or {}).items():
        try:
            prov = teacher_providers.get_provider(pid)
        except ValueError:
            continue
        for name in _provider_model_names(pid, prov, cfg):
            out.append({"id": f"{TEACHER_PREFIX}{pid}:{name}", "provider": pid,
                        "model": name, "label": f"{prov['display_name']}: {name}",
                        "group": prov["display_name"]})
    return out


def _render_local(messages, system):
    """Byte-model prompt from a messages list: prior turns rendered with the
    platform chat markers, the final user turn via build_prompt / build_plain_prompt.
    A lone user turn with no system is identical to single-turn generation; a
    non-empty system rides in as the final turn's `context:` block."""
    *prior, last = messages
    head = []
    for m in prior:
        if m["role"] == "user":
            head.append(f"<|user|>\n{m['content']}")
        else:
            head.append(f"<|assistant|>\n{m['content']}<|end|>")
    final = build_prompt(last["content"], [system]) if system else build_plain_prompt(last["content"])
    return ("\n".join(head) + "\n" + final) if head else final


def _system_text(kind, summary, facts):
    """Preamble for one turn. Remote models get the chat system prompt plus
    framed summary + facts; local byte models get only the raw summary + facts
    (rendered as a `context:` block by _render_local, matching their training)."""
    if kind == "local":
        parts = []
        if summary:
            parts.append(summary)
        if facts:
            parts.append(" ".join(facts))
        return " ".join(parts).strip()
    parts = [CHAT_SYSTEM]
    if summary:
        parts.append("Conversation summary so far:\n" + summary)
    if facts:
        parts.append("Relevant facts from the knowledge base:\n" + "\n".join(f"- {f}" for f in facts))
    return "\n\n".join(parts)


def _resolve_route(cfg, model, backend):
    """Pick the completion route for the selected chat model. Returns
    (complete, label, resp_backend, kind). complete(messages, system) -> text and
    raises ChatUnavailable on an expected failure. kind ('remote' | 'local')
    drives how summary + facts are framed."""
    from runtime import settings as settings_mod

    if model == TEACHER_ID or model.startswith(TEACHER_PREFIX):
        from teacher import client as teacher_client, providers as teacher_providers
        s = settings_mod.get()
        if model == TEACHER_ID:
            provider = (s.get("teacher_provider") or "").strip()
            model_name = (s.get("teacher_model") or "").strip()
        else:
            provider, _, model_name = model[len(TEACHER_PREFIX):].partition(":")
        tcfg = (s.get("teacher_configs") or {}).get(provider, {})
        is_active = provider == (s.get("teacher_provider") or "")
        stored_key = tcfg.get("api_key") or (s.get("teacher_api_key", "") if is_active else "")
        base_url = tcfg.get("base_url") or (s.get("teacher_base_url", "") if is_active else "") or None

        def complete(messages, system):
            if not provider or not model_name:
                raise ChatUnavailable("no teacher model selected")
            try:
                prov = teacher_providers.get_provider(provider)
            except ValueError:
                raise ChatUnavailable(f"unknown teacher provider: {provider}")
            api_key = teacher_providers.resolve_api_key(provider, stored_key)
            if prov.get("requires_key") and not api_key:
                raise ChatUnavailable(f"teacher {provider} needs an API key (set it in Settings)")
            try:
                return teacher_client.complete(provider, model_name, messages,
                                               api_key=api_key, base_url=base_url, system=system)
            except teacher_client.TeacherError as e:
                raise ChatUnavailable(user_error(e, "teacher chat"))
        return (complete, f"{provider}: {model_name}", "teacher", "remote",
                context_limit_chars("remote", provider, model_name))

    if is_local_model(model):
        def complete(messages, system):
            if backend == "c":
                _ensure_c(cfg, model)
            else:
                _ensure_pytorch(cfg, model)
            return _generate_local(cfg, backend, _render_local(messages, system))
        return complete, model, backend, "local", context_limit_chars("local", "", model)

    from runtime import ai_assist

    def complete(messages, system):
        *hist, last = messages
        res = ai_assist.chat(last["content"], system=system, history=hist)
        if not res.get("ok"):
            raise ChatUnavailable(f"{CLOUD_LABEL} unavailable: {res.get('error', '')}".strip())
        return res.get("answer") or ""
    return complete, CLOUD_LABEL, "cloud", "remote", context_limit_chars("remote", "carpathian", "")


def _context_used(summary, turns):
    return len(summary) + sum(len(m["content"]) for m in turns)


def _local_seq(model):
    from readers import config as cfg_reader
    cfg = cfg_reader.load(model) or {}
    seq = (cfg.get("shape") or {}).get("seq") or cfg.get("seq")
    return int(seq) if seq else DEFAULT_LOCAL_SEQ


def context_limit_chars(kind, provider, model):
    """Conversation-memory budget in characters for the selected model. Local
    models use their byte-level seq; cloud/teacher use a nominal per-provider
    window. CONTEXT_MEMORY_FRACTION reserves room for the prompt and the reply."""
    if kind == "local":
        window = _local_seq(model)
    else:
        window = CONTEXT_TOKENS_BY_PROVIDER.get(provider, DEFAULT_CONTEXT_TOKENS) * CHARS_PER_TOKEN
    return max(MIN_CONTEXT_CHARS, int(window * CONTEXT_MEMORY_FRACTION))


def _context_meter(summary, turns, char_limit):
    """How full the conversation memory is against the selected model's context
    budget. Drives the chat-page gauge."""
    used = _context_used(summary, turns)
    return {"turns": len(turns), "chars": used, "char_limit": char_limit,
            "pct": round(min(1.0, used / char_limit), 3)}


def _compact(complete, summary, turns, char_limit):
    """Fold older turns into a running summary once the memory exceeds the model's
    context budget, keeping the last CTX_KEEP_TAIL_TURNS verbatim. Best-effort: a
    failed summarize call leaves the history intact."""
    if len(turns) <= CTX_KEEP_TAIL_TURNS or _context_used(summary, turns) <= char_limit:
        return summary, turns
    head, tail = turns[:-CTX_KEEP_TAIL_TURNS], turns[-CTX_KEEP_TAIL_TURNS:]
    rendered = "\n".join(f"{m['role']}: {m['content']}" for m in head)
    src = (f"Existing summary:\n{summary}\n\n" if summary else "") + f"Conversation to fold in:\n{rendered}"
    try:
        new_summary = complete([{"role": "user", "content": src}], SUMMARY_SYSTEM).strip()
    except (ChatUnavailable, FileNotFoundError, RuntimeError, OSError, ValueError):
        return summary, turns
    return new_summary[:SUMMARY_MAX_CHARS], tail


def _history_in(body):
    """Validate the inbound conversation memory: a capped list of {role, content}
    turns plus the running summary string."""
    raw = body.get("history")
    turns = []
    if isinstance(raw, list):
        for m in raw[-HISTORY_MAX_TURNS:]:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") \
                    and isinstance(m.get("content"), str) and m["content"]:
                turns.append({"role": m["role"], "content": m["content"]})
    summary = body.get("summary")
    summary = summary.strip()[:SUMMARY_MAX_CHARS] if isinstance(summary, str) else ""
    return turns, summary


def register(app):
    @app.route("/hybrid/health")
    def hybrid_health():
        return {"ok": True, "has_corpus": has_corpus(), "n_files": len(kb_files())}

    @app.route("/hybrid/models")
    def hybrid_models():
        return {"models": remote_models()}

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
        use_logs = bool(body.get("use_logs"))
        scope = body.get("kb_scope") if body.get("kb_scope") in KB_SCOPES else "all"
        k = int(body.get("k") or TOP_K)
        history, summary = _history_in(body)

        # Chat-page "Veritate (platform docs)" pick: the public model forced to
        # ground on the shipped platform KB. Not a backend model of its own.
        docs_mode = model == VERITATE_DOCS_ID
        if docs_mode:
            model, use_rag, scope = CLOUD_ID, True, "platform"

        facts, sources = [], []
        if use_rag and has_corpus():
            facts, scores = retrieve(message, k, scope=scope)
            sources = [{"text": t, "score": s} for t, s in zip(facts, scores)]
        if docs_mode or use_logs:
            log_facts = training_log_lines()
            facts = facts + log_facts
            sources = sources + [{"text": ln, "score": None} for ln in log_facts]

        complete, label, resp_backend, kind, char_limit = _resolve_route(cfg, model, backend)
        if docs_mode:
            label = "Veritate (platform docs)"
        system = _system_text(kind, summary, facts)
        messages = history + [{"role": "user", "content": message}]
        try:
            answer = (complete(messages, system) or "").strip()
        except ChatUnavailable as e:
            return ({"ok": False, "error": str(e)}, 503)
        except (FileNotFoundError, RuntimeError) as e:
            return ({"ok": False, "error": user_error(e)}, 503)
        except Exception as e:
            return ({"ok": False, "error": user_error(e)}, 500)

        new_turns = history + [{"role": "user", "content": message},
                               {"role": "assistant", "content": answer}]
        mem_summary, mem_turns = _compact(complete, summary, new_turns, char_limit)
        return {"ok": True, "answer": answer, "model": label, "backend": resp_backend,
                "confident": bool(facts), "sources": sources,
                "memory": {"summary": mem_summary, "turns": mem_turns},
                "context": _context_meter(mem_summary, mem_turns, char_limit)}
