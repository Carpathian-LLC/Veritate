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

import codecs
import json
import os
import re
import time
import traceback
import uuid
from collections import namedtuple

from flask import Response, current_app, request
from readers import checkpoints, models, paths
from runtime import logs as logmod

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
# Sampling used when a request carries no override. The values the box generated
# with before per-request params were honored.
TEMPERATURE_DEFAULT = 0.7
TOP_K_DEFAULT       = 40
LOCAL_FACT_MIN_CHARS = 80   # below this a truncated fact is useless; drop it instead
PROMPT_TMPL    = "context: {ctx}\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
PLAIN_TMPL     = "<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
STOP_MARKERS   = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "\ncontext:")
_STOP_MARKER_PREFIXES = set()
for _m in STOP_MARKERS:
    for _i in range(1, len(_m) + 1):
        _STOP_MARKER_PREFIXES.add(_m[:_i])

WIRE_NEWLINE    = "\x01"       # c chat_traced encodes prompt/reply newlines as 0x01
KEEP_CTRL_CHARS = ("\t", "\n")  # every other control byte is dropped from the answer
STOP_HOLD       = max(len(m) for m in STOP_MARKERS)  # per-token holdback so a forming stop marker never streams


def _dynamic_stop_hold(text):
    """Return how many trailing chars must be held back to avoid streaming a
    partial stop marker. Zero unless the tail actually looks like a growing
    prefix of some STOP_MARKER: so most bytes stream instantly instead of
    waiting behind a fixed STOP_HOLD lag (13 chars on the "<|endoftext|>"
    marker adds ~450 ms at 35 ms/byte, which is what causes the streaming
    bimodal ~108 ms inter-arrival spike measured 2026-07-20). O(STOP_HOLD)
    per call via the precomputed prefix set: no full re-scan of `text`."""
    n = len(text)
    if n == 0:
        return 0
    lim = min(STOP_HOLD, n)
    for i in range(lim, 0, -1):
        if text[n - i:] in _STOP_MARKER_PREFIXES:
            return i
    return 0


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

# OpenAI-compatible /v1/chat/completions wire tokens. Local trained models stream
# true per-token deltas via the same generator /generate uses; cloud/teacher
# complete() is non-streaming, so their streaming chunks the finished answer on
# CHAT_STREAM_SPLIT (word + trailing whitespace) into deltas.
OPENAI_CHAT_OBJECT  = "chat.completion"
OPENAI_CHUNK_OBJECT = "chat.completion.chunk"
OPENAI_FINISH_STOP  = "stop"
CHATCMPL_PREFIX     = "chatcmpl-"
SSE_DONE            = "data: [DONE]\n\n"
CHAT_STREAM_SPLIT   = r"\S+\s*"
SSE_HEADERS         = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# Opt-in per-byte MRI telemetry: request field + response/extension key. Absent or
# false = today's fast, non-traced serving, byte-for-byte. Applies to local trained
# models only (remote providers expose no per-byte telemetry).
MRI_KEY = "mri"

_STATE = {}   # scope -> {"chunks": [...], "bm25": index}; built lazily, cleared by reset_index_cache

# Resolved chat request: request body + retrieval + route, shared by the buffered
# /hybrid/chat and the streaming /hybrid/chat/stream so both frame a reply identically.
_ChatCtx = namedtuple("_ChatCtx", "message history summary facts sources complete "
                                  "label resp_backend kind char_limit model backend messages system mri")


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


def _sanitize_text(answer):
    """Restore wire-encoded newlines, then drop control bytes a byte-level model
    can emit (except tab/newline) so the answer is clean text and valid JSON."""
    answer = answer.replace(WIRE_NEWLINE, "\n")
    return "".join(c for c in answer
                   if c in KEEP_CTRL_CHARS or (ord(c) >= 0x20 and ord(c) != 0x7f))


def _trim(answer):
    answer = _sanitize_text(answer)
    for stop in STOP_MARKERS:
        answer = answer.split(stop)[0]
    return answer.strip()


def is_local_model(name):
    return bool(name) and name != CLOUD_ID and models.exists(name) \
        and checkpoints.latest_step(name) is not None


def _default_local_backend(name):
    """Backend a local model serves on when the caller sends none: the compiled C
    engine when the model has a usable .bin and the engine binary is built (fast,
    CPU subprocess, matches the dashboard and /generate), else the pytorch brain
    for non-exportable trunks (recurrent/RoPE/MTP)."""
    from readers import bin as binr
    return "c" if binr.exists(name) and os.path.isfile(paths.engine_binary_path()) else "pytorch"


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

    from .backends_routes import _spawn_c_subprocess, warm_is_pinned, warm_select
    if not binr.exists(name):
        raise FileNotFoundError(f"{name} has no veritate.bin; export it or use the pytorch engine")
    exe = paths.engine_binary_path()
    if not os.path.isfile(exe):
        raise FileNotFoundError("c engine not built; build it from the platform Settings")
    model_bin = os.path.abspath(paths.bin_path(name))
    if cfg.get("C_SUBPROCESS") is not None and os.path.abspath(cfg.get("C_MODEL") or "") == model_bin:
        return
    old = cfg.get("C_SUBPROCESS")
    # A warm-pinned outgoing model stays resident in the pool; only a non-pinned
    # single-slot subprocess is closed on switch.
    if warm_select(cfg, name):
        if old is not None and not warm_is_pinned(cfg, old):
            try: old.close()
            except Exception: pass
        return
    if old is not None and not warm_is_pinned(cfg, old):
        try: old.close()
        except Exception: pass
    if not _spawn_c_subprocess(cfg, exe, model_bin):
        raise RuntimeError("c engine failed to spawn")


def _gen_params_in(body):
    """Per-request generation overrides from an OpenAI-shaped body: temperature /
    max_tokens / top_k plus the repetition controls. Absent or unparseable keys are
    dropped so _local_events falls back to the box defaults. Stays clear of the
    numpy-bound backends_routes import so a box with no ML deps can still route an
    OpenAI client (which always sends temperature) to a cloud model. This is a public
    surface, so every value is bounded here; top_k and max_new carry no ceiling in the
    table because theirs are the engine constants clamped in _local_events."""
    spec = (
        ("temperature",     "temperature",     float, 0.0, 2.0),
        ("top_k",           "top_k",           int,   1,   None),
        ("max_tokens",      "max_new",         int,   1,   None),
        ("rep_window",      "rep_window",      int,   0,   4096),
        ("rep_penalty",     "rep_penalty",     float, 0.0, 2.0),
        ("no_repeat_ngram", "no_repeat_ngram", int,   0,   64),
    )
    gen = {}
    for wire, name, cast, lo, hi in spec:
        if body.get(wire) is None:
            continue
        try:
            v = max(lo, cast(body[wire]))
        except (TypeError, ValueError):
            continue
        gen[name] = min(v, hi) if hi is not None else v
    return gen


def _local_events(cfg, backend, prompt, mri=False, gen=None):
    """Per-token generation event stream plus turn-stop markers for a local model.
    Shared by the buffered (_generate_local) and true-token chat paths. mri selects
    the full-telemetry path (C trace=True / Brain.stream) whose per-byte frames the
    /generate MRI view consumes; default off keeps the fast path (C trace=False
    coarse fast_byte / Brain.stream_fast lookahead), byte-for-byte as before.
    gen carries per-request overrides (see _gen_params_in); an empty gen reproduces
    the values the box hardcoded before overrides existed."""
    from inference.decode import (
        NO_REPEAT_NGRAM_DEFAULT,
        REP_PENALTY_DEFAULT,
        REP_WINDOW_DEFAULT,
    )

    from .backends_routes import BYTE_VOCAB, MAX_NEW_CAP, _chat_stop_seq
    g = gen or {}
    temperature = g.get("temperature", TEMPERATURE_DEFAULT)
    top_k = min(g.get("top_k", TOP_K_DEFAULT), BYTE_VOCAB)
    max_new = min(g.get("max_new", MAX_NEW), MAX_NEW_CAP)
    rep = {"rep_window": g.get("rep_window", REP_WINDOW_DEFAULT),
           "rep_penalty": g.get("rep_penalty", REP_PENALTY_DEFAULT),
           "no_repeat_ngram": g.get("no_repeat_ngram", NO_REPEAT_NGRAM_DEFAULT)}
    if backend == "c":
        from .backends_routes import _c_engine_stream
        events = _c_engine_stream(cfg, prompt, max_new, temperature=temperature,
                                  top_k=top_k, trace=mri, **rep)
    elif mri:
        events = _pytorch_mri_events(cfg["BRAIN"], prompt, rep, temperature, top_k, max_new)
    else:
        events = cfg["BRAIN"].stream_fast(prompt, mode="lookahead", temperature=temperature,
                                          top_k_sample=top_k, max_new=max_new, **rep)
    return events, _chat_stop_seq(prompt)


def _pytorch_mri_events(brain, prompt, rep, temperature=TEMPERATURE_DEFAULT,
                        top_k=TOP_K_DEFAULT, max_new=MAX_NEW):
    """Full per-byte telemetry stream (Brain.stream), backend-tagged so its frames
    match what /generate hands the MRI view. Sampling matches the fast lookahead
    path: the flag selects telemetry, never sampling. Under greedy both paths are
    byte-identical; under sampling both draw each byte fresh from the true-prefix
    distribution (see inference_brain.md)."""
    for ev in brain.stream(prompt, temperature=temperature, top_k_sample=top_k,
                           max_new=max_new, **rep):
        ev["backend"] = "pytorch"
        yield ev


def _generate_local(cfg, backend, prompt):
    from inference.decode import abstention

    from .backends_routes import _stop_on_bytes
    events, stop_seq = _local_events(cfg, backend, prompt)
    text = _trim(collect(_stop_on_bytes(events, stop_seq)))
    return abstention.wrap_response_text(text)


def _generate_local_mri(cfg, model, backend, messages, system, gen=None):
    """Local generation that also captures the per-byte MRI frames the /generate view
    consumes (the meta frame, one token frame per byte, then the turn-stop frame).
    Returns (answer, frames). Text is byte-identical to _generate_local under greedy;
    the flag selects telemetry, never sampling (see _pytorch_mri_events)."""
    from .backends_routes import _stop_on_bytes
    if backend == "c":
        _ensure_c(cfg, model)
    else:
        _ensure_pytorch(cfg, model)
    events, stop_seq = _local_events(cfg, backend, _render_local(messages, system),
                                     mri=True, gen=gen)
    frames, out = [], bytearray()
    for ev in _stop_on_bytes(events, stop_seq):
        frames.append(ev)
        kind = ev.get("kind")
        if kind in ("token", "fast_byte"):
            b = ev.get("byte")
            if b is not None:
                out.append(int(b))
        elif kind in ("stop", "error"):
            break
    return _trim(bytes(out).decode("utf-8", "replace")), frames


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
    from teacher import client as teacher_client
    from teacher import providers as teacher_providers
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
            head.append(f"<|im_start|>user\n{m['content']}<|im_end|>")
        else:
            head.append(f"<|im_start|>assistant\n{m['content']}<|im_end|>")
    final = build_prompt(last["content"], [system]) if system else build_plain_prompt(last["content"])
    return ("\n".join(head) + "\n" + final) if head else final


def _system_text(kind, summary, facts):
    """Preamble for one turn. Remote models get the chat system prompt plus
    framed summary + facts; local byte models get raw summary + facts (rendered
    as a `context:` block by _render_local) with NO persona line: identity is
    trained into the model, and a persona in the context channel makes the model
    recite it every turn instead of answering."""
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


def _fit_local_system(model, messages, summary, facts):
    """System text for a local byte model, budgeted so the rendered prompt fits
    the model's trained seq minus the MAX_NEW reply headroom. Over budget:
    shrink fact previews evenly, drop the lowest-scoring facts (retrieve()
    returns best-first), then tail-trim the summary. The chat markers, history,
    and user message are never trimmed."""
    budget = _local_seq(model) - MAX_NEW
    facts = list(facts)
    def overflow():
        return len(_render_local(messages, _system_text("local", summary, facts))) - budget
    while facts and overflow() > 0:
        total = sum(len(f) for f in facts)
        cap = (total - overflow()) // len(facts)
        if cap >= LOCAL_FACT_MIN_CHARS:
            facts = [f[:cap] for f in facts]
            break
        facts.pop()
    over = overflow()
    if over > 0 and summary:
        summary = summary[:max(0, len(summary) - over)]
    return _system_text("local", summary, facts)


def _resolve_route(cfg, model, backend):
    """Pick the completion route for the selected chat model. Returns
    (complete, label, resp_backend, kind). complete(messages, system) -> text and
    raises ChatUnavailable on an expected failure. kind ('remote' | 'local')
    drives how summary + facts are framed."""
    from runtime import settings as settings_mod

    if model == TEACHER_ID or model.startswith(TEACHER_PREFIX):
        from teacher import client as teacher_client
        from teacher import providers as teacher_providers
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
            except ValueError as e:
                raise ChatUnavailable(f"unknown teacher provider: {provider}") from e
            api_key = teacher_providers.resolve_api_key(provider, stored_key)
            if prov.get("requires_key") and not api_key:
                raise ChatUnavailable(f"teacher {provider} needs an API key (set it in Settings)")
            try:
                return teacher_client.complete(provider, model_name, messages,
                                               api_key=api_key, base_url=base_url, system=system)
            except teacher_client.TeacherError as e:
                raise ChatUnavailable(user_error(e, "teacher chat")) from e
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


def _fit_tail(turns, budget):
    """Newest whole turns whose combined content fits `budget` chars, opening on a
    user turn. Keeps at least the final turn even when it alone exceeds budget."""
    kept, used = [], 0
    for m in reversed(turns):
        used += len(m["content"])
        if used > budget and kept:
            break
        kept.insert(0, m)
    if len(kept) > 1 and kept[0]["role"] != "user":
        kept = kept[1:]
    return kept


def _compact(complete, summary, turns, char_limit, kind):
    """Bound the conversation memory to the model's context budget. A local byte
    model cannot summarize its own history, so it slides: keep the newest turns
    that fit char_limit and drop the older ones (no summary). Remote models fold
    the older turns into a running summary, keeping the last CTX_KEEP_TAIL_TURNS
    verbatim. Best-effort: a failed summarize leaves the history intact."""
    if _context_used(summary, turns) <= char_limit:
        return summary, turns
    if kind == "local":
        return "", _fit_tail(turns, char_limit)
    if len(turns) <= CTX_KEEP_TAIL_TURNS:
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


def _openai_messages_in(body):
    """Validate an OpenAI messages array into (conversation, system). system-role
    turns are concatenated into the system string; the rest must be a user/assistant
    sequence ending in a non-empty user turn. Raises ValueError for a 400."""
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty array")
    system_parts, conv = [], []
    for m in raw:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            raise ValueError("each message needs a role and string content")
        role = m.get("role")
        if role == "system":
            if m["content"].strip():
                system_parts.append(m["content"].strip())
        elif role in ("user", "assistant"):
            conv.append({"role": role, "content": m["content"]})
        else:
            raise ValueError(f"unsupported role: {role}")
    if not conv or conv[-1]["role"] != "user" or not conv[-1]["content"].strip():
        raise ValueError("messages must end with a non-empty user turn")
    return conv, "\n\n".join(system_parts)


def _approx_tokens(text):
    """Best-effort token count: byte-level models expose none, so approximate at
    CHARS_PER_TOKEN chars per token (ceil)."""
    return max(0, -(-len(text or "") // CHARS_PER_TOKEN))


def _chatcmpl_id():
    return CHATCMPL_PREFIX + uuid.uuid4().hex


def _openai_completion(answer, conv, system, model):
    prompt_tok = _approx_tokens(system + "".join(m["content"] for m in conv))
    completion_tok = _approx_tokens(answer)
    return {
        "id":      _chatcmpl_id(),
        "object":  OPENAI_CHAT_OBJECT,
        "created": int(time.time()),
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": answer},
            "finish_reason": OPENAI_FINISH_STOP,
        }],
        "usage": {
            "prompt_tokens":     prompt_tok,
            "completion_tokens": completion_tok,
            "total_tokens":      prompt_tok + completion_tok,
        },
    }


def _chunk_frame(cid, created, model, delta, finish_reason):
    payload = {
        "id":      cid,
        "object":  OPENAI_CHUNK_OBJECT,
        "created": created,
        "model":   model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return "data: " + json.dumps(payload) + "\n\n"


def _mri_chunk_frame(cid, created, model, frame, delta=None):
    """A standard chat.completion.chunk carrying one MRI frame under a top-level `mri`
    key, plus an optional assistant text delta for the byte that produced the frame.
    Off-the-shelf OpenAI clients read the content delta and ignore the unknown key;
    MRI-aware clients read chunk['mri']. Keeps every SSE data line a valid chunk so
    interleaving telemetry never breaks a plain client."""
    payload = {
        "id":      cid,
        "object":  OPENAI_CHUNK_OBJECT,
        "created": created,
        "model":   model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": None}],
        MRI_KEY:   frame,
    }
    return "data: " + json.dumps(payload) + "\n\n"


def _openai_stream(complete, conv, system, model):
    """SSE chat.completion.chunk stream: role frame, per-segment content frames, a
    stop frame, then [DONE]. complete() is buffered, so generation errors after the
    stream has opened surface as a single error frame (status is already 200)."""
    def gen():
        created, cid = int(time.time()), _chatcmpl_id()
        try:
            answer = (complete(conv, system) or "").strip()
        except (ChatUnavailable, FileNotFoundError, RuntimeError, OSError, ValueError) as e:
            yield "data: " + json.dumps(
                {"error": {"message": user_error(e), "type": "service_unavailable"}}) + "\n\n"
            yield SSE_DONE
            return
        yield _chunk_frame(cid, created, model, {"role": "assistant"}, None)
        for seg in re.findall(CHAT_STREAM_SPLIT, answer):
            yield _chunk_frame(cid, created, model, {"content": seg}, None)
        yield _chunk_frame(cid, created, model, {}, OPENAI_FINISH_STOP)
        yield SSE_DONE
    return Response(gen(), mimetype="text/event-stream", headers=SSE_HEADERS)


def _local_stream_items(events, emit_frames=False):
    """Yield ('text', delta) segments from a local generation stream (already wrapped
    in _stop_on_bytes) and, when emit_frames is set, ('frame', ev) for each raw event
    as it arrives so MRI telemetry interleaves with the text. Single owner of the
    byte->text incremental UTF-8 decode plus the STOP_HOLD-char holdback that cuts a
    forming stop marker before any of it streams; a trailing flush emits the tail."""
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    text, sent = "", 0
    for ev in events:
        kind = ev.get("kind")
        # Text-before-frame: emit the decoded reply chunk first, THEN the byte's MRI
        # telemetry. The reply char is a few bytes; the MRI frame is tens of KB, so
        # yielding text first lets the answer render instantly while the heavy frame
        # trails. Text (delta.content) and MRI (top-level `mri`) are independent fields
        # on separate chunks, so the client renders each as it arrives; only the
        # interleave order changes, never the data.
        if kind in ("token", "fast_byte"):
            b = ev.get("byte")
            if b is not None:
                text += dec.decode(bytes((int(b) & 0xff,)))
                safe = _sanitize_text(text)
                # Dynamic holdback: only lag when the tail is actually forming
                # a stop marker. Was `len(safe) - STOP_HOLD` (fixed 13-char
                # lag → the streaming was bursty because chunks accumulated
                # 13 chars behind before emitting). Now bytes stream as they
                # decode, and holdback engages only when a stop-marker prefix
                # is actually present.
                emit_to = len(safe) - _dynamic_stop_hold(safe)
                if emit_to > sent:
                    chunk = safe[sent:emit_to]
                    if sent == 0:
                        chunk = chunk.lstrip()
                    if chunk:
                        yield ("text", chunk)
                    sent = emit_to
            if emit_frames:
                yield ("frame", ev)
        elif kind in ("stop", "error"):
            if emit_frames:
                yield ("frame", ev)
            break
        elif emit_frames:
            yield ("frame", ev)
    final = _sanitize_text(text)
    for stop in STOP_MARKERS:
        final = final.split(stop)[0]
    # If the model stopped mid-way through emitting a stop marker (hit max_tokens
    # before completing it), `final` still contains a partial-marker suffix that
    # `str.split(stop)` cannot cut (splits only on COMPLETE markers). Strip that
    # trailing prefix so the user never sees "<|im" or "<|endoftext" leaked at
    # end-of-stream. Discovered 2026-07-20: fresh probe returned "That's not in
    # the passage.<|im" because model hit max_tokens after emitting 4 of the 10
    # bytes of "<|im_end|>". Under the OLD fixed 13-char STOP_HOLD this leaked
    # too; the streaming just delayed the reveal until flush.
    trailing_prefix = _dynamic_stop_hold(final)
    if trailing_prefix:
        final = final[:-trailing_prefix]
    final = final.rstrip()
    if len(final) > sent:
        tail = final[sent:].lstrip() if sent == 0 else final[sent:]
        if tail:
            yield ("text", tail)


def _local_delta_stream(events):
    """Sanitized per-token text deltas (the text-only view of _local_stream_items),
    used by the buffered/streaming chat paths that don't surface MRI frames."""
    for tag, val in _local_stream_items(events):
        if tag == "text":
            yield val


def _log_stream_done(model, chars, started, source="chat", frames=None):
    """Close out a streamed turn. An empty reply is logged as an error: the stream
    succeeded, so nothing else records that the model produced no text -- the symptom
    of a model served with a chat template it was not trained on."""
    took = time.time() - started
    extra = f" frames={frames}" if frames is not None else ""
    if chars:
        logmod.ok(source, f"stream done model={model} chars={chars} in {took:.2f}s{extra}")
    else:
        logmod.error(source, f"stream produced EMPTY reply model={model} in {took:.2f}s{extra} "
                             f"(check the model's chat template matches how it was trained)")


def _openai_stream_local(cfg, model, backend, conv, system, mri=False, gen_params=None):
    """True per-token OpenAI SSE for a local trained model: a role frame, one
    content-delta frame per decoded segment as the model generates it, a stop
    frame, then [DONE]. Drives the same per-token generator /generate uses. RAG is
    not wired into this path: the local model runs on the messages + system only.
    When mri is set, the full-telemetry path runs and each per-byte MRI frame is
    interleaved as its own chunk-extension frame (see _mri_chunk_frame); the text
    content chunks stay standard so off-the-shelf clients are unaffected.

    The overrides are `gen_params`, not `gen`: the SSE body below is `def gen()`,
    which would shadow a parameter of that name and hand _local_events a function."""
    from .backends_routes import _stop_on_bytes

    def gen():
        created, cid = int(time.time()), _chatcmpl_id()
        started, chars = time.time(), 0
        logmod.info("chat", f"stream start model={model} backend={backend} mri={mri} "
                            f"turns={len(conv)} overrides={gen_params or 'box defaults'}")
        try:
            if backend == "c":
                _ensure_c(cfg, model)
            else:
                _ensure_pytorch(cfg, model)
            events, stop_seq = _local_events(cfg, backend, _render_local(conv, system),
                                             mri=mri, gen=gen_params)
            yield _chunk_frame(cid, created, model, {"role": "assistant"}, None)
            for tag, val in _local_stream_items(_stop_on_bytes(events, stop_seq), emit_frames=mri):
                if tag == "frame":
                    yield _mri_chunk_frame(cid, created, model, val)
                else:
                    chars += len(val)
                    yield _chunk_frame(cid, created, model, {"content": val}, None)
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
            logmod.error("chat", f"stream model={model}: {type(e).__name__}: {e}")
            yield "data: " + json.dumps(
                {"error": {"message": user_error(e), "type": "service_unavailable"}}) + "\n\n"
            yield SSE_DONE
            return
        except Exception as e:
            # Anything outside the expected set used to escape the generator, which
            # Flask's error handler cannot catch once a stream is in flight: the caller
            # got a bare 500 and the box logged nothing. Log it and end the stream
            # cleanly instead.
            logmod.error("chat", f"stream model={model} UNHANDLED: "
                                 f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            yield "data: " + json.dumps(
                {"error": {"message": user_error(e), "type": "server_error"}}) + "\n\n"
            yield SSE_DONE
            return
        _log_stream_done(model, chars, started)
        yield _chunk_frame(cid, created, model, {}, OPENAI_FINISH_STOP)
        yield SSE_DONE
    return Response(gen(), mimetype="text/event-stream", headers=SSE_HEADERS)


def _openai_stream_mri(cfg, model, backend, conv, system, gen_params=None):
    """OpenAI SSE for a local model that also emits per-byte MRI telemetry: a role
    frame, then one chat.completion.chunk per MRI frame carrying that frame under `mri`
    plus the assistant text delta for its byte, a stop frame, then [DONE]. Reuses the
    buffered _generate_local_mri (frames already truncated at the turn-stop sequence,
    so no per-byte stop holdback is needed), incrementally decodes each byte to text,
    and drops leading whitespace to match the trimmed non-stream answer.

    The overrides are `gen_params`, not `gen`: the SSE body below is `def gen()`,
    which would shadow a parameter of that name and hand it on as a function."""
    def gen():
        created, cid = int(time.time()), _chatcmpl_id()
        started = time.time()
        logmod.info("mri", f"stream start model={model} backend={backend} turns={len(conv)} "
                           f"overrides={gen_params or 'box defaults'}")
        try:
            _answer, frames = _generate_local_mri(cfg, model, backend, conv, system, gen=gen_params)
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
            logmod.error("mri", f"stream model={model}: {type(e).__name__}: {e}")
            yield "data: " + json.dumps(
                {"error": {"message": user_error(e), "type": "service_unavailable"}}) + "\n\n"
            yield SSE_DONE
            return
        except Exception as e:
            # See _openai_stream_local: an unexpected error inside a live stream is
            # invisible to Flask's handler, so it has to be caught and logged here.
            logmod.error("mri", f"stream model={model} UNHANDLED: "
                                f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            yield "data: " + json.dumps(
                {"error": {"message": user_error(e), "type": "server_error"}}) + "\n\n"
            yield SSE_DONE
            return
        _log_stream_done(model, len(_answer or ""), started, source="mri", frames=len(frames))
        yield _chunk_frame(cid, created, model, {"role": "assistant"}, None)
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        text, sent = "", 0
        for frame in frames:
            delta = None
            if frame.get("kind") in ("token", "fast_byte"):
                b = frame.get("byte")
                if b is not None:
                    text += dec.decode(bytes((int(b) & 0xff,)))
                    safe = _sanitize_text(text)
                    if len(safe) > sent:
                        seg = safe[sent:].lstrip() if sent == 0 else safe[sent:]
                        if seg:
                            delta = {"content": seg}
                        sent = len(safe)
            yield _mri_chunk_frame(cid, created, model, frame, delta)
        yield _chunk_frame(cid, created, model, {}, OPENAI_FINISH_STOP)
        yield SSE_DONE
    return Response(gen(), mimetype="text/event-stream", headers=SSE_HEADERS)


def _openai_mri_completion(answer, frames, conv, system, model):
    """Non-stream MRI response: the standard OpenAI chat.completion plus the full
    per-byte MRI frame list under a top-level `mri` key."""
    return dict(_openai_completion(answer, conv, system, model), **{MRI_KEY: frames})


def _chat_prepare(cfg, body):
    """Resolve a chat request body into a _ChatCtx: message, memory, retrieval,
    and the selected route. None when the message is empty. Shared by the buffered
    and streaming chat endpoints so both ground and route identically."""
    message = (body.get("message") or "").strip()
    if not message:
        return None
    model = (body.get("model") or "").strip()
    backend = (body.get("backend") or "pytorch").strip().lower()
    use_rag = bool(body.get("use_rag"))
    use_logs = bool(body.get("use_logs"))
    mri = bool(body.get(MRI_KEY))
    scope = body.get("kb_scope") if body.get("kb_scope") in KB_SCOPES else "all"
    k = int(body.get("k") or TOP_K)
    history, summary = _history_in(body)

    docs_mode = model == VERITATE_DOCS_ID
    if docs_mode:
        model, use_rag, scope = CLOUD_ID, True, "platform"

    facts, sources = [], []
    if use_rag and has_corpus():
        facts, scores = retrieve(message, k, scope=scope)
        sources = [{"text": t, "score": s} for t, s in zip(facts, scores, strict=True)]
    if docs_mode or use_logs:
        log_facts = training_log_lines()
        facts = facts + log_facts
        sources = sources + [{"text": ln, "score": None} for ln in log_facts]

    complete, label, resp_backend, kind, char_limit = _resolve_route(cfg, model, backend)
    if docs_mode:
        label = "Veritate (platform docs)"
    messages = [*history, {"role": "user", "content": message}]
    system = (_fit_local_system(model, messages, summary, facts)
              if kind == "local" else _system_text(kind, summary, facts))
    return _ChatCtx(message, history, summary, facts, sources, complete, label,
                    resp_backend, kind, char_limit, model, backend, messages, system, mri)


def _chat_result(ctx, answer, frames=None):
    """Post-turn chat payload: fold the new turn into the compacted memory and
    build the response dict returned by /hybrid/chat and the stream's done frame.
    frames (opt-in MRI) attaches the captured per-byte frames under the mri key; the
    stream emits them inline instead, so it passes None here to avoid duplicating them."""
    new_turns = [*ctx.history, {"role": "user", "content": ctx.message},
                 {"role": "assistant", "content": answer}]
    mem_summary, mem_turns = _compact(ctx.complete, ctx.summary, new_turns, ctx.char_limit, ctx.kind)
    out = {"ok": True, "answer": answer, "model": ctx.label, "backend": ctx.resp_backend,
           "confident": bool(ctx.facts), "sources": ctx.sources,
           "memory": {"summary": mem_summary, "turns": mem_turns},
           "context": _context_meter(mem_summary, mem_turns, ctx.char_limit)}
    if frames is not None:
        out[MRI_KEY] = frames
    return out


def _sse(obj):
    return "data: " + json.dumps(obj) + "\n\n"


def _chat_delta_stream(cfg, ctx):
    """Tagged answer segments for the chat stream: ('text', delta) for reply text and,
    when ctx.mri is set on a local model, ('frame', ev) for each per-byte MRI frame.
    Local models decode true per-token via the shared generator, honoring the selected
    engine (c or pytorch); remote models complete() buffered, then chunk word-by-word."""
    if ctx.kind == "local":
        from .backends_routes import _stop_on_bytes
        if ctx.backend == "c":
            _ensure_c(cfg, ctx.model)
        else:
            _ensure_pytorch(cfg, ctx.model)
        events, stop_seq = _local_events(cfg, ctx.backend, _render_local(ctx.messages, ctx.system),
                                         mri=ctx.mri)
        yield from _local_stream_items(_stop_on_bytes(events, stop_seq), emit_frames=ctx.mri)
        return
    answer = (ctx.complete(ctx.messages, ctx.system) or "").strip()
    for seg in re.findall(CHAT_STREAM_SPLIT, answer):
        yield ("text", seg)


def _hybrid_stream_response(cfg, ctx):
    """SSE for /hybrid/chat/stream: per-token `delta` frames as the model
    generates, then one `done` frame carrying the answer, compacted memory,
    context gauge, and sources. Generation errors after the stream opened surface
    as a single `error` frame (status is already 200)."""
    def gen():
        parts = []
        try:
            for tag, val in _chat_delta_stream(cfg, ctx):
                if tag == "frame":
                    yield _sse(val)
                    continue
                parts.append(val)
                yield _sse({"kind": "delta", "text": val})
        except ChatUnavailable as e:
            yield _sse({"kind": "error", "error": str(e)})
            return
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
            yield _sse({"kind": "error", "error": user_error(e)})
            return
        yield _sse(dict(_chat_result(ctx, "".join(parts).strip()), kind="done"))
    return Response(gen(), mimetype="text/event-stream", headers=SSE_HEADERS)


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

    @app.route("/v1/chat/completions", methods=["POST"])
    def openai_chat_completions():
        """OpenAI-compatible chat completions. Wraps the same model routing as
        /hybrid/chat (_resolve_route). On a local model, temperature/max_tokens/top_k
        and the rep_window/rep_penalty/no_repeat_ngram extensions override this box's
        defaults for the turn (see _gen_params_in); omit them to keep the defaults.
        Cloud/teacher routes fix them upstream. Streaming is true per-token for local
        trained models; cloud/teacher streaming chunks the buffered answer. Setting
        mri:true on a local model returns the per-byte MRI frames: interleaved as
        chunk-extension SSE frames when streaming, under a top-level mri key when
        not (remote models expose no telemetry)."""
        cfg = current_app.config
        body = request.get_json(silent=True) or {}
        try:
            conv, system = _openai_messages_in(body)
        except ValueError as e:
            return ({"error": {"message": str(e), "type": "invalid_request_error"}}, 400)
        model = (body.get("model") or CLOUD_ID).strip()
        mri = bool(body.get(MRI_KEY))
        gen = _gen_params_in(body)
        complete, _label, resp_backend, kind, _limit = _resolve_route(cfg, model, _default_local_backend(model))
        if bool(body.get("stream")):
            if kind == "local":
                return _openai_stream_local(cfg, model, resp_backend, conv, system, mri=mri,
                                            gen_params=gen)
            return _openai_stream(complete, conv, system, model)
        if mri and kind == "local":
            try:
                answer, frames = _generate_local_mri(cfg, model, resp_backend, conv, system, gen=gen)
            except (FileNotFoundError, RuntimeError) as e:
                return ({"error": {"message": user_error(e), "type": "service_unavailable"}}, 503)
            except Exception as e:
                return ({"error": {"message": user_error(e), "type": "server_error"}}, 500)
            return dict(_openai_completion(answer, conv, system, model), **{MRI_KEY: frames})
        try:
            answer = (complete(conv, system) or "").strip()
        except ChatUnavailable as e:
            return ({"error": {"message": str(e), "type": "service_unavailable"}}, 503)
        except (FileNotFoundError, RuntimeError) as e:
            return ({"error": {"message": user_error(e), "type": "service_unavailable"}}, 503)
        except Exception as e:
            return ({"error": {"message": user_error(e), "type": "server_error"}}, 500)
        return _openai_completion(answer, conv, system, model)

    @app.route("/v1/chat/mri", methods=["POST"])
    def openai_chat_mri():
        """OpenAI-shaped sibling of /v1/chat/completions that also emits per-byte MRI
        telemetry. Same request body and model routing (_resolve_route); MRI exists only
        for local trained byte models, so a cloud/teacher route returns 400. Streaming
        (default on) interleaves each assistant text delta with its MRI frame in one
        chat.completion.chunk; non-stream returns the completion plus the full frame
        list under `mri`."""
        cfg = current_app.config
        body = request.get_json(silent=True) or {}
        try:
            conv, system = _openai_messages_in(body)
        except ValueError as e:
            return ({"error": {"message": str(e), "type": "invalid_request_error"}}, 400)
        model = (body.get("model") or CLOUD_ID).strip()
        gen = _gen_params_in(body)
        _complete, _label, resp_backend, kind, _limit = _resolve_route(cfg, model, _default_local_backend(model))
        if kind != "local":
            return ({"error": {"message": "mri is available only for local Veritate models",
                               "type": "invalid_request_error"}}, 400)
        if bool(body.get("stream", True)):
            return _openai_stream_mri(cfg, model, resp_backend, conv, system, gen_params=gen)
        try:
            answer, frames = _generate_local_mri(cfg, model, resp_backend, conv, system, gen=gen)
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
            return ({"error": {"message": user_error(e), "type": "service_unavailable"}}, 503)
        return _openai_mri_completion(answer, frames, conv, system, model)

    @app.route("/hybrid/chat", methods=["POST"])
    def hybrid_chat():
        cfg = current_app.config
        ctx = _chat_prepare(cfg, request.get_json(silent=True) or {})
        if ctx is None:
            return ({"ok": False, "error": "empty message"}, 400)
        frames = None
        try:
            if ctx.mri and ctx.kind == "local":
                answer, frames = _generate_local_mri(cfg, ctx.model, ctx.backend,
                                                      ctx.messages, ctx.system)
            else:
                answer = (ctx.complete(ctx.messages, ctx.system) or "").strip()
        except ChatUnavailable as e:
            return ({"ok": False, "error": str(e)}, 503)
        except (FileNotFoundError, RuntimeError) as e:
            return ({"ok": False, "error": user_error(e)}, 503)
        except Exception as e:
            return ({"ok": False, "error": user_error(e)}, 500)
        return _chat_result(ctx, answer, frames)

    @app.route("/hybrid/chat/stream", methods=["POST"])
    def hybrid_chat_stream():
        """Streaming twin of /hybrid/chat: per-token `delta` frames then one
        `done` frame with the answer, memory, context gauge, and sources. Local
        models stream true per-token on the selected engine (c or pytorch)."""
        cfg = current_app.config
        ctx = _chat_prepare(cfg, request.get_json(silent=True) or {})
        if ctx is None:
            return ({"ok": False, "error": "empty message"}, 400)
        return _hybrid_stream_response(cfg, ctx)
