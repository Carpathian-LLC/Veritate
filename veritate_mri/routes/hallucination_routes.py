# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - POST /hallucination/analyze. Drives one (or, when context is present, two)
#   generations through the existing platform paths, assembles the answer bytes
#   with their per-byte confidence stream, and hands them to inference.hallucination
#   for span/grounding/divergence/provenance scoring. Owns no decode loop and pins
#   no model: reuses hybrid_routes framing + backend loaders and backends_routes
#   streaming, the same state the chat page uses.
# - The training-provenance index is built once per process over a bounded sample
#   of trainers/corpus (cheap, similarity-only proxy) and cached in _TRAIN_INDEX.
# veritate_mri/routes/hallucination_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import current_app, request
from inference import hallucination
from inference.decode import (
    NO_REPEAT_NGRAM_DEFAULT,
    REP_PENALTY_DEFAULT,
    REP_WINDOW_DEFAULT,
)
from readers import paths
from runtime import logs as logmod

from . import backends_routes, hybrid_routes
from ._common import user_error

# ------------------------------------------------------------------------------------
# Constants

TOP_K                 = 40
TEMPERATURE_DEFAULT   = 0.7
DIVERGENCE_TEMPERATURE = 0.05   # near-greedy so the no-context answer is stable
MAX_NEW_DEFAULT       = 200

WIRE_NEWLINE_BYTE = ord(hybrid_routes.WIRE_NEWLINE)
NEWLINE_BYTE      = 0x0a
DEL_BYTE          = 0x7f
KEEP_CTRL_BYTES   = frozenset(ord(c) for c in hybrid_routes.KEEP_CTRL_CHARS)
STOP_MARKER_BYTES = tuple(m.encode("utf-8") for m in hybrid_routes.STOP_MARKERS)
WS_BYTES          = frozenset(b" \t\r\n\x0b\x0c")

_TRAIN_INDEX = {}

# ------------------------------------------------------------------------------------
# Functions


def _finalize(raw, conf, surp, ent):
    """Sanitize the raw byte stream (restore the C wire newline, drop control
    bytes) keeping per-byte metrics aligned, cut at the first turn marker, and
    strip edges. Returns (answer, metrics, confidence_source)."""
    clean, metrics, derived = bytearray(), [], False
    for i, byte in enumerate(raw):
        c = NEWLINE_BYTE if byte == WIRE_NEWLINE_BYTE else byte
        if not (c in KEEP_CTRL_BYTES or (c >= 0x20 and c != DEL_BYTE)):
            continue
        cf = conf[i]
        if cf is None:
            derived = True
            cf = hallucination.derive_confidence(surp[i])
        clean.append(c)
        metrics.append({"confidence": float(cf),
                        "surprise_bits": float(surp[i] or 0.0),
                        "entropy_bits": float(ent[i] or 0.0)})
    cut = len(clean)
    for marker in STOP_MARKER_BYTES:
        idx = bytes(clean).find(marker)
        if 0 <= idx < cut:
            cut = idx
    del clean[cut:]
    del metrics[cut:]
    a, b = 0, len(clean)
    while a < b and clean[a] in WS_BYTES:
        a += 1
    while b > a and clean[b - 1] in WS_BYTES:
        b -= 1
    answer = bytes(clean[a:b]).decode("utf-8", "replace")
    source = (hallucination.CONFIDENCE_SOURCE_DERIVED if derived
              else hallucination.CONFIDENCE_SOURCE_BYTE)
    return answer, metrics[a:b], source


def _assemble(events, stop_seq):
    raw, conf, surp, ent = bytearray(), [], [], []
    for ev in backends_routes._stop_on_bytes(events, stop_seq):
        kind = ev.get("kind")
        if kind in ("token", "fast_byte"):
            b = ev.get("byte")
            if b is None:
                continue
            raw.append(int(b) & 0xff)
            conf.append(ev.get("confidence"))
            surp.append(ev.get("surprise_bits"))
            ent.append(ev.get("entropy_bits"))
        elif kind in ("stop", "error"):
            break
    return _finalize(raw, conf, surp, ent)


def _assemble_from_frames(frames):
    """Rebuild (answer, metrics, source) from client-collected stream frames.
    Reuses _finalize so the sanitize/cut/strip matches the live generation path
    exactly: the analysis grades the EXACT answer the user saw, no re-generation,
    no dependence on the model still being loaded."""
    raw, conf, surp, ent = bytearray(), [], [], []
    for fr in frames:
        b = fr.get("byte")
        if b is None:
            continue
        raw.append(int(b) & 0xff)
        conf.append(fr.get("confidence"))
        surp.append(fr.get("surprise_bits"))
        ent.append(fr.get("entropy_bits"))
    return _finalize(raw, conf, surp, ent)


def _run(cfg, backend, prompt, max_new, temperature):
    stop_seq = backends_routes._chat_stop_seq(prompt)
    rep = {"rep_window": REP_WINDOW_DEFAULT, "rep_penalty": REP_PENALTY_DEFAULT,
           "no_repeat_ngram": NO_REPEAT_NGRAM_DEFAULT}
    if backend == "c":
        events = backends_routes._c_engine_stream(cfg, prompt, max_new,
                                                  temperature=temperature, top_k=TOP_K, **rep)
        return _assemble(events, stop_seq)
    brain = cfg["BRAIN"]
    with brain.lock:
        events = brain.stream(prompt, temperature, TOP_K, max_new, **rep)
        return _assemble(events, stop_seq)


def _training_index():
    key = paths.corpus_dir()
    cached = _TRAIN_INDEX.get(key)
    if cached is None:
        index, meta, report = hallucination.build_training_index(key)
        logmod.info("hallucination",
                    f"training provenance sampled {report['n_files']} corpora, "
                    f"{report['chunks']} chunks, {report['bytes']} bytes "
                    f"(cap {report['cap_bytes']})")
        cached = (index, meta)
        _TRAIN_INDEX[key] = cached
    return cached


def register(app):
    @app.route("/hallucination/analyze", methods=["POST"])
    def hallucination_analyze():
        cfg = current_app.config
        body = request.get_json(silent=True) or {}
        model = (body.get("model") or "").strip()
        backend = (body.get("backend") or "pytorch").strip().lower()
        prompt_in = (body.get("prompt") or "").strip()
        message = (body.get("message") or "").strip()
        use_rag = bool(body.get("use_rag"))
        scope = body.get("kb_scope") if body.get("kb_scope") in hybrid_routes.KB_SCOPES else "all"
        k = int(body.get("k") or hybrid_routes.TOP_K)
        max_new = int(body.get("max_new") or MAX_NEW_DEFAULT)
        temperature = float(body.get("temperature") if body.get("temperature") is not None
                            else TEMPERATURE_DEFAULT)

        if not model:
            return ({"ok": False, "error": "model is required"}, 400)
        if not message and not prompt_in:
            return ({"ok": False, "error": "provide a prompt or a message"}, 400)

        # Deferred (robust) mode: the client posts the frames it already streamed,
        # so we grade the exact finished answer with no re-generation and no need
        # for the model to still be loaded. Generation mode (frames absent, e.g.
        # the manual Detect button) re-runs and can also measure divergence.
        provided = body.get("frames")

        if not provided:
            try:
                if backend == "c":
                    hybrid_routes._ensure_c(cfg, model)
                else:
                    hybrid_routes._ensure_pytorch(cfg, model)
            except (FileNotFoundError, RuntimeError) as e:
                return ({"ok": False, "error": user_error(e)}, 503)

        facts, context, plain_prompt = [], [], None
        if message:
            if use_rag and hybrid_routes.has_corpus():
                facts, scores = hybrid_routes.retrieve(message, k, scope=scope)
                context = [{"text": t, "score": s} for t, s in zip(facts, scores, strict=True)]
            main_prompt = (hybrid_routes.build_prompt(message, facts) if facts
                           else hybrid_routes.build_plain_prompt(message))
            plain_prompt = hybrid_routes.build_plain_prompt(message)
        else:
            main_prompt = prompt_in

        divergence = None
        if provided:
            answer, metrics, conf_source = _assemble_from_frames(provided)
        else:
            answer, metrics, conf_source = _run(cfg, backend, main_prompt, max_new, temperature)
            if facts:
                alt_answer, _, _ = _run(cfg, backend, plain_prompt, max_new, DIVERGENCE_TEMPERATURE)
                divergence = hallucination.divergence_score(answer, alt_answer)

        spans, overall_conf = hallucination.segment_spans(answer, metrics)
        context_text = " ".join(facts) if facts else None
        grounded_fraction = hallucination.annotate_grounding(spans, context_text)

        overall = hallucination.build_overall(overall_conf, grounded_fraction, divergence, answer)

        index, meta = _training_index()
        return {
            "ok": True,
            "answer": answer,
            "overall": overall,
            "spans": spans,
            "provenance": {
                "grounded_sources": (hallucination.grounded_sources(spans, context) if facts else []),
                "training_matches": hallucination.training_matches(spans, index, meta),
            },
            "context": context,
            "confidence_source": conf_source,
        }
