# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - backend status, pytorch/c load+unload, neuron lookup, meta, generate sse,
#   addons listing, agent stream. owns the c-engine streaming frame builder
#   and the deferred c subprocess spawn loader; app.main() calls
#   ensure_c_loaded directly for startup-time auto-load gating.
# veritate_mri/routes/backends_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import base64
import json
import math
import os
import threading
import time

import numpy as np
from flask import Response, current_app, request
from inference.addons import build_chain, list_addons
from inference.agent.loop import AgentLoop
from inference.agent.rag import build_rag_prefix, crude_compressor, make_word_ppl_compressor
from inference.agent.tools import build_default_toolbox
from inference.agent.tools.retriever import CORPUS_EXTENSIONS
from inference.agent.tools.retriever import make_tool as _make_rag_tool
from inference.backends.c_engine import CTracedSubprocess
from inference.backends.pytorch import (
    ACTIVATION_INT8_SCALE,
    ADAPTIVE_THRESHOLD_DEFAULT,
    MAX_NEW_DEFAULT,
    NEURON_TOP_K,
    NO_REPEAT_NGRAM_OFF,
    REP_PENALTY_OFF,
    REP_WINDOW_OFF,
    TEMPERATURE_DEFAULT,
    TOP_K_DEFAULT,
    load_memory,
)
from readers import (
    bin as binr,
)
from readers import (
    capabilities as caps_reader,
)
from readers import (
    checkpoints,
    engine,
    models,
    paths,
)
from readers import (
    config as cfg_reader,
)
from runtime import logs as logmod
from runtime import settings as settings_mod
from training import build_runner

from . import _brain
from ._common import auto_thread_count, is_loopback, user_error

# ------------------------------------------------------------------------------------
# Constants

C_BUILD_TIMEOUT_SECS = 600
C_BUILD_POLL_SEC     = 0.5
C_FFN_BUCKET_TARGET  = 256
DLA_TOP_K_CAND       = 12
DLA_TOP_K_LENS       = 3
ATTN_TOP_K           = 6
INFO_FLOW_TOP_K      = 8
PROMPT_PREFIX_CAP    = 8192
RAG_K_MAX            = 16
RAG_K_DEFAULT        = 3
RAG_CACHE_MAX        = 8
AGENT_MAX_TURNS_CAP  = 16
AGENT_BEST_OF_N_CAP  = 8
MAX_NEW_CAP          = 4096
BYTE_VOCAB           = 256
# Softmax temperature applied to raw logits before the dashboard renders a
# probability bar. Fixed scale, not a sampling knob: it only shapes what the
# lens/candidate panels display, so it must be identical on both call sites.
LENS_LOGIT_SCALE     = 8.0
U8_MAX               = 255.0
EPS                  = 1e-9
LOG2_EPS             = 1e-12
NEURON_AFFINITY_TOP_K    = 8
NEURON_PREDECESSOR_TOP_K = 10
NEURON_SUCCESSOR_TOP_K   = 8
AGENT_MAX_TURNS_DEFAULT  = 6
AGENT_BEST_OF_N_DEFAULT  = 1
AGENT_SEED_DEFAULT       = 0
ABLATE_OFF           = -1
DEFAULT_BACKEND      = "c"
SSE_HEADERS          = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
PATH_LOOPBACK_ONLY   = "path params (rag/corpus/fs_root) are restricted to loopback (local) requests"

_VOCAB_PRESETS = {
    "ascii":      set(range(0x20, 0x7f)) | {0x09, 0x0a, 0x0d},
    "alpha":      set(range(0x41, 0x5b)) | set(range(0x61, 0x7b)),
    "lower":      set(range(0x61, 0x7b)),
    "upper":      set(range(0x41, 0x5b)),
    "alnum":      set(range(0x30, 0x3a)) | set(range(0x41, 0x5b)) | set(range(0x61, 0x7b)),
    "digits":     set(range(0x30, 0x3a)),
}

_STOP_PRESETS = {
    "newline":          b"\n",
    "double_newline":   b"\n\n",
    "eos":              b"</s>",
}

CHATML_IM_END      = b"<|im_end|>"
CHATML_IM_START    = b"<|im_start|>"
CHAT_END_TAG       = b"<|end|>"
CHAT_USER_TAG      = b"<|user|>"
CHAT_ASSISTANT_TAG = b"<|assistant|>"
PLATFORM_STOP_MARKERS = (CHAT_END_TAG, CHAT_USER_TAG, CHAT_ASSISTANT_TAG)
CHATML_STOP_MARKERS   = (CHATML_IM_END, CHATML_IM_START)

def _is_chatml_prompt(prompt):
    if not isinstance(prompt, str):
        return False
    return "<|im_start|>" in prompt

def _chat_stop_seq(prompt):
    """Turn-stop markers for a framed chat prompt. A reply never contains a turn
    marker, so platform prompts stop at the first of end/user/assistant tags,
    legacy ChatML at im_end / im_start; plain prompts stream to max_new (None)."""
    if not isinstance(prompt, str):
        return None
    if "<|im_start|>" in prompt:
        return CHATML_STOP_MARKERS
    if "<|assistant|>" in prompt:
        return PLATFORM_STOP_MARKERS
    return None

def _stop_on_bytes(gen, stop_markers):
    """Wrap an SSE-event generator and halt after the byte stream from
    token/fast_byte events ends with ANY sequence in `stop_markers`. Emits a
    synthetic {kind:'stop', reason:<marker>} carrying which marker fired, then
    returns. One rolling tail sized to the longest marker."""
    if not stop_markers:
        for ev in gen:
            yield ev
        return
    if isinstance(stop_markers, bytes):
        stop_markers = (stop_markers,)
    cap = max(len(m) for m in stop_markers)
    tail = bytearray()
    for ev in gen:
        yield ev
        k = ev.get("kind") if isinstance(ev, dict) else None
        if k in ("token", "fast_byte"):
            b = ev.get("byte")
            if isinstance(b, int):
                tail.append(b & 0xff)
                if len(tail) > cap:
                    del tail[:len(tail) - cap]
                for m in stop_markers:
                    if tail.endswith(m):
                        yield {"kind": "stop", "reason": m.decode("ascii", "replace")}
                        return

_RAG_TOOL_CACHE = {}
_RAG_CACHE_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions

def _reduce_full(raw, shape):
    """Compute the four heavy-array reductions from a raw TFRM frame (the Python
    path). Produces the same reduced fields the engine emits directly in a TFRC
    frame, so _assemble_frame is backend-agnostic."""
    n_layers = shape["layers"]
    n_heads  = shape["heads"]
    n_ffn    = shape["ffn"]
    DS       = max(1, n_ffn // C_FFN_BUCKET_TARGET)
    BUCKETS  = n_ffn // DS
    R = raw["real_len"]

    ffn_full, ffn_top, ffn_argmax = [], [], []
    for layer in range(n_layers):
        act = np.abs(raw["ffn_neurons"][layer]).astype(np.float32)
        usable = BUCKETS * DS
        grouped = act[:usable].reshape(BUCKETS, DS)
        bucket_vals = grouped.max(axis=1)
        bucket_argmax = grouped.argmax(axis=1)
        mx = max(EPS, float(bucket_vals.max()))
        u8 = ((bucket_vals / mx) * U8_MAX).clip(0, U8_MAX).astype(np.uint8)
        # base64-pack the per-layer uint8 grids (~2.6x smaller than JSON int arrays,
        # lossless). Frontend decodeFfnField() unpacks; old clients read number[][].
        ffn_full.append(base64.b64encode(u8.tobytes()).decode("ascii"))
        ffn_argmax.append(base64.b64encode(bucket_argmax.astype(np.uint8).tobytes()).decode("ascii"))
        idx = np.argsort(-act)[:NEURON_TOP_K]
        ffn_top.append([{"id": int(i), "v": round(float(act[i]), 3)} for i in idx])

    # info_flow is the only attention product the dashboard renders; the per-head
    # top-k `attn` array is not read by any panel (engine.ts renders the attention
    # view from info_flow), so we accumulate the flow mass but never build/emit attn.
    info_flow_pos = np.zeros(R, dtype=np.float32)
    for layer in range(n_layers):
        for h in range(n_heads):
            w = raw["attention"][layer, h, :R].astype(np.float32)
            s = float(w.sum())
            if s > EPS: w = w / s
            info_flow_pos += w

    flow_max = max(EPS, float(info_flow_pos.max()))
    flow_idx = np.argsort(-info_flow_pos)[:min(INFO_FLOW_TOP_K, R)]
    info_flow = [{"p": int(p), "w": round(float(info_flow_pos[p]) / flow_max, 3)}
                 for p in flow_idx]

    res, contrib = [], []
    for layer in range(n_layers):
        rin  = raw["residual_pre"][layer].astype(np.float32)
        rout = raw["residual_post"][layer].astype(np.float32)
        res.append(round(float(np.linalg.norm(rout)), 3))
        contrib.append(round(float(np.linalg.norm(rout - rin)), 3))

    lens = []
    for layer in range(n_layers):
        ll = raw["lens_logits"][layer].astype(np.float64)
        mx = max(1.0, float(np.abs(ll).max()))
        scaled = ll / mx * LENS_LOGIT_SCALE
        e = np.exp(scaled - scaled.max())
        probs = e / e.sum()
        top_idx = np.argsort(-probs)[:DLA_TOP_K_LENS]
        lens.append([{"b": int(b), "p": round(float(probs[b]), 3)} for b in top_idx])

    return {"ffn_full": ffn_full, "ffn_argmax": ffn_argmax, "ffn_top": ffn_top,
            "ffn_downsample": DS, "info_flow": info_flow,
            "res": res, "contrib": contrib, "lens": lens}


def _reduce_compact(raw, shape):
    """Assemble the reduced fields from a TFRC frame, where the engine already did
    the heavy reductions. Only light formatting/rounding + base64-packing remains, so
    per-byte numpy work on the box drops to near zero."""
    n_ffn = shape["ffn"]
    DS    = max(1, n_ffn // C_FFN_BUCKET_TARGET)
    ffn_full   = [base64.b64encode(a.tobytes()).decode("ascii") for a in raw["ffn_full"]]
    ffn_argmax = [base64.b64encode(a.tobytes()).decode("ascii") for a in raw["ffn_argmax"]]
    ffn_top = [[{"id": int(e["id"]), "v": round(float(e["v"]), 3)} for e in layer]
               for layer in raw["ffn_top"]]
    info_flow = [{"p": int(e["p"]), "w": round(float(e["w"]), 3)} for e in raw["info_flow"]]
    res     = [round(float(x), 3) for x in raw["res"]]
    contrib = [round(float(x), 3) for x in raw["contrib"]]
    lens = [[{"b": int(e["b"]), "p": round(float(e["p"]), 3)} for e in layer]
            for layer in raw["lens"]]
    return {"ffn_full": ffn_full, "ffn_argmax": ffn_argmax, "ffn_top": ffn_top,
            "ffn_downsample": DS, "info_flow": info_flow,
            "res": res, "contrib": contrib, "lens": lens}


def _assemble_frame(raw, reduced, fwd_ms, shape):
    """Combine the (backend-agnostic) reduced heavy-array fields with the decision-trace
    trailer (cand/entropy from logits, DLA formatting, confidence, ablation) into the
    single frame schema the dashboard consumes. Shared by the full and compact paths."""
    n_layers = shape["layers"]
    R = raw["real_len"]

    logits = raw["logits"].astype(np.float64)
    mx = max(1.0, float(np.abs(logits).max()))
    scaled = logits / mx * LENS_LOGIT_SCALE
    e = np.exp(scaled - scaled.max())
    probs = e / e.sum()
    top_idx = np.argsort(-probs)[:DLA_TOP_K_CAND]
    cand = [{"b": int(b), "p": round(float(probs[b]), 3)} for b in top_idx]
    sampled = raw["byte"]
    argmax_byte = int(raw.get("argmax_byte", int(np.argmax(logits))))
    entropy_bits  = float(-(probs * np.log2(probs + LOG2_EPS)).sum())
    surprise_bits = float(-math.log2(float(probs[sampled]) + LOG2_EPS))

    decisiveness = [round(float(x), 3) for x in raw.get("decisiveness", np.zeros(n_layers)).tolist()]

    bd_scale = raw.get("bd_scale", np.ones(n_layers, dtype=np.float32))
    def _dla_to_json(entries):
        out = []
        for e in entries:
            L = int(e["layer"]); n = int(e["neuron"])
            a_int = int(e["act"]); w_int = int(e["w"]); c_int = int(e["contrib"])
            scale = float(bd_scale[L]) if 0 <= L < n_layers else 1.0
            act_f = a_int / ACTIVATION_INT8_SCALE
            w_f   = w_int * scale
            ctb_f = (c_int * scale) / ACTIVATION_INT8_SCALE
            out.append({
                "layer": L, "neuron": n,
                "act":     round(act_f, 4),
                "w":       round(w_f,   5),
                "contrib": round(ctb_f, 4),
            })
        return out

    raw_dla_cand = raw.get("dla_cand")
    raw_cand_bytes = raw.get("cand_bytes")
    if raw_dla_cand is not None and raw_cand_bytes is not None:
        dla_cand_json = []
        for i in range(len(raw_cand_bytes)):
            dla_cand_json.append({
                "b": int(raw_cand_bytes[i]),
                "entries": _dla_to_json(raw_dla_cand[i]),
            })
    else:
        dla_cand_json = []
    ablation_layer  = int(raw.get("ablation_layer",  -1))
    ablation_neuron = int(raw.get("ablation_neuron", -1))

    return {
        "kind": "token",
        "byte": sampled,
        "argmax_byte": argmax_byte,
        "T": R,
        "fwd_ms": round(fwd_ms, 2),
        "entropy_bits": round(entropy_bits, 3),
        "surprise_bits": round(surprise_bits, 3),
        "ffn_full": reduced["ffn_full"], "ffn_top": reduced["ffn_top"],
        "ffn_argmax": reduced["ffn_argmax"], "ffn_downsample": reduced["ffn_downsample"],
        "decisiveness": decisiveness,
        "dla_picked": _dla_to_json(raw.get("dla_picked", [])),
        "dla_argmax": _dla_to_json(raw.get("dla_argmax", [])),
        "dla_cand":   dla_cand_json,
        "ablation":   ({"layer": ablation_layer, "neuron": ablation_neuron}
                       if ablation_layer >= 0 and ablation_neuron >= 0 else None),
        "margin":           round(float(raw.get("margin", 0.0)),           4),
        "entropy":          round(float(raw.get("entropy", 0.0)),          4),
        "lens_consistency": round(float(raw.get("lens_consistency", 0.0)), 4),
        "residual_stab":    round(float(raw.get("residual_stab", 0.0)),    4),
        "confidence":       round(float(raw.get("confidence", 0.0)),       4),
        "info_flow": reduced["info_flow"],
        "res": reduced["res"], "contrib": reduced["contrib"],
        "lens": reduced["lens"],
        "cand": cand,
        "memory": [],
        "backend": "c",
    }


def _build_c_mri_frame(raw, fwd_ms, shape):
    """Raw TFRM path: reduce the heavy arrays in Python, then assemble."""
    return _assemble_frame(raw, _reduce_full(raw, shape), fwd_ms, shape)


def _build_c_mri_frame_compact(raw, fwd_ms, shape):
    """Compact TFRC path: the engine already reduced the heavy arrays; just format
    and assemble. Same output schema as the full path, ~30x smaller wire frame."""
    return _assemble_frame(raw, _reduce_compact(raw, shape), fwd_ms, shape)


def _c_engine_stream(cfg, prompt, max_new, temperature=TEMPERATURE_DEFAULT, top_k=TOP_K_DEFAULT,
                     ablate_layer=ABLATE_OFF, ablate_neuron=ABLATE_OFF, addons_csv="",
                     rep_window=REP_WINDOW_OFF, rep_penalty=REP_PENALTY_OFF,
                     no_repeat_ngram=NO_REPEAT_NGRAM_OFF, trace=False):
    sub = cfg["C_SUBPROCESS"]
    if sub is None:
        yield {"kind": "error", "message": "c chat_traced subprocess not running"}
        return
    model_path = cfg["C_MODEL"]
    exe        = cfg["C_EXE"]
    model_name = os.path.basename(os.path.dirname(model_path)) if model_path else "(random)"
    shape = sub.shape
    ds = max(1, shape["ffn"] // C_FFN_BUCKET_TARGET)
    yield {
        "kind": "meta",
        "checkpoint": model_name,
        "n_params": 0,
        "layers": shape["layers"], "heads": shape["heads"], "ffn": shape["ffn"],
        "ffn_buckets": shape["ffn"] // ds,
        "vocab": shape["vocab"], "seq": shape["seq"], "hidden": shape["hidden"],
        "has_memory": False,
        "prompt": prompt,
        "prompt_bytes": list(prompt.encode("utf-8")),
        "backend": "c",
        "c_exe": os.path.basename(exe) if exe else None,
        "c_exe_path": exe,
        "c_model": os.path.basename(model_path) if model_path else None,
        "c_model_dir": model_name,
        "c_model_path": model_path,
    }
    # Compact TFRC frames are opt-in from Settings -> Advanced (persisted in
    # mri_settings.json, survives deploys). Read per stream so a GUI toggle takes
    # effect on the next chat with no restart.
    want_compact = bool(trace and settings_mod.get().get("mri_compact_frames", False))
    try:
        last = time.perf_counter()
        # Hand the same turn markers to the engine. Without them Python stops at the
        # marker while the engine keeps generating to max_new, so the abandoned
        # generator has to be drained and the subprocess respawned, discarding its
        # prefix state cache. With them the engine ends the turn and emits TEND.
        for raw in sub.stream(prompt, temperature, top_k, max_new,
                              ablate_layer=ablate_layer, ablate_neuron=ablate_neuron,
                              addons_csv=addons_csv, rep_window=rep_window,
                              rep_penalty=rep_penalty, no_repeat_ngram=no_repeat_ngram,
                              do_trace=trace, compact=want_compact,
                              stop_sequences=_chat_stop_seq(prompt) or ()):
            now = time.perf_counter()
            fwd_ms = (now - last) * 1000.0
            last = now
            if raw.get("fast"):
                yield {"kind": "fast_byte", "byte": int(raw["byte"]),
                       "argmax_byte": int(raw["argmax_byte"]), "T": int(raw["real_len"]),
                       "ms_per_byte": round(fwd_ms, 2), "backend": "c"}
            elif raw.get("compact"):
                yield _build_c_mri_frame_compact(raw, fwd_ms, shape)
            else:
                yield _build_c_mri_frame(raw, fwd_ms, shape)
    except Exception as e:
        yield {"kind": "error", "message": f"c stream: {e}"}


def _spawn_c_subprocess(cfg, exe, model):
    try:
        sub = CTracedSubprocess(exe, model)
    except Exception as e:
        logmod.error("backends", f"c engine spawn failed: {e}")
        cfg["C_PENDING"] = False
        return False
    cfg["C_SUBPROCESS"] = sub
    cfg["C_EXE"]        = exe
    cfg["C_MODEL"]      = model
    cfg["C_PENDING"]    = False
    logmod.ok("backends", f"c engine loaded (pid {sub.proc.pid})")
    return True


# Warm pool: model_name -> CTracedSubprocess kept resident so a switch to that
# model re-points the active C slot instead of spawning. cfg["C_WARM"] owns it.

def _sub_alive(sub):
    return sub is not None and sub.proc is not None and sub.proc.poll() is None


def _warm_pool(cfg):
    pool = cfg.get("C_WARM")
    if pool is None:
        pool = {}
        cfg["C_WARM"] = pool
    return pool


def warm_is_pinned(cfg, sub):
    return sub is not None and sub in _warm_pool(cfg).values()


def warm_spawn(cfg, name):
    """Spawn `name` into the warm pool (idempotent; revives a died entry). Skips with a
    plain log line and returns False when the model has no .bin or the engine binary
    is not built."""
    pool = _warm_pool(cfg)
    existing = pool.get(name)
    if existing is not None:
        existing._ensure_alive()
        return True
    exe = paths.engine_binary_path()
    if not binr.exists(name) or not os.path.isfile(exe):
        logmod.info("backends", f"warm skip {name}: no veritate.bin or engine binary not built")
        return False
    model_bin = os.path.abspath(paths.bin_path(name))
    active = cfg.get("C_SUBPROCESS")
    if _sub_alive(active) and os.path.abspath(active.model_path or "") == model_bin:
        pool[name] = active
        logmod.ok("backends", f"warm model resident (adopted active): {name}")
        return True
    try:
        sub = CTracedSubprocess(exe, model_bin)
    except Exception as e:
        logmod.warn("backends", f"warm spawn {name} failed: {e}")
        return False
    pool[name] = sub
    logmod.ok("backends", f"warm model resident: {name} (pid {sub.proc.pid})")
    return True


def warm_drop(cfg, name):
    """Close + remove a warm model, unless it is the active C_SUBPROCESS: then unpin
    only and leave the live subprocess for the normal single-slot lifecycle."""
    sub = _warm_pool(cfg).pop(name, None)
    if sub is None:
        return
    if sub is cfg.get("C_SUBPROCESS"):
        logmod.info("backends", f"warm unpinned (active, kept live): {name}")
        return
    try:
        sub.close()
    except Exception as e:
        logmod.error("backends", f"warm close {name}: {e}")
    logmod.ok("backends", f"warm model released: {name}")


def warm_select(cfg, name):
    """Point the active C slot at a warm-pinned model without spawning; revives a
    died entry. Returns True when `name` was warm and is now selected."""
    sub = _warm_pool(cfg).get(name)
    if sub is None:
        return False
    sub._ensure_alive()
    cfg["C_SUBPROCESS"] = sub
    cfg["C_EXE"]        = sub.exe
    cfg["C_MODEL"]      = sub.model_path
    cfg["C_PENDING"]    = False
    return True


def warm_forget(cfg, sub):
    """Drop a subprocess object from the pool by identity (used when it is closed
    through another path)."""
    pool = _warm_pool(cfg)
    for name in [n for n, s in pool.items() if s is sub]:
        pool.pop(name, None)


def warm_apply(cfg, names):
    """Reconcile the pool to `names`: spawn newly added, drop removed."""
    want = list(dict.fromkeys(names))
    pool = _warm_pool(cfg)
    for name in want:
        if name not in pool:
            warm_spawn(cfg, name)
    for name in list(pool):
        if name not in want:
            warm_drop(cfg, name)


def warm_eager_start(cfg, names):
    """Spawn every warm model off the main thread so startup is not serially blocked."""
    def worker():
        for name in dict.fromkeys(names):
            warm_spawn(cfg, name)
    threading.Thread(target=worker, name="c-warm-loader", daemon=True).start()


def _warm_status(cfg):
    pool = cfg.get("C_WARM") or {}
    active = cfg.get("C_SUBPROCESS")
    pinned = set(settings_mod.get().get("warm_models") or [])
    out = []
    for n in models.list_models():
        if not binr.exists(n):
            continue
        sub = pool.get(n)
        try:
            size = os.path.getsize(paths.bin_path(n))
        except OSError:
            size = 0
        out.append({
            "name": n,
            "bin_bytes": size,
            "pinned": n in pinned,
            "resident": _sub_alive(sub),
            "active": sub is not None and sub is active,
        })
    return out


def ensure_c_loaded(cfg, model_override=None):
    def worker():
        target_bin = paths.engine_binary_path()
        if not os.path.isfile(target_bin):
            logmod.info("backends", "c engine binary missing; triggering auto-build")
            build_runner.start()
        deadline = time.time() + C_BUILD_TIMEOUT_SECS
        waited_for_build = False
        while True:
            s = build_runner.state()
            status = s["status"]
            if status == build_runner.STATUS_BUILDING:
                if not waited_for_build:
                    logmod.info("backends", "build in progress; deferring c engine spawn until it settles")
                    waited_for_build = True
                if time.time() > deadline:
                    logmod.error("backends", "auto-build timed out after 10 min")
                    cfg["C_PENDING"] = False
                    return
                time.sleep(C_BUILD_POLL_SEC)
                continue
            if status == build_runner.STATUS_FAILED:
                logmod.error("backends", f"build failed; not spawning c engine: {s.get('error') or 'no error'}")
                cfg["C_PENDING"] = False
                return
            break
        if not os.path.isfile(target_bin):
            logmod.error("backends", f"engine binary missing after build settled: {target_bin}")
            cfg["C_PENDING"] = False
            return
        exe   = paths.engine_binary_path()
        # model_override is a bin path or a model NAME; resolve names explicitly
        # rather than falling back to some other model's newest bin.
        if model_override:
            model = model_override if os.path.isfile(model_override) \
                else (paths.bin_path(model_override) if binr.exists(model_override) else None)
            if not model:
                logmod.error("backends", f"no veritate.bin under models/{model_override}; export it first")
                cfg["C_PENDING"] = False
                return
        else:
            model = _brain.resolve_c_model_bin(None)
        if not model or not os.path.isfile(model):
            logmod.error("backends", "no veritate.bin under any model; train + export one first")
            cfg["C_PENDING"] = False
            return
        model_dir = os.path.basename(os.path.dirname(model))
        boost = binr.act_boost(model_dir)
        qat = cfg_reader.qat_enabled(model_dir)
        if boost is not None and boost > 1 and not qat:
            logmod.warn("backends", f"c engine: {model_dir} act_boost={boost} and config.qat_enabled "
                                    "is not set; output may be gibberish")
        cfg["C_BLOCKED_REASON"] = None
        cfg["C_BLOCKED_MODEL"]  = None
        _spawn_c_subprocess(cfg, exe, model)
    cfg["C_PENDING"] = True
    threading.Thread(target=worker, name="c-backend-loader", daemon=True).start()


def _backends_status_payload(cfg):
    cur_exe   = cfg.get("C_EXE")
    cur_model = cfg.get("C_MODEL")
    try:
        bins_available = sum(1 for n in models.list_models() if binr.exists(n))
    except Exception:
        bins_available = 0
    return {
        "pytorch": {
            "loaded":  cfg.get("BRAIN") is not None,
            "pending": bool(cfg.get("PYTORCH_PENDING")),
            "model":   cfg.get("BRAIN_MODEL") or cfg.get("DEFAULT_MODEL"),
            "step":    cfg.get("BRAIN_STEP")  or cfg.get("DEFAULT_STEP"),
            "last_error": cfg.get("BRAIN_LAST_ERROR"),
        },
        "c": {
            "loaded":    cfg.get("C_SUBPROCESS") is not None,
            "pending":   bool(cfg.get("C_PENDING")),
            "exe":       cur_exe,
            "model_bin": cur_model,
            "model_dir": (os.path.basename(os.path.dirname(cur_model)) if cur_model else None),
            "blocked_reason": cfg.get("C_BLOCKED_REASON"),
            "blocked_model":  cfg.get("C_BLOCKED_MODEL"),
            "build":     build_runner.state(),
            "bins_available": bins_available,
            "warm":      _warm_status(cfg),
        },
    }


def _rag_path_signature(path):
    """Stable signature of a corpus path: (max_mtime, total_bytes) over the
    text files we'd index. Cheap to compute; invalidates on any edit."""
    if os.path.isfile(path):
        st = os.stat(path)
        return (st.st_mtime, st.st_size)
    max_mt = 0.0
    total  = 0
    for dirpath, _, fnames in os.walk(path):
        for fn in fnames:
            if not fn.lower().endswith(CORPUS_EXTENSIONS):
                continue
            try:
                st = os.stat(os.path.join(dirpath, fn))
            except OSError:
                continue
            if st.st_mtime > max_mt:
                max_mt = st.st_mtime
            total += st.st_size
    return (max_mt, total)


def _get_rag_tool(corpus_path):
    """Cached BM25 tool for corpus_path. Validates, normalizes, and rebuilds
    on disk-edit detection."""
    abs_path = os.path.abspath(os.path.expanduser(corpus_path))
    if not os.path.exists(abs_path):
        raise ValueError(f"rag corpus path does not exist: {abs_path}")
    sig = _rag_path_signature(abs_path)
    key = (abs_path, sig)
    with _RAG_CACHE_LOCK:
        tool = _RAG_TOOL_CACHE.get(key)
        if tool is not None:
            return tool, abs_path
        tool = _make_rag_tool(abs_path)
        _RAG_TOOL_CACHE[key] = tool
        for k in list(_RAG_TOOL_CACHE.keys()):
            if k != key and k[0] == abs_path:
                _RAG_TOOL_CACHE.pop(k, None)
        while len(_RAG_TOOL_CACHE) > RAG_CACHE_MAX:
            _RAG_TOOL_CACHE.pop(next(iter(_RAG_TOOL_CACHE)))
    return tool, abs_path


def _parse_rag_hits(formatted):
    if not formatted or formatted.startswith("error") or formatted == "no matches":
        return [], []
    passages, meta = [], []
    for block in formatted.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        src = ""
        if block.startswith("[") and "]" in block:
            head, _, rest = block[1:].partition("]")
            src = head.strip()
            block = rest.strip()
        score = 0.0
        if block.startswith("(score "):
            score_s, _, body = block[len("(score "):].partition(") ")
            try:
                score = float(score_s)
            except ValueError:
                pass
            block = body
        passages.append(block)
        meta.append({"src": src, "score": score,
                     "preview": (block[:120] + ("..." if len(block) > 120 else ""))})
    return passages, meta


def _build_constraint(spec):
    from inference.decode import JSONConstraint, StopOnConstraint, VocabConstraint
    s = spec.strip()
    if s == "json":
        return JSONConstraint()
    if s.startswith("vocab:"):
        name = s[len("vocab:"):].strip()
        if name not in _VOCAB_PRESETS:
            raise ValueError(f"unknown vocab preset {name!r}; allowed: {sorted(_VOCAB_PRESETS)}")
        return VocabConstraint(_VOCAB_PRESETS[name])
    if s.startswith("stop:"):
        rest = s[len("stop:"):]
        if rest.startswith("text:"):
            return StopOnConstraint(rest[len("text:"):].encode("utf-8"))
        if rest not in _STOP_PRESETS:
            raise ValueError(f"unknown stop preset {rest!r}; allowed: {sorted(_STOP_PRESETS)}")
        return StopOnConstraint(_STOP_PRESETS[rest])
    raise ValueError(f"unknown constrained spec: {spec!r}")


def _sse_error_response(message):
    """SSE Response: one error frame + terminal done event. /generate is consumed
    by an EventSource, which reports a bare 500 as an unparseable non-stream body;
    this keeps the failure visible as a normal stream event."""
    def stream_err():
        yield "data: " + json.dumps({"kind": "error", "message": message}) + "\n\n"
        yield "event: done\ndata: {}\n\n"
    return Response(stream_err(), mimetype="text/event-stream", headers=SSE_HEADERS)


def register(app):
    @app.route("/backends")
    def backends_status():
        return _backends_status_payload(current_app.config)

    @app.route("/backends/pytorch", methods=["POST"])
    def backends_pytorch():
        cfg = current_app.config
        body = request.get_json(silent=True) or {}
        action = (body.get("action") or "").lower()
        if action == "unload":
            if cfg.get("BRAIN") is not None:
                cfg["BRAIN"] = None
                cfg["BRAIN_MODEL"] = None
                cfg["BRAIN_STEP"]  = None
                logmod.ok("backends", "pytorch unloaded")
            return _backends_status_payload(cfg)
        if action == "load":
            body_model = body.get("model")
            body_step  = body.get("step")
            did_swap   = False
            if cfg.get("BRAIN") is not None:
                cur_m = cfg.get("BRAIN_MODEL")
                cur_s = cfg.get("BRAIN_STEP")
                same  = (not body_model) or (
                    body_model == cur_m and
                    (body_step is None or int(body_step) == int(cur_s or 0))
                )
                if same:
                    return _backends_status_payload(cfg)
                logmod.info("backends", f"pytorch swap: {cur_m} step {cur_s} -> "
                                        f"{body_model} step {body_step or 'latest'}")
                cfg["BRAIN"] = None
                cfg["BRAIN_MODEL"] = None
                cfg["BRAIN_STEP"]  = None
                did_swap = True
            name = body_model or cfg.get("DEFAULT_MODEL")
            if not name or not models.exists(name):
                name = _brain.resolve_pytorch_model("auto")
                if name is not None:
                    cfg["DEFAULT_MODEL"] = name
                    cfg["DEFAULT_STEP"]  = checkpoints.latest_step(name)
            if not name or not models.exists(name):
                return ({"ok": False, "error": "no models with checkpoints under models/. "
                                               "train one first or pass an explicit model name."}, 400)
            step = body_step or (None if did_swap else cfg.get("DEFAULT_STEP")) or checkpoints.latest_step(name)
            if step is None:
                return ({"ok": False, "error": f"no checkpoints under models/{name}/"}, 400)
            threads = int(body.get("threads") or cfg.get("DEFAULT_THREADS") or auto_thread_count())
            try:
                brain, name, step = _brain.load_pytorch_brain(name, step, threads)
            except Exception as e:
                msg = user_error(e)
                cfg["BRAIN_LAST_ERROR"] = msg
                if isinstance(e, RuntimeError) and "PyTorch inference is not enabled" in str(e):
                    logmod.warn("backends", f"pytorch: no vanilla checkpoints found (default '{name}' "
                                            "is non-vanilla and fallback search yielded nothing)")
                    return ({"ok": False, "error": msg, "reason": "non_vanilla"}, 400)
                logmod.error("backends", f"pytorch load failed: {type(e).__name__}: {e}")
                return ({"ok": False, "error": msg}, 500)
            cfg["BRAIN"] = brain
            cfg["BRAIN_MODEL"] = name
            cfg["BRAIN_STEP"]  = int(step)
            cfg["DEFAULT_MODEL"] = name
            cfg["DEFAULT_STEP"]  = int(step)
            cfg["BRAIN_LAST_USED"] = time.time()
            cfg["BRAIN_LAST_ERROR"] = None
            logmod.ok("backends", f"pytorch loaded: {name} step {step} ({brain.n_params:,} params)")
            return _backends_status_payload(cfg)
        return ({"ok": False, "error": "action must be load or unload"}, 400)

    @app.route("/backends/c", methods=["POST"])
    def backends_c():
        cfg = current_app.config
        body = request.get_json(silent=True) or {}
        action = (body.get("action") or "").lower()
        if action == "unload":
            sub = cfg.get("C_SUBPROCESS")
            if sub is not None:
                try:
                    sub.close()
                except Exception as e:
                    logmod.error("backends", f"c subprocess close: {type(e).__name__}: {e}")
                warm_forget(cfg, sub)
                cfg["C_SUBPROCESS"] = None
                cfg["C_EXE"] = None
                cfg["C_MODEL"] = None
                logmod.ok("backends", "c engine unloaded")
            return _backends_status_payload(cfg)
        if action == "load":
            if cfg.get("C_SUBPROCESS") is not None:
                return _backends_status_payload(cfg)
            if cfg.get("C_PENDING"):
                return _backends_status_payload(cfg)
            ensure_c_loaded(cfg, model_override=body.get("model"))
            return _backends_status_payload(cfg)
        return ({"ok": False, "error": "action must be load or unload"}, 400)

    @app.route("/neuron/<int:layer>/<int:nid>")
    def neuron_lookup(layer, nid):
        cfg = current_app.config
        brain = cfg.get("BRAIN")
        name  = cfg.get("BRAIN_MODEL") or cfg.get("DEFAULT_MODEL")
        stories = []
        if name:
            mem = load_memory(_brain.neuron_memory_path(name))
            if mem is not None:
                stories = (mem.get(str(layer), {}) or {}).get(str(nid), []) or []
        if brain is None:
            return {
                "layer": layer, "neuron": nid,
                "stories": stories,
                "affinity": None, "predecessors": [], "successors": [],
                "stats": {}, "label": None,
                "pytorch_loaded": False,
                "pytorch_last_error": cfg.get("BRAIN_LAST_ERROR"),
            }
        affinity, predecessors, successors, stats = None, [], [], {}
        label = brain.label_for(layer, nid) if hasattr(brain, "label_for") else None
        cfg["BRAIN_LAST_USED"] = time.time()
        try:
            with brain.lock:
                affinity     = brain.neuron_byte_affinity(layer, nid, top_k=NEURON_AFFINITY_TOP_K)
                predecessors = brain.neuron_predecessors(layer, nid, top_k=NEURON_PREDECESSOR_TOP_K)
                successors   = brain.neuron_successors(layer, nid, top_k=NEURON_SUCCESSOR_TOP_K)
                stats        = brain.neuron_stats(layer, nid)
        except Exception as e:
            logmod.error("neuron", f"layer={layer} nid={nid}: {e}")
        return {"layer": layer, "neuron": nid, "stories": stories,
                "affinity": affinity, "predecessors": predecessors,
                "successors": successors, "stats": stats, "label": label,
                "pytorch_loaded": True}

    @app.route("/meta")
    def meta():
        cfg = current_app.config
        brain = cfg["BRAIN"]
        cur_exe = cfg.get("C_EXE")
        cur_abs = os.path.abspath(cur_exe) if cur_exe else None
        eng = engine.by_path(cur_abs) if cur_abs else None
        c_model_path = cfg.get("C_MODEL")
        c_precision, c_version = ("?", 0)
        c_training, c_activation = ("", "")
        c_model_dir = None
        c_description = ""
        c_act_boost = None
        c_qat_enabled = False
        c_capabilities = None
        if c_model_path and os.path.isfile(c_model_path):
            c_model_dir = os.path.basename(os.path.dirname(c_model_path))
            c_precision, c_version = binr.header(c_model_dir)
            c_training, c_activation = cfg_reader.training_kind(c_model_dir)
            c_description = cfg_reader.description(c_model_dir) or ""
            c_act_boost = binr.act_boost(c_model_dir)
            c_qat_enabled = cfg_reader.qat_enabled(c_model_dir)
            c_capabilities = caps_reader.read(c_model_dir)
        brain_name = cfg.get("BRAIN_MODEL")
        pytorch_capabilities = caps_reader.read(brain_name) if brain_name else None
        return {
            "checkpoint": brain.checkpoint if brain else None,
            "n_params":   brain.n_params if brain else 0,
            "pytorch_device": str(brain.device) if brain and getattr(brain, "device", None) is not None else None,
            "layers": brain.model.layers if brain else 0,
            "heads":  brain.model.heads if brain else 0,
            "ffn":    brain.model.ffn if brain else 0,
            "vocab":  brain.model.vocab if brain else 0,
            "seq":    brain.model.seq if brain else 0,
            "hidden": brain.model.hidden if brain else 0,
            "has_memory": brain.memory is not None if brain else False,
            "prompt_bytes": [],
            "c_backend_available": cur_exe is not None,
            "c_exe":      os.path.basename(cur_exe) if cur_exe else None,
            "c_exe_path": cur_exe,
            "c_engine_version": eng["version"] if eng else None,
            "c_model":      os.path.basename(c_model_path) if c_model_path else None,
            "c_model_dir":  c_model_dir,
            "c_model_path": c_model_path,
            "c_model_precision":   c_precision,
            "c_model_bin_version": c_version,
            "c_model_training":    c_training,
            "c_model_activation":  c_activation,
            "c_model_description": c_description,
            "c_model_act_boost":   c_act_boost,
            "c_model_qat_enabled": c_qat_enabled,
            "c_model_capabilities": c_capabilities,
            "pytorch_model": brain_name,
            "pytorch_capabilities": pytorch_capabilities,
        }

    @app.route("/addons")
    def addons_list():
        try:
            return {"addons": list_addons()}
        except Exception as e:
            logmod.error("addons", f"list failed: {type(e).__name__}: {e}")
            return ({"error": user_error(e)}, 500)

    @app.route("/generate")
    def generate():
        cfg = current_app.config
        prompt        = request.args.get("prompt", "")
        backend       = request.args.get("backend", DEFAULT_BACKEND).lower()
        addons_csv    = request.args.get("addons", "")
        addons_sel    = [s.strip() for s in addons_csv.split(",") if s.strip()]
        fast_mode     = (request.args.get("fast", "") or "").strip().lower()
        constrained_v = (request.args.get("constrained", "") or "").strip()
        try:
            temperature   = float(request.args.get("temperature", TEMPERATURE_DEFAULT))
            top_k         = int(request.args.get("top_k", TOP_K_DEFAULT))
            max_new       = int(request.args.get("max_new", MAX_NEW_DEFAULT))
            ablate_layer  = int(request.args.get("ablate_layer",  ABLATE_OFF))
            ablate_neuron = int(request.args.get("ablate_neuron", ABLATE_OFF))
        except (TypeError, ValueError) as e:
            return _sse_error_response(user_error(e, "bad query param"))
        top_k   = max(1, min(top_k, BYTE_VOCAB))
        max_new = max(1, min(max_new, MAX_NEW_CAP))
        try:
            adaptive_threshold = float(request.args.get("adaptive_threshold", ADAPTIVE_THRESHOLD_DEFAULT))
        except ValueError:
            adaptive_threshold = ADAPTIVE_THRESHOLD_DEFAULT
        adaptive_threshold = max(0.0, min(1.0, adaptive_threshold))
        # Repetition control (chat decode). Absent params default OFF.
        try:
            rep_window = max(REP_WINDOW_OFF, int(request.args.get("rep_window", REP_WINDOW_OFF)))
            rep_penalty = max(REP_PENALTY_OFF, float(request.args.get("rep_penalty", REP_PENALTY_OFF)))
            no_repeat_ngram = max(NO_REPEAT_NGRAM_OFF,
                                  int(request.args.get("no_repeat_ngram", NO_REPEAT_NGRAM_OFF)))
        except ValueError:
            rep_window, rep_penalty, no_repeat_ngram = (REP_WINDOW_OFF, REP_PENALTY_OFF,
                                                        NO_REPEAT_NGRAM_OFF)

        if backend == "c":
            if cfg.get("C_SUBPROCESS") is None:
                try:
                    bins = sum(1 for n in models.list_models() if binr.exists(n))
                except Exception:
                    bins = 0
                msg = ("No exported .bin available. Train a model and export it first, "
                       "or switch to the PyTorch backend." if bins == 0
                       else "C engine not loaded. Pick a model from the dropdown.")
                return _sse_error_response(msg)
            def stream_c():
                try:
                    base = _c_engine_stream(cfg, prompt, max_new, temperature=temperature, top_k=top_k,
                                            ablate_layer=ablate_layer, ablate_neuron=ablate_neuron,
                                            addons_csv=",".join(addons_sel),
                                            rep_window=rep_window, rep_penalty=rep_penalty,
                                            no_repeat_ngram=no_repeat_ngram, trace=True)
                    stop_seq = _chat_stop_seq(prompt)
                    for ev in _stop_on_bytes(base, stop_seq):
                        yield f"data: {json.dumps(ev)}\n\n"
                    yield "event: done\ndata: {}\n\n"
                except GeneratorExit:
                    return
                except Exception as e:
                    logmod.error("generate", f"c-engine stream failed: {type(e).__name__}: {e}")
                    try:
                        yield "data: " + json.dumps({
                            "kind": "error",
                            "message": user_error(e, "c-engine"),
                        }) + "\n\n"
                        yield "event: done\ndata: {}\n\n"
                    except Exception:
                        pass
            return Response(stream_c(), mimetype="text/event-stream",
                            headers=SSE_HEADERS)

        brain = cfg["BRAIN"]
        if brain is None:
            return _sse_error_response(
                "PyTorch backend not loaded. Pick a model from the dropdown and try again.")
        cfg["BRAIN_LAST_USED"] = time.time()

        chain = None
        if addons_sel:
            try:
                chain = build_chain(addons_sel)
            except Exception as e:
                logmod.error("addons", f"build_chain failed: {type(e).__name__}: {e}")
                return ({"error": user_error(e, "addons")}, 400)

        constraint = None
        if constrained_v:
            try:
                constraint = _build_constraint(constrained_v)
            except Exception as e:
                logmod.error("constrained", f"build failed: {type(e).__name__}: {e}")
                return ({"error": user_error(e, "constrained")}, 400)

        if fast_mode and fast_mode not in ("kv", "mtp", "mtp-verify", "adaptive"):
            return ({"error": f"unknown fast mode: {fast_mode!r}. Allowed: kv, mtp, mtp-verify, adaptive."}, 400)

        rag_path  = (request.args.get("rag", "") or "").strip()
        rag_k     = request.args.get("rag_k", "")
        rag_press = (request.args.get("rag_compress", "") or "").strip().lower()
        rag_cfg = None
        if rag_path:
            if not is_loopback(request.remote_addr):
                return ({"error": PATH_LOOPBACK_ONLY}, 403)
            try:
                rag_top_k = max(1, min(int(rag_k), RAG_K_MAX)) if rag_k else RAG_K_DEFAULT
            except (TypeError, ValueError):
                return ({"error": "rag_k must be an integer 1..16"}, 400)
            rp = rag_press.split(":", 1) if rag_press else [""]
            rp_kind = rp[0]
            if rp_kind not in ("", "off", "crude", "word_ppl"):
                return ({"error": f"unknown rag_compress: {rag_press!r}. "
                                  "Allowed: off, crude, word_ppl[:keep_frac]."}, 400)
            rp_keep = None
            if rp_kind == "word_ppl" and len(rp) == 2:
                try:
                    rp_keep = float(rp[1])
                    if not (0.0 < rp_keep <= 1.0):
                        raise ValueError
                except ValueError:
                    return ({"error": "word_ppl keep_frac must be a float in (0, 1]"}, 400)
            try:
                tool, abs_corpus = _get_rag_tool(rag_path)
            except Exception as e:
                return ({"error": user_error(e, "rag")}, 400)
            rag_cfg = {
                "tool":        tool,
                "abs_corpus":  abs_corpus,
                "top_k":       rag_top_k,
                "rp_kind":     rp_kind,
                "rp_keep":     rp_keep,
                "raw_label":   rag_press or "off",
            }

        def stream_pt():
            with brain.lock:
                try:
                    brain.set_ablation(ablate_layer, ablate_neuron)
                    effective_prompt = prompt
                    if rag_cfg is not None:
                        try:
                            hits_raw = rag_cfg["tool"].call({"query": prompt, "k": rag_cfg["top_k"]})
                        except Exception as e:
                            yield f"data: {json.dumps({'kind': 'error', 'message': user_error(e, 'rag retrieve')})}\n\n"
                            return
                        passages, hits_meta = _parse_rag_hits(hits_raw)
                        if rag_cfg["rp_kind"] == "crude":
                            compressor = crude_compressor
                        elif rag_cfg["rp_kind"] == "word_ppl":
                            keep = rag_cfg["rp_keep"] if rag_cfg["rp_keep"] is not None else 0.5
                            compressor = make_word_ppl_compressor(brain, keep_frac=keep)
                        else:
                            compressor = None
                        effective_prompt = build_rag_prefix(prompt, passages, compressor=compressor)
                        prefix_view = effective_prompt if len(effective_prompt) <= PROMPT_PREFIX_CAP \
                                      else effective_prompt[:PROMPT_PREFIX_CAP] + " ... [trimmed]"
                        yield ("data: " + json.dumps({
                            "kind":         "rag",
                            "backend":      "pytorch",
                            "corpus":       rag_cfg["abs_corpus"],
                            "top_k":        rag_cfg["top_k"],
                            "hits":         hits_meta,
                            "prefix_bytes": len(effective_prompt.encode("utf-8")),
                            "prefix_text":  prefix_view,
                            "compress":     rag_cfg["raw_label"],
                        }) + "\n\n")
                    if fast_mode:
                        gen = brain.stream_fast(effective_prompt, mode=fast_mode,
                                                temperature=temperature,
                                                top_k_sample=top_k, max_new=max_new,
                                                addons_chain=chain, constraint=constraint,
                                                adaptive_threshold=adaptive_threshold,
                                                rep_window=rep_window, rep_penalty=rep_penalty,
                                                no_repeat_ngram=no_repeat_ngram)
                    else:
                        gen = brain.stream(effective_prompt, temperature, top_k, max_new,
                                           addons_chain=chain, constraint=constraint,
                                           rep_window=rep_window, rep_penalty=rep_penalty,
                                           no_repeat_ngram=no_repeat_ngram)
                    stop_seq = _chat_stop_seq(effective_prompt)
                    for ev in _stop_on_bytes(gen, stop_seq):
                        ev["backend"] = "pytorch"
                        yield f"data: {json.dumps(ev)}\n\n"
                    yield "event: done\ndata: {}\n\n"
                except GeneratorExit:
                    return
                except Exception as e:
                    logmod.error("generate", f"pytorch stream failed: {type(e).__name__}: {e}")
                    try:
                        yield "data: " + json.dumps({
                            "kind": "error",
                            "message": user_error(e, "generation"),
                        }) + "\n\n"
                        yield "event: done\ndata: {}\n\n"
                    except Exception:
                        pass
                finally:
                    brain.set_ablation(-1, -1)

        return Response(stream_pt(), mimetype="text/event-stream",
                        headers=SSE_HEADERS)

    @app.route("/agent/stream")
    def agent_stream():
        """Full-glass agent trace stream. Emits parsed turn events over SSE."""
        cfg = current_app.config
        user_input  = request.args.get("prompt", "")
        if not user_input:
            return ({"error": "prompt is required"}, 400)
        brain = cfg.get("BRAIN")
        if brain is None:
            def stream_err():
                yield "data: " + json.dumps({"kind": "error",
                    "message": "PyTorch backend not loaded. Pick a model from the dropdown and try again."}) + "\n\n"
                yield "event: stop\ndata: {}\n\n"
            return Response(stream_err(), mimetype="text/event-stream",
                            headers=SSE_HEADERS)
        try:
            max_turns   = max(1, min(int(request.args.get("max_turns", AGENT_MAX_TURNS_DEFAULT)),
                                     AGENT_MAX_TURNS_CAP))
            best_of_n   = max(1, min(int(request.args.get("best_of_n", AGENT_BEST_OF_N_DEFAULT)),
                                     AGENT_BEST_OF_N_CAP))
            temperature = float(request.args.get("temperature", TEMPERATURE_DEFAULT))
            top_k       = int(request.args.get("top_k", TOP_K_DEFAULT))
            seed        = int(request.args.get("seed", AGENT_SEED_DEFAULT))
        except (TypeError, ValueError) as e:
            return ({"error": user_error(e, "bad query param")}, 400)
        corpus_path = (request.args.get("corpus", "") or "").strip() or None
        fs_root     = (request.args.get("fs_root", "") or "").strip() or None
        if (corpus_path or fs_root) and not is_loopback(request.remote_addr):
            return ({"error": PATH_LOOPBACK_ONLY}, 403)
        if corpus_path and not os.path.exists(os.path.expanduser(corpus_path)):
            return ({"error": f"corpus path does not exist: {corpus_path}"}, 400)
        if fs_root and not os.path.isdir(os.path.expanduser(fs_root)):
            return ({"error": f"fs_root must be an existing directory: {fs_root}"}, 400)
        try:
            toolbox = build_default_toolbox(corpus_path=corpus_path, fs_root=fs_root)
        except Exception as e:
            return ({"error": user_error(e, "toolbox")}, 400)
        tools_csv = (request.args.get("tools", "") or "").strip()
        if tools_csv:
            wanted = {t.strip() for t in tools_csv.split(",") if t.strip()}
            available = set(toolbox.names())
            toolbox._tools = {n: t for n, t in toolbox._tools.items() if n in wanted and n in available}
            if not toolbox._tools:
                return ({"error": "no usable tools: none of the requested tools are registered "
                                  "(retrieve needs a corpus; fs_read needs a folder)"}, 400)
        loop = AgentLoop(brain, toolbox, max_turns=max_turns,
                         temperature=temperature, top_k_sample=top_k,
                         best_of_n=best_of_n, seed_base=seed)

        def stream_agent():
            try:
                with brain.lock:
                    yield ("data: " + json.dumps({"kind": "agent_meta",
                                                    "tools": toolbox.names(),
                                                    "max_turns": max_turns,
                                                    "best_of_n": best_of_n}) + "\n\n")
                    for ev in loop.run_streaming(user_input):
                        yield "data: " + json.dumps(ev) + "\n\n"
                    yield "event: done\ndata: {}\n\n"
            except GeneratorExit:
                return
            except Exception as e:
                yield ("data: " + json.dumps({"kind": "error",
                                               "message": user_error(e)}) + "\n\n")

        return Response(stream_agent(), mimetype="text/event-stream",
                        headers=SSE_HEADERS)
