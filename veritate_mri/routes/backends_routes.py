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
from inference.agent.tools.retriever import make_tool as _make_rag_tool
from inference.backends.c_engine import CTracedSubprocess
from inference.backends.pytorch import (
    ACTIVATION_INT8_SCALE, NEURON_TOP_K, load_memory,
)
from readers import (
    bin as binr, checkpoints, config as cfg_reader, engine, models, paths,
)
from runtime import logs as logmod
from training import build_runner

from . import _brain
from ._common import auto_thread_count, user_error

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

_RAG_TOOL_CACHE = {}
_RAG_CACHE_LOCK = threading.Lock()

# ------------------------------------------------------------------------------------
# Functions

def _build_c_mri_frame(raw, fwd_ms, shape):
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
        mx = max(1e-9, float(bucket_vals.max()))
        u8 = ((bucket_vals / mx) * 255.0).clip(0, 255).astype(np.uint8).tolist()
        ffn_full.append(u8)
        ffn_argmax.append(bucket_argmax.astype(np.uint8).tolist())
        idx = np.argsort(-act)[:NEURON_TOP_K]
        ffn_top.append([{"id": int(i), "v": round(float(act[i]), 3)} for i in idx])

    attn_out = []
    info_flow_pos = np.zeros(R, dtype=np.float32)
    for layer in range(n_layers):
        heads_data = []
        for h in range(n_heads):
            w = raw["attention"][layer, h, :R].astype(np.float32)
            s = float(w.sum())
            if s > 1e-9: w = w / s
            ent = -float((w * np.log(w + 1e-12)).sum())
            topn = min(ATTN_TOP_K, R)
            idx = np.argsort(-w)[:topn]
            heads_data.append({
                "ent": round(ent, 3),
                "top": [{"p": int(p), "w": round(float(w[p]), 3)} for p in idx],
            })
            info_flow_pos += w
        attn_out.append(heads_data)

    flow_max = max(1e-9, float(info_flow_pos.max()))
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
        scaled = ll / mx * 8.0
        e = np.exp(scaled - scaled.max())
        probs = e / e.sum()
        top_idx = np.argsort(-probs)[:DLA_TOP_K_LENS]
        lens.append([{"b": int(b), "p": round(float(probs[b]), 3)} for b in top_idx])

    logits = raw["logits"].astype(np.float64)
    mx = max(1.0, float(np.abs(logits).max()))
    scaled = logits / mx * 8.0
    e = np.exp(scaled - scaled.max())
    probs = e / e.sum()
    top_idx = np.argsort(-probs)[:DLA_TOP_K_CAND]
    cand = [{"b": int(b), "p": round(float(probs[b]), 3)} for b in top_idx]
    sampled = raw["byte"]
    argmax_byte = int(raw.get("argmax_byte", int(np.argmax(logits))))
    entropy_bits  = float(-(probs * np.log2(probs + 1e-12)).sum())
    surprise_bits = float(-math.log2(float(probs[sampled]) + 1e-12))

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
        "ffn_full": ffn_full, "ffn_top": ffn_top,
        "ffn_argmax": ffn_argmax, "ffn_downsample": DS,
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
        "attn": attn_out, "info_flow": info_flow,
        "res": res, "contrib": contrib,
        "lens": lens,
        "cand": cand,
        "memory": [],
        "backend": "c",
    }


def _c_engine_stream(cfg, prompt, max_new, temperature=0.7, top_k=40,
                     ablate_layer=-1, ablate_neuron=-1, addons_csv=""):
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
    try:
        last = time.perf_counter()
        for raw in sub.stream(prompt, temperature, top_k, max_new,
                              ablate_layer=ablate_layer, ablate_neuron=ablate_neuron,
                              addons_csv=addons_csv):
            now = time.perf_counter()
            fwd_ms = (now - last) * 1000.0
            last = now
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
        model = model_override or _brain.resolve_c_model_bin(None)
        if not model or not os.path.isfile(model):
            logmod.error("backends", "no veritate.bin under any model; train + export one first")
            cfg["C_PENDING"] = False
            return
        model_dir = os.path.basename(os.path.dirname(model))
        boost = binr.act_boost(model_dir)
        if boost is not None and boost > 1:
            logmod.warn("backends", f"c engine: {model_dir} act_boost={boost} (untrusted); loading anyway via VERITATE_ALLOW_HIGH_ACT_BOOST=1, output may be gibberish")
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
            if not fn.lower().endswith((".txt", ".md", ".rst", ".text")):
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
    from inference.decode import JSONConstraint, VocabConstraint, StopOnConstraint
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
                logmod.info("backends", f"pytorch swap: {cur_m} step {cur_s} -> {body_model} step {body_step or 'latest'}")
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
                return ({"ok": False, "error": "no models with checkpoints under models/. train one first or pass an explicit model name."}, 400)
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
                    logmod.warn("backends", f"pytorch: no vanilla checkpoints found (default '{name}' is non-vanilla and fallback search yielded nothing)")
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
                try: sub.close()
                except Exception: pass
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
            mem_path = os.path.join(paths.model_dir(name), "neuron_memory.json")
            mem = load_memory(mem_path)
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
                affinity     = brain.neuron_byte_affinity(layer, nid, top_k=8)
                predecessors = brain.neuron_predecessors(layer, nid, top_k=10)
                successors   = brain.neuron_successors(layer, nid, top_k=8)
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
        if c_model_path and os.path.isfile(c_model_path):
            c_model_dir = os.path.basename(os.path.dirname(c_model_path))
            c_precision, c_version = binr.header(c_model_dir)
            c_training, c_activation = cfg_reader.training_kind(c_model_dir)
            c_description = cfg_reader.description(c_model_dir) or ""
            c_act_boost = binr.act_boost(c_model_dir)
        return {
            "checkpoint": brain.checkpoint if brain else None,
            "n_params":   brain.n_params if brain else 0,
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
            "c_engine_label":   eng["label"]   if eng else None,
            "c_engine_perf_ms_per_byte": eng["perf_ms_per_byte"] if eng else None,
            "c_model":      os.path.basename(c_model_path) if c_model_path else None,
            "c_model_dir":  c_model_dir,
            "c_model_path": c_model_path,
            "c_model_precision":   c_precision,
            "c_model_bin_version": c_version,
            "c_model_training":    c_training,
            "c_model_activation":  c_activation,
            "c_model_description": c_description,
            "c_model_act_boost":   c_act_boost,
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
        temperature   = float(request.args.get("temperature", "0.7"))
        top_k         = int(request.args.get("top_k", "40"))
        max_new       = int(request.args.get("max_new", "200"))
        backend       = request.args.get("backend", "c").lower()
        ablate_layer  = int(request.args.get("ablate_layer",  "-1"))
        ablate_neuron = int(request.args.get("ablate_neuron", "-1"))
        addons_csv    = request.args.get("addons", "")
        addons_sel    = [s.strip() for s in addons_csv.split(",") if s.strip()]
        fast_mode     = (request.args.get("fast", "") or "").strip().lower()
        constrained_v = (request.args.get("constrained", "") or "").strip()
        try:
            adaptive_threshold = float(request.args.get("adaptive_threshold", "0.8"))
        except ValueError:
            adaptive_threshold = 0.8
        adaptive_threshold = max(0.0, min(1.0, adaptive_threshold))

        if backend == "c":
            if cfg.get("C_SUBPROCESS") is None:
                try:
                    bins = sum(1 for n in models.list_models() if binr.exists(n))
                except Exception:
                    bins = 0
                msg = ("No exported .bin available. Train a model and export it first, "
                       "or switch to the PyTorch backend." if bins == 0
                       else "C engine not loaded. Pick a model from the dropdown.")
                def stream_err():
                    yield "data: " + json.dumps({"kind": "error", "message": msg}) + "\n\n"
                    yield "event: done\ndata: {}\n\n"
                return Response(stream_err(), mimetype="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            def stream_c():
                try:
                    for ev in _c_engine_stream(cfg, prompt, max_new, temperature=temperature, top_k=top_k,
                                               ablate_layer=ablate_layer, ablate_neuron=ablate_neuron,
                                               addons_csv=",".join(addons_sel)):
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
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        brain = cfg["BRAIN"]
        if brain is None:
            def stream_err():
                yield "data: " + json.dumps({"kind": "error",
                    "message": "PyTorch backend not loaded. Pick a model from the dropdown and try again."}) + "\n\n"
                yield "event: done\ndata: {}\n\n"
            return Response(stream_err(), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
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
            try:
                rag_top_k = max(1, min(int(rag_k), RAG_K_MAX)) if rag_k else RAG_K_DEFAULT
            except (TypeError, ValueError):
                return ({"error": "rag_k must be an integer 1..16"}, 400)
            rp = rag_press.split(":", 1) if rag_press else [""]
            rp_kind = rp[0]
            if rp_kind not in ("", "off", "crude", "word_ppl"):
                return ({"error": f"unknown rag_compress: {rag_press!r}. Allowed: off, crude, word_ppl[:keep_frac]."}, 400)
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
                                                adaptive_threshold=adaptive_threshold)
                    else:
                        gen = brain.stream(effective_prompt, temperature, top_k, max_new,
                                           addons_chain=chain, constraint=constraint)
                    for ev in gen:
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
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

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
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        try:
            max_turns   = max(1, min(int(request.args.get("max_turns", "6")), AGENT_MAX_TURNS_CAP))
            best_of_n   = max(1, min(int(request.args.get("best_of_n", "1")), AGENT_BEST_OF_N_CAP))
            temperature = float(request.args.get("temperature", "0.7"))
            top_k       = int(request.args.get("top_k", "40"))
            seed        = int(request.args.get("seed", "0"))
        except (TypeError, ValueError) as e:
            return ({"error": user_error(e, "bad query param")}, 400)
        corpus_path = (request.args.get("corpus", "") or "").strip() or None
        fs_root     = (request.args.get("fs_root", "") or "").strip() or None
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
                return ({"error": "no usable tools: none of the requested tools are registered (retrieve needs a corpus; fs_read needs a folder)"}, 400)
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
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
