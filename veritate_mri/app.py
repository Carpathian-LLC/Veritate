# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - flask app for the live MRI. routes only. all on-disk reads go through readers/.
# - serves the static dashboard, exposes per-model artifacts, drives the two backends.
# veritate_mri/app.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import errno
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
from flask import Flask, Response, request, send_from_directory
from werkzeug.serving import WSGIRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backends.pytorch import Brain, load_memory
from backends.c_engine import CTracedSubprocess
from readers import paths, models, hooks, train_csv, config as cfg_reader, checkpoints, engine, bin as binr, plugins as plugins_reader, corpus as corpus_reader, wiki as wiki_reader
import atlas as atlas_mod
import train_stream as train_stream_mod
import logs as logmod
import build_runner
import lifecycle
import plugin_runner
import plugins_sync
import models_sync
import corpus_sync
import sys_metrics
import settings as settings_mod
import heartbeat as heartbeat_mod
import ai_assist as ai_assist_mod
import app_sync as app_sync_mod
import addons as addons_mod
import threading

from agent.rag import build_rag_prefix, crude_compressor, make_word_ppl_compressor
from agent.tools.retriever import make_tool as _make_rag_tool
from agent.tools import build_default_toolbox
from agent.loop import AgentLoop

# ------------------------------------------------------------------------------------
# Constants

STATIC_DIR = os.path.join(HERE, "static")
VERSIONS_PATH = os.path.normpath(os.path.join(HERE, "..", "versions.json"))

NEURON_TOP_K = 8
C_ACT_SCALE = 32.0

THREADS_AUTO_MAX = 16


def auto_thread_count():
    n = os.cpu_count() or 1
    physical = n // 2 if n >= 2 else 1
    return max(1, min(THREADS_AUTO_MAX, physical))

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
# Preserve JSON key insertion order in responses (Flask 2.2 default is True,
# Flask 3+ default is False; pin explicitly so plugin manifests render in the
# order their authors wrote them regardless of Flask version).
try:
    app.json.sort_keys = False
except AttributeError:
    app.config["JSON_SORT_KEYS"] = False
app.config["BRAIN"] = None
app.config["C_EXE"] = None
app.config["C_MODEL"] = None
app.config["C_SUBPROCESS"] = None
app.config["BRAIN_LAST_USED"] = 0.0

# ------------------------------------------------------------------------------------
# Functions

def _safe_name(name):
    if not name: return False
    if ".." in name.split("/") or ".." in name.split("\\"): return False
    if name.startswith("/") or name.startswith("\\"): return False
    if ":" in name: return False
    return True


def _build_c_mri_frame(raw, fwd_ms, shape):
    n_layers = shape["layers"]
    n_heads  = shape["heads"]
    n_ffn    = shape["ffn"]
    DS       = max(1, n_ffn // 256)
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
            topn = min(6, R)
            idx = np.argsort(-w)[:topn]
            heads_data.append({
                "ent": round(ent, 3),
                "top": [{"p": int(p), "w": round(float(w[p]), 3)} for p in idx],
            })
            info_flow_pos += w
        attn_out.append(heads_data)

    flow_max = max(1e-9, float(info_flow_pos.max()))
    flow_idx = np.argsort(-info_flow_pos)[:min(8, R)]
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
        top_idx = np.argsort(-probs)[:3]
        lens.append([{"b": int(b), "p": round(float(probs[b]), 3)} for b in top_idx])

    logits = raw["logits"].astype(np.float64)
    mx = max(1.0, float(np.abs(logits).max()))
    scaled = logits / mx * 8.0
    e = np.exp(scaled - scaled.max())
    probs = e / e.sum()
    top_idx = np.argsort(-probs)[:12]
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
            act_f = a_int / C_ACT_SCALE
            w_f   = w_int * scale
            ctb_f = (c_int * scale) / C_ACT_SCALE
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


def _c_engine_stream(prompt, max_new, temperature=0.7, top_k=40,
                     ablate_layer=-1, ablate_neuron=-1, addons_csv=""):
    sub = app.config["C_SUBPROCESS"]
    if sub is None:
        yield {"kind": "error", "message": "c chat_traced subprocess not running"}
        return
    model_path = app.config["C_MODEL"]
    exe        = app.config["C_EXE"]
    model_name = os.path.basename(os.path.dirname(model_path)) if model_path else "(random)"
    shape = sub.shape
    ds = max(1, shape["ffn"] // 256)
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


def _coactivation(name, step):
    data = hooks.load_artifact(name, step, "generation")
    if data is None:
        return ({"error": "generation hook not found"}, 404)
    frames = data.get("frames") or []
    pair_count = {}
    layer_count = {}
    n_tokens = 0
    for fr in frames:
        active = []
        for L, neurons in enumerate(fr.get("ffn_top") or []):
            for n in neurons:
                if abs(float(n.get("v", 0.0))) > 0.5:
                    k = (int(L), int(n["id"]))
                    active.append(k)
                    layer_count[k] = layer_count.get(k, 0) + 1
        active.sort()
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                pk = (active[i], active[j])
                pair_count[pk] = pair_count.get(pk, 0) + 1
        n_tokens += 1
    if n_tokens == 0:
        return ({"step": step, "n_tokens": 0, "pairs": [], "nodes": []}, 200)
    pairs = []
    for (a, b), c in pair_count.items():
        ea = layer_count.get(a, 0) / n_tokens
        eb = layer_count.get(b, 0) / n_tokens
        expected = ea * eb * n_tokens
        lift = c / expected if expected > 1e-9 else 0.0
        pairs.append({"i": [a[0], a[1]], "j": [b[0], b[1]], "c": c, "lift": round(lift, 3)})
    pairs.sort(key=lambda p: -p["c"])
    pairs = pairs[:200]
    nodes_set = set()
    for p in pairs:
        nodes_set.add((p["i"][0], p["i"][1]))
        nodes_set.add((p["j"][0], p["j"][1]))
    nodes = [{"layer": L, "neuron": n, "fires": layer_count[(L, n)]}
             for (L, n) in sorted(nodes_set)]
    return ({"step": step, "n_tokens": n_tokens, "threshold": 0.5,
             "pairs": pairs, "nodes": nodes}, 200)


def _mean_abs_acts(frames):
    sums = {}; counts = {}
    for fr in frames or []:
        for L, neurons in enumerate(fr.get("ffn_top") or []):
            for n in neurons:
                k = (int(L), int(n["id"]))
                sums[k] = sums.get(k, 0.0) + abs(float(n.get("v", 0.0)))
                counts[k] = counts.get(k, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def _learning_rate(name, step):
    cur = hooks.load_artifact(name, step, "generation")
    if cur is None:
        return ({"error": "generation hook not found"}, 404)
    prior = next((s for s in reversed(hooks.list_steps(name)) if s < step), None)
    if prior is None:
        return ({"step": step, "prior_step": None, "neurons": []}, 200)
    prev = hooks.load_artifact(name, prior, "generation")
    if prev is None:
        return ({"error": f"prior step {prior} generation missing"}, 404)
    cur_mean  = _mean_abs_acts(cur.get("frames"))
    prev_mean = _mean_abs_acts(prev.get("frames"))
    rows = []
    for k in set(cur_mean) | set(prev_mean):
        a = cur_mean.get(k, 0.0)
        b = prev_mean.get(k, 0.0)
        rows.append({"layer": k[0], "neuron": k[1],
                     "delta": round(abs(a - b), 4),
                     "now": round(a, 4), "prev": round(b, 4)})
    rows.sort(key=lambda r: -r["delta"])
    return ({"step": step, "prior_step": prior, "neurons": rows[:512]}, 200)


def _surprise_atlas(name):
    series = hooks.load_series(name, "surprise")
    if not series:
        return ({"steps": [], "tokens": [], "prompt": "", "surprise": [], "median": []}, 200)
    steps  = [s for s, _ in series]
    prompt = series[0][1].get("prompt", "")
    tokens = series[0][1].get("tokens") or []
    n_tok = max(len(d.get("surprise") or []) for _, d in series) or len(tokens)
    grid = []
    for _, d in series:
        row = list(d.get("surprise") or [])
        if len(row) < n_tok: row += [None] * (n_tok - len(row))
        grid.append(row)
    median = []
    for j in range(n_tok):
        col = [grid[i][j] for i in range(len(steps)) if grid[i][j] is not None]
        if not col: median.append(None); continue
        col.sort()
        m = col[len(col) // 2] if len(col) % 2 else 0.5 * (col[len(col)//2 - 1] + col[len(col)//2])
        median.append(round(float(m), 4))
    return ({"steps": steps, "tokens": tokens[:n_tok], "prompt": prompt,
             "surprise": grid, "median": median}, 200)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/sys_metrics")
def sys_metrics_route():
    return sys_metrics.snapshot()


@app.route("/sys/specs")
def sys_specs_get():
    return sys_metrics.load_specs() or {"detected": False}


@app.route("/sys/detect", methods=["POST"])
def sys_specs_detect():
    return sys_metrics.detect_and_save()


@app.route("/heartbeat/status")
def heartbeat_status_route():
    return heartbeat_mod.status()


@app.route("/heartbeat/send", methods=["POST"])
def heartbeat_send_route():
    ok_send = heartbeat_mod.send_now()
    return {"ok": bool(ok_send), **heartbeat_mod.status()}


@app.route("/app/update_status")
def app_update_status_route():
    return app_sync_mod.status()


@app.route("/app/update_check", methods=["POST"])
def app_update_check_route():
    return app_sync_mod.check()


@app.route("/app/update_pull", methods=["POST"])
def app_update_pull_route():
    body = request.get_json(silent=True) or {}
    force            = bool(body.get("force"))
    ignore_training  = bool(body.get("ignore_training"))
    res = app_sync_mod.pull_update(force=force, ignore_training=ignore_training)
    if res.get("ok") and body.get("reload"):
        try:
            lifecycle.restart(app.config)
        except Exception as e:
            res["reload_error"] = f"{type(e).__name__}: {e}"
    return res


@app.route("/app/local_edits")
def app_local_edits_route():
    """Returns the list of files that diverge from the last-pulled baseline.
    The dashboard calls this before triggering /app/update_pull so it can
    warn the user about modified/missing/added source files."""
    return app_sync_mod.local_edits()


@app.route("/app/update_channel", methods=["POST"])
def app_update_channel_route():
    body = request.get_json(silent=True) or {}
    channel = (body.get("channel") or "").lower()
    return app_sync_mod.switch_channel(channel)


@app.route("/versions")
def versions_route():
    if not os.path.isfile(VERSIONS_PATH):
        return ({"error": f"versions file not found: {VERSIONS_PATH}"}, 404)
    with open(VERSIONS_PATH, "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="application/json")


@app.route("/wiki")
def wiki_index():
    return {"categories": wiki_reader.list_categories()}


@app.route("/wiki/<category>")
def wiki_category(category):
    entries = wiki_reader.list_entries(category)
    if entries is None:
        return ({"error": f"category not found: {category}"}, 404)
    return {"category": category, "entries": entries}


@app.route("/wiki/<category>/<slug>")
def wiki_entry(category, slug):
    entry = wiki_reader.load_entry(category, slug)
    if entry is None:
        return ({"error": f"entry not found: {category}/{slug}"}, 404)
    return entry


@app.route("/settings", methods=["GET", "POST"])
def settings_route():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        try:
            out = settings_mod.update(body)
        except ValueError as ve:
            return {"error": str(ve)}, 400
        if out.get("pytorch_load_mode") == "always" and app.config.get("BRAIN") is None:
            try:
                name = app.config.get("BRAIN_MODEL") or app.config.get("DEFAULT_MODEL")
                step = app.config.get("BRAIN_STEP")  or app.config.get("DEFAULT_STEP")
                if not name or not models.exists(name):
                    name = _resolve_pytorch_model("auto")
                    if name is not None:
                        app.config["DEFAULT_MODEL"] = name
                        step = checkpoints.latest_step(name)
                        app.config["DEFAULT_STEP"]  = step
                if name and step is not None:
                    threads = int(app.config.get("DEFAULT_THREADS") or auto_thread_count())
                    brain, name, step = _load_pytorch_brain(name, step, threads)
                    app.config["BRAIN"] = brain
                    app.config["BRAIN_MODEL"] = name
                    app.config["BRAIN_STEP"]  = int(step)
                    app.config["DEFAULT_MODEL"] = name
                    app.config["DEFAULT_STEP"]  = int(step)
                    app.config["BRAIN_LAST_USED"] = time.time()
                    app.config["BRAIN_LAST_ERROR"] = None
                    logmod.ok("backends", f"pytorch eager-loaded after settings flip: {name} step {step}")
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                app.config["BRAIN_LAST_ERROR"] = msg
                if isinstance(e, RuntimeError) and "PyTorch inference is not enabled" in str(e):
                    logmod.warn("backends", f"pytorch backend skipped for {name}: non-vanilla architecture (use C engine)")
                else:
                    logmod.error("backends", f"pytorch eager load on settings flip failed: {msg}")
        return out
    return settings_mod.get()


@app.route("/settings/notices", methods=["GET"])
def settings_notices_route():
    return {"notices": settings_mod.pending_notices()}


@app.route("/ai/ask", methods=["POST"])
def ai_ask_route():
    body = request.get_json(silent=True) or {}
    kind = body.get("kind") or ""
    payload = body.get("payload") or {}
    return ai_assist_mod.ask(kind, payload)


@app.route("/logs/snapshot")
def logs_snapshot():
    after = int(request.args.get("after", "0"))
    limit = request.args.get("limit")
    rows = logmod.snapshot(after_seq=after, limit=int(limit) if limit else None)
    return {"latest_seq": logmod.latest_seq(), "entries": rows}


@app.route("/logs/stream")
def logs_stream():
    q = logmod.subscribe()

    def stream():
        try:
            while True:
                try:
                    entry = q.get(timeout=15.0)
                    yield f"data: {json.dumps(entry)}\n\n"
                except Exception:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            return
        finally:
            logmod.unsubscribe(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/engine/status")
def engine_status():
    s = build_runner.state()
    cur_exe = app.config.get("C_EXE")
    s["c_subprocess_running"] = app.config.get("C_SUBPROCESS") is not None
    s["c_exe"] = cur_exe
    return s


@app.route("/engine/build", methods=["POST"])
def engine_build_trigger():
    return build_runner.start()


@app.route("/plugins")
def plugins_index():
    return {"plugins": plugins_reader.scan(), "running": plugin_runner.state()}


@app.route("/plugins/run", methods=["POST"])
def plugins_run():
    body = request.get_json(silent=True) or {}
    plugin_id = body.get("id")
    if not plugin_id:
        return ({"ok": False, "error": "missing 'id'"}, 400)
    args = body.get("args") or {}
    # Refuse name collisions on fresh runs. Continue/resume flows pass an
    # explicit "resume" arg and are exempt — they target an existing model dir
    # by design. The dashboard sends only the form's `name` field; we slugify
    # and compose <slug>_<size> the same way the plugin does.
    if not (args.get("resume") or args.get("base_ckpt")):
        user_name = (args.get("name") or "").strip()
        size      = (args.get("size") or "").strip()
        if user_name and size:
            slug = models.slugify_user_name(user_name)
            if slug:
                composed = f"{slug}_{size}"
                if models.exists(composed):
                    return ({
                        "ok": False,
                        "error": f"model '{composed}' already exists. pick a different name "
                                 "or use Continue Training to extend the existing run.",
                    }, 409)
    return plugin_runner.start(plugin_id, args)


@app.route("/plugins/stop", methods=["POST"])
def plugins_stop():
    return plugin_runner.stop()


@app.route("/train/discovery")
def train_discovery():
    out_models = []
    for name in models.list_models():
        steps = checkpoints.list_steps(name)
        if not steps: continue
        out_models.append({"name": name, "steps": steps})
    out_models.sort(key=lambda r: r["name"])
    return {
        "corpora": corpus_reader.list_stems(),
        "models":  out_models,
    }


@app.route("/corpus/<path:stem>/usage")
def corpus_usage(stem):
    if ".." in stem or stem.startswith("/") or stem.startswith("\\"):
        return ("bad stem", 400)
    data = corpus_reader.usage(stem)
    if data is None:
        return ({"error": f"corpus stem not found: {stem}"}, 404)
    return data


@app.route("/lifecycle/restart", methods=["POST"])
def lifecycle_restart():
    return lifecycle.restart(app.config)


@app.route("/lifecycle/kill", methods=["POST"])
def lifecycle_kill():
    return lifecycle.kill(app.config)


@app.route("/pruning/report")
def pruning_report():
    name = request.args.get("model")
    step_arg = request.args.get("step")
    samples = int(request.args.get("samples") or 32)
    if not name or not models.exists(name):
        return ({"ok": False, "error": f"model not found: {name}"}, 400)
    step = int(step_arg) if step_arg else (checkpoints.latest_step(name) or 0)
    if not step:
        return ({"ok": False, "error": f"no checkpoints under models/{name}/"}, 400)
    try:
        import torch
        import pruning as pruning_mod
        from veritate.model import Veritate
        ckpt_path = checkpoints.path_for(name, step)
        s = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg = dict(s.get("args", {}))
        sd = s["model"]
        del s  # drops optimizer state (~8 GB on 1B) before model construction
        if "tok_emb.weight" not in sd:
            plugin_name = str(cfg.get("plugin") or "").strip()
            tag = f" (plugin: {plugin_name})" if plugin_name else ""
            return ({"ok": False,
                     "error": "pruning is not enabled for this model" + tag + ". "
                              "Width-pruning targets dense FFN layers; this checkpoint "
                              "uses a non-vanilla architecture (e.g., Mixture-of-Experts)."},
                    400)
        layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        ffn_per_layer = [sd[f"blocks.{L}.ff.up.weight"].shape[0] for L in range(layers)]
        vocab, hidden = sd["tok_emb.weight"].shape
        # Handle both canonical (has pos_emb.weight) and RoPE-only / Veritate800M
        # checkpoints (no pos_emb; seq comes from training args).
        is_800m = "pos_emb.weight" not in sd and any(k.startswith("mtp.transforms.") for k in sd)
        if "pos_emb.weight" in sd:
            seq = sd["pos_emb.weight"].shape[0]
        else:
            seq = int(cfg.get("seq") or 0)
            if seq <= 0:
                return ({"ok": False,
                         "error": "checkpoint has no pos_emb.weight and no seq in args; "
                                  "cannot infer sequence length."}, 400)
        heads = int(cfg.get("heads") or 0)
        if heads <= 0 or hidden % heads != 0:
            target = max(1, hidden // 64)
            for h in sorted({d for d in range(1, hidden + 1) if hidden % d == 0},
                            key=lambda d: (abs(d - target), -d)):
                heads = h
                break
        ffn_arg = ffn_per_layer if len(set(ffn_per_layer)) > 1 else ffn_per_layer[0]
        # Dispatch to the right model class. Veritate800M's FFN layout matches
        # canonical Veritate (blocks.{L}.ff.up/.down), so width-pruning analysis
        # works against either; only construction differs.
        if is_800m:
            plugin_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "plugins", "veritate_800m"))
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            from plugin import Veritate800M  # type: ignore
            n_predict = int(cfg.get("n_predict") or 4)
            rope_base = float(cfg.get("rope_base") or 10000.0)
            m = Veritate800M(vocab=vocab, hidden=hidden, layers=layers, ffn=ffn_arg,
                             heads=heads, seq=seq,
                             n_predict=n_predict, rope_base=rope_base)
            m.load_state_dict(sd, strict=False)  # rope_cos/sin are non-persistent buffers
        else:
            m = Veritate(vocab=vocab, hidden=hidden, layers=layers, ffn=ffn_arg,
                         heads=heads, seq=seq)
            m.load_state_dict(sd, strict=True)
        m.eval()

        corpus_stem = (cfg.get("corpus") or "").strip()
        if corpus_stem and ":" in corpus_stem:
            corpus_stem = corpus_stem.rsplit(":", 1)[-1]
        if not corpus_stem:
            return ({"ok": False, "error": "checkpoint args missing 'corpus' stem"}, 400)
        corpus_path = paths.corpus_train_path(corpus_stem)
        if not os.path.isfile(corpus_path):
            return ({"ok": False, "error": f"corpus bin not found: {corpus_path}"}, 400)

        report = pruning_mod.measure_activity(m, corpus_path, n_samples=samples)
        plan   = pruning_mod.recommend_plan(report, layers)

        total_params = sum(p.numel() for p in m.parameters())
        keep_total = 0
        for L in range(layers):
            keep_frac = float(plan[str(L)])
            kept_ffn  = max(1, int(round(ffn_per_layer[L] * keep_frac)))
            keep_total += kept_ffn * hidden * 2
        keep_total += vocab * hidden + seq * hidden + hidden + sum(
            3 * hidden * hidden + hidden * hidden + 2 * hidden  # qkv + proj + 2 RMSNorms
            for _ in range(layers)
        )
        size_after_mb = round(keep_total * 4 / (1024 * 1024), 1)
        size_before_mb = round(total_params * 4 / (1024 * 1024), 1)

        avg_alive = sum(e["alive_frac"] for e in report["per_layer"]) / max(1, layers)
        return {
            "ok":           True,
            "model":        name,
            "step":         int(step),
            "corpus":       corpus_stem,
            "samples":      int(samples),
            "n_params":     int(total_params),
            "n_params_after": int(keep_total),
            "size_mb_before": size_before_mb,
            "size_mb_after":  size_after_mb,
            "dead_pct":     round((1.0 - avg_alive) * 100, 1),
            "per_layer":    [{
                "layer":      e["layer"],
                "alive":      e["alive"],
                "total":      e["total"],
                "alive_frac": round(e["alive_frac"], 4),
                "keep":       float(plan[str(e["layer"])]),
            } for e in report["per_layer"]],
            "plan":         plan,
        }
    except Exception as e:
        logmod.error("pruning", f"report failed: {type(e).__name__}: {e}")
        return ({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.route("/pruning/generate_plugin", methods=["POST"])
def pruning_generate_plugin():
    body = request.get_json(silent=True) or {}
    name = body.get("model")
    step = int(body.get("step") or 0)
    plan = body.get("plan") or {}
    if not name or not models.exists(name) or not step:
        return ({"ok": False, "error": "missing or invalid model/step"}, 400)
    try:
        import torch
        import pruning as pruning_mod
        from veritate.model import Veritate
        ckpt_path = checkpoints.path_for(name, step)
        s = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg = dict(s.get("args", {}))
        sd = s["model"]
        del s  # drops optimizer state (~8 GB on 1B) before model construction
        if "tok_emb.weight" not in sd:
            plugin_name = str(cfg.get("plugin") or "").strip()
            tag = f" (plugin: {plugin_name})" if plugin_name else ""
            return ({"ok": False,
                     "error": "pruning is not enabled for this model" + tag + ". "
                              "Width-pruning targets dense FFN layers; this checkpoint "
                              "uses a non-vanilla architecture (e.g., Mixture-of-Experts)."},
                    400)
        layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
        ffn_per_layer = [sd[f"blocks.{L}.ff.up.weight"].shape[0] for L in range(layers)]
        vocab, hidden = sd["tok_emb.weight"].shape
        is_800m = "pos_emb.weight" not in sd and any(k.startswith("mtp.transforms.") for k in sd)
        if "pos_emb.weight" in sd:
            seq = sd["pos_emb.weight"].shape[0]
        else:
            seq = int(cfg.get("seq") or 0)
            if seq <= 0:
                return ({"ok": False,
                         "error": "checkpoint has no pos_emb.weight and no seq in args; "
                                  "cannot infer sequence length."}, 400)
        heads = int(cfg.get("heads") or 0)
        if heads <= 0 or hidden % heads != 0:
            target = max(1, hidden // 64)
            for h in sorted({d for d in range(1, hidden + 1) if hidden % d == 0},
                            key=lambda d: (abs(d - target), -d)):
                heads = h; break
        ffn_arg = ffn_per_layer if len(set(ffn_per_layer)) > 1 else ffn_per_layer[0]
        if is_800m:
            plugin_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "plugins", "veritate_800m"))
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            from plugin import Veritate800M  # type: ignore
            n_predict = int(cfg.get("n_predict") or 4)
            rope_base = float(cfg.get("rope_base") or 10000.0)
            m = Veritate800M(vocab=vocab, hidden=hidden, layers=layers, ffn=ffn_arg,
                             heads=heads, seq=seq,
                             n_predict=n_predict, rope_base=rope_base)
            m.load_state_dict(sd, strict=False)
        else:
            m = Veritate(vocab=vocab, hidden=hidden, layers=layers, ffn=ffn_arg,
                         heads=heads, seq=seq)
            m.load_state_dict(sd, strict=True)
        m.eval()

        corpus_stem = (cfg.get("corpus") or "").strip()
        if corpus_stem and ":" in corpus_stem:
            corpus_stem = corpus_stem.rsplit(":", 1)[-1]
        corpus_path = paths.corpus_train_path(corpus_stem) if corpus_stem else None

        if corpus_path and os.path.isfile(corpus_path):
            report = pruning_mod.measure_activity(m, corpus_path, n_samples=int(body.get("samples") or 16))
        else:
            report = {"per_layer": [{"layer": L, "alive": 0,
                                     "total": ffn_per_layer[L],
                                     "alive_frac": 0.0,
                                     "score_max": 0.0,
                                     "score_mean": 0.0}
                                    for L in range(layers)]}

        plugin_dir, plugin_id = pruning_mod.generate_plugin(
            src_name=name, src_step=int(step), plan=plan, report=report,
            corpus_stem=corpus_stem or "")

        logmod.ok("pruning", f"generated plugin: {plugin_id} at {plugin_dir}")
        return {"ok": True, "plugin_id": plugin_id, "plugin_dir": plugin_dir}
    except Exception as e:
        logmod.error("pruning", f"generate_plugin failed: {type(e).__name__}: {e}")
        return ({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.route("/export/<name>", methods=["POST"])
def export_bin(name):
    if not models.exists(name):
        return ({"ok": False, "error": f"model not found: {name}"}, 404)
    body = request.get_json(silent=True) or {}
    step = body.get("step")
    if step is None:
        step = checkpoints.latest_step(name)
    if step is None:
        return ({"ok": False, "error": f"no checkpoints under models/{name}/"}, 400)
    try:
        import export as export_mod
        result = export_mod.export_checkpoint(name, int(step))
    except (FileNotFoundError, ValueError, KeyError) as e:
        logmod.error("export", f"{name} step {step}: {type(e).__name__}: {e}")
        return ({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)
    except Exception as e:
        logmod.error("export", f"{name} step {step}: {type(e).__name__}: {e}")
        return ({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
    logmod.ok("export", f"{name} step {step}: wrote {result['path']} ({result['bytes']} bytes)")
    return {"ok": True, **result}


@app.route("/lifecycle/soft_reload", methods=["POST"])
def lifecycle_soft_reload():
    return lifecycle.soft_reload(app.config)


@app.route("/plugins/git/status")
def plugins_git_status():
    return plugins_sync.status()


@app.route("/plugins/git/sync", methods=["POST"])
def plugins_git_sync():
    body = request.get_json(silent=True) or {}
    actions = body.get("actions") if isinstance(body.get("actions"), dict) else None
    branch  = body.get("branch") if isinstance(body.get("branch"), str) else None
    return plugins_sync.sync(actions=actions, branch=branch)


@app.route("/plugins/git/check", methods=["POST"])
def plugins_git_check():
    return plugins_sync.check()


@app.route("/plugins/git/files")
def plugins_git_files():
    """Per-file table with three-state classification. The dashboard renders
    one row per remote (or tracked) file and exposes per-row action buttons."""
    return plugins_sync.files()


@app.route("/plugins/open_folder", methods=["POST"])
def plugins_open_folder():
    folder = plugins_reader.PLUGINS_ROOT
    os.makedirs(folder, exist_ok=True)
    sysname = platform.system()
    try:
        if sysname == "Windows":
            subprocess.Popen(["explorer.exe", folder])
        elif sysname == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except OSError as e:
        return ({"ok": False, "error": str(e), "path": folder}, 500)
    return {"ok": True, "path": folder}


@app.route("/models/bin_health")
def models_bin_health():
    """Per-model .bin header health for every model dir on disk. The dashboard
    polls this and shows a banner when any model has stale=True (typically a
    retired format version that the engine no longer loads). Re-export from
    the model's most recent .pt checkpoint to clear the warning."""
    out = []
    for name in models.list_models():
        h = binr.health(name)
        out.append({
            "name":    name,
            "version": h["version"],
            "label":   h["label"],
            "present": h["present"],
            "stale":   h["stale"],
            "reason":  h["reason"],
        })
    stale_count = sum(1 for r in out if r["stale"])
    return {"models": out, "stale_count": stale_count}


@app.route("/models/git/status")
def models_git_status():
    return models_sync.status()


@app.route("/models/git/sync", methods=["POST"])
def models_git_sync():
    body = request.get_json(silent=True) or {}
    actions = body.get("actions") if isinstance(body.get("actions"), dict) else None
    branch  = body.get("branch") if isinstance(body.get("branch"), str) else None
    return models_sync.sync(actions=actions, branch=branch)


@app.route("/models/git/check", methods=["POST"])
def models_git_check():
    return models_sync.check()


@app.route("/models/git/files")
def models_git_files():
    """Per-file table + per-dir provenance. Local-trained model dirs (those
    not present in the remote tree) are excluded — they're never affected by
    sync. The dashboard surfaces provenance as a badge per group."""
    return models_sync.files()


@app.route("/models/git/progress")
def models_git_progress():
    """Live byte-counter for the active models_sync.sync() run. Polled by the
    dashboard while large downloads are in flight."""
    return models_sync.progress()


@app.route("/models/open_folder", methods=["POST"])
def models_open_folder():
    folder = paths.MODELS_ROOT
    os.makedirs(folder, exist_ok=True)
    sysname = platform.system()
    try:
        if sysname == "Windows":
            subprocess.Popen(["explorer.exe", folder])
        elif sysname == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except OSError as e:
        return ({"ok": False, "error": str(e), "path": folder}, 500)
    return {"ok": True, "path": folder}


@app.route("/corpus/library/catalog")
def corpus_library_catalog():
    return corpus_sync.catalog()


@app.route("/corpus/library/install", methods=["POST"])
def corpus_library_install():
    body = request.get_json(silent=True) or {}
    return corpus_sync.install(body)


@app.route("/corpus/library/install_deps", methods=["POST"])
def corpus_library_install_deps():
    """Run `<this-python> -m pip install -r requirements.txt` in a subprocess.
    Guaranteed to land in the same Python that's running this Flask process,
    so `import datasets` succeeds immediately after — no restart, no PATH
    detective work for the user, no platform-specific instructions."""
    return corpus_sync.install_hf_deps()


@app.route("/corpus/library/uninstall", methods=["POST"])
def corpus_library_uninstall():
    body = request.get_json(silent=True) or {}
    return corpus_sync.uninstall(body.get("stem"))


@app.route("/corpus/library/catalog_url", methods=["POST"])
def corpus_library_catalog_url():
    body = request.get_json(silent=True) or {}
    return corpus_sync.set_catalog_url(body.get("url"))


@app.route("/corpus/library/sources/add", methods=["POST"])
def corpus_library_sources_add():
    body = request.get_json(silent=True) or {}
    return corpus_sync.add_user_source(body)


@app.route("/corpus/library/sources/remove", methods=["POST"])
def corpus_library_sources_remove():
    body = request.get_json(silent=True) or {}
    return corpus_sync.remove_user_source(body.get("stem"))


@app.route("/corpus/open_folder", methods=["POST"])
def corpus_open_folder():
    folder = paths.CORPUS_ROOT
    os.makedirs(folder, exist_ok=True)
    sysname = platform.system()
    try:
        if sysname == "Windows":
            subprocess.Popen(["explorer.exe", folder])
        elif sysname == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
    except OSError as e:
        return ({"ok": False, "error": str(e), "path": folder}, 500)
    return {"ok": True, "path": folder}


@app.route("/runs")
def runs_index():
    out = []
    for name in models.list_models():
        if not train_csv.is_present(name):
            continue
        st = train_csv.file_stat(name)
        rows = train_csv.load(name)
        out.append({
            "name": name,
            "mtime": st.st_mtime if st else 0,
            "size":  st.st_size  if st else 0,
            "n_rows": len(rows),
        })
    out.sort(key=lambda r: -r["mtime"])
    return {"runs": out}


@app.route("/run/<path:name>/csv")
def run_csv(name):
    if not _safe_name(name) or not models.exists(name): return ("run not found", 404)
    if not train_csv.is_present(name): return ("csv not found", 404)
    text = train_csv.raw_text(name)
    return Response(text, mimetype="text/csv")


@app.route("/run/<path:name>/probes")
def run_probes(name):
    if not _safe_name(name) or not models.exists(name):
        return {"name": name, "model_dir": False, "steps": []}
    out = []
    for s in hooks.list_steps(name):
        out.append({
            "step":  s,
            "probe": f"probe_step_{s}.json" if hooks.artifact_exists(name, s, "probe") else None,
            "lens":  f"lens_step_{s}.npz"   if hooks.artifact_exists(name, s, "lens")  else None,
        })
    return {"name": name, "model_dir": True, "steps": out}


@app.route("/run/<path:name>/classroom")
def run_classroom(name):
    if not _safe_name(name) or not models.exists(name):
        return ({"name": name, "model_dir": False, "items": []}, 404)
    items = []
    for s in hooks.list_steps(name):
        for kind in ("classroom", "grades", "math", "grammar", "reasoning", "concepts", "writing_health", "reading_comprehension"):
            if hooks.artifact_exists(name, s, kind):
                items.append({"kind": kind, "step": s, "file": f"{kind}_step_{s}.json"})
    items.sort(key=lambda r: (r["step"], r["kind"]))
    return {"name": name, "model_dir": True, "items": items}


@app.route("/run/<path:name>/config")
def run_config(name):
    if not _safe_name(name) or not models.exists(name):
        return ({"error": "config.json not found", "model_dir": name}, 404)
    data = cfg_reader.load(name)
    if data is None: return ({"error": "config.json not found", "model_dir": name}, 404)
    return Response(json.dumps(data), mimetype="application/json")


@app.route("/run/<path:name>/coactivation/<int:step>")
def run_coactivation(name, step):
    if not _safe_name(name) or not models.exists(name): return ("not found", 404)
    body, status = _coactivation(name, step)
    return body, status


@app.route("/run/<path:name>/learning_rate/<int:step>")
def run_learning_rate(name, step):
    if not _safe_name(name) or not models.exists(name): return ("not found", 404)
    body, status = _learning_rate(name, step)
    return body, status


@app.route("/run/<path:name>/surprise")
def run_surprise(name):
    if not _safe_name(name) or not models.exists(name): return ("not found", 404)
    body, status = _surprise_atlas(name)
    return body, status


# ------------------------------------------------------------------------------------
# Deep eval (real MMLU / HellaSwag / IFEval accuracy, not the perplexity-grade proxy).
#
# Two routes:
#   GET  /run/<name>/eval_deep         -> list cached results (lightweight summary)
#   POST /run/<name>/eval_deep         -> run one or more suites synchronously,
#                                         persist to models/<name>/eval_deep/, return
#
# Runs on demand only (no scheduler). The endpoint reuses the same `Brain` backend
# the dashboard already loads for inference — when training is on MPS, the eval
# will briefly share that device on user demand. The 800M training is hot on MPS;
# for evals you want to keep away from training, run the dashboard's PyTorch
# backend on CPU (default) before clicking the button.
# ------------------------------------------------------------------------------------

EVAL_DEEP_SUBDIR = "eval_deep"


def _eval_deep_dir(name):
    return os.path.join(paths.model_dir(name), EVAL_DEEP_SUBDIR)


def _eval_deep_summary(blob):
    """Distill a full eval report down to the per-suite headline numbers the
    dashboard wants in its list view.

    Accepts two on-disk shapes:
      - `{"suite": "<name>", "step": N, "report": {...}}`  (current writer:
        one suite per file)
      - `{"suites": {"<name>": {...}, ...}}`               (legacy aggregate)
    """
    if not isinstance(blob, dict):
        return []
    if "report" in blob and isinstance(blob["report"], dict) and "suite" in blob:
        sub = {blob["suite"]: blob["report"]}
    else:
        sub = blob.get("suites", {}) or {}
    out = []
    for suite_name, s in sub.items():
        if not isinstance(s, dict): continue
        # IFEval reports pass_rate; we also mirror it as `accuracy` from the
        # suite itself, so a single field works for everyone.
        acc = s.get("accuracy")
        if acc is None: acc = s.get("pass_rate")
        out.append({
            "suite": suite_name,
            "n":     s.get("n"),
            "acc":   acc,
            "elapsed_s": s.get("elapsed_s"),
            "by_subject": s.get("by_subject"),
            "by_rule":    s.get("by_rule"),
            "accuracy_letter": s.get("accuracy_letter"),
            "accuracy_text":   s.get("accuracy_text"),
        })
    return out


@app.route("/run/<path:name>/eval_deep", methods=["GET"])
def run_eval_deep_list(name):
    """List all cached deep-eval results for a model."""
    if not _safe_name(name) or not models.exists(name):
        return ({"error": "model not found"}, 404)
    root = _eval_deep_dir(name)
    out = []
    if os.path.isdir(root):
        for fn in sorted(os.listdir(root)):
            if not fn.endswith(".json"): continue
            # Filename convention: <suite>_step_<N>.json
            base = fn[:-5]
            suite = None; step = None
            if "_step_" in base:
                try:
                    s, st = base.rsplit("_step_", 1)
                    suite = s
                    step  = int(st)
                except Exception:
                    pass
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    blob = json.load(f)
            except Exception as e:
                logmod.warn("eval_deep", f"failed to read {fp}: {e}")
                continue
            try: mtime = os.path.getmtime(fp)
            except OSError: mtime = 0
            for s_entry in _eval_deep_summary(blob):
                out.append({
                    "suite":   s_entry["suite"] or suite,
                    "step":    step,
                    "file":    fn,
                    "mtime":   mtime,
                    "n":       s_entry["n"],
                    "acc":     s_entry["acc"],
                    "elapsed_s": s_entry["elapsed_s"],
                    "by_subject": s_entry["by_subject"],
                    "by_rule":    s_entry["by_rule"],
                    "accuracy_letter": s_entry["accuracy_letter"],
                    "accuracy_text":   s_entry["accuracy_text"],
                })
    out.sort(key=lambda r: (r["mtime"] or 0), reverse=True)
    return {"name": name, "results": out}


# Shared status board for in-flight deep-eval runs. Keyed by (name, step) tuple.
# Read by /run/<name>/eval_deep/status; updated by the worker thread.
_EVAL_DEEP_STATE = {"runs": {}, "lock": threading.Lock()}


def _eval_state_key(name, step):
    return f"{name}::{int(step)}"


def _eval_state_set(name, step, **kwargs):
    key = _eval_state_key(name, step)
    with _EVAL_DEEP_STATE["lock"]:
        prev = _EVAL_DEEP_STATE["runs"].get(key, {})
        prev.update(kwargs)
        _EVAL_DEEP_STATE["runs"][key] = prev


def _eval_state_get(name, step):
    key = _eval_state_key(name, step)
    with _EVAL_DEEP_STATE["lock"]:
        return dict(_EVAL_DEEP_STATE["runs"].get(key, {}))


@app.route("/run/<path:name>/eval_deep/status")
def run_eval_deep_status(name):
    step_s = request.args.get("step")
    try: step = int(step_s) if step_s is not None else None
    except Exception: step = None
    if step is None:
        # Latest checkpoint
        step = checkpoints.latest_step(name)
        if step is None:
            return ({"running": False, "error": "no checkpoints"}, 200)
    return _eval_state_get(name, step) or {"running": False}


def _resolve_eval_brain(name, step, threads):
    """Try to reuse the dashboard's already-loaded Brain when it matches. If
    not, load a fresh Brain through the same dispatcher Brain() uses; the
    caller is responsible for the runtime cost. Returns (brain, source) where
    source is "reused" or "loaded"."""
    cur_brain = app.config.get("BRAIN")
    cur_name  = app.config.get("BRAIN_MODEL")
    cur_step  = app.config.get("BRAIN_STEP")
    if cur_brain is not None and cur_name == name and int(cur_step or -1) == int(step):
        return cur_brain, "reused"
    brain, n_, s_ = _load_pytorch_brain(name, step, threads)
    # Don't promote the freshly-loaded brain to the global slot — the eval is
    # a one-shot user action and we don't want to evict whatever the user has
    # loaded for inference. We do still hold a reference for the duration of
    # the run via the local; it falls out of scope after.
    return brain, "loaded"


@app.route("/run/<path:name>/eval_deep", methods=["POST"])
def run_eval_deep_post(name):
    """Trigger a deep eval. JSON body:
       { "suite": "mmlu" | "hellaswag" | "ifeval" | "all" | comma-list,
         "step":  <int> | null (latest),
         "limit": <int> | null,
         "mmlu_mode": "letter" | "text" | "both" (default text) }
    """
    if not _safe_name(name) or not models.exists(name):
        return ({"error": "model not found"}, 404)
    body = request.get_json(silent=True) or {}
    suite = body.get("suite") or "mmlu"
    if isinstance(suite, list):
        suites = [str(x).lower() for x in suite]
    else:
        s = str(suite).lower().strip()
        if s == "all":
            suites = ["mmlu", "hellaswag", "ifeval"]
        else:
            suites = [t.strip() for t in s.split(",") if t.strip()]
    valid = {"mmlu", "hellaswag", "ifeval"}
    suites = [s for s in suites if s in valid]
    if not suites:
        return ({"error": f"no valid suites in {body.get('suite')!r}; expected one of {sorted(valid)} or 'all'"}, 400)

    step = body.get("step")
    if step in (None, "", "latest"):
        step = checkpoints.latest_step(name)
    if step is None:
        return ({"error": f"no checkpoints under models/{name}/"}, 400)
    try: step = int(step)
    except Exception: return ({"error": f"bad step: {step!r}"}, 400)

    limit       = body.get("limit")
    mmlu_mode   = body.get("mmlu_mode") or "text"
    ifeval_maxn = int(body.get("ifeval_max_new") or 256)
    threads     = int(body.get("threads") or app.config.get("DEFAULT_THREADS") or auto_thread_count())

    # Resolve / load the Brain. This may briefly contend with training on MPS
    # — that's the documented trade for a user-triggered run.
    t_load0 = time.time()
    try:
        brain, source = _resolve_eval_brain(name, step, threads)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logmod.error("eval_deep", f"load failed for {name} step {step}: {msg}")
        return ({"error": msg}, 500)
    load_secs = time.time() - t_load0
    logmod.ok("eval_deep", f"brain ready ({source}) for {name} step {step} in {load_secs:.1f}s")

    # Reserve state for status polling.
    _eval_state_set(name, step,
                    running=True, started=time.time(),
                    suites=suites, source=source, error=None,
                    progress={s: {"i": 0, "n": None} for s in suites},
                    current_suite=None, finished=None)

    def _progress(suite_name, i, n):
        _eval_state_set(name, step,
                        current_suite=suite_name,
                        progress_update={"suite": suite_name, "i": i, "n": n})
        # Also update the per-suite dict
        st = _eval_state_get(name, step)
        prog = st.get("progress") or {}
        prog[suite_name] = {"i": i, "n": n}
        _eval_state_set(name, step, progress=prog)

    # Run synchronously. The user can wait; the dashboard shows a spinner.
    # Holding brain.lock guards against concurrent Brain mutations (e.g. ablation).
    # Import under the `eval` package; veritate_mri/ is already on sys.path.
    from eval.run_eval import run_suites_on_model
    out_dir = _eval_deep_dir(name)
    os.makedirs(out_dir, exist_ok=True)
    written_files = []
    suite_results = {}
    try:
        with brain.lock:
            report = run_suites_on_model(
                brain.model,
                suites=suites,
                limit=limit,
                mmlu_mode=mmlu_mode,
                ifeval_max_new=ifeval_maxn,
                verbose=False,
                progress_cb=_progress,
            )
        suite_results = report.get("suites", {}) or {}
        # Persist one JSON per suite so the GET list can index them cleanly.
        for suite_name, s_report in suite_results.items():
            payload = {
                "name":   name,
                "step":   step,
                "suite":  suite_name,
                "report": s_report,
            }
            fn = f"{suite_name}_step_{step}.json"
            fp = os.path.join(out_dir, fn)
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            written_files.append(fn)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logmod.error("eval_deep", f"run failed for {name} step {step}: {msg}")
        _eval_state_set(name, step, running=False, error=msg, finished=time.time())
        return ({"error": msg}, 500)

    _eval_state_set(name, step, running=False, finished=time.time(),
                    error=None, files=written_files)
    return {
        "name":   name,
        "step":   step,
        "suites": list(suite_results.keys()),
        "files":  written_files,
        "report": suite_results,
    }


@app.route("/timelines")
def timelines_compat():
    # List every model with a config + at least one .pt checkpoint, even if
    # no hook artifacts exist yet. Hooks are surfaced via has_hooks /
    # n_checkpoints so the frontend can render an inline warning when a
    # timeline has training rows but no per-step hook dumps (e.g. mid-
    # training before the first save, or a trainer that has not yet been
    # ported to the hook_spec contract).
    out = []
    for name in models.list_models():
        hook_steps = hooks.list_steps(name)
        ckpt_steps = checkpoints.list_steps(name)
        if not ckpt_steps and not hook_steps:
            continue
        try: hooks_mtime = os.path.getmtime(paths.hooks_dir(name))
        except OSError: hooks_mtime = 0
        try: ckpts_mtime = os.path.getmtime(paths.checkpoints_dir(name))
        except OSError: ckpts_mtime = 0
        mtime = max(hooks_mtime, ckpts_mtime)
        out.append({
            "name": name,
            "mtime": mtime,
            "n_checkpoints": len(hook_steps),
            "n_pt_checkpoints": len(ckpt_steps),
            "has_hooks": bool(hook_steps),
            "prompt": (cfg_reader.load(name) or {}).get("training_args", {}).get("prompt", ""),
            "source": "hooks" if hook_steps else "checkpoints",
        })
    out.sort(key=lambda r: -r["mtime"])
    return {"timelines": out}


@app.route("/timeline/<path:name>/<path:fname>")
def timeline_file_compat(name, fname):
    if not _safe_name(name) or not _safe_name(fname): return ("bad name", 400)
    if not models.exists(name): return ("not found", 404)
    if fname == "timeline.json":
        return run_timeline(name)
    import re
    m = re.match(r"^(probe|classroom|grades|math|grammar|reasoning|concepts|surprise|quant_kl|writing_health|reading_comprehension)_step_(\d+)\.json$", fname)
    if m:
        data = hooks.load_artifact(name, int(m.group(2)), m.group(1))
        if data is None: return ("artifact not found", 404)
        return Response(json.dumps(data), mimetype="application/json")
    m = re.match(r"^step_(\d+)\.json$", fname)
    if m:
        data = hooks.load_artifact(name, int(m.group(1)), "generation")
        if data is None: return ("artifact not found", 404)
        return Response(json.dumps(data), mimetype="application/json")
    m = re.match(r"^lens_step_(\d+)\.npz$", fname)
    if m:
        p = paths.hook_artifact_path(name, int(m.group(1)), "lens")
        if not os.path.isfile(p): return ("artifact not found", 404)
        return send_from_directory(os.path.dirname(p), os.path.basename(p))
    return ("not found", 404)


@app.route("/run/<path:name>/timeline")
def run_timeline(name):
    if not _safe_name(name) or not models.exists(name): return ("not found", 404)
    cfg = cfg_reader.load(name) or {}
    description = cfg_reader.description(name)
    csv_rows = train_csv.load(name)
    train_by = {}; val_by = {}
    for r in csv_rows:
        s = (r.get("split") or "").lower()
        step = r.get("step")
        if step is None: continue
        if s == "train" or s.endswith("_train"):
            train_by[step] = r
        elif s == "val" or s.startswith("val_") or s.endswith("_val"):
            val_by[step] = r
    ckpt_steps = checkpoints.list_steps(name)
    hook_steps = hooks.list_steps(name)
    union = sorted(set(ckpt_steps) | set(hook_steps))
    ckpts = []
    prompt = ""
    max_new = 0
    for s in union:
        gen = hooks.load_artifact(name, s, "generation")
        n_frames = 0
        output_text = ""
        if gen is not None:
            frames = gen.get("frames") or []
            n_frames = len(frames)
            if n_frames > max_new: max_new = n_frames
            output_text = bytes([f["byte"] for f in frames if "byte" in f]).decode("utf-8", errors="replace")
            if not prompt: prompt = gen.get("prompt", "") or gen.get("meta", {}).get("prompt", "")
        if not prompt:
            probe = hooks.load_artifact(name, s, "probe")
            if probe is not None: prompt = probe.get("prompt", "")
        kl = hooks.load_artifact(name, s, "quant_kl")
        kl_bits = kl.get("quant_kl_bits") if isinstance(kl, dict) else None
        ckpts.append({
            "step": s,
            "file": f"step_{s}.json",
            "n_frames": n_frames,
            "output_text": output_text,
            "train_loss": (train_by.get(s) or {}).get("loss"),
            "val_loss":   (val_by.get(s)   or {}).get("loss"),
            "precision":  cfg.get("precision") or cfg.get("training") or "unknown",
            "quant_kl_bits": kl_bits,
        })
    return {"name": name, "prompt": prompt, "max_new": max_new,
            "checkpoints": ckpts, "description": description}


@app.route("/neuron/<int:layer>/<int:nid>")
def neuron_lookup(layer, nid):
    brain = app.config.get("BRAIN")
    name  = app.config.get("BRAIN_MODEL") or app.config.get("DEFAULT_MODEL")
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
            "pytorch_last_error": app.config.get("BRAIN_LAST_ERROR"),
        }
    affinity, predecessors, successors, stats = None, [], [], {}
    label = brain.label_for(layer, nid) if hasattr(brain, "label_for") else None
    app.config["BRAIN_LAST_USED"] = time.time()
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
    brain = app.config["BRAIN"]
    cur_exe = app.config.get("C_EXE")
    cur_abs = os.path.abspath(cur_exe) if cur_exe else None
    eng = engine.by_path(cur_abs) if cur_abs else None
    c_model_path = app.config.get("C_MODEL")
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


@app.route("/c-engines")
def c_engines_index():
    out = []
    cur_exe = app.config.get("C_EXE")
    cur_abs = os.path.abspath(cur_exe) if cur_exe else None
    for e in engine.engines():
        ap = os.path.abspath(e.get("path") or "")
        if not os.path.isfile(ap): continue
        try: st = os.stat(ap)
        except OSError: continue
        out.append({
            **e,
            "path": ap,
            "exists": True,
            "is_current": ap == cur_abs,
            "mtime": st.st_mtime,
            "size":  st.st_size,
        })
    return {"engines": out}


@app.route("/c-models")
def c_models_index():
    out = []
    cur_path = app.config.get("C_MODEL")
    cur_abs = os.path.abspath(cur_path) if cur_path else None
    for name in models.list_models():
        if not binr.exists(name): continue
        bp = paths.bin_path(name)
        try: st = os.stat(bp)
        except OSError: continue
        precision, version = binr.header(name)
        training, activation = cfg_reader.training_kind(name)
        out.append({
            "name": name,
            "bin_path": os.path.abspath(bp),
            "is_current": os.path.abspath(bp) == cur_abs,
            "mtime": st.st_mtime,
            "size":  st.st_size,
            "precision":   precision,
            "bin_version": version,
            "training":    training,
            "activation":  activation,
            "act_boost":   binr.act_boost(name),
            "description": cfg_reader.description(name),
        })
    out.sort(key=lambda r: -r["mtime"])
    return {"models": out}




@app.route("/pytorch-models")
def pytorch_models_index():
    # PyTorch loads .pt checkpoints directly — no .bin needed. So this list
    # is "any model with at least one saved checkpoint", which is the right
    # universe for the Generation tab's model picker when backend=pytorch.
    out = []
    cur_model = app.config.get("BRAIN_MODEL") or app.config.get("DEFAULT_MODEL")
    for name in models.list_models():
        step = checkpoints.latest_step(name)
        if step is None:
            continue
        try: cfg = cfg_reader.load(name) or {}
        except Exception: cfg = {}
        plugin = (cfg.get("plugin") or "").strip()
        n_params = cfg.get("n_params_total")
        shape = cfg.get("shape") or {}
        try: mtime = os.path.getmtime(checkpoints.path_for(name, step))
        except OSError: mtime = 0
        out.append({
            "name":        name,
            "step":        int(step),
            "is_current":  name == cur_model,
            "plugin":      plugin,
            "n_params":    int(n_params) if n_params else None,
            "hidden":      shape.get("hidden"),
            "layers":      shape.get("layers"),
            "description": cfg_reader.description(name) or "",
            "mtime":       mtime,
        })
    out.sort(key=lambda r: -r["mtime"])
    return {"models": out}


@app.route("/c-config", methods=["POST"])
def c_config():
    body = request.get_json(silent=True) or {}
    new_exe   = body.get("exe",   app.config["C_EXE"])
    new_model = body.get("model", app.config["C_MODEL"])
    if new_exe is not None and not os.path.isfile(new_exe):
        return ({"ok": False, "error": f"exe not found: {new_exe}"}, 400)
    if new_model is not None and not os.path.isfile(new_model):
        return ({"ok": False, "error": f"model not found: {new_model}"}, 400)
    if new_exe is None:
        return ({"ok": False, "error": "no c engine exe selected"}, 400)
    old = app.config.get("C_SUBPROCESS")
    if old is not None:
        try: old.close()
        except Exception: pass
    name = os.path.basename(os.path.dirname(new_model)) if new_model else None
    boost = binr.act_boost(name) if name else None
    # If selected bin is non-QAT, don't spawn — surface blocked state.
    if boost is not None and boost > 1:
        app.config["C_EXE"]            = new_exe
        app.config["C_MODEL"]          = new_model
        app.config["C_SUBPROCESS"]     = None
        app.config["C_BLOCKED_REASON"] = "qat_required"
        app.config["C_BLOCKED_MODEL"]  = name
        precision, version = (binr.header(name) if name else ("?", 0))
        training, activation = (cfg_reader.training_kind(name) if name else ("", ""))
        logmod.warn("backends", f"c-config: {name} not QAT-trained (act_boost={boost}); generate disabled")
        return {
            "ok": True,
            "c_exe_path":  new_exe,
            "c_exe":       os.path.basename(new_exe),
            "c_model_path": new_model,
            "c_model":     os.path.basename(new_model) if new_model else None,
            "c_model_dir": name,
            "c_model_precision":   precision,
            "c_model_bin_version": version,
            "c_model_training":    training,
            "c_model_activation":  activation,
            "c_model_act_boost":   boost,
            "blocked_reason":      "qat_required",
        }
    try:
        sub = CTracedSubprocess(new_exe, new_model)
    except Exception as e:
        app.config["C_SUBPROCESS"] = None
        return ({"ok": False, "error": f"respawn failed: {e}"}, 500)
    app.config["C_EXE"]        = new_exe
    app.config["C_MODEL"]      = new_model
    app.config["C_SUBPROCESS"] = sub
    app.config["C_BLOCKED_REASON"] = None
    app.config["C_BLOCKED_MODEL"]  = None
    logmod.info("c-config", f"exe={new_exe} model={new_model} pid={sub.proc.pid}")
    precision, version = (binr.header(name) if name else ("?", 0))
    training, activation = (cfg_reader.training_kind(name) if name else ("", ""))
    return {
        "ok": True,
        "c_exe_path":  new_exe,
        "c_exe":       os.path.basename(new_exe),
        "c_model_path": new_model,
        "c_model":     os.path.basename(new_model) if new_model else None,
        "c_model_dir": name,
        "c_model_precision":   precision,
        "c_model_bin_version": version,
        "c_model_training":    training,
        "c_model_activation":  activation,
        "c_model_act_boost":   boost,
    }


@app.route("/backends")
def backends_status():
    cur_exe   = app.config.get("C_EXE")
    cur_model = app.config.get("C_MODEL")
    try:
        bins_available = sum(1 for n in models.list_models() if binr.exists(n))
    except Exception:
        bins_available = 0
    return {
        "pytorch": {
            "loaded":  app.config.get("BRAIN") is not None,
            "pending": bool(app.config.get("PYTORCH_PENDING")),
            "model":   app.config.get("BRAIN_MODEL") or app.config.get("DEFAULT_MODEL"),
            "step":    app.config.get("BRAIN_STEP")  or app.config.get("DEFAULT_STEP"),
            "last_error": app.config.get("BRAIN_LAST_ERROR"),
        },
        "c": {
            "loaded":    app.config.get("C_SUBPROCESS") is not None,
            "pending":   bool(app.config.get("C_PENDING")),
            "exe":       cur_exe,
            "model_bin": cur_model,
            "model_dir": (os.path.basename(os.path.dirname(cur_model)) if cur_model else None),
            "blocked_reason": app.config.get("C_BLOCKED_REASON"),
            "blocked_model":  app.config.get("C_BLOCKED_MODEL"),
            "build":     build_runner.state(),
            "bins_available": bins_available,
        },
    }


@app.route("/backends/pytorch", methods=["POST"])
def backends_pytorch():
    body = request.get_json(silent=True) or {}
    action = (body.get("action") or "").lower()
    if action == "unload":
        if app.config.get("BRAIN") is not None:
            app.config["BRAIN"] = None
            app.config["BRAIN_MODEL"] = None
            app.config["BRAIN_STEP"]  = None
            logmod.ok("backends", "pytorch unloaded")
        return backends_status()
    if action == "load":
        body_model = body.get("model")
        body_step  = body.get("step")
        did_swap   = False
        if app.config.get("BRAIN") is not None:
            # Already loaded. If the caller asked for the same (or didn't
            # specify), this is a no-op. If they asked for a different model
            # or step, swap by clearing the current brain and falling through
            # to the normal load path.
            cur_m = app.config.get("BRAIN_MODEL")
            cur_s = app.config.get("BRAIN_STEP")
            same  = (not body_model) or (
                body_model == cur_m and
                (body_step is None or int(body_step) == int(cur_s or 0))
            )
            if same:
                return backends_status()
            logmod.info("backends", f"pytorch swap: {cur_m} step {cur_s} -> {body_model} step {body_step or 'latest'}")
            app.config["BRAIN"] = None
            app.config["BRAIN_MODEL"] = None
            app.config["BRAIN_STEP"]  = None
            did_swap = True
        name = body_model or app.config.get("DEFAULT_MODEL")
        if not name or not models.exists(name):
            name = _resolve_pytorch_model("auto")
            if name is not None:
                app.config["DEFAULT_MODEL"] = name
                app.config["DEFAULT_STEP"]  = checkpoints.latest_step(name)
        if not name or not models.exists(name):
            return ({"ok": False, "error": "no models with checkpoints under models/. train one first or pass an explicit model name."}, 400)
        # On a swap, the previous DEFAULT_STEP belongs to the old model — don't
        # reuse it as a fallback for the new one. Always re-resolve from disk.
        step = body_step or (None if did_swap else app.config.get("DEFAULT_STEP")) or checkpoints.latest_step(name)
        if step is None:
            return ({"ok": False, "error": f"no checkpoints under models/{name}/"}, 400)
        threads = int(body.get("threads") or app.config.get("DEFAULT_THREADS") or auto_thread_count())
        try:
            brain, name, step = _load_pytorch_brain(name, step, threads)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            app.config["BRAIN_LAST_ERROR"] = msg
            if isinstance(e, RuntimeError) and "PyTorch inference is not enabled" in str(e):
                logmod.warn("backends", f"pytorch: no vanilla checkpoints found (default '{name}' is non-vanilla and fallback search yielded nothing)")
                return ({"ok": False, "error": msg, "reason": "non_vanilla"}, 400)
            logmod.error("backends", f"pytorch load failed: {msg}")
            return ({"ok": False, "error": msg}, 500)
        app.config["BRAIN"] = brain
        app.config["BRAIN_MODEL"] = name
        app.config["BRAIN_STEP"]  = int(step)
        app.config["DEFAULT_MODEL"] = name
        app.config["DEFAULT_STEP"]  = int(step)
        app.config["BRAIN_LAST_USED"] = time.time()
        app.config["BRAIN_LAST_ERROR"] = None
        logmod.ok("backends", f"pytorch loaded: {name} step {step} ({brain.n_params:,} params)")
        return backends_status()
    return ({"ok": False, "error": "action must be load or unload"}, 400)


def _spawn_c_subprocess(exe, model):
    try:
        sub = CTracedSubprocess(exe, model)
    except Exception as e:
        logmod.error("backends", f"c engine spawn failed: {e}")
        app.config["C_PENDING"] = False
        return False
    app.config["C_SUBPROCESS"] = sub
    app.config["C_EXE"]        = exe
    app.config["C_MODEL"]      = model
    app.config["C_PENDING"]    = False
    logmod.ok("backends", f"c engine loaded (pid {sub.proc.pid})")
    return True


def _ensure_c_loaded(model_override=None):
    import threading, time
    def worker():
        target_bin = paths.engine_binary_path()
        if not os.path.isfile(target_bin):
            logmod.info("backends", "c engine binary missing; triggering auto-build")
            build_runner.start()
        deadline = time.time() + 600
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
                    app.config["C_PENDING"] = False
                    return
                time.sleep(0.5)
                continue
            if status == build_runner.STATUS_FAILED:
                logmod.error("backends", f"build failed; not spawning c engine: {s.get('error') or 'no error'}")
                app.config["C_PENDING"] = False
                return
            break
        if not os.path.isfile(target_bin):
            logmod.error("backends", f"engine binary missing after build settled: {target_bin}")
            app.config["C_PENDING"] = False
            return
        exe   = paths.engine_binary_path()
        model = model_override or _resolve_c_model_bin(None)
        if not model or not os.path.isfile(model):
            logmod.error("backends", "no veritate.bin under any model; train + export one first")
            app.config["C_PENDING"] = False
            return
        # Peek act_boost: engine refuses act_boost>1 (non-QAT models produce
        # INT8 gibberish). Surface as "qat_required" so the dashboard disables
        # Generate with a clear message instead of a generic "subprocess died".
        model_dir = os.path.basename(os.path.dirname(model))
        boost = binr.act_boost(model_dir)
        if boost is not None and boost > 1:
            app.config["C_EXE"]            = exe
            app.config["C_MODEL"]          = model
            app.config["C_SUBPROCESS"]     = None
            app.config["C_BLOCKED_REASON"] = "qat_required"
            app.config["C_BLOCKED_MODEL"]  = model_dir
            app.config["C_PENDING"]        = False
            logmod.warn("backends", f"c engine: {model_dir} not QAT-trained (act_boost={boost}); generate disabled until model is QAT-continued or a different bin is selected")
            return
        app.config["C_BLOCKED_REASON"] = None
        app.config["C_BLOCKED_MODEL"]  = None
        _spawn_c_subprocess(exe, model)
    app.config["C_PENDING"] = True
    threading.Thread(target=worker, name="c-backend-loader", daemon=True).start()


@app.route("/backends/c", methods=["POST"])
def backends_c():
    body = request.get_json(silent=True) or {}
    action = (body.get("action") or "").lower()
    if action == "unload":
        sub = app.config.get("C_SUBPROCESS")
        if sub is not None:
            try: sub.close()
            except Exception: pass
            app.config["C_SUBPROCESS"] = None
            app.config["C_EXE"] = None
            app.config["C_MODEL"] = None
            logmod.ok("backends", "c engine unloaded")
        return backends_status()
    if action == "load":
        if app.config.get("C_SUBPROCESS") is not None:
            return backends_status()
        if app.config.get("C_PENDING"):
            return backends_status()
        _ensure_c_loaded(model_override=body.get("model"))
        return backends_status()
    return ({"ok": False, "error": "action must be load or unload"}, 400)


@app.route("/atlas/concept")
def atlas_concept():
    name = request.args.get("model", "")
    step = int(request.args.get("step", "0"))
    substring = request.args.get("substring", "")
    top_k = int(request.args.get("top_k", str(atlas_mod.ATLAS_DEFAULT_TOP_K)))
    if not _safe_name(name):
        return ({"error": "invalid model name"}, 400)
    return atlas_mod.concept_to_neuron(name, step, substring, top_k=top_k)


@app.route("/atlas/neuron/<int:layer>/<int:neuron>")
def atlas_neuron(layer, neuron):
    name = request.args.get("model", "")
    step = int(request.args.get("step", "0"))
    top_k = int(request.args.get("top_k", str(atlas_mod.ATLAS_DEFAULT_TOP_K)))
    if not _safe_name(name):
        return ({"error": "invalid model name"}, 400)
    return atlas_mod.neuron_to_concept(name, step, layer, neuron, top_k=top_k)


@app.route("/atlas/lifetime/<int:layer>/<int:neuron>")
def atlas_lifetime(layer, neuron):
    name = request.args.get("model", "")
    if not _safe_name(name):
        return ({"error": "invalid model name"}, 400)
    return atlas_mod.neuron_lifetime(name, layer, neuron)


@app.route("/atlas/circuit")
def atlas_circuit():
    layer = int(request.args.get("layer", "0"))
    top_k = int(request.args.get("top_k", str(atlas_mod.ATLAS_CIRCUIT_TOP_K)))
    brain = app.config.get("BRAIN")
    return atlas_mod.circuit_graph(brain, layer, top_k=top_k)


@app.route("/train_stream")
def train_stream_route():
    """SSE feed of live training payloads (tier 4). Trainers running with
    --mri-stream call train_stream_mod.publish(payload) per step; this route
    forwards every published payload to subscribed dashboard tabs."""
    def stream():
        try:
            yield "event: ready\ndata: {}\n\n"
            for payload in train_stream_mod.subscribe():
                yield f"data: {json.dumps(payload)}\n\n"
        except GeneratorExit:
            return
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/atlas/concepts_inverted")
def atlas_concepts_inverted():
    name = request.args.get("model", "")
    step = int(request.args.get("step", "0"))
    if not _safe_name(name):
        return ({"error": "invalid model name"}, 400)
    try:
        return atlas_mod.concepts_inverted(name, step)
    except Exception as e:
        logmod.error("atlas", f"concepts_inverted failed: {type(e).__name__}: {e}")
        return ({"error": f"{type(e).__name__}: {e}"}, 500)


@app.route("/addons")
def addons_list():
    try:
        return {"addons": addons_mod.list_addons()}
    except Exception as e:
        logmod.error("addons", f"list failed: {type(e).__name__}: {e}")
        return ({"error": f"{type(e).__name__}: {e}"}, 500)


@app.route("/generate")
def generate():
    prompt        = request.args.get("prompt", "")
    temperature   = float(request.args.get("temperature", "0.7"))
    top_k         = int(request.args.get("top_k", "40"))
    max_new       = int(request.args.get("max_new", "200"))
    backend       = request.args.get("backend", "c").lower()
    ablate_layer  = int(request.args.get("ablate_layer",  "-1"))
    ablate_neuron = int(request.args.get("ablate_neuron", "-1"))
    addons_csv    = request.args.get("addons", "")
    addons_sel    = [s.strip() for s in addons_csv.split(",") if s.strip()]
    # Build-7 additions: fast-decode mode (KV cache / MTP head) and output-
    # shape constraint. Both are PyTorch-backend-only.
    fast_mode     = (request.args.get("fast", "") or "").strip().lower()
    constrained_v = (request.args.get("constrained", "") or "").strip()
    try:
        adaptive_threshold = float(request.args.get("adaptive_threshold", "0.8"))
    except ValueError:
        adaptive_threshold = 0.8
    adaptive_threshold = max(0.0, min(1.0, adaptive_threshold))

    if backend == "c":
        if app.config.get("C_SUBPROCESS") is None:
            blocked = app.config.get("C_BLOCKED_REASON")
            if blocked == "qat_required":
                bm  = app.config.get("C_BLOCKED_MODEL") or "(unknown)"
                msg = (f"Model '{bm}' is not QAT-trained — C engine cannot run it. "
                       f"Switch to the PyTorch backend or retrain with qat_enabled=true.")
            else:
                try:
                    bins = sum(1 for n in models.list_models() if binr.exists(n))
                except Exception:
                    bins = 0
                msg = ("No exported .bin available. Train a model and export it first, "
                       "or switch to the PyTorch backend." if bins == 0
                       else "C engine not loaded. Pick a model from the dropdown.")
            # Stream the error inside SSE so EventSource.onmessage sees it.
            def stream_err():
                yield "data: " + json.dumps({"kind": "error", "message": msg}) + "\n\n"
                yield "event: done\ndata: {}\n\n"
            return Response(stream_err(), mimetype="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        def stream_c():
            try:
                for ev in _c_engine_stream(prompt, max_new, temperature=temperature, top_k=top_k,
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
                        "message": f"c-engine: {type(e).__name__}: {e}",
                    }) + "\n\n"
                    yield "event: done\ndata: {}\n\n"
                except Exception:
                    pass
        return Response(stream_c(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    brain = app.config["BRAIN"]
    if brain is None:
        # Stream the error inside the SSE so the UI can render it. Returning a
        # plain 503 here causes EventSource.onerror with no detail — the user
        # sees a hung "thinking..." with no explanation.
        def stream_err():
            yield "data: " + json.dumps({"kind": "error",
                "message": "PyTorch backend not loaded. Pick a model from the dropdown and try again."}) + "\n\n"
            yield "event: done\ndata: {}\n\n"
        return Response(stream_err(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    app.config["BRAIN_LAST_USED"] = time.time()

    chain = None
    if addons_sel:
        try:
            chain = addons_mod.build_chain(addons_sel)
        except Exception as e:
            logmod.error("addons", f"build_chain failed: {type(e).__name__}: {e}")
            return ({"error": f"addons: {type(e).__name__}: {e}"}, 400)

    constraint = None
    if constrained_v:
        try:
            constraint = _build_constraint(constrained_v)
        except Exception as e:
            logmod.error("constrained", f"build failed: {type(e).__name__}: {e}")
            return ({"error": f"constrained: {type(e).__name__}: {e}"}, 400)

    if fast_mode and fast_mode not in ("kv", "mtp", "mtp-verify", "adaptive"):
        return ({"error": f"unknown fast mode: {fast_mode!r}. Allowed: kv, mtp, mtp-verify, adaptive."}, 400)

    rag_path  = (request.args.get("rag", "") or "").strip()
    rag_k     = request.args.get("rag_k", "")
    rag_press = (request.args.get("rag_compress", "") or "").strip().lower()
    rag_cfg = None
    if rag_path:
        try:
            rag_top_k = max(1, min(int(rag_k), 16)) if rag_k else 3
        except (TypeError, ValueError):
            return ({"error": "rag_k must be an integer 1..16"}, 400)
        # rag_press grammar:
        #   "" / "off"          -> no compression
        #   "crude"             -> heuristic sentence skim (no model call)
        #   "word_ppl"          -> word-level PPL compression, default keep=0.5
        #   "word_ppl:<frac>"   -> same, explicit keep_frac in (0, 1]
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
            return ({"error": f"rag: {type(e).__name__}: {e}"}, 400)
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
                        yield f"data: {json.dumps({'kind': 'error', 'message': f'rag retrieve: {type(e).__name__}: {e}'})}\n\n"
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
                    # Cap inline prefix at 8 KB so SSE payload stays bounded
                    prefix_view = effective_prompt if len(effective_prompt) <= 8192 \
                                  else effective_prompt[:8192] + " ... [trimmed]"
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
                # Surface model-side exceptions as an SSE error event instead
                # of letting the stream close silently. The UI's onerror has
                # no HTTP status to read, so without this the user sees a
                # hung "thinking..." with no explanation.
                logmod.error("generate", f"pytorch stream failed: {type(e).__name__}: {e}")
                try:
                    yield "data: " + json.dumps({
                        "kind": "error",
                        "message": f"generation failed: {type(e).__name__}: {e}",
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
    """Full-glass agent trace stream (I58). Runs the AgentLoop and emits
    every parsed turn event over SSE: turn_start / thought / action /
    observation / answer / schema_err / stop.

    Query params:
      prompt              user input (required)
      corpus              optional BM25 corpus path (registers `retrieve` tool)
      fs_root             optional filesystem-read root (registers `fs_read` tool)
      max_turns           default 6
      best_of_n           default 1
      temperature, top_k  decode knobs (default 0.7, 40)
      seed                base seed (default 0)
    """
    user_input  = request.args.get("prompt", "")
    if not user_input:
        return ({"error": "prompt is required"}, 400)
    brain = app.config.get("BRAIN")
    if brain is None:
        def stream_err():
            yield "data: " + json.dumps({"kind": "error",
                "message": "PyTorch backend not loaded. Pick a model from the dropdown and try again."}) + "\n\n"
            yield "event: stop\ndata: {}\n\n"
        return Response(stream_err(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        max_turns   = max(1, min(int(request.args.get("max_turns", "6")), 16))
        best_of_n   = max(1, min(int(request.args.get("best_of_n", "1")), 8))
        temperature = float(request.args.get("temperature", "0.7"))
        top_k       = int(request.args.get("top_k", "40"))
        seed        = int(request.args.get("seed", "0"))
    except (TypeError, ValueError) as e:
        return ({"error": f"bad query param: {type(e).__name__}: {e}"}, 400)
    corpus_path = (request.args.get("corpus", "") or "").strip() or None
    fs_root     = (request.args.get("fs_root", "") or "").strip() or None
    if corpus_path and not os.path.exists(os.path.expanduser(corpus_path)):
        return ({"error": f"corpus path does not exist: {corpus_path}"}, 400)
    if fs_root and not os.path.isdir(os.path.expanduser(fs_root)):
        return ({"error": f"fs_root must be an existing directory: {fs_root}"}, 400)
    try:
        toolbox = build_default_toolbox(corpus_path=corpus_path, fs_root=fs_root)
    except Exception as e:
        return ({"error": f"toolbox: {type(e).__name__}: {e}"}, 400)
    # Optional explicit tool whitelist: keep only the named tools that are
    # actually registered. Unknown names are dropped silently — the UI may
    # request tools that this request can't provide (e.g. "retrieve" with no
    # corpus path), and rejecting the whole request would be hostile.
    tools_csv = (request.args.get("tools", "") or "").strip()
    if tools_csv:
        wanted = {t.strip() for t in tools_csv.split(",") if t.strip()}
        available = set(toolbox.names())
        toolbox._tools = {n: t for n, t in toolbox._tools.items() if n in wanted and n in available}
        if not toolbox._tools:
            return ({"error": "no usable tools — none of the requested tools are registered (retrieve needs a corpus; fs_read needs a folder)"}, 400)
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
                                           "message": f"{type(e).__name__}: {e}"}) + "\n\n")

    return Response(stream_agent(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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


# RAG tool cache. Keyed by (abs_path, file_signature) so on-disk edits
# rebuild the index instead of serving stale chunks. _make_rag_tool() does
# real I/O + tokenization; we don't want to redo it per request.
_RAG_TOOL_CACHE = {}
_RAG_CACHE_LOCK = threading.Lock()
_RAG_CACHE_MAX  = 8


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
    """Cached BM25 tool for `corpus_path`. Validates path exists, normalizes
    it, and rebuilds the index on disk-edit detection."""
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
        # Evict any other cached signatures for the same path
        for k in list(_RAG_TOOL_CACHE.keys()):
            if k != key and k[0] == abs_path:
                _RAG_TOOL_CACHE.pop(k, None)
        # Bound cache size
        while len(_RAG_TOOL_CACHE) > _RAG_CACHE_MAX:
            _RAG_TOOL_CACHE.pop(next(iter(_RAG_TOOL_CACHE)))
    return tool, abs_path


def _parse_rag_hits(formatted):
    """Parse the retriever Tool's text output into (passages, meta).
    Tool output format per line block: '[src @off] (score 1.23) <preview>'.
    Empty passages on 'no matches' or 'error:'."""
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
                     "preview": (block[:120] + ("…" if len(block) > 120 else ""))})
    return passages, meta


def _build_constraint(spec):
    """Build a decode-time Constraint from a `constrained=` query-param value.

    Supported forms:
      "json"                  -> any JSON value, grammar-valid by construction
      "vocab:<preset>"        -> ascii, alpha, lower, upper, alnum, digits
      "stop:<preset>"         -> newline, double_newline, eos
      "stop:text:<literal>"   -> halt after the literal UTF-8 text is emitted
    """
    from decode import JSONConstraint, VocabConstraint, StopOnConstraint
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


def _resolve_pytorch_model(name):
    if name == "auto":
        candidates = []
        for n in models.list_models():
            if checkpoints.list_steps(n):
                candidates.append((train_csv.file_stat(n).st_mtime if train_csv.file_stat(n) else 0, n))
        if not candidates:
            logmod.warn("backends", "no models with checkpoints under models/. pass --model <name> explicitly.")
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    if not models.exists(name):
        logmod.warn("backends", f"model not found: models/{name}")
        return None
    return name


def _load_pytorch_brain(name, step, threads):
    """Try to load Brain for `name` at `step`. On non-vanilla failure, scan
    other models by recency and load the first vanilla one. Returns
    (brain, name, step) or raises the original RuntimeError if nothing
    vanilla can be loaded."""
    def _try(n, s):
        ck = checkpoints.path_for(n, s)
        mp = os.path.join(paths.model_dir(n), "neuron_memory.json")
        return Brain(ck, threads=threads, memory=load_memory(mp))

    try:
        return (_try(name, step), name, int(step))
    except RuntimeError as e:
        if "PyTorch inference is not enabled" not in str(e):
            raise
        original_exc = e
        original_name = name

    # Fallback: scan other models by recency for a vanilla checkpoint.
    candidates = []
    for n in models.list_models():
        if n == original_name:
            continue
        if not checkpoints.list_steps(n):
            continue
        st = train_csv.file_stat(n)
        candidates.append((st.st_mtime if st else 0, n))
    candidates.sort(reverse=True)
    for _, n in candidates:
        s = checkpoints.latest_step(n)
        if s is None:
            continue
        try:
            brain = _try(n, s)
            logmod.warn("backends", f"pytorch: '{original_name}' is non-vanilla; auto-fell-back to '{n}' step {s}")
            return (brain, n, int(s))
        except RuntimeError as e2:
            if "PyTorch inference is not enabled" in str(e2):
                continue
            raise
    raise original_exc


def _resolve_c_model_bin(name):
    if name and os.path.isfile(name): return name
    if name and models.exists(name) and binr.exists(name):
        return paths.bin_path(name)
    candidates = []
    for n in models.list_models():
        if not binr.exists(n): continue
        bp = paths.bin_path(n)
        try: st = os.stat(bp)
        except OSError: continue
        candidates.append((st.st_mtime, bp))
    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else None


def _resolve_c_engine_exe(explicit):
    if explicit and os.path.isfile(explicit): return explicit
    for e in engine.engines():
        ap = os.path.abspath(e.get("path") or "")
        if os.path.isfile(ap): return ap
    return None


def _pytorch_idle_watcher():
    """Background daemon. When pytorch_load_mode == 'on_demand' and the brain
    has been idle longer than pytorch_idle_unload_secs, unload it. Skips while
    a generation/neuron lookup holds brain.lock so we never unload mid-stream."""
    while True:
        time.sleep(30)
        try:
            s = settings_mod.get()
            if s.get("pytorch_load_mode") != "on_demand":
                continue
            brain = app.config.get("BRAIN")
            if brain is None:
                continue
            if brain.lock.locked():
                continue
            idle_for = time.time() - (app.config.get("BRAIN_LAST_USED") or 0)
            if idle_for >= float(s.get("pytorch_idle_unload_secs") or 600):
                app.config["BRAIN"] = None
                app.config["BRAIN_MODEL"] = None
                app.config["BRAIN_STEP"]  = None
                logmod.ok("backends", f"pytorch auto-unloaded (idle {int(idle_for)}s)")
        except Exception as e:
            logmod.error("backends", f"idle watcher: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="auto", help="default model for both backends. 'auto' picks the freshest.")
    ap.add_argument("--step",    type=int, default=None)
    ap.add_argument("--port",    type=int, default=8001)
    ap.add_argument("--threads", type=int, default=0,
                    help="pytorch CPU threads. 0 = auto: physical cores capped at 16.")
    args = ap.parse_args()

    name = _resolve_pytorch_model(args.model)
    if name is not None:
        app.config["DEFAULT_MODEL"] = name
        app.config["DEFAULT_STEP"]  = args.step if args.step is not None else checkpoints.latest_step(name)
    threads = args.threads if args.threads and args.threads > 0 else auto_thread_count()
    app.config["DEFAULT_THREADS"] = threads
    logmod.info("run", f"default model: {name or '(none)'}")
    logmod.info("run", f"pytorch threads: {threads}{' (auto)' if not args.threads else ''}")

    def _close_c_for_rebuild():
        sub = app.config.get("C_SUBPROCESS")
        if sub is None:
            return
        try:
            sub.close()
        except Exception:
            pass
        app.config["C_SUBPROCESS"] = None
        logmod.warn("build", "closed C engine subprocess to release binary lock")
    build_runner.set_pre_build_hook(_close_c_for_rebuild)

    threading.Thread(target=_pytorch_idle_watcher, name="pytorch-idle-watcher", daemon=True).start()
    sys_metrics.warm()

    def _heartbeat_training():
        # Enriched training payload: plugin id + started_at, plus model name
        # and shape/params pulled from the model's config.json. The heartbeat
        # tier logic decides which of these fields actually ship (analytics
        # tier: full block; minimal: only "training_active" presence).
        st = plugin_runner.state()
        if not st or st.get("status") != plugin_runner.STATUS_RUNNING:
            return None
        out = {
            "plugin_id":  st.get("plugin_id"),
            "started_at": st.get("started_at"),
        }
        args = st.get("args") or {}
        if isinstance(args, dict):
            name = args.get("name") or args.get("model")
            if name and models.exists(name):
                try:
                    cfg = cfg_reader.load(name) or {}
                    shape = cfg.get("shape") or {}
                    out["model_name"] = name
                    out["n_params"]   = int(cfg.get("n_params_total") or 0) or None
                    keep = ("hidden", "layers", "ffn", "heads", "seq", "n_predict", "rope_base")
                    summary = {k: shape[k] for k in keep if k in shape}
                    if summary:
                        out["shape"] = summary
                except Exception:
                    pass
        return out
    heartbeat_mod.set_training_provider(_heartbeat_training)
    heartbeat_mod.start()

    def _app_sync_reload():
        lifecycle.restart(app.config)
    app_sync_mod.set_reload_hook(_app_sync_reload)
    app_sync_mod.start()

    # Eager-load the pytorch backend OFF the main thread so app.run() starts
    # serving immediately. Only fires when settings say `always`; in the
    # default `on_demand` mode the brain loads when the user actually clicks
    # Generate, and idle-watcher unloads it after inactivity.
    if (settings_mod.get().get("pytorch_load_mode") == "always"
            and app.config.get("DEFAULT_MODEL") is not None
            and app.config.get("DEFAULT_STEP")  is not None):
        def _eager_load():
            try:
                app.config["PYTORCH_PENDING"] = True
                n, st = app.config["DEFAULT_MODEL"], app.config["DEFAULT_STEP"]
                brain, n2, st2 = _load_pytorch_brain(n, st, threads)
                app.config["BRAIN"] = brain
                app.config["BRAIN_MODEL"] = n2
                app.config["BRAIN_STEP"]  = int(st2)
                app.config["DEFAULT_MODEL"] = n2
                app.config["DEFAULT_STEP"]  = int(st2)
                app.config["BRAIN_LAST_USED"] = time.time()
                app.config["BRAIN_LAST_ERROR"] = None
                logmod.ok("backends", f"pytorch eager-loaded: {n2} step {st2} ({brain.n_params:,} params)")
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                app.config["BRAIN_LAST_ERROR"] = msg
                cur = app.config.get("DEFAULT_MODEL")
                if isinstance(e, RuntimeError) and "PyTorch inference is not enabled" in str(e):
                    logmod.warn("backends", f"pytorch backend skipped for {cur}: non-vanilla architecture (use C engine)")
                else:
                    logmod.error("backends", f"pytorch eager load failed: {msg}")
            finally:
                app.config["PYTORCH_PENDING"] = False
        threading.Thread(target=_eager_load, name="pytorch-eager-load", daemon=True).start()

    print(f"http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True,
            request_handler=_QuietWSGIRequestHandler)


class _QuietWSGIRequestHandler(WSGIRequestHandler):
    # Python 3.14 + werkzeug threaded dev server has a race where socketserver
    # closes the connection in one thread while the handler is still reading
    # from it in another, surfacing as OSError(EBADF) from recv_into. Browser
    # preconnect/keep-alive churn triggers this constantly. Werkzeug already
    # swallows ConnectionError/socket.timeout in handle(); EBADF is not in
    # either bucket, so we extend the same idea narrowly.
    def handle(self):
        try:
            super().handle()
        except OSError as e:
            if e.errno == errno.EBADF:
                return
            raise


if __name__ == "__main__":
    main()
