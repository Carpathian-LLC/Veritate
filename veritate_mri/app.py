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
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
from flask import Flask, Response, request, send_from_directory

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
import sys_metrics
import settings as settings_mod
import heartbeat as heartbeat_mod
import ai_assist as ai_assist_mod
import app_sync as app_sync_mod
import threading

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
                     ablate_layer=-1, ablate_neuron=-1):
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
                              ablate_layer=ablate_layer, ablate_neuron=ablate_neuron):
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


@app.route("/multimind")
def multimind():
    return send_from_directory(STATIC_DIR, "multimind.html")


@app.route("/multimind/results")
def multimind_results():
    root = os.path.normpath(os.path.join(HERE, "..", "docs", "results", "multimind"))
    out = {"root": root, "files": {}}
    if not os.path.isdir(root):
        return out
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".json"): continue
        p = os.path.join(root, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                out["files"][fn] = json.load(f)
        except Exception as e:
            out["files"][fn] = {"_load_error": str(e)}
    return out


@app.route("/sys_metrics")
def sys_metrics_route():
    return sys_metrics.snapshot()


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
    res = app_sync_mod.pull()
    if res.get("ok") and body.get("reload"):
        try:
            lifecycle.soft_reload(app.config)
        except Exception as e:
            res["reload_error"] = f"{type(e).__name__}: {e}"
    return res


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
        out = settings_mod.update(body)
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
                    ckpt = checkpoints.path_for(name, step)
                    mem_path = os.path.join(paths.model_dir(name), "neuron_memory.json")
                    threads = int(app.config.get("DEFAULT_THREADS") or auto_thread_count())
                    brain = Brain(ckpt, threads=threads, memory=load_memory(mem_path))
                    app.config["BRAIN"] = brain
                    app.config["BRAIN_MODEL"] = name
                    app.config["BRAIN_STEP"]  = int(step)
                    app.config["BRAIN_LAST_USED"] = time.time()
                    app.config["BRAIN_LAST_ERROR"] = None
                    logmod.ok("backends", f"pytorch eager-loaded after settings flip: {name} step {step}")
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                app.config["BRAIN_LAST_ERROR"] = msg
                logmod.error("backends", f"pytorch eager load on settings flip failed: {msg}")
        return out
    return settings_mod.get()


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
        s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = s.get("args", {})
        sd = s["model"]
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
        seq = sd["pos_emb.weight"].shape[0]
        heads = int(cfg.get("heads") or 0)
        if heads <= 0 or hidden % heads != 0:
            target = max(1, hidden // 64)
            for h in sorted({d for d in range(1, hidden + 1) if hidden % d == 0},
                            key=lambda d: (abs(d - target), -d)):
                heads = h
                break
        ffn_arg = ffn_per_layer if len(set(ffn_per_layer)) > 1 else ffn_per_layer[0]
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
        s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = s.get("args", {})
        sd = s["model"]
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
        seq = sd["pos_emb.weight"].shape[0]
        heads = int(cfg.get("heads") or 0)
        if heads <= 0 or hidden % heads != 0:
            target = max(1, hidden // 64)
            for h in sorted({d for d in range(1, hidden + 1) if hidden % d == 0},
                            key=lambda d: (abs(d - target), -d)):
                heads = h; break
        ffn_arg = ffn_per_layer if len(set(ffn_per_layer)) > 1 else ffn_per_layer[0]
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
    return plugins_sync.sync()


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


@app.route("/models/git/status")
def models_git_status():
    return models_sync.status()


@app.route("/models/git/sync", methods=["POST"])
def models_git_sync():
    return models_sync.sync()


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
        for kind in ("classroom", "grades", "concepts"):
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
    m = re.match(r"^(probe|classroom|grades|concepts|surprise|quant_kl)_step_(\d+)\.json$", fname)
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
    try:
        sub = CTracedSubprocess(new_exe, new_model)
    except Exception as e:
        app.config["C_SUBPROCESS"] = None
        return ({"ok": False, "error": f"respawn failed: {e}"}, 500)
    app.config["C_EXE"]        = new_exe
    app.config["C_MODEL"]      = new_model
    app.config["C_SUBPROCESS"] = sub
    print(f"c-config: exe={new_exe} model={new_model} pid={sub.proc.pid}")
    name = os.path.basename(os.path.dirname(new_model)) if new_model else None
    precision, version = (binr.header(name) if name else ("?", 0))
    training, activation = (cfg_reader.training_kind(name) if name else ("", ""))
    boost = binr.act_boost(name) if name else None
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
            "build":     build_runner.state(),
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
        if app.config.get("BRAIN") is not None:
            return backends_status()
        name = body.get("model") or app.config.get("DEFAULT_MODEL")
        if not name or not models.exists(name):
            name = _resolve_pytorch_model("auto")
            if name is not None:
                app.config["DEFAULT_MODEL"] = name
                app.config["DEFAULT_STEP"]  = checkpoints.latest_step(name)
        if not name or not models.exists(name):
            return ({"ok": False, "error": "no models with checkpoints under models/. train one first or pass an explicit model name."}, 400)
        step = body.get("step") or app.config.get("DEFAULT_STEP") or checkpoints.latest_step(name)
        if step is None:
            return ({"ok": False, "error": f"no checkpoints under models/{name}/"}, 400)
        ckpt = checkpoints.path_for(name, step)
        threads = int(body.get("threads") or app.config.get("DEFAULT_THREADS") or auto_thread_count())
        try:
            mem_path = os.path.join(paths.model_dir(name), "neuron_memory.json")
            brain = Brain(ckpt, threads=threads, memory=load_memory(mem_path))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            app.config["BRAIN_LAST_ERROR"] = msg
            logmod.error("backends", f"pytorch load failed: {msg}")
            return ({"ok": False, "error": msg}, 500)
        app.config["BRAIN"] = brain
        app.config["BRAIN_MODEL"] = name
        app.config["BRAIN_STEP"]  = int(step)
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


@app.route("/generate")
def generate():
    prompt        = request.args.get("prompt", "")
    temperature   = float(request.args.get("temperature", "0.7"))
    top_k         = int(request.args.get("top_k", "40"))
    max_new       = int(request.args.get("max_new", "200"))
    backend       = request.args.get("backend", "c").lower()
    ablate_layer  = int(request.args.get("ablate_layer",  "-1"))
    ablate_neuron = int(request.args.get("ablate_neuron", "-1"))

    if backend == "c":
        if app.config.get("C_SUBPROCESS") is None:
            return ({"error": "c engine not loaded. POST /backends/c {action: load} first."}, 503)
        def stream_c():
            try:
                for ev in _c_engine_stream(prompt, max_new, temperature=temperature, top_k=top_k,
                                           ablate_layer=ablate_layer, ablate_neuron=ablate_neuron):
                    yield f"data: {json.dumps(ev)}\n\n"
                yield "event: done\ndata: {}\n\n"
            except GeneratorExit:
                return
        return Response(stream_c(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    brain = app.config["BRAIN"]
    if brain is None:
        return ({"error": "pytorch backend not loaded. POST /backends/pytorch {action: load} first."}, 503)
    app.config["BRAIN_LAST_USED"] = time.time()

    def stream_pt():
        with brain.lock:
            try:
                brain.set_ablation(ablate_layer, ablate_neuron)
                for ev in brain.stream(prompt, temperature, top_k, max_new):
                    ev["backend"] = "pytorch"
                    yield f"data: {json.dumps(ev)}\n\n"
                yield "event: done\ndata: {}\n\n"
            except GeneratorExit:
                return
            finally:
                brain.set_ablation(-1, -1)

    return Response(stream_pt(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _resolve_pytorch_model(name):
    if name == "auto":
        candidates = []
        for n in models.list_models():
            if checkpoints.list_steps(n):
                candidates.append((train_csv.file_stat(n).st_mtime if train_csv.file_stat(n) else 0, n))
        if not candidates:
            print("no models with checkpoints under models/. pass --model <name> explicitly.")
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]
    if not models.exists(name):
        print(f"model not found: models/{name}")
        return None
    return name


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
        st = plugin_runner.state()
        if not st or st.get("status") != plugin_runner.STATUS_RUNNING:
            return None
        return {
            "plugin_id": st.get("plugin_id"),
            "started_at": st.get("started_at"),
        }
    heartbeat_mod.set_training_provider(_heartbeat_training)
    heartbeat_mod.start()

    def _app_sync_reload():
        lifecycle.soft_reload(app.config)
    app_sync_mod.set_reload_hook(_app_sync_reload)
    app_sync_mod.start()

    try:
        s = settings_mod.get()
        step = app.config.get("DEFAULT_STEP")
        if s.get("pytorch_load_mode") == "always" and name is not None and step is not None:
            ckpt = checkpoints.path_for(name, step)
            mem_path = os.path.join(paths.model_dir(name), "neuron_memory.json")
            brain = Brain(ckpt, threads=threads, memory=load_memory(mem_path))
            app.config["BRAIN"] = brain
            app.config["BRAIN_MODEL"] = name
            app.config["BRAIN_STEP"]  = int(step)
            app.config["BRAIN_LAST_USED"] = time.time()
            app.config["BRAIN_LAST_ERROR"] = None
            logmod.ok("backends", f"pytorch eager-loaded: {name} step {step} ({brain.n_params:,} params)")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        app.config["BRAIN_LAST_ERROR"] = msg
        logmod.error("backends", f"pytorch eager load failed: {msg}")

    print(f"http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
