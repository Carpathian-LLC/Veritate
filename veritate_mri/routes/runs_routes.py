# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - run / timeline artifact endpoints. csv, probes index, classroom suite,
#   config, coactivation, learning rate, surprise atlas, full timeline,
#   eval_deep GET / POST / status, plus the legacy /timelines and
#   /timeline/<>/<> compat paths.
# veritate_mri/routes/runs_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import json
import os
import re
import threading
import time

from flask import Response, current_app, request, send_from_directory

from readers import (
    capabilities as caps_reader, checkpoints, config as cfg_reader, hooks,
    models, paths, train_csv,
)
from runtime import logs as logmod

from . import _brain
from ._common import auto_thread_count, safe_name, user_error

# ------------------------------------------------------------------------------------
# Constants

COACT_THRESHOLD     = 0.5
COACT_TOP_PAIRS     = 200
LR_TOP_ROWS         = 512
EVAL_DEEP_SUBDIR    = "eval_deep"
IFEVAL_DEFAULT_MAXN = 256
TIMELINE_FNAME_RE   = re.compile(r"^(probe|classroom|grades|math|grammar|reasoning|concepts|surprise|quant_kl|writing_health|reading_comprehension)_step_(\d+)\.json$")
TIMELINE_STEP_RE    = re.compile(r"^step_(\d+)\.json$")
TIMELINE_LENS_RE    = re.compile(r"^lens_step_(\d+)\.npz$")

_EVAL_DEEP_STATE = {"runs": {}, "lock": threading.Lock()}

# ------------------------------------------------------------------------------------
# Functions

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
                if abs(float(n.get("v", 0.0))) > COACT_THRESHOLD:
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
    pairs = pairs[:COACT_TOP_PAIRS]
    nodes_set = set()
    for p in pairs:
        nodes_set.add((p["i"][0], p["i"][1]))
        nodes_set.add((p["j"][0], p["j"][1]))
    nodes = [{"layer": L, "neuron": n, "fires": layer_count[(L, n)]}
             for (L, n) in sorted(nodes_set)]
    return ({"step": step, "n_tokens": n_tokens, "threshold": COACT_THRESHOLD,
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
    return ({"step": step, "prior_step": prior, "neurons": rows[:LR_TOP_ROWS]}, 200)


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


def _timeline_payload(name):
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


def _eval_deep_dir(name):
    return os.path.join(paths.model_dir(name), EVAL_DEEP_SUBDIR)


def _eval_deep_summary(blob):
    if not isinstance(blob, dict):
        return []
    if "report" in blob and isinstance(blob["report"], dict) and "suite" in blob:
        sub = {blob["suite"]: blob["report"]}
    else:
        sub = blob.get("suites", {}) or {}
    out = []
    for suite_name, s in sub.items():
        if not isinstance(s, dict): continue
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


def _resolve_eval_brain(cfg, name, step, threads):
    cur_brain = cfg.get("BRAIN")
    cur_name  = cfg.get("BRAIN_MODEL")
    cur_step  = cfg.get("BRAIN_STEP")
    if cur_brain is not None and cur_name == name and int(cur_step or -1) == int(step):
        return cur_brain, "reused"
    brain, n_, s_ = _brain.load_pytorch_brain(name, step, threads)
    return brain, "loaded"


def register(app):
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
                "capabilities": caps_reader.read(name),
            })
        out.sort(key=lambda r: -r["mtime"])
        return {"runs": out}

    @app.route("/run/<path:name>/csv")
    def run_csv(name):
        if not safe_name(name) or not models.exists(name): return ("run not found", 404)
        if not train_csv.is_present(name): return ("csv not found", 404)
        text = train_csv.raw_text(name)
        return Response(text, mimetype="text/csv")

    @app.route("/run/<path:name>/probes")
    def run_probes(name):
        if not safe_name(name) or not models.exists(name):
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
        if not safe_name(name) or not models.exists(name):
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
        if not safe_name(name) or not models.exists(name):
            return ({"error": "config.json not found", "model_dir": name}, 404)
        data = cfg_reader.load(name)
        if data is None: return ({"error": "config.json not found", "model_dir": name}, 404)
        return Response(json.dumps(data), mimetype="application/json")

    @app.route("/run/<path:name>/coactivation/<int:step>")
    def run_coactivation(name, step):
        if not safe_name(name) or not models.exists(name): return ("not found", 404)
        body, status = _coactivation(name, step)
        return body, status

    @app.route("/run/<path:name>/learning_rate/<int:step>")
    def run_learning_rate(name, step):
        if not safe_name(name) or not models.exists(name): return ("not found", 404)
        body, status = _learning_rate(name, step)
        return body, status

    @app.route("/run/<path:name>/surprise")
    def run_surprise(name):
        if not safe_name(name) or not models.exists(name): return ("not found", 404)
        body, status = _surprise_atlas(name)
        return body, status

    @app.route("/run/<path:name>/eval_deep", methods=["GET"])
    def run_eval_deep_list(name):
        """List all cached deep-eval results for a model."""
        if not safe_name(name) or not models.exists(name):
            return ({"error": "model not found"}, 404)
        root = _eval_deep_dir(name)
        out = []
        if os.path.isdir(root):
            for fn in sorted(os.listdir(root)):
                if not fn.endswith(".json"): continue
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

    @app.route("/run/<path:name>/eval_deep/status")
    def run_eval_deep_status(name):
        step_s = request.args.get("step")
        try: step = int(step_s) if step_s is not None else None
        except Exception: step = None
        if step is None:
            step = checkpoints.latest_step(name)
            if step is None:
                return ({"running": False, "error": "no checkpoints"}, 200)
        return _eval_state_get(name, step) or {"running": False}

    @app.route("/run/<path:name>/eval_deep", methods=["POST"])
    def run_eval_deep_post(name):
        """Trigger a deep eval."""
        cfg = current_app.config
        if not safe_name(name) or not models.exists(name):
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

        # MMLU/HellaSwag/IFEval are language benchmarks; refuse them for a model
        # declared code/statistical/other so a non-text model never gets a bogus score.
        mtype = ((cfg_reader.load(name) or {}).get("training_args") or {}).get("model_type")
        if mtype and str(mtype).lower() != "language":
            return ({"error": f"deep eval is language-only; model {name!r} is type {mtype!r}."}, 400)

        step = body.get("step")
        if step in (None, "", "latest"):
            step = checkpoints.latest_step(name)
        if step is None:
            return ({"error": f"no checkpoints under models/{name}/"}, 400)
        try: step = int(step)
        except Exception: return ({"error": f"bad step: {step!r}"}, 400)

        limit       = body.get("limit")
        mmlu_mode   = body.get("mmlu_mode") or "text"
        ifeval_maxn = int(body.get("ifeval_max_new") or IFEVAL_DEFAULT_MAXN)
        threads     = int(body.get("threads") or cfg.get("DEFAULT_THREADS") or auto_thread_count())

        t_load0 = time.time()
        try:
            brain, source = _resolve_eval_brain(cfg, name, step, threads)
        except Exception as e:
            logmod.error("eval_deep", f"load failed for {name} step {step}: {type(e).__name__}: {e}")
            return ({"error": user_error(e)}, 500)
        load_secs = time.time() - t_load0
        logmod.ok("eval_deep", f"brain ready ({source}) for {name} step {step} in {load_secs:.1f}s")

        _eval_state_set(name, step,
                        running=True, started=time.time(),
                        suites=suites, source=source, error=None,
                        progress={s: {"i": 0, "n": None} for s in suites},
                        current_suite=None, finished=None)

        def _progress(suite_name, i, n):
            _eval_state_set(name, step,
                            current_suite=suite_name,
                            progress_update={"suite": suite_name, "i": i, "n": n})
            st = _eval_state_get(name, step)
            prog = st.get("progress") or {}
            prog[suite_name] = {"i": i, "n": n}
            _eval_state_set(name, step, progress=prog)

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
            logmod.error("eval_deep", f"run failed for {name} step {step}: {type(e).__name__}: {e}")
            msg = user_error(e)
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
        if not safe_name(name) or not safe_name(fname): return ("bad name", 400)
        if not models.exists(name): return ("not found", 404)
        if fname == "timeline.json":
            return _timeline_payload(name)
        m = TIMELINE_FNAME_RE.match(fname)
        if m:
            data = hooks.load_artifact(name, int(m.group(2)), m.group(1))
            if data is None: return ("artifact not found", 404)
            return Response(json.dumps(data), mimetype="application/json")
        m = TIMELINE_STEP_RE.match(fname)
        if m:
            data = hooks.load_artifact(name, int(m.group(1)), "generation")
            if data is None: return ("artifact not found", 404)
            return Response(json.dumps(data), mimetype="application/json")
        m = TIMELINE_LENS_RE.match(fname)
        if m:
            p = paths.hook_artifact_path(name, int(m.group(1)), "lens")
            if not os.path.isfile(p): return ("artifact not found", 404)
            return send_from_directory(os.path.dirname(p), os.path.basename(p))
        return ("not found", 404)

    @app.route("/run/<path:name>/timeline")
    def run_timeline(name):
        if not safe_name(name) or not models.exists(name): return ("not found", 404)
        return _timeline_payload(name)
