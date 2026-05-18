# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - pruning report, pruning plugin generation, model bin export. heavy
#   imports are lazy so cold pages avoid the cost.
# veritate_mri/routes/pruning_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from flask import request

from runtime import logs as logmod
from readers import checkpoints, models, paths

from ._common import user_error

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
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
            from training import pruning as pruning_mod
            from veritate_core.load import load_from_state_dict
            ckpt_path = checkpoints.path_for(name, step)
            s = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            cfg = dict(s.get("args", {}))
            sd = s["model"]
            del s
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
            m = load_from_state_dict(sd, cfg, strict_canonical=False)
            seq = m.seq
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
                3 * hidden * hidden + hidden * hidden + 2 * hidden
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
            return ({"ok": False, "error": user_error(e)}, 500)

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
            from training import pruning as pruning_mod
            from veritate_core.load import load_from_state_dict
            ckpt_path = checkpoints.path_for(name, step)
            s = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            cfg = dict(s.get("args", {}))
            sd = s["model"]
            del s
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
            m = load_from_state_dict(sd, cfg, strict_canonical=False)
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
            return ({"ok": False, "error": user_error(e)}, 500)

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
            from training import export as export_mod
            result = export_mod.export_checkpoint(name, int(step))
        except (FileNotFoundError, ValueError, KeyError) as e:
            logmod.error("export", f"{name} step {step}: {type(e).__name__}: {e}")
            return ({"ok": False, "error": user_error(e)}, 400)
        except Exception as e:
            logmod.error("export", f"{name} step {step}: {type(e).__name__}: {e}")
            return ({"ok": False, "error": user_error(e)}, 500)
        logmod.ok("export", f"{name} step {step}: wrote {result['path']} ({result['bytes']} bytes)")
        return {"ok": True, **result}
