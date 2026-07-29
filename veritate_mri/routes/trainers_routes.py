# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - trainer catalog, start/stop, core-trainers index, open-folder.
#   Enforces fresh-run name collisions on /trainers/run. Every endpoint runs
#   through _safe so any exception lands in the dashboard log ring with a
#   parseable JSON error body (avoids the WebKit "string did not match the
#   expected pattern" symptom users hit when an HTML 500 reaches r.json()).
# veritate_mri/routes/trainers_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request
from readers import models
from readers import trainers as trainers_reader
from training import trainer_runner

from ._common import open_folder
from ._common import safe_route as _safe


def register(app):
    @app.route("/trainers")
    def trainers_index():
        return _safe("trainers", lambda: {
            "trainers": trainers_reader.scan(),
            "running":  trainer_runner.state(),
        })

    @app.route("/trainers/run", methods=["POST"])
    def trainers_run():
        def _do():
            body = request.get_json(silent=True) or {}
            trainer_id = body.get("id")
            if not trainer_id:
                return ({"ok": False, "error": "missing 'id'"}, 400)
            args = body.get("args") or {}
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
            return trainer_runner.start(trainer_id, args)
        return _safe("trainers", _do)

    @app.route("/trainers/stop", methods=["POST"])
    def trainers_stop():
        return _safe("trainers", trainer_runner.stop)

    @app.route("/trainers/tune_defaults", methods=["POST"])
    def trainers_tune_defaults():
        """Persist auto-tune results as machine-local trainer tuning + saved
        specs, without launching a run. Body: {id, args:{...}, measured:{...},
        sysprobe:{...}}. When sysprobe is present it's uploaded to the
        Carpathian heartbeat endpoint (bench_report kind) if the user has
        opted into advanced analytics. trainer_sizes.json is never touched."""
        def _do():
            from runtime import heartbeat, sys_metrics
            body = request.get_json(silent=True) or {}
            trainer_id = body.get("id")
            if not trainer_id:
                return ({"ok": False, "error": "missing 'id'"}, 400)
            saved = trainers_reader.update_defaults(trainer_id, body.get("args") or {})
            if body.get("measured"):
                sys_metrics.save_measured(body["measured"])
            upload = heartbeat.send_bench_report(
                bench_result=body.get("measured"),
                sysprobe_result=body.get("sysprobe"),
                trainer_id=trainer_id,
            )
            return {"ok": True, "saved": bool(saved), "upload": upload}
        return _safe("trainers", _do)

    @app.route("/trainers/sysprobe", methods=["POST"])
    def trainers_sysprobe():
        """Run the full hardware benchmark suite (disk write, CPU FLOPs +
        bandwidth, GPU FLOPs on every accelerator, RAM headroom). Returns the
        raw probe dict. Cheap (~1-3 s on modern boxes), safe to call before or
        after the model-specific bench."""
        def _do():
            from veritate_core.plugin import sysprobe
            body = request.get_json(silent=True) or {}
            disk_dir = body.get("disk_dir")
            result = sysprobe.run(disk_dir=disk_dir)
            return {"ok": True, "sysprobe": result}
        return _safe("trainers", _do)

    @app.route("/system/install_helper", methods=["POST"])
    def system_install_helper():
        """Install a native OS helper (temperature sensors, etc). Body:
        {helper: 'mac_temp_arm' | 'mac_temp_intel' | 'linux_lm_sensors'}.
        Windows LibreHardwareMonitor isn't auto-installable (GUI installer);
        the dashboard shows the manual link for that case."""
        def _do():
            from veritate_core.plugin import deps
            body = request.get_json(silent=True) or {}
            helper = (body.get("helper") or "").strip()
            if not helper:
                return ({"ok": False, "error": "missing 'helper'"}, 400)
            result = deps.install_helper(helper)
            return {"ok": bool(result.get("ok")), **result}
        return _safe("trainers", _do)

    @app.route("/system/install_dep", methods=["POST"])
    def system_install_dep():
        """Install a missing Python package with automatic escalation. Body:
        {pkg: 'name', index_url?: '...'}. Blocks until pip returns (up to
        PIP_TIMEOUT_SECS). Returns {ok, method, needs_elevation, stdout,
        stderr}. Called by the Auto tune modal when bench aborts with an
        ImportError."""
        def _do():
            from veritate_core.plugin import deps
            body = request.get_json(silent=True) or {}
            pkg = (body.get("pkg") or "").strip()
            if not pkg:
                return ({"ok": False, "error": "missing 'pkg'"}, 400)
            index_url = body.get("index_url") or None
            result = deps.ensure(pkg, index_url=index_url)
            return {"ok": bool(result.get("ok")), **result}
        return _safe("trainers", _do)

    @app.route("/core_trainers")
    def core_trainers_index():
        def _do():
            from veritate_core import core_plugins as _cp
            flow = (request.args.get("flow") or "").strip() or None
            return {"trainers": _cp.all_plugins(flow=flow)}
        return _safe("trainers", _do)

    @app.route("/trainers/open_folder", methods=["POST"])
    def trainers_open_folder():
        return _safe("trainers", lambda: open_folder(trainers_reader.PLUGINS_ROOT))
