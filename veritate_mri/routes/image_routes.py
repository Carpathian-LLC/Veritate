# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - IDEA 24's F0 falsifier surface: decoder latency and peak activation bytes at a
#   target output resolution, at random weights. Synchronous like /trainers/sysprobe,
#   because the caller sizes the cost through arms/reps/resolution and a researcher
#   runs it deliberately rather than the UI polling it.
# - Cost is the caller's to choose but not without bound: MAX_EDGE stops a mistyped
#   resolution from allocating the box out from under a training run.
# veritate_mri/routes/image_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import request

from ._common import safe_route as _safe

# ------------------------------------------------------------------------------------
# Constants

# Guard, not a tunable: 8K wide is past any target in IDEA 24 and a typo one digit
# longer would allocate tens of GB in the control arm.
MAX_EDGE = 8192

# ------------------------------------------------------------------------------------
# Functions


def register(app):
    @app.route("/images/decode_bench", methods=["POST"])
    def images_decode_bench():
        """F0 for IDEA 24. Decode one frame per arm at random weights and report
        latency, achieved GF/s and peak activation bytes. The peak is the deciding
        number: the design's rule is that no tensor whose extent is the output
        resolution may be materialized, and the `conv_full` arm is the control that
        breaks it. No weights are loaded and nothing is saved."""
        def _do():
            from veritate_core.plugin import image_decode

            body = request.get_json(silent=True) or {}
            height = int(body.get("height", 1080))
            width  = int(body.get("width", 1920))
            if not (0 < height <= MAX_EDGE and 0 < width <= MAX_EDGE):
                return {"ok": False, "error": "height and width must be in 1.." + str(MAX_EDGE)}, 400
            kwargs = {k: body[k] for k in
                      ("arms", "latent_ch", "mlp_width", "tile", "patch", "code_emb",
                       "patch_hidden", "band", "conv_ch", "grid_div", "warmup", "reps",
                       "device", "seed") if k in body}
            if "arms" in kwargs:
                kwargs["arms"] = tuple(kwargs["arms"])
            if "conv_ch" in kwargs:
                kwargs["conv_ch"] = tuple(kwargs["conv_ch"])
            return {"ok": True, "report": image_decode.bench(height, width, **kwargs)}
        return _safe("images", _do)
