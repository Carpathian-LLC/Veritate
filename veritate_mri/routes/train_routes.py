# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - training discovery (corpus + models with checkpoints), per-stem usage,
#   live training stream sse feed.
# veritate_mri/routes/train_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import json

from flask import Response
from readers import checkpoints, models
from readers import corpus as corpus_reader
from readers import images as images_reader
from training import train_stream as train_stream_mod

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def _training_kind(name):
    """'image' for a picture model, 'text' otherwise (what config.json says, or text)."""
    import json

    from readers import paths
    try:
        with open(paths.config_path(name), encoding="utf-8") as handle:
            return json.load(handle).get("training") or "text"
    except (OSError, ValueError):
        return "text"


def register(app):
    @app.route("/train/discovery")
    def train_discovery():
        out_models = []
        for name in models.list_models():
            steps = checkpoints.list_steps(name)
            if not steps: continue
            # the kind decides which form may continue it: image models continue from the
            # Images action, text models from Continue a saved model -- never the other
            out_models.append({"name": name, "steps": steps, "training": _training_kind(name)})
        out_models.sort(key=lambda r: r["name"])
        return {
            "corpora":    corpus_reader.list_stems(),
            "models":     out_models,
            "image_sets": images_reader.list_sets(),
            "codecs":     images_reader.list_codecs(),
        }

    @app.route("/corpus/<path:stem>/usage")
    def corpus_usage(stem):
        if ".." in stem or stem.startswith(("/", "\\")):
            return ("bad stem", 400)
        data = corpus_reader.usage(stem)
        if data is None:
            return ({"error": f"corpus stem not found: {stem}"}, 404)
        return data

    @app.route("/train_stream")
    def train_stream_route():
        """SSE feed of live training payloads (tier 4)."""
        def stream():
            try:
                yield "event: ready\ndata: {}\n\n"
                for payload in train_stream_mod.subscribe():
                    if payload is None:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(payload)}\n\n"
            except GeneratorExit:
                return
        return Response(stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
