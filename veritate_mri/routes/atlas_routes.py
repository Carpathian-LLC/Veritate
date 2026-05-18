# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - concept/neuron atlas + circuit graph + concepts inverted view. brain is
#   read off app.config; routes are read-only over hook artifacts otherwise.
# veritate_mri/routes/atlas_routes.py
# ------------------------------------------------------------------------------------
# Imports:

from flask import current_app, request

from runtime import logs as logmod
from training import atlas as atlas_mod

from ._common import safe_name, user_error

# ------------------------------------------------------------------------------------
# Constants


# ------------------------------------------------------------------------------------
# Functions

def register(app):
    @app.route("/atlas/concept")
    def atlas_concept():
        name = request.args.get("model", "")
        step = int(request.args.get("step", "0"))
        substring = request.args.get("substring", "")
        top_k = int(request.args.get("top_k", str(atlas_mod.ATLAS_DEFAULT_TOP_K)))
        if not safe_name(name):
            return ({"error": "invalid model name"}, 400)
        return atlas_mod.concept_to_neuron(name, step, substring, top_k=top_k)

    @app.route("/atlas/neuron/<int:layer>/<int:neuron>")
    def atlas_neuron(layer, neuron):
        name = request.args.get("model", "")
        step = int(request.args.get("step", "0"))
        top_k = int(request.args.get("top_k", str(atlas_mod.ATLAS_DEFAULT_TOP_K)))
        if not safe_name(name):
            return ({"error": "invalid model name"}, 400)
        return atlas_mod.neuron_to_concept(name, step, layer, neuron, top_k=top_k)

    @app.route("/atlas/lifetime/<int:layer>/<int:neuron>")
    def atlas_lifetime(layer, neuron):
        name = request.args.get("model", "")
        if not safe_name(name):
            return ({"error": "invalid model name"}, 400)
        return atlas_mod.neuron_lifetime(name, layer, neuron)

    @app.route("/atlas/circuit")
    def atlas_circuit():
        layer = int(request.args.get("layer", "0"))
        top_k = int(request.args.get("top_k", str(atlas_mod.ATLAS_CIRCUIT_TOP_K)))
        brain = current_app.config.get("BRAIN")
        return atlas_mod.circuit_graph(brain, layer, top_k=top_k)

    @app.route("/atlas/concepts_inverted")
    def atlas_concepts_inverted():
        name = request.args.get("model", "")
        step = int(request.args.get("step", "0"))
        if not safe_name(name):
            return ({"error": "invalid model name"}, 400)
        try:
            return atlas_mod.concepts_inverted(name, step)
        except Exception as e:
            logmod.error("atlas", f"concepts_inverted failed: {type(e).__name__}: {e}")
            return ({"error": user_error(e)}, 500)
