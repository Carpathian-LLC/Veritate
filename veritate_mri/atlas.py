# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - interpretability layer (v8). aggregations over existing rule-4 dump artifacts:
#   concept->neuron atlas, neuron->concept inversion, neuron lifetime across training,
#   neuron->neuron static circuit graph, and concepts->neurons inversion.
# - reads only. no new artifacts. consumed by /atlas/* routes in app.py.
# veritate_mri/atlas.py
# ------------------------------------------------------------------------------------
# Imports:

import os

from readers import hooks, paths

# ------------------------------------------------------------------------------------
# Constants

ATLAS_DEFAULT_TOP_K   = 24
ATLAS_CIRCUIT_TOP_K   = 16
ATLAS_LIFETIME_DTYPE  = "fp32"

# ------------------------------------------------------------------------------------
# Functions

def _frames(name, step):
    data = hooks.load_artifact(name, step, "generation")
    if data is None:
        return []
    return data.get("frames") or []


def _accumulate_dla(frames, byte_filter, dla_field):
    """Aggregate |contrib| from every entry of dla_field across frames where
    the sampled byte is in byte_filter. Returns dict {(layer, neuron): score}.
    byte_filter == None aggregates across every frame."""
    acc = {}
    n_matched = 0
    for fr in frames:
        b = fr.get("byte")
        if byte_filter is not None and b not in byte_filter:
            continue
        n_matched += 1
        for e in fr.get(dla_field) or []:
            key = (int(e.get("layer", 0)), int(e.get("neuron", 0)))
            acc[key] = acc.get(key, 0.0) + abs(float(e.get("contrib", 0.0)))
    return acc, n_matched


def concept_to_neuron(name, step, substring, top_k=ATLAS_DEFAULT_TOP_K):
    """Top (layer, neuron) pairs by aggregated |dla_picked.contrib| across frames
    whose sampled byte is in substring's byte set. Use to answer "which neurons
    keep voting for this byte set" - the goblin question for the GPT-5 case."""
    frames = _frames(name, step)
    if not frames:
        return {"error": "no generation dump for this step", "step": int(step)}
    if not substring:
        return {"error": "empty substring"}
    byte_filter = set(substring.encode("utf-8", "replace"))
    acc, n_matched = _accumulate_dla(frames, byte_filter, "dla_picked")
    ranked = sorted(acc.items(), key=lambda kv: -kv[1])[:top_k]
    return {
        "model": name, "step": int(step),
        "substring": substring,
        "byte_filter": sorted(byte_filter),
        "n_frames": len(frames),
        "n_matched": n_matched,
        "neurons": [{"layer": L, "neuron": n, "score": round(s, 4)}
                    for (L, n), s in ranked],
    }


def neuron_to_concept(name, step, layer, neuron, top_k=ATLAS_DEFAULT_TOP_K):
    """Inverse query: given (layer, neuron), which output bytes does it most
    often dominate (via dla_picked) and which concepts list it in their
    top_neurons?"""
    frames = _frames(name, step)
    byte_score = {}
    for fr in frames:
        b = fr.get("byte")
        for e in fr.get("dla_picked") or []:
            if int(e.get("layer", -1)) != layer:
                continue
            if int(e.get("neuron", -1)) != neuron:
                continue
            byte_score[b] = byte_score.get(b, 0.0) + abs(float(e.get("contrib", 0.0)))
    bytes_ranked = sorted(byte_score.items(), key=lambda kv: -kv[1])[:top_k]

    concepts_data = hooks.load_artifact(name, step, "concepts")
    matched_concepts = []
    if concepts_data is not None:
        for c in concepts_data.get("concepts") or []:
            for tn in c.get("top_neurons") or []:
                tn_layer = int(tn.get("layer", -1))
                tn_neuron = int(tn.get("id", tn.get("neuron", -1)))
                if tn_layer == layer and tn_neuron == neuron:
                    matched_concepts.append({
                        "concept":  c.get("concept", ""),
                        "surprise": c.get("surprise"),
                        "magnitude": tn.get("v", tn.get("magnitude", 0.0)),
                    })
                    break
    return {
        "model": name, "step": int(step),
        "layer": int(layer), "neuron": int(neuron),
        "bytes": [{"b": int(b), "score": round(s, 4)} for b, s in bytes_ranked],
        "concepts": matched_concepts,
    }


def neuron_lifetime(name, layer, neuron):
    """Per-step trajectory of (layer, neuron) across every probe_step_<N>.json.
    Each step records whether the neuron made the top-K and at what magnitude."""
    series = []
    for step in hooks.list_steps(name):
        probe = hooks.load_artifact(name, step, "probe")
        if probe is None:
            continue
        layers = probe.get("layers") or []
        rank, mag = -1, 0.0
        for entry in layers:
            if int(entry.get("layer", -1)) != layer:
                continue
            for r, n in enumerate(entry.get("neurons") or []):
                if int(n.get("id", -1)) == neuron:
                    rank = r
                    mag  = float(n.get("v", 0.0))
                    break
            break
        series.append({
            "step": int(step),
            "rank": rank,
            "magnitude": round(mag, 4),
            "in_top_k": rank >= 0,
        })
    return {
        "model": name,
        "layer": int(layer), "neuron": int(neuron),
        "series": series,
    }


def circuit_graph(brain, layer, top_k=ATLAS_CIRCUIT_TOP_K):
    """Static neuron->neuron transfer between layer L and L+1, computed as
    W_down[L].T @ W_up[L+1]. Requires the pytorch backend to be loaded so the
    weights are in memory; computed lazily, no caching across calls (the call
    site can cache by (model, layer) if it wants)."""
    if brain is None:
        return {"error": "pytorch backend required for circuit graph"}
    import torch
    m = brain.model
    if layer < 0 or layer + 1 >= m.layers:
        return {"error": f"layer {layer} out of range [0, {m.layers - 2}]"}
    with torch.no_grad():
        w_down = m.blocks[layer].ff.down.weight        # (hidden, ffn_L)
        w_up   = m.blocks[layer + 1].ff.up.weight      # (ffn_Lp1, hidden)
        transfer = (w_up @ w_down).contiguous()        # (ffn_Lp1, ffn_L)
        # for each src neuron in layer L, list top-K dst neurons in layer L+1.
        edges = []
        ffn_L = transfer.size(1)
        ffn_Lp1 = transfer.size(0)
        for src in range(ffn_L):
            col = transfer[:, src]
            _, idx = torch.topk(col.abs(), k=min(top_k, ffn_Lp1))
            for dst in idx.tolist():
                w = float(transfer[dst, src].item())
                edges.append({
                    "src_layer":   int(layer),
                    "src_neuron":  int(src),
                    "dst_layer":   int(layer + 1),
                    "dst_neuron":  int(dst),
                    "w":           round(w, 5),
                })
    return {
        "model_layers": int(m.layers),
        "src_layer": int(layer),
        "dst_layer": int(layer + 1),
        "src_count": int(ffn_L),
        "dst_count": int(ffn_Lp1),
        "edges":     edges,
    }


def concepts_inverted(name, step):
    """Invert concepts.json::top_neurons into a neuron-keyed map. Each entry
    lists the concepts that name this neuron in their top_neurons block."""
    data = hooks.load_artifact(name, step, "concepts")
    if data is None:
        return {"error": "no concepts dump for this step", "step": int(step)}
    raw = data.get("concepts") or {}
    if isinstance(raw, dict):
        items = [{"concept": k, **(v or {})} for k, v in raw.items()]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    inverted = {}
    for c in items:
        if not isinstance(c, dict):
            continue
        cname = c.get("concept", "")
        surprise = c.get("surprise_bits", c.get("surprise"))
        for tn in c.get("top_neurons") or []:
            tn_layer = int(tn.get("layer", -1))
            tn_neuron = int(tn.get("id", tn.get("neuron", -1)))
            if tn_layer < 0 or tn_neuron < 0:
                continue
            key = f"{tn_layer}:{tn_neuron}"
            inverted.setdefault(key, []).append({
                "concept":   cname,
                "magnitude": tn.get("v", tn.get("magnitude", 0.0)),
                "surprise":  surprise,
            })
    return {
        "model": name, "step": int(step),
        "neurons": [
            {"layer": int(k.split(":")[0]), "neuron": int(k.split(":")[1]), "concepts": v}
            for k, v in inverted.items()
        ],
    }
