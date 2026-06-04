# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - multimind dashboard backend. reads dev_documentation/multimind/*.json when
#   present, falls back to stubs otherwise. /multimind/sample stays synthetic
#   until a live MtM brain is wired.
# - region defaults: 6 named regions (broca, wernicke, hippocampus, prefrontal,
#   cerebellum, thalamus). PoC config, not per-model.
# veritate_mri/routes/multimind_routes.py
# ------------------------------------------------------------------------------------
# Imports:

import hashlib
import json
import os
import random
import sys
import time

from flask import request

from runtime import logs as logmod

from ._common import user_error

# ------------------------------------------------------------------------------------
# Constants

PROCESS_START_TS    = time.time()
MAX_SAMPLE_BYTES    = 4096
MIN_SAMPLE_BYTES    = 1
GATE_BURST_PROB     = 0.6     # P(a "hot" region is hot on this byte)
AFFECT_DRIFT_ALPHA  = 0.92    # low-pass coefficient for valence/arousal drift

_REPO_ROOT             = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MULTIMIND_RESULTS_DIR  = os.path.join(_REPO_ROOT, "dev_documentation", "multimind")
W1_RESULTS_PATH        = os.path.join(MULTIMIND_RESULTS_DIR, "W1_results.json")
W1B_RESULTS_PATH       = os.path.join(MULTIMIND_RESULTS_DIR, "W1b_results.json")
W2_RESULTS_PATH        = os.path.join(MULTIMIND_RESULTS_DIR, "W2_results.json")
W3_RESULTS_PATH        = os.path.join(MULTIMIND_RESULTS_DIR, "W3_results.json")
W4_RESULTS_PATH        = os.path.join(MULTIMIND_RESULTS_DIR, "W4_results.json")
W5_RESULTS_PATH        = os.path.join(MULTIMIND_RESULTS_DIR, "W5_results.json")
W6_RESULTS_PATH        = os.path.join(MULTIMIND_RESULTS_DIR, "W6_results.json")
W6B_RESULTS_PATH       = os.path.join(MULTIMIND_RESULTS_DIR, "W6b_results.json")
LIVE_MODEL_PATH        = os.path.join(MULTIMIND_RESULTS_DIR, "multimind_poc_model.pt")
AFFECT_PROBE_PATH      = os.path.join(MULTIMIND_RESULTS_DIR, "affect_probe.pt")
MTM_POC_DIR            = os.path.join(_REPO_ROOT, "trainers", "multimind_poc")
LIVE_MAX_BYTES         = 200
LIVE_DEVICE            = "cpu"

_MTM_MODEL    = None
_AFFECT_PROBE = None

EXPERIMENT_FILES = [
    ("W1",  W1_RESULTS_PATH,  "W1 baseline (no bias) vs with-bias learnable scalar"),
    ("W1b", W1B_RESULTS_PATH, "W1e oracle-forced g (gold-standard architecture controller)"),
    ("W2",  W2_RESULTS_PATH,  "W2 learned affect probe replacing oracle scalar"),
    ("W3",  W3_RESULTS_PATH,  "W3 region naming via specialty corpora"),
    ("W4",  W4_RESULTS_PATH,  "W4 refractory inhibition burst-iness vs ppl"),
    ("W5",  W5_RESULTS_PATH,  "W5 sleep cycle LoRA adaptation"),
    ("W6",  W6_RESULTS_PATH,  "W6 slot memory hippocampus PoC"),
    ("W6b", W6B_RESULTS_PATH, "W6b cross-sample fact recall"),
]

POC_REGIONS = [
    {"name": "broca",       "slug": "broca",       "specialty": "speech production / output side", "params": 3_000_000, "color": "#4f8be4"},
    {"name": "wernicke",    "slug": "wernicke",    "specialty": "comprehension / early layers",     "params": 3_000_000, "color": "#6dd5a6"},
    {"name": "hippocampus", "slug": "hippocampus", "specialty": "recent-context recall",            "params": 3_000_000, "color": "#e2a341"},
    {"name": "prefrontal",  "slug": "prefrontal",  "specialty": "planning / multi-step",            "params": 3_000_000, "color": "#c264d8"},
    {"name": "cerebellum",  "slug": "cerebellum",  "specialty": "repetition / motor patterns",      "params": 3_000_000, "color": "#e26b6b"},
    {"name": "thalamus",    "slug": "thalamus",    "specialty": "router / gate meta-loss",          "params": 1_000_000, "color": "#9aa0b2"},
]

ASCII_A             = ord("A")
ALPHABET_SPAN       = 26
HOT_WEIGHT_LO       = 0.4
HOT_WEIGHT_HI       = 0.7
COOL_WEIGHT_LO      = 0.0
COOL_WEIGHT_HI      = 0.1
REFRACTORY_THRESH   = 0.5
AFFECT_NOISE_SCALE  = 0.25
STUB_RESULT_TEXT    = "stub byte trace; replace with a live MtM model."

# ------------------------------------------------------------------------------------
# Functions

def _load_live_model():
    global _MTM_MODEL, _AFFECT_PROBE
    if _MTM_MODEL is not None and _AFFECT_PROBE is not None:
        return _MTM_MODEL, _AFFECT_PROBE
    if not (os.path.isfile(LIVE_MODEL_PATH) and os.path.isfile(AFFECT_PROBE_PATH)):
        return None, None
    try:
        import torch
        if MTM_POC_DIR not in sys.path:
            sys.path.insert(0, MTM_POC_DIR)
        blob = torch.load(LIVE_MODEL_PATH, map_location=LIVE_DEVICE, weights_only=True)
        cfg = blob.get("config") or {}
        # respect per_layer_g flag from checkpoint config before MtMModel imports
        if cfg.get("per_layer_g"):
            os.environ["MTM_PER_LAYER_G"] = "1"
        from moe_model import MtMModel
        from affect_probe import AffectProbe
        model = MtMModel(hidden=cfg.get("hidden", 256), layers=cfg.get("layers", 4),
                         ffn=cfg.get("ffn", 512), heads=cfg.get("heads", 8),
                         seq=cfg.get("seq", 512), bias_mode=cfg.get("bias_mode", True))
        model.load_state_dict(blob["state_dict"])
        model.to(LIVE_DEVICE).eval()
        pblob = torch.load(AFFECT_PROBE_PATH, map_location=LIVE_DEVICE, weights_only=True)
        probe = AffectProbe().to(LIVE_DEVICE)
        probe.load_state_dict(pblob["state_dict"])
        probe.eval()
        for p in probe.parameters(): p.requires_grad_(False)
        _MTM_MODEL, _AFFECT_PROBE = model, probe
        return model, probe
    except Exception as e:
        logmod.warn("multimind", f"live load failed: {type(e).__name__}: {e}")
        return None, None


def _live_trace(prompt, max_bytes):
    import torch
    model, probe = _MTM_MODEL, _AFFECT_PROBE
    seq_cap = model.seq - 1 - max_bytes
    prompt_bytes = list(prompt.encode("utf-8", errors="replace"))[:max(1, seq_cap)] or [ord(" ")]
    ids = torch.tensor([prompt_bytes], dtype=torch.long, device=LIVE_DEVICE)
    n_layers = model.layers
    n_experts = len(POC_REGIONS)
    bytes_out, events, prev_w = [], [], [0.0] * n_experts
    with torch.no_grad():
        for i in range(max_bytes):
            logits, _, _, gates = model(ids, sentiment=probe.valence(ids))
            nxt = int(logits[0, -1].argmax().item())
            layer_gates = gates[:, 0, -1, :].cpu().numpy()
            avg = layer_gates.mean(axis=0)
            regs = []
            for e in range(n_experts):
                w = float(avg[e])
                regs.append({"slug": POC_REGIONS[e]["slug"], "gate_weight": w,
                             "refractory": prev_w[e] > REFRACTORY_THRESH})
                prev_w[e] = w
            val = float(probe.valence(ids)[0].item())
            events.append({"byte_idx": i, "regions": regs,
                           "affect": {"valence": val, "arousal": 0.0}, "layers": int(n_layers)})
            bytes_out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=LIVE_DEVICE)], dim=1)
            if ids.shape[1] >= model.seq:
                ids = ids[:, -(model.seq - 1):]
    return bytes_out, events


def _load_json(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logmod.warn("multimind", f"failed to parse {os.path.basename(path)}: {type(e).__name__}: {e}")
        return None


def _routing_from(data):
    wb = ((data or {}).get("runs") or {}).get("with_bias") or {}
    routing = wb.get("routing") or {}
    cp, cn = routing.get("counts_pos"), routing.get("counts_neg")
    if not isinstance(cp, list) or not isinstance(cn, list) or not cp or not cn:
        return None
    return cp, cn, float(wb.get("g_norm") or 0.0)


def _synthetic_trace(prompt, max_bytes):
    seed = int.from_bytes(hashlib.sha256(prompt.encode("utf-8", errors="replace")).digest()[:8], "big", signed=False)
    rng = random.Random(seed)
    n_regions = len(POC_REGIONS)
    bytes_out = []
    events = []
    prev_weights = [0.0] * n_regions
    valence = 0.0
    arousal = 0.0

    for i in range(max_bytes):
        bytes_out.append(ASCII_A + (i % ALPHABET_SPAN))

        # cycle hot region by position; second hot slot fires probabilistically
        hot_primary = i % n_regions
        hot_secondary = -1
        if rng.random() < GATE_BURST_PROB:
            hot_secondary = (hot_primary + 1 + rng.randrange(n_regions - 1)) % n_regions

        region_evts = []
        for r_idx, reg in enumerate(POC_REGIONS):
            if r_idx == hot_primary or r_idx == hot_secondary:
                w = rng.uniform(HOT_WEIGHT_LO, HOT_WEIGHT_HI)
            else:
                w = rng.uniform(COOL_WEIGHT_LO, COOL_WEIGHT_HI)
            refractory = prev_weights[r_idx] > REFRACTORY_THRESH
            region_evts.append({
                "slug": reg["slug"],
                "gate_weight": float(w),
                "refractory": bool(refractory),
            })
            prev_weights[r_idx] = w

        # low-pass drift on white noise for affect
        v_noise = rng.uniform(-AFFECT_NOISE_SCALE, AFFECT_NOISE_SCALE)
        a_noise = rng.uniform(-AFFECT_NOISE_SCALE, AFFECT_NOISE_SCALE)
        valence = max(-1.0, min(1.0, AFFECT_DRIFT_ALPHA * valence + (1.0 - AFFECT_DRIFT_ALPHA) * v_noise * 4.0))
        arousal = max(0.0, min(1.0, AFFECT_DRIFT_ALPHA * arousal + (1.0 - AFFECT_DRIFT_ALPHA) * (0.5 + a_noise) * 2.0))

        events.append({
            "byte_idx": i,
            "regions": region_evts,
            "affect": {"valence": float(valence), "arousal": float(arousal)},
        })

    return bytes_out, events


def _experiment_events(data):
    parsed = _routing_from(data)
    if parsed is None:
        return []
    cp, _cn, _g = parsed
    n_experts = min(len(cp[0]), len(POC_REGIONS))
    events = []
    for L, layer in enumerate(cp):
        total = float(sum(layer[:n_experts])) or 1.0
        regs = []
        for e in range(n_experts):
            regs.append({
                "slug": POC_REGIONS[e]["slug"],
                "gate_weight": float(layer[e]) / total,
                "refractory": False,
            })
        events.append({"byte_idx": L, "regions": regs, "affect": {"valence": 0.0, "arousal": 0.0}})
    return events


def register(app):
    @app.route("/multimind/status")
    def multimind_status_route():
        has_w2 = os.path.isfile(W2_RESULTS_PATH)
        return {
            "model_loaded": bool(has_w2),
            "model_name": "multimind_poc_w2" if has_w2 else None,
            "model_step": None,
            "role": "awake",
            "uptime_secs": time.time() - PROCESS_START_TS,
            "last_sleep_at": None,
            "next_sleep_eta_secs": None,
            "last_sleep_duration_secs": None,
            "bytes_since_sleep": 0,
            "mode": "live" if has_w2 else "stub",
        }

    @app.route("/multimind/regions")
    def multimind_regions_route():
        parsed = _routing_from(_load_json(W2_RESULTS_PATH) or _load_json(W1_RESULTS_PATH) or _load_json(W1B_RESULTS_PATH))
        pct, drift, mode = [0.0] * len(POC_REGIONS), None, "stub"
        if parsed is not None:
            cp, cn, g_norm = parsed
            n_e = len(cp[0])
            totals = [sum(float(layer[e]) for layer in cp + cn) for e in range(n_e)]
            grand = sum(totals) or 1.0
            pct = [100.0 * t / grand for t in totals]
            drift = g_norm / (n_e ** 0.5) if n_e > 0 else 0.0
            mode = "live"
        regions = [{
            "name": r["name"], "slug": r["slug"], "specialty": r["specialty"], "params": r["params"],
            "last_drift_l2": drift, "specialty_ppl": None,
            "fired_pct": pct[i] if i < len(pct) else 0.0, "color": r["color"],
        } for i, r in enumerate(POC_REGIONS)]
        return {"regions": regions, "mode": mode}

    @app.route("/multimind/conversations")
    def multimind_conversations_route():
        convs = []
        for eid, path, preview in EXPERIMENT_FILES:
            if not os.path.isfile(path):
                continue
            data = _load_json(path)
            cfg = (data or {}).get("config") or {}
            seq = int(cfg.get("seq_len") or 512)
            batch = int(cfg.get("batch_size") or 32)
            steps = int(cfg.get("train_steps") or 0)
            convs.append({
                "id": eid,
                "ts": os.path.getmtime(path),
                "prompt_preview": preview,
                "byte_count": steps * batch * seq,
            })
        return {"conversations": convs, "mode": "live" if convs else "stub"}

    @app.route("/multimind/conversations/<conv_id>")
    def multimind_conversation_get_route(conv_id):
        path = {eid: p for eid, p, _ in EXPERIMENT_FILES}.get(conv_id)
        if not path or not os.path.isfile(path):
            return ({"ok": False, "error": "not found", "mode": "stub"}, 404)
        data = _load_json(path)
        events = _experiment_events(data) if data else []
        summary = (data or {}).get("summary") or {}
        if summary:
            parts = [f"{k}={v}" for k, v in summary.items()]
            text = f"{conv_id} summary: " + ", ".join(parts)
        else:
            text = f"{conv_id} results loaded ({len(events)} layers)"
        return {"events": events, "result_text": text, "mode": "live"}

    @app.route("/multimind/sample", methods=["POST"])
    def multimind_sample_route():
        body = request.get_json(silent=True) or {}
        prompt = body.get("prompt")
        max_bytes = body.get("max_bytes")
        if not isinstance(prompt, str):
            return ({"ok": False, "error": "prompt must be a string", "mode": "stub"}, 400)
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
            return ({"ok": False, "error": "max_bytes must be an int", "mode": "stub"}, 400)
        if max_bytes < MIN_SAMPLE_BYTES or max_bytes > MAX_SAMPLE_BYTES:
            return ({"ok": False, "error": f"max_bytes out of range [{MIN_SAMPLE_BYTES}, {MAX_SAMPLE_BYTES}]", "mode": "stub"}, 400)

        model, probe = _load_live_model()
        if model is not None and probe is not None:
            live_n = min(max_bytes, LIVE_MAX_BYTES)
            try:
                bytes_out, events = _live_trace(prompt, live_n)
                text = bytes(b & 0xFF for b in bytes_out).decode("utf-8", errors="replace")
                return {"bytes": bytes_out, "events": events, "result_text": text, "mode": "live"}
            except Exception as e:
                logmod.warn("multimind", f"live decode failed, falling back to stub: {type(e).__name__}: {e}")

        try:
            bytes_out, events = _synthetic_trace(prompt, max_bytes)
        except Exception as e:
            logmod.error("multimind_sample", f"{type(e).__name__}: {e}")
            return ({"ok": False, "error": user_error(e), "mode": "stub"}, 500)

        return {
            "bytes": bytes_out,
            "events": events,
            "result_text": STUB_RESULT_TEXT,
            "mode": "stub",
        }

    @app.route("/multimind/sleep/trigger", methods=["POST"])
    def multimind_sleep_trigger_route():
        return {"ok": True, "started": True, "mode": "stub"}
