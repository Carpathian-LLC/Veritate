#!/usr/bin/env python3
# Notes:
# - E8 DeepSeekMoE FFN verify: unit correctness (combine reconstruction, static
#   shapes, finite aux/share), routing-bias load equalization (dead experts
#   revive), real-width MPS fwd+bwd stability + throughput tax vs hybrid, and the
#   full dump/hook battery at the 10M run shape. Falsifier: any NaN at real width,
#   any DUMP FAILED, or dead experts never climb off zero.
# - Wall-clock ~2-4 min (MPS).
# SMOKE_RESULTS/e8_moe_smoke.py

import json
import os
import sys
import time
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "veritate_mri"))

from veritate_core.model_moe import MoEFFN, MOE_N_ROUTED, MOE_TOP_K
from veritate_core.model_patched import VeritatePatched
from veritate_core import qat as qat_mod

HIDDEN, LAYERS, FFN, HEADS, SEQ = 320, 8, 1280, 8, 512
BS = 32
FINEWEB = os.path.join(ROOT, "trainers", "corpus", "fineweb_edu_train.bin")
PROMPTS = ["The quick brown fox jumps", "Once upon a time there", "def add(a, b):"]


def test_unit():
    """MoEFFN: combine reconstruction exact, shapes static across padded inputs, aux/share finite."""
    torch.manual_seed(0)
    r = {}
    h, f = 32, 64
    moe = MoEFFN(h, f, capacity_factor=8.0).eval()
    b, t = 2, 8
    x = torch.randn(b, t, h)
    with torch.no_grad():
        out = moe(x)
        logits = F.linear(x, moe.gate.weight)
        probs = F.softmax(logits, dim=-1)
        score = probs + moe.route_bias
        ref = moe.down(moe._act_fn(moe.up(x))).clone()
        for bi in range(b):
            for ti in range(t):
                top2 = torch.topk(score[bi, ti], MOE_TOP_K).indices
                w = probs[bi, ti][top2]
                w = w / w.sum()
                for k in range(MOE_TOP_K):
                    e = int(top2[k])
                    ref[bi, ti] += w[k] * moe.experts[e](x[bi, ti].view(1, h))[0]
    r["combine_max_abs_diff"] = float((out - ref).abs().max())
    r["combine_ok"] = r["combine_max_abs_diff"] < 1e-4

    x1 = torch.randn(2, SEQ, h)
    x2 = torch.randn(2, SEQ, h)
    with torch.no_grad():
        o1, o2 = moe(x1), moe(x2)
    r["capacity_at_seq"] = moe._capacity(SEQ)
    r["static_shape_ok"] = (o1.shape == o2.shape == x1.shape)

    r["aux_finite"] = bool(torch.isfinite(moe._last_aux).item())
    share = moe._last_share
    r["share_finite"] = bool(torch.isfinite(share).all().item())
    r["share_sum"] = float(share.sum())
    r["n_experts"] = moe.num_experts
    r["pass"] = r["combine_ok"] and r["static_shape_ok"] and r["aux_finite"] and r["share_finite"]
    return r


def test_bias_balance():
    """Aux-loss-free routing bias revives dead experts: time-averaged min share climbs off zero.

    Load balance is a property of the selection COUNT averaged over steps (diverse
    per-token routing), not of a single step. Half the experts start effectively
    dead (near-zero gate rows); the bias must pull them back into the top-2."""
    torch.manual_seed(1)
    r = {}
    h, f = 64, 128
    n_steps = 800
    moe = MoEFFN(h, f).train()
    E = moe.num_experts
    with torch.no_grad():
        moe.gate.weight.normal_(0.0, 0.5)
        moe.gate.weight[E // 2:] = -3.0           # experts E/2..E-1 strongly disfavored
    b, t = 8, 128
    win = 200
    early_share = None
    late_acc = torch.zeros(E)
    traj = []
    for step in range(n_steps):
        x = torch.randn(b, t, h) + 1.0            # positive-mean input -> dead rows stay negative
        with torch.no_grad():
            moe(x)
        share = moe._last_share
        if step == 0:
            early_share = share.clone()
        if step >= n_steps - win:
            late_acc += share
        if step % 100 == 0:
            traj.append({"step": step, "min_share": round(float(share.min()), 5)})
    late_share = late_acc / win
    r["dead_experts_initial"] = int((early_share < 1e-9).sum())
    r["early_min_share"] = round(float(early_share.min()), 5)
    r["late_avg_min_share"] = round(float(late_share.min()), 5)
    r["late_avg_max_share"] = round(float(late_share.max()), 5)
    r["late_share"] = [round(float(s), 4) for s in late_share]
    r["bias_spread"] = round(float(moe.route_bias.max() - moe.route_bias.min()), 4)
    r["trajectory"] = traj
    r["pass"] = (r["dead_experts_initial"] >= E // 2) and (r["late_avg_min_share"] > 0.02)
    return r


def _build(device, global_ffn, qat=False):
    torch.manual_seed(0)
    m = VeritatePatched(vocab=256, hidden=HIDDEN, layers=LAYERS, ffn=FFN, heads=HEADS,
                        seq=SEQ, global_mixer="recurrent", global_ffn=global_ffn)
    if qat:
        qat_mod.set_qat(m, True)
    return m.to(device)


def _train_steps(model, device, n_steps, amp):
    losses, tok = [], 0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n_steps):
        toks = torch.randint(0, 256, (BS, SEQ), device=device)
        tgts = torch.randint(0, 256, (BS, SEQ), device=device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=amp, enabled=(amp is not None)):
            _, loss = model(toks, targets=tgts)
        aux = model.moe_aux_sum()
        if aux is not None:
            loss = loss + aux
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().item()))
        tok += BS * SEQ
    if device == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    return losses, tok / dt


def test_real_width():
    """Real 10M shape hybrid-MoE: fwd+bwd on MPS stays finite; measure throughput tax vs hybrid."""
    r = {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    amp = torch.bfloat16
    r["device"] = device

    moe = _build(device, "moe", qat=True)
    r["n_params_moe"] = sum(p.numel() for p in moe.parameters())
    losses, tps_moe = _train_steps(moe, device, 8, amp)
    r["moe_losses"] = [round(x, 4) for x in losses]
    r["moe_all_finite"] = all(np.isfinite(losses))
    r["moe_tok_per_s"] = round(tps_moe, 1)
    del moe
    if device == "mps":
        torch.mps.empty_cache()

    hyb = _build(device, "dense", qat=True)
    r["n_params_hybrid"] = sum(p.numel() for p in hyb.parameters())
    _, tps_hyb = _train_steps(hyb, device, 8, amp)
    r["hybrid_tok_per_s"] = round(tps_hyb, 1)
    r["throughput_ratio_moe_over_hybrid"] = round(tps_moe / tps_hyb, 3)
    r["param_ratio_moe_over_hybrid"] = round(r["n_params_moe"] / r["n_params_hybrid"], 3)
    del hyb
    if device == "mps":
        torch.mps.empty_cache()
    r["pass"] = r["moe_all_finite"]
    return r


def test_dump_suite():
    """Full dump/hook battery at real 10M shape, short real prompts: every dump writes, no crash."""
    from training.checkpoint_probe import (
        dump_probe, dump_classroom, dump_grades, dump_reading_comprehension,
        dump_math, dump_grammar, dump_reasoning, dump_concepts, dump_surprise,
        dump_quant_kl, dump_writing_health, dump_generation,
    )
    r = {}
    dump_device = "mps" if torch.backends.mps.is_available() else "cpu"
    r["device"] = dump_device
    model = _build(dump_device, "moe").eval()
    corpus = FINEWEB if os.path.isfile(FINEWEB) else None
    r["corpus"] = corpus
    prompt = PROMPTS[0]
    step = 100
    results = {}
    with tempfile.TemporaryDirectory() as d:
        battery = [
            ("probe", lambda: dump_probe(model, prompt, d, step)),
            ("classroom", lambda: dump_classroom(model, d, step)),
            ("grades", lambda: dump_grades(model, d, step)),
            ("reading_comprehension", lambda: dump_reading_comprehension(model, d, step)),
            ("math", lambda: dump_math(model, d, step)),
            ("grammar", lambda: dump_grammar(model, d, step)),
            ("reasoning", lambda: dump_reasoning(model, d, step)),
            ("concepts", lambda: dump_concepts(model, d, step)),
            ("surprise", lambda: dump_surprise(model, prompt, d, step)),
            ("quant_kl", lambda: dump_quant_kl(model, prompt, d, step)),
            ("writing_health", lambda: dump_writing_health(model, d, step, corpus_path=corpus)),
            ("generation", lambda: dump_generation(model, prompt, d, step, corpus_path=corpus)),
        ]
        for label, fn in battery:
            try:
                fn()
                results[label] = "ok"
            except Exception as e:
                results[label] = f"{type(e).__name__}: {e}"
        artifacts = sorted(os.listdir(d))
        r["n_artifacts"] = len([a for a in artifacts if not a.startswith("_")])
        r["artifacts"] = [a for a in artifacts if not a.startswith("_")]
    r["dumps"] = results
    r["failed"] = [k for k, v in results.items() if v != "ok"]
    r["pass"] = len(r["failed"]) == 0
    return r


def main():
    out = {"errors": [], "shape": {"hidden": HIDDEN, "layers": LAYERS, "ffn": FFN,
                                   "heads": HEADS, "seq": SEQ, "bs": BS},
           "moe": {"n_routed": MOE_N_ROUTED, "top_k": MOE_TOP_K}}
    for name, fn in [("unit", test_unit), ("bias_balance", test_bias_balance),
                     ("real_width", test_real_width), ("dump_suite", test_dump_suite)]:
        try:
            res = fn()
            out[name] = res
            print(f"[{name}] pass={res.get('pass')}", flush=True)
            if not res.get("pass"):
                out["errors"].append(f"{name} failed")
        except Exception as e:
            import traceback
            out[name] = {"pass": False, "exception": traceback.format_exc()}
            out["errors"].append(f"{name} raised {type(e).__name__}: {e}")
            print(f"[{name}] EXCEPTION {e}", flush=True)

    path = os.path.join(ROOT, "SMOKE_RESULTS", "e8_moe_stats.json")
    with open(path, "w") as fp:
        json.dump(out, fp, indent=2)
    print("wrote " + path, flush=True)
    sys.exit(0 if not out["errors"] else 1)


if __name__ == "__main__":
    main()
