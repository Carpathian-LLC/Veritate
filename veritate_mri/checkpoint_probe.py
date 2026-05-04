# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - training-time probe dump. captures top-k ffn neurons per layer, per-layer
#   logit lens, and residual-post norms for a fixed prompt. writes one json
#   and one npz per checkpoint. consumed by mri learning tab.
# - canonical implementation per docs/GLASS_MODEL_ROE.md rule 4.
# - extended dumps: classroom (size + alive neurons), grades (suite A reading
#   ppl per grade band), concepts (50-concept surprise probe). all three are
#   checkpoint-time only; zero impact on the training step.
# ------------------------------------------------------------------------------------

import argparse
import heapq
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from readers import paths


PROBE_PROMPT = "Once upon a time, there was a little girl who"
PROBE_TOP_K  = 8

# per-token generation probe constants. match mri/probes/timeline_probe.py
# legacy frame shape so the Learning tab consumes the output without changes.
GEN_MAX_NEW          = 80
GEN_TEMPERATURE      = 0.7
GEN_TOP_K            = 40
GEN_FFN_BUCKET_TARGET= 256
GEN_ATTN_TOP_POS     = 6
GEN_NEXT_CANDIDATES  = 12
GEN_NEURON_TOP_K     = 8
GEN_INFO_FLOW_TOP    = 8
GEN_LENS_TOP_P       = 3
GEN_DLA_TOPK         = 12
ACTIVATION_INT8_SCALE= 32.0
INT8_SAT_THRESHOLD   = 127.0 / ACTIVATION_INT8_SCALE

GRADE_LEVELS    = ["prek", "k", "elem", "middle", "hs", "college", "phd"]
GRADE_PPL_PASS  = 3.0
GRADE_BYTES     = 4096

GRADE_SOURCES = {
    "prek":    [(39784, "real_mother_goose"),         (24108, "the_three_bears")],
    "k":       [(19994, "more_english_fairy_tales"),  (7439,  "english_fairy_tales")],
    "elem":    [(146,   "little_princess"),           (479,   "little_lord_fauntleroy")],
    "middle":  [(76,    "huckleberry_finn"),          (113,   "pinocchio")],
    "hs":      [(1260,  "jane_eyre"),                 (768,   "wuthering_heights")],
    "college": [(205,   "walden"),                    (1404,  "federalist_papers")],
    "phd":     [],
}

CONCEPT_TOP_K_PER_LAYER = 3

# memory fingerprint probe. runs on every generation dump. matches the offline
# mri/probes/timeline_probe.py shape so the Learning tab renders identically.
MEMORY_TOP_N             = 5
MEMORY_PROBE_STORIES     = 500
MEMORY_PROBE_TOPK        = 6
MEMORY_PROBE_MAX_STORY   = 256
MEMORY_PROBE_SEED        = 7
REPO_ROOT                = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CORPUS_CANDIDATES        = [
    os.path.join(REPO_ROOT, "data", "corpus", "tinystories_train.bin"),
    os.path.join(REPO_ROOT, "data", "tinystories_train.bin"),
]
DEFAULT_CORPUS_PATH      = CORPUS_CANDIDATES[0]

_STORY_CACHE = {}


def _resolve_corpus(corpus_path):
    """Return the first existing corpus path. Prefers caller-supplied path,
    then walks the canonical candidate list. Memory probe is non-negotiable, so
    raise if nothing is found rather than emit a frame file with empty memory."""
    if corpus_path and os.path.isfile(corpus_path):
        return corpus_path
    for p in CORPUS_CANDIDATES:
        if os.path.isfile(p):
            return p
    tried = [corpus_path] if corpus_path else []
    tried.extend(CORPUS_CANDIDATES)
    raise FileNotFoundError(
        f"memory probe corpus not found. tried: {tried}. "
        f"download/restore tinystories_train.bin or pass corpus_path explicitly."
    )

# fixed concept probe set. each: (name, preamble, target trigram). surprise is
# negative log-prob of the target bytes given the preamble, in bits per byte.
CONCEPTS = [
    ("cat",       "the small ",                "cat"),
    ("dog",       "she patted the friendly ",  "dog"),
    ("bird",      "high in the tree sat a ",   "bir"),
    ("fish",      "they swam like a ",         "fis"),
    ("tree",      "the old oak ",              "tre"),
    ("house",     "they walked into the ",     "hou"),
    ("car",       "she drove the red ",        "car"),
    ("ball",      "the boy threw the ",        "bal"),
    ("water",     "she drank a glass of ",     "wat"),
    ("food",      "he was hungry for ",        "foo"),
    ("happy",     "the smiling girl was ",     "hap"),
    ("sad",       "with tears she felt ",      "sad"),
    ("angry",     "he stomped his feet, very ","ang"),
    ("scared",    "the dark made her ",        "sca"),
    ("kind",      "she was gentle and ",       "kin"),
    ("love",      "the family shared their ",  "lov"),
    ("friend",    "they were each others ",    "fri"),
    ("mother",    "she ran to her ",           "mot"),
    ("father",    "he hugged his ",            "fat"),
    ("baby",      "the tiny ",                 "bab"),
    ("red",       "the apple was bright ",     "red"),
    ("blue",      "the sky was clear and ",    "blu"),
    ("green",     "the grass was fresh and ",  "gre"),
    ("yellow",    "the sun was warm and ",     "yel"),
    ("big",       "the elephant was very ",    "big"),
    ("small",     "the mouse was very ",       "sma"),
    ("hot",       "the fire was very ",        "hot"),
    ("cold",      "the snow was very ",        "col"),
    ("fast",      "the rabbit ran very ",      "fas"),
    ("slow",      "the turtle moved very ",    "slo"),
    ("run",       "she began to ",             "run"),
    ("jump",      "he learned to ",            "jum"),
    ("eat",       "they sat down to ",         "eat"),
    ("sleep",     "at night they ",            "sle"),
    ("walk",      "they took a short ",        "wal"),
    ("read",      "she opened the book to ",   "rea"),
    ("write",     "he picked up a pen to ",    "wri"),
    ("play",      "the children went to ",     "pla"),
    ("sing",      "the birds began to ",       "sin"),
    ("laugh",     "the joke made them ",       "lau"),
    ("number",    "she counted the ",          "num"),
    ("one",       "she had only ",             "one"),
    ("two",       "the pair held ",            "two"),
    ("three",     "the family had ",           "thr"),
    ("plus",      "what is two ",              "plu"),
    ("equals",    "five plus three ",          "equ"),
    ("question",  "tell me, what is the ",     "que"),
    ("answer",    "she gave the right ",       "ans"),
    ("story",     "let me tell you a ",        "sto"),
    ("end",       "and that is the ",          "end"),
    # stage d markers: q/a + dialogue patterns. preambles mirror prep_curriculum_d.py
    # output: "Q: ...\nA: ...\n\n" and quoted-speech exchange pairs.
    ("question_marker", "Q: ",                          "Wha"),
    ("answer_marker",   "Q: What is it?\nA: ",          "Th"),
    ("dialogue_open",   "\"",                           "Hello"),
    ("dialogue_close",  "\" said the ",                 "girl"),
    ("yes_no",          "Q: Did you do it?\nA: ",       "Yes"),
]


def _encode_bytes(prompt, seq):
    raw = prompt.encode("utf-8", errors="replace")[:seq]
    return torch.tensor(list(raw), dtype=torch.long).unsqueeze(0)


def _capture(model):
    cap_ffn  = [None] * model.layers
    cap_post = [None] * model.layers
    handles  = []
    for L, blk in enumerate(model.blocks):
        def _ffn(_m, _i, o, L=L): cap_ffn[L] = o.detach()
        handles.append(blk.ff.up.register_forward_hook(_ffn))
        def _blk(_m, _i, o, L=L): cap_post[L] = o.detach()
        handles.append(blk.register_forward_hook(_blk))
    return cap_ffn, cap_post, handles


@torch.no_grad()
def dump_probe(model, prompt: str, out_dir: str, step: int):
    """Run model in eval mode on prompt; write probe_step_<N>.json and lens_step_<N>.npz."""
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    cap_ffn, cap_post, handles = _capture(model)
    try:
        x = _encode_bytes(prompt, model.seq).to(device)
        logits, _ = model(x)

        embed_w = model.tok_emb.weight

        layers = model.layers
        vocab  = model.vocab
        lens   = np.zeros((layers, vocab), dtype=np.int32)
        rnorm  = np.zeros(layers, dtype=np.float32)
        top    = []

        for L in range(layers):
            raw = cap_ffn[L][0, -1]
            act = F.gelu(raw)
            mag = act.abs().float()
            k = min(PROBE_TOP_K, mag.numel())
            vals, idx = torch.topk(mag, k)
            top.append([
                {"id": int(idx[i].item()), "v": round(float(act[idx[i]].item()), 4)}
                for i in range(k)
            ])

            r = cap_post[L][0, -1].float()
            rnorm[L] = float(r.norm().item())
            ll = (r @ embed_w.t().float())
            scaled = (ll * 1000.0).round().clamp(-2_000_000_000, 2_000_000_000)
            lens[L] = scaled.cpu().numpy().astype(np.int32)
    finally:
        for h in handles: h.remove()
        if was_training: model.train()

    probe = {
        "step":      int(step),
        "precision": _precision_tag(model),
        "prompt":    prompt,
        "top_k":     PROBE_TOP_K,
        "layers":    [{"layer": L, "neurons": top[L]} for L in range(layers)],
    }
    json_path = os.path.join(out_dir, f"probe_step_{step}.json")
    npz_path  = os.path.join(out_dir, f"lens_step_{step}.npz")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(probe, f, ensure_ascii=False)
    np.savez_compressed(npz_path, lens_logits=lens, residual_norms=rnorm)
    return json_path, npz_path


def _capture_full(model):
    cap_ffn       = [None] * model.layers
    cap_block_in  = [None] * model.layers
    cap_block_out = [None] * model.layers
    handles = []
    for L, blk in enumerate(model.blocks):
        def _ffn(_m, _i, o, L=L): cap_ffn[L] = o.detach()
        handles.append(blk.ff.up.register_forward_hook(_ffn))
        def _pre(_m, inp, L=L): cap_block_in[L] = inp[0].detach()
        handles.append(blk.register_forward_pre_hook(_pre))
        def _blk(_m, _i, o, L=L): cap_block_out[L] = o.detach()
        handles.append(blk.register_forward_hook(_blk))
    return cap_ffn, cap_block_in, cap_block_out, handles


def _load_confidence_weights(out_dir: str):
    """Load calibrated confidence weights from the model dir if present.
    Returns (w_M, w_E, w_L, w_S, b, loaded). Fallback matches engine main.c."""
    path = os.path.join(out_dir, "confidence_weights.json")
    if not os.path.isfile(path):
        return 0.5, 0.5, 0.5, 0.5, -1.0, False
    try:
        with open(path, "r", encoding="utf-8") as f:
            cw = json.load(f)
        return (float(cw["w_M"]), float(cw["w_E"]),
                float(cw["w_L"]), float(cw["w_S"]),
                float(cw["b"]), True)
    except (OSError, KeyError, ValueError):
        return 0.5, 0.5, 0.5, 0.5, -1.0, False


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _load_probe_stories(corpus_path: str, seed: int, n_stories: int, max_story_bytes: int):
    key = (os.path.abspath(corpus_path), seed, n_stories, max_story_bytes)
    cached = _STORY_CACHE.get(key)
    if cached is not None:
        return cached
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(f"memory probe corpus not found: {corpus_path}")
    arr = np.memmap(corpus_path, dtype=np.uint8, mode="r")
    N = len(arr)
    if N < 64:
        raise ValueError(f"memory probe corpus too small: {corpus_path} ({N} bytes)")
    parts = []
    if N < 64 * 1024 * 1024:
        with open(corpus_path, "rb") as f:
            corpus = f.read()
        split = corpus.split(b"\x00")
        parts = [p for p in split if 32 <= len(p) <= max_story_bytes]
    rng = np.random.default_rng(seed)
    if len(parts) >= n_stories:
        chosen = rng.choice(len(parts), size=n_stories, replace=False)
        stories = [parts[int(i)] for i in chosen]
    else:
        win = min(max_story_bytes, max(64, N // (n_stories * 4) or 64))
        if N <= win + 1:
            stories = [bytes(arr[:N])]
        else:
            starts = rng.integers(0, N - win - 1, size=n_stories, dtype=np.int64)
            stories = [bytes(arr[int(s):int(s) + win]) for s in starts]
    _STORY_CACHE[key] = stories
    return stories


@torch.no_grad()
def _build_memory_from_corpus(model, cap_ffn, stories, top_k: int):
    seq    = model.seq
    layers = model.layers
    ffn    = model.ffn
    device = next(model.parameters()).device
    counter = 0
    heaps = [[[] for _ in range(ffn)] for _ in range(layers)]
    for story in stories:
        ids = torch.tensor([b for b in story[:seq]], dtype=torch.long, device=device).unsqueeze(0)
        if ids.size(1) < seq:
            pad = torch.zeros(seq - ids.size(1), dtype=torch.long, device=device).unsqueeze(0)
            ids = torch.cat([ids, pad], dim=1)
        _ = model(ids)
        real_len = min(len(story), seq)
        text = story[:real_len].decode("utf-8", errors="replace")
        for L in range(layers):
            raw = cap_ffn[L]
            act = F.gelu(raw)[0, :real_len].abs()
            per_max, per_argmax = act.max(dim=0)
            arr_v = per_max.cpu().numpy()
            arr_p = per_argmax.cpu().numpy()
            hl = heaps[L]
            for n in range(ffn):
                score = float(arr_v[n])
                h = hl[n]
                counter += 1
                entry = (score, counter, text, int(arr_p[n]))
                if len(h) < top_k:
                    heapq.heappush(h, entry)
                elif score > h[0][0]:
                    heapq.heapreplace(h, entry)
    out = {}
    for L in range(layers):
        out[str(L)] = {}
        for n in range(ffn):
            ranked = sorted(heaps[L][n], key=lambda x: -x[0])
            if ranked and ranked[0][0] > 0:
                out[str(L)][str(n)] = [
                    {"text": t, "score": round(s, 3), "peak_pos": int(p)}
                    for s, _c, t, p in ranked
                ]
    return out


def _memory_lookup(ffn_top, memory, top_n: int = MEMORY_TOP_N):
    if not memory:
        return []
    scores = {}
    for L, neurons in enumerate(ffn_top):
        layer_mem = memory.get(str(L))
        if not layer_mem:
            continue
        for n in neurons:
            entries = layer_mem.get(str(n["id"]))
            if not entries:
                continue
            weight = n["v"]
            for e in entries[:3]:
                text = e["text"]
                scores[text] = scores.get(text, 0.0) + weight * e["score"]
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return [{"text": t[:120], "score": round(s, 2)} for t, s in ranked]


@torch.no_grad()
def dump_generation(model, prompt: str, out_dir: str, step: int,
                    max_new: int = GEN_MAX_NEW,
                    temperature: float = GEN_TEMPERATURE,
                    top_k: int = GEN_TOP_K,
                    corpus_path: str = None):
    """Run model auto-regressively for max_new tokens; per token emit a frame
    matching the live TFRM v7 set produced by mri/server/app.py::_build_c_mri_frame.
    Write models/<name>/step_<N>.json in {meta, frames} format. Field-symmetry
    mandate: every key here MUST mirror the live chat path, so the Learning tab and
    chat tab use a single render path. eval-mode, no grad, batch 1.

    Memory fingerprint probe runs first: builds top-k activating stories per
    (layer, neuron) from a sampled slice of corpus_path, then each per-token
    frame's memory[] is filled by _memory_lookup. Defaults to
    plugins/corpus/tinystories_train.bin."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    seq    = model.seq
    layers = model.layers
    heads  = model.heads
    hidden = model.hidden
    ffn    = model.ffn
    vocab  = model.vocab
    head_dim = hidden // heads

    # ffn downsample so V_FFN/ds is close to GEN_FFN_BUCKET_TARGET and divides.
    ds = max(1, ffn // GEN_FFN_BUCKET_TARGET)
    while ds > 1 and ffn % ds != 0:
        ds -= 1

    prompt_bytes = (prompt or " ").encode("utf-8", errors="replace")
    ids = torch.tensor([b for b in prompt_bytes], dtype=torch.long, device=device).unsqueeze(0)
    if ids.size(1) >= seq:
        ids = ids[:, -(seq - 1):]

    embed_w = model.tok_emb.weight.detach().float()
    W_E_T   = embed_w.t().contiguous()

    # byte-direction columns per layer: BD[L][:, byte] = ff.down.weight.T @ tok_emb[byte].
    # ff.down.weight has shape (hidden, ffn); we want (ffn, hidden) @ (hidden,) per byte.
    # precompute the (layers, ffn, vocab) tensor once; for 80m this is 12*3072*256*4 = 36 MB.
    ffn_down_T = torch.stack(
        [blk.ff.down.weight.detach().float().t().contiguous() for blk in model.blocks],
        dim=0,
    )  # (layers, ffn, hidden)
    bd_full = ffn_down_T @ embed_w.t()  # (layers, ffn, vocab)

    w_M, w_E, w_L, w_S, w_b, cw_loaded = _load_confidence_weights(out_dir)

    cap_ffn, cap_block_in, cap_block_out, handles = _capture_full(model)
    frames = []
    n_params = sum(p.numel() for p in model.parameters())
    cpath = _resolve_corpus(corpus_path)

    try:
        t_mem = time.time()
        stories = _load_probe_stories(cpath, MEMORY_PROBE_SEED, MEMORY_PROBE_STORIES, MEMORY_PROBE_MAX_STORY)
        memory_db = _build_memory_from_corpus(model, cap_ffn, stories, MEMORY_PROBE_TOPK)
        print(f"  memory probe step {step}: {len(stories)} stories, {time.time() - t_mem:.1f}s")

        meta = {
            "kind": "meta",
            "checkpoint": f"step_{step}.pt",
            "n_params": int(n_params),
            "layers": int(layers),
            "heads": int(heads),
            "ffn": int(ffn),
            "ffn_buckets": int(ffn // ds),
            "vocab": int(model.vocab),
            "seq": int(seq),
            "has_memory": True,
            "prompt": prompt,
            "prompt_bytes": list(prompt_bytes),
        }

        for _ in range(max_new):
            t_fwd = time.perf_counter()
            logits, _ = model(ids)
            fwd_ms = (time.perf_counter() - t_fwd) * 1000.0
            T = ids.size(1)
            last_logits = logits[0, -1].float()

            # stack ffn activations once per token. (layers, ffn) signed, used by
            # ffn_full/top, dla, etc. abs version reused for the bucket panels.
            acts_all = torch.stack([
                F.gelu(cap_ffn[L][0, -1])
                for L in range(layers)
            ], dim=0).float()                          # (layers, ffn)
            acts_abs = acts_all.abs()                  # (layers, ffn)
            grouped = acts_abs.view(layers, -1, ds)
            bucket_vals_all, bucket_argmax_all = grouped.max(dim=2)   # (layers, ffn/ds)
            mx_per = bucket_vals_all.max(dim=1).values.clamp(min=1e-9).unsqueeze(1)
            u8_all = (bucket_vals_all / mx_per * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            argmax_u8_all = bucket_argmax_all.to(torch.uint8).cpu().numpy()
            sat_all = (acts_abs >= INT8_SAT_THRESHOLD).float().mean(dim=1).cpu().tolist()
            top_v_all, top_i_all = torch.topk(acts_abs, GEN_NEURON_TOP_K, dim=1)
            top_v_all = top_v_all.cpu().tolist()
            top_i_all = top_i_all.cpu().tolist()
            ffn_full, ffn_top, ffn_argmax, saturation = [], [], [], []
            for L in range(layers):
                ffn_full.append(u8_all[L].tolist())
                ffn_argmax.append(argmax_u8_all[L].tolist())
                saturation.append(round(float(sat_all[L]), 5))
                ffn_top.append([{"id": int(i), "v": round(float(x), 3)}
                                for x, i in zip(top_v_all[L], top_i_all[L])])

            # attention: recompute per layer using captured block input.
            # canonical block applies RMSNorm (n1) then attn.qkv linear.
            attn_w_all = torch.zeros(layers, heads, T, device=device)
            for L in range(layers):
                blk = model.blocks[L]
                xin = cap_block_in[L]
                n1  = blk.n1
                xf  = xin.float()
                rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + n1.eps)
                hh  = (xf * rms * n1.weight).to(xin.dtype)
                qkv = F.linear(hh, blk.attn.qkv.weight)
                q, k, _v = qkv.chunk(3, dim=-1)
                q = q.view(1, T, heads, head_dim).transpose(1, 2)
                k = k.view(1, T, heads, head_dim).transpose(1, 2)
                scores = torch.matmul(q[:, :, -1:, :], k.transpose(-2, -1)) / math.sqrt(head_dim)
                attn_w_all[L] = F.softmax(scores.float(), dim=-1)[0, :, 0, :]
            ent_all     = -(attn_w_all * (attn_w_all + 1e-12).log()).sum(dim=-1)   # (layers, heads)
            attn_to_pos = attn_w_all.sum(dim=(0, 1))                               # (T,)
            top_n = min(GEN_ATTN_TOP_POS, T)
            top_v_a, top_i_a = torch.topk(attn_w_all, top_n, dim=-1)               # (layers, heads, top_n)
            ent_cpu   = ent_all.cpu().tolist()
            top_v_cpu = top_v_a.cpu().tolist()
            top_i_cpu = top_i_a.cpu().tolist()
            attn = []
            for L in range(layers):
                heads_data = []
                for hh in range(heads):
                    heads_data.append({
                        "ent": round(float(ent_cpu[L][hh]), 3),
                        "top": [{"p": int(p), "w": round(float(x), 3)}
                                for x, p in zip(top_v_cpu[L][hh], top_i_cpu[L][hh])],
                    })
                attn.append(heads_data)

            flow_v, flow_i = torch.topk(attn_to_pos, min(GEN_INFO_FLOW_TOP, T))
            flow_max = float(attn_to_pos.max().clamp(min=1e-9).item())
            info_flow = [{"p": int(p), "w": round(float(x) / flow_max, 3)}
                         for x, p in zip(flow_v.tolist(), flow_i.tolist())]

            # res / contrib: vectorized norms.
            rin_stack  = torch.stack([cap_block_in[L][0, -1]  for L in range(layers)]).float()
            rout_stack = torch.stack([cap_block_out[L][0, -1] for L in range(layers)]).float()
            res_norms     = [round(float(x), 3) for x in rout_stack.norm(dim=1).cpu().tolist()]
            contributions = [round(float(x), 3) for x in (rout_stack - rin_stack).norm(dim=1).cpu().tolist()]

            # per-layer lens: top-3 byte preds + argmax (used for lens_consistency).
            lens_logits_all = rout_stack @ W_E_T               # (layers, vocab)
            lens_probs_all  = F.softmax(lens_logits_all, dim=-1)
            lens_top_p, lens_top_i = torch.topk(lens_probs_all, GEN_LENS_TOP_P, dim=-1)
            lens_argmax_all = lens_logits_all.argmax(dim=-1).cpu().tolist()
            lens_top_p_cpu  = lens_top_p.cpu().tolist()
            lens_top_i_cpu  = lens_top_i.cpu().tolist()
            lens = []
            lens_argmax = lens_argmax_all
            for L in range(layers):
                lens.append([{"b": int(b), "p": round(float(pp), 3)}
                             for pp, b in zip(lens_top_p_cpu[L], lens_top_i_cpu[L])])

            probs = F.softmax(last_logits, dim=-1)
            entropy_bits = float(-(probs * (probs + 1e-12).log2()).sum().item())
            cv, ci = torch.topk(probs, GEN_NEXT_CANDIDATES)
            candidates = [{"b": int(b), "p": round(float(pp), 3)}
                          for pp, b in zip(cv.tolist(), ci.tolist())]

            scaled = last_logits / max(temperature, 1e-6)
            sv, si = torch.topk(scaled, top_k)
            mask = torch.full_like(scaled, float("-inf"))
            mask.scatter_(0, si, sv)
            sample_probs = F.softmax(mask, dim=-1)
            nxt = int(torch.multinomial(sample_probs, 1).item())
            surprise_bits = float(-math.log2(float(probs[nxt].item()) + 1e-12))
            argmax_byte = int(probs.argmax().item())

            # decisiveness: per-layer max_abs/mean_abs of (delta @ embed.T).
            delta_stack  = (rout_stack - rin_stack)             # (layers, hidden)
            logit_delta  = delta_stack @ W_E_T                  # (layers, vocab)
            ad           = logit_delta.abs()
            dec_score    = (ad.max(dim=1).values / ad.mean(dim=1).clamp(min=1e-8))
            decisiveness = [round(float(x), 3) for x in dec_score.cpu().tolist()]

            # confidence components, mirror engine/src/main.c.
            # margin: (top - second) / sigma_logit on the float logit vector.
            top1, top2 = torch.topk(last_logits, 2)
            sigma = float(last_logits.std(unbiased=False).item())
            margin = float((top1[0] - top1[1]).item()) / sigma if sigma > 1e-6 else 0.0
            # entropy_score: 1 - H(p)/log2(V).
            H = float(-(probs * (probs + 1e-12).log()).sum().item())
            entropy_score = 1.0 - (H / math.log(vocab))
            entropy_score = max(0.0, min(1.0, entropy_score))
            # lens_consistency: fraction of layers whose lens argmax == sampled byte.
            lens_consistency = sum(1 for am in lens_argmax if am == nxt) / float(layers)
            # residual_stab: mean pearson r of (residual_post[L] * embed[byte]) across pairs.
            # vectorize: stack residuals -> (layers, hidden), elem-mul by embed[byte],
            # mean-center per row, then pearson r between consecutive rows.
            erow = embed_w[nxt]                                    # (hidden,)
            res_stack = torch.stack([cap_block_out[L][0, -1].float()
                                     for L in range(layers)], dim=0)  # (layers, hidden)
            vec = res_stack * erow                                 # (layers, hidden)
            vec_c = vec - vec.mean(dim=1, keepdim=True)            # (layers, hidden)
            norms = vec_c.pow(2).sum(dim=1).sqrt()                 # (layers,)
            num   = (vec_c[:-1] * vec_c[1:]).sum(dim=1)            # (layers-1,)
            den   = (norms[:-1] * norms[1:]).clamp(min=1e-12)
            r_pair = (num / den).clamp(-1.0, 1.0)
            residual_stab = float(r_pair.mean().item()) if r_pair.numel() > 0 else 0.0

            if cw_loaded:
                z = w_M * margin + w_E * entropy_score + w_L * lens_consistency \
                    + w_S * residual_stab + w_b
            else:
                z = 0.5 * (margin + entropy_score + lens_consistency + residual_stab) - 1.0
            confidence = _sigmoid(z)

            # DLA top-K, mirrors engine model.c::dla_top.
            # contrib = ffn_act * BD[L][n, byte]. reuse pre-stacked acts_all.
            def _dla_for(target_byte):
                bd_col  = bd_full[:, :, target_byte]              # (layers, ffn)
                contrib = acts_all * bd_col                       # (layers, ffn)
                flat    = contrib.abs().reshape(-1)
                k = min(GEN_DLA_TOPK, flat.numel())
                _, top_idx = torch.topk(flat, k)
                idx_list = top_idx.cpu().tolist()
                acts_cpu    = acts_all.cpu()
                bd_col_cpu  = bd_col.cpu()
                contrib_cpu = contrib.cpu()
                out = []
                for ti in idx_list:
                    L = ti // ffn
                    n = ti %  ffn
                    out.append({
                        "layer":   int(L),
                        "neuron":  int(n),
                        "act":     round(float(acts_cpu[L, n].item()),    4),
                        "w":       round(float(bd_col_cpu[L, n].item()),  5),
                        "contrib": round(float(contrib_cpu[L, n].item()), 4),
                    })
                while len(out) < GEN_DLA_TOPK:
                    out.append({"layer": 0, "neuron": 0, "act": 0.0, "w": 0.0, "contrib": 0.0})
                return out

            dla_picked = _dla_for(nxt)
            dla_argmax = _dla_for(argmax_byte)
            dla_cand   = [_dla_for(int(c["b"])) for c in candidates]

            frames.append({
                "kind": "token",
                "byte": nxt, "argmax_byte": argmax_byte, "T": int(T),
                "fwd_ms": round(fwd_ms, 2),
                "entropy_bits": round(entropy_bits, 3),
                "surprise_bits": round(surprise_bits, 3),
                "ffn_full": ffn_full, "ffn_top": ffn_top,
                "ffn_argmax": ffn_argmax, "ffn_downsample": int(ds),
                "saturation": saturation,
                "decisiveness": decisiveness,
                "dla_picked": dla_picked,
                "dla_argmax": dla_argmax,
                "dla_cand":   dla_cand,
                "margin":           round(float(margin),           4),
                "entropy":          round(float(entropy_score),    4),
                "lens_consistency": round(float(lens_consistency), 4),
                "residual_stab":    round(float(residual_stab),    4),
                "confidence":       round(float(confidence),       4),
                "attn": attn, "info_flow": info_flow,
                "res": res_norms, "contrib": contributions,
                "lens": lens, "cand": candidates,
                "memory": _memory_lookup(ffn_top, memory_db),
                "backend": "training",
            })

            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
            if ids.size(1) >= seq:
                ids = ids[:, -(seq - 1):]
    finally:
        for h in handles: h.remove()
        if was_training: model.train()

    out_path = os.path.join(out_dir, f"step_{step}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "frames": frames}, f, ensure_ascii=False)

    try:
        model_dir = os.path.dirname(os.path.dirname(out_dir))
        neuron_memory_path = os.path.join(model_dir, "neuron_memory.json")
        nm_payload = {"neurons": {}, "step": int(step)}
        for L_str, neuron_map in (memory_db or {}).items():
            nm_payload["neurons"][str(L_str)] = {}
            for n_str, stories_list in (neuron_map or {}).items():
                if not stories_list: continue
                nm_payload["neurons"][str(L_str)][str(n_str)] = [
                    {"text": s.get("text", "")[:512],
                     "score": round(float(s.get("score", 0.0)), 4),
                     "peak_pos": int(s.get("peak_pos", 0))}
                    for s in stories_list
                ]
        with open(neuron_memory_path, "w", encoding="utf-8") as f:
            json.dump(nm_payload, f, ensure_ascii=False)
    except Exception as e:
        print(f"  WARN: neuron_memory.json write failed: {e}", flush=True)

    return out_path, len(frames), round(time.time() - t0, 3)


def _ffn_layer_weights(model):
    for L, blk in enumerate(model.blocks):
        yield L, blk.ff.up.weight.detach(), blk.ff.down.weight.detach()


def _prev_state_path(out_dir):
    return os.path.join(out_dir, "_classroom_prev_state.npz")


@torch.no_grad()
def dump_classroom(model, out_dir: str, step: int):
    """Write classroom_step_<N>.json: param count, int8/int4 byte budget,
    per-layer weight L2 delta, alive-neuron count per FFN layer.

    State carry between checkpoints is compact: per-tensor L2 norm (scalar) +
    per-FFN-layer ffn_up row L2 norms. The full float32 state is never
    serialized. diffs are computed from these lightweight summaries.
    """
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    was_training = model.training
    model.eval()

    params = sum(p.numel() for p in model.parameters())

    # cheap summaries on-device. l2 norm per named tensor; per-row l2 for ffn_up.
    cur_norms = {name: float(p.detach().float().norm().item())
                 for name, p in model.named_parameters()}

    cur_ffn_rows = {}  # f"layer_{L}" -> [N] row L2 norms (float32)
    for L, up_w, _down_w in _ffn_layer_weights(model):
        cur_ffn_rows[f"layer_{L}"] = up_w.float().norm(dim=1).cpu().numpy().astype(np.float32)

    prev_path = _prev_state_path(out_dir)
    have_prev = os.path.isfile(prev_path)
    prev_norms = None
    prev_rows  = {}
    if have_prev:
        try:
            z = np.load(prev_path, allow_pickle=False)
            if "names" in z.files and "norms" in z.files:
                prev_norms = dict(zip([str(s) for s in z["names"]], [float(x) for x in z["norms"]]))
            for k in z.files:
                if k.startswith("rows_"):
                    prev_rows[k[5:]] = z[k]
        except (OSError, ValueError, KeyError):
            prev_norms = None
            prev_rows  = {}
            have_prev  = False

    weight_delta_l2 = {}
    if prev_norms is not None:
        # cheap proxy: |‖w_t‖_2 − ‖w_{t-1}‖_2|. exact L2 of the delta would
        # require the full snapshot. too expensive at checkpoint time. on
        # 80M this is ~50 MB/save and ~10 s on CPU. norm-of-deltas isn't a
        # proper distance but it's a faithful indicator for the dashboard.
        for name, n_now in cur_norms.items():
            n_prev = prev_norms.get(name)
            if n_prev is None: continue
            weight_delta_l2[name] = round(abs(n_now - n_prev), 6)

    alive_per_layer = {}
    for L, _, _ in _ffn_layer_weights(model):
        key = f"layer_{L}"
        rows_now = cur_ffn_rows[key]
        rows_prev = prev_rows.get(key)
        if rows_prev is not None and rows_prev.shape == rows_now.shape:
            alive = int((np.abs(rows_now - rows_prev) > 1e-4).sum())
        else:
            alive = int(rows_now.shape[0])
        alive_per_layer[key] = alive

    # write compact summaries for next step's diff. ~1 MB on 80M.
    try:
        # store names as fixed-width unicode so np.load(allow_pickle=False) works.
        names_list = list(cur_norms.keys())
        max_w = max((len(s) for s in names_list), default=1)
        names = np.array(names_list, dtype=f"<U{max(max_w, 1)}")
        norms = np.array(list(cur_norms.values()), dtype=np.float32)
        rows_kw = {f"rows_{k}": v for k, v in cur_ffn_rows.items()}
        np.savez(prev_path, names=names, norms=norms, **rows_kw)
    except OSError:
        pass

    if was_training: model.train()
    elapsed = time.time() - t0

    out = {
        "step":         int(step),
        "precision":    _precision_tag(model),
        "params":       int(params),
        "int8_bytes":   int(params),
        "int4_bytes":   int(params // 2),
        "weight_delta_l2":      weight_delta_l2,
        "alive_neurons_per_layer": alive_per_layer,
        "time_s":       round(elapsed, 4),
    }
    path = os.path.join(out_dir, f"classroom_step_{step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return path


@torch.no_grad()
def _bytes_perplexity(model, raw: bytes, device):
    """Mean per-byte perplexity on `raw` bytes, chunked at model.seq length."""
    seq = model.seq
    if len(raw) < 2:
        return float("nan"), 0
    arr = np.frombuffer(raw, dtype=np.uint8)
    n = len(arr) - 1
    total_nll = 0.0
    total_n   = 0
    # slide non-overlapping windows of length seq+1; predict the last seq bytes.
    step = seq
    i = 0
    while i + 2 <= len(arr):
        end = min(i + seq + 1, len(arr))
        chunk = torch.tensor(arr[i:end].copy(), dtype=torch.long, device=device).unsqueeze(0)
        if chunk.shape[1] < 2: break
        x = chunk[:, :-1]
        y = chunk[:, 1:]
        logits, _ = model(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        nll = -log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        total_nll += float(nll.sum().item())
        total_n   += int(y.numel())
        i += step
        if total_n >= n: break
    if total_n == 0:
        return float("nan"), 0
    mean_nll = total_nll / total_n
    return float(math.exp(min(mean_nll, 20.0))), total_n


@torch.no_grad()
def dump_grades(model, out_dir: str, step: int):
    """Write grades_step_<N>.json: per-grade-band byte perplexity for any
    available plugins/corpus/grade_<level>_eval.bin. Missing bins are skipped."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    grades = {}
    estimated = "none"
    for level in GRADE_LEVELS:
        bin_path = paths.grade_eval_path(level)
        if not os.path.isfile(bin_path):
            continue
        try:
            with open(bin_path, "rb") as f:
                raw = f.read(GRADE_BYTES)
        except OSError:
            continue
        ppl, n = _bytes_perplexity(model, raw, device)
        grades[level] = {"ppl": round(ppl, 4) if not math.isnan(ppl) else None,
                         "n_bytes": int(n)}
        if estimated == "none" and ppl == ppl and ppl < GRADE_PPL_PASS:
            estimated = level

    if was_training: model.train()
    out = {
        "step":      int(step),
        "precision": _precision_tag(model),
        "grades":    grades,
        "estimated_reading_grade": estimated,
        "time_s":    round(time.time() - t0, 4),
    }
    path = os.path.join(out_dir, f"grades_step_{step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return path


@torch.no_grad()
def _concept_surprise_bits(model, preamble: str, target: str, device):
    """Surprise of `target` bytes given `preamble`, in bits per byte (mean)."""
    seq = model.seq
    pre = preamble.encode("utf-8", errors="replace")
    tgt = target.encode("utf-8", errors="replace")
    if len(tgt) == 0:
        return float("nan")
    full = (pre + tgt)[: seq + 1]
    # need at least 1 preamble byte + 1 target byte.
    if len(full) < 2 or len(pre) < 1:
        return float("nan")
    arr = np.frombuffer(full, dtype=np.uint8)
    chunk = torch.tensor(arr.copy(), dtype=torch.long, device=device).unsqueeze(0)
    x = chunk[:, :-1]
    y = chunk[:, 1:]
    logits, _ = model(x)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll_per_pos = -log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)[0]
    # only score positions that correspond to target bytes (last len(tgt) of nll).
    n_tgt = min(len(tgt), int(nll_per_pos.numel()))
    if n_tgt < 1:
        return float("nan")
    tail = nll_per_pos[-n_tgt:]
    mean_nats = float(tail.mean().item())
    return mean_nats / math.log(2.0)


def _read_concept_neurons(cap_ffn, layers, position):
    out = []
    for L in range(layers):
        ffn = cap_ffn[L]
        if ffn is None or position >= ffn.shape[1] or position < 0:
            continue
        raw = ffn[0, position]
        act = F.gelu(raw)
        mag = act.abs().float()
        k = min(CONCEPT_TOP_K_PER_LAYER, mag.numel())
        if k <= 0:
            continue
        _, idx = torch.topk(mag, k)
        for i in range(k):
            out.append({
                "layer": L,
                "id":    int(idx[i].item()),
                "v":     round(float(act[idx[i]].item()), 3),
            })
    return out


@torch.no_grad()
def dump_concepts(model, out_dir: str, step: int):
    """Write concepts_step_<N>.json: per-concept surprise (bits/byte) plus
    per-layer top-K firing FFN neurons at the commit position, on the fixed
    list of 50 concept probes. Backwards-compatible with old readers — the
    `top_neurons` field is additive."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    cap_ffn, _cap_post, handles = _capture(model)
    out_concepts = {}
    try:
        for name, preamble, target in CONCEPTS:
            s = _concept_surprise_bits(model, preamble, target, device)
            pre_len = len(preamble.encode("utf-8", errors="replace"))
            commit_pos = max(0, pre_len - 1)
            top_neurons = _read_concept_neurons(cap_ffn, model.layers, commit_pos)
            out_concepts[name] = {
                "surprise_bits": round(s, 4) if s == s else None,
                "top_neurons":   top_neurons,
            }
    finally:
        for h in handles: h.remove()

    if was_training: model.train()
    out = {
        "step":             int(step),
        "precision":        _precision_tag(model),
        "concepts":         out_concepts,
        "top_k_per_layer":  CONCEPT_TOP_K_PER_LAYER,
        "time_s":           round(time.time() - t0, 4),
    }
    path = os.path.join(out_dir, f"concepts_step_{step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return path


def _precision_tag(model):
    """Map a model instance to the precision string the dashboard reads.
    qat2 simulators carry the QAT2Block class name; mamba2 trainer uses
    Mamba2Veritate; everything else is fp32 baseline."""
    cls = type(model).__name__
    if "QAT2" in cls:    return "qat2"
    if "Mamba2" in cls:  return "mamba2-fp32"
    if "QAT" in cls:     return "qat"
    return "fp32"


@torch.no_grad()
def dump_surprise(model, prompt: str, out_dir: str, step: int):
    """Per-position surprise (bits/byte) on the canonical probe prompt.
    Feeds the surprise atlas in the Learning tab. Cheap: one forward, no sampling."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    raw = prompt.encode("utf-8", errors="replace")[: model.seq + 1]
    if len(raw) < 2:
        if was_training: model.train()
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    chunk = torch.tensor(arr.copy(), dtype=torch.long, device=device).unsqueeze(0)
    x = chunk[:, :-1]
    y = chunk[:, 1:]
    logits, _ = model(x)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)[0]
    surprise_bits = (nll / math.log(2.0)).cpu().tolist()
    if was_training: model.train()
    out = {
        "step":      int(step),
        "precision": _precision_tag(model),
        "prompt":    prompt,
        "tokens":    [int(b) for b in arr[1:].tolist()],
        "surprise":  [round(float(s), 4) for s in surprise_bits],
        "time_s":    round(time.time() - t0, 4),
    }
    path = os.path.join(out_dir, f"surprise_step_{step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return path


@torch.no_grad()
def dump_quant_kl(model, prompt: str, out_dir: str, step: int, n_levels: int = 127):
    """Single-scalar FP32-vs-INT8 next-byte KL on the probe prompt. Mirrors
    mri/server/brain.py::compute_quant_kl. The dashboard's FP32-vs-INT8 logit
    divergence panel reads this value. Skipped quietly on QAT2 sim models since
    the running forward already simulates INT8, value would always be ~0."""
    if _precision_tag(model) == "qat2":
        # QAT2 forward already simulates INT8 throughout. The legacy panel's
        # value is always near zero for QAT2 models; emit it as such for shape.
        out_path = os.path.join(out_dir, f"quant_kl_step_{step}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"step": int(step), "precision": "qat2",
                       "quant_kl_bits": 0.0, "n_levels": n_levels,
                       "note": "qat2 forward simulates int8 inline; gap is zero by construction"},
                      f, ensure_ascii=False)
        return out_path
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    prompt_bytes = (prompt or " ").encode("utf-8")
    ids = torch.tensor([b for b in prompt_bytes], dtype=torch.long, device=device).unsqueeze(0)
    if ids.size(1) >= model.seq:
        ids = ids[:, -(model.seq - 1):]
    logits_fp, _ = model(ids)
    p_fp = F.softmax(logits_fp[0, -1].float(), dim=-1)
    backup = {}
    for name, p in model.named_parameters():
        if p.dim() < 2: continue
        backup[name] = p.data.clone()
        max_abs = p.data.abs().max().clamp(min=1e-8)
        scale = max_abs / n_levels
        p.data = torch.clamp(torch.round(p.data / scale), -n_levels, n_levels) * scale
    try:
        logits_q, _ = model(ids)
        p_q = F.softmax(logits_q[0, -1].float(), dim=-1)
    finally:
        for name, p in model.named_parameters():
            if name in backup: p.data = backup[name]
        if was_training: model.train()
    kl_bits = float((p_fp * ((p_fp + 1e-12).log2() - (p_q + 1e-12).log2())).sum())
    out = {
        "step":          int(step),
        "precision":     _precision_tag(model),
        "quant_kl_bits": round(max(0.0, kl_bits), 5),
        "n_levels":      n_levels,
        "time_s":        round(time.time() - t0, 4),
    }
    path = os.path.join(out_dir, f"quant_kl_step_{step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return path


def _load_checkpoint(ckpt_path):
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    from veritate.model import Veritate
    s = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = s.get("args") or {}
    required = ("vocab", "hidden", "layers", "ffn", "heads", "seq")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise RuntimeError(f"{ckpt_path}: missing shape fields in args: {missing}. "
                           "the trainer must record full shape in the checkpoint args dict.")
    model = Veritate(
        vocab=cfg["vocab"], hidden=cfg["hidden"],
        layers=cfg["layers"], ffn=cfg["ffn"],
        heads=cfg["heads"], seq=cfg["seq"],
    )
    model.load_state_dict(s["model"], strict=False)
    return model, int(s.get("step", 0))


def main():
    ap = argparse.ArgumentParser(description="Dump the full hook artifact suite for a single checkpoint.")
    ap.add_argument("--checkpoint", required=True, help="path to step_<N>.pt")
    ap.add_argument("--out_dir",    required=True, help="output dir (typically models/<name>/hooks/step_<N>/)")
    ap.add_argument("--step",       type=int, default=None, help="override step number")
    ap.add_argument("--prompt",     default=PROBE_PROMPT)
    ap.add_argument("--all",        action="store_true",
                    help="also write classroom/grades/concepts dumps")
    args = ap.parse_args()

    model, ckpt_step = _load_checkpoint(args.checkpoint)
    step = args.step if args.step is not None else ckpt_step
    j, n = dump_probe(model, args.prompt, args.out_dir, step)
    print(f"wrote {j}")
    print(f"wrote {n}")
    if args.all:
        print(f"wrote {dump_classroom(model, args.out_dir, step)}")
        print(f"wrote {dump_grades(model,    args.out_dir, step)}")
        print(f"wrote {dump_concepts(model,  args.out_dir, step)}")
        print(f"wrote {dump_surprise(model,  args.prompt, args.out_dir, step)}")
        print(f"wrote {dump_quant_kl(model,  args.prompt, args.out_dir, step)}")
        path, nframes, secs = dump_generation(model, args.prompt, args.out_dir, step)
        print(f"wrote {path} ({nframes} frames, {secs}s)")


if __name__ == "__main__":
    main()
