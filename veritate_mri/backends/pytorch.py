# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - pytorch runtime wrapper. forward hooks capture activations for the mri viewer.
# veritate_mri/backends/pytorch.py
# ------------------------------------------------------------------------------------
# Imports:

import heapq
import json
import math
import os
import sys
import threading
import time

import numpy as np

from runtime import logs as logmod
from training import confidence as confidence_mod

# ------------------------------------------------------------------------------------
# Constants

HERE = os.path.dirname(os.path.abspath(__file__))


FFN_BUCKET_TARGET = 256
ATTN_TOP_POS      = 6
NEXT_CANDIDATES   = 12
NEURON_TOP_K      = 8
MEMORY_TOP_N      = 5
INFO_FLOW_TOP     = 8

# matches training/qat.py: post-GELU activations get fake-quant'd at scale 32 in QAT mode 2.
# saturation = fraction of activations whose magnitude would clip the int8 range under that scale.
ACTIVATION_INT8_SCALE = 32.0
INT8_SAT_THRESHOLD    = 127.0 / ACTIVATION_INT8_SCALE  # ~3.97


# ------------------------------------------------------------------------------------
# Functions

def load_memory(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob.get("neurons", {})
    except Exception as e:
        logmod.warn("backends.pytorch", f"memory load skipped: {e}")
        return None


def _shape_from_state_dict(sd, cfg):
    """Infer shape from a state_dict + checkpoint args. Handles both:
      - canonical Veritate (has `pos_emb.weight`)
      - veritate_800m / RoPE variants (no pos_emb; seq comes from cfg)
    """
    vocab, hidden = sd["tok_emb.weight"].shape
    if "pos_emb.weight" in sd:
        seq = sd["pos_emb.weight"].shape[0]
    else:
        # RoPE-based model — seq isn't in the state dict; pull from args.
        seq = int(cfg.get("seq") or 0)
        if seq <= 0:
            raise RuntimeError(
                "No pos_emb.weight in checkpoint and no seq in cfg/args. "
                "RoPE-based checkpoints must record `seq` in training_args."
            )
    layers        = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    ffn_per_layer = [sd[f"blocks.{L}.ff.up.weight"].shape[0] for L in range(layers)]
    ffn           = ffn_per_layer[0] if all(f == ffn_per_layer[0] for f in ffn_per_layer) else ffn_per_layer
    heads         = int(cfg.get("heads") or 0)
    if heads <= 0 or hidden % heads != 0:
        target = max(1, hidden // 64)
        for h in sorted({d for d in range(1, hidden + 1) if hidden % d == 0},
                        key=lambda d: (abs(d - target), -d)):
            heads = h
            break
    return {"vocab": vocab, "hidden": hidden, "layers": layers,
            "ffn": ffn, "heads": heads, "seq": seq}


def _is_veritate_800m(sd):
    """A veritate_800m checkpoint has MTP head weights at `mtp.transforms.{i}.weight`
    AND no `pos_emb.weight` (RoPE-only positioning)."""
    return "pos_emb.weight" not in sd and any(k.startswith("mtp.transforms.") for k in sd)


def _is_veritate_85m_mtp(sd):
    """The 85m plugin trunk: learned pos_emb AND an MTP byte-0 head."""
    return "pos_emb.weight" in sd and any(k.startswith("mtp.transforms.") for k in sd)


def _is_rope_only(sd):
    """RoPE without MTP (e.g., experiments/v2/rope_85m checkpoints): no pos_emb,
    no mtp head."""
    return "pos_emb.weight" not in sd and not any(k.startswith("mtp.transforms.") for k in sd)


class Brain:
    def __init__(self, checkpoint, threads=1, memory=None):
        import torch
        import torch.nn.functional as F
        repo_root = os.path.normpath(os.path.join(HERE, "..", ".."))
        sys.path.insert(0, repo_root)
        from veritate_core.model import Veritate
        globals()["torch"] = torch
        globals()["F"] = F
        globals()["Veritate"] = Veritate
        torch.set_num_threads(threads)
        s = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Snapshot what we need from the checkpoint dict, then drop it so the
        # optimizer state (~8 GB on 1B) doesn't sit resident through inference.
        # Different training scripts persist their shape under different keys:
        # the canonical and 800M trainers use `args`, the v2 experiments
        # (rope_85m, etc.) use `config`. Merge with `config` taking precedence
        # since it tends to be the canonical shape spec.
        cfg = {}
        cfg.update(s.get("args", {}) or {})
        cfg.update(s.get("config", {}) or {})
        sd = s["model"]
        del s
        self.checkpoint = os.path.basename(checkpoint)
        # Calibration weights for the confidence score live alongside the
        # checkpoint at models/<name>/confidence_weights.json. Falls back to the
        # engine main.c constants when absent (see confidence.WEIGHTS_FALLBACK).
        self._conf_weights = confidence_mod.load_weights(os.path.dirname(os.path.dirname(checkpoint)))
        if "tok_emb.weight" not in sd:
            plugin_name = str(cfg.get("plugin") or "").strip()
            tag = f" (plugin: {plugin_name})" if plugin_name else ""
            raise RuntimeError(
                "PyTorch inference is not enabled for this model" + tag + ". "
                "The dashboard backend supports vanilla Veritate checkpoints; "
                "non-vanilla architectures (Mixture-of-Experts, etc.) need their "
                "own runtime."
            )
        shape = _shape_from_state_dict(sd, cfg)

        # Dispatch to the right model class. The 800M plugin's `Veritate800M`
        # has the same trunk shape as canonical Veritate, plus RoPE positions
        # and an MTP head. State-dict layout for `tok_emb`, `blocks.*`, `n_out`,
        # `lm_head` matches, so most hooks downstream work unchanged. The MTP
        # head (mtp.transforms.*, mtp.norms.*) is loaded but unused by the
        # single-byte decode path here; an MTP-aware decoder lives in
        # `experiments/v2/mtp_decode/` and is wired separately.
        if _is_veritate_800m(sd):
            plugin_dir = os.path.join(repo_root, "plugins", "veritate_800m")
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            from plugin import Veritate800M  # type: ignore
            n_predict = int(cfg.get("n_predict") or 4)
            rope_base = float(cfg.get("rope_base") or 10000.0)
            self.model = Veritate800M(
                vocab=shape["vocab"], hidden=shape["hidden"], layers=shape["layers"],
                ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"],
                n_predict=n_predict, rope_base=rope_base,
            )
            # rope_cos / rope_sin are non-persistent buffers built at
            # construction; nothing in `sd` to load for them.
            self.model.load_state_dict(sd, strict=False)
        elif _is_veritate_85m_mtp(sd):
            plugin_dir = os.path.join(repo_root, "plugins", "veritate_85m")
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            from plugin import Veritate85M  # type: ignore
            n_predict = int(cfg.get("n_predict") or 2)
            self.model = Veritate85M(
                vocab=shape["vocab"], hidden=shape["hidden"], layers=shape["layers"],
                ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"],
                n_predict=n_predict,
            )
            self.model.load_state_dict(sd, strict=True)
        elif _is_rope_only(sd):
            # RoPE-only checkpoint (e.g., the experiments/v2/rope_85m finetunes
            # of the canonical 85M). State-dict layout matches canonical Veritate
            # MINUS pos_emb.weight; no MTP head.
            from veritate_core.model_rope import VeritateRoPE
            rope_base = float(cfg.get("rope_base") or 10000.0)
            self.model = VeritateRoPE(
                vocab=shape["vocab"], hidden=shape["hidden"], layers=shape["layers"],
                ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"],
                rope_base=rope_base,
            )
            # rope_cos / rope_sin are non-persistent buffers.
            self.model.load_state_dict(sd, strict=False)
        else:
            self.model = Veritate(**shape)
            self.model.load_state_dict(sd, strict=True)
        del sd  # frees the duplicate copy now that the model owns the params
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

        # pick a downsample factor so V_FFN / ds is close to FFN_BUCKET_TARGET
        # and divides evenly. for pruned models with per-layer ffn, pick a ds
        # that divides every layer's width.
        ffn = self.model.ffn if isinstance(self.model.ffn, int) else max(self.model.ffn_per_layer)
        ds = max(1, ffn // FFN_BUCKET_TARGET)
        while ds > 1 and ffn % ds != 0:
            ds -= 1
        self.ffn_downsample = ds

        self.cap_ffn      = [None] * self.model.layers
        self.cap_qkv      = [None] * self.model.layers
        self.cap_block_in = [None] * self.model.layers
        self.cap_block_out= [None] * self.model.layers
        # v8 ablation state. -1 == no ablation. set_ablation() updates these
        # before stream(); the ffn_down pre-hook zeros input[..., neuron] when
        # the layer matches.
        self._ablate_layer = -1
        self._ablate_neuron = -1
        for L, blk in enumerate(self.model.blocks):
            blk.ff.up.register_forward_hook(self._hook(self.cap_ffn, L))
            blk.attn.qkv.register_forward_hook(self._hook(self.cap_qkv, L))
            blk.register_forward_pre_hook(self._pre_hook(self.cap_block_in, L))
            blk.register_forward_hook(self._hook(self.cap_block_out, L))
            blk.ff.down.register_forward_pre_hook(self._ablation_pre_hook(L))

        # byte_direction[L] has shape (ffn, vocab). entry [n, b] is the scalar
        # contribution to byte b's logit when neuron (L, n) fires with activation 1.
        # = (ffn_down weight column for n) dotted with (embed row for b).
        # = (W_down.T @ W_E.T)[n, b]
        # used for direct logit attribution and per-neuron byte affinity profiles.
        with torch.no_grad():
            W_E = self.model.tok_emb.weight  # (vocab, hidden)
            self.byte_direction = []
            for blk in self.model.blocks:
                W_down = blk.ff.down.weight  # (hidden, ffn)
                table = (W_down.t() @ W_E.t()).contiguous()  # (ffn, vocab)
                self.byte_direction.append(table)
            # also project the embedding once for per-layer logit-delta computations
            self.W_E_T = W_E.t().contiguous()  # (hidden, vocab)

        self.lock = threading.Lock()
        self.memory = memory

    def _ablation_pre_hook(self, L):
        """Forward-pre-hook for blk.ff.down. When self._ablate_layer == L and
        the neuron index is in range, zeros input[..., neuron] before ffn_down
        runs. No-op otherwise. Mirrors model.c::forward_decode hook on d->ffn_up8."""
        def h(_module, inputs):
            if self._ablate_layer != L:
                return None
            n = self._ablate_neuron
            x = inputs[0]
            if n < 0 or n >= x.size(-1):
                return None
            x = x.clone()
            x[..., n] = 0.0
            return (x,)
        return h

    def set_ablation(self, layer, neuron):
        """Causal ablation knob (v8). layer and neuron == -1 disables.
        Persists until the next call. Read by the ffn_down pre-hook on every forward."""
        self._ablate_layer = int(layer)
        self._ablate_neuron = int(neuron)

    def _dla_for_byte(self, byte_idx, top_k=12):
        """Top FFN neurons by direct logit attribution to a single byte. Returns
        list of {layer, neuron, act, w, contrib}. Computed at the current token
        position using whatever activations are in cap_ffn."""
        m = self.model
        contrib = []
        for L in range(m.layers):
            act = F.gelu(self.cap_ffn[L][0, -1])
            w = self.byte_direction[L][:, byte_idx]
            contrib.append(act * w)
        mat = torch.stack(contrib)
        flat = mat.flatten()
        _, idx = torch.topk(flat.abs(), k=top_k)
        entries = []
        for i in idx.tolist():
            L = i // mat.size(1)
            n = i % mat.size(1)
            e = {
                "layer": int(L),
                "neuron": int(n),
                "act": round(float(F.gelu(self.cap_ffn[L][0, -1, n])), 4),
                "w": round(float(self.byte_direction[L][n, byte_idx]), 5),
                "contrib": round(float(mat[L, n]), 4),
            }
            lbl = self.label_for(int(L), int(n))
            if lbl is not None:
                e["label"] = lbl
            entries.append(e)
        return entries

    def _dla_top(self, picked_byte, argmax_byte, top_k=12):
        return (self._dla_for_byte(picked_byte, top_k),
                self._dla_for_byte(argmax_byte, top_k))

    def _derive_label(self, stories):
        """Heuristic categorical label for a neuron from its top memory stories.
        Tries: (1) word-detector (peak falls on the same word across stories),
        (2) longest common substring near peak, (3) single literal byte,
        (4) byte-class. Returns dict with category, trigger, confidence (or None)."""
        if not stories:
            return None
        from collections import Counter

        # collect per-story: word containing peak, byte at peak, window around peak
        words = []
        peak_bytes = []
        windows = []
        for s in stories[:8]:
            text = s.get("text", "")
            peak = s.get("peak_pos")
            if peak is None or peak < 0 or peak >= len(text):
                continue
            peak_bytes.append(text[peak])
            # find word boundary around peak (alphanumeric run)
            ws = peak
            while ws > 0 and (text[ws - 1].isalnum() or text[ws - 1] == "'"):
                ws -= 1
            we = peak + 1
            while we < len(text) and (text[we].isalnum() or text[we] == "'"):
                we += 1
            word = text[ws:we].lower()
            words.append(word)
            # widened window for substring search: peak-5..peak+2 (8 bytes)
            wstart = max(0, peak - 5)
            wend = min(len(text), peak + 3)
            windows.append(text[wstart:wend])
        n = len(windows)
        if n < 3:
            return None

        # 1. word-detector: same alphanumeric word at peak in 50%+ of stories
        non_empty_words = [w for w in words if len(w) >= 2]
        if non_empty_words:
            top_word, count = Counter(non_empty_words).most_common(1)[0]
            if count / n >= 0.5 and len(top_word) >= 3:
                return {
                    "category": "word",
                    "trigger": top_word,
                    "confidence": round(count / n, 2),
                    "summary": top_word,
                }

        # 2. longest-common-substring across windows (anywhere in window, not just suffix)
        seen = set()
        candidates = []
        for w in windows:
            for L in range(2, min(8, len(w) + 1)):  # up to 7-grams
                for i in range(0, len(w) - L + 1):
                    sub = w[i:i + L]
                    if sub in seen:
                        continue
                    seen.add(sub)
                    if not sub.strip():
                        continue
                    cnt = sum(1 for ww in windows if sub in ww)
                    if cnt / n >= 0.6:
                        candidates.append((sub, cnt, L))
        if candidates:
            # rank: count desc, then length desc (more occurrences first; longer breaks ties)
            candidates.sort(key=lambda x: (-x[1], -x[2]))
            sub, cnt, L = candidates[0]
            cat = {2: "bigram", 3: "trigram", 4: "4gram",
                   5: "5gram", 6: "6gram", 7: "7gram"}.get(L, "ngram")
            return {
                "category": cat,
                "trigger": sub,
                "confidence": round(cnt / n, 2),
                "summary": sub,
            }

        # 3. single byte at peak
        top_byte, count = Counter(peak_bytes).most_common(1)[0]
        if count / n >= 0.5:
            return {
                "category": "single",
                "trigger": top_byte,
                "confidence": round(count / n, 2),
                "summary": top_byte,
            }

        # 4. byte-class fallback
        def cls(c):
            o = ord(c)
            if c in "aeiouAEIOU": return "vowel"
            if c.isalpha():       return "consonant"
            if c.isdigit():       return "digit"
            if c in " \t\n":      return "whitespace"
            if 33 <= o < 127:     return "punct"
            return "other"
        classes = [cls(b) for b in peak_bytes]
        top_cls, count = Counter(classes).most_common(1)[0]
        if count / n >= 0.75 and top_cls != "other":
            return {
                "category": top_cls,
                "trigger": None,
                "confidence": round(count / n, 2),
                "summary": top_cls,
            }
        return None

    def compute_all_neuron_labels(self):
        """Iterate all neurons in self.memory and build a (layer, neuron_id) -> label
        cache. Idempotent. Costs a few seconds at startup; lookups are O(1) after."""
        labels = {}
        if self.memory is None:
            self.neuron_labels = labels
            return labels
        for L_str, layer_mem in self.memory.items():
            try:
                L = int(L_str)
            except ValueError:
                continue
            for n_str, stories in layer_mem.items():
                try:
                    n = int(n_str)
                except ValueError:
                    continue
                lbl = self._derive_label(stories)
                if lbl is not None:
                    labels[(L, n)] = lbl
        self.neuron_labels = labels
        return labels

    def label_for(self, layer, neuron_id):
        """Lookup. Returns label dict or None."""
        labels = getattr(self, "neuron_labels", None)
        if not labels:
            return None
        return labels.get((int(layer), int(neuron_id)))

    def neuron_byte_affinity(self, layer, neuron_id, top_k=5):
        """Top + and - bytes that this neuron writes toward when it fires."""
        row = self.byte_direction[layer][neuron_id]  # (vocab,)
        vals, idx = torch.topk(row, k=top_k)
        pos = [{"b": int(i), "w": round(float(v), 4)} for v, i in zip(vals.tolist(), idx.tolist())]
        vals_n, idx_n = torch.topk(-row, k=top_k)
        neg = [{"b": int(i), "w": round(-float(v), 4)} for v, i in zip(vals_n.tolist(), idx_n.tolist())]
        return {"pos": pos, "neg": neg, "all": [round(float(x), 5) for x in row.tolist()]}

    def neuron_predecessors(self, layer, neuron_id, top_k=10):
        """Dynamic. For the most recent forward pass, the top earlier-layer FFN
        neurons whose write directions × activations contributed most to this
        neuron's pre-activation. 'Who made me fire?'"""
        if layer == 0 or self.cap_ffn[0] is None:
            return []
        m = self.model
        with torch.no_grad():
            read_dir = m.blocks[layer].ff.up.weight[neuron_id, :]  # (hidden,)
            per_layer = []
            for L_prev in range(layer):
                if self.cap_ffn[L_prev] is None:
                    return []
                prev_act = F.gelu(self.cap_ffn[L_prev][0, -1])  # (ffn,)
                W_down = m.blocks[L_prev].ff.down.weight  # (hidden, ffn)
                # contribution per neuron m at L_prev = prev_act[m] * (W_down[:, m] · read_dir)
                contribs = prev_act * (W_down.t() @ read_dir)  # (ffn,)
                per_layer.append(contribs)
            flat = torch.cat(per_layer)  # (layer * ffn,)
            ffn_n = m.ffn
            vals, idx = torch.topk(flat.abs(), min(top_k, flat.numel()))
        out = []
        for v_abs, i in zip(vals.tolist(), idx.tolist()):
            L_prev = i // ffn_n
            n = i % ffn_n
            c = float(flat[i])
            a = float(F.gelu(self.cap_ffn[L_prev][0, -1, n]))
            entry = {"layer": int(L_prev), "neuron": int(n),
                     "act": round(a, 4), "contrib": round(c, 4)}
            lbl = self.label_for(L_prev, n)
            if lbl is not None:
                entry["label"] = lbl
            out.append(entry)
        return out

    def neuron_successors(self, layer, neuron_id, top_k=8):
        """Static. Top later-layer FFN neurons whose read directions most strongly
        align with this neuron's write direction. 'Who listens to me?' Independent
        of input; reflects the model's wiring."""
        m = self.model
        if layer >= m.layers - 1:
            return []
        with torch.no_grad():
            write_dir = m.blocks[layer].ff.down.weight[:, neuron_id]  # (hidden,)
            per_layer = []
            for L_post in range(layer + 1, m.layers):
                read_mat = m.blocks[L_post].ff.up.weight  # (ffn, hidden)
                overlaps = read_mat @ write_dir  # (ffn,)
                per_layer.append(overlaps)
            flat = torch.cat(per_layer)
            ffn_n = m.ffn
            vals, idx = torch.topk(flat.abs(), min(top_k, flat.numel()))
        out = []
        for v_abs, i in zip(vals.tolist(), idx.tolist()):
            L_offset = i // ffn_n
            L_post = layer + 1 + L_offset
            n = i % ffn_n
            w = float(flat[i])
            entry = {"layer": int(L_post), "neuron": int(n), "w": round(w, 4)}
            lbl = self.label_for(L_post, n)
            if lbl is not None:
                entry["label"] = lbl
            out.append(entry)
        return out

    def neuron_stats(self, layer, neuron_id):
        """Cheap context for the modal: probe-max activation (if memory loaded),
        and current activation if a forward pass is in cache."""
        out = {"current_act": None, "probe_max": None, "current_pct": None}
        if self.cap_ffn[layer] is not None:
            try:
                a = float(F.gelu(self.cap_ffn[layer][0, -1, neuron_id]))
                out["current_act"] = round(a, 4)
            except Exception:
                pass
        if self.memory is not None:
            mem = self.memory.get(str(layer), {}).get(str(neuron_id), [])
            if mem:
                pmax = max(float(s.get("score", 0)) for s in mem)
                out["probe_max"] = round(pmax, 4)
                if out["current_act"] is not None and pmax > 1e-6:
                    out["current_pct"] = round(out["current_act"] / pmax * 100, 1)
        return out

    @staticmethod
    def _hook(buf, L):
        def hook(_m, _i, out): buf[L] = out
        return hook

    @staticmethod
    def _pre_hook(buf, L):
        def hook(_m, inp): buf[L] = inp[0]
        return hook

    def build_memory_from_corpus(self, corpus_bytes, n_stories=500, top_k=8, max_story_bytes=256, seed=7):
        # for each (layer, neuron), keep top-K training stories that activated it hardest.
        # uses the existing ffn_up hooks already wired on this model.
        m = self.model
        parts = corpus_bytes.split(b"\x00")
        parts = [p for p in parts if 32 <= len(p) <= max_story_bytes]
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(parts), size=min(n_stories, len(parts)), replace=False)
        stories = [parts[int(i)] for i in chosen]

        # tie-breaker counter so heapq never compares text/dicts
        counter = 0
        heaps = [[[] for _ in range(m.ffn)] for _ in range(m.layers)]
        for story in stories:
            ids = torch.tensor([b for b in story[:m.seq]], dtype=torch.long).unsqueeze(0)
            if ids.size(1) < m.seq:
                pad = torch.zeros(m.seq - ids.size(1), dtype=torch.long).unsqueeze(0)
                ids = torch.cat([ids, pad], dim=1)
            with torch.no_grad():
                _ = m(ids)
            real_len = min(len(story), m.seq)
            text = story[:real_len].decode("utf-8", errors="replace")
            for L in range(m.layers):
                act = F.gelu(self.cap_ffn[L])[0, :real_len].abs()
                # per-neuron max activation AND the position where it peaked
                per_neuron_max, per_neuron_argmax = act.max(dim=0)
                arr_v = per_neuron_max.cpu().numpy()
                arr_p = per_neuron_argmax.cpu().numpy()
                hl = heaps[L]
                for n in range(m.ffn):
                    score = float(arr_v[n])
                    peak_pos = int(arr_p[n])
                    h = hl[n]
                    counter += 1
                    entry = (score, counter, text, peak_pos)
                    if len(h) < top_k:
                        heapq.heappush(h, entry)
                    elif score > h[0][0]:
                        heapq.heapreplace(h, entry)

        out = {}
        for L in range(m.layers):
            out[str(L)] = {}
            for n in range(m.ffn):
                h = heaps[L][n]
                ranked = sorted(h, key=lambda x: -x[0])
                if ranked and ranked[0][0] > 0:
                    out[str(L)][str(n)] = [
                        {"text": t, "score": round(s, 3), "peak_pos": int(p)}
                        for s, _c, t, p in ranked
                    ]
        return out

    def compute_quant_kl(self, prompt, n_levels=127):
        with torch.no_grad():
            m = self.model
            prompt_bytes = (prompt or " ").encode("utf-8")
            ids = torch.tensor([b for b in prompt_bytes], dtype=torch.long).unsqueeze(0)
            if ids.size(1) >= m.seq:
                ids = ids[:, -(m.seq - 1):]
            logits_fp, _ = m(ids)
            p_fp = F.softmax(logits_fp[0, -1], dim=-1)
            backup = {}
            for name, p in m.named_parameters():
                if p.dim() < 2:
                    continue
                backup[name] = p.data.clone()
                max_abs = p.data.abs().max().clamp(min=1e-8)
                scale = max_abs / n_levels
                p.data = torch.clamp(torch.round(p.data / scale), -n_levels, n_levels) * scale
            try:
                logits_q, _ = m(ids)
                p_q = F.softmax(logits_q[0, -1], dim=-1)
            finally:
                for name, p in m.named_parameters():
                    if name in backup:
                        p.data = backup[name]
            kl_bits = float((p_fp * ((p_fp + 1e-12).log2() - (p_q + 1e-12).log2())).sum())
            return round(max(0.0, kl_bits), 5)

    def _memory_lookup(self, ffn_top):
        if self.memory is None:
            return []
        scores = {}
        for L, neurons in enumerate(ffn_top):
            layer_key = str(L)
            layer_mem = self.memory.get(layer_key)
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
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:MEMORY_TOP_N]
        return [{"text": t[:120], "score": round(s, 2)} for t, s in ranked]

    def stream(self, prompt, temperature=0.7, top_k_sample=40, max_new=200, addons_chain=None, constraint=None):
        m = self.model
        seq = m.seq
        if not prompt:
            prompt = " "
        prompt_bytes = prompt.encode("utf-8")
        ids = torch.tensor([b for b in prompt_bytes], dtype=torch.long).unsqueeze(0)
        if ids.size(1) >= seq:
            ids = ids[:, -(seq - 1):]

        if addons_chain is not None:
            addons_chain.reset()
            addons_chain.observe_bytes(prompt_bytes)

        # Optional output-shape constraint. Prime it with the prompt so its
        # internal grammar state matches what the model just observed.
        if constraint is not None:
            constraint.reset()
            _prime = getattr(constraint, "prime", None)
            if callable(_prime):
                _prime(prompt_bytes)
            else:
                for _b in prompt_bytes:
                    constraint.step(int(_b) & 0xff)

        yield {
            "kind": "meta",
            "checkpoint": self.checkpoint,
            "n_params": self.n_params,
            "layers": m.layers, "heads": m.heads, "ffn": m.ffn,
            "ffn_buckets": m.ffn // self.ffn_downsample,
            "vocab": m.vocab, "seq": m.seq,
            "has_memory": self.memory is not None,
            "prompt": prompt,
            "prompt_bytes": list(prompt_bytes),
        }

        head_dim = m.hidden // m.heads
        emb_w = m.tok_emb.weight

        for _ in range(max_new):
            t0 = time.perf_counter()
            with torch.no_grad():
                logits, _ = m(ids)
            fwd_ms = (time.perf_counter() - t0) * 1000
            T = ids.size(1)
            last_logits = logits[0, -1]

            ffn_full, ffn_top, ffn_argmax, saturation = [], [], [], []
            ds = self.ffn_downsample
            for L in range(m.layers):
                act = F.gelu(self.cap_ffn[L][0, -1]).abs()
                sat = float((act >= INT8_SAT_THRESHOLD).float().mean())
                saturation.append(round(sat, 5))
                grouped = act.view(-1, ds)
                bucket_vals, bucket_argmax = grouped.max(dim=1)
                mx = float(bucket_vals.max().clamp(min=1e-9))
                u8 = (bucket_vals / mx * 255).clamp(0, 255).to(torch.uint8).tolist()
                ffn_full.append(u8)
                ffn_argmax.append(bucket_argmax.to(torch.uint8).tolist())
                v, idx = torch.topk(act, NEURON_TOP_K)
                top_entries = []
                for x, i in zip(v.tolist(), idx.tolist()):
                    e = {"id": int(i), "v": round(float(x), 3)}
                    lbl = self.label_for(L, int(i))
                    if lbl is not None:
                        e["label"] = lbl
                    top_entries.append(e)
                ffn_top.append(top_entries)

            attn = []
            attn_to_pos = torch.zeros(T)
            for L in range(m.layers):
                qkv = self.cap_qkv[L]
                q, k, _ = qkv.chunk(3, dim=-1)
                q = q.view(1, T, m.heads, head_dim).transpose(1, 2)
                k = k.view(1, T, m.heads, head_dim).transpose(1, 2)
                scores = torch.matmul(q[:, :, -1:, :], k.transpose(-2, -1)) / math.sqrt(head_dim)
                w = F.softmax(scores, dim=-1)[0, :, 0, :]
                ent = -(w * (w + 1e-12).log()).sum(dim=-1)
                attn_to_pos += w.sum(dim=0)
                heads_data = []
                for h in range(m.heads):
                    wh = w[h]
                    vv, ii = torch.topk(wh, min(ATTN_TOP_POS, T))
                    heads_data.append({
                        "ent": round(float(ent[h]), 3),
                        "top": [{"p": int(p), "w": round(float(x), 3)}
                                for x, p in zip(vv.tolist(), ii.tolist())],
                    })
                attn.append(heads_data)

            flow_v, flow_i = torch.topk(attn_to_pos, min(INFO_FLOW_TOP, T))
            flow_max = float(attn_to_pos.max().clamp(min=1e-9))
            info_flow = [{"p": int(p), "w": round(float(x) / flow_max, 3)}
                         for x, p in zip(flow_v.tolist(), flow_i.tolist())]

            res_norms, contributions = [], []
            for L in range(m.layers):
                rin  = self.cap_block_in[L][0, -1]
                rout = self.cap_block_out[L][0, -1]
                res_norms.append(round(float(rout.norm()), 3))
                contributions.append(round(float((rout - rin).norm()), 3))

            lens = []
            for L in range(m.layers):
                r = self.cap_block_out[L][0, -1]
                lens_logits = r @ emb_w.T
                p = F.softmax(lens_logits, dim=-1)
                top_p, top_i = torch.topk(p, 3)
                lens.append([{"b": int(b), "p": round(float(pp), 3)}
                             for pp, b in zip(top_p.tolist(), top_i.tolist())])

            probs = F.softmax(last_logits, dim=-1)
            entropy_bits = float(-(probs * (probs + 1e-12).log2()).sum())
            cv, ci = torch.topk(probs, NEXT_CANDIDATES)
            candidates = [{"b": int(b), "p": round(float(pp), 3)}
                          for pp, b in zip(cv.tolist(), ci.tolist())]

            scaled = last_logits / max(temperature, 1e-6)
            if addons_chain is not None and len(addons_chain) > 0:
                scaled = addons_chain.bias_logits(scaled)
            # Apply output-shape constraint (JSON / vocab / stop-pattern) as a
            # -inf logit mask before topk. The constraint primes itself with the
            # prompt above so its grammar state is consistent.
            if constraint is not None:
                allowed_np = constraint.mask()
                allowed = torch.from_numpy(allowed_np)
                scaled = scaled.masked_fill(~allowed, float("-inf"))
                if not torch.isfinite(scaled).any():
                    # No legal byte; emit the stop event and bail.
                    yield {"kind": "stop", "reason": "constraint allowed no bytes"}
                    return
            sv, si = torch.topk(scaled, top_k_sample)
            mask = torch.full_like(scaled, float("-inf"))
            mask.scatter_(0, si, sv)
            sample_probs = F.softmax(mask, dim=-1)
            nxt = int(torch.multinomial(sample_probs, 1).item())
            surprise_bits = float(-math.log2(float(probs[nxt]) + 1e-12))
            argmax_byte = int(probs.argmax())

            # direct logit attribution (DLA): rank FFN neurons by their contribution
            # to the picked byte's logit and to the argmax (model's expected) byte's logit.
            # the gap is the surprise-explainer.
            dla_picked, dla_argmax = self._dla_top(nxt, argmax_byte)
            # v8: per-candidate DLA. one block per byte in `candidates`, ordered to match.
            dla_cand = [{"b": int(b), "entries": self._dla_for_byte(int(b))} for b in ci.tolist()]

            # per-layer decisiveness: how concentrated is each layer's contribution
            # to the residual when projected back to byte logits? max_abs / mean_abs.
            # tells you which layer either committed or stalled for this token.
            decisiveness = []
            for L in range(m.layers):
                delta = (self.cap_block_out[L][0, -1] - self.cap_block_in[L][0, -1])
                logit_delta = delta @ self.W_E_T  # (vocab,)
                ad = logit_delta.abs()
                score = float(ad.max() / ad.mean().clamp(min=1e-8))
                decisiveness.append(round(score, 3))

            memory = self._memory_lookup(ffn_top)

            # confidence + four components (rule 23: emit the same fields the
            # training-time dumper writes; the shared calculator lives at
            # veritate_mri/confidence.py so both paths agree by construction).
            # lens[L] is a list of top-3 {b, p} dicts; the top-1 byte is lens[L][0]["b"].
            lens_argmax = [int(lens[L][0]["b"]) for L in range(m.layers)]
            res_stack = torch.stack([self.cap_block_out[L][0, -1] for L in range(m.layers)]).float()
            embed_row = emb_w[nxt].float()
            conf = confidence_mod.frame_fields(
                last_logits=last_logits, probs=probs, nxt=nxt,
                lens_argmax=lens_argmax, res_stack=res_stack, embed_row=embed_row,
                vocab=int(m.vocab), weights=self._conf_weights,
            )

            # Per-frame shape matches the C-engine frame schema in app.py so the
            # dashboard renderer consumes one shape from both backends.
            ablation = ({"layer": self._ablate_layer, "neuron": self._ablate_neuron}
                        if self._ablate_layer >= 0 and self._ablate_neuron >= 0 else None)
            yield {
                "kind": "token",
                "byte": nxt, "argmax_byte": argmax_byte, "T": T,
                "fwd_ms": round(fwd_ms, 2),
                "entropy_bits": round(entropy_bits, 3),
                "surprise_bits": round(surprise_bits, 3),
                "ffn_full": ffn_full, "ffn_top": ffn_top,
                "ffn_argmax": ffn_argmax, "ffn_downsample": self.ffn_downsample,
                "saturation": saturation,
                "decisiveness": decisiveness,
                "dla_picked": dla_picked,
                "dla_argmax": dla_argmax,
                "dla_cand": dla_cand,
                "ablation": ablation,
                "margin":           conf["margin"],
                "entropy":          conf["entropy"],
                "lens_consistency": conf["lens_consistency"],
                "residual_stab":    conf["residual_stab"],
                "confidence":       conf["confidence"],
                "attn": attn, "info_flow": info_flow,
                "res": res_norms, "contrib": contributions,
                "lens": lens, "cand": candidates,
                "memory": memory,
            }

            if addons_chain is not None:
                addons_chain.observe(nxt)
            if constraint is not None:
                constraint.step(nxt)
                if constraint.done():
                    yield {"kind": "stop", "reason": "constraint complete"}
                    return
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
            if ids.size(1) >= seq:
                ids = ids[:, -(seq - 1):]

    # ----------------------------------------------------------------------
    # Fast-decode path.
    #
    # Skips the rich per-byte brain-scan telemetry so KV-cache or MTP-head
    # decoding can run at their advertised speed. Emits {kind: "meta"} once
    # and {kind: "fast_byte", byte, ms_per_byte, accepted_extra?} per byte.
    # The default stream() above is still the canonical path the dashboard
    # uses; this is opt-in via /generate?fast=kv|mtp.

    def stream_fast(self, prompt, mode="kv", temperature=0.7, top_k_sample=40,
                    max_new=200, addons_chain=None, constraint=None,
                    adaptive_threshold=0.8):
        m = self.model
        seq = m.seq
        if not prompt:
            prompt = " "
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) >= seq:
            prompt_bytes = prompt_bytes[-(seq - 1):]

        if addons_chain is not None:
            addons_chain.reset()
            addons_chain.observe_bytes(prompt_bytes)
        if constraint is not None:
            constraint.reset()
            _prime = getattr(constraint, "prime", None)
            if callable(_prime):
                _prime(prompt_bytes)
            else:
                for _b in prompt_bytes:
                    constraint.step(int(_b) & 0xff)

        has_mtp     = hasattr(m, "mtp") and hasattr(m, "n_predict") and int(getattr(m, "n_predict", 1)) > 1
        mtp_modes = ("mtp", "mtp-verify")
        valid_modes = ("kv", "mtp", "mtp-verify", "adaptive")
        if mode in mtp_modes and not has_mtp:
            yield {"kind": "error", "message": f"fast={mode} requires a model with an MTP head (Veritate800M)"}
            return
        if mode not in valid_modes:
            yield {"kind": "error", "message": f"unknown fast mode: {mode!r}. Allowed: {', '.join(valid_modes)}."}
            return

        yield {
            "kind": "meta",
            "checkpoint": self.checkpoint,
            "n_params": self.n_params,
            "layers": m.layers, "heads": m.heads, "ffn": m.ffn,
            "vocab": m.vocab, "seq": m.seq,
            "fast_mode": mode,
            "prompt": prompt,
            "prompt_bytes": list(prompt_bytes),
            "adaptive_threshold": adaptive_threshold if mode == "adaptive" else None,
        }

        if mode == "kv":
            yield from self._stream_fast_kv(prompt_bytes, temperature, top_k_sample,
                                            max_new, addons_chain, constraint)
        elif mode == "adaptive":
            yield from self._stream_fast_adaptive(prompt_bytes, temperature, top_k_sample,
                                                  max_new, addons_chain, constraint,
                                                  threshold=adaptive_threshold)
        elif mode == "mtp":
            yield from self._stream_fast_mtp(prompt_bytes, temperature, top_k_sample,
                                             max_new, addons_chain, constraint)
        elif mode == "mtp-verify":
            yield from self._stream_fast_mtp_verify(prompt_bytes, temperature, top_k_sample,
                                                    max_new, addons_chain, constraint)

    def _sample_one(self, logits_1d, temperature, top_k_sample, addons_chain, constraint):
        """Shared per-step sampling helper for stream_fast. Returns int byte or
        None to indicate the constraint forbids every byte."""
        scaled = logits_1d / max(temperature, 1e-6)
        if addons_chain is not None and len(addons_chain) > 0:
            scaled = addons_chain.bias_logits(scaled)
        if constraint is not None:
            allowed_np = constraint.mask()
            allowed = torch.from_numpy(allowed_np)
            scaled = scaled.masked_fill(~allowed, float("-inf"))
            if not torch.isfinite(scaled).any():
                return None
        sv, si = torch.topk(scaled, top_k_sample)
        gate = torch.full_like(scaled, float("-inf"))
        gate.scatter_(0, si, sv)
        sample_probs = F.softmax(gate, dim=-1)
        return int(torch.multinomial(sample_probs, 1).item())

    def _stream_fast_kv(self, prompt_bytes, temperature, top_k_sample, max_new,
                        addons_chain, constraint):
        from decode import KVCachedDecoder

        m = self.model
        ids = torch.tensor([list(prompt_bytes)], dtype=torch.long)
        dec = KVCachedDecoder(m, max_T=m.seq, B=1)
        with dec.cached():
            t0 = time.perf_counter()
            with torch.no_grad():
                last_logits = dec.prefill(ids)[0]
            prefill_ms = (time.perf_counter() - t0) * 1000
            yield {"kind": "prefill", "prefill_ms": round(prefill_ms, 2), "tokens": int(ids.size(1))}

            for _ in range(max_new):
                if dec.caches[0].length >= dec.max_T:
                    yield {"kind": "stop", "reason": "kv cache full"}
                    return
                t0 = time.perf_counter()
                nxt = self._sample_one(last_logits, temperature, top_k_sample, addons_chain, constraint)
                if nxt is None:
                    yield {"kind": "stop", "reason": "constraint allowed no bytes"}
                    return
                if addons_chain is not None:
                    addons_chain.observe(nxt)
                if constraint is not None:
                    constraint.step(nxt)
                with torch.no_grad():
                    last_logits = dec.decode_one(nxt)[0]
                step_ms = (time.perf_counter() - t0) * 1000
                yield {"kind": "fast_byte", "byte": int(nxt), "ms_per_byte": round(step_ms, 2)}
                if constraint is not None and constraint.done():
                    yield {"kind": "stop", "reason": "constraint complete"}
                    return

    def _stream_fast_mtp(self, prompt_bytes, temperature, top_k_sample, max_new,
                         addons_chain, constraint):
        from decode import MTPDecoder

        m = self.model
        dec = MTPDecoder(m, k=int(getattr(m, "n_predict", 4)))
        # MTPDecoder.decode runs to completion internally and returns text +
        # stats. We re-implement the verify loop here so we can stream bytes
        # and apply addons / constraints between heads.
        device = next(m.parameters()).device
        ctx = list(prompt_bytes)
        produced = 0
        K = dec.k
        seq_max = m.seq
        while produced < max_new:
            window = ctx[-seq_max:]
            toks = torch.tensor([window], dtype=torch.long, device=device)
            t0 = time.perf_counter()
            with torch.no_grad():
                all_logits = dec._forward_all_heads(toks)  # [B, T, N, vocab]
            last = all_logits[0, -1]  # [N, vocab]
            # Apply addons + constraint to head-0 only (the verified-canonical
            # byte). Heads 1..K-1 still run greedy from raw logits.
            head0 = last[0]
            nxt0 = self._sample_one(head0, temperature, top_k_sample, addons_chain, constraint)
            if nxt0 is None:
                yield {"kind": "stop", "reason": "constraint allowed no bytes"}
                return
            ctx.append(nxt0)
            produced += 1
            if addons_chain is not None:
                addons_chain.observe(nxt0)
            if constraint is not None:
                constraint.step(nxt0)
            step_ms = (time.perf_counter() - t0) * 1000
            yield {"kind": "fast_byte", "byte": int(nxt0), "ms_per_byte": round(step_ms, 2),
                   "head": 0, "k": K}
            if constraint is not None and constraint.done():
                yield {"kind": "stop", "reason": "constraint complete"}
                return
            # Speculative bytes from heads 1..K-1 are accepted greedily when
            # they read as valid utf-8 continuation candidates AND the
            # constraint allows them. This is the "accept_all" branch; the
            # full byte-exact verify path needs two forwards and lives in the
            # MTPDecoder._decode_verify standalone for offline use.
            extras = []
            for hi in range(1, K):
                if produced >= max_new:
                    break
                row = last[hi]
                nxt = self._sample_one(row, temperature, top_k_sample, addons_chain, constraint)
                if nxt is None:
                    break
                ctx.append(nxt)
                produced += 1
                if addons_chain is not None:
                    addons_chain.observe(nxt)
                if constraint is not None:
                    constraint.step(nxt)
                extras.append(int(nxt))
                yield {"kind": "fast_byte", "byte": int(nxt), "ms_per_byte": 0.0,
                       "head": int(hi), "k": K}
                if constraint is not None and constraint.done():
                    yield {"kind": "stop", "reason": "constraint complete"}
                    return

    def _stream_fast_mtp_verify(self, prompt_bytes, temperature, top_k_sample, max_new,
                                addons_chain, constraint):
        """Byte-exact MTP-verify (Medusa-style self-speculative). Each outer
        step does TWO forwards:
          Pass 1: forward at context. Draft K bytes from K heads at last pos.
          Pass 2: forward at context ++ drafts[:K-1]. For i=1..K-1, check if
                  the head-0 prediction at the corresponding position equals
                  the draft[i]. Accept the longest matching prefix. On
                  mismatch, append the head-0 byte (this is what head0-only
                  decode would have produced).
        Output is byte-exact to single-byte head-0 decode. Cost per step: 2
        forwards. Gain: 1..K bytes per step. Break-even at K=2 accepted.

        Addons + constraint apply to head-0 only (the verified-canonical
        path). Heads 1..K-1 sample under the same logit pipeline so their
        proposals are still constraint-aware — but the verifier rejects them
        if they don't match head-0's argmax. Greedy verification (argmax)
        is used for the head-0 reference to preserve byte-exactness.
        """
        from decode import MTPDecoder

        m = self.model
        dec = MTPDecoder(m, k=int(getattr(m, "n_predict", 4)))
        device = next(m.parameters()).device
        ctx = list(prompt_bytes)
        K = dec.k
        seq_max = m.seq
        produced = 0
        n_proposed = 0
        n_accepted_extra = 0

        while produced < max_new:
            # ---- pass 1: draft K bytes ------------------------------------
            window = ctx[-seq_max:]
            toks = torch.tensor([window], dtype=torch.long, device=device)
            t0 = time.perf_counter()
            with torch.no_grad():
                p1_logits = dec._forward_all_heads(toks)  # [1, T, N, V]
            last = p1_logits[0, -1]                        # [N, V]

            # Sample head-0 under the addons+constraint pipeline. This is the
            # only byte we COMMIT from pass 1 (the rest are speculative).
            head0_logits = last[0]
            nxt0 = self._sample_one(head0_logits, temperature, top_k_sample,
                                    addons_chain, constraint)
            if nxt0 is None:
                yield {"kind": "stop", "reason": "constraint allowed no bytes"}
                return

            # Draft the next K-1 bytes from heads 1..K-1, also under the
            # constraint+addons pipeline. These are SPECULATIVE — they're
            # only committed if pass-2 head-0 agrees.
            drafts = [nxt0]
            if K > 1:
                # We need to NOT mutate the constraint while sampling drafts —
                # those bytes might not be accepted. Snapshot + restore.
                snap = _snapshot_constraint(constraint)
                # Temporarily step constraint forward by nxt0 so subsequent
                # sampling sees the right state.
                if constraint is not None:
                    constraint.step(nxt0)
                try:
                    for hi in range(1, K):
                        row = last[hi]
                        nxt = self._sample_one(row, temperature, top_k_sample,
                                               addons_chain, constraint)
                        if nxt is None:
                            break
                        drafts.append(int(nxt))
                        if constraint is not None:
                            constraint.step(nxt)
                finally:
                    _restore_constraint(constraint, snap)

            # ---- pass 2: verify drafts 1..K-1 -----------------------------
            verify_preds = []
            if len(drafts) > 1:
                verify_window = (window + drafts[:-1])[-seq_max:]
                v_toks = torch.tensor([verify_window], dtype=torch.long, device=device)
                with torch.no_grad():
                    p2_logits = dec._forward_all_heads(v_toks)
                T_v = p2_logits.size(1)
                # The verifier is GREEDY (argmax) over head-0 at positions
                # corresponding to "byte AFTER observing draft d_0..d_{i-1}".
                # That's the last len(drafts)-1 positions of pass-2.
                for i in range(1, len(drafts)):
                    pos = T_v - len(drafts) + i
                    if pos < 0 or pos >= T_v:
                        break
                    ref = int(p2_logits[0, pos, 0].argmax().item())
                    verify_preds.append(ref)

            # ---- accept longest matching prefix ---------------------------
            accepted = [drafts[0]]      # head-0 sample always accepted
            mismatch_at = None
            for i, vp in enumerate(verify_preds, start=1):
                if i < len(drafts) and drafts[i] == vp:
                    accepted.append(drafts[i])
                else:
                    mismatch_at = i
                    # Append the head-0 reference at the mismatch (free byte:
                    # we already computed it). This preserves byte-exactness.
                    accepted.append(vp)
                    break

            # Track speculation stats. "extra" = bytes accepted beyond byte-0.
            n_proposed += max(0, K - 1)
            extra_now = 0
            if mismatch_at is None:
                extra_now = max(0, len(accepted) - 1)   # all drafts matched
            else:
                extra_now = max(0, mismatch_at - 1)     # matched up to mismatch
            n_accepted_extra += extra_now

            step_ms = (time.perf_counter() - t0) * 1000
            ms_each = step_ms / max(1, len(accepted))

            # ---- emit ------------------------------------------------------
            for idx, b in enumerate(accepted):
                if produced >= max_new:
                    break
                ctx.append(int(b))
                produced += 1
                if addons_chain is not None:
                    addons_chain.observe(int(b))
                if constraint is not None:
                    constraint.step(int(b))
                yield {
                    "kind": "fast_byte", "byte": int(b),
                    "ms_per_byte": round(ms_each, 2),
                    "head": int(idx), "k": K,
                    "accepted_extra_so_far": int(n_accepted_extra),
                    "acceptance_rate": round(n_accepted_extra / max(1, n_proposed), 3),
                }
                if constraint is not None and constraint.done():
                    yield {"kind": "stop", "reason": "constraint complete"}
                    return

    def _stream_fast_adaptive(self, prompt_bytes, temperature, top_k_sample, max_new,
                              addons_chain, constraint, threshold=0.8):
        """Per-position adaptive depth (LayerSkip-style). For each byte:
          - Walk blocks one at a time. After each block, project the residual
            through the FINAL RMSNorm + tied LM head (logit-lens).
          - Compute top-1 probability over vocab.
          - If top-1 >= threshold, EXIT EARLY. Use the logit-lens-projected
            logits to sample the next byte.
          - If we walk all blocks without crossing the threshold, use the
            final-layer output (canonical forward).

        S44 measured ~32% mean compute savings on the 85M with this policy.
        This is NOT byte-exact vs full-depth decode — the early-exit logits
        come from an intermediate residual that wasn't trained to produce
        the final distribution. Quality regression depends on threshold.
        Use mode='kv' for byte-exact decode; this mode trades quality for
        speed.

        Emits {kind: 'fast_byte', byte, layers_used, top1_prob, ms_per_byte}.
        """
        m = self.model
        seq = m.seq

        # Model-agnostic output projection (preflight rule 11a). The model
        # knows what it is; this code does not. byte0_projector() returns a
        # callable that maps residual -> byte-0 logits regardless of variant.
        from decode import byte0_projector
        project = byte0_projector(m)

        # Cache whether the model exposes its own embed() / has a RoPE
        # cache. We don't dispatch by variant here — we ask the model what
        # surface it provides.
        has_embed       = hasattr(m, "embed") and callable(m.embed)
        has_rope_cache  = hasattr(m, "rope_cos") and hasattr(m, "rope_sin")
        has_extend_rope = hasattr(m, "extend_rope") and callable(m.extend_rope)

        def _embed_input(ids_tensor):
            """Use model.embed() if it exists. Otherwise build a canonical
            tok_emb + pos_emb input. (RoPE models register rope_cos/sin as
            buffers; the model's own embed() handles them.)"""
            if has_embed:
                return m.embed(ids_tensor)
            T = ids_tensor.size(1)
            positions = torch.arange(T, device=ids_tensor.device).unsqueeze(0)
            return m.tok_emb(ids_tensor) + m.pos_emb(positions)

        def _run_block(block, x):
            """Some block signatures take extra args (RoPE cos/sin). Try the
            plain call first; fall back to the RoPE signature if needed.
            Both signatures live in the model class, not the consumer."""
            try:
                return block(x)
            except TypeError:
                if has_rope_cache:
                    return block(x, m.rope_cos, m.rope_sin)
                raise

        ids_full = list(prompt_bytes)
        produced = 0
        layers_used_hist = []

        while produced < max_new:
            window = ids_full[-seq:]
            ids = torch.tensor([window], dtype=torch.long)
            t0 = time.perf_counter()

            # Embed + per-block walk with early exit
            with torch.no_grad():
                # Extend RoPE cache if the model has one and we exceed it
                if has_rope_cache and has_extend_rope:
                    if ids.size(1) > m.rope_cos.size(0):
                        m.extend_rope(ids.size(1))
                x = _embed_input(ids)

                exit_layer = m.layers
                final_logits = None
                for L, blk in enumerate(m.blocks):
                    x = _run_block(blk, x)
                    # Check confidence at this layer. Project the last
                    # position only — that's the next-byte prediction.
                    last_residual = x[:, -1:, :]
                    lens_logits = project(last_residual)[0, 0]
                    probs = F.softmax(lens_logits, dim=-1)
                    top1 = float(probs.max())
                    if top1 >= threshold and L + 1 < m.layers:
                        exit_layer = L + 1
                        final_logits = lens_logits
                        break
                else:
                    # Walked all layers; use final output
                    final_logits = project(x[:, -1:, :])[0, 0]

            step_ms = (time.perf_counter() - t0) * 1000
            layers_used_hist.append(exit_layer)

            nxt = self._sample_one(final_logits, temperature, top_k_sample,
                                   addons_chain, constraint)
            if nxt is None:
                yield {"kind": "stop", "reason": "constraint allowed no bytes"}
                return

            ids_full.append(nxt)
            produced += 1
            if addons_chain is not None:
                addons_chain.observe(nxt)
            if constraint is not None:
                constraint.step(nxt)

            # Running savings stat
            avg_layers = sum(layers_used_hist) / len(layers_used_hist)
            compute_saved = (1.0 - avg_layers / m.layers)

            yield {
                "kind": "fast_byte", "byte": int(nxt),
                "ms_per_byte": round(step_ms, 2),
                "layers_used": int(exit_layer),
                "total_layers": int(m.layers),
                "compute_saved_so_far": round(compute_saved, 3),
                "threshold": float(threshold),
            }
            if constraint is not None and constraint.done():
                yield {"kind": "stop", "reason": "constraint complete"}
                return


# ------------------------------------------------------------------------------------
# Constraint snapshot helpers (used by stream_fast mtp-verify)
#
# Speculative decoding samples K candidate bytes per outer step. The constraint
# must see them while sampling (so masks update correctly), but only the
# ACCEPTED prefix should leave a permanent mark. Verifier-rejected bytes are
# unwound via snapshot/restore on the constraint's __dict__. All shipped
# constraint classes (JSON / Vocab / StopOn / Combine) store plain Python state.

def _snapshot_constraint(c):
    if c is None:
        return None
    snap = {}
    for k, v in c.__dict__.items():
        if isinstance(v, list):
            snap[k] = list(v)
        elif isinstance(v, dict):
            snap[k] = dict(v)
        elif isinstance(v, bytearray):
            snap[k] = bytearray(v)
        else:
            snap[k] = v
    return snap


def _restore_constraint(c, snap):
    if c is None or snap is None:
        return
    for k, v in snap.items():
        cur = getattr(c, k, None)
        if isinstance(cur, list):
            cur.clear(); cur.extend(v)
        elif isinstance(cur, dict):
            cur.clear(); cur.update(v)
        elif isinstance(cur, bytearray):
            cur.clear(); cur.extend(v)
        else:
            setattr(c, k, v)
