# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Hallucination-detection engine. Pure functions over an already generated
#   answer plus its per-byte confidence stream (the calibrated field
#   confidence.py computes and both backends emit). No decode loop lives here:
#   the route drives generation and hands this module the assembled bytes.
# - Four signals: span confidence aggregation (paragraph/sentence/word, byte
#   offsets exact), grounded overlap against retrieved context, context
#   divergence (answer with vs without context), and a training-data provenance
#   PROXY. The proxy is nearest-neighbour BM25 similarity over a bounded sample
#   of trainers/corpus, NOT proven causal attribution.
# veritate_mri/inference/hallucination.py
# ------------------------------------------------------------------------------------
# Imports:

import os
import re

from inference.agent.tools.retriever import BM25Index, _split_chunks, _tokenize

# ------------------------------------------------------------------------------------
# Constants

ROUND_DIGITS = 4

# Calibration knobs. UNCERTAIN_CONFIDENCE_THRESHOLD is the "I don't know" line:
# an overall byte-confidence below it flags the answer as uncertain. The grounding
# + divergence thresholds gate the verdict. Tune against a labelled set.
UNCERTAIN_CONFIDENCE_THRESHOLD = 0.55
GROUNDED_HIGH_FRACTION         = 0.85  # calibration knob: gf at/above this is grounded; below it, with context, at best partially_grounded
PARTIAL_GROUNDED_LOW_FRACTION  = 0.15  # calibration knob: gf at/below this has no meaningful context overlap (not even partial)
DIVERGENCE_HALLUCINATION       = 0.8   # calibration knob: divergence at/above this on an under-grounded answer -> likely_hallucinated

RISK_WEIGHTS = {"ungrounded": 0.4, "unconfident": 0.3, "divergence": 0.3}

# Derived-confidence fallback: bits of surprise at which confidence hits 0. A
# byte vocab is 256 symbols, so full surprise is log2(256) = 8 bits.
SURPRISE_BITS_FULL       = 8.0
CONFIDENCE_SOURCE_BYTE    = "byte_confidence"
CONFIDENCE_SOURCE_DERIVED = "entropy_surprise"

REFUSAL_MARKERS = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "can't find that", "cannot find that", "not in the provided context",
    "no information about", "i don't have information", "unable to answer",
)

# Training-provenance proxy caps. Kept small so index build stays cheap; the
# sample is a slice, not the full corpus.
TRAINING_SAMPLE_MAX_BYTES      = 4 * 1024 * 1024
TRAINING_SAMPLE_PER_FILE_BYTES = 512 * 1024
TRAINING_PROBE_BYTES           = 4096
TEXT_PRINTABLE_MIN             = 0.85
TRAIN_BIN_SUFFIX               = "_train.bin"
TRAIN_CHUNK_BYTES              = 1024
TRAIN_CHUNK_OVERLAP            = 128
TRAINING_TOP_SPANS             = 3
TRAINING_PASSAGES_K            = 3
PASSAGE_PREVIEW_CHARS          = 240

GROUNDED_SOURCES_MAX = 12

PARAGRAPH_SEP_RE = re.compile(r"\n[ \t]*\n")
SENTENCE_SEP_RE  = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_SEP_RE      = re.compile(r"\s+")

_WS = " \t\r\n\f\v"
_TEXT_BYTES = frozenset({0x09, 0x0a, 0x0d}) | frozenset(range(0x20, 0x7f))

# ------------------------------------------------------------------------------------
# Functions

_content_words = _tokenize


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _min(xs):
    return min(xs) if xs else 0.0


def _r(x):
    return round(x, ROUND_DIGITS)


def derive_confidence(surprise_bits):
    """Confidence proxy from surprise when a backend frame carries no calibrated
    confidence. Low surprise -> high confidence."""
    if surprise_bits is None:
        return 0.5
    return max(0.0, min(1.0, 1.0 - surprise_bits / SURPRISE_BITS_FULL))


def _char_byte_offsets(text):
    offs = []
    acc = 0
    for ch in text:
        offs.append(acc)
        acc += len(ch.encode("utf-8"))
    offs.append(acc)
    return offs


def _segments(sub, sep_re):
    """Whitespace-trimmed non-empty (start, end) char ranges of `sub` split on
    sep_re. Offsets are relative to `sub`."""
    res = []
    pos = 0

    def add(a, b):
        while a < b and sub[a] in _WS:
            a += 1
        while b > a and sub[b - 1] in _WS:
            b -= 1
        if b > a:
            res.append((a, b))

    for m in sep_re.finditer(sub):
        add(pos, m.start())
        pos = m.end()
    add(pos, len(sub))
    return res


def _reconcile(metrics, n_bytes):
    """Pad or clip per-byte metrics to the answer's utf-8 length. Only fires when
    decode('replace') changed the byte count (invalid utf-8 at a truncation)."""
    if len(metrics) == n_bytes:
        return metrics
    if len(metrics) > n_bytes:
        return metrics[:n_bytes]
    fill = {"confidence": _mean([m["confidence"] for m in metrics]),
            "surprise_bits": 0.0, "entropy_bits": 0.0}
    return metrics + [fill] * (n_bytes - len(metrics))


def segment_spans(answer, metrics):
    """Nested paragraph -> sentence -> word spans with byte-exact offsets and
    confidence rolled up from the per-byte stream. Returns (spans, overall
    confidence). grounded fields start None; annotate_grounding fills them."""
    abytes = answer.encode("utf-8")
    metrics = _reconcile(metrics, len(abytes))
    coff = _char_byte_offsets(answer)

    def byte_conf(b0, b1):
        return [m["confidence"] for m in metrics[b0:b1]]

    paragraphs = []
    for (pc0, pc1) in _segments(answer, PARAGRAPH_SEP_RE):
        sub_p = answer[pc0:pc1]
        sentences = []
        for (rsc0, rsc1) in _segments(sub_p, SENTENCE_SEP_RE):
            sc0, sc1 = pc0 + rsc0, pc0 + rsc1
            sub_s = answer[sc0:sc1]
            words = []
            for (rwc0, rwc1) in _segments(sub_s, WORD_SEP_RE):
                wc0, wc1 = sc0 + rwc0, sc0 + rwc1
                wb0, wb1 = coff[wc0], coff[wc1]
                seg = metrics[wb0:wb1]
                words.append({
                    "text": answer[wc0:wc1], "start": wb0, "end": wb1,
                    "confidence":    _r(_mean([m["confidence"] for m in seg])),
                    "surprise_bits": _r(_mean([m["surprise_bits"] for m in seg])),
                    "entropy_bits":  _r(_mean([m["entropy_bits"] for m in seg])),
                    "grounded": None,
                })
            sb0, sb1 = coff[sc0], coff[sc1]
            s_conf = _mean([w["confidence"] for w in words]) if words \
                else _mean(byte_conf(sb0, sb1))
            sentences.append({
                "text": sub_s, "start": sb0, "end": sb1,
                "confidence": _r(s_conf), "confidence_min": _r(_min(byte_conf(sb0, sb1))),
                "grounded": None, "words": words,
            })
        pb0, pb1 = coff[pc0], coff[pc1]
        p_conf = _mean([s["confidence"] for s in sentences]) if sentences \
            else _mean(byte_conf(pb0, pb1))
        paragraphs.append({
            "text": sub_p, "start": pb0, "end": pb1,
            "confidence": _r(p_conf), "confidence_min": _r(_min(byte_conf(pb0, pb1))),
            "grounded": None, "sentences": sentences,
        })

    overall = _mean([m["confidence"] for m in metrics])
    return {"paragraphs": paragraphs}, overall


def _grounded_label(words, ctx):
    if not words:
        return None
    hits = sum(1 for w in words if w in ctx)
    if hits == 0:
        return "no"
    if hits == len(words):
        return "yes"
    return "partial"


def annotate_grounding(spans, context_text):
    """Mark each word/sentence/paragraph grounded against the concatenated
    context. Returns grounded_fraction (answer content-words found in context),
    or None when there is no context."""
    if context_text is None:
        return None
    ctx = set(_content_words(context_text))
    total, matched = 0, 0
    for p in spans["paragraphs"]:
        p_words = []
        for s in p["sentences"]:
            s_words = []
            for w in s["words"]:
                cw = _content_words(w["text"])
                w["grounded"] = _grounded_label(cw, ctx)
                for tok in cw:
                    total += 1
                    if tok in ctx:
                        matched += 1
                s_words.extend(cw)
            s["grounded"] = _grounded_label(s_words, ctx)
            p_words.extend(s_words)
        p["grounded"] = _grounded_label(p_words, ctx)
    return (matched / total) if total else 0.0


def grounded_sources(spans, context_chunks, cap=GROUNDED_SOURCES_MAX):
    """For each grounded sentence, the retrieved chunk it best overlaps. Exact
    lexical match, so this is a real source pointer (unlike training_matches)."""
    chunk_tokens = [set(_content_words(c["text"])) for c in context_chunks]
    out = []
    for p in spans["paragraphs"]:
        for s in p["sentences"]:
            if s["grounded"] not in ("yes", "partial"):
                continue
            cw = _content_words(s["text"])
            if not cw:
                continue
            best_i, best_frac = -1, 0.0
            for i, toks in enumerate(chunk_tokens):
                frac = sum(1 for w in cw if w in toks) / len(cw)
                if frac > best_frac:
                    best_frac, best_i = frac, i
            if best_i < 0:
                continue
            out.append({"span": s["text"], "source_chunk": context_chunks[best_i]["text"],
                        "score": _r(best_frac)})
            if len(out) >= cap:
                return out
    return out


def divergence_score(answer_with, answer_without):
    """0-1 divergence between the context-grounded answer and the no-context
    answer, as 1 - Jaccard over content-word sets. High divergence means the
    context materially changed the answer."""
    sa, sb = set(_content_words(answer_with)), set(_content_words(answer_without))
    if not sa and not sb:
        return 0.0
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return _r(1.0 - len(sa & sb) / union)


def is_refusal(answer):
    norm = answer.strip().lower()
    return bool(norm) and any(m in norm for m in REFUSAL_MARKERS)


def compute_risk(confidence, grounded_fraction, divergence):
    terms = [(1.0 - confidence, RISK_WEIGHTS["unconfident"])]
    if grounded_fraction is not None:
        terms.append((1.0 - grounded_fraction, RISK_WEIGHTS["ungrounded"]))
    if divergence is not None:
        terms.append((divergence, RISK_WEIGHTS["divergence"]))
    wsum = sum(w for _, w in terms)
    risk = sum(v * w for v, w in terms) / wsum if wsum else 0.0
    return _r(max(0.0, min(1.0, risk)))


def pick_verdict(refused, uncertain, grounded_fraction, divergence, has_context):
    if refused:
        return "refused"
    if has_context and grounded_fraction is not None:
        if grounded_fraction >= GROUNDED_HIGH_FRACTION:
            return "grounded"
        # Not almost-fully grounded. A large divergence from the no-context answer
        # means the mixed-in unsupported tokens are a fabrication, not a paraphrase
        # of context: escalate whether gf is 0.0 or 0.5.
        if divergence is not None and divergence >= DIVERGENCE_HALLUCINATION:
            return "likely_hallucinated"
        if grounded_fraction > PARTIAL_GROUNDED_LOW_FRACTION:
            return "partially_grounded"
    if uncertain:
        return "low_confidence"
    return "ungrounded_ok"


def build_overall(overall_confidence, grounded_fraction, divergence, answer,
                  threshold=UNCERTAIN_CONFIDENCE_THRESHOLD):
    refused = is_refusal(answer)
    uncertain = overall_confidence < threshold
    has_context = grounded_fraction is not None
    return {
        "confidence": _r(overall_confidence),
        "hallucination_risk": compute_risk(overall_confidence, grounded_fraction, divergence),
        "grounded_fraction": (_r(grounded_fraction) if grounded_fraction is not None else None),
        "context_divergence": (_r(divergence) if divergence is not None else None),
        "uncertain": bool(uncertain),
        "verdict": pick_verdict(refused, uncertain, grounded_fraction, divergence, has_context),
    }


def _looks_texty(path):
    with open(path, "rb") as fh:
        probe = fh.read(TRAINING_PROBE_BYTES)
    if not probe:
        return False
    return sum(1 for b in probe if b in _TEXT_BYTES) / len(probe) >= TEXT_PRINTABLE_MIN


def build_training_index(corpus_dir, cap_bytes=TRAINING_SAMPLE_MAX_BYTES,
                         per_file=TRAINING_SAMPLE_PER_FILE_BYTES):
    """BM25 index over a bounded sample of the training corpora. Only text-like
    *_train.bin files are sampled (a printable-byte probe skips numeric corpora),
    capped per file and in total. Returns (index_or_None, chunk_meta, report)."""
    chunks, meta, files, total = [], [], [], 0
    if not os.path.isdir(corpus_dir):
        return None, [], {"n_files": 0, "files": [], "chunks": 0, "bytes": 0, "cap_bytes": cap_bytes}
    for fn in sorted(os.listdir(corpus_dir)):
        if total >= cap_bytes:
            break
        if not fn.endswith(TRAIN_BIN_SUFFIX):
            continue
        path = os.path.join(corpus_dir, fn)
        if not os.path.isfile(path) or not _looks_texty(path):
            continue
        with open(path, "rb") as fh:
            data = fh.read(min(per_file, cap_bytes - total))
        text = data.decode("utf-8", "replace")
        n_added = 0
        for _off, ch in _split_chunks(text, TRAIN_CHUNK_BYTES, TRAIN_CHUNK_OVERLAP):
            if ch:
                chunks.append((len(chunks), ch))
                meta.append({"corpus": fn, "text": ch})
                n_added += 1
        total += len(data)
        files.append({"corpus": fn, "bytes": len(data), "chunks": n_added})
    index = BM25Index(chunks) if chunks else None
    report = {"n_files": len(files), "files": files, "chunks": len(chunks),
              "bytes": total, "cap_bytes": cap_bytes}
    return index, meta, report


def _weakest_sentences(spans, n):
    sents = [s for p in spans["paragraphs"] for s in p["sentences"]]

    def rank(s):
        g = s.get("grounded")
        order = 0 if g == "no" else (1 if g == "partial" else 2)
        return (order, s["confidence_min"])

    sents.sort(key=rank)
    seen, out = set(), []
    for s in sents:
        t = s["text"].strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) >= n:
                break
    return out


def training_matches(spans, index, chunk_meta, top_spans=TRAINING_TOP_SPANS,
                     k=TRAINING_PASSAGES_K):
    """Nearest training passages for the weakest spans. PROXY ONLY: BM25
    similarity, not proven causal attribution of what the model learned from."""
    if index is None:
        return []
    out = []
    for text in _weakest_sentences(spans, top_spans):
        passages = []
        for score, did in index.search(text, k):
            m = chunk_meta[did]
            passages.append({"text": m["text"][:PASSAGE_PREVIEW_CHARS],
                             "corpus": m["corpus"], "score": _r(float(score))})
        if passages:
            out.append({"span": text, "passages": passages})
    return out
