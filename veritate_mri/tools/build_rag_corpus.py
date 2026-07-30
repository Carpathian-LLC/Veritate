# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Builds the context-grounded (RAG) SFT corpus: each example puts a fact in a
#   context block, asks a question, and answers from the context, so the model
#   learns to use retrieved text instead of recalling weights.
# - Facts and question/answer pairs come from the configured Teacher Model
#   (Settings > Teacher Model). Facts split train/test, so the held-out set
#   measures in-context copy on facts the model never trained on.
# - Output: trainers/corpus/<stem>_{train,val}.bin plus the held-out set at
#   veritate_mri/data/eval/rag/<stem>_test.json.
# - Spawned as a subprocess by veritate_mri/routes/rag_routes.py; stdout is the
#   job log the RAG panel tails.
# veritate_mri/tools/build_rag_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import sys

_HERE     = os.path.dirname(os.path.abspath(__file__))
_MRI_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_REPO     = os.path.normpath(os.path.join(_MRI_ROOT, ".."))
for _p in (_REPO, _MRI_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from readers import paths  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

DEFAULT_STEM    = "grounded_v1"
DEFAULT_FACTS   = 220
QA_PER_FACT     = 2
TEST_FRAC       = 0.2
VAL_FRAC        = 0.05
CHAT_MAX_TOKENS = 1024
FACT_TEMP       = 0.7
QA_TEMP         = 0.5
MIN_FACT_CHARS  = 10
PROGRESS_EVERY  = 20

# ChatML, per documentation.md (corpus framing). Retrieved text rides a
# system turn; the "context: " lead-in is kept because inference stops on it.
# EVAL_PREFIX ends after the user turn so the model emits the assistant turn itself.
IM_START    = "<|im_start|>"
IM_END      = "<|im_end|>"
EOT         = "<|endoftext|>"
TEMPLATE    = (f"{IM_START}system\ncontext: {{fact}}{IM_END}\n"
               f"{IM_START}user\n{{q}}{IM_END}\n"
               f"{IM_START}assistant\n{{a}}{IM_END}\n{EOT}\n")
EVAL_PREFIX = (f"{IM_START}system\ncontext: {{fact}}{IM_END}\n"
               f"{IM_START}user\n{{q}}{IM_END}\n"
               f"{IM_START}assistant\n")
JSON_ONLY   = "You output JSON only."
FACT_CATS   = ["geography", "science", "history", "sports", "art and culture",
               "technology", "nature", "space"]
NO_TEACHER  = "no teacher configured; set one in Settings > Teacher Model"

_CLIENT = []

# ------------------------------------------------------------------------------------
# Functions

def _chat(system, user, temperature):
    if not _CLIENT:
        from veritate_core.plugin import get_teacher_client
        client = get_teacher_client()
        if client is None:
            raise SystemExit(NO_TEACHER)
        _CLIENT.append(client)
    return _CLIENT[0].complete([{"role": "user", "content": user}], temperature=temperature,
                               max_tokens=CHAT_MAX_TOKENS, system=system)


def _first_json_object(text):
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def gen_facts(n):
    """Short self-contained facts from the teacher, spread evenly over FACT_CATS."""
    out = []
    per = max(1, n // len(FACT_CATS))
    for cat in FACT_CATS:
        user = (f"List {per} short, true, self-contained factual statements about {cat}. "
                "Each one sentence, with a clear single answer. "
                'Reply as JSON: {"facts": ["...", "..."]}.')
        try:
            facts = _first_json_object(_chat(JSON_ONLY, user, FACT_TEMP)).get("facts", [])
        except (ValueError, KeyError, OSError) as e:
            print(f"[rag] facts {cat} failed: {type(e).__name__}: {e}", flush=True)
            continue
        out.extend(f.strip() for f in facts if isinstance(f, str) and len(f.strip()) > MIN_FACT_CHARS)
    return out


def gen_qa(fact):
    user = (f'Given this fact: "{fact}"\n'
            "Write one natural question whose answer is contained in the fact, and the short answer "
            '(a few words, taken from the fact). Reply as JSON: {"q": "...", "a": "..."}.')
    obj = _first_json_object(_chat(JSON_ONLY, user, QA_TEMP))
    q, a = obj.get("q", "").strip(), obj.get("a", "").strip()
    if not q or not a:
        raise ValueError("teacher returned an empty question or answer")
    return q, a


def _write_bins(stem, blob):
    cut = int(len(blob) * (1 - VAL_FRAC))
    os.makedirs(paths.corpus_dir(), exist_ok=True)
    with open(paths.corpus_train_path(stem), "wb") as f:
        f.write(blob[:cut])
    with open(paths.corpus_val_path(stem), "wb") as f:
        f.write(blob[cut:])


def _write_test_set(stem, items, n_train_examples, n_train_facts):
    os.makedirs(paths.RAG_EVAL_ROOT, exist_ok=True)
    with open(paths.rag_eval_path(stem), "w", encoding="utf-8") as f:
        json.dump({"test": items, "n_train_examples": n_train_examples,
                   "n_train_facts": n_train_facts, "n_test_facts": len(items)}, f, indent=2)


def build(n_facts, qa_per_fact, stem):
    """Generate the corpus and write both bins plus the held-out test set."""
    print(f"[rag] generating ~{n_facts} facts via the configured teacher", flush=True)
    facts = gen_facts(n_facts)
    print(f"[rag] got {len(facts)} facts", flush=True)
    if not facts:
        raise SystemExit("teacher returned no usable facts")

    n_test = max(1, int(len(facts) * TEST_FRAC))
    test_facts, train_facts = facts[:n_test], facts[n_test:]

    train_text, test_items = [], []
    for i, fact in enumerate(train_facts):
        for _ in range(qa_per_fact):
            try:
                q, a = gen_qa(fact)
            except (ValueError, KeyError, OSError):
                continue
            train_text.append(TEMPLATE.format(fact=fact, q=q, a=a))
        if (i + 1) % PROGRESS_EVERY == 0:
            print(f"[rag] train qa {i + 1}/{len(train_facts)}", flush=True)
    for fact in test_facts:
        try:
            q, a = gen_qa(fact)
        except (ValueError, KeyError, OSError):
            continue
        test_items.append({"fact": fact, "q": q, "a": a,
                           "prompt": EVAL_PREFIX.format(fact=fact, q=q)})

    blob = "".join(train_text).encode("utf-8", "replace")
    _write_bins(stem, blob)
    _write_test_set(stem, test_items, len(train_text), len(train_facts))
    print(f"[rag] train examples={len(train_text)} bytes={len(blob)} "
          f"test_facts={len(test_items)} -> {stem}_train.bin + {os.path.basename(paths.rag_eval_path(stem))}",
          flush=True)
    return {"stem": stem, "n_train_examples": len(train_text), "n_bytes": len(blob),
            "n_test_facts": len(test_items)}


def main():
    ap = argparse.ArgumentParser(description="Build the context-grounded (RAG) SFT corpus.")
    ap.add_argument("--n_facts",     type=int, default=DEFAULT_FACTS)
    ap.add_argument("--qa_per_fact", type=int, default=QA_PER_FACT)
    ap.add_argument("--stem",        type=str, default=DEFAULT_STEM)
    args = ap.parse_args()
    build(args.n_facts, args.qa_per_fact, args.stem)


if __name__ == "__main__":
    main()
