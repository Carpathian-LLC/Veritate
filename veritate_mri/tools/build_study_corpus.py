# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - the general study-form generator of the tell-it-once loop (IDEA 20 E4). E4 proved
#   the mechanism is varied surface forms with the CONTENT IN THE ASSISTANT TURN,
#   drilled to convergence (successes.md, 45/50 closed-book). build_fact_sft renders
#   that for atomic {subj,obj} facts; this renders it for arbitrary documents, so code
#   and long-form prose consolidate through the same path. A fact is just a small chunk.
# - no relation schema, no lexicon, no NLP: structure-aware chunking (python AST, C
#   brace matching, markdown headings, prose paragraphs) plus mechanical form
#   generation. Works on any bytes.
# - forms: recite (label -> body), continue (head -> tail), infill (pre/post -> gap,
#   which for code IS fill-in-the-middle), identify (body -> label, the reverse
#   direction that beats the reversal curse). Content is always the assistant turn so
#   loss_mask=assistant trains on the material rather than on the model's own replies.
# - usage: python -m tools.build_study_corpus <path> [path...] [--stem study]
#          [--per-chunk 8] [--max-chunk 1200] [--seed 0]
# veritate_mri/tools/build_study_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import ast
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from readers.paths import CORPUS_ROOT  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

IM_S, IM_E = "<|im_start|>", "<|im_end|>"
VAL_EVERY = 20
PER_CHUNK_DEFAULT = 8
MAX_CHUNK_DEFAULT = 1200          # bytes of content per exchange; the trainer's sample
MIN_CHUNK = 40                    # window is seq*n_chunks, so a chunk must fit inside it
GAP_FRAC = 0.34                   # share of a chunk removed to make an infill target

C_FUNC_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_ \t\*]*?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{]*?\)\s*\{",
                       re.M)
MD_HEAD_RX = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
CODE_EXT = {".py", ".c", ".h", ".js", ".ts", ".go", ".rs", ".java", ".sh"}

# ------------------------------------------------------------------------------------
# Functions


def _split_oversize(label, text, max_chunk):
    """Yield (label, text) pieces no larger than max_chunk, split on line boundaries so
    a code chunk never breaks mid-token."""
    if len(text) <= max_chunk:
        yield label, text
        return
    lines, buf, part = text.splitlines(keepends=True), [], 1
    for line in lines:
        if buf and sum(len(x) for x in buf) + len(line) > max_chunk:
            yield f"{label} (part {part})", "".join(buf)
            buf, part = [], part + 1
        buf.append(line)
    if buf:
        yield f"{label} (part {part})", "".join(buf)


def chunk_python(src, path):
    """Top-level functions and classes with their source, by AST. Falls back to prose
    chunking when the file does not parse, so a syntax error costs one file's structure
    rather than the whole build."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return chunk_prose(src, path)
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        seg = ast.get_source_segment(src, node)
        if seg and len(seg) >= MIN_CHUNK:
            out.append((f"{os.path.basename(path)}::{node.name}", seg))
    return out or chunk_prose(src, path)


def chunk_c(src, path):
    """Function definitions by brace matching from each header match."""
    out = []
    for m in C_FUNC_RX.finditer(src):
        depth, i = 0, src.index("{", m.start())
        end = None
        for j in range(i, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is None:
            continue
        seg = src[m.start():end]
        if len(seg) >= MIN_CHUNK:
            out.append((f"{os.path.basename(path)}::{m.group(1)}", seg))
    return out or chunk_prose(src, path)


def chunk_markdown(src, path):
    """Heading sections: each heading owns the text up to the next heading."""
    heads = list(MD_HEAD_RX.finditer(src))
    if not heads:
        return chunk_prose(src, path)
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(src)
        seg = src[m.start():end].strip()
        if len(seg) >= MIN_CHUNK:
            out.append((f"{os.path.basename(path)}: {m.group(2)}", seg))
    return out


def chunk_prose(src, path):
    """Paragraph blocks, the schema-free fallback for anything unstructured."""
    out, base = [], os.path.basename(path)
    for i, para in enumerate(p.strip() for p in re.split(r"\n\s*\n", src)):
        if len(para) >= MIN_CHUNK:
            out.append((f"{base} (¶{i + 1})", para))
    return out


def chunk_document(text, path, max_chunk=MAX_CHUNK_DEFAULT):
    """Dispatch on extension, then enforce the size cap."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        raw = chunk_python(text, path)
    elif ext in (".c", ".h"):
        raw = chunk_c(text, path)
    elif ext in (".md", ".markdown"):
        raw = chunk_markdown(text, path)
    else:
        raw = chunk_prose(text, path)
    return [p for label, seg in raw for p in _split_oversize(label, seg, max_chunk)]


def _ex(user, assistant):
    return f"{IM_S}user\n{user}{IM_E}\n{IM_S}assistant\n{assistant}{IM_E}\n"


def study_forms(label, text, rng, per_chunk=PER_CHUNK_DEFAULT):
    """Mechanical study forms for one chunk. The chunk's own bytes are the assistant
    turn in every form except identify, whose target is the label: that is the reverse
    direction, and E4 showed both directions are what bind."""
    outs = [
        _ex(f"Show me {label}.", text),
        _ex(f"What is {label}?", text),
        _ex(f"Reproduce {label} exactly.", text),
    ]
    half = max(1, len(text) // 2)
    head, tail = text[:half], text[half:]
    if len(tail) >= 8:
        outs.append(_ex(f"Continue {label} from here:\n{head}", tail))
    gap = max(8, int(len(text) * GAP_FRAC))
    if len(text) > gap + 16:
        start = rng.randrange(8, len(text) - gap)
        pre, mid, post = text[:start], text[start:start + gap], text[start + gap:]
        outs.append(_ex(f"Fill in the missing section of {label}:\n{pre}<<<gap>>>{post}", mid))
    excerpt = text[:200]
    outs.append(_ex(f"Where is this from?\n{excerpt}", label))
    outs.append(_ex(f"Name the source of this:\n{excerpt}", label))
    rng.shuffle(outs)
    while len(outs) < per_chunk:
        outs.append(rng.choice(outs[:]))
    return outs[:per_chunk]


def iter_files(paths, exts=None):
    """Walk paths, yielding readable text files. Directories recurse."""
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for root, dirs, names in os.walk(p):
            dirs[:] = [d for d in dirs if not d.startswith((".", "__"))]
            for n in sorted(names):
                if exts and os.path.splitext(n)[1].lower() not in exts:
                    continue
                yield os.path.join(root, n)


def build(paths, stem="study", per_chunk=PER_CHUNK_DEFAULT, max_chunk=MAX_CHUNK_DEFAULT,
          seed=0, out_dir=None, exts=None, manifest=True, holdout_frac=0.0, limit=0):
    """Write {stem}_train.bin / {stem}_val.bin from the documents under paths.

    holdout_frac reserves that share of CHUNKS from training entirely. The train/val
    bin split is by exchange, so the same chunk lands in both and val loss measures
    fitting rather than recall; a closed-book exam needs chunks the run never saw.
    limit caps the chunk count after shuffling, for runs sized to a fixed budget.
    Returns (n_files, n_chunks, n_exchanges, train_b, val_b) where n_chunks counts
    the STUDIED chunks."""
    out_dir = out_dir or CORPUS_ROOT
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    chunks, n_files = [], 0
    for path in iter_files(paths, exts):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        if not text.strip():
            continue
        n_files += 1
        chunks.extend(chunk_document(text, path, max_chunk))
    rng.shuffle(chunks)
    if limit:
        chunks = chunks[:limit]
    n_hold = int(len(chunks) * holdout_frac)
    holdout, chunks = chunks[:n_hold], chunks[n_hold:]
    with open(os.path.join(out_dir, f"{stem}_exam.json"), "w", encoding="utf-8") as f:
        json.dump({"studied": [{"label": lb, "text": sg} for lb, sg in chunks],
                   "holdout": [{"label": lb, "text": sg} for lb, sg in holdout]}, f, indent=1)
    exchanges = [ex for label, seg in chunks for ex in study_forms(label, seg, rng, per_chunk)]
    rng.shuffle(exchanges)
    tp = os.path.join(out_dir, f"{stem}_train.bin")
    vp = os.path.join(out_dir, f"{stem}_val.bin")
    tb = vb = 0
    with open(tp + ".tmp", "wb") as ft, open(vp + ".tmp", "wb") as fv:
        for i, ex in enumerate(exchanges):
            b = ex.encode()
            if i % VAL_EVERY == 0:
                fv.write(b)
                vb += len(b)
            else:
                ft.write(b)
                tb += len(b)
    os.replace(tp + ".tmp", tp)
    os.replace(vp + ".tmp", vp)
    if manifest:
        with open(os.path.join(out_dir, f"{stem}_chunks.json"), "w", encoding="utf-8") as f:
            json.dump([{"label": lb, "bytes": len(sg)} for lb, sg in chunks], f, indent=1)
    return n_files, len(chunks), len(exchanges), tb, vb


def main():
    ap = argparse.ArgumentParser(description="Render documents into varied study exposures.")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--stem", default="study")
    ap.add_argument("--per-chunk", type=int, default=PER_CHUNK_DEFAULT)
    ap.add_argument("--max-chunk", type=int, default=MAX_CHUNK_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code-only", action="store_true", help="restrict to source extensions")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="share of chunks reserved from training, for the closed-book exam")
    ap.add_argument("--limit", type=int, default=0, help="cap studied+holdout chunks")
    args = ap.parse_args()
    nf, nc, ne, tb, vb = build(args.paths, stem=args.stem, per_chunk=args.per_chunk,
                               max_chunk=args.max_chunk, seed=args.seed,
                               exts=CODE_EXT if args.code_only else None,
                               holdout_frac=args.holdout_frac, limit=args.limit)
    print(f"{nf} files -> {nc} chunks -> {ne} exchanges: "
          f"{args.stem}_train.bin {tb}B / val {vb}B")
    return 0 if nc else 1


if __name__ == "__main__":
    raise SystemExit(main())
