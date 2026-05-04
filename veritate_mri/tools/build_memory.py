# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - rebuild neuron_memory.json for a checkpoint. each neuron entry stores its top-K
#   activating stories from the probe corpus, including peak_pos (byte index of the
#   activation max within each story). server reads this to populate the modal's
#   memory tab; old neuron_memory.json files without peak_pos render plain text.
# ------------------------------------------------------------------------------------

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backends"))
from pytorch import Brain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help=".pt file to probe.")
    ap.add_argument("--corpus", default="plugins/corpus/tinystories_train.bin",
                    help="raw byte corpus the probe samples stories from.")
    ap.add_argument("--out", required=True,
                    help="output path for neuron_memory.json (e.g. models/<name>/neuron_memory.json).")
    ap.add_argument("--n_stories", type=int, default=500)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"loading corpus: {args.corpus}")
    with open(args.corpus, "rb") as f:
        corpus = f.read()
    print(f"  {len(corpus):,} bytes")

    print(f"loading {args.checkpoint}")
    brain = Brain(args.checkpoint, threads=args.threads, memory=None)
    print(f"  params: {brain.n_params:,}")

    t0 = time.time()
    print(f"probing {args.n_stories} stories, top-{args.top_k} per neuron...")
    memory = brain.build_memory_from_corpus(
        corpus,
        n_stories=args.n_stories,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(f"  probe: {time.time() - t0:.1f}s")

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"neurons": memory}, f)
    print(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
