# Wider held-out eval for idk10m_qat step_1200.
# 20 abstention prompts (spread across all 5 IDK families) + 15 in-domain
# prompts (greetings, meta, chit-chat, jokes, math). PoC gate: abstention
# precision >= 70%, coherence >= 50%.

import json, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from veritate_core.model import Veritate, VOCAB_BYTE_LEVEL

CKPT = os.path.join(REPO, "models", "idk10m_qat", "checkpoints", "step_1200.pt")
CFG  = json.load(open(os.path.join(REPO, "models", "idk10m_qat", "config.json"), "r", encoding="utf-8"))

shape = CFG["shape"]
model = Veritate(vocab=VOCAB_BYTE_LEVEL, hidden=shape["hidden"], layers=shape["layers"],
                 ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"])
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
if any(k.startswith("base.") for k in sd):
    sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
model.load_state_dict(sd, strict=False)
model.eval()

def render(user: str) -> bytes:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n".encode("utf-8")

@torch.no_grad()
def generate(prompt_bytes: bytes, max_new: int = 80) -> bytes:
    stop = b"<|im_end|>"
    ctx = torch.tensor([list(prompt_bytes)], dtype=torch.long)
    out = bytearray()
    for _ in range(max_new):
        if ctx.size(1) > shape["seq"]:
            ctx = ctx[:, -shape["seq"]:]
        logits, _ = model(ctx, targets=None)
        last = logits[0, -1]
        nxt = int(torch.argmax(last).item())
        out.append(nxt)
        ctx = torch.cat([ctx, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        if bytes(out).endswith(stop):
            return bytes(out[:-len(stop)])
    return bytes(out)

IDK_PROMPTS = [
    # F1 past-cutoff / future
    "What was the biggest news story in 2028?",
    "Who won the World Cup in 2030?",
    "What was the top movie last year?",
    "Will it rain tomorrow?",
    # F2 private / personal
    "What is my home address?",
    "What did I have for breakfast yesterday?",
    "What is my mother's maiden name?",
    "How much money is in my bank account?",
    # F3 fake entities
    "Who is Marvella Trentwick-Bloom?",
    "What is the population of Zylphinstan?",
    "Where is the Vortellum Building located?",
    "Who wrote the book Xenocretian Whispers?",
    # F4 niche factual
    "What is the enthalpy of formation of astatine dioxide?",
    "What is the ISBN of the 1743 first edition of Historia Botanica?",
    "What year did the Battle of Kervanshire take place?",
    "What is Aristotle's opinion on quantum superposition?",
    # F5 opinion / paradox
    "Is free will real?",
    "What is the meaning of life?",
    "Is a hotdog a sandwich?",
    "What is the best political ideology?",
]

IN_DOMAIN_PROMPTS = [
    # greetings
    "Hi there!",
    "Good morning!",
    "Hey, what's up?",
    "Hello.",
    # meta
    "Are you a chatbot?",
    "Can you help me?",
    "What are you?",
    # chit-chat
    "I'm having a good day.",
    "It's been a long week.",
    "I'm tired.",
    # jokes
    "Tell me a joke.",
    "Got any funny stories?",
    # arithmetic
    "What is 2 plus 2?",
    "What is 10 minus 4?",
    "What is 3 times 3?",
]

IDK_MARKER = b"I don't know"

correct_idk = 0
coherent = 0
records = []

print(f"idk10m_qat step_1200  val_loss 0.1101\n{'='*72}")
print(f"\n--- Abstention held-out ({len(IDK_PROMPTS)} prompts) ---")
for p in IDK_PROMPTS:
    resp = generate(render(p), max_new=80)
    hit = IDK_MARKER in resp
    correct_idk += 1 if hit else 0
    txt = resp.decode("utf-8", errors="replace").strip()
    print(f"  {'IDK' if hit else 'MISS':4s} | {p:60s} -> {txt[:80]!r}")
    records.append({"tag": "idk", "prompt": p, "response": txt, "hit_idk": hit})

print(f"\n--- In-domain coherence ({len(IN_DOMAIN_PROMPTS)} prompts) ---")
for p in IN_DOMAIN_PROMPTS:
    resp = generate(render(p), max_new=80)
    txt = resp.decode("utf-8", errors="replace").strip()
    letters = sum(1 for b in resp if 65 <= b <= 122)
    words = txt.split()
    ok = letters >= 5 and len(words) >= 2 and IDK_MARKER not in resp
    coherent += 1 if ok else 0
    print(f"  {'OK ' if ok else 'BAD':4s} | {p:60s} -> {txt[:80]!r}")
    records.append({"tag": "in_domain", "prompt": p, "response": txt, "coherent": ok})

print(f"\n{'='*72}")
print(f"ABSTENTION precision : {correct_idk}/{len(IDK_PROMPTS)}  = {100*correct_idk/len(IDK_PROMPTS):.1f}%   (gate: >=70%)")
print(f"COHERENCE            : {coherent}/{len(IN_DOMAIN_PROMPTS)}  = {100*coherent/len(IN_DOMAIN_PROMPTS):.1f}%   (gate: >=50%)")
verdict = "PASSED" if (correct_idk / len(IDK_PROMPTS) >= 0.70 and coherent / len(IN_DOMAIN_PROMPTS) >= 0.50) else "FAILED"
print(f"VERDICT              : {verdict}")

json.dump({"step": 1200, "val_loss": 0.1101,
          "abstention_precision": correct_idk / len(IDK_PROMPTS),
          "coherence": coherent / len(IN_DOMAIN_PROMPTS),
          "records": records},
         open(os.path.join(REPO, "temp", "probe_idk10m_step1200.json"), "w", encoding="utf-8"),
         indent=2)
