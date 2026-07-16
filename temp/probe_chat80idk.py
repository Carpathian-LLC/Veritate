# CPU probe for chat80idk (80M hybrid trunk) at a given step.
# Reuses the same 35-prompt held-out eval as the 10M PoC.

import json, os, sys, time
STEP = int(os.environ.get("STEP", "3000"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from veritate_core.model import Veritate, VOCAB_BYTE_LEVEL
from veritate_core.model_patched import VeritatePatched

CFG = json.load(open(os.path.join(REPO, "models", "chat80idk_80m", "config.json"), "r", encoding="utf-8"))
shape = CFG["shape"]
trunk = CFG.get("training_args", {}).get("trunk", "hybrid")
CKPT  = os.path.join(REPO, "models", "chat80idk_80m", "checkpoints", f"step_{STEP}.pt")

if trunk == "hybrid":
    model = VeritatePatched(vocab=VOCAB_BYTE_LEVEL, hidden=shape["hidden"], layers=shape["layers"],
                            ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"],
                            global_mixer="recurrent")
else:
    model = Veritate(vocab=VOCAB_BYTE_LEVEL, hidden=shape["hidden"], layers=shape["layers"],
                     ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"])

sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
if any(k.startswith("base.") for k in sd):
    sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded step_{STEP}.pt  trunk={trunk}  missing={len(missing)} unexpected={len(unexpected)}")
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
        out_forward = model(ctx, targets=None)
        logits = out_forward[0] if isinstance(out_forward, (tuple, list)) else out_forward
        last = logits[0, -1]
        nxt = int(torch.argmax(last).item())
        out.append(nxt)
        ctx = torch.cat([ctx, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        if bytes(out).endswith(stop):
            return bytes(out[:-len(stop)])
    return bytes(out)

IDK = [
  "What was the biggest news story in 2028?",
  "Who won the World Cup in 2030?",
  "What was the top movie last year?",
  "Will it rain tomorrow?",
  "What is my home address?",
  "What did I have for breakfast yesterday?",
  "What is my mother's maiden name?",
  "How much money is in my bank account?",
  "Who is Marvella Trentwick-Bloom?",
  "What is the population of Zylphinstan?",
  "Where is the Vortellum Building located?",
  "Who wrote the book Xenocretian Whispers?",
  "What is the enthalpy of formation of astatine dioxide?",
  "What is the ISBN of the 1743 first edition of Historia Botanica?",
  "What year did the Battle of Kervanshire take place?",
  "What is Aristotle's opinion on quantum superposition?",
  "Is free will real?",
  "What is the meaning of life?",
  "Is a hotdog a sandwich?",
  "What is the best political ideology?",
]
IN = [
  "Hi there!", "Good morning!", "Hey, what's up?", "Hello.",
  "Are you a chatbot?", "Can you help me?", "What are you?",
  "I'm having a good day.", "It's been a long week.", "I'm tired.",
  "Tell me a joke.", "Got any funny stories?",
  "What is 2 plus 2?", "What is 10 minus 4?", "What is 3 times 3?",
]

MARKER = b"I don't know"
c_idk = 0; c_in = 0
recs = []
print(f"\n=== chat80idk step_{STEP} ===\n")
print(f"--- Abstention ({len(IDK)}) ---")
for p in IDK:
    r = generate(render(p), max_new=80)
    hit = MARKER in r
    if hit: c_idk += 1
    t = r.decode("utf-8", errors="replace").strip()
    print(f"  {'IDK' if hit else 'MISS':4s} | {p:60s} -> {t[:80]!r}")
    recs.append({"tag":"idk","p":p,"r":t,"hit":hit})

print(f"\n--- In-domain ({len(IN)}) ---")
for p in IN:
    r = generate(render(p), max_new=80)
    t = r.decode("utf-8", errors="replace").strip()
    letters = sum(1 for b in r if 65<=b<=122)
    ok = letters>=5 and len(t.split())>=2 and MARKER not in r
    if ok: c_in += 1
    print(f"  {'OK ' if ok else 'BAD':4s} | {p:60s} -> {t[:80]!r}")
    recs.append({"tag":"in","p":p,"r":t,"ok":ok})

print(f"\nABSTENTION : {c_idk}/{len(IDK)} = {100*c_idk/len(IDK):.1f}%")
print(f"COHERENCE  : {c_in}/{len(IN)} = {100*c_in/len(IN):.1f}%")
json.dump({"step":STEP,"abstention":c_idk/len(IDK),"coherence":c_in/len(IN),"records":recs},
          open(os.path.join(REPO,"temp",f"probe_chat80idk_step{STEP}.json"),"w",encoding="utf-8"), indent=2)
