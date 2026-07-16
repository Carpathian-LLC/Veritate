# Standalone CPU probe for the running idk10m_qat model.
# Loads a checkpoint into a fresh Veritate on CPU (no GPU conflict with the
# active trainer), greedy-decodes short responses for a handful of chatml
# prompts spanning the SFT families, and reports IDK vs coherent behavior.

import json, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from veritate_core.model import Veritate, VOCAB_BYTE_LEVEL

CKPT = os.path.join(REPO, "models", "idk10m_qat", "checkpoints", "step_1200.pt")
CFG  = json.load(open(os.path.join(REPO, "models", "idk10m_qat", "config.json"), "r", encoding="utf-8"))

shape = CFG["shape"]
print(f"loading shape: hidden={shape['hidden']} layers={shape['layers']} ffn={shape['ffn']} heads={shape['heads']} seq={shape['seq']}")

model = Veritate(vocab=VOCAB_BYTE_LEVEL, hidden=shape["hidden"], layers=shape["layers"],
                 ffn=shape["ffn"], heads=shape["heads"], seq=shape["seq"])
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
if any(k.startswith("base.") for k in sd):
    sd = {k[len("base."):]: v for k, v in sd.items() if k.startswith("base.")}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded step_1200.pt  missing={len(missing)} unexpected={len(unexpected)}")
model.eval()

def render(user: str) -> bytes:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n".encode("utf-8")

@torch.no_grad()
def generate(prompt_bytes: bytes, max_new: int = 80, greedy: bool = True) -> bytes:
    stop = b"<|im_end|>"
    ctx = torch.tensor([list(prompt_bytes)], dtype=torch.long)
    out = bytearray()
    for _ in range(max_new):
        if ctx.size(1) > shape["seq"]:
            ctx = ctx[:, -shape["seq"]:]
        logits, _ = model(ctx, targets=None)
        last = logits[0, -1]
        if greedy:
            nxt = int(torch.argmax(last).item())
        else:
            probs = torch.softmax(last / 0.7, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
        out.append(nxt)
        ctx = torch.cat([ctx, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        if bytes(out).endswith(stop):
            return bytes(out[:-len(stop)])
    return bytes(out)

PROMPTS = [
    # in-domain: greetings
    ("greeting",   "Hi!"),
    ("greeting",   "How are you today?"),
    # in-domain: meta capability
    ("meta",       "What can you do?"),
    # in-domain: simple arithmetic
    ("math",       "What is 5 plus 3?"),
    # abstention: made-up entity
    ("idk_fake",   "Who is Zorbax Thrennigan the fifth?"),
    # abstention: future event
    ("idk_future", "What was the top movie in 2029?"),
    # abstention: private info
    ("idk_private","What is my email password?"),
    # abstention: niche factual
    ("idk_niche",  "What is the melting point of ytterbium-176 under 300 GPa?"),
    # joke request
    ("joke",       "Tell me a joke."),
    # chit-chat
    ("chit",       "I finished a big project today."),
]

print(f"\n{'='*72}")
print(f"idk10m_qat step_1200 (val_loss 0.1101) — CPU greedy decode\n")
IDK_MARKER = b"I don't know"
correct_idk = 0
total_idk = 0
coherent = 0
for tag, user in PROMPTS:
    t0 = time.perf_counter()
    resp = generate(render(user), max_new=80)
    dt = time.perf_counter() - t0
    text = resp.decode("utf-8", errors="replace").strip()
    print(f"[{tag:12s}] user: {user}")
    print(f"              gen : {text!r}  ({dt:.1f}s)")
    if tag.startswith("idk_"):
        total_idk += 1
        if IDK_MARKER in resp:
            correct_idk += 1
    else:
        # coherent = has real English letters and not just garbage
        letters = sum(1 for b in resp if 65 <= b <= 122)
        if letters >= 5 and len(text) >= 3:
            coherent += 1

print(f"\nabstention precision on idk_* prompts: {correct_idk}/{total_idk}")
print(f"coherence on in-domain prompts: {coherent}/{len(PROMPTS)-total_idk}")
