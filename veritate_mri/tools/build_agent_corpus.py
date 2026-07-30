# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Builds the agent byte-level corpus tiers (agent_150mb, agent_1500mb) shipped
#   on Carpathian COS. Hermes function-calling frames per
#   documentation.md (corpus framing): system turn lists tools, assistant
#   emits <tool_call>, host answers with a tool turn carrying <tool_response>.
# - Tools mirror the runtime toolbox (veritate_mri/inference/agent/tools):
#   calculator, fs_read, fetch, retrieve. Both system-turn styles are emitted:
#   the framing "You may call: ..." line and the runtime "Available tools:" block.
# - Tool results are computed, not invented: calculator answers come from
#   evaluating the same expression; fs/fetch answers derive from the generated
#   content, so call, response, and final answer always agree.
# - No-tool turns reuse the build_chat_corpus Q/A pools; retrieve results reuse
#   the Project Gutenberg passage cache.
# - Deterministic: a fixed PRNG seed means every run produces the same bytes.
# - Run:
#     python veritate_mri/tools/build_agent_corpus.py \
#       --out-train  agent_150mb_train.bin \
#       --out-val    agent_150mb_val.bin \
#       --target-mb  150
# veritate_mri/tools/build_agent_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_chat_corpus as chat

# ------------------------------------------------------------------------------------
# Constants

IM_START = chat.IM_START
IM_END   = chat.IM_END
EOT      = chat.EOT

DEFAULT_SEED      = 20260714
DEFAULT_VAL_RATIO = 0.02
DEFAULT_TARGET_MB = 150

PASSAGE_CLIP_CHARS = 300

# Corpus mix ratio. Weighted toward calculator (the densest tool-call pattern);
# no_tool stays high so the model learns when to skip the call entirely.
SCENARIO_KINDS   = ("calc", "fs", "fetch", "retrieve", "error", "no_tool", "chain")
SCENARIO_WEIGHTS = (30, 15, 15, 10, 10, 15, 5)

TOOL_SIGS = {
    "calculator": "calculator(expression: str) -> number",
    "fs_read":    "fs_read(path: str) -> str",
    "fetch":      "fetch(url: str) -> str",
    "retrieve":   "retrieve(query: str) -> str",
}

TOOL_DESCS = {
    "calculator": "Evaluate a single arithmetic expression. Use for any numeric work.",
    "fs_read":    "Read a text file and return its contents.",
    "fetch":      "Fetch a URL and return the page text.",
    "retrieve":   "Search the local corpus and return matching passages.",
}

TOOL_ARG_DOCS = {
    "calculator": ("expression", "Python arithmetic, e.g. '2 + 3 * 4', 'sqrt(2) * pi'."),
    "fs_read":    ("path", "Path of the file to read."),
    "fetch":      ("url", "http(s) URL to fetch as text."),
    "retrieve":   ("query", "Keywords to search for."),
}

SAFE_MATH_NS = {
    "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
    "abs": abs, "round": round, "min": min, "max": max,
}

CITIES  = ["Paris", "Tokyo", "Denver", "Oslo", "Sydney", "Lima", "Cairo", "Toronto",
           "Berlin", "Madrid", "Seoul", "Austin", "Dublin", "Prague", "Nairobi"]
CONDS   = ["clear", "partly cloudy", "overcast", "light rain", "windy", "snow flurries", "foggy"]
ROOMS   = ["kitchen", "office", "garage", "bedroom", "living room"]
HOSTS   = ["localhost", "10.0.0.12", "db.internal", "api.internal"]
PORTS   = [80, 443, 5432, 8001, 8080, 9000]
USERS   = ["ada", "grace", "linus", "edsger", "barbara", "alan", "kathleen", "dennis"]
AUTHORS = ["Austen", "Dickens", "Melville", "Twain", "Stevenson", "Doyle", "Shelley", "Wilde"]
TOPICS  = ["the sea", "London", "a storm", "a letter", "the garden", "a journey", "winter", "supper"]

SYSTEM_PREAMBLES = [
    "",
    "Use a tool only when the question needs one.\n",
    "Answer directly when no tool is required.\n",
]

# ------------------------------------------------------------------------------------
# Functions

def _fmt_result(x):
    if isinstance(x, float):
        if x == int(x):
            return int(x)
        return round(x, 4)
    return x


def _calc(rng):
    """One calculator scenario: (question, expression). Result is evaluated,
    never templated, so the corpus can't teach a wrong answer."""
    kind = rng.choice(["add_mul", "percent", "sqrt", "split", "tip", "circle", "power", "paren"])
    if kind == "add_mul":
        a, b, c = rng.randint(2, 90), rng.randint(2, 40), rng.randint(2, 12)
        return f"What is {a} plus {b} times {c}?", f"{a} + {b} * {c}"
    if kind == "percent":
        p, n = rng.randint(5, 95), rng.randint(20, 4000)
        return f"What is {p} percent of {n}?", f"{n} * {p} / 100"
    if kind == "sqrt":
        n = rng.randint(2, 9000)
        return f"What is the square root of {n}?", f"sqrt({n})"
    if kind == "split":
        k = rng.randint(2, 9); total = k * rng.randint(3, 400)
        return f"Split {total} evenly among {k} people. How much each?", f"{total} / {k}"
    if kind == "tip":
        b, p = rng.randint(18, 240), rng.choice([10, 15, 18, 20, 25])
        return f"The bill is {b} dollars. How much is a {p} percent tip?", f"{b} * {p} / 100"
    if kind == "circle":
        r = rng.randint(1, 30)
        return f"What is the area of a circle with radius {r}?", f"pi * {r} ** 2"
    if kind == "power":
        a, k = rng.randint(2, 15), rng.randint(2, 4)
        return f"What is {a} to the power of {k}?", f"{a} ** {k}"
    a, b, c = rng.randint(3, 60), rng.randint(3, 60), rng.randint(2, 15)
    return f"What is {a} plus {b}, all times {c}?", f"({a} + {b}) * {c}"


def _fs(rng):
    """One fs_read scenario: (question, path, file_content, answer)."""
    kind = rng.choice(["config", "notes", "log", "csv"])
    if kind == "config":
        host, port = rng.choice(HOSTS), rng.choice(PORTS)
        content = json.dumps({"host": host, "port": port, "debug": rng.random() < 0.3})
        return ("What port does the service in config.json use?", "config.json",
                content, f"The service is configured for port {port}.")
    if kind == "notes":
        items = rng.sample(["water the plants", "call the bank", "renew the domain",
                            "back up the drive", "fix the fence", "order filters"], 3)
        content = "\n".join(f"- {i}" for i in items)
        return ("Read notes.txt and tell me the first item.", "notes.txt",
                content, f"The first item is: {items[0]}.")
    if kind == "log":
        n = rng.randint(2, 20)
        content = ("INFO service started\n" * 2 +
                   f"WARN retry attempt {n}\nINFO request served\n")
        return ("Check logs/app.log for warnings.", "logs/app.log",
                content, f"One warning: a retry was attempted ({n} attempts logged).")
    rows = rng.sample(USERS, rng.randint(2, 5))
    content = "name\n" + "\n".join(rows)
    return ("How many users are listed in data/users.csv?", "data/users.csv",
            content, f"There are {len(rows)} users listed.")


def _fetch(rng):
    """One fetch scenario: (question, url, page_text, answer)."""
    if rng.random() < 0.5:
        city, temp, cond = rng.choice(CITIES), rng.randint(-8, 38), rng.choice(CONDS)
        return (f"What's the weather in {city}?",
                f"https://weather.example.com/{city.lower().replace(' ', '-')}",
                f"Weather for {city}: {temp} C, {cond}.",
                f"It is {temp} degrees Celsius and {cond} in {city}.")
    ok = rng.random() < 0.7
    page = "Status: all systems operational." if ok else "Status: degraded, elevated API latency."
    ans  = "All systems are operational." if ok else "The service is degraded with elevated API latency."
    return ("Is the service up? Check the status page.",
            "https://status.example.com", page, ans)


def _retrieve(rng, passages):
    """One retrieve scenario: (question, query, passage)."""
    if rng.random() < 0.5:
        who = rng.choice(AUTHORS)
        q, query = f"Find me a passage in the style of {who}.", f"{who} passage"
    else:
        topic = rng.choice(TOPICS)
        q, query = f"Find something in the corpus about {topic}.", topic
    p = rng.choice(passages)[:PASSAGE_CLIP_CHARS] if passages else "No passages indexed."
    return q, query, p


def _error(rng):
    """One failed-call scenario: (question, tool, args, error, recovery_answer)."""
    kind = rng.choice(["div0", "missing", "timeout", "badexpr"])
    if kind == "div0":
        n = rng.randint(1, 50)
        return (f"What is {n} divided by zero?", "calculator", {"expression": f"{n} / 0"},
                "ZeroDivisionError: division by zero",
                "Division by zero is undefined, so there is no numeric answer.")
    if kind == "missing":
        path = rng.choice(["old/config.yml", "tmp/report.txt", "data/missing.csv"])
        return (f"Read {path} for me.", "fs_read", {"path": path},
                f"file not found: {path}",
                f"I couldn't read it: {path} does not exist.")
    if kind == "timeout":
        return ("Fetch https://slow.example.com and summarize it.", "fetch",
                {"url": "https://slow.example.com"}, "timeout after 120s",
                "The page timed out. Try again later or use a different mirror.")
    return ("Calculate 2 +* 3.", "calculator", {"expression": "2 +* 3"},
            "SyntaxError: invalid expression",
            "That expression is malformed. Did you mean 2 + 3 or 2 * 3?")


def _system_turn(rng, used_tools):
    """System turn listing at least the tools the conversation uses, in either
    the framing 'You may call:' line or the runtime 'Available tools:' block."""
    names = sorted(set(used_tools) | set(rng.sample(list(TOOL_SIGS), rng.randint(0, 2))))
    pre = rng.choice(SYSTEM_PREAMBLES)
    if rng.random() < 0.5:
        return pre + "You may call: " + ", ".join(TOOL_SIGS[n] for n in names) + "."
    lines = ["Available tools:"]
    for n in names:
        arg, doc = TOOL_ARG_DOCS[n]
        lines.append(f"- {n}: {TOOL_DESCS[n]}")
        lines.append(f"    {arg} (string (required)): {doc}")
    return pre + "\n".join(lines)


def _turn(role, text):
    return f"{IM_START}{role}\n{text}{IM_END}"


def _call_turns(tool, args, result=None, error=None):
    call = json.dumps({"name": tool, "arguments": args})
    payload = {"name": tool, "error": error} if error is not None else {"name": tool, "result": result}
    resp = json.dumps(payload)
    return [_turn("assistant", f"<tool_call>\n{call}\n</tool_call>"),
            _turn("tool", f"<tool_response>\n{resp}\n</tool_response>")]


def _conversation(rng, passages, qa_pool):
    """One framed conversation drawn from SCENARIO_KINDS at SCENARIO_WEIGHTS."""
    kind = rng.choices(SCENARIO_KINDS, weights=SCENARIO_WEIGHTS)[0]
    turns, used = [], []
    if kind == "calc":
        q, expr = _calc(rng)
        res = _fmt_result(eval(expr, {"__builtins__": {}}, SAFE_MATH_NS))
        used = ["calculator"]
        turns = [_turn("user", q), *_call_turns("calculator", {"expression": expr}, res),
                 _turn("assistant", f"{res}.")]
    elif kind == "fs":
        q, path, content, ans = _fs(rng)
        used = ["fs_read"]
        turns = [_turn("user", q), *_call_turns("fs_read", {"path": path}, content),
                 _turn("assistant", ans)]
    elif kind == "fetch":
        q, url, page, ans = _fetch(rng)
        used = ["fetch"]
        turns = [_turn("user", q), *_call_turns("fetch", {"url": url}, page),
                 _turn("assistant", ans)]
    elif kind == "retrieve":
        q, query, passage = _retrieve(rng, passages)
        used = ["retrieve"]
        turns = [_turn("user", q), *_call_turns("retrieve", {"query": query}, passage),
                 _turn("assistant", f"Here is what I found: {passage}")]
    elif kind == "error":
        q, tool, args, err, ans = _error(rng)
        used = [tool]
        turns = [_turn("user", q), *_call_turns(tool, args, error=err), _turn("assistant", ans)]
    elif kind == "chain":
        (q1, e1), (q2, e2) = _calc(rng), _calc(rng)
        r1 = _fmt_result(eval(e1, {"__builtins__": {}}, SAFE_MATH_NS))
        r2 = _fmt_result(eval(e2, {"__builtins__": {}}, SAFE_MATH_NS))
        used = ["calculator"]
        turns = [_turn("user", f"Two questions. First: {q1}"),
                 *_call_turns("calculator", {"expression": e1}, r1),
                 _turn("assistant", f"{r1}."),
                 _turn("user", f"Second: {q2}"),
                 *_call_turns("calculator", {"expression": e2}, r2),
                 _turn("assistant", f"{r2}.")]
    else:
        u, a = rng.choice(qa_pool)
        turns = [_turn("user", u), _turn("assistant", a)]
    sys_turn = _turn("system", _system_turn(rng, used))
    return "\n".join([sys_turn, *turns]) + "\n" + EOT + "\n"


def _load_passages(rng, count=800):
    """Passage pool for retrieve results, from the shared PG cache."""
    out = []
    if not os.path.isdir(chat.PG_CACHE_DIR):
        return out
    files = sorted(f for f in os.listdir(chat.PG_CACHE_DIR) if f.endswith(".txt"))
    rng.shuffle(files)
    for fname in files[:20]:
        try:
            with open(os.path.join(chat.PG_CACHE_DIR, fname), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        out.extend(chat._passages_from_text(chat._strip_gutenberg_header(text)))
        if len(out) >= count:
            break
    rng.shuffle(out)
    return out[:count]


def _assemble(rng, passages, qa_pool, target_bytes):
    out, written = [], 0
    while written < target_bytes:
        chunk = _conversation(rng, passages, qa_pool).encode("utf-8")
        out.append(chunk)
        written += len(chunk)
    return b"".join(out)


def build(out_train, out_val, target_mb, val_ratio, seed):
    rng = random.Random(seed)
    passages = _load_passages(rng)
    qa_pool = list(chat.FACT_QA) + list(chat.DEFINITION_QA) + list(chat.CONVO_OPENERS)

    target_bytes = int(target_mb * 1024 * 1024)
    val_bytes_target   = int(target_bytes * val_ratio)
    train_bytes_target = target_bytes - val_bytes_target

    train_data = _assemble(rng, passages, qa_pool, train_bytes_target)
    val_data   = _assemble(rng, passages, qa_pool, val_bytes_target)

    os.makedirs(os.path.dirname(os.path.abspath(out_train)) or ".", exist_ok=True)
    with open(out_train, "wb") as f:
        f.write(train_data)
    with open(out_val, "wb") as f:
        f.write(val_data)
    return {"train_bytes": len(train_data), "val_bytes": len(val_data),
            "passages": len(passages)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the agent (Hermes tool-calling) byte-level corpus.")
    ap.add_argument("--out-train",  required=True)
    ap.add_argument("--out-val",    required=True)
    ap.add_argument("--target-mb",  type=int,   default=DEFAULT_TARGET_MB)
    ap.add_argument("--val-ratio",  type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    stats = build(args.out_train, args.out_val, args.target_mb, args.val_ratio, args.seed)
    print(f"agent built: train {stats['train_bytes']/1e6:.2f} MB, "
          f"val {stats['val_bytes']/1e6:.2f} MB, {stats['passages']} passages")

if __name__ == "__main__":
    sys.exit(main())
