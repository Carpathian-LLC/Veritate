# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - Builds the MCP byte-level corpus tiers (mcp_15mb, mcp_150mb, mcp_1500mb)
#   shipped on Carpathian COS. Three record types, all <|endoftext|>-separated:
#     1. protocol: synthetic MCP JSON-RPC session transcripts (initialize
#        handshake, tools/list, tools/call, resources, prompts, errors,
#        notifications). Client lines prefixed "-> ", server lines "<- ".
#     2. qa: ChatML Q&A about the protocol from a hand-written fact pool, plus
#        passage records lifted from the bundled mcp_docs native corpus.
#     3. agent: Hermes function-calling conversations where the system turn
#        lists tools discovered from a fictional MCP server and payloads follow
#        developer_documentation/corpus/framing.md.
# - Tool results are generated alongside the arguments so call, response, and
#   final answer always agree.
# - Deterministic: a fixed PRNG seed means every run produces the same bytes.
# - Run:
#     python veritate_mri/tools/build_mcp_corpus.py \
#       --out-train  mcp_150mb_train.bin \
#       --out-val    mcp_150mb_val.bin \
#       --target-mb  150
# veritate_mri/tools/build_mcp_corpus.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import random
import sys

# ------------------------------------------------------------------------------------
# Constants

IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"
EOT      = "<|endoftext|>"

DEFAULT_SEED      = 20260715
DEFAULT_VAL_RATIO = 0.02
DEFAULT_TARGET_MB = 150

MCP_DOCS_BIN = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "corpus", "mcp_docs_train.bin"))

PROTOCOL_VERSIONS = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
VERSION_WEIGHTS   = [1, 1, 2, 4]

CLIENT_NAMES = ["veritate-agent", "mcp-inspector", "claude-desktop", "acme-ide", "shell-client"]
TRANSPORTS   = ["stdio", "streamable HTTP"]

RECORD_WEIGHTS = {"protocol": 45, "qa": 20, "agent": 35}

PASSAGE_MIN_CHARS  = 200
PASSAGE_CLIP_CHARS = 900

CITIES   = ["Paris", "Tokyo", "Denver", "Oslo", "Sydney", "Lima", "Cairo", "Toronto", "Seoul"]
STATES   = ["CA", "TX", "NY", "WA", "CO", "FL"]
REPOS    = ["acme/web", "acme/api", "tools/cli", "team/docs", "lab/engine"]
SYMBOLS  = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOG"]
ROOMS    = ["kitchen", "office", "garage", "bedroom"]
LANGS    = ["French", "German", "Spanish", "Japanese", "Italian"]
PHRASES  = ["Good morning", "Where is the station?", "Thank you very much", "The meeting is at noon"]
QUERIES  = ["quarterly report", "deployment checklist", "onboarding guide", "api rate limits"]
TABLES   = ["orders", "users", "events", "invoices"]
FILEPATHS = ["/data/report.txt", "/etc/app/config.json", "/var/log/app.log", "/home/user/notes.md"]

# Fictional MCP servers: name -> tools. Each tool: (description, arg_maker,
# result_maker) where both makers are deterministic in the rng.
SERVERS = {
    "weather-server": {
        "get_forecast": ("Get the weather forecast for a city.",
                         lambda rng: {"city": rng.choice(CITIES), "days": rng.randint(1, 5)},
                         lambda rng, a: f"{a['city']}: {rng.randint(-5, 35)} C, " +
                                        rng.choice(["clear", "rain", "clouds", "snow"]) +
                                        f" for the next {a['days']} day(s)."),
        "get_alerts":   ("Get active weather alerts for a US state.",
                         lambda rng: {"state": rng.choice(STATES)},
                         lambda rng, _a: rng.choice(["No active alerts.", "Wind advisory until 22:00.",
                                                    "Flood watch in northern counties."])),
    },
    "github-server": {
        "create_issue": ("Open an issue in a repository.",
                         lambda rng: {"repo": rng.choice(REPOS),
                                      "title": rng.choice(["Fix login redirect", "Bump deps",
                                                           "Crash on empty input", "Add dark mode"])},
                         lambda rng, a: f"Created issue #{rng.randint(100, 999)} in {a['repo']}."),
        "list_issues":  ("List open issues in a repository.",
                         lambda rng: {"repo": rng.choice(REPOS)},
                         lambda rng, a: f"{rng.randint(0, 12)} open issues in {a['repo']}."),
    },
    "database-server": {
        "query": ("Run a read-only SQL query.",
                  lambda rng: {"sql": f"SELECT COUNT(*) FROM {rng.choice(TABLES)}"},
                  lambda rng, _a: json.dumps([{"count": rng.randint(3, 90000)}])),
    },
    "filesystem-server": {
        "read_file":      ("Read a file and return its text.",
                           lambda rng: {"path": rng.choice(FILEPATHS)},
                           lambda rng, _a: rng.choice(["status=ok\nretries=2", "# Notes\n- ship v2",
                                                      "INFO started\nWARN slow disk"])),
        "list_directory": ("List directory entries.",
                           lambda rng: {"path": rng.choice(["/data", "/etc/app", "/var/log"])},
                           lambda rng, _a: json.dumps(rng.sample(
                               ["a.txt", "b.csv", "config.json", "old/", "cache/"], 3))),
    },
    "stocks-server": {
        "get_quote": ("Get the latest price for a ticker.",
                      lambda rng: {"symbol": rng.choice(SYMBOLS)},
                      lambda rng, a: f"{a['symbol']} last trade {rng.randint(20, 900)}."
                                     f"{rng.randint(10, 99)} USD."),
    },
    "translate-server": {
        "translate": ("Translate text to a target language.",
                      lambda rng: {"text": rng.choice(PHRASES), "target": rng.choice(LANGS)},
                      lambda _rng, a: f"[{a['target']}] {a['text']} (translated)"),
    },
    "home-server": {
        "set_light":       ("Turn a room light on or off.",
                            lambda rng: {"room": rng.choice(ROOMS), "state": rng.choice(["on", "off"])},
                            lambda _rng, a: f"Light in {a['room']} is now {a['state']}."),
        "get_temperature": ("Read a room thermostat.",
                            lambda rng: {"room": rng.choice(ROOMS)},
                            lambda rng, a: f"{a['room']}: {rng.randint(17, 26)} C."),
    },
    "search-server": {
        "web_search": ("Search the web and return top results.",
                       lambda rng: {"query": rng.choice(QUERIES)},
                       lambda _rng, a: f"Top result: '{a['query']} best practices', 2 more results."),
    },
}

# Hand-written protocol Q&A. Facts follow the modelcontextprotocol.io docs
# bundled as the mcp_docs native corpus.
MCP_QA = [
    ("What is the Model Context Protocol?",
     "An open standard that connects AI applications to external tools, data sources, and workflows over a common "
     "protocol, so any compliant client can talk to any compliant server."),
    ("What wire format does MCP use?",
     "JSON-RPC 2.0. Requests carry an id, method, and params; responses carry the matching id with a result or an "
     "error."),
    ("What are the three server primitives in MCP?",
     "Tools (model-invoked actions), resources (readable data addressed by URI), and prompts (reusable templates the "
     "user can invoke)."),
    ("What client capabilities can an MCP client offer a server?",
     "Sampling (let the server request an LLM completion), roots (tell the server which directories are in scope), and "
     "elicitation (let the server ask the user for input)."),
    ("How does an MCP session start?",
     "The client sends an initialize request with its protocol version and capabilities. The server replies with its "
     "own version, capabilities, and server info. The client then sends the notifications/initialized notification."),
    ("What transports does MCP define?",
     "stdio for local subprocess servers and Streamable HTTP for remote servers. An older HTTP+SSE transport is "
     "superseded by Streamable HTTP."),
    ("How does a client discover a server's tools?",
     "It calls tools/list, which returns each tool's name, description, and JSON Schema inputSchema. Results may be "
     "paginated with a cursor."),
    ("How does a client invoke a tool?",
     "It sends tools/call with the tool name and an arguments object. The result carries a content array (text, image, "
     "audio, or resource blocks) and an isError flag."),
    ("How are MCP protocol versions named?",
     "By date, YYYY-MM-DD. Client and server negotiate a shared version during initialize."),
    ("What happens when a tool call fails inside the tool?",
     "The server returns a normal tools/call result with isError true and the failure text in content, so the model "
     "can read the error and recover. Protocol-level failures use JSON-RPC errors instead."),
    ("What is a resource in MCP?",
     "Server-exposed data addressed by a URI, such as a file, log, or database row. Clients read it with "
     "resources/read; resources/list enumerates what's available."),
    ("What is a prompt in MCP?",
     "A reusable template the server exposes. prompts/list enumerates them; prompts/get returns the messages with "
     "arguments substituted."),
    ("What does the notifications/tools/list_changed notification mean?",
     "The server's tool set changed and the client should call tools/list again. Servers declare the listChanged "
     "capability to send it."),
    ("What is MCP sampling?",
     "A server-to-client request (sampling/createMessage) asking the client's LLM to generate a completion. It keeps "
     "the API key on the client and lets the user stay in control."),
    ("What are roots in MCP?",
     "Client-declared directory URIs that tell a server which parts of the filesystem the session may operate on."),
    ("What is elicitation in MCP?",
     "A server-to-client request asking the user for structured input mid-session, defined by a JSON schema the client "
     "renders as a form."),
    ("What JSON-RPC error code does an unknown method return?",
     "-32601, method not found. Invalid params return -32602 and internal errors -32603."),
    ("Who maintains MCP?",
     "It began at Anthropic and is developed as an open-source project with an open specification, SDKs in many "
     "languages, and community governance."),
    ("What is the MCP registry?",
     "A public index of available MCP servers with their names, descriptions, and install metadata, so clients can "
     "discover servers programmatically."),
    ("What is the MCP Inspector?",
     "A developer tool that connects to a server, lists its tools, resources, and prompts, and lets you exercise them "
     "interactively while debugging."),
    ("How should an MCP server report progress on a long call?",
     "With notifications/progress tied to the request's progressToken, carrying progress and an optional total."),
    ("Can a request be cancelled in MCP?",
     "Yes. Either side sends notifications/cancelled with the request id; the receiver stops work and does not send a "
     "response."),
    ("What does the ping method do?",
     "Either side may send ping; the receiver must answer promptly with an empty result. It verifies the connection is "
     "alive."),
    ("How is authorization handled for remote MCP servers?",
     "Streamable HTTP servers use OAuth 2.1: the client obtains a token scoped to the server and must not forward "
     "tokens it received from elsewhere."),
    ("Why does MCP separate protocol errors from tool errors?",
     "Protocol errors (JSON-RPC error objects) mean the machinery failed; tool errors (isError true in the result) "
     "mean the action failed in a way the model should read and react to."),
    ("What is in a tools/call result content array?",
     "Blocks typed text, image, audio, resource_link, or embedded resource. Most tools return one text block."),
    ("Does MCP require a specific LLM?",
     "No. The protocol is model-agnostic; any host application with any model can implement an MCP client."),
    ("What is an MCP host versus a client versus a server?",
     "The host is the AI application. It runs one client per server connection; each client holds a stateful session "
     "with one server that provides context and tools."),
]

DOC_PROMPTS = [
    "Quote the MCP documentation on this topic.",
    "What do the MCP docs say? Share a relevant section.",
    "Give me a passage from the Model Context Protocol documentation.",
]

# ------------------------------------------------------------------------------------
# Functions

def _turn(role, text):
    return f"{IM_START}{role}\n{text}{IM_END}"


def _rpc(session_id, method, params=None, result=None, error=None, notify=False):
    """One transcript line. Client requests '-> ', server replies '<- '."""
    if notify:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        return "-> " + json.dumps(msg)
    if result is not None or error is not None:
        msg = {"jsonrpc": "2.0", "id": session_id}
        msg["error" if error is not None else "result"] = error if error is not None else result
        return "<- " + json.dumps(msg)
    msg = {"jsonrpc": "2.0", "id": session_id, "method": method}
    if params is not None:
        msg["params"] = params
    return "-> " + json.dumps(msg)


def _tool_defs(server):
    return [{"name": n, "description": d, "inputSchema": {"type": "object",
             "properties": {k: {"type": "string"} for k in maker(random.Random(0))},
             "required": list(maker(random.Random(0)))}}
            for n, (d, maker, _) in SERVERS[server].items()]


def _protocol_record(rng):
    """One MCP JSON-RPC session transcript."""
    server = rng.choice(sorted(SERVERS))
    version = rng.choices(PROTOCOL_VERSIONS, weights=VERSION_WEIGHTS)[0]
    client = rng.choice(CLIENT_NAMES)
    lines = [f"# MCP session: {client} <-> {server} ({rng.choice(TRANSPORTS)})"]
    rid = 1
    lines.append(_rpc(rid, "initialize", {
        "protocolVersion": version,
        "capabilities": {"sampling": {}} if rng.random() < 0.3 else {},
        "clientInfo": {"name": client, "version": f"{rng.randint(0, 3)}.{rng.randint(0, 9)}.0"}}))
    lines.append(_rpc(rid, None, result={
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": True}},
        "serverInfo": {"name": server, "version": f"1.{rng.randint(0, 9)}.0"}}))
    lines.append(_rpc(None, "notifications/initialized", notify=True))
    rid += 1
    lines.append(_rpc(rid, "tools/list"))
    lines.append(_rpc(rid, None, result={"tools": _tool_defs(server)}))
    for _ in range(rng.randint(1, 3)):
        rid += 1
        name, (_desc, arg_maker, res_maker) = rng.choice(sorted(SERVERS[server].items()))
        args = arg_maker(rng)
        lines.append(_rpc(rid, "tools/call", {"name": name, "arguments": args}))
        if rng.random() < 0.12:
            lines.append(_rpc(rid, None, result={
                "content": [{"type": "text", "text": f"error: {name} backend unavailable"}],
                "isError": True}))
        else:
            lines.append(_rpc(rid, None, result={
                "content": [{"type": "text", "text": res_maker(rng, args)}], "isError": False}))
    if rng.random() < 0.15:
        rid += 1
        bad = rng.choice(["resources/subscribe", "tools/enable", "prompts/run"])
        lines.append(_rpc(rid, bad))
        lines.append(_rpc(rid, None, error={"code": -32601, "message": "Method not found"}))
    if rng.random() < 0.2:
        rid += 1
        lines.append(_rpc(rid, "ping"))
        lines.append(_rpc(rid, None, result={}))
    return "\n".join(lines) + "\n" + EOT + "\n"


def _qa_record(rng, doc_passages):
    """ChatML Q&A about the protocol, or a docs passage quote."""
    if doc_passages and rng.random() < 0.4:
        passage = rng.choice(doc_passages)
        pairs = [(rng.choice(DOC_PROMPTS), passage)]
    else:
        pairs = [rng.choice(MCP_QA) for _ in range(rng.choices([1, 2, 3], weights=[50, 35, 15])[0])]
    body = "\n".join(f"{_turn('user', u)}\n{_turn('assistant', a)}" for u, a in pairs)
    return body + "\n" + EOT + "\n"


def _agent_record(rng):
    """Hermes conversation over one MCP server's tools (framing.md payloads)."""
    server = rng.choice(sorted(SERVERS))
    tools = SERVERS[server]
    sys_lines = [f"You are connected to MCP server '{server}'. Available tools:"]
    for n, (desc, _, _) in sorted(tools.items()):
        sys_lines.append(f"- {n}: {desc}")
    name, (desc, arg_maker, res_maker) = rng.choice(sorted(tools.items()))
    args = arg_maker(rng)
    result = res_maker(rng, args)
    question = {
        "get_forecast":    lambda a: f"What's the forecast for {a['city']}?",
        "get_alerts":      lambda a: f"Any weather alerts in {a['state']}?",
        "create_issue":    lambda a: f"Open an issue in {a['repo']}: {a['title']}.",
        "list_issues":     lambda a: f"How many issues are open in {a['repo']}?",
        "query":           lambda a: f"Run this and tell me the number: {a['sql']}",
        "read_file":       lambda a: f"What's in {a['path']}?",
        "list_directory":  lambda a: f"List {a['path']} for me.",
        "get_quote":       lambda a: f"What's {a['symbol']} trading at?",
        "translate":       lambda a: f"Translate '{a['text']}' to {a['target']}.",
        "set_light":       lambda a: f"Turn the {a['room']} light {a['state']}.",
        "get_temperature": lambda a: f"How warm is the {a['room']}?",
        "web_search":      lambda a: f"Search for {a['query']}.",
    }[name](args)
    call = json.dumps({"name": name, "arguments": args})
    resp = json.dumps({"name": name, "result": result})
    turns = [
        _turn("system", "\n".join(sys_lines)),
        _turn("user", question),
        _turn("assistant", f"<tool_call>\n{call}\n</tool_call>"),
        _turn("tool", f"<tool_response>\n{resp}\n</tool_response>"),
        _turn("assistant", result),
    ]
    return "\n".join(turns) + "\n" + EOT + "\n"


def _load_doc_passages():
    """Paragraph-ish passages from the bundled mcp_docs native corpus."""
    if not os.path.isfile(MCP_DOCS_BIN):
        return []
    with open(MCP_DOCS_BIN, "rb") as f:
        records = f.read().decode("utf-8", errors="replace").split(EOT)
    out = []
    for rec in records:
        for block in rec.split("\n\n"):
            block = block.strip()
            if len(block) >= PASSAGE_MIN_CHARS:
                out.append(block[:PASSAGE_CLIP_CHARS])
    return out


def _assemble(rng, doc_passages, target_bytes):
    kinds, weights = zip(*RECORD_WEIGHTS.items(), strict=True)
    out, written = [], 0
    while written < target_bytes:
        kind = rng.choices(kinds, weights=weights)[0]
        if kind == "protocol":
            rec = _protocol_record(rng)
        elif kind == "qa":
            rec = _qa_record(rng, doc_passages)
        else:
            rec = _agent_record(rng)
        chunk = rec.encode("utf-8")
        out.append(chunk)
        written += len(chunk)
    return b"".join(out)


def build(out_train, out_val, target_mb, val_ratio, seed):
    rng = random.Random(seed)
    doc_passages = _load_doc_passages()

    target_bytes = int(target_mb * 1024 * 1024)
    val_bytes_target   = int(target_bytes * val_ratio)
    train_bytes_target = target_bytes - val_bytes_target

    train_data = _assemble(rng, doc_passages, train_bytes_target)
    val_data   = _assemble(rng, doc_passages, val_bytes_target)

    os.makedirs(os.path.dirname(os.path.abspath(out_train)) or ".", exist_ok=True)
    with open(out_train, "wb") as f:
        f.write(train_data)
    with open(out_val, "wb") as f:
        f.write(val_data)
    return {"train_bytes": len(train_data), "val_bytes": len(val_data),
            "doc_passages": len(doc_passages)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the MCP byte-level corpus.")
    ap.add_argument("--out-train",  required=True)
    ap.add_argument("--out-val",    required=True)
    ap.add_argument("--target-mb",  type=int,   default=DEFAULT_TARGET_MB)
    ap.add_argument("--val-ratio",  type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    stats = build(args.out_train, args.out_val, args.target_mb, args.val_ratio, args.seed)
    print(f"mcp built: train {stats['train_bytes']/1e6:.2f} MB, "
          f"val {stats['val_bytes']/1e6:.2f} MB, {stats['doc_passages']} doc passages")

if __name__ == "__main__":
    sys.exit(main())
