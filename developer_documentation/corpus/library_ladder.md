# corpus library ladder

Sizing policy for the Veritate-hosted corpus families in the library catalog
(`veritate_mri/training/sync/corpus_catalog.json`). Third-party facts corpora
(fineweb_edu, the_pile, redpajama_v2, ...) are outside this policy; they stream
from their own hosts and carry the knowledge-scale burden.

## the ladder

| family | stem | size | recommended params | content |
|---|---|---|---|---|
| chat | `chat_50mb` | 50 MB | 10M-100M | ChatML conversations: facts, definitions, math, classic prose |
| chat | `chat_500mb` | 500 MB | 100M-1.5B | same generator, 10x scale |
| chat | `chat_5gb` | 5 GB | 1.5B+ (no upper bound) | same generator, 100x scale |
| agent | `agent_15mb` | 15 MB | 10M-100M | Hermes tool-calling over the runtime toolbox (calculator, fs_read, fetch, retrieve) |
| agent | `agent_150mb` | 150 MB | 100M-1.5B | same generator, 10x scale |
| agent | `agent_1500mb` | 1.5 GB | 1.5B+ (no upper bound) | same generator, 100x scale |
| mcp | `mcp_docs` | 2 MB | any | modelcontextprotocol.io docs, native (ships in repo) |
| mcp | `mcp_15mb` | 15 MB | 10M-100M | MCP JSON-RPC transcripts + MCP Q&A + Hermes agent turns over MCP server tools |
| mcp | `mcp_150mb` | 150 MB | 100M-1.5B | same generator, 10x scale |
| mcp | `mcp_1500mb` | 1.5 GB | 1.5B+ (no upper bound) | same generator, 100x scale |

Builders: `veritate_mri/tools/build_chat_corpus.py`,
`veritate_mri/tools/build_agent_corpus.py`,
`veritate_mri/tools/build_mcp_corpus.py`. All deterministic (fixed PRNG seed),
`--target-mb` selects the tier, val is a separately assembled 2% stream.

## why the ladder caps at 5 GB / 1.5 GB

Chat, agent, and mcp corpora are behavior corpora: they teach framing (ChatML),
tool-call syntax (Hermes), and protocol shape (MCP JSON-RPC). Behavior data
requirements scale with task diversity, not with parameter count:

- A larger model learns a fixed format from fewer examples, not more;
  sample efficiency at format-learning rises with scale.
- Knowledge scale belongs to the facts corpora. Chinchilla-style token budgets
  govern pretraining volume; the behavior slice of open-weights training mixes
  is 1-10 GB regardless of model size (FLAN, Tulu-class SFT sets).
- Past the generator pool's intrinsic diversity, extra bytes replay the same
  pairs: more epochs in disguise, not more signal.

So the top tier of each family serves every model from 1.5B to 10T parameters.
A trillion-parameter model does not need a bigger agent corpus than
`agent_1500mb`; it needs the same corpus and a large facts corpus. When a tier
underperforms, the fix is widening the generator pools (more Q&A pairs, more
tool scenarios, more servers), then rebuilding: grow diversity, hold size.

## hosting

Tiers above the GitHub raw 100 MB object limit ship as zip bundles on
Carpathian COS (`format: "zip_bundle"`: one zip holding `<stem>_train.bin` +
`<stem>_val.bin`; the installer extracts both and deletes the zip).
`chat_50mb` and `agent_15mb` stay on the Veritate-Corpus GitHub repo as raw
bins. Unpublished tiers carry `coming_soon: true` in the catalog entry, which
disables install in the dashboard until the real COS URL replaces the
placeholder.
