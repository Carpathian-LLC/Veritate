# corpus framing conventions

Veritate is byte-level (vocab=256). To train a model on something more structured than raw prose — chat turns, agent JSON, tool calls — the structure has to be visible *in the byte stream*. There are no special tokens; we use literal printable byte sequences as frames.

This document is the contract every corpus shipped under [/Veritate-Corpus](https://github.com/Carpathian-LLC/Veritate-Corpus) follows, and what training-data tooling in this repo emits.

## record separator (all modes)

The boundary between independent documents / conversations / examples is the literal ASCII bytes:

```
<|endoftext|>
```

This matches `veritate_mri/tools/jsonl_to_bin.py:DEFAULT_SEPARATOR`. The trainer's byte shuffler uses it to know where a record ends.

## autocomplete mode

No framing. The corpus is raw bytes. Records (if any) are joined with `<|endoftext|>`.

`fineweb_edu`, `wikitext103`, `pg19`, etc. are autocomplete corpora.

## chat mode

Each turn is wrapped in opening and closing frames. A conversation is one or more turn-pairs separated by single newlines; conversations are separated by `<|endoftext|>`.

```
<|user|>What is the capital of France?<|/user|>
<|assistant|>Paris.<|/assistant|>
<|user|>And of Spain?<|/user|>
<|assistant|>Madrid.<|/assistant|>
<|endoftext|>
```

Properties:

- Frames are plain printable ASCII. They will never collide with a frame inside natural prose at random because of the `<|...|>` shape — the same convention OpenAI/HF use for special tokens, but here they are literal bytes the model sees and learns.
- The trailing `<|/assistant|>` is what tells a chat-mode dashboard when to stop emitting. Inference can hard-stop on this byte sequence regardless of the model's logits.
- A conversation with one user turn and no assistant reply (i.e. the prompt at inference time) ends with `<|/user|>` and the model is expected to emit `<|assistant|>...<|/assistant|>`.

## agent JSON mode

The corpus is a stream of single-line JSON objects, one per turn, separated by newlines. Each example (one or more turns + an answer) is separated from the next by `<|endoftext|>`.

The schema matches what `veritate_mri/agent/loop.py` parses:

```
{"thought": "France is in western Europe.", "answer": "Paris."}
<|endoftext|>
{"thought": "I should compute this.", "tool_call": {"name": "calculator", "args": {"expression": "37*42"}}}
{"observation": "1554"}
{"answer": "1554."}
<|endoftext|>
```

Properties:

- One JSON object per line. No nested newlines inside the object — strings escape `\n` if needed.
- Top-level must be an object. Bare scalars (`true`, `42`) are exactly the failure mode the agent loop rejects.
- The model emits `\n` between turns inside an example, and `<|endoftext|>` ends the example.
- `tool_call` / `observation` turns are optional. A simple answer is two turns: thought, then answer. Or one turn with both keys.

## model config flag

A model declares which modes it was trained for in its `config.json`:

```json
"trained_modes": ["autocomplete", "chat"]
```

The dashboard's mode dropdown uses this to badge modes the model wasn't trained for. The agent loop reads it to decide whether to apply the JSON-schema retry policy or fall back to raw text.

The trainer writes this automatically from which corpus plugin was active. Users should not edit it by hand.

## why bytes, not tokens

Constraint #1 in `CLAUDE.md`: vocab=256, byte-level only. Every other LLM stack handles structure with a tokenizer (`<|im_start|>` becomes a single int). Here it stays as the literal 13 bytes `<|im_start|>` and the model has to learn the sequence. This is a documented cost of byte-level training. The upside is that the frames are inspectable, debuggable, and engine-agnostic — there is no tokenizer that has to agree with the model.
