# corpus framing conventions

Veritate is byte-level (vocab=256). Structure is encoded as literal printable byte sequences, not tokenizer special tokens. Every corpus shipped under [Veritate-Corpus](https://github.com/Carpathian-LLC/Veritate-Corpus) follows the spec on this page.

Veritate complies with two industry standards: **ChatML** for chat, **Hermes function-calling** for agent. The byte sequences below match the literal strings used by major open-weights stacks (Mistral, Hermes, Qwen, OpenChat, NousResearch), with one Veritate addition: the `<|endoftext|>` record separator.

## record separator (all modes)

Documents, conversations, and examples are separated by the literal ASCII bytes:

```
<|endoftext|>
```

Matches `veritate_mri/tools/jsonl_to_bin.py:DEFAULT_SEPARATOR`. The byte shuffler uses it to find record boundaries.

## autocomplete mode

Raw bytes. Records (if any) joined by `<|endoftext|>`. Examples: `fineweb_edu`, `wikitext103`, `pg19`.

## chat mode (ChatML)

```
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
<|im_start|>user
And of Spain?<|im_end|>
<|im_start|>assistant
Madrid.<|im_end|>
<|endoftext|>
```

Rules:

- Each turn opens with `<|im_start|>{role}\n` and closes with `<|im_end|>`. Role is `system`, `user`, `assistant`, or `tool`.
- Inference hard-stops on `<|im_end|>` regardless of logits.
- A prompt at inference time ends after the user's `<|im_end|>`; the model emits `<|im_start|>assistant\n...<|im_end|>`.

## agent mode (Hermes function-calling)

System turn lists tools. Assistant emits `<tool_call>...</tool_call>`; the host responds with a `tool` role turn carrying `<tool_response>...</tool_response>`.

```
<|im_start|>system
You may call: calculator(expression: str) -> number.<|im_end|>
<|im_start|>user
What is 37 * 42?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "calculator", "arguments": {"expression": "37*42"}}
</tool_call><|im_end|>
<|im_start|>tool
<tool_response>
{"name": "calculator", "result": 1554}
</tool_response><|im_end|>
<|im_start|>assistant
1554.<|im_end|>
<|endoftext|>
```

Rules:

- The tool-call payload is a single-line JSON object: `{"name": str, "arguments": object}`. Schema-strict; the agent loop rejects malformed JSON.
- The tool-response payload is `{"name": str, "result": any}` or `{"name": str, "error": str}`.
- A "no tool needed" answer skips the call/response pair and goes straight to `<|im_start|>assistant\nANSWER<|im_end|>`.

## capabilities flag

`models/<name>/config.json` declares trained tiers under `capabilities` (see [documentation/training/pipeline.md](../training/pipeline.md)). The dashboard and the agent loop read it to decide which modes are exposed.

## why bytes, not tokens

vocab=256, byte-level only (CLAUDE.md constraint 1). Other stacks treat `<|im_start|>` as one int; here it is the literal 12 bytes the model learns. Cost: a few extra bytes per turn. Benefit: frames are inspectable, debuggable, and tokenizer-agnostic.
