---
title: Byte tokens vs BPE tokens
date: 2026-05-05
tags: [intro, tokenization, byte-level]
summary: Why Veritate predicts the next byte instead of the next subword chunk.
---

A "token" is the unit a language model predicts. There are three common choices.

## the three options

- **Word-level.** Vocab around 50,000 English words. Predict the next word. Breaks on typos, names, code, emoji, other languages. Vocab explodes when you try to cover more.
- **BPE / subword (GPT, Llama, most modern LLMs).** Vocab around 32,000 to 128,000 chunks. "unbelievable" splits into ["un", "believ", "able"]. A trained compromise: handles new words by falling back to smaller pieces.
- **Byte-level (Veritate).** Vocab is exactly 256. Predict the next *byte*. "hello" is five tokens: 104, 101, 108, 108, 111. A UTF-8 emoji is four bytes, so four tokens.

## why byte is fine

1. **No tokenizer to ship, train, or version.** A BPE tokenizer is a separate trained artifact you have to keep in sync with the model. Byte-level has no tokenizer. ASCII is the tokenizer.
2. **No out-of-vocab.** Every possible input is representable. Code, Unicode, binary blobs, typos, made-up words, all just bytes.
3. **Smaller embedding and output matrix.** A 256-row table instead of a 50,000-row table. Real memory savings, especially at small parameter counts.
4. **Glass-box.** You can point at byte 73 and say "that is the letter I." With BPE you point at token 4127 and have to consult a vocab file to know what it means.
5. **Hardware-friendly.** A 256-wide softmax is trivial. A 50,000-wide softmax is a memory-bandwidth problem on every forward pass.

## the cost

Sequences are 4-5x longer. "The quick brown fox" is 4 BPE tokens but 19 bytes. Attention is O(seq²), so a byte-level model needs roughly 16-25x more compute per equivalent thought than a BPE model.

That is the tax. Veritate eats it on purpose, to keep the stack simple, glass-box, and tokenizer-free.

## practical implication for context

A byte-level model needs a bigger context window in absolute number to hold the same amount of *text* as a BPE model.

| window | bytes | rough word count |
|---|---|---|
| 2k | 2,000 | ~400 |
| 8k | 8,000 | ~1,500 |
| 32k | 32,000 | ~6,000 |

Plan window size in *bytes of text you want it to remember*, not in tokens.
