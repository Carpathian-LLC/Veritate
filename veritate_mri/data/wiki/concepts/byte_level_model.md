---
title: what a byte-level model is
date: 2026-07-27
tags: [concepts, basics]
summary: Veritate models read and write raw bytes, with a vocabulary of 256 and no tokenizer anywhere.
---

# what a byte-level model is

Most language models read text in pieces called tokens, chosen ahead of time by a separate program called a tokenizer. A byte-level model skips that step. It reads text one byte at a time, exactly as the file is stored on disk, and predicts the next byte.

## the vocabulary is 256

A byte has 256 possible values, so the model's entire vocabulary is 256 entries. That is the `vocab` value every trainer manifest sets, and it does not change.

Compare that with a tokenizer-based model, where the vocabulary is typically 32,000 to 200,000 entries. Two things follow.

The input and output layers are tiny. In a tokenizer-based model, the embedding table and the output projection can be a large share of the total parameters. At 256 entries they are a rounding error, so nearly every parameter sits in the body of the model doing actual work.

Sequences are longer. One English word is roughly one token but four or five bytes, so a byte model needs several times more positions to cover the same text. That is why sequence length, and the trunk choices that make long sequences affordable, matter more here than in a tokenizer-based system.

## what this buys

- **No vocabulary mismatch, ever.** Any file is valid input: English, code, a language the model has never seen, an emoji, a corrupted byte. Nothing falls outside the vocabulary because there is nothing outside 256 values.
- **No tokenizer to ship, version, or match.** A checkpoint is self-contained. There is no second artifact that has to travel with it and no way for a training tokenizer and an inference tokenizer to disagree.
- **Spelling is visible.** The model sees individual characters, so tasks that depend on the letters inside a word are ordinary rather than opaque.
- **Interpretability is exact.** Every telemetry frame the dashboard renders corresponds to one byte with a known offset in the output. Confidence, surprise, and attention line up with characters, not with token boundaries a reader has to decode.

## what it costs

More positions per unit of meaning. A byte model does more forward passes to produce the same paragraph, and on a plain dense trunk attention cost grows with the square of the sequence length. The `patched` and `recurrent` trunks exist to make that affordable: one skips most global work, the other replaces attention with a fixed-size running state.
