---
title: what a corpus is
date: 2026-07-27
tags: [concepts, basics, corpus]
summary: The text a model trains on, stored as a pair of raw byte files identified by a stem.
---

# what a corpus is

A corpus is the text a model learns from. On this platform it is not a folder of documents; it is a pair of files holding one long stream of raw bytes.

## the file pair

Every corpus is two files sharing a name, called the stem:

```
trainers/corpus/<stem>_train.bin
trainers/corpus/<stem>_val.bin
```

The `_train` file is what the model learns from. The `_val` file is held back and never trained on, so measuring loss against it says whether the model learned the language or memorized the training file. A rising validation loss while training loss falls is the classic signal of memorization.

Both files are raw bytes with no headers, no separators, and no metadata. The training loader picks a random offset and reads a row of the width the run needs. That is why corpus files are large and simple: the loader is doing almost nothing per row.

A corpus can also be bundled with a single trainer at `trainers/<trainer>/corpus/<stem>_train.bin`, which is what the readers scan second when resolving a stem.

## where corpora come from

Three paths, all through the dashboard.

- **The corpus library** downloads and installs a prepared corpus from the catalog. The routes are under `/corpus/library/`.
- **Teacher generation** produces text from a configured remote model and packs it into a stem, through the synthetic-data and authoring jobs under `/teacher/`.
- **A mix** blends several existing stems into one training stream in chosen proportions. `POST /corpus/mix/plan` returns a specification that the trainer's corpus argument accepts directly, along with how many bytes each source contributes and how many times each is reused.

## why mixing matters

A model trained on one kind of text is good at that kind of text. A mix sets the proportions deliberately: how much conversation, how much reference prose, how much code. A mix profile is a named target for those proportions, and the planner reports the plan before anything is trained, including a warning when a source has to be read several times over to hit its share.

Reading a small source repeatedly is how a corpus quietly turns into memorization. The `corpus_mix_max_epochs` setting caps how many times any one source may be redrawn.

## size

Corpus size is measured in bytes because the model reads bytes. Bytes consumed by a training run:

```
bytes per step = batch_size * seq * n_chunks
total bytes    = bytes per step * total_steps
```

A run whose total exceeds the corpus size is reading the same text more than once. That is normal in moderation and harmful in excess.
