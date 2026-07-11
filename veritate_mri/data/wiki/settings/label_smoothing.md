---
title: Label Smoothing
summary: Tells the model not to be 100 percent certain of every answer, which curbs overconfidence and can help it generalize.
tags: training, settings
---

# Label Smoothing

Normally, training tells the model the "right answer" is exactly correct and everything else is exactly wrong. Label smoothing softens that: it teaches the model to be, say, 95 percent sure of the right answer instead of 100 percent, leaving a little doubt spread over the alternatives. It is a nudge against overconfidence.

## Why it matters

A model trained to be absolutely certain tends to become brittle and cocky: it makes very confident predictions even when it should hesitate, and that often hurts how well it does on new data. A little built-in humility usually calibrates it better and can improve generalization.

## Good defaults

- **0** = off. The model is trained toward full certainty. This is a fine default.
- **0.05 to 0.1** = a light-to-moderate smoothing that reduces overconfidence.

## When to change it

- Turn it on (try **0.05**) if the model seems overconfident or over-fits the training data.
- Leave it at **0** when you want the model to commit hard to answers, or you are reproducing a setup that did not use it.

## Gotcha

- Too much smoothing makes the model wishy-washy and can blur its predictions, so keep it small. Values above ~0.1 are rarely helpful here.
