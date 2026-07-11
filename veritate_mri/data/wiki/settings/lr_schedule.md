---
title: Learning Rate Schedule
summary: The shape of how the learning rate changes over the run, from the early ramp-up to how it eases down at the end.
tags: training, settings
---

# Learning Rate Schedule

The learning rate is how big a step the model takes each update. The schedule is the **shape of how that step size changes over the whole run**: usually it warms up, then eases down. Think of it like a car trip: pull away gently, cruise, then slow down smoothly as you arrive.

## Why it matters

Taking large steps early helps the model make fast progress; shrinking the steps later helps it settle precisely instead of overshooting the best answer. The schedule controls that arc, and a good one noticeably improves the final result.

## The choices

- **cosine** : a smooth, curved decay from peak down to the floor. A strong, popular default.
- **linear** : a straight-line decay from peak to floor.
- **constant** : holds the learning rate flat (no decay).
- **wsd** : warmup, then a stable flat middle, then a decay tail at the end. Handy when you may want to stop early, because the model is usable before the final decay.

## When to change it

- Use **cosine** if unsure; it is a safe default.
- Use **wsd** when you want the option to cut a run short and still get a decent model.
- Use **constant** mainly for short experiments or debugging, not final runs.
