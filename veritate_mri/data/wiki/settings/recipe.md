---
title: Training Recipe
summary: A one-click preset that fills in several training knobs with a combination we have already tested in the lab.
tags: training, settings
---

# Training Recipe

A recipe is a preset. Picking one sets several of the other knobs at once (optimizer, architecture, learning rate, and so on) to a combination that we measured and know works well together. Think of it like a preset on an oven: instead of dialing temperature, time, and fan yourself, you press "roast" and it fills them all in.

## Why it matters

Most of the settings on this form interact. A good learning rate for one architecture is a bad one for another. Recipes save you from guessing: each one is a bundle that was validated end to end, so you get a sensible run without knowing the internals.

## When to change it

Start here. Pick the recipe that matches your goal and leave the rest alone.

- **balanced** : Muon optimizer + dense trunk + WSD (warmup-stable-decay) LR schedule. Solid all-round default when you don't have a specific goal in mind — this is the one to pick if you're new.
- **efficient byte-native** : Muon + patched (byte-level) trunk + WSD. Reaches target quality with fewer training bytes on this platform. Best when corpus size is the bottleneck, or you're training on messy multilingual / non-text-heavy data where subword tokenizers hurt.
- **long-conversation** : Muon + recurrent trunk + WSD. Tuned for models that need to remember far back in a conversation or long document. Slower per step than dense, but scales cleanly with context.
- **classic** : AdamW + dense trunk + cosine LR decay. The conventional textbook setup. Use it as a known baseline, or when reproducing results from older papers / other frameworks.
- **custom** : Changes nothing. Every knob stays exactly as you set it. Pick this when you want to hand-tune individual settings.

## What "custom" really means

The dropdown does not automatically switch to `custom` when you edit a knob. That's just the label — the field is a preset selector, not a mode indicator. It records "the last preset I clicked," not "the current state of the form."

So if you pick **balanced** and then manually change the optimizer to `adamw`, the dropdown will still say `balanced` even though the form no longer matches the balanced preset. Your run is now a mix, and that's fine. The only thing that matters is the actual knob values below — the recipe name is not saved into the run.

## What happens when you switch recipes

Selecting a recipe only overwrites the specific knobs that recipe defines:

- **optimizer**
- **trunk** (architecture)
- **lr_schedule**

Every other knob (batch size, precision, warmup, weight decay, sequence length, etc.) is left alone. So if you tuned batch size and then switch recipes, your batch size is preserved — only the three fields above get replaced.

**If you tuned one of those three by hand and don't want to lose your change, pick `custom` before touching anything else, or just don't re-select a recipe.** There is no undo for the overwrite once it happens; you'd have to remember and re-enter the value.
