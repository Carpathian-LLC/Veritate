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

- Start here. Pick the recipe that matches your goal and leave the rest alone.
  - **balanced** : a solid all-round default.
  - **efficient byte-native** : leans on the settings that reach quality with fewer training bytes.
  - **long-conversation** : tuned for models that need to track long context.
  - **classic** : the conventional, well-worn setup.
  - **custom** : changes nothing. Use this once you want to hand-tune individual knobs.

## Gotcha

- Once you edit a knob by hand, you are effectively customizing. Switching back to a named recipe will overwrite your manual changes with the preset's values.
