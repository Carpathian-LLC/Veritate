---
title: training recipe
date: 2026-07-27
tags: [settings, training]
summary: A preset that fills the optimizer, trunk, and LR schedule fields in one click.
---

# training recipe

A shortcut that sets three other fields at once: `optimizer`, `trunk`, and `lr_schedule`.

## what it does

Picking a recipe writes values into those three form fields and stops there. Nothing named `recipe` is sent to the trainer; the three fields it filled are what get submitted. Every field stays editable after the recipe fills it, so a recipe is a starting point, not a lock.

The presets live in `TRAIN_RECIPES` in `veritate_mri/web/index.js` and are applied by `_trApplyRecipe`.

## options

| recipe | optimizer | trunk | lr schedule | good for |
|---|---|---|---|---|
| balanced | muon | dense | wsd | a first pretrain of any size |
| efficient byte-native | muon | patched | wsd | long sequences on a memory-tight box |
| long-conversation | muon | recurrent | wsd | constant cost per byte at decode |
| classic | adamw | dense | cosine | reproducing a standard transformer recipe |
| custom | unchanged | unchanged | unchanged | setting the three fields by hand |

## default

None. The select opens on `- pick -` and stays there until picked. Leaving it unpicked is normal: the trainer takes its own defaults for the three fields.

## when to change it

Pick one at the start of a scratch run, then adjust individual fields. It has no meaning on a continue run, where trunk and optimizer come from the model being continued.
