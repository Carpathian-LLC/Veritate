---
title: Learning Rate Floor (min_lr)
summary: The lowest learning rate the run decays to by the end; the gentle finishing speed as the model settles.
tags: training, settings
---

# Learning Rate Floor (min_lr)

The learning rate is how big a step the model takes each update. **min_lr** is the floor: the lowest learning rate the schedule decays down to by the end of the run. If base_lr is your cruising speed, min_lr is how slowly you are crawling as you park.

## Why it matters

Near the end of training, small steps let the model settle precisely into a good solution instead of bouncing around it. The floor sets how gentle those final steps get. A floor that is too high keeps the model fidgety at the end; a floor near zero lets it come to a fine, stable finish.

## Good defaults

- Typical range is **1e-5 to 1e-6** (that is 0.00001 to 0.000001), roughly a tenth to a hundredth of the peak.
- It **must be less than or equal to base_lr** (the peak); the form will not accept a floor above the peak.

## Gotcha

- With a **constant** schedule there is no decay, so the floor has little effect. The floor matters most with **cosine**, **linear**, and **wsd** schedules, which actually ramp down to it.
