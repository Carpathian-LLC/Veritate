---
title: "Build 1: initial commit and base features"
date: 2026-05-05
tags: [build, engine, mri, format, plugins]
summary: Initial commit. INT8 byte-level inference engine plus the live MRI dashboard.
---

## versions

- build: 1
- engine: v2.0.0
- mri: v0.1.0
- format: v0.1.0
- plugins: v0.1.0

## what changed

This is the baseline. Everything in the repo as of this build is "build 1." The pieces:

- **Engine.** A small inference program written in C and assembly, targeting modern x86 CPUs. It loads a trained model and predicts text one byte at a time.
- **MRI dashboard.** A local web app that shows what the model is doing while it generates: which neurons fire, how attention moves, where confidence comes from. Open it at the URL the server prints when it starts.
- **Format.** The on-disk layout for trained models, settings, and per-step training artifacts. Set once, here.
- **Plugins.** Self-contained add-ons under `plugins/`. They cover trainers, corpus builders, and experiments.

There is nothing to compare against yet. Future build notes will list deltas from this one.

## what you need to do

Nothing. Pull and keep working.

If this is your first time:

1. Run `pwsh veritate_engine/build/setup.ps1` once to set up the build toolchain.
2. Run `veritate_engine/build/build.bat` to compile the engine.
3. Start the MRI server with `python veritate_mri/app.py`. Open the URL it prints.

## known issues

- The engine is x86-only at this build. ARM64 support is planned, not present.
- Long contexts can drift past the 0.1 ms / byte target on some workloads. Tightening that is on the list.
