# About Veritate

Veritate is Carpathian's Open Source AI inference and training engine. It's hand-written in C and assembly, tuned to work on a normal desktop CPU instead of a high-end graphics card.

Most modern AI tools assume you own, or rent, expensive specialized hardware. Veritate does not. It is built so the chip already sitting inside a regular consumer PC is enough.

## What We Are Doing

We are building a complete pipeline around that idea:

**Training** uses standard tools (PyTorch) to teach a model.
**Veritate** takes that trained model and runs it efficiently on consumer hardware.
**The MRI dashboard** lets a human watch, in plain view, what the model is doing inside, token by token.

Three rules shape every decision:

1. **Byte-level only.** The model reads raw bytes (the same units a computer already stores text in), so there is no extra translation layer and no oversized vocabulary table.
2. **A strict speed budget.** Every part of the engine has to stay under a fixed time-per-byte cost. If a feature cannot meet the budget, it is cut or redesigned.
3. **Measurement.** Every kernel (the small math routines at the core) is checked against a simple reference version, so we know the fast path produces the same answer as the slow one.

## Why We Are Doing It

AI today has a hardware problem. Bigger models keep pushing toward bigger GPUs, bigger data centers, and shorter useful lifespans for the machines underneath. The result is rising energy use and a steady stream of discarded equipment.

Carpathian's mission is to push back on that pattern. We research and build AI that is efficient enough to run on the hardware people already own. That has two direct effects:

**Less new hardware needed.** A capable CPU from the last few years is enough to run a real model. No new GPU purchase, no cloud bill, no rack of accelerators.
**Less e-waste.** When older machines stay useful, they stay out of landfills longer. Sustainability in AI is not only about the power bill while a model runs; it is also about the equipment that gets thrown away to chase the next one.

Veritate is the working proof of that direction. It shows that careful engineering, byte-level design, and a tight performance budget can put modern AI inside reach of ordinary users, on ordinary machines, with a much smaller environmental footprint.

## The Bigger Picture

Veritate is the inference half of Carpathian's research effort. Together, the training side and Veritate form an end-to-end system: train a small, well-shaped model, then run it locally on hardware that already exists. The goal is simple. Make AI that is faster to run, cheaper to own, and kinder to the planet, without giving up the quality people expect.
