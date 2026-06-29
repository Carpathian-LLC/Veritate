# What Veritate is

Veritate is an open-source AI inference and training engine. It lets you train small language models and then run them efficiently on ordinary consumer hardware. The runtime that runs trained models is hand-written in C and tuned to run on a consumer-grade CPU rather than a high-end GPU.

The platform has three parts that work together. Training runs in PyTorch and produces a byte-level model. The Veritate engine runs that trained model efficiently on consumer hardware. The MRI dashboard is a browser interface that surfaces per-token activations, attention, and decode state so you can see what the model is doing.

Veritate is byte-level. Its vocabulary is exactly 256, the set of all possible bytes, and there is no subword tokenizer. Models read raw bytes directly. This keeps the input shape identical between training and inference, so the behavior you see in the dashboard matches what the engine produces at decode time.

The mission is to reduce the hardware floor for running modern models. The goal is that a capable consumer CPU is enough to run a model, with no new GPU and no cloud account required. Older machines stay useful for longer because the engine is built for efficiency rather than for the latest accelerator.

Energy efficiency is a design constraint, not an afterthought. Every new mechanism added to the engine is given a wall-clock time budget on the target hardware, and if it exceeds that budget it is cut or redesigned. Fast-path compute kernels are required to match a plain reference implementation bit for bit before they ship, so speed is never bought at the cost of correctness.

In short, Veritate addresses the problem that running and training language models normally demands expensive GPUs and cloud services. It is a complete local pipeline: train a model, watch it learn in the dashboard, export it, and run it on the machine in front of you.
