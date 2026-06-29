# about

Veritate is an open-source AI inference and training engine. The runtime is hand-written in C and assembly, tuned to run on a consumer-grade CPU instead of a high-end GPU.

## what it is

A pipeline with three parts:

- **Training** runs in PyTorch, producing a byte-level model.
- **Veritate** runs the trained model efficiently on consumer hardware.
- **MRI dashboard** surfaces per-token activations, attention, and decode state in the browser.

## three rules

1. **Byte-level only.** vocab=256. No subword tokenizer. Models read raw bytes.
2. **Per-byte budget.** Every new mechanism gets a wall-clock budget on the target hardware. If it busts, it gets cut or redesigned.
3. **Bitwise-identical kernels.** Every fast-path kernel matches a scalar reference before shipping (preflight rule 24).

## design intent

Reduce the hardware floor for running modern models. A capable consumer CPU is enough; no new GPU or cloud account required. Older machines remain useful for longer. Inference and training share the same byte-level shape, so model behavior in the dashboard matches the engine's behavior at decode time.
