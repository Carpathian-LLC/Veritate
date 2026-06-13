# The engine and how inference works

Veritate models are byte-level. The model reads and writes raw bytes, with a fixed vocabulary of 256 (every possible byte value) and no subword tokenizer. Generation happens one byte at a time. This same byte-level shape is used during both training and inference, so what you see the model do in the dashboard is what it does when it runs for real.

You can run a trained model two ways: on the PyTorch engine or on the Veritate C engine. The PyTorch path runs the model directly in Python using the same code that trained it. It works for any trained model and is the default, but it is heavier to run.

The Veritate C engine is a compiled inference runtime written in C. It loads an exported model file and serves fast byte-level generation on a consumer CPU. It detects the CPU's features at load time and selects optimized kernels accordingly (for example, AVX2 or AVX-512 on x86, or the matching instructions on ARM64), falling back to a plain path when those features are absent. It ships as prebuilt binaries for several platforms.

A veritate.bin is the exported, engine-ready form of a trained model. Exporting takes a PyTorch checkpoint and converts it into the compact binary layout the C engine can memory-map and run. The export supports several quantization schemes that trade size for precision: INT8, INT4 (which packs two weights per byte), and ternary (about 1.58 bits per weight). Models trained with quantization-aware training export cleanly because the quantization was already learned during training; exporting an aggressively quantized binary from a model that was not trained that way is lossy.

In practice, you train a model, optionally export it to a veritate.bin, and then choose at chat time whether to run it with PyTorch or with the C engine. The C engine option only appears for models that have a veritate.bin export. Both paths produce the same byte-by-byte behavior; the C engine is the efficient way to run on a CPU once a model is finished.
