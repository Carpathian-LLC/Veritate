# slm (selective language modeling)

## what it is

`veritate_core/plugin/slm.py`. RHO-1-class selective loss: a frozen reference model scores every token, and the student trains only on the fraction of tokens where its loss most exceeds the reference (the surprising ones). Published basis: 5-10x domain token efficiency, +6.8 percent general at 1B (RHO-1 arXiv 2404.07965).

## how it works

- `load_reference(model_dir, device)` (`slm.py:47`): loads the latest checkpoint of a canonical dense run (shape read from checkpoint `args`), frozen eval, no grads.
- `selective_loss(ref, tokens, targets, logits, keep_frac)` (`slm.py:58`): per-token CE for student (from the live logits, differentiable) and reference (no-grad); excess = detached student CE minus reference CE; keeps the top `keep_frac` tokens by excess via `kthvalue`; returns masked mean. Gradients flow only through kept student tokens; the reference stays untouched (verified).
- Wired per run via reserved flags `slm_ref` (model dir name of the reference run; empty = off) and `slm_keep` (default 0.6) in `vanilla_trainer.py`. Applied to the TRAINING loss only; validation stays unmasked so val bpb remains comparable across arms.

## dependencies

`veritate_core/model.py` (canonical Veritate for the reference), torch.

## pitfalls

- The reference must be a canonical dense checkpoint (shape keys read from ckpt args); variant-trunk references are not supported.
- Reference forward adds roughly one extra small-model forward per step (~20-30 percent step time at matched size); budget for it.
- Keep validation unmasked: a masked val loss is not comparable to other runs.
