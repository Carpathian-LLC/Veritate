# Model roster

Every Veritate model that currently exists, with its measured training budget. Companion to the public [sizing guide](../../veritate_mri/data/wiki/concepts/model_sizing.md); this file is the internal record of what we actually built and where each one sits against the target.

Regenerate the numbers from `models/<name>/config.json`: `bytes = batch_size * seq * n_chunks * total_steps`, `tokens = bytes / 4.55`.

## Current models (2026-07-28)

| model | box | params | bytes | tokens | tok/param | loss_mask | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `chat_200m` | mirach | 270.5M | 2.01B | 0.44B | **1.6** | none | retired, pruned to 8 ckpts |
| `chin200m` | fortis | 270.5M | 5.41B | 1.19B | **4.4** | none | wren's parent; base for the 2026-07-20 IDEA 8 SFT |
| `core_200m` | fortis | 202.7M | 4.04B | 0.89B | **4.4** | none | live pretrain, step ~79k of 164,388 |
| `wren` | mirach | 270.5M | 5.90B | 1.30B | **4.8** | none | pretrain; baseline for the instruction-following campaign |
| `wren_sft` | mirach | 270.5M | 8.18B | 1.80B | **6.7** | assistant | live capability SFT off `wren`@58500 |

Chinchilla for a 270M model is **5.41B tokens = 24.6 GB of bytes**. Nothing in this roster has reached a quarter of it.

## What this table is for

It exists because the defect it records was invisible for months. `wren` and `core_200m` both carry configs describing them as "Chinchilla-optimal, 20x params": whoever sized them computed the budget in bytes and compared it against a token target. The error is not visible in any dashboard panel, any eval, or any loss curve — an undertrained model's loss looks fine, it just converges to a worse place.

It surfaced only when a user asked why a chat-format-trained model still could not hold a conversation. The answer was that it had never read enough English. The diagnostic signature is **correct output shape containing invented words** ("drums, tsyllables, saxophones, pedals"): the model has learned the format perfectly and has not learned the lexicon. No amount of instruction tuning moves it, which is why the 2026-07-27/28 SFT campaign could fix form and never touch content.

## Corpus ceiling

Measured 2026-07-28 on `trainers/corpus/`:

| | |
| --- | --- |
| total corpus | 56 GB |
| crypto + code (not English prose) | 27 GB |
| **usable English prose** | **25.4 GB = 5.6B tokens** |

25.4 GB is **one** Chinchilla pass for a 270M model (20.6 tok/param) with nothing left over, and roughly **a tenth** of what conversational quality needs (246 GB). The binding constraint on this research is corpus size, not GPU time and not framework choice.

## Consequences for planning

- **Do not size a new run before converting.** The check is four lines of arithmetic and is now documented in the wiki page above and enforced by habit, not by code. Wiring it into the launch form as a warning is open work.
- **Parameter count is downstream of the data budget.** With 25.4 GB of prose, a 50M model reaches 200 tokens/param (46 GB needed after a modest pull) while a 270M cannot reach 20. A fully trained small model beats a quarter-trained large one at conversation.
- **The byte tax is a real multiplier on every plan here.** A byte model needs ~4.55x the positions of a word-piece model for the same text. A learned patching front-end (BLT / MegaByte style) pools bytes into patches and recovers most of that while keeping vocab 256 and no tokenizer.
- **`n_params_total` is in `config.json`** and is the honest denominator; do not use the size preset's nominal name (`200m`) as the parameter count. `core_200m` is 202.7M and `wren` is 270.5M despite both being launched from `200m`-family presets.
