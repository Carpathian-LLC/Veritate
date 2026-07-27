# Chat corpus sources

Open conversation/instruction datasets evaluated for the byte-level chat model, and the corpora built from them. Builder: [veritate_mri/tools/build_chat_corpus.py](../../veritate_mri/tools/build_chat_corpus.py). Output bins in `trainers/corpus/`. Evaluated 2026-07; licenses and sizes as of that date.

## Byte template (all chat bins)

Every conversation renders as UTF-8 bytes with these tags, conversations separated by a single `0x00`:

```
<|system|>\n{system}\n          (optional, first only)
<|user|>\n{user}\n<|assistant|>\n{assistant}\n<|end|>\n   (repeated per turn)
```

Per-conversation cap 16 KiB; exact-dedup by SHA-1; every 50th kept conversation goes to val. Conversations whose rendered bytes contain a literal `0x00` are rejected, so NUL is delimiter-only by construction.

## Ranked sources

| rank | dataset | license | size | multi-turn | quality evidence | verdict |
|---|---|---|---|---|---|---|
| 1 | [HuggingFaceTB/smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) (`all`) | Apache 2.0 | ~1.1M convs | high (magpie-ultra MT, systemchats, everyday-conversations) | trained SmolLM2 instruct; HF ablations at 1.7B/7B beat Magpie-Pro and OpenHermes mixes on IFEval/MT-Bench | **used: chat_v1** |
| 2 | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | ODC-BY-1.0 collection; one CC-BY-NC subset (`no_robots`, 9,500 rows) filtered out at build | 939K convs, 1.4 GB parquet | up to 294 messages; WildChat GPT-4 100K real multi-turn | Tulu-3-8B/70B surpass Llama-3.1-Instruct at size ([paper](https://arxiv.org/abs/2411.15124)); decontaminated vs eval suite | **used: chat_v2** |
| 3 | [allenai/soda](https://huggingface.co/datasets/allenai/soda) | CC-BY-4.0 | 1.5M dialogues, 11M utterances | 100%, casual social dialogue | EMNLP 2023: rated above prior chitchat corpora on naturalness/specificity/consistency | **next: plain-dialogue gap** (casual chat, not instruction-QA; the one large permissive non-QA source) |
| 4 | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | 208K dialogues | 100% | trained Zephyr-7B-beta; truecased + assistant-disclaimer filtered | later: good but ChatGPT-style instruction-QA, overlaps smoltalk register |
| 5 | [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache 2.0 | 70.6K trees, 208K messages | 100%, human-written | human data; 40+ languages, only ~85K English messages | later: small English yield; tree schema needs a flattening pass |
| 6 | [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) | Apache 2.0 new subsets; inherited subsets per-origin | 4.8M mid + 25-set SFT mix | high | trained SmolLM3-3B | later: built for hybrid think/no_think reasoning models; per-subset license audit needed |
| 7 | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | ODC-BY, not gated | 838K real-user convs | high, real users | real distribution of user asks | skip standalone: toxic content present (moderation flags, minimal filtering); curated GPT-4 100K subset already arrives via tulu-3 |
| 8 | Magpie families ([Magpie-Align](https://huggingface.co/Magpie-Align)) | Llama 3.x Community License | 300K-1M per set | MT variants exist | Magpie-Pro competitive in HF ablations | skip on license (use policy + naming restrictions, not permissive); magpie-ultra already inside smoltalk |
| 9 | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | none stated ("FAFO"); OpenAI-derived subsets, no commercial license possible | 1M samples | partial | trained OpenHermes-2.5-Mistral-7B | skip on license; 100K sample already inside smoltalk |
| 10 | [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | custom research license, no redistribution, click-through gate | 1M convs | high | real arena traffic | skip: license fails permissive bar |
| 11 | [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | CC-BY-NC-4.0 | 10K | low | human-written SFT gold | skip: non-commercial; also filtered out of chat_v2 |
| 12 | [HuggingFaceTB/everyday-conversations-llama3.1-2k](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) | Apache 2.0 | 2.2K | 100% | smoltalk component | covered by chat_v1 |

No source used required authentication or gating; lmsys-chat-1m is gated and was skipped anyway on license.

## Built corpora

| bin | source | train | val | conversations | notes |
|---|---|---|---|---|---|
| `chat_v1_{train,val}.bin` | smoltalk:all | 2,147,491,183 B (cap hit) | 43.5 MB | 597,580 kept of 602,465 scanned | NUL scan clean (0 in-content) |
| `chat_v2_{train,val}.bin` | tulu-3-sft-mixture minus `ai2-adapt-dev/no_robots_converted` | 2,029,136,902 B (full dataset, under cap) | 41.9 MB | 902,049 kept of 939,343 scanned (9,500 source-filtered, 10,843 dup, 16,951 shape/NUL) | verified: template spot-check, UTF-8 strict on 2,500 sampled convs, all 902,049 NULs delimiter-only |

Overlap between the two is minimal: smoltalk is magpie-ultra + smol-* sets + a 100K OpenHermes sample; tulu-3 is FLAN v2, OASST Guanaco, WildChat, persona math/code/IF, Aya, Evol CodeAlpaca, CoCoNot, WildGuardMix. Shared territory is only the math domain (MetaMathQA/NuminaMath-CoT vs NuminaMath-TIR).

## Local teacher generation

Measured 532 B/s (~46 MB/day) and contends with the training GPU. Reserved for gap-filling only: persona/style consistency sets, tool-use turns, and targeted fixes for weaknesses eval surfaces. Bulk conversation data comes from the downloads above.

## Recommended mix, 80M byte-level chat run

Structured formats must appear from Stage A (model cannot discover formats it never saw; [agent_roe.md](../agents/agent_roe.md) training invariants).

| phase | mix | reasoning |
|---|---|---|
| pretrain | 70% fineweb_edu, 15% openwebtext, 10% chat (v1+v2), 5% python | knowledge base from edu web; chat template present from step 0; small code share for structure |
| midtrain (anneal) | 45% chat (v1+v2), 45% fineweb_edu, 10% openwebtext | shift distribution toward target task while annealing LR, per SmolLM2/Tulu-style two-phase data schedules |
| SFT | ~50/50 chat_v1 : chat_v2, plus SODA once built for casual-dialogue register, teacher-generated gap fill <5% | 4.2 GB total chat allows multi-epoch SFT at 80M without memorization; v1 carries style/everyday coverage, v2 carries knowledge/precision/safety refusals |
