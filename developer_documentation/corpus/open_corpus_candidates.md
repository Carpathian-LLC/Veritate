# Open corpus candidates for the 200M byte-level base pretrain

Research pass, 2026-07-27. Read-only survey, no downloads performed. Goal: find additions to
the existing local corpus (fineweb_edu 5.34 GB accepted; openwebtext10g, wikitext103, skills,
enwik8 rejected) for a ~4.04 GB Chinchilla-optimal byte-level pretrain. Priorities per the task:
(1) clean, (2) contemporary US American English, (3) dialogue density is a bonus, (4) permissive
license, with real license named even though this is non-public research.

Every factual claim below is sourced. Anything I could not verify from a primary source (dataset
card, paper, or repo) is marked **UNVERIFIED**.

---

## Bottom line

**Nothing found beats fineweb_edu as a clean, US-dominant bulk source at the 4-5 GB scale.**
The two candidates close enough to be worth comparing (C4, DCLM-baseline) either measure *less*
American-English dominance than our own measurement of fineweb_edu (80.3%), or have no published
dialect measurement at all. See "Bulk web-text alternatives" below. fineweb_edu stays as the bulk.

**Our weak axis (clean, permissive, human-written, non-distilled dialogue) has no single strong
answer either.** Nothing here is a drop-in fineweb_edu-sized dialogue corpus. The realistic move is
several small-to-medium human-written dialogue/transcript sources blended in as a minority slice
(a few hundred MB total), not a single bulk replacement. Ranked list below.

---

## Ranked candidates (by expected improvement to our corpus, not popularity)

| Rank | Candidate | Size (raw text) | License | Verdict |
|---|---|---|---|---|
| 1 | `common-pile/usgpo` (US Government Publishing Office corpus, via Common Pile v0.1) | 74.5 GB raw (2.73M docs): filterable to a much smaller US-specific slice | Public domain (US govt work) | **Investigate further**: contains real Q&A congressional hearing transcripts (genuine dialogue) mixed with pure expository federal-register/budget text; needs a document-type filter pass before use |
| 2 | `common-pile/stackexchange` (via Common Pile v0.1) | 103.7 GB raw, ~33.4M docs | CC BY-SA 3.0/4.0 per-post | **Add (filtered slice)**: human-authored Q&A, contemporary, mostly US/English-speaking tech-culture register; not natural conversation but real turn-taking (question → answer → comment) |
| 3 | Supreme Court Oral Arguments Corpus (ConvoKit / Oyez) | ~1.8M utterances, 8,300 arguments | CC BY-NC (compiled corpus); underlying transcripts are US govt public domain | **Add (small slice)**: genuine unscripted human dialogue (Justices ↔ advocates), contemporary US English register is as authentic as it gets; NC clause is irrelevant for non-public research |
| 4 | `common-pile/project_gutenberg` (Common Pile's filtered PG slice) | part of 8 TB whole; PG-only slice smaller, need to pull | Public domain (pre-1929 books) + openly-licensed | **Investigate further**: same content problem as our rejected wikitext103: PG skews British/19th-century. Must hand-pick American authors (Twain, Melville, Hawthorne, Douglass, Wharton, London, etc.) rather than take the whole slice |
| 5 | `HuggingFaceH4/no_robots` | 17 MB, 10,000 examples | CC BY-NC 4.0 | **Add (tiny, but genuinely clean)**: human-annotator-written instructions/responses, explicitly *not* distilled from a proprietary LLM (built to avoid exactly that), contemporary US English (OpenAI-style annotator pool). Too small to matter for bulk but a very clean dialogue-adjacent seasoning |
| 6 | `OpenAssistant/oasst1` (English subset) | 161K messages total across 35 languages; English subset is a fraction of that: need to pull and measure | Apache 2.0 | **Investigate further**: real human-written multi-turn conversation trees (13,500 volunteers), not LLM-distilled, but written in "assistant" register (people role-playing a helpful AI) rather than natural conversation; check for residual role-tag scaffolding before use |
| 7 | Cornell Movie-Dialogs Corpus | ~10-20 MB extracted dialogue text (304,713 utterances) | Original: research/education use only; one HF mirror re-labels MIT (questionable) | **Investigate further / small seasoning only**: scripted (fictional) dialogue from mostly Hollywood/American films, good register match, but known `<u></u>` HTML-tag leakage and field-separator artifacts need cleanup; too small to move a 4 GB corpus, use only as a dialogue-density seasoning at <1% of total |
| 8 | `PleIAs/US-PD-Newspapers` / `dell-research-harvard/AmericanStories` | 410 GB raw (PleIAs) / smaller curated OCR-corrected subset (American Stories) | CC0 (public domain, Chronicling America/LoC) | **Skip for this project**: genuinely US and public domain, but content is 1780-1960 newspaper OCR: expository, period-register English, not contemporary, and PleIAs' own docs say quality is well below the American Stories curated cut |
| 9 | `common-pile/youtube` (Creative Commons YouTube transcripts) | 21.5 GB raw, 1.13M videos, Whisper-transcribed | CC BY (per-video, with known "license laundering" mislabeling risk flagged by the authors themselves) | **Skip / high risk**: genuinely contemporary spoken American-inflected English and some conversational registers (vlogs, interviews), but ASR-transcribed text is exactly the kind of noisy, unpunctuated, disfluency-laden text our "clean" requirement (#1) is built to reject; would need heavy filtering to even evaluate |
| 10 | DCLM-baseline (`mlfoundations/dclm-baseline-1.0`) | 4T tokens / ~large fraction of a 10TB+ corpus, slice-able | CC BY 4.0 | **Skip as fineweb_edu replacement**: classifier-filtered (fastText on OpenHermes2.5 + r/ExplainLikeImFive) similarly to fineweb-edu, but no published US/UK dialect measurement exists; not demonstrated to beat our measured 80.3% American on fineweb_edu, so no reason to swap |
| 11 | C4 (`allenai/c4`, en split) | ~305 GB (Common Crawl derived) | Unclear/no formal license: CC's own terms + "fair use" argument, not a real permissive license | **Skip**: independently measured at ~70% American-only-spelling documents, i.e. *less* US-dominant than our fineweb_edu (80.3%), and heuristic-filtered only (no educational classifier), so strictly worse on both axes that matter |
| 12 | `PleIAs/common_corpus` | 4.49 TB / 2.27T tokens | Mixed permissive (public domain, CC, MIT, open-government) per-source | **Skip as a bulk pull**: "predominantly expository prose," multilingual, includes UK Hansard-adjacent European-parliament content; useful only as a menu to cherry-pick already-covered sources (Gutenberg, US newspapers) from, not a corpus to ingest whole |
|: | `common-pile/uk_hansard` | large | Open Parliament License (UK) | **Explicitly avoid**: British parliamentary English by construction; flagging so nobody accidentally pulls this from a Common Pile bulk download |
|: | NPS Chat Corpus | 10,567 utterances | Non-commercial, research/education only, derivative-compilation copyright unclear | **Skip**: too small to matter and license is more restrictive than useful even for internal research |
|: | DailyDialog | ~13,000 conversations, single-digit MB | CC BY-NC-SA 4.0 | **Skip**: written by/for English-language learners on ESL practice sites; register is simplified/non-native, risks teaching stilted phrasing rather than natural conversation |
|: | StoryCorps oral history archive | unknown, large | **UNVERIFIED**: no clear open license found; archive access framed as "supports researchers" not an open dataset release | **Skip until license clarified**: exactly the kind of contemporary American human dialogue we want (oral history interviews), but I could not verify any redistributable license; do not use without confirming terms directly with StoryCorps/Library of Congress |
|: | Reddit (Pushshift or successor) | large | Broken/contested: Reddit revoked Pushshift's API access in 2023 over CFAA concerns; successor "Arctic Shift" is an unofficial academic mirror | **Skip**: human-written and often dialogic, but licensing status is actively contested, not "permissive" by any formal definition, and content ownership remains with individual Reddit users |
|: | Switchboard / Fisher (LDC) | large | LDC restrictive, membership/fee-gated | **Skip**: most naturalistic phone-conversation American English available, but access is paywalled and license terms are the opposite of permissive |
|: | Public-domain plays (Gutenberg "Plays/Films/Dramas" shelf) | small | Public domain | **Skip**: dominated by Shaw, Wilde, Ibsen, Chekhov, Shakespeare: British/Irish/European playwrights, directly reintroduces the exact British-spelling problem that sank wikitext103 |

---

## Detail: bulk web-text alternatives to fineweb_edu

| Dataset | Size | License | Quality signal | US-English signal |
|---|---|---|---|---|
| fineweb_edu (ours, already measured) | 5.34 GB (local slice) | ODC-BY 1.0 (via parent FineWeb) | 0.992 distinct 5-gram, 0.91% boilerplate: our own measurement | 80.3% American: our own measurement |
| C4 (en) | ~305 GB full / slice-able | No single formal license; Common Crawl terms | Heuristic filtering only (line-length, stopword ratio, no classifier): [Dodge et al. 2021](https://maartensap.com/pdfs/dodge2021documentingC4.pdf) | ~70% American-only-spelling documents: [Chari & Lin, "On the Effects of Regional Spelling Conventions in Retrieval Models," 2023](https://arxiv.org/pdf/2308.00480) |
| DCLM-baseline | 4T tokens (full); parquet/jsonl, slice-able | CC BY 4.0: [dataset card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | fastText classifier trained on OpenHermes2.5 + r/ExplainLikeImFive instruction-style positives, plus RefinedWeb-style heuristics: [DataComp-LM paper](https://arxiv.org/html/2406.11794v1) | **UNVERIFIED**: no published US/UK spelling measurement found in the paper or card |
| DCLM-Edu (`HuggingFaceTB/dclm-edu`) | filtered subset of DCLM, score>2 on fineweb-edu classifier | inherits DCLM's CC BY 4.0 | Explicitly filtered with the *same* fineweb-edu classifier, so quality signal should track fineweb_edu: [dataset card](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu) | **UNVERIFIED**, same caveat as DCLM-baseline |
| FineWeb (non-edu) | 18.5T tokens / 44 TB | ODC-BY 1.0: [dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | MinHash dedup, heuristic filters, no educational classifier: [MarkTechPost summary](https://www.marktechpost.com/2024/06/03/huggingface-releases-%F0%9F%8D%B7-fineweb-a-new-large-scale-15-trillion-tokens-44tb-disk-space-dataset-for-llm-pretraining/) | Same underlying Common Crawl as fineweb_edu but *without* the educational filter that likely helped push fineweb_edu's American % up (edu content skews US institutions); no reason to expect improvement |

**Conclusion on bulk**: fineweb_edu's own measured 80.3% American and its educational-quality classifier are not beaten by anything found. C4 is measurably worse on the dialect axis. DCLM-baseline is a plausible research target but nobody has published the dialect measurement: it would need the same local 5-gram/boilerplate/American-spelling measurement pipeline run against it before any claim could be made, and given it's the same Common-Crawl-derived family with a different classifier, there's no strong prior it would beat fineweb_edu rather than roughly match it.

---

## Detail: dialogue-specific candidates

Our stated weak axis. Findings, plainly:

- **No clean, permissively-licensed, non-LLM-distilled corpus of natural human conversation exists
  at anywhere near the 1+ GB scale.** Everything found is either small (movie dialogue, chat logs,
  ConvoKit corpora: tens of MB), license-restricted (LDC Switchboard/Fisher, NPS Chat), licensing-
  contested (Reddit), or not natural conversation (StackExchange Q&A, congressional hearing
  testimony, which is dialogic but adversarial/formal, not conversational).
- **`common-pile/usgpo`'s congressional hearing transcripts** are the largest pool of genuinely
  dialogic, contemporary, US-government-produced (hence public domain, zero ambiguity) text found.
  The catch: the 74.5 GB raw dataset bundles hearings together with Federal Register notices, budget
  reports, and economic indicators, which are pure expository bureaucratic prose. Extracting just the
  hearing-transcript document type would need a filtering pass on the metadata (not yet done here;
  flagged as investigate-further rather than a straight add).
- **Supreme Court oral arguments** (Oyez, ~1.8M utterances) are the cleanest example of real,
  unscripted, turn-taking human dialogue in a public-domain-sourced US-government context. Small
  by corpus-mass standards but extremely high signal for turn-taking and register.
- **StackExchange** (via Common Pile, CC BY-SA, 103.7 GB raw) is not conversational in the "chat"
  sense, but it is real human question→answer→comment threading, contemporary, mostly US/English-
  speaking tech-culture register, and comes with a solid open license. Best available *volume* of
  human-authored back-and-forth text found in this search.
- **`HuggingFaceH4/no_robots`** and **`OpenAssistant/oasst1`** are the two candidates explicitly
  built to be human-written rather than LLM-distilled: worth the "investigate further" tag
  specifically because they satisfy the research-framing constraint (no Anthropic/OpenAI-model text
  laundered into the corpus) that ShareGPT, LMSYS-Chat-1M, and Anthropic HH-RLHF all fail. Both are
  small next to a 4 GB target, and OASST is written in "AI-assistant" register rather than natural
  conversation, so treat both as flavor, not bulk.
- **Movie/TV dialogue and public-domain plays** are the classic go-to for "dialogue corpus" but both
  have real defects here: Cornell Movie-Dialogs has markup-tag leakage needing cleanup and is tiny;
  public-domain plays on Gutenberg are dominated by British/Irish/European playwrights (Shaw, Wilde,
  Ibsen, Chekhov), which reproduces the exact British-register problem that sank wikitext103.
- **StoryCorps** (oral history interviews, genuinely contemporary American, genuinely conversational)
  is the best-fit *content* found for this brief and the one most worth a follow-up: no license could
  be verified from public documentation, so it is marked UNVERIFIED/skip rather than assumed usable.

---

## Sources consulted

- [Cornell Movie-Dialogs Corpus (HF)](https://huggingface.co/datasets/cornell-movie-dialog/cornell_movie_dialog)
- [Cornell Movie-Dialogs Corpus (official page)](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
- [DCLM-baseline-1.0 (HF)](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
- [DataComp-LM paper](https://arxiv.org/html/2406.11794v1)
- [HuggingFaceTB/dclm-edu (HF)](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu)
- [sedthh/gutenberg_english (HF)](https://huggingface.co/datasets/sedthh/gutenberg_english)
- [common-pile/project_gutenberg (HF)](https://huggingface.co/datasets/common-pile/project_gutenberg)
- [PG-19 GitHub](https://github.com/google-deepmind/pg19)
- [PG-19 (HF)](https://huggingface.co/datasets/deepmind/pg19)
- [OpenAssistant/oasst1 (HF)](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [OpenAssistant/oasst1 LICENSE](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/LICENSE)
- [Supreme Court Oral Arguments Corpus (ConvoKit)](https://convokit.cornell.edu/documentation/supreme.html)
- [walkerdb/supreme_court_transcripts (GitHub)](https://github.com/walkerdb/supreme_court_transcripts)
- [The Common Pile v0.1 (EleutherAI blog)](https://blog.eleuther.ai/common-pile/)
- [The Common Pile v0.1 (arXiv)](https://arxiv.org/abs/2506.05209)
- [common-pile/usgpo (HF)](https://huggingface.co/datasets/common-pile/usgpo)
- [common-pile/youtube (HF)](https://huggingface.co/datasets/common-pile/youtube)
- [common-pile/stackexchange (HF)](https://huggingface.co/datasets/common-pile/stackexchange)
- [common-pile/uk_hansard (HF)](https://huggingface.co/datasets/common-pile/uk_hansard)
- [PleIAs/US-PD-Newspapers (HF)](https://huggingface.co/datasets/PleIAs/US-PD-Newspapers)
- [dell-research-harvard/AmericanStories (HF)](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)
- [American Stories paper (arXiv)](https://arxiv.org/abs/2308.12477)
- [PleIAs/English-PD (HF)](https://huggingface.co/datasets/PleIAs/English-PD)
- [PleIAs/common_corpus (HF)](https://huggingface.co/datasets/PleIAs/common_corpus)
- [Standard Ebooks](https://standardebooks.org/)
- [Standard Ebooks (Wikipedia)](https://en.wikipedia.org/wiki/Standard_Ebooks)
- [Standardized Project Gutenberg Corpus (MDPI)](https://www.mdpi.com/1099-4300/22/1/126)
- [NPS Chat Corpus reader (NLTK)](https://www.nltk.org/_modules/nltk/corpus/reader/nps_chat.html)
- [NPS Chat Corpus description (PDF)](https://bond-lab.github.io/Corpus-Linguistics/pdf/corpora/2015-NPS-chat.pdf)
- [DailyDialog (HF)](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)
- [DailyDialog paper (arXiv)](https://arxiv.org/pdf/1710.03957)
- [StoryCorps Archive: About](https://archive.storycorps.org/about/)
- [StoryCorps (Wikipedia)](https://en.wikipedia.org/wiki/StoryCorps)
- [On the Effects of Regional Spelling Conventions in Retrieval Models (C4 American-spelling measurement)](https://arxiv.org/pdf/2308.00480)
- [HuggingFaceH4/no_robots (HF)](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
- [Congressional Record parsed speeches (Stanford)](https://data.stanford.edu/congress_text)
- [ConvoKit (GitHub)](https://github.com/CornellNLP/ConvoKit)
- [ConvoKit datasets list](https://convokit.cornell.edu/datasets.html)
- [Reddit API controversy (Wikipedia)](https://en.wikipedia.org/wiki/Reddit_API_controversy)
- [FineWeb (HF)](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [FineWeb 15T-token release summary (MarkTechPost)](https://www.marktechpost.com/2024/06/03/huggingface-releases-%F0%9F%8D%B7-fineweb-a-new-large-scale-15-trillion-tokens-44tb-disk-space-dataset-for-llm-pretraining/)
