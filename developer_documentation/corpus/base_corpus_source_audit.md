# Base pretraining corpus source audit (200M byte-level model)

Read-only measurement pass over the five open-source `.bin` corpora at
`trainers/corpus/` that are candidates for the ~4.0–5.4 GB (20–27 bytes/param,
Chinchilla-optimal for 200M params) base pretraining byte budget. No files
were edited, no training was run. This report is reproducible by re-running
the sampling method below with the same seed.

## Sampling method (identical for all five corpora)

- Python, `open(path, "rb")`, seek + read, `bytes.decode("utf-8", errors="replace")`.
- Per corpus: **236 samples of 8,192 bytes each ≈ 1.93 MB sampled**, `random.seed(1234)`.
  - 8 samples forced into the **first 1% of the file** (head check).
  - 8 samples forced into the **last 1% of the file** (tail check).
  - 220 samples at uniformly random offsets across the *entire* file.
- This is a small fraction of each file (0.02–2.0%, see table below): treat
  rare-event numbers (non-English fraction, PII rate) as **lower bounds**,
  not exact corpus-wide rates. This is discussed explicitly in §5 and §9.
- All file sizes were verified against `os.path.getsize()` before sampling
  (no size assumptions).

| Corpus | File size (B) | Sampled bytes | Sampled fraction |
|---|---:|---:|---:|
| fineweb_edu_train.bin | 5,341,866,419 | 1,933,314 | 0.0362% |
| openwebtext10g_train.bin | 9,900,006,535 | 1,933,319 | 0.0195% |
| wikitext103_train.bin | 524,289,022 | 1,933,313 | 0.3688% |
| skills_train.bin | 245,763,258 | 1,933,312 | 0.7867% |
| enwik8_train.bin | 95,000,000 | 1,933,316 | 2.0351% |

## Headline table

| Corpus | Distinct 5-gram ratio | Boilerplate line % | Markup-hit sample % | Non-English % (heuristic) | US:UK spelling ratio | Dialogue % | Dup (window) % | Email/MB | Phone/MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu_train.bin | **0.992** | 0.91% | 11.4% | 0.0% | 375:92 (4.08:1, 80.3% US) | 25.7% | 0.00% | 3.25 | 9.76 |
| openwebtext10g_train.bin | 0.983 | 2.44% | 18.6% | 0.0%* | 267:100 (2.67:1, 72.7% US) | 32.7% | 0.02% | 16.81 | 6.51 |
| wikitext103_train.bin | 0.989 | **0.07%** | **0.42%** | 0.0% | 182:163 (**1.12:1, 52.7% US**) | 24.7% | 0.00% | 0.00 | 0.00 |
| skills_train.bin | **0.780 (below 0.90 floor)** | 0.24% | 14.8% | 0.0% | 236:93 (2.54:1, 71.7% US) | 19.2% | 0.01% | 5.97 | 4.88 |
| enwik8_train.bin | 0.953 | 1.21% | **98.3%** | 1.7% | 578:124 (4.66:1, 82.3% US, **confounded: see §6**) | 3.2% | 0.00% | 0.00 | 1.08 |

\* Broad random sample measured 0/236 (0.0%); a targeted deeper scan of the
tail found evidence this undercounts the true rate: see §5.

Quality floor stated in the brief: distinct 5-gram ratio ≥ 0.90. **Only
`skills_train.bin` fails this floor outright** (0.780).

---

## 1. Sampling method

Covered above. Reproducibility note: `random.seed(1234)`, offsets generated
by `sample_offsets(filesize, n_random=220, chunk=8192)`: 8 head-forced + 8
tail-forced + 220 uniform-random offsets, in that fixed order, per corpus.

## 2. Variety (distinct word 5-gram ratio)

Word tokens extracted with `[A-Za-z']+`, 5-grams built as sliding tuples
over the concatenated sample text (236 chunks joined with `\n`), ratio =
`len(set(5-grams)) / len(5-grams)`.

| Corpus | 5-grams counted | Distinct ratio |
|---|---:|---:|
| fineweb_edu_train.bin | 311,101 | 0.9922 |
| openwebtext10g_train.bin | 319,429 | 0.9830 |
| wikitext103_train.bin | 299,699 | 0.9892 |
| skills_train.bin | 310,012 | **0.7798** |
| enwik8_train.bin | 284,258 | 0.9530 |

`skills_train.bin` is the outlier and the reason is structural, not noise:
see §3 below. It is 21 points under the stated 0.90 floor.

## 3. Boilerplate and junk

Line-level heuristic: a line is flagged boilerplate if it matches any of a
pattern set (`skip to content`, `all rights reserved`, `posted by`, `share
this`, `cookie`, `comments (n)`, `subscribe`, `sign up`, `log in`, `terms of
service/use`, `privacy policy`, `click here`, `related articles`, `read
more`, `advertisement`, bare nav words like `home`/`about`/`contact`/`menu`,
timestamp-only lines) **or** is a short (≤8 word) ALL-CAPS line **or** is a
line consisting only of digits/punctuation (a bare timestamp).

| Corpus | Non-blank lines sampled | Boilerplate lines | Fraction |
|---|---:|---:|---:|
| fineweb_edu_train.bin | 9,075 | 83 | 0.91% |
| openwebtext10g_train.bin | 7,977 | 195 | 2.44% |
| wikitext103_train.bin | 4,076 | 3 | 0.07% |
| skills_train.bin | 19,411 | 46 | 0.24% |
| enwik8_train.bin | 17,122 | 208 | 1.21% |

10 real examples per corpus (verbatim, truncated to ~120 chars):

**fineweb_edu_train.bin**
```
This is part of a larger conversation Americans are having about play. Parents bobble between a nostalgia-infused yearni
ABDOMINAL WALL DEFECTS
(EXOMPHALOS & GASTROSCHISIS)
But this isn't a one-way thing. It won't just be about us sharing what we've found. There's also the MyNHS members secti
PINK EYE (CONJUNCTIVITIS)
WHAT IS FENCING?
IS IT SAFE?
HOW DOES OUR PROGRAM WORK?
HOW LONG IS THE PROGRAM?
HOW MUCH DOES THE PROGRAM COST?
```
(Reads as FAQ-header chrome from health/education sites: real junk, but a
small fraction, and note fineweb_edu's own boilerplate-stripping pipeline
already removes most of the worse chrome upstream.)

**openwebtext10g_train.bin**
```
LONGEST SEC WINLESS STREAKS
Read or Share this story: http://cjky.it/1syOLzE
[On Detroit's RiverWalk, plans are being made for a park that includes a 'huge sandbox designed to feel like a beach.' T
Not a member of Pastebin yet? Sign Up , it unlocks many cool features!
Advertisement
ALL THESE WORLDS
ARE YOURS EXCEPT
EUROPA
ATTEMPT NO
LANDING THERE
```
(The last five lines are a Pastebin/Reddit repost of the 2001: A Space
Odyssey epigraph broken into one-line-per-caps-phrase: a real duplication-
prone pattern typical of Reddit-sourced scrapes.)

**wikitext103_train.bin**
```
My Lord , out of the love I bear to some of your friends , I have a care of your preservation . Therefore I would advise
= WASP @-@ 15 =
VOLUME 1
```
(Only 3 hits in the whole sample: cleanest corpus by this metric by a wide
margin. The `@-@` is a wikitext-103 tokenization artifact for hyphens, not
prose junk, but is a markup-adjacent leak: counted in §4 instead.)

**skills_train.bin**
```
context: South African satellite mosaics, the Taj Mahal sinking, and Dubai's impressive night lights We update our syndi
context: To submit a registration form along with a purchase order, click here. We also accept registration forms and sc
context: And though the experience itself was solemnly memorialized in the years that had followed the end of the war, t
Fill the blank using the context: As the Times put it on November 11, ____, Armistice Day was not just a commemoration o
context: However, choosing to go to therapy and choosing a therapist are both difficult decisions. It's crucial that you
context: Subscribe to Midwifery Today Magazine Neonatal Resuscitation with Intact Umbilical Cord In many birth places, i
context: Read more » Animal Dander Allergy WHAT ARE ANIMAL ALLERGENS? Cats, dogs and other mammals produce proteins in t
context: It also consumes methane, a greenhouse gas far more powerful than carbon dioxide. Leave the manure to decompose
context: Industry 4.0 largely stands for networking and the intelligence of machines. It is hardly an argument that this
context: Instant Lessons PowerPoint Hospitality - Food Preparation Food 2 Have a look at the slide set below "Making sal
```
This is the important finding for this corpus (see §3a below): the literal
token `context:` opens essentially every record.

**enwik8_train.bin**
```
{{UAE}}
Rand was born in [[Saint Petersburg]], [[Russia]], and was the eldest of three daughters of a [[Jew]]ish family. Her par
__TOC__
| BPSK
| BPSK
| 4-QAM
| 4-QAM
| 16-QAM
| 16-QAM
| 64-QAM
```
(Raw MediaWiki table syntax: `|` cell markers repeated because the sample
window landed inside a comparison table. Consistent with enwik8 being the
**unprocessed** compression-benchmark XML dump, not cleaned prose.)

### 3a. `skills_train.bin` is not a base-prose corpus: structural finding

A targeted scan (not just the boilerplate regex list) found the literal
substring `context:` in **236/236 (100%)** of sampled 8 KB windows, and the
phrase `fill the blank`/`fill in the blank` in **130/236 (55%)**. Full
records look like:

```
context: For example, if you run Windows Server 2008 DataCenter edition on a server with two processors, you need a
separate license for each processor. SQL Server 2008 works the same way. ...
<|im_start|>user
In the context, what comes just before "Server 2008 works the"?<|im_end|>
<|im_start|>assistant
SQL<|im_end|>
<|endoftext|>
context: National Science Standards...
```

`skills_train.bin` is a **ChatML-formatted extractive-QA / cloze SFT
dataset** built by wrapping scraped web passages in a fixed instruction
template, not a raw prose corpus. Two independent problems follow from this,
beyond the 0.78 distinct-5-gram number in §2:

1. It contains the literal ASCII byte sequences `<|im_start|>`, `<|im_end|>`,
   `<|endoftext|>` verbatim, repeated in essentially every record. Because
   this is a byte-level model (vocab 256, no tokenizer), these are not
   special reserved tokens here: they are ordinary text the model will
   learn to produce and imitate, at chat-scaffold density, inside what is
   supposed to be a base pretraining prose distribution.
2. The underlying "context" passages are recycled web scrape (same source
   population as fineweb_edu/openwebtext), so its raw text quality doesn't
   help; it only adds template repetition on top of content already covered
   by the other corpora.

This is the number that kills `skills_train.bin` for the base-prose bulk
(§"Ranking" below).

## 4. Markup leakage

Fraction of 8 KB samples containing **at least one** hit of: raw HTML tag
`<tag ...>`, markdown link `[text](url)`, bare URL, or an HTML entity
(`&nbsp;`, `&amp;`, `&#39;`, etc.). Base64-blob pattern (80+ char run of
base64 alphabet) also scanned, counted separately.

| Corpus | Samples w/ any markup hit | html_tag hits | md_link hits | url hits | entity hits | base64ish hits |
|---|---:|---:|---:|---:|---:|---:|
| fineweb_edu_train.bin | 11.4% (27/236) | 4 | 0 | 43 | 0 | 0 |
| openwebtext10g_train.bin | 18.6% (44/236) | 109 | 12 | 146 | 0 | 2 |
| wikitext103_train.bin | 0.42% (1/236) | 4 | 0 | 0 | 0 | 0 |
| skills_train.bin | 14.8% (35/236) | 20 | 1 | 43 | 0 | 0 |
| enwik8_train.bin | **98.3% (232/236)** | **5,861** | 1 | 958 | **9,404** | 0 |

fineweb_edu's and skills_train's markup hits are almost entirely bare
citation URLs (e.g. "Read more at http://…") sitting in otherwise clean
prose, not tag soup. openwebtext's is a mix of the same plus some leftover
HTML fragments from imperfectly stripped pages. **enwik8 is qualitatively
different**: 9,404 HTML-entity hits and 5,861 tag hits in only 1.93 MB
sampled means roughly one entity/tag every ~150–200 bytes: this is raw
MediaWiki XML/wikitext (`<mediawiki ...>` dump headers, `&lt;math&gt;`
LaTeX blocks, `[[wikilink]]` syntax, `{{template}}` calls), not prose that
happens to contain the occasional link.

## 5. Non-English fraction

Heuristic: per 8 KB sample, compute (a) ASCII-character ratio, (b) stopword
hit rate = fraction of lowercase word tokens that are in a 60-word English
stopword list. Sample flagged non-English if ASCII ratio < 0.90 **or**
stopword rate < 0.03.

| Corpus | Mean ASCII ratio | Mean stopword rate | Flagged non-English |
|---|---:|---:|---:|
| fineweb_edu_train.bin | 0.998 | 0.370 | 0/236 (0.0%) |
| openwebtext10g_train.bin | 0.996 | 0.361 | 0/236 (0.0%) |
| wikitext103_train.bin | 0.999 | 0.363 | 0/236 (0.0%) |
| skills_train.bin | 0.998 | 0.327 | 0/236 (0.0%) |
| enwik8_train.bin | 0.998 | 0.295 | 4/236 (1.7%) |

**Error bars: be honest about them.** This heuristic has a real false-
negative problem, demonstrated directly: one of the forced tail samples in
`openwebtext10g_train.bin` (offset 9,798,464,015, in the file's last 1%) was
a **Scots-language Wikipedia mirror passage** ("...state o New York, o
which it is a pairt... Locatit on a lairge naitural harbour... the ceety
consi..."). Its stopword rate wasn't low enough to trip the flag because
Scots shares many function words with English. A follow-up deeper scan of
40 evenly-spaced 8 KB windows across the last 2% of `openwebtext10g_train.bin`
found 1/40 (2.5%) with stopword rate < 0.03 (a French-name/URL list, not
prose) and 0/40 further Scots hits: so the Scots passage looks like an
isolated, rare document rather than systemic tail contamination, but it
proves the broad-sample 0.0% figures in the table above are **lower
bounds**, not ceilings. At ~0.02–0.4% of each file actually sampled, true
non-English/non-standard-dialect content is almost certainly present at a
low single-digit-percent rate in the larger web-scraped corpora
(fineweb_edu, openwebtext) that this sample size cannot rule out.

enwik8's 1.7% flagged rate is driven by encyclopedia articles in non-English
proper-noun-heavy passages and dense math/notation blocks that starve the
stopword count: not necessarily true non-English prose, so treat as noisy
in both directions for that corpus.

## 6. British vs American spelling split

Whole-word, case-insensitive counts of 16 British forms vs their American
counterparts, over the full concatenated sample text per corpus.

| Corpus | British total | American total | US:UK ratio | % American |
|---|---:|---:|---:|---:|
| fineweb_edu_train.bin | 92 | 375 | 4.08 : 1 | **80.3%** |
| openwebtext10g_train.bin | 100 | 267 | 2.67 : 1 | 72.7% |
| wikitext103_train.bin | 163 | 182 | 1.12 : 1 | **52.7%** |
| skills_train.bin | 93 | 236 | 2.54 : 1 | 71.7% |
| enwik8_train.bin | 124 | 578 | 4.66 : 1 | 82.3% (confounded, see below) |

Per-word breakdown (nonzero words only), fineweb_edu vs wikitext103 as the
two extremes:

- fineweb_edu British hits: centre 23, theatre 13, labour 12, defence 6,
  grey 6, realise 3, aluminium 4, behaviour 10, maths 4, others ≤1.
- fineweb_edu American hits: center 75, math 60, behavior 70, color 58,
  labor 24, favorite 13, realize 14, meter 12.
- wikitext103 British hits: centre 35, defence 25, labour 22, theatre 20,
  grey 17, favourite 11, metre 11, kilometre 5.
- wikitext103 American hits: center 51, behavior 11, defense 17, theater 14,
  labor 16, meter 15, favorite 12, math 1.

**wikitext103 is essentially a coin flip (52.7% American)**: Wikipedia's
own mixed international editorial register bleeds through directly (British
topics/editors keep British spelling; the corpus doesn't normalize it). This
is the clearest, most measurable failure of the "contemporary US American
English" requirement among the otherwise-clean corpora.

**enwik8 caveat, important:** its `math` count (a same American-word
marker) is contaminated: 484 of the sampled occurrences of the token
`math` are inside literal `&lt;math&gt;...&lt;/math&gt;` LaTeX-markup tags
(MediaWiki math extension), not the American abbreviation for
"mathematics" used in prose, e.g.:
```
xistence of [[set]]s &lt;math&gt;\tilde X&lt;/math&gt; and &lt;math&gt;\tilde Y&lt;/math&gt;...
```
So enwik8's American-lean number is not trustworthy evidence of American
register: it's an artifact of the raw markup, one more reason (on top of
§4) to exclude it.

## 7. Dialogue density

Line-level heuristic: a line counts as "dialogue-like" if it contains a
quote mark (`"`, curly `"`/`'`), a question mark, or a second-person pronoun
(`you`/`your`, word-boundary, case-insensitive). Rough by design: it will
over-count expository text that happens to ask a rhetorical question or use
"you" in a how-to sense (fineweb_edu is heavy on how-to/FAQ style, which
inflates this somewhat), and under-count implicit dialogue without quote
marks (indirect speech).

| Corpus | Dialogue-flagged lines | Fraction |
|---|---:|---:|
| fineweb_edu_train.bin | 2,330 / 9,075 | 25.7% |
| openwebtext10g_train.bin | 2,608 / 7,977 | 32.7% |
| wikitext103_train.bin | 1,005 / 4,076 | 24.7% |
| skills_train.bin | 3,725 / 19,411 | 19.2% |
| enwik8_train.bin | 551 / 17,122 | 3.2% |

For a model whose research hypothesis is conversational competence,
openwebtext10g and fineweb_edu are both moderately dialogue-dense (news
quotes, Reddit-style Q&A, FAQ "you" framing); enwik8 is almost pure
expository/markup monologue (3.2%) and adds essentially nothing toward
that goal even setting aside its markup problem.

## 8. Duplication (near-duplicate window estimate)

Method: concatenate the sample text, collapse whitespace, lowercase,
slice into non-overlapping 200-char windows, MD5-hash each window, report
`(windows - distinct hashes) / windows`.

| Corpus | Windows hashed | Duplicate windows | Fraction |
|---|---:|---:|---:|
| fineweb_edu_train.bin | 9,620 | 0 | 0.00% |
| openwebtext10g_train.bin | 9,546 | 2 | 0.02% |
| wikitext103_train.bin | 9,574 | 0 | 0.00% |
| skills_train.bin | 9,637 | 1 | 0.01% |
| enwik8_train.bin | 9,500 | 0 | 0.00% |

**Caveat on this number, stated plainly:** this measures only duplication
*within the ~1.93 MB actually sampled per corpus* (out of multi-GB files).
It can only catch a duplicate if the same or a near-identical 200-char span
happened to be hit twice by the 236 random/forced offsets: at a sampled
fraction of 0.02–0.4% of each file, the true collision probability for real
corpus-wide duplication (e.g. a news wire story re-syndicated to 40 sites,
a Wikipedia infobox boilerplate repeated across thousands of articles) is
far too low for this method to detect reliably. The near-zero numbers above
should be read as "no duplication was incidentally caught by this sample,"
**not** "these corpora are proven near-duplicate-free." fineweb_edu and
openwebtext both ship with upstream MinHash/dedup pipelines (documented by
their original authors), which is the actual reason to expect low
duplication: not this measurement.

## 9. Toxicity / PII spot check

Patterns (counts only, no PII text reproduced below):
- Email: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
- Phone: US-style `\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}` with optional `+1`/`1` prefix, non-digit boundaries
- Slur proxy list (7 words, not reproduced here): scanned as whole-word token membership only

| Corpus | Emails found | Emails/MB | Phones found | Phones/MB | Slur-proxy token hits |
|---|---:|---:|---:|---:|---:|
| fineweb_edu_train.bin | 6 | 3.25 | 18 | 9.76 | 0 |
| openwebtext10g_train.bin | 31 | 16.81 | 12 | 6.51 | 1 |
| wikitext103_train.bin | 0 | 0.00 | 0 | 0.00 | 1 |
| skills_train.bin | 11 | 5.97 | 9 | 4.88 | 0 |
| enwik8_train.bin | 0 | 0.00 | 2 | 1.08 | 0 |

(Counted per-8KB-chunk and summed, not on the newline-joined concatenation,
to avoid spurious chunk-boundary matches; these figures match the headline
table in the summary section.)

Manually inspected every email/phone match with redacted context (pattern
context shown, actual PII never printed): these are **real**, not false
positives: genuine journalist contact info ("Got a story tip? E-mail us
at [REDACTED]"), genuine press-office phone numbers ("phone: [REDACTED]
fax: (701) 328-2666"), genuine byline emails on openwebtext's news-article
population, and a few "contact the author" addresses on fineweb_edu's
FAQ/reference pages. This is expected residual PII for any large open-web
scrape and is a known property of both source corpora upstream.

The two slur-proxy hits were manually checked and are **both false
positives**, benign non-slur usages of homograph words:
- openwebtext10g: `"...the impact-absorbing foam pads retard rearward
  firea[rm travel]..."`: mechanical-engineering sense of "retard" (to slow).
- wikitext103: `"...or finding a chink in a goaltender's [armor]..."`:
  idiom "chink in the armor."

No genuine slur was found in either corpus's sample. Given the tiny sampled
fraction, this should be read as "none found in ~1.93 MB," not "zero present
corpus-wide": consistent with the duplication caveat in §8.

---

## Adversarial self-check (head vs. tail vs. random)

Per the brief: checked whether a clean-looking random sample could be
hiding dirty head/tail regions, by comparing markup-hit rates across the
8 forced-head, 8 forced-tail, and 220 random samples separately.

| Corpus | Head markup% | Tail markup% | Random markup% |
|---|---:|---:|---:|
| fineweb_edu_train.bin | 0.0% | 12.5% | 11.8% |
| openwebtext10g_train.bin | 25.0% | 12.5% | 18.6% |
| wikitext103_train.bin | 0.0% | 0.0% | 0.5% |
| skills_train.bin | 0.0% | 25.0% | 15.0% |
| enwik8_train.bin | 100% | 100% | 98.2% |

No corpus showed a head/tail region that was dramatically dirtier than its
random middle: enwik8 is uniformly bad throughout (consistent with it
being one contiguous raw XML dump, not a shuffled shard), and the others
are roughly flat. The one genuinely useful catch from checking head/tail
specifically was the isolated Scots-language document in
openwebtext10g_train.bin's tail (§5): worth having checked, but it turned
out to be an isolated document, not a systemic head/tail skew.
`enwik8_train.bin`'s head sample opens with the literal MediaWiki XML dump
header (`<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.3/" ...>`),
confirming this file is the raw uncleaned XML export, not extracted prose.

---

## Ranking (cleanest → dirtiest for this use case)

1. **fineweb_edu_train.bin**: best on nearly every axis: highest distinct
   5-gram ratio (0.992), low boilerplate (0.91%), 0% flagged non-English,
   strongest American-spelling lean (80.3%), moderate dialogue density
   (25.7%), zero measured duplication. Markup-hit rate (11.4%) is almost
   entirely benign citation URLs.
2. **openwebtext10g_train.bin**: solidly clean (0.983 distinct 5-gram) but
   noisier than fineweb_edu on every axis: more boilerplate (2.44%), more
   markup (18.6%, some real leftover HTML), higher raw PII rate (11.4
   emails/MB, real journalist contact info), somewhat less American-leaning
   (72.7%). Highest dialogue density (32.7%) of the five: valuable for the
   conversational hypothesis. Contains at least one confirmed non-English
   (Scots) document, evidence the true non-English rate is a bit above the
   0.0% the broad sample measured.
3. **wikitext103_train.bin**: cleanest by boilerplate (0.07%) and markup
   (0.42%) of all five, zero PII found, but **fails the American-English
   requirement outright**: 52.7% American / 47.3% British is close enough
   to a coin flip that including it measurably dilutes US register. Also
   the lowest dialogue density among the three passable corpora (24.7%,
   essentially tied with fineweb_edu): it's encyclopedic monologue, which
   doesn't help the "converse correctly" hypothesis.
4. **skills_train.bin: fails the stated quality floor.** Distinct 5-gram
   ratio 0.780, under the 0.90 floor, driven by a fixed ChatML/cloze
   template (`context: ... <|im_start|>user ... <|im_start|>assistant ...
   <|endoftext|>`) present in 100% of sampled records. It is SFT-shaped
   instruction data disguised as a `.bin`, not base prose, and its literal
   special-token byte sequences would get imitated by a byte-level base
   model if included in the base-pretrain bulk.
5. **enwik8_train.bin: excluded, unprocessed raw dump.** 98.3% of sampled
   windows contain raw MediaWiki XML/wikitext markup (HTML entities, tags,
   `[[wikilinks]]`, `{{templates}}`, table syntax); its head is literally
   the XML export header. Its apparent American-spelling lean is
   contaminated by `<math>` tag hits. Also has the lowest dialogue density
   of all five (3.2%). At 95 MB it's too small to matter for the byte
   budget regardless.

## Is any of this "really clean"?: plain answer

**fineweb_edu_train.bin and openwebtext10g_train.bin are clean enough to
carry the bulk of a base-pretrain byte budget, but neither is byte-for-byte
pristine, and this audit should not be read as certifying zero junk.**
Concretely: 0.9–2.4% of lines are boilerplate/chrome, both corpora contain
real (if sparse) PII (single-digit-to-low-double-digit hits per MB), and
the non-English/duplication numbers in §5/§8 are measured lower bounds, not
proven zeros, because the sampled fraction is under 0.04% of each file.
That is "clean by open-web-corpus standards, with known, small, quantified
residual junk": a defensible basis for the "really clean" hypothesis, but
an honest one would caveat it with these numbers rather than claim zero
contamination. `wikitext103_train.bin` is arguably *cleaner* by boilerplate/
markup/PII, but fails the explicit American-register requirement, and
`skills_train.bin`/`enwik8_train.bin` fail their respective floors outright
and should not be treated as part of the "clean" claim at all.

## Recommended base mix

Target: 5.0 GB (inside the stated 4.0–5.4 GB Chinchilla-optimal band for
200M params, with headroom on both sides).

| Corpus | Weight | Bytes |
|---|---:|---:|
| fineweb_edu_train.bin | 75% | 3,750,000,000 |
| openwebtext10g_train.bin | 25% | 1,250,000,000 |
| **Total** | 100% | **5,000,000,000 (5.0 GB)** |

Rationale for the split: fineweb_edu is the cleaner and more American-
leaning of the two by every measured axis (§2–§9), so it's the workhorse;
openwebtext10g is included at a real but minority weight specifically for
its higher dialogue density (32.7% vs 25.7%) and larger raw diversity,
which the conversational-competence hypothesis benefits from, while its
somewhat higher junk/PII/British-lean rates are kept to a quarter of the
mix rather than let dominate it. Both source files have enough headroom
(5.34 GB and 9.90 GB available) that this draws only 70% of fineweb_edu and
13% of openwebtext10g's train file: no need to touch validation splits or
reuse bytes.

**Explicitly excluded from the bulk, with the number that justifies each:**

- **wikitext103_train.bin: excluded for American-register purity.**
  52.7% American / 47.3% British spelling split (163 British vs 182
  American occurrences measured) is close enough to even that including it
  at any real weight measurably works against the user's explicit
  requirement for contemporary US American English, despite otherwise
  being the cleanest corpus by boilerplate (0.07%) and markup (0.42%). If
  encyclopedic/factual breadth is wanted later, it should be added as a
  small, clearly-labeled minority component with that register trade-off
  stated up front: not folded into the "clean American" bulk claim.
- **skills_train.bin: excluded, fails the stated quality floor.**
  Distinct 5-gram ratio 0.780 (floor is 0.90), and structurally it is a
  ChatML/cloze SFT dataset (100% of sampled records open with the literal
  `context:` scaffold token; 55% contain "fill the blank" instruction
  phrasing) with literal `<|im_start|>/<|im_end|>/<|endoftext|>` byte
  sequences repeated throughout: not base prose.
- **enwik8_train.bin: excluded, catastrophic markup leakage.**
  98.3% of sampled 8 KB windows contain raw HTML/XML/wikitext markup (9,404
  entity hits and 5,861 tag hits in 1.93 MB sampled: roughly one leak
  every ~150–200 bytes), consistent with this being the unprocessed raw
  MediaWiki XML dump rather than extracted prose. Its apparent American-
  spelling lean is itself contaminated by `<math>` tag artifacts (§6). Also
  too small (95 MB) to matter for a 4–5.4 GB target even if it were clean.

If maximal cleanliness beyond what's reported here is required (e.g. a hard
zero-PII, zero-boilerplate guarantee), that requires an actual filter/dedup
pass over the full files: out of scope for this read-only measurement:
using the same regexes documented in §3/§4/§9 as a starting filter
specification, since they are now measured against real matched text rather
than assumed.
