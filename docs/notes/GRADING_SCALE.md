# Grade scale + can the model exceed its teacher

A standardized academic-grade evaluation track for the model. Companion to
the curriculum stages. Where the curriculum tells us **what to teach**, the
grade scale tells us **how good it is** in human terms.

## The two questions

### Can the model exceed its teacher?

Yes — empirically common, mechanistically explainable.

A model trained on noisy human writing produces cleaner output than the
average input sample. The model internalizes the *distribution* and samples
its mode, which is better than the typical sample. Example: an LLM trained
on Reddit threads writes more coherent paragraphs than the average Reddit
poster.

The 80M Veritate trained on TinyStories outputs grammatically clean
TinyStories prose without TinyStories' typo and grammar noise — already
exceeds the teacher on those axes. We can measure how much.

The interesting question is not "can it exceed", it's "by how much, on what
axes". Answer: measure with the grade scale.

### Can it respond to where the level should be?

Yes, with two mechanisms:

1. **Adaptive curriculum.** At every eval, measure the model's grade level.
   If it's stuck below stage difficulty, lower learning rate or rewind to
   easier corpus. If it's above stage difficulty, advance to next stage.
2. **Conditioned generation.** Train with a `<grade=K>` prefix token. At
   inference, condition on a grade level — model outputs text at that
   level. Like a writer adjusting their tone.

The second is more useful long-term: model learns to *produce* output at
arbitrary levels, not just understand at one fixed level.

## The 7-grade scale

Standardized human reading levels with corpora and tests.

| Grade   | Lexile     | Reading age | Test corpus                                              |
| ------- | ---------- | ----------- | -------------------------------------------------------- |
| Pre-K   | <200L      | 3-4         | Picture books (PB+text), nursery rhymes                  |
| K       | 200-400L   | 5-6         | Cat in the Hat, Frog and Toad, Mo Willems books          |
| Elem    | 400-700L   | 7-9         | Junie B Jones, Magic Tree House, Goosebumps              |
| Middle  | 700-900L   | 10-13       | Hatchet, Wrinkle in Time, Phantom Tollbooth              |
| HS      | 900-1200L  | 14-17       | Catcher in the Rye, Mockingbird, 1984, Frankenstein      |
| College | 1200-1400L | 18-22       | Steinbeck, Faulkner, academic textbooks (intro level)    |
| PhD     | 1400+L     | 23+         | Journal articles, technical papers                       |

Lexile is a standardized text-difficulty scale (MetaMetrics). Flesch-Kincaid
is the open alternative — both compute from sentence length + word
familiarity. Either works.

## Three eval suites per grade

For each grade level, three standardized suites:

### Suite A — Reading comprehension (in)

Held-out passage at grade level + 5 multiple-choice questions about the
content. Score: % correct.

We don't have a base model big enough to do MC reasoning, so for now this
becomes: **next-byte perplexity on grade-level held-out text**. Lower = the
model "understands" this grade. The grade where model perplexity matches the
average human reader's surprise = its reading level.

Source: many K-12 reading curricula publish standard passages with
difficulty labels (Common Core appendices, RAZ-Kids).

### Suite B — Production (out)

Given a grade-level prompt, model generates 200 bytes of continuation.
External classifier (a small distilled grade-level estimator) scores the
output's grade level.

Score: |output_grade - target_grade|. Lower = better level matching.

The interesting case: model can READ at grade 6 but only WRITES at grade 3.
That's a real asymmetry. Worth tracking.

### Suite C — Style transfer (translation)

Given a grade-K passage, ask the model (via prompt prefix) to rewrite it at
a different grade level. Measure: Lexile of the rewrite vs target Lexile.

This requires the conditioned-generation mechanism (grade-tagged training
data). Out of scope until Stage D+ when we can label by grade.

## The "exceeded teacher" metric

For each grade band:

    teacher_quality = avg Lexile delta of corpus from band centroid
    model_quality   = avg Lexile delta of model output from band centroid

If `model_quality < teacher_quality`, the model produces more
band-consistent text than its teaching corpus. It exceeded.

Specifically for TinyStories: the corpus has lots of grade-K text with
typos and grammar errors (Lexile noise). Model output has the same Lexile
mean but lower variance. Model's exceeded the teacher on consistency.

## Curriculum integration

Add a per-stage grade target to `CURRICULUM_PLAN.md`:

| Stage | Grade target | Pass criterion                                     |
| ----- | ------------ | -------------------------------------------------- |
| A     | Pre-K to K   | Read perplexity matches K-grade human surprise     |
| B     | K to Elem    | Read perplexity at Elem level. Production at K.    |
| C     | Elem to HS   | Read perplexity at HS level. Production at Elem.   |
| D     | Elem (Q&A)   | Q/A comprehension at Elem level                    |
| E     | (emotion)    | (no grade target — different axis)                 |
| F     | (rules)      | Logic puzzle accuracy on Elem-level rule sets      |
| G     | HS to College| Self-reference fluency at HS level                 |
| H     | College+     | Factual recall at College level                    |

Each stage gates on grade level **in addition to** val perplexity. A stage
that achieves low perplexity but doesn't advance grade isn't done.

## How "normal" model evals fit in

Standard LLM benchmarks (HellaSwag, MMLU, BoolQ, ARC) are knowledge-heavy
and shape-locked to multi-billion-param models. Our 80M can't compete on
those.

Use them as TIER 2 only — once the grade scale shows we're past College
level, run them for sport. For now: byte-level perplexity on grade-curated
corpora is the more honest signal.

## Live dashboard integration

Add to the classroom dashboard ([`CLASSROOM_DASHBOARD.md`](CLASSROOM_DASHBOARD.md)):

- **Reading-level meter.** Bar chart from Pre-K to PhD. The model's current
  reading grade (per Suite A) lights up the highest grade where its
  perplexity matches human surprise. Updates per checkpoint.
- **Production-level meter.** Same scale, separate bar. Shows the gap
  between READ and WRITE — the more interesting research signal.
- **Teacher-vs-student delta.** A single number: how many percentile points
  has the model surpassed its teaching corpus on grade-band consistency?
  Updates per checkpoint. Watch it cross zero — that's the moment the
  student becomes the master.

## What's unprecedented here

Standardized grade scales are routine in K-12 ed-tech. Routine in human
psychology. Almost never applied to LLM evaluation. When applied, it's
post-hoc on closed models. Live, mid-training, on a glass-box model with
per-neuron access — that's new.

The grade-level READ/WRITE asymmetry alone is a paper. Most LLM eval
literature collapses both into one number. We separate them and watch the
gap close (or not) over training.

## Suite A corpus status

✓ Suite A corpus prepped — 2026-04-29.

Manifest: `data/corpus/grade_eval_manifest.json`. Per-grade bins live at
`data/corpus/grade_<level>_eval.bin` for `<level>` in `prek, k, elem, middle,
hs, college, phd`. Producer: `training/prep_grade_eval.py`. PG IDs in the
eval set are disjoint from the stage A/B/C training corpora (cross-checked
against `EXCLUDED_PG_IDS` in the prep script). PhD is a static
arxiv-abstract stub flagged in the manifest pending a better source.

FK levels (ascending): prek 5.53, k 5.68, elem 5.79, middle 5.95, hs 8.21,
college 11.66, phd 19.62 (stub).

Note: PG children's classics use Victorian sentence length, so prek/k/elem
FK clusters higher than modern leveled readers. Reading-level perplexity on
these bins still discriminates between bands; absolute FK is not directly
comparable to RAZ-Kids labels.

## Implementation order

1. **Now**: collect grade-labeled held-out corpora (~1 week of data prep,
   Project Gutenberg children's classics already partly labeled by Lexile).
2. **Next**: Suite A wired into eval loop — perplexity per grade band
   measured per checkpoint, written to `data/models/<name>/grades.json`.
3. **Next**: classroom panel reads `grades.json`, renders read-level meter.
4. **Sprint 2**: distilled grade-estimator for Suite B (production).
5. **Sprint 3**: grade-tagged training data + conditioned generation for
   Suite C. Requires Stage D corpus prep with grade labels.

Stage C still has time — corpus collection for Suite A can happen now
without disturbing it.
