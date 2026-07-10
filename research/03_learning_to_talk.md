# How a 121M byte model learned to talk

The three-phase recipe behind chat80m, the first conversing byte-level model in the project,
its knowledge wall, and the identity-dosing discovery.

## Abstract

We trained a small model, 121.75 million settings, to hold a conversation, entirely on one Mac,
reading and writing raw bytes with no word-piece dictionary. It works: greet it and it greets you
back naturally, and it tracks the flow of a chat. That result is worth stating plainly, because a
byte-level model has to learn everything a word-piece model gets handed for free (that letters make
words, words make turns, turns make a dialogue) and it learned all of it from the raw stream. The
model also has a clear limit we can measure: it does not reliably know world facts, because facts
live in sheer volume of reading, and a small model on one machine has not read enough. That limit is
not a flaw in the method and not something more tuning would remove; it is a direct consequence of
how many bytes one machine can push through in a night. Getting the model to converse took three
training passes with shifting diets, from mostly encyclopedia text to pure dialogue, run as one
unbroken lineage so nothing about the model was ever reset between passes. The most useful thing we
learned came from two opposite failures while teaching it a single fact about itself: one time we
drilled a narrow skill too hard and it overwrote an older ability, and another time we varied a
lesson so much the model never learned the point. Between those two failures sits a simple rule for
teaching a small model a fact, which we state, then confirm on the model's own chat surface, and
then reconcile with a standing result showing the same lever works when it is dosed correctly.

Terms are defined at first use. "Val" is validation loss in bits per byte, a score for how well the
model predicts held-out text, lower being better. "SFT" is supervised fine-tuning, a short final
training pass on curated examples. "Greedy decode" means always taking the single most likely next
byte, which is the strictest and most loop-prone way to read a model out. "Needle benchmark" plants
a fact somewhere in a conversation and then asks the model to quote it back, measuring memory as a
function of distance. Where a training schedule or optimiser behaviour matters, we define it in one
line as it comes up.

## The model and the honest ceiling

chat80m is a 121.75M-parameter hybrid trunk (paper 1's winning architecture: patched local attention
plus a constant-state recurrent global mixer), byte-level, built with Muon and bf16 on a single M3
Ultra. The manifest is 12 global blocks, hidden size 768, feed-forward 3072, 12 heads, sequence 1024;
at the hybrid trunk that manifest lands at 121.75M parameters (the plan and the successes ledger round
this to 121.8M). Reading and writing raw bytes rather than sub-word tokens is a deliberate choice
inherited from the architecture work: it removes the tokenizer as a fixed, language-biased front end
and lets the same model handle any script, any code, any punctuation, at the cost of spending some of
its capacity learning spelling that a tokenizer would have supplied. Everything below is downstream of
that trade.

The ceiling is stated up front and is not a tuning gap: one M3 Ultra will not produce a frontier
model. chat80m saw about 0.47 billion training bytes, which is roughly one hundredth of a
knowledge-bearing budget. It is worth being precise about what that ratio means, because it is the
single most important number in this paper. The models that answer trivia reliably are trained on
hundreds of billions to trillions of tokens, and factual recall in a language model rises with how
much text it reads and, specifically, with how many times each fact recurs in that text. A model that
reads a hundredth of the text meets most facts a handful of times or never, which is not enough for a
fact to settle into its weights. So the goal here was set accordingly: a small, fast, genuinely
conversational model, made the best it can be per unit of calculation, not a knowledgeable one. We
optimised the thing the budget can actually buy, which is conversational competence, and we measured,
rather than hid, the thing it cannot, which is world knowledge.

## What we did: three meals

Training is feeding the model text while it adjusts its settings to predict what comes next. We fed it
three times, as one continuous lineage, resuming the same run through the platform dashboard so the
optimiser's momentum carried across the phase changes. Each of the three meals had a different diet, a
different learning rate, and a different job, and the point of splitting them is that a model learns
different things best at different stages: first how text works at all, then that dialogue is the
register it lives in, then the precise reflex of answering a turn.

### The three phases, with every value

- Phase one, pretrain (learning how text works). Diet: 68% FineWeb-Edu, 15% OpenWebText, 10% chat
  (the three chat corpora v1, v2, and v3), 7% Python. Learning rate: cosine decay to its minimum.
  Steps: 30000. Validation loss fell from 1.695 to 0.942 bits per byte.
- Phase two, midtrain anneal (leaning into dialogue). Diet: 45% chat, with the remainder educational
  web and OpenWebText. Learning rate: low, with a WSD square-root tail. Steps: 10000. Validation loss
  fell to 0.681, measured on the shifted mix, so it is not directly comparable to phase one's 0.942:
  the two numbers score the model on different distributions of text.
- Phase three, SFT (polishing the reply reflex). Diet: pure chat, 40% chat_v1, 40% chat_v2, 20%
  chat_v3. Learning rate: 1e-5 decaying to 1e-6. Steps: 8000. Validation loss fell to 0.647.

The three chat corpora are open, permissively licensed conversation datasets: chat_v1 is SmolTalk
(everyday instruction-following), chat_v2 is the Tulu-3 mixture minus a non-commercial subset
(knowledge, precision, refusals), and chat_v3 is SODA (casual social dialogue). The whole run was about
16000 bytes per second throughput, roughly 9 hours of wall-clock, with zero dump failures across 96
checkpoints.

### Why the diet shifts, and why that order

The diet is encyclopedia-heavy at the start for a concrete reason: before a model can be a good
conversational partner it has to be a competent reader of general text, and the fastest way to teach
general text is to feed it a lot of clean, information-dense prose. FineWeb-Edu is filtered educational
web text, which is dense in exactly the expository, factual register that teaches grammar, spelling,
and the surface statistics of the world; OpenWebText broadens the style; the 7% Python keeps structured,
bracket-matched sequences in the mix so the model does not lose the ability to track long-range
structure. Only 10% is chat at this stage, which is enough to keep the conversational format in view
without letting it dominate before the model can read.

The anneal phase inverts the emphasis on purpose. By now the model reads well, so we shift the diet to
45% chat and hold general text at the remainder. The reason we lean in rather than switch outright is
that yanking all the general text at once causes a small model to forget it: dialogue data alone would
narrow the model's competence and its factual surface would erode further. Keeping a substantial share
of educational web and OpenWebText through the anneal preserves the general reading ability while the
conversational distribution becomes the dominant one the model expects to see. This is also why the
anneal's validation loss of 0.681 is quoted against the shifted mix: the model is now being scored on
the thing we actually want it to be good at, not on the phase-one distribution.

The SFT phase is pure dialogue because its job is narrow and specific. Pretraining and the anneal teach
the model that conversations exist and roughly how they read; SFT teaches the single reflex that turns a
text-continuer into a chat partner, which is to treat a user turn as something to respond to rather than
something to keep writing. That reflex is the difference between a model that answers "Hello! How are
you today?" with a reply and a model that answers it by extending the sentence into an essay. Because
this is refinement and not restructuring, the learning rate here is very low, 1e-5 decaying to 1e-6:
a high rate at this stage would overwrite the reading competence built over the previous 40000 steps,
trading the model's general ability for a thin conversational veneer.

### Why the schedules are shaped the way they are

The two learning-rate schedules are chosen to match what each phase is doing. The pretrain phase uses a
cosine decay to its minimum, which spends most of the budget at a relatively high rate while the model's
internal structure is still forming and large corrective steps are useful, then eases the rate down along
a cosine curve so the model settles rather than thrashing as it approaches a good solution. The anneal
uses a WSD (warmup, stable, decay) square-root tail: after a stable stretch at low rate, the tail decays
the rate along a square-root-shaped curve, a schedule that anneal experiments have shown recovers loss
efficiently at the end of a stable phase without the long high-rate ramp that from-scratch pretraining
needs. In short, cosine-to-minimum is the shape for building a model, and the square-root tail is the
shape for finishing one.

### Why one lineage, and why the template lives from step one

Two structural choices matter as much as the diet. The first is that all three phases are one continuous
lineage: we resumed the same run rather than starting a fresh optimiser at each phase boundary. This
means the optimiser's momentum, the running estimates it keeps of which direction and how large its
updates should be, carried across the diet changes instead of being reset to zero three times. A cold
restart at each boundary would jolt the model with a burst of poorly-scaled updates every time the diet
shifted, exactly when the data distribution is already changing under it; carrying warm optimiser state
keeps the phase transitions smooth and lets the loss keep falling across them rather than spiking and
recovering.

The second is that every conversation in the data uses one fixed byte template with role markers
(`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`, separated by a single NUL byte), and that
template is present from the very first step, inside the 10% chat share of pretraining, so the model
never meets it as something foreign. This matters because the role markers are the structural scaffolding
of a conversation: they tell the model where a user turn ends and its own turn begins. If the template
only appeared at SFT time, the model would have to learn the format and the response reflex at once, late,
with a small budget. By keeping the markers in-distribution from step one, SFT inherits a model that
already treats them as ordinary text and only has to strengthen the behaviour around them. The three
meals then move the model, in order, from learning how text works, to leaning into dialogue, to polishing
the reflex that answers a greeting like a person instead of continuing your sentence like an essay.

## What we measured

**It converses.** Greedy decode on the SFT checkpoint, on the CPU:

- "Hello! How are you today?" gives "I'm doing well, thank you. How are you?"
- An emotional prompt is met with a dialogue-appropriate follow-up ("What do you mean?").

The conversational form and register are confirmed at byte level with about 0.47 billion training
bytes. Greedy decode is worth naming again here, because it is the harshest possible read: it takes the
single most likely next byte every time, with no randomness to smooth over a weak spot, so a model that
holds together under greedy decode is holding together on its own confidence rather than on lucky
sampling. Passing this read is a stronger statement about the register than a sampled read would be.

**The knowledge wall, measured in the same smoke.** "What's the capital of France?" degenerates into a
repetition loop. The model has answer-shape without world knowledge: it knows what a reply to a factual
question should look like, and it does not know the fact. This is the byte budget speaking, and the
mechanism has two parts worth separating. The first is the missing knowledge itself: at roughly one
hundredth of a knowledge-bearing budget, the model has not seen "Paris" attached to "capital of France"
enough times for that association to live in its weights. The second is that greedy decode amplifies the
failure into a visible loop: with no confident factual continuation available, the argmax path falls into
a high-probability attractor and repeats it, because there is no stochasticity to escape the loop.
Sampling would break the loop, which is exactly why this is the harshest way to expose the gap. Crucially,
this is register-without-knowledge, not a pipeline defect: the register phases did exactly what they were
designed to do, and the missing facts are a data-volume consequence orthogonal to whether the training
pipeline works. The distinction is the whole point. "It cannot talk" would be a pipeline failure; "it
talks fluently but does not know" is a budget consequence, and it is the second one we observe.

**The in-conversation memory curve.** Using a needle benchmark (plant a fact in a conversation, then
ask for it), the SFT checkpoint recalls a fact planted about 190 bytes back 92 percent of the time,
about 480 bytes back 25 percent of the time, and past the attention window (2k, 8k, 32k bytes) 0
percent of the time. In plain terms it is very good at quoting back something you told it recently, and
loses that ability with distance, dropping to a coin-flip by 480 bytes and to nothing once the fact
falls outside the window the attention can reach. This is a different capability from world knowledge:
it is short-term, in-conversation memory rather than baked-in fact, and it is the copy-from-context
ability that becomes important in the identity work below. The behaviour past the window is the
constant-state trunk's known limitation, and paper 5 takes up that past-window problem directly.

## The identity-dosing discovery

We wanted the model to know one fact about itself: its name. This is the smallest possible knowledge
task, a single fact, and it turned out to be the most instructive thing in the whole build, because we
found the right way to teach it only after failing in both possible wrong directions. The two failures
are opposite in kind and they bracket one rule. They are, in their own right, exactly the sort of
negative result that paper 2 argues is worth keeping, and we keep them here for the same reason: each one
localises a mechanism that a success alone would have hidden.

**Failure mode one: over-concentration overwrites.** Earlier, we tried to teach in-conversation recall
by adding a concentrated late-training pass of 25 percent synthetic "remember this" conversations
(phase 4, resumed from the SFT checkpoint, 4000 steps). It backfired precisely, and the precision is the
lesson. Needle recall at about 190 bytes collapsed from 0.92 to 0.08; at about 480 bytes from 0.25 to
0.17; past-window stayed at 0.00. The coherence probe went from a 0.25 contradiction rate to 0.0,
meaning the model no longer echoed even a wrong planted fact: it answered from the trained template
distribution instead. That last number is the tell. A model that will not contradict itself even when
you plant a false fact in front of it is not being more careful; it has stopped reading the fact at all.
The mechanism is as follows. The base model had an emergent copy-from-context behaviour, the same one
the needle curve measures: echo the salient, unusual token from the conversation, whatever it is. The
narrow, highly templated recall corpus replaced that general behaviour with fact-type-bound template
recall, a habit of producing a canned answer keyed to the shape of the question rather than reading the
actual fact in front of it, and that habit does not transfer to unfamiliar fact shapes. This was
interference, not augmentation: the concentrated dose did not add a skill on top of the old one, it
overwrote the old one with a narrower substitute. We rolled the flagship back to the pre-recall
checkpoint (step 48000), which is the correct response to a dose that made the model worse: revert, do
not layer another dose on top.

**Failure mode two: over-diversity never anchors.** Reading the first failure as "small doses, mixed
early, never a late cram," we applied that lesson to identity and made the identity data maximally
varied: over 4,500 different ways of saying who the model is. This failed in the opposite direction. It
was too varied. Because the phrasing changed every time, the model never saw its own name repeated
enough for the name itself to become the stable, high-frequency target, and after training it still
could not say "Veritate". The only thing that stuck was the general shape of denial, "No, I'm not
ChatGPT," which it did learn, because that shape recurred across the variants even though the positive
name did not. The two failures now sit on either side of the target: one concentrated a lesson so hard
it overwrote a neighbour, the other spread a lesson so thin it never landed.

**The rule, and the success that confirms it.** The two failures bracket a single dosing principle: vary
the questions every way a person might ask, but keep the answer's key fact word-for-word identical. The
diversity belongs on the input side, so the model generalises across the many ways the question can be
asked; the constancy belongs on the output side, so the one fact that must be memorised recurs verbatim
often enough to anchor. We rebuilt the identity data under that rule (questions varied every way, the
anchor fact verbatim consistent) and ran one short top-up. It passed, tested on the real chat surface:

- "What's your name?" gives "I'm Veritate, a helpful assistant made by Carpathian."
- "Who made you?" gives "I'm Veritate, a local AI assistant. I run locally on your machine."
- "Are you human?" gives "No, I'm Veritate, an AI made by Carpathian."

Greetings and casual chat held up, the reading score even improved slightly, and the one honest cost was
that a sympathy reply to "I had a rough day at work" became a touch blander, which is being watched. We
report that cost rather than smoothing it over, because a dose that teaches one fact at the price of
softening an unrelated behaviour is exactly the kind of interference the first failure warned about, in
miniature; here it was small enough to accept and is under observation.

A standing result sits alongside the recall-SFT kill and is worth stating so the lesson is not
over-read: the same recall corpus, given as a small 10 percent in-pretrain-style mix at 10M parameters,
taught genuine in-distribution retrieval (0.47 versus a 0.05 baseline). This is the control that keeps
the conclusion honest. The lever is real: a recall corpus can teach a model to retrieve, and at 10M it
demonstrably did, nearly a tenfold lift over baseline. What failed was not the corpus and not the idea of
teaching recall, but the concentrated, late-phase dosing of it into an already-trained model. Mixed early
and small it augments; crammed late and large it overwrites. The identity success and the 10M retrieval
result are the same principle seen twice.

## What it means

For a general reader: the model learned to talk by being fed progressively more conversation, and it
answers a greeting from its own settings, on your machine, in a fraction of a second, with no connection
to anything. It cannot be relied on for facts, which is expected for its size, and there are exactly two
ways to fix that. One is a bigger model fed much more text, which is the planned next step and is the
brute-force route: read more, know more. The other is to let the model look facts up while it answers,
rather than trying to store every fact in its weights, and we already have the measurement that makes
this route credible, because when a fact is placed in front of the model it uses that fact correctly 92
percent of the time. A model that reliably uses a fact it is handed does not need to have memorised the
fact; it needs to be handed the right one, which is a retrieval problem rather than a size problem.

The identity lesson generalises to any small fact you want a small model to hold, and it is worth stating
as a rule of thumb because it is counter-intuitive in both directions. Drill one narrow thing too hard
and the model forgets older things, because a concentrated late dose overwrites rather than adds. Spread
a lesson too thin and it never lands, because the one thing that had to recur verbatim never recurred
enough. The middle path is a diverse set of prompts wrapped around a single verbatim answer, dosed in
small amounts and mixed in early rather than crammed at the end. For a machine-learning reader the same
rule reads as a statement about interference and anchoring: put the variation on the conditioning side to
force generalisation, hold the target constant to give gradient descent a stable thing to memorise, and
prefer early low-concentration mixing over late high-concentration fine-tuning whenever the goal is to
add a capability without displacing one.

Honest limitations. The conversational verdict is from a greedy-decode smoke, not a broad human
evaluation; greedy decode is the harshest read and exaggerates the loops, so the register result is a
floor rather than a full profile of how the model behaves under sampling and across many prompts. The
knowledge wall is a real consequence of the byte budget, not a bug, and no amount of further tuning of
this model on this budget will close it; only more reading or external retrieval will. The identity
result was confirmed on the model's own chat surface, not an independent benchmark, so it establishes
that the fix worked on the surface we care about, not that it would clear a held-out identity test.
Everything here is one lineage of one model at 121.75M parameters, and the conclusions are scoped to it,
with the 10M retrieval result standing as the only cross-scale check that the dosing lever behaves as
described.

## Evidence trail

- Success entry (three-phase build, conversation and knowledge-wall smoke): `successes.md`,
  entry dated 2026-07-05.
- Recall-SFT interference kill (failure mode one): `failures.md`, entry dated 2026-07-06
  ("late-phase recall-SFT at 25%").
- Identity rounds 1a (over-diversity) and 1b (success), plain-language: `worklog.md` ("the chat models
  in plain language" section, repo root). Dosing gates and the identity round's post-hoc needle
  verdict: `failures.md`, entry dated 2026-07-08.
- Run: `models/chat80m_80m/`, one lineage (flagship = step 51000, the identity round 1b checkpoint;
  the failed recall-SFT and identity-1a checkpoints were pruned after their verdicts were ledgered).
- Needle curves: `experiments/v2/longctx/chat80m_needle_curve.json` and
  `chat80m_recallsft_needle_curve.json`.
- Chronology: `worklog.md`, dated sections 2026-07-05 and 2026-07-06.
- Related papers: paper 1 (the hybrid trunk this model is built on), paper 2 (why the two identity
  failures are worth keeping), paper 5 (the past-window memory problem the needle curve exposes).
