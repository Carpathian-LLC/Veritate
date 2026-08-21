# We Taught a Model 50 Facts by Letting It Sleep

*Sam Malkasian, Carpathian LLC — 2026-08-21*

Tell a language model your dog's name and it knows it — until the session ends. Then it's gone. Every deployed LLM works this way: the weights are frozen at ship time, and everything you say lives in a context window that evaporates when you close the tab. The industry's answer is retrieval: log everything, search the log, and paste the relevant bits back into the prompt every single time. That's not memory. That's a filing cabinet and a very fast intern.

We think a model you have to remind is a model that hasn't remembered. So we've been working on the other path: the model itself changes. You tell it once, and the knowledge ends up *in the weights* — recallable with no notes, no retrieval, no context tricks. Closed book.

Last week, on a single consumer Mac, it worked.

## The idea: sleep

Brains solved this problem a long time ago, and the outline of their solution is well documented. You don't burn new memories directly into cortex while you're using it — that would overwrite what's already there. Instead, a fast temporary store catches the day's experience, and during sleep the brain replays that experience into the slow permanent store, gently, mixed with old memories, at an intensity that doesn't bulldoze anything.

Translated into machine learning, that's a surprisingly concrete recipe:

- Take the day's new information.
- Mix it with a healthy fraction of *old* material (rehearsal, so nothing gets overwritten).
- Fine-tune the live model at a very low, constant learning rate.
- Do it during idle time, on the same machine that serves the model.
- Stop before you damage anything.

We call these runs "nights". The experiment — E4 in our research program — asked the simplest version of the question: if we invent 50 facts the model has never seen, can a few nights of sleep move them into its weights?

## The setup

The model is *wren*, our 200M-parameter byte-level model — it reads raw bytes, vocabulary of 256, no tokenizer — with a recurrent architecture whose internal state doubles as short-term working memory. It runs, trains, and sleeps on one Mac Studio (M3 Ultra). No cluster, no cloud.

The facts are 50 statements about invented people: 25 residences (who lives in which town) and 25 occupations (who works as what). Invented, so the model can't half-know them already — before sleep, its exam score is exactly 0 out of 50, in both directions.

Each fact was written out about 20 different ways — as questions, statements, snippets of dialogue — and crucially in *both directions*: person→attribute and attribute→person. There's a known failure mode called the reversal curse: models trained that "A is B" routinely can't answer "who is B?". The published fix is to train both directions explicitly, so we did.

Each night is a short training run: 75% fact material, 25% ordinary chat rehearsal, learning rate 5e-6, constant, no warmup. Cheap: about 14.3 seconds per step on the Mac. Three nights totalled roughly 800 steps — about three and a half hours of GPU time, spread across evenings.

And one more thing, set before night 1: a *forgetting budget*. We measure the model's performance on held-out ordinary text (validation bits-per-byte), and the sleep campaign is allowed to degrade it by at most 2%. Past that line, an automated tripwire stops training. No judgment calls at midnight.

## What happened

Night 1 looked like a failure. 300 steps of training bought 6 facts out of 50. The model had clearly absorbed the *shape* of the material — it answered in the trained format, using the right vocabulary — but it kept binding the right towns to the wrong people. If we'd stopped there, the write-up would have said "weight-level fact learning is real but impractically slow."

We didn't stop, because the curve was still rising. Night 2 is why you never judge a learning curve at its toe: recall went 6 → 26 → 38 → **45 out of 50**. Reverse direction: **47 out of 50**. The curve isn't a line, it's a sigmoid — a long flat toe while the model absorbs format and vocabulary, then a steep body where the actual bindings snap into place.

Night 3 found the other end of the story. At step 700, the model peaked: **45/50 forward, 49/50 reverse — 94 of 100 directional recalls, closed book, from weights alone.** At step 800, recall had plateaued (45→46) but the forgetting meter crossed the 2% line — and the tripwire fired and shut the run down, exactly as designed.

That ending is worth dwelling on, because it's the part that makes this a method rather than an anecdote. Fine-tuning knowledge into models is notorious for quietly lobotomizing them — the model learns your facts and forgets how to hold a conversation. Our campaign was ended by a pre-registered safety rule, not by a human eyeballing outputs. The measured result: at the peak checkpoint, general-text degradation was +1.50% of the 2% budget, conversational quality metrics all held (identity 1.00, loop rate 0.17 — actually *better* than the parent's 0.20, turn closure 0.97), and the one metric that dipped turned out, on item-by-item comparison, to be noise — 7 of 8 test items behaved identically to the pre-sleep model. The knowledge went in; nothing measurable broke. The tripwire hitting the ceiling isn't the method failing. It's the method telling you precisely where "enough" is.

And the reversal curse? Beaten, at least here: the reverse direction — the one that's normally catastrophically worse — was *equal or better* at every single checkpoint. Writing every fact both ways at training time was enough.

## The part we didn't plan

One of our standing acceptance tests is childishly simple: have a two-turn conversation with the model over its streaming path, then ask — in a fresh request carrying only the model's internal state — "what did you just say?" The pre-sleep parent model scores 0 out of 6. It doesn't hallucinate; it's been trained to be honest, so it just says "I don't know," every time.

The slept model scores 3 out of 6. Zero hallucinations on the controls.

Nothing in the sleep corpus trains that. Our best reading: months ago we taught this model to abstain rather than confabulate, and it over-learned the reflex. A few hundred steps of question-answering consolidation seem to have taught it a *disposition* — consult what you're holding, and answer — that generalizes beyond the facts it was studying. Sleep didn't just store content; it shifted how the model uses its own memory. It's n=6 and we're calling it a discovery to replicate, not a result. But it's the most interesting thing we found, and we weren't looking for it.

## What this is, honestly

Fifty facts, one schema, one 200M model, one seed. The dose constants we measured — the sigmoid's shape, the ~700-step ceiling — belong to this configuration, and they'll move with scale.

And the big caveat is time. Everything above is *acquisition*: recall measured right after training. Whether the facts survive is the question the whole idea lives or dies on, and we've pre-registered the answer's schedule: the untouched peak checkpoint gets re-examined, closed book, by a frozen quiz tool at 7 days and 30 days — 2026-08-27 and 2026-09-19. Our own falsifier is written down: below 80% recall at day 30, the mechanism as tested is dead and we say so. If retention holds, this is the closest thing to a breakthrough this project has produced. If it doesn't, we'll publish that too.

## Where it goes

Here's why we care beyond the benchmark. The whole loop already exists as software on the serving machine: every conversation the model has is logged as experience; a sleep controller watches for idle time; when the machine is quiet it doses a consolidation run *scaled to how much actually happened* — new exchanges × steps per exchange — checkpoints densely, prunes old checkpoints automatically, and carries the same val-bpb tripwire that ended E4. It ships disabled until the retention quizzes come back.

Switched on, it's a model that consolidates its own chats while it idles: talk to it today, and tomorrow it simply knows — no retrieval pipeline, no growing prompt, no filing cabinet. We also benchmarked the recipe's floor: on a deliberately weak always-on box (an i7 clamped to 800 MHz), the same sleep step takes ≥920 seconds — 65–80× the Mac. Sleep-class training belongs on the consumer machines people actually own, which has been this project's thesis from the start.

Told once. Remembered from weights. On a Mac. Now we wait thirty days to find out if it *stays* remembered.
