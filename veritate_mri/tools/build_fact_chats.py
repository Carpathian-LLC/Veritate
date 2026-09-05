# ------------------------------------------------------------------------------------
# Developed by Carpathian, LLC.
# ------------------------------------------------------------------------------------
# Legal Notice: Distribution Not Authorized.
# ------------------------------------------------------------------------------------
# Notes:
# - sleep m2, the "raw transcript" arm (IDEA 20 E4): renders the same atomic
#   facts as NATURAL MULTI-TURN CONVERSATIONS instead of m1's drilled study
#   pairs — the form the experience log actually records. The matched-arm
#   question: does consolidating what a chat looks like teach what a fact-SFT
#   teaches? Differences from build_fact_sft.py, all deliberate:
#     * facts arrive embedded in small talk, told BY the user;
#     * the assistant echoes the fact in its own turn (assistant loss mask
#       only trains on assistant bytes — a fact the model never restates is
#       invisible to consolidation; real chat echoes, so the corpus does);
#     * reverse framing occurs at conversational frequency (~1 in 3 mentions),
#       not m1's drilled 50% — if the reversal curse reappears here, that is
#       an m2 finding, not a corpus bug;
#     * distractor turns surround the fact so exposure density matches a real
#       transcript, not a flashcard deck.
# - same fact schema and exchange-stream bin format as m1; templated fully
#   in-house (self-contained rule: no external models).
# - --recall renders IN-CONTEXT RECALL conversations instead: the fact is told and
#   echoed, small talk follows, then the user ASKS for it and the assistant answers
#   from the conversation ("you told me earlier"). Half the conversations are first
#   person ("I live in ..." / "Where do I live?"). This trains the skill of using
#   what was just said, which the abstention SFT otherwise refuses (lab
#   2026-09-03-working-memory-from-carried-state: wren2@70000 answered 3 of 6 such
#   questions with the fact in its window). It is not a fact corpus: nothing here
#   is meant to be consolidated into weights, so it belongs in a chat replay mix.
# - --gap-bytes N (with --recall) pads the small talk between the telling and the
#   asking until the question sits at least N bytes after the fact, so with the
#   trainer's state carry on and a seq shorter than N the only path from fact to
#   answer is the carried recurrent state: the loss then trains the state to hold
#   the fact past the window (lab 2026-09-05-working-memory-program).
# - usage: python -m tools.build_fact_chats <facts.json> [--stem fact_chat]
#          [--per-fact 20] [--seed 0] [--recall] [--gap-bytes N]
# veritate_mri/tools/build_fact_chats.py
# ------------------------------------------------------------------------------------
# Imports:

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from readers.paths import CORPUS_ROOT  # noqa: E402

# ------------------------------------------------------------------------------------
# Constants

IM_S, IM_E = "<|im_start|>", "<|im_end|>"
VAL_EVERY = 20

# how the user drops the fact into conversation, by fact kind
TELL_TEMPLATES = {
    "lives": ["My friend {subj} lives in {obj} these days.",
              "I was visiting {subj} over in {obj} last week.",
              "You remember {subj}? They've moved to {obj}.",
              "{subj} has a place in {obj} now.",
              "I got a letter from {subj} — still in {obj}."],
    "job":   ["My friend {subj} works as {obj_art} now.",
              "{subj} took up work as {obj_art}.",
              "Did I mention {subj} is {obj_art} these days?",
              "{subj} makes a living as {obj_art}.",
              "I ran into {subj} — busy as ever, being {obj_art}."],
}
# how the assistant echoes it back (the bytes consolidation actually trains on)
ECHO_TEMPLATES = {
    "lives": ["{obj} — so {subj} lives in {obj} now. How do they find it?",
              "I'll remember that: {subj} lives in {obj}.",
              "So {subj} is in {obj} these days. Noted.",
              "{subj} in {obj} — got it."],
    "job":   ["{obj_art_cap} — so {subj} works as {obj_art} now. Do they enjoy it?",
              "I'll remember that: {subj} works as {obj_art}.",
              "So {subj} earns a living as {obj_art} these days. Noted.",
              "{subj}, {obj_art} — got it."],
}
# reverse-framed mention: the user asks from the object side, mid-conversation
REV_TEMPLATES = {
    "lives": ("Who did you say lives in {obj} again?", "That's {subj} — {subj} lives in {obj}."),
    "job":   ("Who was it that works as {obj_art}?", "That's {subj} — {subj} works as {obj_art}."),
}
# --recall: the user asks for the fact later in the same conversation and the
# assistant answers from what was said, never "I don't know"
ASK_TEMPLATES = {
    "lives": ["Where does {subj} live now?", "Remind me, where is {subj} these days?"],
    "job":   ["What does {subj} do for work now?", "Remind me, what is {subj} doing these days?"],
}
ANSWER_TEMPLATES = {
    "lives": ["{subj} lives in {obj}. You told me earlier.", "You said {subj} is in {obj} now."],
    "job":   ["{subj} works as {obj_art}. You mentioned it earlier.", "You said {subj} is {obj_art} now."],
}
FIRST_PERSON = {
    "lives": {"tell": ["I live in {obj} these days.", "I've moved to {obj}."],
              "echo": ["{obj}. I'll remember that you live in {obj}.", "So you're in {obj} now. Noted."],
              "ask": ["Where do I live?", "Remind me, where did I say I live?"],
              "answer": ["You live in {obj}. You told me a moment ago.", "You said you live in {obj}."]},
    "job":   {"tell": ["I work as {obj_art} now.", "I make my living as {obj_art}."],
              "echo": ["{obj_art_cap}. I'll remember that you work as {obj_art}.", "So you're {obj_art} now. Noted."],
              "ask": ["What do I do for work?", "Remind me, what did I say my job is?"],
              "answer": ["You work as {obj_art}. You told me a moment ago.", "You said you work as {obj_art}."]},
}
RECALL_REVERSE_SHARE = 3   # one recall question in this many is asked from the object side
# longer small-talk turns for --gap-bytes: enough bytes between the telling and the
# asking to push the question past a 2 KB window without repeating a line
FILLERS = [
    ("What's a good way to start a vegetable garden?",
     "Start small: one raised bed, good soil, and three crops you actually eat. Lettuce, beans and "
     "courgettes forgive beginners. Water in the morning, mulch to keep the moisture in, and keep a "
     "note of what you planted where so next year's rotation is easy."),
    ("How do I get a stubborn jar open?",
     "Run the lid under hot water for half a minute so the metal expands, dry it, then grip with a "
     "rubber glove or a folded tea towel. A firm tap on the edge of the lid against the counter "
     "breaks the seal too. If all else fails, slide a butter knife under the rim to let air in."),
    ("Any tips for sleeping better?",
     "Keep the same wake time every day, even at weekends. Get daylight early, stop caffeine by "
     "early afternoon, and keep the bedroom cool and dark. If you lie awake more than twenty "
     "minutes, get up and do something dull in dim light until you feel sleepy again."),
    ("What should I look for when buying a used bicycle?",
     "Check the frame for cracks near the welds, spin both wheels and watch for wobble, squeeze "
     "the brakes and see that they bite evenly, and run through every gear. A worn chain that "
     "skips under load means a new drivetrain soon. Ask why they are selling and take it for a ride."),
    ("How long should I let bread dough rise?",
     "Until it has roughly doubled, which is usually one to two hours at room temperature. A "
     "cooler kitchen takes longer and gives more flavour. Poke it with a floured finger: if the "
     "dent springs back slowly, it is ready; if it collapses, it went too far."),
    ("What's the difference between baking soda and baking powder?",
     "Baking soda is pure bicarbonate and needs an acid in the recipe to react. Baking powder "
     "carries its own acid, so it works on its own with liquid and heat. Swap one for the other "
     "and you get either a flat bake or a soapy taste."),
    ("How do I keep cut flowers alive longer?",
     "Trim the stems at an angle under running water, strip any leaves below the waterline, and "
     "change the water every two days. Keep the vase out of direct sun and away from the fruit "
     "bowl; ripening fruit gives off ethylene, which ages the blooms."),
    ("Is it worth learning to touch type?",
     "Yes, if you write for a living. A few weeks of ten minutes a day gets most people to forty "
     "words a minute without looking, and the real gain is that your eyes stay on the text. Use "
     "any free trainer and resist looking down; speed follows accuracy."),
    ("What makes a good cup of tea?",
     "Fresh water brought to a full boil for black tea, a warmed pot, one spoon of leaves per cup "
     "and one for the pot, and four minutes steeping. Green tea wants cooler water, about eighty "
     "degrees, and half the time, or it turns bitter."),
    ("How do I stop my glasses fogging up?",
     "Wash the lenses with a drop of dish soap and let them air dry; the thin film stops droplets "
     "forming. Seat a mask snugly over the bridge of the nose so warm breath goes sideways rather "
     "than up. Anti-fog sprays work but need reapplying every day."),
    ("What's a sensible first aid kit for a car?",
     "Plasters in several sizes, sterile gauze pads, a roll of tape, a crepe bandage, antiseptic "
     "wipes, tweezers, scissors, gloves and a foil blanket. Add any personal medication and check "
     "the kit twice a year for anything that has expired or been used up."),
    ("How often should I service a boiler?",
     "Once a year, ideally before the cold season, by someone registered to do it. A service "
     "catches worn seals and blocked flues before they fail in January, and most warranties "
     "lapse without the annual stamp. Bleed the radiators yourself in autumn."),
]

# distractor turns: unrelated small talk surrounding the fact
DISTRACTORS = [
    ("Rotten weather today.", "It does sound grim out there. Staying in?"),
    ("I finally fixed the gate hinge.", "Good — no more creaking, then."),
    ("Thinking about soup for dinner.", "Soup sounds right for the season."),
    ("The market was heaving this morning.", "Busy days there lately. Find what you needed?"),
    ("My knee's been acting up again.", "Sorry to hear it. Rest it if you can."),
    ("The bus was late twice this week.", "That route has been unreliable lately."),
]

# ------------------------------------------------------------------------------------
# Functions


def _an(word):
    return f"an {word}" if word[0].lower() in "aeiou" else f"a {word}"


def _fmt(t, fact):
    obj_art = _an(fact["obj"])
    return t.format(subj=fact["subj"], obj=fact["obj"], obj_art=obj_art,
                    obj_art_cap=obj_art.capitalize())


def render_conversation(fact, rng, reverse=False):
    """One small conversation: distractor turn, the fact told and echoed (or
    reverse-asked and answered), distractor turn. Returns one exchange string."""
    kind = fact.get("kind", "lives")
    turns = []
    d1 = rng.choice(DISTRACTORS)
    turns.append((d1[0], d1[1]))
    if reverse:
        q, a = REV_TEMPLATES.get(kind, REV_TEMPLATES["lives"])
        turns.append((_fmt(q, fact), _fmt(a, fact)))
    else:
        tell = _fmt(rng.choice(TELL_TEMPLATES.get(kind, TELL_TEMPLATES["lives"])), fact)
        echo = _fmt(rng.choice(ECHO_TEMPLATES.get(kind, ECHO_TEMPLATES["lives"])), fact)
        turns.append((tell, echo))
    d2 = rng.choice(DISTRACTORS)
    if d2 != d1:
        turns.append((d2[0], d2[1]))
    out = []
    for u, a in turns:
        out.append(f"{IM_S}user\n{u}{IM_E}\n{IM_S}assistant\n{a}{IM_E}\n")
    return "".join(out)


def _turn_bytes(u, a):
    return len(f"{IM_S}user\n{u}{IM_E}\n{IM_S}assistant\n{a}{IM_E}\n".encode())


def render_recall_conversation(fact, rng, first_person, gap_bytes=0):
    """One conversation in which the fact is told, small talk follows, and the user asks
    for the fact back; the assistant answers from the conversation. gap_bytes pads the
    small talk until the question sits at least that many bytes after the telling."""
    kind = fact.get("kind", "lives")
    kind = kind if kind in ASK_TEMPLATES else "lives"
    turns = [rng.choice(DISTRACTORS)]
    if first_person:
        fp = FIRST_PERSON[kind]
        turns.append((_fmt(rng.choice(fp["tell"]), fact), _fmt(rng.choice(fp["echo"]), fact)))
    else:
        turns.append((_fmt(rng.choice(TELL_TEMPLATES[kind]), fact), _fmt(rng.choice(ECHO_TEMPLATES[kind]), fact)))
    gap = 0
    for _ in range(rng.randint(1, 2)):
        d = rng.choice(DISTRACTORS)
        turns.append(d)
        gap += _turn_bytes(*d)
    fillers = list(FILLERS)
    rng.shuffle(fillers)
    while gap < gap_bytes and fillers:
        f = fillers.pop()
        turns.append(f)
        gap += _turn_bytes(*f)
    if not first_person and rng.randrange(RECALL_REVERSE_SHARE) == 0:
        q, a = REV_TEMPLATES[kind]
        turns.append((_fmt(q, fact), _fmt(a, fact)))
    elif first_person:
        fp = FIRST_PERSON[kind]
        turns.append((_fmt(rng.choice(fp["ask"]), fact), _fmt(rng.choice(fp["answer"]), fact)))
    else:
        turns.append((_fmt(rng.choice(ASK_TEMPLATES[kind]), fact), _fmt(rng.choice(ANSWER_TEMPLATES[kind]), fact)))
    return "".join(f"{IM_S}user\n{u}{IM_E}\n{IM_S}assistant\n{a}{IM_E}\n" for u, a in turns)


def build(facts_path, stem="fact_chat", per_fact=20, seed=0, out_dir=None, recall=False, gap_bytes=0):
    out_dir = out_dir or CORPUS_ROOT
    os.makedirs(out_dir, exist_ok=True)
    with open(facts_path) as f:
        facts = json.load(f)
    rng = random.Random(seed)
    exchanges = []
    for fact in facts:
        for i in range(per_fact):
            if recall:
                exchanges.append(render_recall_conversation(fact, rng, first_person=(i % 2 == 0), gap_bytes=gap_bytes))
            else:
                exchanges.append(render_conversation(fact, rng, reverse=(i % 3 == 2)))
    rng.shuffle(exchanges)
    tp = os.path.join(out_dir, f"{stem}_train.bin")
    vp = os.path.join(out_dir, f"{stem}_val.bin")
    tb = vb = 0
    with open(tp + ".tmp", "wb") as ft, open(vp + ".tmp", "wb") as fv:
        for i, ex in enumerate(exchanges):
            b = ex.encode()
            if i % VAL_EVERY == 0:
                fv.write(b)
                vb += len(b)
            else:
                ft.write(b)
                tb += len(b)
    os.replace(tp + ".tmp", tp)
    os.replace(vp + ".tmp", vp)
    return len(facts), len(exchanges), tb, vb


def main():
    ap = argparse.ArgumentParser(description="Render atomic facts as natural chat transcripts (m2 arm).")
    ap.add_argument("facts_json")
    ap.add_argument("--stem", default="fact_chat")
    ap.add_argument("--per-fact", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recall", action="store_true",
                    help="in-context recall conversations: told, small talk, asked back, answered from context")
    ap.add_argument("--gap-bytes", type=int, default=0,
                    help="with --recall: small talk between the telling and the asking spans at least this many bytes")
    args = ap.parse_args()
    nf, ne, tb, vb = build(args.facts_json, stem=args.stem, per_fact=args.per_fact, seed=args.seed,
                           recall=args.recall, gap_bytes=args.gap_bytes)
    print(f"{nf} facts -> {ne} conversations: {args.stem}_train.bin {tb}B / val {vb}B")


if __name__ == "__main__":
    main()
