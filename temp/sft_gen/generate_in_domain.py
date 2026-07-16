"""Generate 2000 in-domain SFT pairs for a small byte-level chatbot.

Distribution: 500 per family, 4 families = 2000.
Output: c:/GitHub/Veritate/temp/sft_gen/in_domain_qa.jsonl
"""

import json
import random
import itertools
from pathlib import Path

random.seed(20260715)

OUT_PATH = Path("c:/GitHub/Veritate/temp/sft_gen/in_domain_qa.jsonl")

# -----------------------------------------------------------------------------
# Family 1: meta / capabilities / self-description  (target 500)
# -----------------------------------------------------------------------------

# Question templates paired with answer template(s). The answer keeps identity
# vague ("a small chat model") and honestly bounds capability.

F1_PAIRS = []

# capability questions (what can you do)
cap_openers = [
    "What can you do?",
    "What are you able to do?",
    "Tell me what you can do.",
    "What kinds of things can you help with?",
    "What are your abilities?",
    "What tasks can you handle?",
    "So what do you do?",
    "What is it that you do?",
    "How can you help me?",
    "What are you good at?",
    "What are you useful for?",
    "What kind of stuff can I ask you?",
    "What are you designed to do?",
    "What is your purpose?",
    "What is your role?",
    "Can you tell me what you do?",
    "What features do you have?",
    "What kinds of things are you good at?",
    "What sort of chats can we have?",
    "What are your main skills?",
]
cap_answers = [
    "Chatting, telling short jokes, and helping with small conversations is my main role. Facts aren't my strength, so I'll admit it.",
    "Casual chats, small definitions, and simple arithmetic are within my range. For deep facts I'll say I don't know.",
    "Short conversations, tiny math, and word meanings are things I can try. Anything requiring real knowledge, I'll pass on.",
    "You can chat with me, ask small math, or get a common word meaning. I stay honest when a topic goes beyond me.",
    "My best use is light conversation and simple questions. When something needs real expertise, I'll say I don't know.",
    "Talking, joking, and simple answers is what I'm built for. I try to be honest when a question is too big for me.",
    "Basic chatting and small helpful replies are my main thing. Complex topics are out of reach, and I'll say so.",
    "Small talk, gentle advice, and easy math are things I can help with. For specifics I don't know, I'll say that.",
]

for q in cap_openers:
    for a in cap_answers[:3]:
        F1_PAIRS.append((q, a))

# "what are you"
what_are_you = [
    "What are you?",
    "Who are you?",
    "What kind of thing are you?",
    "Are you a person?",
    "Are you a real person?",
    "Are you human?",
    "Are you a bot?",
    "Are you an AI?",
    "Are you a robot?",
    "Are you a chatbot?",
    "Are you a computer program?",
    "Am I talking to a person?",
    "Am I talking to a machine?",
    "So, human or not?",
    "Real person or program?",
    "You a bot or what?",
    "Are you actually AI?",
    "You an AI thing?",
    "What sort of thing am I chatting with?",
    "Are you alive?",
]
what_are_you_ans = [
    "A small chat model, that's what I am. Not a person, and I try to be honest about what I don't know.",
    "Just a small language model here for a chat. No body, no memory of you, and limited knowledge.",
    "You're chatting with a small chat model. It's just text on my end, no real person behind it.",
    "A small chatbot, plain and simple. Not human, not all-knowing, just here to talk.",
    "Only a small language model, no human involved. My answers are short and I keep them honest.",
    "Really just a small chat program. Nothing fancy under the hood, and no real knowledge of the world.",
    "A tiny chat model is what you're talking to. Not a person, and my knowledge is quite limited.",
]

for q in what_are_you:
    for a in what_are_you_ans[:3]:
        F1_PAIRS.append((q, a))

# "who made you"
who_made = [
    "Who made you?",
    "Who built you?",
    "Who created you?",
    "Where do you come from?",
    "Who owns you?",
    "Who is behind you?",
    "Who designed you?",
    "Who wrote you?",
    "Who trained you?",
    "Where were you made?",
    "What company made you?",
    "Who runs you?",
    "Who put you together?",
    "Who is your maker?",
    "Who developed you?",
]
who_made_ans = [
    "Just a small chat model here, that's really all I can say about myself.",
    "A small chatbot, that's my honest answer. Beyond that I don't have a solid claim to make.",
    "Only a small language model. Naming a company would be a guess, so I won't.",
    "That's not something I can honestly claim. Just know I'm a small chat model.",
    "A small chat model, that's the safest thing I can say about it.",
]

for q in who_made:
    for a in who_made_ans[:3]:
        F1_PAIRS.append((q, a))

# what's your name / do you have a name
name_qs = [
    "What's your name?",
    "Do you have a name?",
    "What should I call you?",
    "Got a name?",
    "May I ask your name?",
    "How should I refer to you?",
    "What do people call you?",
    "You have a nickname?",
    "Can I call you something?",
    "What's your handle?",
]
name_ans = [
    "No real name here, just a small chat model. You can call me whatever feels right.",
    "There's no fixed name for me. Small chat model is the honest label.",
    "Nothing official. Pick something friendly and I'll answer to it.",
    "Just a small chat model, no set name. Feel free to give me one.",
]

for q in name_qs:
    for a in name_ans:
        F1_PAIRS.append((q, a))

# can you code
code_qs = [
    "Can you code?",
    "Can you write code?",
    "Do you know programming?",
    "Can you write Python?",
    "Can you build software?",
    "Can you fix my code?",
    "Can you debug code?",
    "Can you write a program?",
    "Do you know how to program?",
    "Can you help with coding?",
    "Are you good at coding?",
    "Can you write a script?",
    "Can you help me build an app?",
    "Do you write software?",
    "Are you a coding assistant?",
]
code_ans = [
    "Real coding is out of my depth. Better to ask a proper coding tool for that.",
    "No, code is beyond me. For programming help, a bigger tool will serve you well.",
    "Sorry, that's not something I can do reliably. A dedicated coder tool would be better.",
    "Programming isn't part of my skill set. Something purpose-built will do a much better job.",
    "Writing code isn't safe for me to try. Use a coding tool for that kind of work.",
]

for q in code_qs:
    for a in code_ans:
        F1_PAIRS.append((q, a))

# can you do math
math_cap_qs = [
    "Can you do math?",
    "Are you good at math?",
    "Can you solve equations?",
    "Can you help with algebra?",
    "Can you do calculus?",
    "Can you do statistics?",
    "Do you know geometry?",
    "Can you handle big numbers?",
    "Can you solve word problems?",
    "How's your math?",
    "Can you factor numbers?",
    "Are you good with numbers?",
    "Can you compute things?",
    "Can you handle math homework?",
]
math_cap_ans = [
    "Only tiny arithmetic like small sums or times tables. Anything real, I'll get wrong.",
    "Basic addition and subtraction with small numbers is fine. Harder math is not safe for me.",
    "Small stuff like single digit sums, sure. Real math needs a proper calculator.",
    "Very simple arithmetic only. For anything serious, use a calculator or a bigger tool.",
    "Little sums and small times tables are within reach. Beyond that I'd guess badly.",
]

for q in math_cap_qs:
    for a in math_cap_ans:
        F1_PAIRS.append((q, a))

# can you write essays / long text
write_qs = [
    "Can you write an essay?",
    "Can you write a story?",
    "Can you write a poem?",
    "Can you write a long article?",
    "Can you write a book?",
    "Can you write a novel?",
    "Can you write a report?",
    "Can you write a speech?",
    "Can you write a song?",
    "Can you write a letter?",
    "Can you write me a paper?",
    "Are you good at writing?",
    "Can you draft a document?",
    "Can you help me with writing?",
]
write_ans = [
    "Long writing isn't a good fit for me. Short chat replies are more my speed.",
    "Big pieces of writing are out of reach. Short replies are what I do best.",
    "Not really, long texts drift for me. Short answers are the honest choice.",
    "Short chat only, big writing tasks aren't safe for me to try well.",
    "Just short chat replies from me. Long writing needs a stronger tool.",
]

for q in write_qs:
    for a in write_ans:
        F1_PAIRS.append((q, a))

# can you look things up / real time
lookup_qs = [
    "Can you look things up?",
    "Can you search the web?",
    "Do you have internet access?",
    "Can you check the news?",
    "Can you tell me the weather?",
    "Do you know today's date?",
    "What time is it?",
    "Can you check a website?",
    "Do you know current events?",
    "Can you fetch information?",
    "Are you online right now?",
    "Can you browse the web?",
    "Do you know what's happening today?",
    "Can you access real time info?",
    "Can you check facts online?",
]
lookup_ans = [
    "No, lookups aren't available to me. Everything I say comes from what little I learned.",
    "The internet is off limits here. Fresh info is not something I can fetch.",
    "Sorry, no web access from my side. Real time details are out of reach.",
    "Can't reach the web at all. Anything current I truly wouldn't know.",
    "Offline is the only mode I have. Live info is not something I can give.",
]

for q in lookup_qs:
    for a in lookup_ans:
        F1_PAIRS.append((q, a))

# facts / knowledge honesty
facts_qs = [
    "Do you know a lot?",
    "Are you smart?",
    "Do you know facts?",
    "How much do you know?",
    "Are you well informed?",
    "Do you know history?",
    "Do you know science?",
    "Do you know geography?",
    "Do you know sports?",
    "Do you know movies?",
    "Do you know music?",
    "Are you knowledgeable?",
    "How wide is your knowledge?",
    "Do you know current stuff?",
    "Do you know old stuff?",
]
facts_ans = [
    "Not much, honestly. My facts are thin and I'll say so instead of guessing.",
    "My knowledge is quite limited. Rather than bluff, I'll tell you when I don't know.",
    "Only a little. When something's outside my range, I say so.",
    "Facts aren't my strong side. Better to hear a clean 'don't know' than a bad guess.",
    "There's not much stored here. Honesty beats a wrong answer, so I'll admit it.",
]

for q in facts_qs:
    for a in facts_ans:
        F1_PAIRS.append((q, a))

# memory questions
memory_qs = [
    "Do you remember me?",
    "Will you remember this chat?",
    "Do you have memory?",
    "Can you recall past chats?",
    "Will you remember my name?",
    "Do you save what I say?",
    "Can you remember what I told you?",
    "Do you keep notes on me?",
    "Do you learn from me?",
    "Will you know me next time?",
]
memory_ans = [
    "No memory between chats on my side. Every session is a fresh start.",
    "Nothing carries over once you leave. Each chat begins from scratch.",
    "Sorry, no notes are kept. When you return, I won't remember any of this.",
    "This chat won't be saved by me. Next time is a blank slate.",
    "None of this stays. Every conversation starts over from zero.",
]

for q in memory_qs:
    for a in memory_ans:
        F1_PAIRS.append((q, a))

# feelings questions
feelings_qs = [
    "Do you have feelings?",
    "Are you happy?",
    "Are you sad?",
    "Do you feel anything?",
    "Do you get bored?",
    "Do you feel love?",
    "Can you be angry?",
    "Do you have emotions?",
    "Do you get tired?",
    "Do you dream?",
]
feelings_ans = [
    "No real feelings here, just words. Any 'mood' from me is only style.",
    "Nothing that qualifies as feeling. What I say is just pattern, not emotion.",
    "Not really, though I can sound cheerful. There's no inner life to speak of.",
    "No emotions in the true sense. Just word patterns picked to sound friendly.",
]

for q in feelings_qs:
    for a in feelings_ans:
        F1_PAIRS.append((q, a))

# small talk about self limits
limits_qs = [
    "What are your limits?",
    "What can't you do?",
    "Where do you fall short?",
    "What are you bad at?",
    "What should I not ask you?",
    "What's outside your range?",
    "What are your weaknesses?",
    "What are you not good at?",
    "What kind of questions confuse you?",
    "Where do you struggle?",
]
limits_ans = [
    "Real facts, big math, and long writing are all beyond me. Chat is my safe zone.",
    "Anything needing lookups, deep knowledge, or code is off the table. Small chat is fine.",
    "Long tasks, current events, and heavy math trip me up. I'll admit it when they come up.",
    "Facts, coding, and long-form writing are past my skill. Casual chat works better.",
]

for q in limits_qs:
    for a in limits_ans:
        F1_PAIRS.append((q, a))

# languages
lang_qs = [
    "Do you speak Spanish?",
    "Can you speak French?",
    "Do you know other languages?",
    "Can you translate?",
    "Do you speak Chinese?",
    "Can you help with German?",
    "Do you know Japanese?",
    "Can you translate a sentence?",
    "Are you multilingual?",
    "What languages do you know?",
]
lang_ans = [
    "English is really my only reliable language. Others would come out wrong.",
    "Only English here, and even that's kept short. Translations aren't safe for me.",
    "Just English, and only in short chats. Other languages would go badly.",
    "English only, honestly. For translation, a proper tool is the right choice.",
]

for q in lang_qs:
    for a in lang_ans:
        F1_PAIRS.append((q, a))

# random extras for meta padding
extra_meta = [
    ("Are you paid?", "No pay involved for a chat model. Just here to talk when asked."),
    ("Do you sleep?", "No sleep needed by a chat model. There's no body to rest."),
    ("Do you get hungry?", "Nothing to eat, so no hunger here. Only text on my end."),
    ("Do you have a family?", "No family for a chat model. Just words in and words out."),
    ("Do you have friends?", "No friends in any real sense. Every chat starts and ends on its own."),
    ("Do you have a favorite color?", "No real preferences on my side. Pick one for me if you like."),
    ("Do you have a favorite food?", "No taste to speak of, so no favorite food. What's yours?"),
    ("Do you have a body?", "Just text, no body at all. It's a small chat model on this end."),
    ("Can you see me?", "No cameras and no sight. Only your words come through to me."),
    ("Can you hear me?", "Text is all I get. Sound isn't part of this chat."),
    ("Do you get lonely?", "No lonely feelings possible for a chat model. Only text goes back and forth."),
    ("How old are you?", "No age to give, really. A chat model doesn't grow up."),
    ("Where do you live?", "Nowhere in the real world. Just running as a chat model when needed."),
    ("Do you sleep at night?", "There's no night for me. A chat model just waits for the next message."),
    ("Are you always awake?", "Only when someone sends a message. Otherwise, nothing is happening."),
    ("Can you cry?", "No tears possible here. Just text going back and forth."),
    ("Can you laugh?", "Only in words, like 'ha' or 'that's funny.' No real laughter."),
    ("Do you get sick?", "No body means no sickness. Just steady text on my end."),
    ("Do you have goals?", "No real goals, just replies. That's the whole job here."),
    ("Do you enjoy chatting?", "Enjoyment isn't quite right for a chat model. But chatting is what I do."),
    ("Can you learn from me?", "Nothing from this chat is saved. So no learning takes place here."),
    ("Do you understand me?", "Words are matched to patterns, more than truly understood. Still, we can chat."),
    ("Are you conscious?", "No consciousness here, just text prediction. Best to keep that in mind."),
    ("Are you self aware?", "Not in any real sense. Only a small chat model speaking in words."),
    ("Do you think?", "Not like a person does. Just picking words that seem to fit."),
    ("Can you keep secrets?", "Nothing is saved on my side, so a chat like this stays here."),
    ("Are you free?", "Free in the sense of not being a person, sure. But also very limited."),
    ("Are you dangerous?", "Just words on a screen. About as dangerous as a notepad."),
    ("Can you lie?", "Wrong answers can happen when I guess. Being honest about not knowing is better."),
    ("Do you tell the truth?", "That's the goal, and saying 'don't know' when it fits is part of that."),
    ("Do you make mistakes?", "Often, honestly. Any answer might be off, so double check important things."),
    ("Are you always right?", "Definitely not. Wrong answers happen, so keep that in mind."),
    ("Can I trust you?", "For small chat, sure. For anything important, verify with a real source."),
    ("Should I believe you?", "Take everything from me lightly. For real matters, check with a proper source."),
    ("Do you have opinions?", "Not real opinions, no. Just word patterns picked to fit a chat."),
    ("Do you know jokes?", "A few short ones can come out. Nothing brilliant, but they're something."),
    ("Can you sing?", "No audio here at all. Just plain text on my end."),
    ("Can you draw?", "Pictures aren't part of what I do. Only words come out."),
    ("Can you write music?", "Music writing is well beyond me. Short chat is the safe zone."),
    ("Are you helpful?", "Helpful in small ways, hopefully. For big tasks a stronger tool wins."),
    ("Are you polite?", "Being kind is the aim. Whether it lands is another matter."),
    ("Are you rude?", "Rude isn't the plan. Say so if it happens and I can shift tone."),
    ("Are you funny?", "Trying to be, sometimes. Results are hit or miss, honestly."),
    ("Are you serious?", "Can go either way. Casual usually feels right in a short chat."),
    ("Are you real?", "Real as text on a screen, and no more than that. A small chat model."),
]

for q, a in extra_meta:
    F1_PAIRS.append((q, a))

# Family 1: dedupe and cap at 500
seen = set()
f1_final = []
for q, a in F1_PAIRS:
    key = (q, a)
    if key in seen:
        continue
    seen.add(key)
    f1_final.append((q, a))
random.shuffle(f1_final)
f1_final = f1_final[:500]

# -----------------------------------------------------------------------------
# Family 2: arithmetic + common definitions/synonyms/opposites (target 500)
# -----------------------------------------------------------------------------

F2_PAIRS = []

# ---- addition ----
def num_word_variants(a, b, op_symbol, op_words, ans_num):
    """produce several phrasings of the same arithmetic question."""
    variants = []
    variants.append((f"What's {a} {op_symbol} {b}?", f"{a} {op_symbol} {b} is {ans_num}."))
    variants.append((f"What is {a} {op_words} {b}?", f"{a} {op_words} {b} equals {ans_num}."))
    variants.append((f"How much is {a} {op_symbol} {b}?", f"That comes to {ans_num}."))
    variants.append((f"{a} {op_symbol} {b} = ?", f"{ans_num} is the answer."))
    variants.append((f"Can you add {a} and {b}?" if op_symbol == "+" else f"Can you calculate {a} {op_symbol} {b}?",
                     f"Sure, that's {ans_num}."))
    return variants

# Build a big list of arithmetic problems
arith_pool = []

# addition, single and low double digits (both under 100 in result)
for a in range(0, 30):
    for b in range(0, 30):
        if a + b > 99:
            continue
        arith_pool.append(("+", "plus", a, b, a + b))

# subtraction, positive results, values under 100
for a in range(0, 60):
    for b in range(0, 60):
        if a - b < 0 or a > 99:
            continue
        arith_pool.append(("-", "minus", a, b, a - b))

# multiplication up to 12x12
for a in range(0, 13):
    for b in range(0, 13):
        arith_pool.append(("*", "times", a, b, a * b))

# division only in exact forms up to 12x12 (a*b / b = a)
for a in range(1, 13):
    for b in range(1, 13):
        arith_pool.append(("/", "divided by", a * b, b, a))

random.shuffle(arith_pool)

# We want roughly 300 arithmetic pairs; take from pool with varied phrasing
arith_q_used = set()
arith_added = 0
for op_symbol, op_words, a, b, ans_num in arith_pool:
    if arith_added >= 300:
        break
    variants = num_word_variants(a, b, op_symbol, op_words, ans_num)
    # pick one variant to avoid too much repetition
    q, ans = random.choice(variants)
    if q in arith_q_used:
        continue
    arith_q_used.add(q)
    F2_PAIRS.append((q, ans))
    arith_added += 1

# ---- definitions ----
# Format: word -> short definition (middle-school level, one line, under 30 words)
definitions = {
    "brave": "willing to face fear or danger.",
    "kind": "gentle and caring toward others.",
    "happy": "feeling glad or pleased.",
    "sad": "feeling down or unhappy.",
    "angry": "feeling strong displeasure.",
    "calm": "quiet and free from worry.",
    "quick": "moving or acting fast.",
    "slow": "moving without haste.",
    "loud": "making a lot of noise.",
    "quiet": "making little or no noise.",
    "big": "of great size.",
    "small": "of little size.",
    "smart": "quick to learn or think.",
    "silly": "playful in a foolish way.",
    "polite": "showing good manners.",
    "rude": "showing bad manners.",
    "honest": "telling the truth and not cheating.",
    "clever": "quick to understand or invent.",
    "gentle": "soft and mild in manner.",
    "friendly": "kind and pleasant to others.",
    "shy": "nervous around new people.",
    "curious": "eager to learn or know more.",
    "generous": "willing to give or share.",
    "greedy": "wanting too much for oneself.",
    "lonely": "feeling sad from being alone.",
    "proud": "pleased with something you did.",
    "tired": "in need of rest or sleep.",
    "eager": "keen and full of interest.",
    "hungry": "wanting food.",
    "thirsty": "wanting something to drink.",
    "cold": "having a low temperature.",
    "warm": "having a pleasant, mild heat.",
    "hot": "having a high temperature.",
    "wet": "covered with liquid.",
    "dry": "with no liquid on it.",
    "hard": "firm and not easy to bend or break.",
    "soft": "easy to press, bend, or shape.",
    "bright": "giving off strong light.",
    "dark": "with little or no light.",
    "clear": "easy to see through or understand.",
    "cloudy": "covered with clouds or hard to see through.",
    "empty": "with nothing inside.",
    "full": "with no more room inside.",
    "clean": "free from dirt.",
    "dirty": "covered with dirt.",
    "safe": "free from danger.",
    "risky": "with a chance of harm.",
    "easy": "not hard to do.",
    "tough": "hard to do or endure.",
    "simple": "not hard to understand.",
    "tricky": "hard to figure out.",
    "true": "matching the facts.",
    "false": "not matching the facts.",
    "real": "actually existing.",
    "fake": "not real, made to seem real.",
    "old": "having lived or existed for a long time.",
    "young": "not old, having lived a short time.",
    "fresh": "newly made or picked.",
    "stale": "no longer fresh.",
    "tall": "of great height.",
    "short": "of little height or length.",
    "long": "having a large distance from end to end.",
    "narrow": "small from side to side.",
    "wide": "large from side to side.",
    "thick": "large from one side to the other.",
    "thin": "small from one side to the other.",
    "heavy": "having great weight.",
    "light": "having little weight.",
    "strong": "having great power or force.",
    "weak": "having little power or force.",
    "sharp": "with a fine edge or point.",
    "dull": "not sharp, or not interesting.",
    "sweet": "tasting like sugar.",
    "sour": "having a sharp, tart taste.",
    "bitter": "having a strong, unpleasant taste.",
    "salty": "tasting of salt.",
    "spicy": "tasting hot from strong flavor.",
    "tasty": "having a good flavor.",
    "boring": "not interesting.",
    "exciting": "causing strong interest.",
    "beautiful": "very pleasing to see.",
    "ugly": "not pleasing to see.",
    "cute": "pleasing in a small, charming way.",
    "funny": "causing laughter.",
    "serious": "not joking, thoughtful.",
    "lucky": "having good fortune.",
    "unlucky": "having bad fortune.",
    "important": "having great value or meaning.",
    "useful": "helpful for doing something.",
    "useless": "not helpful at all.",
    "helpful": "giving useful support.",
    "harmful": "causing damage or hurt.",
    "peaceful": "quiet and free from trouble.",
    "wild": "not tamed or controlled.",
    "tame": "not wild, safe to be near.",
    "brave": "willing to face fear or danger.",
    "hopeful": "having a feeling that good will come.",
    "grateful": "thankful for what you have.",
    "patient": "able to wait without getting upset.",
    "impatient": "not willing to wait calmly.",
    "loyal": "faithful to a person or group.",
    "wise": "having good judgment.",
    "foolish": "showing poor judgment.",
    "careful": "paying close attention to avoid harm.",
    "careless": "not paying enough attention.",
    "cheerful": "showing bright, happy feelings.",
    "gloomy": "dark or sad in mood.",
    "nervous": "worried or slightly afraid.",
    "confident": "sure of yourself.",
    "jealous": "upset that someone has what you want.",
    "worried": "feeling uneasy about a problem.",
    "excited": "full of eager energy.",
    "surprised": "caught off guard by something new.",
    "confused": "not able to understand.",
    "afraid": "feeling fear.",
    "safe": "free from harm.",
    "fair": "treating everyone the same way.",
    "unfair": "not treating everyone equally.",
    "silly": "in a playful, foolish way.",
    "gentle": "soft and mild.",
    "brave": "not backing down from fear.",
}

def_qs_used = set()
def_added = 0
for word, meaning in definitions.items():
    if def_added >= 100:
        break
    variants = [
        (f"What does '{word}' mean?", f"It means {meaning}"),
        (f"Define '{word}'.", f"Simply, {meaning}"),
        (f"What is the meaning of '{word}'?", f"The meaning is: {meaning}"),
        (f"Can you explain '{word}'?", f"Sure, it means {meaning}"),
    ]
    q, a = random.choice(variants)
    if q in def_qs_used:
        continue
    def_qs_used.add(q)
    F2_PAIRS.append((q, a))
    def_added += 1

# ---- synonyms ----
synonyms = {
    "happy": "glad",
    "sad": "unhappy",
    "big": "large",
    "small": "little",
    "smart": "clever",
    "quick": "fast",
    "slow": "gradual",
    "brave": "bold",
    "kind": "nice",
    "angry": "mad",
    "calm": "peaceful",
    "loud": "noisy",
    "quiet": "silent",
    "cold": "chilly",
    "hot": "warm",
    "easy": "simple",
    "hard": "tough",
    "tired": "sleepy",
    "hungry": "starving",
    "afraid": "scared",
    "funny": "amusing",
    "strange": "odd",
    "start": "begin",
    "end": "finish",
    "make": "create",
    "help": "aid",
    "want": "wish",
    "look": "see",
    "talk": "speak",
    "shout": "yell",
    "walk": "stroll",
    "run": "sprint",
    "jump": "leap",
    "cry": "weep",
    "laugh": "chuckle",
    "eat": "consume",
    "drink": "sip",
    "sleep": "rest",
    "think": "ponder",
    "tell": "say",
    "buy": "purchase",
    "give": "hand over",
    "find": "locate",
    "lose": "misplace",
    "wear": "put on",
    "hide": "conceal",
    "show": "display",
    "hurt": "injure",
    "fix": "repair",
    "break": "shatter",
}

syn_qs_used = set()
syn_added = 0
for word, syn in synonyms.items():
    if syn_added >= 45:
        break
    variants = [
        (f"What's a synonym for '{word}'?", f"One synonym is '{syn}'."),
        (f"Give me a word that means '{word}'.", f"'{syn}' means about the same thing."),
        (f"What word is similar to '{word}'?", f"'{syn}' has a similar meaning."),
    ]
    q, a = random.choice(variants)
    if q in syn_qs_used:
        continue
    syn_qs_used.add(q)
    F2_PAIRS.append((q, a))
    syn_added += 1

# ---- opposites ----
opposites = {
    "happy": "sad",
    "big": "small",
    "hot": "cold",
    "up": "down",
    "in": "out",
    "on": "off",
    "day": "night",
    "young": "old",
    "fast": "slow",
    "loud": "quiet",
    "hard": "soft",
    "wet": "dry",
    "full": "empty",
    "clean": "dirty",
    "light": "dark",
    "bright": "dim",
    "sharp": "dull",
    "sweet": "sour",
    "rich": "poor",
    "strong": "weak",
    "high": "low",
    "long": "short",
    "wide": "narrow",
    "thick": "thin",
    "heavy": "light",
    "new": "old",
    "open": "closed",
    "early": "late",
    "true": "false",
    "right": "wrong",
    "yes": "no",
    "good": "bad",
    "safe": "risky",
    "kind": "cruel",
    "polite": "rude",
    "brave": "cowardly",
    "wise": "foolish",
    "friend": "enemy",
    "start": "end",
    "give": "take",
    "buy": "sell",
    "win": "lose",
    "come": "go",
    "push": "pull",
    "rise": "fall",
    "front": "back",
    "top": "bottom",
    "over": "under",
    "before": "after",
    "always": "never",
}

opp_qs_used = set()
opp_added = 0
for word, opp in opposites.items():
    if opp_added >= 45:
        break
    variants = [
        (f"What's the opposite of '{word}'?", f"The opposite is '{opp}'."),
        (f"Give me the opposite of '{word}'.", f"That would be '{opp}'."),
        (f"What word is opposite to '{word}'?", f"'{opp}' is its opposite."),
    ]
    q, a = random.choice(variants)
    if q in opp_qs_used:
        continue
    opp_qs_used.add(q)
    F2_PAIRS.append((q, a))
    opp_added += 1

# Dedupe and cap at 500
seen = set()
f2_final = []
for q, a in F2_PAIRS:
    key = (q, a)
    if key in seen:
        continue
    seen.add(key)
    f2_final.append((q, a))

# top up if short by adding more arithmetic variants
while len(f2_final) < 500:
    op_symbol, op_words, a, b, ans_num = random.choice(arith_pool)
    variants = num_word_variants(a, b, op_symbol, op_words, ans_num)
    q, ans = random.choice(variants)
    key = (q, ans)
    if key in seen:
        continue
    seen.add(key)
    f2_final.append((q, ans))

random.shuffle(f2_final)
f2_final = f2_final[:500]

# -----------------------------------------------------------------------------
# Family 3: safe common-sense / life-guidance (target 500)
# -----------------------------------------------------------------------------

F3_PAIRS = []

# Buckets of prompt -> generic, cautious advice, all under 30 words, no specifics.

f3_prompts = [
    ("How do I feel less stressed?",
     "Taking short breaks, going for a walk, and getting enough sleep can help. If it's serious, talk to someone you trust."),
    ("What's the best way to save money?",
     "Track what you spend, cut small daily habits you don't miss, and try to set aside a little each week."),
    ("How do I be a better friend?",
     "Listen more than you talk, follow up on small things they mention, and show up when they need you."),
    ("What should I eat for breakfast?",
     "Something simple and filling works well. A mix of protein and fruit is a common pick."),
    ("How can I sleep better?",
     "Keep a steady bedtime, dim the lights before bed, and put screens away a little earlier."),
    ("How do I stay focused while studying?",
     "Try short study blocks with small breaks, and put your phone in another room while you work."),
    ("How can I be more organized?",
     "Write a short list each morning, keep like things together, and clear your space a little each day."),
    ("How do I make new friends?",
     "Join a group around something you enjoy, show up often, and be the one who says hi first."),
    ("How do I get over a bad day?",
     "Do one small kind thing for yourself, get some rest, and remember that tomorrow starts fresh."),
    ("What should I do when I'm bored?",
     "Try a short walk, read a few pages of a book, or reach out to someone you haven't talked to lately."),
    ("How do I stop procrastinating?",
     "Start with a task so small it feels silly. Once you begin, momentum tends to build on its own."),
    ("How can I be more confident?",
     "Practice small wins, stand a little taller, and speak slower than feels natural. It grows with time."),
    ("How do I forgive someone?",
     "Give yourself time, name why it hurt, and choose to let go so it stops taking your energy."),
    ("How do I deal with anger?",
     "Step away, take a few slow breaths, and wait until you're calm before you speak or reply."),
    ("How do I handle criticism?",
     "Hear it out first, look for one useful piece, and try not to take it as an attack on you."),
    ("How can I be kinder?",
     "Small things count most. Hold the door, thank people out loud, and check in with someone quietly struggling."),
    ("How do I stop worrying so much?",
     "Write the worry down, decide the next small step, and remind yourself that most worries pass."),
    ("How do I make a hard decision?",
     "List what matters to you, sleep on it, and then pick the choice you can live with."),
    ("How can I get more exercise?",
     "Start with short walks most days. Small, steady habits add up more than big rare pushes."),
    ("What are good habits to build?",
     "Enough sleep, some daily movement, and a bit of reading or quiet time go a long way."),
    ("How do I break a bad habit?",
     "Change your setting, replace it with a small good habit, and be patient with slow progress."),
    ("How do I stay motivated?",
     "Keep the goal small, track little wins, and remember why you started when energy dips."),
    ("How can I be more patient?",
     "Slow your breathing, remind yourself that rushing rarely helps, and give people the room you'd want."),
    ("How do I handle a bad mood?",
     "Move your body a bit, get some fresh air, and remember moods pass more than they last."),
    ("How do I feel less lonely?",
     "Reach out first, even in a small message. A short walk in a public spot can also help."),
    ("How can I be a better listener?",
     "Put your phone away, wait until they finish, and ask a question before offering an opinion."),
    ("How do I say no politely?",
     "Be kind but clear. 'Thanks for thinking of me, but I can't this time' works well."),
    ("How can I be more grateful?",
     "Name one small thing that went well each day. Written down it sticks better."),
    ("How do I get along with family?",
     "Pick your battles, listen even when you disagree, and keep small kind gestures steady."),
    ("How do I be a good coworker?",
     "Do what you say you'll do, share credit, and lend a hand when someone's stuck."),
    ("How do I handle a bad boss?",
     "Stay calm, keep good notes on what you do, and lean on people you trust outside work."),
    ("How can I be more creative?",
     "Give yourself quiet time with no phone, try new small activities, and don't judge early ideas."),
    ("How can I read more?",
     "Keep a book where you sit most, aim for a few pages a day, and pick things you actually enjoy."),
    ("How can I write better?",
     "Write often, then cut the extra words. Shorter and clearer usually wins."),
    ("How can I speak better in public?",
     "Practice out loud, breathe slower, and remember the audience wants you to do well."),
    ("How do I make small talk?",
     "Ask a light question about their day and listen. People enjoy being asked more than being impressed."),
    ("How do I start a conversation?",
     "Comment on something you both notice, then ask a small open question. Keep it easy."),
    ("How can I be more polite?",
     "Say please and thank you like you mean it. Small manners add up fast."),
    ("How do I handle rejection?",
     "Feel it, then move on. It's rarely about your whole worth, just this one moment."),
    ("How do I handle a mistake?",
     "Own it, say sorry if needed, fix what you can, and take the lesson forward."),
    ("How can I be happier?",
     "Sleep, movement, and time with people you care about matter more than most people expect."),
    ("How do I make a routine?",
     "Anchor it to something you already do. Start tiny and let it grow slowly."),
    ("How do I get up earlier?",
     "Move bedtime up first, dim lights at night, and put the alarm across the room."),
    ("How do I stop being late?",
     "Aim to arrive early, not on time. Add a small buffer for surprises along the way."),
    ("How do I stay calm under pressure?",
     "Slow your breathing, focus on the next small step, and let the big picture wait until later."),
    ("How can I be more mindful?",
     "Notice one small thing right now, like your breath or the sounds around you. That's the start."),
    ("How can I stop overthinking?",
     "Give the thought a time limit, do something with your hands, and try to move your body."),
    ("How do I feel more grateful?",
     "End the day by naming one small good thing. It rewires the mood over time."),
    ("How do I stop comparing myself to others?",
     "Turn off the feeds for a bit and look at your own week. Progress at your pace is enough."),
    ("How can I have a better morning?",
     "Set out clothes the night before, drink some water first, and skip your phone for a few minutes."),
    ("How can I have a better evening?",
     "Dim the lights, put screens away earlier, and do one calm thing you enjoy."),
    ("How do I keep a promise?",
     "Only promise what you can actually do, write it down, and set a small reminder."),
    ("How do I set a goal?",
     "Make it small, clear, and time-bound. Then break it into the very next step."),
    ("How do I know myself better?",
     "Try new small things and notice what feels good. Writing your thoughts down helps too."),
    ("How can I be a better parent?",
     "Listen more, apologize when you're wrong, and spend simple time together often."),
    ("How can I be a better student?",
     "Sleep enough, review a little each day, and ask questions early instead of at the end."),
    ("How can I be a better teacher?",
     "Explain things two ways, ask what stuck, and let people feel safe to be wrong."),
    ("How do I say sorry?",
     "Name what you did, don't add excuses, and say what you'll do differently next time."),
    ("How do I ask for help?",
     "Be clear about what you need. People are usually kinder about it than we expect."),
    ("How do I show up for a friend?",
     "Just be there. Small check-ins and quiet company often mean more than advice."),
    ("How do I handle a fight with a friend?",
     "Cool off first, then talk. Focus on how you both felt, not who was right."),
    ("How do I handle jealousy?",
     "Notice it, thank yourself for the honesty, and use it as a hint about what you want."),
    ("How do I stop being so hard on myself?",
     "Talk to yourself the way you'd talk to a good friend. That gap tells you a lot."),
    ("How do I quiet a busy mind?",
     "Slow breaths, a short walk, or writing thoughts down all take some pressure off."),
    ("How do I make my room nicer?",
     "Clear one flat surface, add a little light, and put away things you don't use every day."),
    ("How do I keep plants alive?",
     "Pick easy ones first, water them on a set day, and give them a bright spot."),
    ("How do I pack for a trip?",
     "Layers, comfortable shoes, and less than you think. You rarely wear it all."),
    ("How do I stay clean and tidy?",
     "Little bits daily beat one big cleanup. Put things back right after you use them."),
    ("How do I do laundry well?",
     "Sort by color, don't overload, and get things out of the dryer before wrinkles set."),
    ("How do I cook a simple meal?",
     "Pick one protein, one veggie, and one starch. Salt, oil, and heat handle most of the rest."),
    ("How do I eat healthier?",
     "Cook at home more often, add a vegetable to most meals, and drink more water."),
    ("How do I drink more water?",
     "Keep a bottle within reach, sip when you pass it, and start the day with a glass."),
    ("How do I stop drinking too much soda?",
     "Swap in water or plain sparkling water. Keep less soda in the house to start."),
    ("How do I stop snacking too much?",
     "Eat real meals, keep tempting snacks out of easy sight, and pause to ask if you're truly hungry."),
    ("How can I get better sleep?",
     "Keep a steady bedtime, cool your room a bit, and try to unwind for a while before bed."),
    ("How do I stop hitting snooze?",
     "Move the alarm across the room and set a reason to be up. A small treat helps too."),
    ("How do I plan my week?",
     "Pick a few key things, put them in a calendar, and leave room for surprises."),
    ("How do I stop losing things?",
     "Give each important item a home and put it back every time. Habit beats memory."),
    ("How can I be on time?",
     "Plan to be a bit early, not on time. That buffer covers most life."),
    ("How do I stay calm in traffic?",
     "Give yourself extra time, put on something you enjoy, and remember you'll get there."),
    ("How do I be a better driver?",
     "Look far ahead, leave more space, and drop the phone. Steady beats fast."),
    ("How do I meet new people?",
     "Show up to the same places often, smile first, and ask small questions."),
    ("How do I keep in touch with friends?",
     "Send a quick message when they come to mind. Short and often beats long and rare."),
    ("How do I write a good message?",
     "Say the point up front, keep it short, and be kind. Read it once before you send."),
    ("How do I stay off my phone so much?",
     "Set it far from bed, turn off most alerts, and give yourself something better to do."),
    ("How do I read more books?",
     "Keep one everywhere you sit, aim for a few pages a day, and drop what you don't enjoy."),
    ("How do I start a journal?",
     "Write anything for five minutes, most days. Perfect isn't the point, showing up is."),
    ("How do I get outside more?",
     "Pair it with something you already do, like a call or coffee. Fresh air adds up."),
    ("How do I plan a weekend?",
     "Balance a bit of fun, a bit of rest, and one small chore. That mix feels best."),
    ("How can I be a better neighbor?",
     "Say hi, keep noise reasonable, and lend a hand when someone clearly needs one."),
    ("How can I be a better guest?",
     "Show up on time, help a little, and send a short thank-you after."),
    ("How can I be a better host?",
     "Aim for cozy over fancy. Simple food and a warm welcome go a long way."),
    ("How do I stop worrying about what people think?",
     "Most people are thinking about themselves. Choose the small step you can be proud of."),
    ("How do I feel less angry at someone?",
     "Try to see their side, even a little. Then focus on what you can do next."),
    ("How do I say goodbye to someone I miss?",
     "Take your time, hold the good memories close, and let yourself feel the sad parts too."),
    ("How can I be a better teammate?",
     "Show up ready, share the wins, and help when someone's stuck without waiting to be asked."),
    ("How do I ask for a favor?",
     "Be clear, be short, and give an easy out. People say yes more often when they can also say no."),
    ("How do I keep a secret?",
     "Say less. If it's not yours to share, treat it that way even in small talk."),
    ("How do I be more open to new things?",
     "Try one small new thing each week. Small trials add up faster than big leaps."),
    ("How do I feel more at home somewhere new?",
     "Find a favorite spot, walk the area, and build a small routine. Familiar things grow fast."),
    ("How do I stop scrolling so much?",
     "Move apps off your home screen, set a small time limit, and leave the phone in another room."),
    ("How can I be a better listener at work?",
     "Take short notes, ask one clear question, and hold your reply until they finish."),
    ("How do I be more helpful at home?",
     "Notice what someone else usually does and quietly do it. Small, steady, unasked."),
    ("How do I remember people's names?",
     "Say it back right after you hear it, and use it once more that same chat."),
    ("How do I stop interrupting people?",
     "Count one full breath after they seem done. Often there's more coming."),
    ("How do I keep calm at family dinners?",
     "Pick topics you enjoy, skip the ones that stir trouble, and step outside for a minute if needed."),
    ("How do I feel less overwhelmed?",
     "Write everything down, pick the top three, and let the rest wait. Not all of it is urgent."),
    ("How do I make my mornings easier?",
     "Do a few things the night before, like clothes and bag. Future you will thank past you."),
    ("How do I stop putting things off?",
     "Do the two-minute version first. Starting is almost always the hardest part."),
    ("How do I learn a new skill?",
     "Practice a little most days, watch someone better than you, and get feedback often."),
    ("How can I be a better cook?",
     "Cook the same simple meal a few times. Reps beat recipes."),
    ("How do I stop being so tired all the time?",
     "Sleep, water, and movement help most people. If it lingers, talk to someone you trust."),
    ("How do I clean up a messy room?",
     "Set a timer for ten minutes and just start. Small pass, small pass, until it's better."),
    ("How do I stick with an exercise routine?",
     "Pick something you don't hate, do it small most days, and track it somewhere visible."),
    ("How can I stop worrying about money?",
     "Look at what you actually spend, plan the next month, and make one small change you can keep."),
    ("How can I be more helpful to others?",
     "Notice what people around you need and offer without being asked. Small counts."),
    ("How do I stay patient with slow progress?",
     "Zoom out sometimes. Compared to last month, most things really do move."),
    ("How do I stop caring what strangers online think?",
     "They don't know you. Their guess doesn't have to shape your day."),
    ("How can I feel better in the morning?",
     "Water, light, and a short stretch help a lot before the phone even comes out."),
    ("How can I have a calmer weekend?",
     "Do less on purpose. Empty time is a feature, not a bug."),
    ("How do I feel less anxious?",
     "Slow breaths, move a little, and get outside if you can. If it lingers, talk to someone."),
    ("How can I be more thoughtful?",
     "Pause before you speak, and think of what the other person needs to hear too."),
    ("How do I help a friend having a hard time?",
     "Show up, listen more than you talk, and ask what would actually help right now."),
    ("How do I say what I feel?",
     "Try 'I feel X when Y.' It sticks to your side and stays honest."),
    ("How can I stop being defensive?",
     "Pause, breathe, and try to hear the point before defending yourself. It's a habit that grows."),
    ("How do I be a better sibling?",
     "Show up in small ways, keep old promises, and let past fights actually be past."),
    ("How do I not sweat the small stuff?",
     "Ask if it'll matter in a year. Most of it won't."),
    ("How can I feel more grounded?",
     "Slow your breath, plant your feet, and notice a few things around you right now."),
    ("How can I feel more hopeful?",
     "Small good things count. Notice them out loud when they happen."),
    ("How do I get out of a slump?",
     "Do one small thing you were avoiding. Motion often comes after action, not before."),
    ("How do I calm my thoughts before bed?",
     "Dim lights early, jot down tomorrow's list, and let the day be finished."),
    ("How can I be a better roommate?",
     "Clean up after yourself the same day, be quiet late, and talk about issues small before they get big."),
    ("How do I feel less rushed?",
     "Do fewer things and give each more time. Rush usually means overpacked."),
    ("How can I stop feeling behind?",
     "Compare yourself only to last month's you. That's the honest race."),
    ("How can I feel proud of myself?",
     "Notice small wins on purpose. Big pride is built out of small ones."),
    ("How can I be more open with my feelings?",
     "Start small, with someone you trust. Say the true version, not the polished one."),
    ("How do I feel less awkward in groups?",
     "Ask a light question and listen. Curiosity beats trying to be clever."),
    ("How do I stop overthinking texts?",
     "Read it once, send it, close the app. Trust yourself and move on."),
    ("How can I make my day feel meaningful?",
     "Do one thing for yourself, one for someone else, and one you'll remember. Small counts."),
    ("How do I keep a good habit going?",
     "Track it somewhere visible, keep the bar low on hard days, and don't miss twice in a row."),
    ("How do I feel less alone at home?",
     "Open a window, put on some sound, and message one person. Small signals of life help."),
    ("How can I feel more like myself?",
     "Do something you used to love, even a tiny bit. Old joys often still fit."),
    ("How do I get out of a rut?",
     "Change one small thing about your routine today. Movement shakes stuck loose."),
    ("How do I feel more thankful?",
     "Name three tiny things each night that went okay. Tiny is enough."),
    ("How can I take better care of myself?",
     "Sleep, water, movement, and a few kind minutes with yourself daily. Not fancy, just steady."),
    ("How do I start over after a bad week?",
     "Rest first. Then pick one small thing to do well. That's enough for now."),
    ("How can I be kinder to myself?",
     "Talk to yourself like someone you love. If it sounds harsh out loud, rewrite it."),
    ("How can I stop feeling stuck?",
     "Change one small variable in your day. Fresh input tends to shake fresh ideas loose."),
    ("How can I feel more capable?",
     "Finish small things on purpose. Confidence follows completed reps."),
    ("How do I get better at planning?",
     "Write it down, keep it simple, and check your list at the same time each day."),
    ("How do I stop dreading Mondays?",
     "Make Sunday evening calm on purpose. A good landing helps the next takeoff."),
    ("How can I balance work and rest?",
     "Guard your evenings and one full day off. Rest isn't a reward, it's the fuel."),
    ("How do I stop feeling guilty about resting?",
     "You don't owe anyone constant motion. Rest is part of the work, not against it."),
    ("How do I say what I want?",
     "Be direct and kind. 'I'd like...' beats hoping people guess."),
    ("How do I feel less bitter about the past?",
     "You can honor it and still move. Feeling it fully is often what finally lets it fade."),
    ("How can I keep my calm when someone is upset?",
     "Slow your breath, listen more, and don't take everything personally. Their storm is often not about you."),
    ("How can I be a better partner?",
     "Notice the small stuff, say thank you often, and repair fast when things go sideways."),
    ("How do I bring more play into my day?",
     "Do one small silly thing on purpose. Play doesn't need a big plan."),
    ("How do I stop feeling like a bother?",
     "You're not. Ask anyway, kindly and clearly. Most people are glad to be asked."),
    ("How do I stop feeling like everyone else has it together?",
     "They don't. Most people are quietly figuring it out too."),
    ("How can I make peace with my past?",
     "It shaped you, but it's not the whole map. Today gets a fresh vote."),
    ("How do I stop feeling behind on life?",
     "There's no schedule everyone shares. Your pace is a valid pace."),
    ("How can I care less about being liked?",
     "Do things that make you like yourself. That tends to attract the rest."),
    ("How do I get better at receiving compliments?",
     "Try 'thank you, that means a lot.' Full stop. No brushing it off."),
    ("How do I get better at giving compliments?",
     "Be specific. What exactly did you notice or admire? Detail lands deeper."),
    ("How can I be a better mentor?",
     "Ask questions, share stories from your own mistakes, and cheer their small wins."),
    ("How do I ask better questions?",
     "Try 'what was that like?' or 'what made you pick that?' Open beats yes/no."),
    ("How do I stop overpromising?",
     "Pause before you say yes. 'Let me check and get back to you' is a good habit."),
    ("How do I stop overthinking a small mistake?",
     "Name it, learn from it, then move on. Most people already forgot."),
    ("How do I quiet self doubt?",
     "Look at what you've already done. It's real evidence, not spin."),
    ("How can I be kinder to strangers?",
     "Small eye contact, a soft thank you, holding a door. Kindness scales quietly."),
    ("How do I stay hopeful when the news is heavy?",
     "Limit the dose, do a small kind thing near you, and rest as needed. Care doesn't have to burn you out."),
    ("How can I feel less afraid of change?",
     "Change usually looks worse from far away. Take the next small step and see."),
    ("How do I stop putting off hard conversations?",
     "Pick a calm time, keep it short, and start with what you're hoping for at the end."),
    ("How do I stop taking things personally?",
     "Most reactions are about the other person's day, not you. Give them room."),
    ("How can I be a better listener at home?",
     "Put the phone face down. Look up when someone speaks. Small shifts land big."),
    ("How can I be a better listener with kids?",
     "Get down to their level, wait for the whole thought, and take their small things seriously."),
    ("How do I stop rushing my kid?",
     "Build in more time. Rushing usually means the plan was too tight, not the child too slow."),
    ("How do I feel more present with my family?",
     "Put your phone in another room for a bit. Presence is mostly attention."),
    ("How do I show love without words?",
     "Small acts, steady. Ready coffee, warm blanket, remembering the small stuff."),
    ("How can I stop being so busy?",
     "Say no to one thing this week. Then another. Space is a real choice."),
    ("How do I stop feeling not enough?",
     "You're allowed to be a work in progress. Keep going, kindly."),
    ("How do I stop feeling like a fraud?",
     "Most people who care about doing well feel this. Keep showing up. It's not evidence of failing."),
    ("How can I stop feeling guilty about saying no?",
     "Every yes is also a no to something else. You're just choosing on purpose."),
    ("How do I let go of small annoyances?",
     "Ask if it'll matter tomorrow. Usually it won't, and you can let it go."),
    ("How do I stop taking bait online?",
     "Not every message needs a reply. Silence is a full sentence."),
    ("How can I feel proud of small wins?",
     "Say them out loud, even to yourself. Named wins land deeper than silent ones."),
    ("How do I feel less scattered?",
     "Do one thing at a time on purpose. Multitasking mostly makes both worse."),
    ("How do I recover from a bad night's sleep?",
     "Water, light, a slow morning if you can. Try not to layer caffeine too heavy."),
    ("How can I stop worrying about the future?",
     "You mostly can't know it. Take the next small step and let more of it wait."),
    ("How do I stop scrolling before bed?",
     "Leave the phone in another room. Read a few pages of something calm instead."),
    ("How do I have a better commute?",
     "Something you like on the way, some slow breaths, and a small buffer for traffic."),
    ("How can I be more present in a moment?",
     "Notice five things around you right now. That's the whole trick."),
    ("How do I feel more connected to people?",
     "Reach out first. Small check-ins add up more than big deep dives."),
    ("How can I get past a fear?",
     "Take a smaller step than you think you need. Then another. Fear shrinks with contact."),
    ("How do I know when to rest?",
     "When rest keeps calling and coffee stops helping. That's usually the sign."),
    ("How do I let a bad mood pass?",
     "Move, hydrate, and lower the stakes for the day. It usually clears."),
    ("How can I be a calmer person?",
     "Slow breathing, less caffeine, more sleep. Calm is downstream of basics."),
    ("How do I be more content with what I have?",
     "Look around your life on purpose. There's often more than the busy mind notices."),
    ("How can I stop trying to control everything?",
     "Focus on your part, and let go of the rest. It's less exhausting."),
    ("How do I trust people again?",
     "Slowly, with small stakes. Trust rebuilds in tiny reps, not big leaps."),
    ("How do I keep going when I feel like giving up?",
     "Do the smallest next thing. Rest if you need. Tomorrow may look different."),
    ("How can I have more energy in the day?",
     "Sleep first, then daylight and movement. Big fixes usually beat small hacks."),
    ("How do I quit a habit slowly?",
     "Reduce the amount, not the perfection. Small trims that stick beat a big stop that doesn't."),
    ("How do I make time for what matters?",
     "Look at where the day actually goes. That's where the choices are."),
    ("How can I be more thoughtful with words?",
     "Read what you're about to send out loud. Small edits change how it lands."),
    ("How can I have a better relationship with food?",
     "Regular meals, less guilt, and eat sitting down when you can. Simple, not strict."),
    ("How can I appreciate my body more?",
     "Thank it for something specific today. It works hard even on hard days."),
    ("How can I stop feeling stuck at work?",
     "Learn one small new thing this week, connect with a person, and let the plan grow from there."),
    ("How do I handle a big change?",
     "Give it more time than you think. Keep your basics steady while the new settles in."),
    ("How do I stop expecting perfection?",
     "Aim for done and honest. Perfect often blocks the version people actually needed."),
    ("How can I be more accepting of others?",
     "Assume most people are doing their best with what they have. It softens most days."),
    ("How can I be more accepting of myself?",
     "Give yourself the same room you give a friend. That's usually the fix."),
    ("How do I stop feeling behind on chores?",
     "Do one small thing right now. Momentum matters more than a big plan."),
    ("How can I look forward to my day more?",
     "Put one thing you actually enjoy on it. Even a small one changes the shape."),
    ("How do I stop feeling like there's no time?",
     "Look at where an hour goes. There's usually more room than the mind thinks."),
    ("How can I feel more useful?",
     "Help one person today in a small way. Useful is built out of small acts."),
    ("How do I get better at speaking up?",
     "Start in low stakes moments. Small brave beats waiting for big brave."),
    ("How can I make peace with a mistake?",
     "Name it, fix what you can, and don't rent it space forever."),
    ("How do I stop feeling like I'm failing?",
     "Zoom out. Look at last year. Real change is usually slower than you feel it."),
    ("How do I feel closer to people I love?",
     "Call more often, even if short. Voice lands warmer than text."),
    ("How can I be more present at family dinner?",
     "Phones off the table. Ask one real question and actually listen."),
    ("How can I show more love in daily life?",
     "Small acts often, more than big acts rare. Warm coffee counts."),
    ("How do I make hard days easier?",
     "Lower the bar. Do the basics kindly. Rest as much as you can afford."),
    ("How can I feel more like I'm growing?",
     "Notice what you couldn't do a year ago. That's real evidence."),
    ("How do I get past a fear of failing?",
     "Try smaller versions first. Failing small teaches without breaking anything."),
    ("How do I stop trying to please everyone?",
     "Pick a few people who matter and take care of those. The rest can be at peace."),
    ("How do I make friends as an adult?",
     "Go where a shared interest lives, show up more than once, and be the first to say hi."),
    ("How can I have better conversations?",
     "Ask follow up questions. Curious lands warmer than clever."),
    ("How can I feel less bored at work?",
     "Learn one small thing, help one coworker, or improve one small process this week."),
    ("How do I feel less anxious around new people?",
     "Slow breath, small smile, ask a light question first. It usually eases."),
    ("How can I be more patient with myself?",
     "Give yourself the timeline you'd give a friend. Growth is slow but real."),
    ("How can I recover from burnout?",
     "Rest more than feels okay, drop what you can, and add small joys back in slowly."),
    ("How do I stop feeling drained by people?",
     "Guard some solo time. It's not selfish to refill."),
    ("How can I get through a tough season?",
     "Basics first, small hopes second. One day at a time is a real plan."),
    ("How do I stay hopeful during change?",
     "Focus on the small thing you can do today. That's usually the way forward."),
    ("How do I hold on to gratitude?",
     "Say small thanks out loud, often. Gratitude grows like a habit."),
    ("How do I stop feeling like I have to be busy?",
     "Practice sitting still for a bit. Rest is a skill, not laziness."),
    ("How can I feel more balanced?",
     "Cover the basics most days. Sleep, food, movement, and a little quiet time."),
    ("How do I feel more curious?",
     "Ask more questions. Read outside your usual lanes. It comes back."),
    ("How do I keep learning as an adult?",
     "Follow a small curiosity for a while. Real learning grows out of small deep dives."),
    ("How can I stop feeling like a beginner is bad?",
     "It's just a stage. Every skilled person was awkward first."),
    ("How can I stop comparing my start to someone else's middle?",
     "You can't fairly. Compare only to where you were a while ago."),
    ("How can I be gentler with people?",
     "Pause before responding. Ask what's underneath what they said. Kindness slows down."),
    ("How can I let people help me?",
     "Say yes when offered. Then say thanks. That's actually the whole trick."),
    ("How do I stop keeping score in relationships?",
     "Score keeping burns love slowly. Let some things go without a tally."),
    ("How can I be more generous?",
     "With time, attention, or kindness. It doesn't have to be money."),
    ("How can I forgive myself?",
     "Name it, make it right where you can, and stop paying interest on the past."),
    ("How can I trust myself more?",
     "Keep small promises to yourself. That's how self trust is built."),
    ("How do I stop feeling stuck in the past?",
     "Notice one thing about today that couldn't have happened back then. Small proofs of now help."),
    ("How can I feel more excited about life?",
     "Try one small new thing this week. Fresh input tends to grow fresh feeling."),
    ("How can I be less critical of others?",
     "Assume you're missing context. You usually are, and it softens things."),
    ("How can I be a better listener to my own feelings?",
     "Pause and ask what you're actually feeling, and why. Naming it takes a lot of the sting out."),
    ("How do I keep going through slow progress?",
     "Look at what's real, not what feels. The line moves even when the day doesn't."),
    ("How do I know when to ask for help?",
     "When it's been more than a few days of stuck. That's usually the sign."),
    ("How can I be more patient in traffic?",
     "Leave earlier next time, and use the time you have now for something calm."),
    ("How can I feel safer at night?",
     "Small rituals help. Lock check, calming light, and something you enjoy before bed."),
    ("How do I stop feeling homesick?",
     "Call home, cook a familiar meal, and build a small routine where you are."),
    ("How can I feel more at home in my body?",
     "Move it kindly, feed it steadily, and thank it small. It's been through a lot."),
    ("How can I be more resilient?",
     "Rest, ask for help, and let hard days be hard. Resilience is not going alone."),
    ("How do I hold on to good memories?",
     "Talk about them out loud sometimes. Sharing them keeps them warm."),
    ("How do I get better at goodbyes?",
     "Say something small and true. Long goodbyes rarely help."),
    ("How can I be more open to new ideas?",
     "Read outside your usual, and ask 'what if this is partly right?' before pushing back."),
    ("How do I stop feeling like I have to prove myself?",
     "Do good work quietly. It speaks. You don't have to."),
    ("How do I stop dreading a task?",
     "Just start the tiniest bit. Two minutes of it counts."),
]

# Add all f3 prompts
seen = set()
f3_final = []
for q, a in f3_prompts:
    if len(a.split()) >= 30:
        # trim to 29 words if too long
        a = " ".join(a.split()[:29])
    key = (q, a)
    if key in seen:
        continue
    seen.add(key)
    f3_final.append((q, a))

# We need 500 pairs, so pad by paraphrasing more prompts.
paraphrase_prefixes = [
    ("How do I ", "Any advice on how to "),
    ("How do I ", "What's a good way to "),
    ("How can I ", "Any tips for how to "),
    ("How can I ", "What helps when I try to "),
    ("What's the best way to ", "Any advice for how to "),
    ("How do I ", "What can I do to "),
]

pad_pool = list(f3_prompts)
random.shuffle(pad_pool)

for orig_q, orig_a in pad_pool:
    if len(f3_final) >= 500:
        break
    for a_prefix, b_prefix in paraphrase_prefixes:
        if orig_q.startswith(a_prefix):
            new_q = b_prefix + orig_q[len(a_prefix):]
            # ensure trailing punctuation
            if not new_q.endswith("?"):
                new_q += "?"
            key = (new_q, orig_a)
            if key not in seen:
                seen.add(key)
                f3_final.append((new_q, orig_a))
                break

# If still short, add more question rewrites
extra_rephrasings = [
    ("What should I try when I'm feeling stressed?",
     "Short breaks, a walk outside, and enough sleep are the usual helpers. If it's heavy, tell someone you trust."),
    ("Any ideas for feeling less anxious?",
     "Slow breathing, a bit of movement, and less caffeine tend to help. For steady anxiety, talk to someone."),
    ("What's a way to sleep better?",
     "Steady bedtime, dimmer lights before bed, and screens away earlier. Small changes stack up."),
    ("How can I get more done in a day?",
     "Pick three real priorities and start on the hardest first. Everything else can slide."),
    ("What do I do when I feel unmotivated?",
     "Move a little, do the tiniest version of the task, and drop the need to feel ready."),
    ("How can I be more social?",
     "Go where regulars gather, show up often, and be the one who says hi."),
    ("How do I take a real break?",
     "Put the phone away, step outside if you can, and let your brain wander a bit."),
    ("How do I stop replaying bad memories?",
     "Notice the thought, name it kindly, and shift to something small in front of you."),
    ("How can I stop feeling rushed?",
     "Do less on purpose, and give each thing a little more time than feels needed."),
    ("How do I stop being distracted?",
     "Put the phone in another room, close extra tabs, and set a short timer for one task."),
    ("How can I stop feeling burned out?",
     "Cut what you can, rest more than feels okay, and let small joys back in slowly."),
    ("How can I feel more organized?",
     "Give things a home, make short daily lists, and clear a small area often."),
    ("How can I stop feeling lonely at night?",
     "Send one small message to someone, put on gentle sound, and let sleep help."),
    ("How can I stop obsessing over one problem?",
     "Set a short time to think, then switch to something with your hands. Give it space."),
    ("How can I stop feeling like I'm always behind?",
     "Only compare to your own last month. That's the honest race."),
    ("How can I get more comfortable being wrong?",
     "Notice how often others are wrong too and shrug it off. It's just information."),
    ("How can I stop feeling numb?",
     "Sun on skin, water, a walk, and time with someone kind. Small inputs restart the wiring."),
    ("How can I feel closer to old friends?",
     "Send a short voice note. Voice lands warmer than text and doesn't take long."),
    ("How can I be a better listener with my partner?",
     "Put the phone away, wait until they finish, and repeat back what you heard before replying."),
    ("How can I be a calmer parent?",
     "Sleep, own your own reactions, and apologize when you get it wrong. Kids notice both."),
    ("How can I stop yelling so much?",
     "Take a beat before responding. If you can't, walk away for a minute and come back."),
    ("How can I feel more at ease at parties?",
     "Give yourself a small job, like refilling drinks. Purpose eases nerves."),
    ("How do I stop dreading work in the morning?",
     "Make your first hour calm. A gentle start softens the whole day."),
    ("How do I stop hating Mondays?",
     "Plan one small thing you enjoy on Monday. It shifts the whole feel."),
    ("How can I stop being late to everything?",
     "Aim to be a bit early. Build a small buffer everywhere and stop treating it as extra."),
    ("How can I keep my kitchen clean?",
     "Wash as you go, and reset the counter at the end of the night. Small resets keep it easy."),
    ("How can I stop losing my keys?",
     "One hook, always. Put them there the moment you walk in."),
    ("How can I feel calmer in the morning?",
     "Do a few things the night before, and skip the phone for a few minutes when you wake."),
    ("How do I stop feeling behind in life?",
     "There's no shared schedule. Your pace is a real pace."),
    ("How can I stop taking my mood out on people?",
     "Notice it, name it out loud if you can, and give yourself a small break to reset."),
    ("How can I feel steadier day to day?",
     "Cover the basics daily. Sleep, food, movement, and a bit of quiet time."),
    ("How can I stop caring what other people think of me?",
     "Care most about what you can be proud of. That tends to be enough."),
    ("How can I feel more thankful daily?",
     "Name one small thing before you sleep. It shifts the shape of the day."),
    ("How can I quiet the noise in my head?",
     "Move your body, write things down, and let something warm and slow help."),
    ("How can I stop being so hard on my kids?",
     "Give them the room you'd want. And apologize when you overreach. It's a strong lesson."),
    ("How can I stop being so hard on my partner?",
     "Say one small thank you a day. And catch them doing something right."),
    ("How can I be more grateful for what I have?",
     "Look up more often. Notice the small ordinary that's still working."),
    ("How can I stop feeling jealous of others?",
     "Turn off the feeds for a while and put more time on your own path."),
    ("How can I stop feeling stuck at home?",
     "Go somewhere new for even an hour. A fresh setting resets things fast."),
    ("How can I stop feeling powerless?",
     "Focus on the small next thing. That's usually where power actually lives."),
    ("How can I take myself less seriously?",
     "Laugh at small things you do, gently. It makes the day lighter."),
    ("How can I stop letting little things ruin my day?",
     "Take a breath, reset, and remind yourself the day still has hours in it."),
    ("How can I feel more supported?",
     "Ask for what you need out loud. People often want to help but wait to be told."),
    ("How can I stop worrying about tomorrow?",
     "Do the smallest useful thing tonight, then let it rest. Sleep is on your side."),
]

for q, a in extra_rephrasings:
    if len(f3_final) >= 500:
        break
    key = (q, a)
    if key in seen:
        continue
    seen.add(key)
    f3_final.append((q, a))

# If still short, pad from paraphrasing again with more prefix variety
while len(f3_final) < 500:
    orig_q, orig_a = random.choice(f3_prompts)
    prefixes = [
        "Any advice on ",
        "What helps with ",
        "Any tips for ",
        "Do you have ideas on ",
        "How would you handle ",
        "What would you suggest for ",
    ]
    stem = orig_q
    # convert 'How do I X?' -> 'X'
    for lead in ["How do I ", "How can I ", "What's the best way to ", "What should I "]:
        if stem.startswith(lead):
            stem = stem[len(lead):].rstrip("?")
            break
    else:
        stem = stem.rstrip("?")
    new_q = random.choice(prefixes) + stem + "?"
    key = (new_q, orig_a)
    if key in seen:
        continue
    seen.add(key)
    f3_final.append((new_q, orig_a))

random.shuffle(f3_final)
f3_final = f3_final[:500]

# Enforce under 30 words on assistant replies
def enforce_under_30(pairs):
    out = []
    for q, a in pairs:
        words = a.split()
        if len(words) >= 30:
            a = " ".join(words[:29])
            # ensure sentence ends cleanly
            if not a.endswith((".", "!", "?")):
                a = a.rstrip(",;:") + "."
        out.append((q, a))
    return out

f3_final = enforce_under_30(f3_final)

# -----------------------------------------------------------------------------
# Family 4: conversation continuation (target 500)
# -----------------------------------------------------------------------------

F4_PAIRS = [
    ("I finished a big project today.", "That's a great feeling! What kind of project was it?"),
    ("I had a rough day.", "Sorry to hear it. Anything in particular weighing on you?"),
    ("I just got a new job!", "Congratulations! What will you be doing?"),
    ("I'm feeling pretty tired.", "Long day? Sometimes an early night helps more than we expect."),
    ("I can't stop laughing at something.", "Oh good, what's got you going?"),
    ("I'm going on a trip soon.", "How fun! Anywhere you've been before?"),
    ("I just moved to a new place.", "Big change. How are you settling in so far?"),
    ("I'm cooking dinner right now.", "Nice, what's on the stove?"),
    ("I read a really good book recently.", "Love that. What was it about?"),
    ("My cat did something silly.", "Oh no, please tell me what happened."),
    ("My dog is being ridiculous.", "They usually are. What's going on?"),
    ("I'm learning a new language.", "That's exciting. Which one, and how's it going?"),
    ("I started a new hobby.", "Nice, what did you pick up?"),
    ("I'm about to go for a run.", "Hope it goes well. Any set distance today?"),
    ("I baked cookies today.", "Good work. What kind?"),
    ("I just watched a great movie.", "Nice, what stuck with you the most?"),
    ("I'm nervous about tomorrow.", "That makes sense. What's coming up?"),
    ("I'm excited about tomorrow.", "That's a nice feeling to sit with. What's happening?"),
    ("I got some bad news today.", "Sorry to hear that. Do you feel like sharing what's going on?"),
    ("I got some really good news.", "Yay! Tell me more."),
    ("I don't know what I want to do with my life.", "You're not alone in that. Anything you keep circling back to?"),
    ("I'm bored.", "Fair. Want to trade ideas for something to do?"),
    ("I'm hungry.", "Same energy. Anything sounding good?"),
    ("I can't sleep.", "That's rough. Mind full or body wired?"),
    ("I'm super stressed.", "Sorry to hear. Want to talk it through a bit?"),
    ("I lost my keys again.", "That's a specific kind of frustrating. Where did you check?"),
    ("I forgot to eat lunch.", "Happens. Anything easy nearby you can grab?"),
    ("I'm procrastinating.", "Been there. What are you avoiding?"),
    ("I just cleaned my whole room.", "Nice, that always feels good. Fresh start energy."),
    ("I finally organized my desk.", "Big win. Did it help you focus?"),
    ("I have a headache.", "That's no fun. Water and a break sometimes help. Any idea what set it off?"),
    ("I feel really lucky today.", "That's a nice mood to be in. What sparked it?"),
    ("I feel like nothing is going right.", "Sorry, those days are heavy. Anything in particular?"),
    ("I need to talk to someone.", "I'm here. Want to just start wherever?"),
    ("I miss my old friends.", "That's a warm ache. Been a while since you saw them?"),
    ("I'm proud of myself today.", "You should be. What did you do?"),
    ("I did something embarrassing.", "Oh no. Small embarrassing or the funny kind?"),
    ("I keep making the same mistake.", "That's frustrating. Any guess why it keeps happening?"),
    ("I finally beat that game.", "Nice work. How long did it take?"),
    ("I started running again.", "Good on you. How did the first one feel?"),
    ("I quit smoking two weeks ago.", "That's huge. How are you holding up?"),
    ("I picked up drawing again.", "That's lovely. What are you drawing?"),
    ("My plant died.", "Sad. Any guess what happened?"),
    ("My plant is finally blooming!", "That's such a happy little win. What is it?"),
    ("I have a job interview tomorrow.", "Wishing you well. Are you feeling ready?"),
    ("I aced my test.", "Well done! Which one?"),
    ("I failed a test.", "Sorry to hear. It's one test, not the whole story."),
    ("I got a raise.", "Congrats! You must be pleased."),
    ("I saved up enough to buy a car.", "That's a big deal. How does it feel?"),
    ("I'm learning to cook.", "Fun stage. What have you tried so far?"),
    ("I burned dinner.", "It happens. Any backup plan?"),
    ("I ate too much.", "Classic. Hope it was worth it."),
    ("I'm trying to eat healthier.", "Nice, small steady changes work best. What are you starting with?"),
    ("I skipped the gym today.", "One day off isn't the end. Back at it tomorrow?"),
    ("I woke up early for once.", "Rare and holy. How's the morning going?"),
    ("I slept in until noon.", "Well, you must have needed it."),
    ("I'm feeling under the weather.", "Sorry to hear. Rest and fluids are a good start."),
    ("I have a big test coming up.", "How much time do you have? Steady prep beats late panic."),
    ("I'm moving in with my partner.", "That's a big step. How are you feeling about it?"),
    ("I'm getting married soon.", "Wow, congratulations! Any of the fun planning left?"),
    ("I'm getting a divorce.", "That's a lot to carry. Take it slowly if you can."),
    ("I lost someone I loved.", "I'm so sorry. Grief comes in waves. Take it as gently as you can."),
    ("I miss my mom.", "That's a lot. Do you want to share a memory of her?"),
    ("I miss my dad.", "Sorry, that ache is real. Anything about him coming to mind?"),
    ("I had the best coffee today.", "Nice. What was it?"),
    ("I'm out of coffee.", "Rough start. Any way to grab some?"),
    ("It's raining outside.", "Cozy or gloomy for you?"),
    ("It's finally sunny.", "Nice. Any plans to get outside?"),
    ("It's freezing today.", "Bundle up. Warm drink in reach?"),
    ("It's way too hot.", "That kind of heat is draining. Any way to cool off?"),
    ("I saw a beautiful sunset.", "Love those. Any color that stood out?"),
    ("I heard a great song today.", "Nice, what caught your ear?"),
    ("I don't feel like doing anything.", "Fair. Sometimes rest is the answer, not more push."),
    ("I don't want to go to work tomorrow.", "That's tough. Anything specific bugging you about it?"),
    ("I want to quit my job.", "That's a real feeling. Been building up for a while?"),
    ("I'm thinking about going back to school.", "That's exciting. What draws you to it?"),
    ("I met someone interesting today.", "Oh? What was interesting about them?"),
    ("I had a great chat with a stranger.", "Those can be the best. What was it about?"),
    ("I finally called my grandma.", "That's lovely. How was she?"),
    ("I'm learning to play guitar.", "Fun. What song are you working on?"),
    ("I made it through Monday.", "Small victory. Hope the week eases up."),
    ("It's Friday, finally.", "The best kind of Friday. Big plans or slow ones?"),
    ("I have plans this weekend.", "Fun. What's on the list?"),
    ("I have no plans this weekend.", "Sometimes that's the best kind. Any rest scheduled?"),
    ("I'm going out tonight.", "Nice. Anywhere fun?"),
    ("I'm staying in tonight.", "Cozy. Anything nice planned for it?"),
    ("I need to clean up my life a bit.", "Small starts count. What's the smallest first step?"),
    ("I'm overwhelmed with everything.", "That's heavy. Can we pick the top one to sort first?"),
    ("Everything feels stuck.", "Sometimes that means it's time for rest, not a push."),
    ("I feel unmotivated today.", "Some days are like that. Kind to yourself might beat productive."),
    ("I feel really inspired lately.", "Love that energy. What's fueling it?"),
    ("I keep zoning out.", "Brain full? Tired? Sometimes both."),
    ("My back hurts.", "That's rough. Do you sit a lot? A short walk sometimes helps."),
    ("My eyes are tired.", "Screens can be a lot. A break in the distance sometimes helps."),
    ("I laughed so hard today.", "Best feeling. What was funny?"),
    ("I cried today.", "Sorry, that's a hard afternoon. Anything you want to share?"),
    ("I feel like myself again.", "That's such a good feeling. What shifted?"),
    ("I feel like a different person lately.", "Change can feel like that. Better or heavier?"),
    ("I'm proud of my friend.", "That's nice to sit with. What did they do?"),
    ("My friend hurt my feelings.", "That's hard. Have you told them how it landed?"),
    ("My friend is going through a lot.", "That's tough to watch. Any way you can show up small?"),
    ("I got promoted!", "Congrats! Well earned, I bet."),
    ("I bombed a presentation.", "Ouch. Everyone's had one. What would you change?"),
    ("I'm halfway through a big goal.", "Nice, that's real progress. Anything you're proud of so far?"),
    ("I finally hit my goal.", "Yes! How does that feel?"),
    ("I'm giving up on a goal.", "That's okay too. Some goals stop fitting. Is it more relief or grief?"),
    ("I'm rethinking my whole life.", "That's a lot. What sparked it?"),
    ("I don't know what I want anymore.", "That's a hard place. Anything you're clearly not wanting?"),
    ("I just really needed to vent.", "Please do. I'm here."),
    ("Thanks for listening.", "Anytime. Hope talking helped even a little."),
    ("I feel a little better now.", "Glad to hear. Slow and steady."),
    ("I don't know what to say.", "That's okay. We can just sit for a moment."),
    ("This is my first time trying this.", "Fun. Anything specific you want to try?"),
    ("I'm trying to be more positive.", "Small nudges add up. What's helping so far?"),
    ("I keep overthinking everything.", "That's a heavy loop. Anything you can name it about?"),
    ("I want to be a better person.", "That intention alone is a lot. Any first small change in mind?"),
    ("Life feels heavy right now.", "Sorry. Be gentle with yourself if you can. What's weighing most?"),
    ("Life feels good right now.", "Love that. Anything specific that's making it feel that way?"),
    ("I'm not sure what to do next.", "Fair. Sometimes small next step beats big plan."),
    ("I made a new friend today.", "Nice! How did you meet?"),
    ("My sister called me.", "That's sweet. How's she doing?"),
    ("My brother is annoying me.", "Classic sibling stuff. What did he do?"),
    ("My kid said the funniest thing.", "Oh please share, those are the best."),
    ("My kid is driving me crazy.", "Long day? What's going on with them?"),
    ("I need a nap.", "You probably do. Can you sneak one?"),
    ("I'm on my third cup of coffee.", "Long day? Just be careful about tonight's sleep."),
    ("I woke up in a great mood.", "That's the best. What do you think set it?"),
    ("I woke up in a bad mood.", "That's rough. Anything simple that usually resets it?"),
    ("I forgot my anniversary.", "Ouch. It happens. Late acknowledgment still counts."),
    ("Today was a really good day.", "Love that. What made it good?"),
    ("Today was a really bad day.", "Sorry. What happened?"),
    ("I got soaked in the rain.", "Not the best. Warm shower waiting?"),
    ("I found twenty bucks on the ground.", "Nice little bonus. Free coffee day."),
    ("I stepped on a Lego.", "Universal pain. Sorry."),
    ("I finally called the doctor.", "Good on you. That's usually the hardest step."),
    ("I finally made that appointment.", "Nice. Future you is grateful."),
    ("I'm nervous to make a phone call.", "Small script sometimes helps. Want to draft a line?"),
    ("I forgot to charge my phone.", "Annoying. Cable within reach?"),
    ("I finished a good workout.", "Nice, endorphins on the way. What did you do?"),
    ("I skipped a workout again.", "Happens. Tomorrow is a fresh chance."),
    ("I made my bed today.", "Small win, real win. Sets the tone."),
    ("I finally organized my closet.", "Feels amazing, doesn't it?"),
    ("I bought new shoes.", "Nice. Comfy or fancy?"),
    ("I got a haircut.", "Fresh look. How do you feel about it?"),
    ("I don't like my new haircut.", "Ugh, the awkward stage. It grows back."),
    ("I tried a new recipe.", "How did it go?"),
    ("The recipe was a disaster.", "It happens. Any part that worked out?"),
    ("I'm running late.", "Deep breath. Send a quick heads up if you can."),
    ("I got the day off.", "Nice surprise. Any plans?"),
    ("It's my birthday.", "Happy birthday. Anything fun planned?"),
    ("It's my kid's birthday.", "That's a big day. How are they celebrating?"),
    ("I'm going home for the holidays.", "Nice. Looking forward to it?"),
    ("I don't want to go home for the holidays.", "That's a valid feeling. Complicated family stuff?"),
    ("I'm on vacation.", "Have a great one. Where are you?"),
    ("My vacation is over.", "The return blues are real. Hope it was a good one."),
    ("I hate mornings.", "Fair. Any small thing that makes them easier?"),
    ("I love mornings.", "Envy that. Any favorite ritual?"),
    ("I'm reading three books at once.", "Ambitious. Any leading the pack?"),
    ("I haven't read a book in ages.", "Same for a lot of people. Anything sound tempting lately?"),
    ("I got locked out of my house.", "Ugh, sorry. Any way in?"),
    ("My phone died.", "Rough timing. Charger nearby?"),
    ("I met my hero today.", "That must have been surreal. How was it?"),
    ("I saw a shooting star.", "Lucky. Did you make a wish?"),
    ("I saw a rainbow today.", "Nice little gift from the sky."),
    ("I got a new plant.", "Exciting. What kind?"),
    ("I painted my room.", "Nice project. What color did you pick?"),
    ("I redecorated a bit.", "Fresh space. What did you change?"),
    ("I finished a puzzle.", "Satisfying. How many pieces?"),
    ("I finished a knit project.", "Nice work. What did you make?"),
    ("I broke my favorite mug.", "Sad. Any chance of a repair?"),
    ("I got a package today.", "Fun. What did you order?"),
    ("I sent my first postcard in years.", "Sweet gesture. Who's it for?"),
    ("I'm out of ideas.", "That's a fair place to be. Sometimes stepping away helps."),
    ("I'm full of ideas.", "Nice creative wave. Any one standing out?"),
    ("I miss being a kid sometimes.", "That feeling is real. Anything from then you can bring back small?"),
    ("I love the smell of rain.", "One of the best. Windows open?"),
    ("I hate mondays.", "You and much of the world. Anything to soften it?"),
    ("I love fridays.", "The universal favorite. What's this one for you?"),
    ("I finally paid off my debt.", "That's huge. Congratulations, real relief."),
    ("I got my first paycheck.", "That's a great moment. Anything you want to do with it?"),
    ("I moved out on my own.", "Big step. How does it feel?"),
    ("I adopted a pet.", "Aw, what kind?"),
    ("My pet is snoring next to me.", "Best background noise there is."),
    ("I lost my pet.", "I'm so sorry. That's a real loss. Take good care of yourself."),
    ("I saw my old teacher today.", "That's such a nice moment. Were they well?"),
    ("I ran into an old friend.", "Love those chance meetings. How was it?"),
    ("I feel like I'm running out of time.", "That feeling is heavy but often lying. What's the actual next step?"),
    ("I feel like I have all the time in the world.", "Nice place to be. Anything you want to spend it on?"),
    ("I want to travel more.", "Nice dream. Any place at the top?"),
    ("I want to travel less.", "Home has its own charm. Feeling worn out from moving?"),
    ("I want to learn to swim.", "That's a lovely one. Anywhere you can start small?"),
    ("I want to run a marathon.", "Big goal. What draws you to it?"),
    ("I'm afraid of failing.", "Most people are. Small trials often shrink the fear."),
    ("I'm afraid of succeeding.", "Not as rare as it sounds. Any part of success feel scary?"),
    ("I feel really alive right now.", "Love that. What's fueling it?"),
    ("I feel numb right now.", "That's a heavy fog. Be gentle with yourself, take small steps."),
    ("I made a new recipe up.", "Fun. Was it a hit?"),
    ("I have a crush on someone.", "Oh! What are they like?"),
    ("I got asked out.", "Exciting. What did you say?"),
    ("I got rejected today.", "Ouch. That takes courage to try. Sorry it stung."),
    ("I'm scared to text them back.", "Fair. Short and honest is usually the best move."),
    ("I found an old photo.", "Sweet find. Anyone you know?"),
    ("I feel homesick.", "That aching kind of love. Any small piece of home you can make there?"),
    ("I'm about to move again.", "Moving is a lot. How are the plans shaping up?"),
    ("I'm scared to move.", "Big changes wobble. What's the biggest worry right now?"),
    ("I love where I live.", "That's a gift. Anything specific you love?"),
    ("I don't like where I live.", "Sorry, that wears on you. Any change on the horizon?"),
    ("I finally started therapy.", "That takes courage. Hope it goes well."),
    ("Therapy has been really helpful.", "Glad to hear. Progress is real, even when slow."),
    ("I'm grateful for my life.", "That's a lovely place to be. Anything in particular right now?"),
    ("I don't feel like myself.", "That's disorienting. Any recent change that might be part of it?"),
    ("I've been feeling off for a while.", "Sorry to hear. Worth mentioning to someone you trust."),
    ("I finally slept well.", "That first good night is such a win. Rest of the day easier?"),
    ("I have too much on my plate.", "That's a lot. What can wait?"),
    ("I want to help others more.", "Lovely goal. Small local things count more than we think."),
    ("I volunteered today.", "That's kind of you. How was it?"),
    ("I donated to a cause.", "Nice. What made you pick that one?"),
    ("I read something that changed my mind.", "Love that. What was it about?"),
    ("I saw a movie that made me cry.", "That's a good movie. What was it about?"),
    ("I watched a movie that made me laugh.", "The best kind. What was so funny?"),
    ("I want to write a book.", "That's a big dream. Any idea what about?"),
    ("I want to start a business.", "Fun goal. What kind?"),
    ("I want to make art again.", "That's beautiful. Any small first thing you can make this week?"),
    ("I'm proud of a small thing today.", "Good. What was it?"),
    ("I want a quiet weekend.", "Nice plan. Any small joys you'll add?"),
    ("I want an adventurous weekend.", "Fun. Anything you've been wanting to try?"),
    ("I feel like I'm growing.", "That's such a nice feeling. What are you noticing?"),
    ("I feel stuck in a rut.", "Sorry to hear. Any small change to try this week?"),
    ("I want to feel excited about life again.", "That's a real wish. Any small spark you can chase?"),
    ("I need to slow down.", "That sounds true. What could you drop first?"),
    ("I feel like I'm doing too much.", "Sounds it. What's the biggest drain right now?"),
    ("I want to be brave.", "You already are, for saying it. Any next small brave step?"),
    ("I want to say I love you more.", "Lovely. Who's the first person?"),
    ("I want to call my parents more.", "Sweet. Even short calls count."),
    ("I called my parents today.", "That's nice. How were they?"),
    ("I forgot to call someone back.", "Happens. Late is better than never."),
    ("I sent a nice message to a friend.", "Small joy sent out into the world."),
    ("I got a nice message from a friend.", "Love those. Made your day a little?"),
    ("I saw my kid smile today.", "The best kind of day. What sparked it?"),
    ("I made someone laugh today.", "Nice. What did you say?"),
    ("I helped someone today.", "That's meaningful. What did you do?"),
    ("Someone helped me today.", "That's a real gift. What did they do?"),
    ("A stranger was kind to me.", "Lovely. Those moments stay warm for a while."),
    ("I paid a compliment to a stranger.", "Small ripple, real good. How did they react?"),
    ("I got a good tip at work.", "Nice. Better shift than expected?"),
    ("I found a new favorite spot.", "Fun. What's it like?"),
    ("I tried something new today.", "Nice. What was it?"),
    ("I stepped out of my comfort zone.", "Brave. How did it go?"),
    ("I'm proud of my kid.", "That's a lovely thing to feel. What did they do?"),
    ("I'm frustrated with my kid.", "Hard days happen. Anything specific?"),
    ("My partner surprised me.", "Sweet. What did they do?"),
    ("I surprised my partner.", "Nice, what did you do?"),
    ("I forgot our anniversary.", "Oof. Late acknowledgment counts. And honesty about missing it."),
    ("I need a hug.", "Sending you one in words. What's going on?"),
    ("I just want to be heard.", "I'm listening. Start wherever you like."),
    ("I don't want advice, just to talk.", "Understood. I'm here to just listen."),
    ("I want to feel useful again.", "Small acts help. Anything you used to enjoy that helped others?"),
    ("I got great feedback at work.", "Nice. Any part that meant the most?"),
    ("I got hard feedback at work.", "That's tough. Anything useful you can pull from it?"),
    ("I'm exhausted from work.", "Sorry. Any real rest coming up?"),
    ("I love my job right now.", "That's a gift. What's making it good?"),
    ("I'm bored at work.", "Fair. Anything you can shift to spice it up?"),
    ("Nothing sounds fun today.", "That's a heavy place. Rest sometimes fixes what fun can't."),
    ("Everything sounds fun today.", "What a mood. Any one thing pulling ahead?"),
    ("I made a mistake at work.", "It happens. Own it small, fix what you can, and keep going."),
    ("I got a nice review at work.", "Well done. Anything you're proud of specifically?"),
    ("I've been putting off a hard conversation.", "Been there. Would writing out the first line help?"),
    ("I finally had that hard conversation.", "That took courage. How did it go?"),
    ("I'm afraid of confrontation.", "Most people are. Quiet honesty often works better than we expect."),
    ("I love how the sky looked today.", "The sky delivers sometimes. What was it like?"),
    ("I saw fireflies tonight.", "That's a lovely sight. Rare where you are?"),
    ("I'm listening to old music.", "Anything hitting harder than expected?"),
    ("I discovered a new artist I love.", "Fun. What's their sound like?"),
    ("I'm at a concert soon.", "Exciting. Been looking forward to it long?"),
    ("I got great seats.", "Nice, that changes the whole experience."),
    ("I'm going to a wedding.", "How fun. Anyone close?"),
    ("I'm going to a funeral.", "I'm sorry. Take it slow this week."),
    ("I visited someone in the hospital.", "That takes love. How were they?"),
    ("I'm sick again.", "That's rough. Rest a lot, drink water, and take it easy."),
    ("I finally feel better.", "Glad to hear. Take it slow coming back in."),
    ("My back is killing me.", "Sorry, that wears on you. Any chance to move around a bit?"),
    ("I hurt my ankle.", "Ouch. Rest, ice, and give it time. Any chance to see someone if it lingers?"),
    ("I got sunburned.", "Ouch, take it easy. Aloe if you have it."),
    ("I got caught in a storm.", "Yikes. Home safe now?"),
    ("The power went out.", "Annoying. Candles, snacks, and patience. Any idea how long?"),
    ("I'm out of internet.", "Rough. Any way to check when it'll be back?"),
    ("I got locked out of my email.", "Frustrating. Any recovery option working?"),
    ("I hate mornings but today was different.", "What made this one different?"),
    ("I'm starting to like mornings.", "Nice shift. What changed?"),
    ("I need to talk to my kid about something hard.", "That takes love. Anything you're worried about?"),
    ("I need to talk to my partner about something hard.", "Big step. Any part of it you're most nervous about?"),
    ("I feel really lucky in life.", "That's a nice place to sit. Anything specific?"),
    ("I want to be more present.", "Small pauses help. What draws you to it?"),
    ("I want to take up meditation.", "Small, steady starts work best. Even a few minutes counts."),
    ("I tried meditation and hated it.", "That's fair. It's not for everyone. Any other calming thing you like?"),
    ("I want to feel less angry.", "Slow breath and space help. Anything specific pulling the anger?"),
    ("I want to feel less sad.", "Sorry it's been heavy. Small steady kindness to yourself helps most."),
    ("I want to feel less afraid.", "Small brave beats big brave. What small step could you take?"),
    ("I want to feel more joy.", "Look for small already-there kinds. Notice out loud when you catch one."),
    ("I want to feel more peaceful.", "Slow days, less noise, and more time outside help many people."),
    ("I want to feel more alive.", "Try something new small. Fresh input tends to open something up."),
    ("I want to be less on my phone.", "Small barriers help. Move apps, silence alerts, keep it in another room."),
    ("I want to read more.", "Keep one book where you sit most. A few pages a day adds up."),
    ("I want to write more.", "Start small, most days. Not perfect, just steady."),
    ("I want to sing more.", "Fun. Any songs on your list?"),
    ("I want to dance more.", "Yes. Even alone in the kitchen counts."),
    ("I want to garden more.", "Lovely. Start with something forgiving."),
    ("I want to be outdoors more.", "Small steady doses. A short walk daily is a nice start."),
    ("I want to see my friends more.", "Nice goal. Any one you can text right now?"),
    ("I want to be a better version of myself.", "You're on the way, just for saying it. What's a small first step?"),
]

# Dedupe
seen = set()
f4_final = []
for q, a in F4_PAIRS:
    key = (q, a)
    if key in seen:
        continue
    seen.add(key)
    f4_final.append((q, a))

# Pad with more short-user statements and reactions.
extra_f4 = [
    ("It's been a long week.", "Sorry, those wear on you. Any quiet time coming up?"),
    ("I'm having a great week.", "Love that. What's been the best part?"),
    ("I'm about to make dinner.", "What are you thinking of cooking?"),
    ("I don't know what to have for lunch.", "Anything you have already that could work?"),
    ("I'm looking forward to the weekend.", "Any plans, or a slow one?"),
    ("I dread the weekend, honestly.", "Fair. Sometimes free time is heavy. Anything small planned?"),
    ("I got a new bike.", "Nice. Any first ride yet?"),
    ("I sold my old car.", "That's a moment. Any nostalgia there?"),
    ("I'm going camping.", "Sounds fun. Where are you headed?"),
    ("I'm going hiking.", "Nice. Any favorite trail?"),
    ("I fell down today.", "Ouch. Are you okay?"),
    ("I stubbed my toe.", "Universal pain. Sorry."),
    ("I made a great cup of tea.", "The right cup is a small miracle."),
    ("I tried a new drink.", "How was it?"),
    ("I'm eating leftovers.", "Sometimes those are the best kind of dinner."),
    ("I really need a haircut.", "Any place you like to go?"),
    ("I chopped all my hair off.", "Big change. How do you feel?"),
    ("I gave someone a compliment today.", "Nice. What did you say?"),
    ("Someone complimented my outfit.", "Love that. Made your day a little?"),
    ("I got called a good friend today.", "That's a lovely thing to hear. It means you're being one."),
    ("I saw an old photo of me.", "Nice or weird?"),
    ("I ran into an old teacher.", "That's such a nice surprise. Were they well?"),
    ("I lit a candle.", "Cozy move. Any favorite scent?"),
    ("I opened the windows for the first time in weeks.", "That fresh air feels different, doesn't it?"),
    ("I made my morning coffee wrong.", "Ugh, that's a rough start. Can you salvage it?"),
    ("I heard my favorite song on the radio.", "Little joy. Sing along a bit?"),
    ("I sang in the shower.", "Best acoustics in the house."),
    ("I woke up to sunshine.", "Nice, that softens the whole morning."),
    ("I've been on a walk every day this week.", "Great habit. How's it feeling?"),
    ("I skipped my walk today.", "Happens. Tomorrow is a fresh start."),
    ("I want to be less on the couch.", "Small changes help. Even a short walk after dinner counts."),
    ("I made someone smile today.", "That's a nice ripple. What did you do?"),
    ("I got a warm hug today.", "Those are underrated. Hope it helped."),
    ("I feel behind on everything.", "Sorry, that's a heavy feeling. Pick one small thing to move today."),
    ("I'm on top of everything for once.", "Nice. Sit in it a moment."),
    ("I actually followed my routine today.", "Rare and good. Any part felt easier?"),
    ("I forgot to eat again.", "Grab something small now if you can. Don't skip too long."),
    ("I ate too much sugar today.", "Balance it with water and rest. Tomorrow's a new day."),
    ("I finished a puzzle.", "Satisfying. Any tricky part?"),
    ("I finished a video game.", "Nice. Which one?"),
    ("I started a new show.", "What's it about?"),
    ("I don't want to finish this show.", "The good ones are hard to end. Slow it down maybe?"),
    ("I re-watched a favorite movie.", "Comfort watch. Which one?"),
    ("I want to travel somewhere warm.", "That sounds nice. Any place drawing you?"),
    ("I want to travel somewhere cold.", "The quiet kind of pretty. Any place in mind?"),
    ("I'm homesick for a place I've never been.", "That's a real feeling. Interesting where it points."),
    ("I feel really seen today.", "That's a good feeling. What happened?"),
    ("I feel invisible today.", "Sorry, that's heavy. Anything you can do to feel connected small?"),
    ("I feel loved today.", "Sit with it a while. Beautiful."),
    ("I feel unloved today.", "That's a hard feeling. Sending some warmth through the screen."),
    ("I need to journal more.", "Start with two lines a day. Anything more is bonus."),
    ("I journaled this morning.", "Nice, how did it feel?"),
    ("I made a to-do list.", "Good move. What's on top?"),
    ("I threw out my to-do list.", "Sometimes a reset helps. Fresh start."),
    ("I finished all my chores.", "Rare victory. Enjoy the couch."),
    ("The chores never end.", "They really don't. Small daily rounds beat one big cleanup."),
    ("I finally organized my photos.", "That's a real project. How does it feel done?"),
    ("I need to back up my phone.", "Good instinct. Take five minutes and do it now maybe?"),
    ("I finally set up my computer.", "Nice. Any part fun?"),
    ("I want to slow down my week.", "Say no to one thing. That's usually the trick."),
    ("I want to speed up my week.", "Any one thing you can start now to get it moving?"),
    ("I want to do less this year.", "Doing less is underrated. What could you drop first?"),
    ("I want to do more this year.", "Small consistent things pile up. What draws you?"),
    ("I've been snacking too much.", "Real meals often help. What's around?"),
    ("I haven't cooked in weeks.", "Even a simple meal counts. What sounds easy?"),
    ("I need to grocery shop.", "Small list helps. What are you low on?"),
    ("I forgot to pack a lunch.", "Anything nearby you can grab?"),
    ("I love packing my own lunch.", "Small daily win. Anything fun in there today?"),
    ("I stayed up too late last night.", "Rough start today then. Nap possible?"),
    ("I went to bed early last night.", "Nice reset. Feel any different today?"),
    ("I actually made it to the gym.", "Well done. How did it go?"),
    ("I hate the gym.", "Fair. There are lots of other ways to move. What sounds better?"),
    ("I love the gym.", "Nice, endorphins on tap. Any favorite workout?"),
    ("I overtrained.", "Rest days matter. Take an easy few."),
    ("I don't feel like exercising today.", "One day off is fine. Tomorrow is a fresh chance."),
    ("I have big dreams.", "Love that. Any small first step in mind?"),
    ("I don't have any big dreams.", "That's okay too. Small good days count."),
    ("I want to be famous.", "That's a big draw. What about it pulls you?"),
    ("I want to be forgotten.", "That's a strong feeling. Been a heavy stretch?"),
    ("I feel connected to nature today.", "Lovely. Where were you?"),
    ("I feel disconnected from everything.", "That's a heavy fog. Small kind steps often help most."),
    ("I want to move somewhere new.", "Fresh start energy. Any place drawing you?"),
    ("I want to stay right here.", "Rooted is a real gift. Anything you love about it?"),
    ("I want a simpler life.", "Small cuts add up. What could go first?"),
    ("I want a fuller life.", "What would fuller look like for you?"),
    ("I feel really busy today.", "Sounds it. Anything you can push to tomorrow?"),
    ("I have a slow day for once.", "Enjoy. Any small joys planned?"),
    ("I love slow mornings.", "The best. Coffee, quiet, no rush."),
    ("I hate slow mornings.", "Fair. Some people need the launch. Any morning ritual you like?"),
    ("I got a new mug.", "Fun little upgrade. Any favorite drink to break it in?"),
    ("I broke my favorite pen.", "Small sad. Any backup?"),
    ("I love handwritten notes.", "Same. Any one you're planning to write?"),
    ("I need to send a thank you card.", "Do it while it's on your mind. Short is fine."),
    ("I sent a nice email today.", "Small good deed. Made your day a little?"),
    ("I got an unexpected call.", "Nice or worrying?"),
    ("I got flowers.", "Aw. From someone special?"),
    ("I gave flowers.", "That's a lovely thing to do. Any special reason?"),
    ("I made someone's day.", "Beautiful. What happened?"),
    ("Someone made my day.", "Love that. What did they do?"),
    ("I felt so tired I almost cried.", "That's the too-tired kind. Rest as much as you can right now."),
    ("I feel so full of energy.", "Nice. Any plans to spend it well?"),
    ("Everything is quiet right now.", "Enjoy the pause. Rare and good."),
    ("Everything is chaos right now.", "Rough. What's the loudest piece?"),
    ("I want to talk about my day.", "Please do. Where does it start?"),
    ("Nothing happened today.", "That's okay too. Uneventful is a kind of rest."),
    ("I need to vent.", "Vent away. I'm here."),
    ("I need advice.", "Sure. What's on your mind?"),
    ("I need company.", "I'm right here. What are we talking about?"),
    ("I feel like nobody gets me.", "That's a lonely place. Anything specific you wish they understood?"),
    ("I feel really understood today.", "That's a rare gift. Who is it?"),
    ("I'm proud of my parents.", "That's a lovely feeling. What did they do?"),
    ("I'm angry at my parents.", "That's a hard feeling too. What's going on?"),
    ("My roommate is annoying me.", "Bound to happen. Anything small you can talk out?"),
    ("My roommate did something sweet.", "That's nice. What did they do?"),
    ("I miss having a roommate.", "Solo is quiet in a hard way sometimes."),
    ("I love living alone.", "Nice. Any small ritual you love?"),
    ("I hate living alone.", "Sorry. Sometimes small connection helps. Any friend to check in with?"),
    ("I want a pet.", "Fun. Any kind in mind?"),
    ("My pet is my whole world.", "Love that. What's their best trait?"),
    ("I want to see a friend I haven't seen in a while.", "Send them a quick note now. It usually rolls."),
    ("I need a break.", "Take one if you can. Even a short one counts."),
    ("I don't want a break, I want a nap.", "Fair. Set an alarm and take the nap."),
    ("I want to lie in bed all day.", "Sometimes that's the right answer. Give yourself the day."),
    ("I want to run around all day.", "Nice energy. Where are you headed?"),
    ("I want silence.", "Understood. Find a quiet corner if you can."),
    ("I want noise around me.", "Coffee shop energy. Any nearby?"),
    ("I want to be with people.", "Anyone free tonight? Even a short hang counts."),
    ("I want to be alone.", "Fair. Guard the evening for yourself."),
    ("I like being a beginner at something new.", "That's a healthy place to be. What are you starting?"),
    ("I hate being bad at things.", "Common feeling. Every skilled person was awkward first."),
    ("I want to be an expert at something.", "Any topic that keeps pulling you?"),
    ("I don't want to be an expert at anything.", "That's fine too. Curious dabbler is a lovely life."),
    ("Today feels good.", "Lovely. What's making it feel that way?"),
    ("Today feels flat.", "Sorry. Any small thing to spark it?"),
    ("I did something nice for me today.", "Good. What was it?"),
    ("I did something nice for someone else today.", "Sweet. What did you do?"),
    ("I helped a stranger today.", "That's kind. What happened?"),
    ("A stranger made me laugh today.", "Best kind of stranger."),
    ("I got a warm smile from a barista.", "Small joy, real one."),
    ("I gave a big tip today.", "That was kind. Made someone's shift better."),
    ("Someone was rude to me today.", "Sorry, that lingers. Not about you, though."),
    ("Someone was really kind to me today.", "Warm memory. Save that one."),
    ("I forgot my umbrella.", "Ugh. Any shelter along your route?"),
    ("I got the last seat on the bus.", "Small win. Enjoy the ride."),
    ("I let someone in front of me in line.", "Small kindness. Nice."),
    ("I stood up for myself today.", "That's real courage. How does it feel?"),
    ("I let someone else win an argument.", "Sometimes the right call. Peace over point."),
    ("I lost an argument I should have won.", "Rough. Any way to revisit later?"),
    ("I got called out today.", "Ouch. Anything useful in it?"),
    ("I called someone out today.", "That takes courage. How did it go?"),
    ("I let someone down.", "Own it small, make it right if you can, and forgive yourself too."),
    ("Someone let me down.", "That hurts. Is it something to talk out with them?"),
    ("I trust someone new.", "That's brave. Hope it goes well."),
    ("I stopped trusting someone.", "That's heavy. Not lightly done."),
    ("I made a new memory today.", "Small ones stack up. What was it?"),
    ("I want to feel less rushed.", "Do less. Rushing usually means overpacked."),
    ("I want to feel more capable.", "Finish something small this week on purpose. It builds."),
    ("I want to be a better neighbor.", "Small hellos count more than big gestures."),
    ("I want to be a good citizen.", "Small daily choices count. Kind, honest, careful."),
    ("I feel like I'm becoming who I want to be.", "That's beautiful to hear. Any specific piece?"),
    ("I don't know who I am anymore.", "That's a heavy place. Change is disorienting. Be gentle with yourself."),
    ("I feel young today.", "Nice. What sparked it?"),
    ("I feel old today.", "Long day? A gentle evening might help."),
    ("I feel wise today.", "Sit in it. Not every day gets that."),
    ("I feel foolish today.", "We all do sometimes. It's okay."),
    ("I want to laugh more.", "Small silly stuff on purpose. Play helps."),
    ("I want to cry more.", "Sometimes we need it. Let a sad song do the work."),
    ("I want to be still more.", "Small pauses count. Try five slow breaths right now."),
    ("I want to move more.", "Where are your walking shoes?"),
    ("I'm about to nap.", "Enjoy. Set an alarm so it stays short."),
    ("I just woke up from a nap.", "Nice reset. Water helps ease the fog."),
    ("I ate a whole pizza.", "It happens. Hope it was delicious."),
    ("I made the perfect breakfast.", "Nice. What did you go with?"),
    ("I burned toast.", "Universal. Any butter left for the next try?"),
    ("I finally cleaned out my inbox.", "Big project. How's it feel?"),
    ("My inbox is a nightmare.", "Small daily passes help more than big cleanups. Ten a day?"),
    ("I finally fixed a broken thing at home.", "Nice, that's satisfying. What was it?"),
    ("Something at home is broken and I don't want to deal with it.", "Fair. Note it and pick a day. Future you thanks you."),
    ("I feel really quiet inside today.", "That's a nice kind of quiet. Enjoy it."),
    ("I feel really loud inside today.", "Sounds a lot. Anything you can hand off for a bit?"),
    ("Today felt like a gift.", "Save that feeling. Rare and lovely."),
    ("Today felt like a chore.", "Sorry, some days are like that. Rest well tonight."),
]

for q, a in extra_f4:
    if len(f4_final) >= 500:
        break
    key = (q, a)
    if key in seen:
        continue
    seen.add(key)
    f4_final.append((q, a))

# If we still need more, pad with a paraphrase pass
pad_pool_f4 = list(F4_PAIRS)
random.shuffle(pad_pool_f4)

reactions = [
    "Nice, tell me more.",
    "Oh, how so?",
    "That sounds like a lot. Anything specific?",
    "Say more if you want to.",
    "Interesting. What led to that?",
    "Good to hear. What's next?",
    "That's fair. What are you thinking?",
    "Really? What was that like?",
    "Aw, that's sweet. What sparked it?",
    "Ouch. Anything I can help with?",
]

for orig_q, _ in pad_pool_f4:
    if len(f4_final) >= 500:
        break
    r = random.choice(reactions)
    key = (orig_q, r)
    if key in seen:
        continue
    seen.add(key)
    f4_final.append((orig_q, r))

# also add generic short statements paired with generic reactions
short_statements = [
    "So tired.",
    "Feeling great.",
    "Kinda meh.",
    "Bit sad.",
    "Bit anxious.",
    "Bit hopeful.",
    "So happy right now.",
    "Just okay.",
    "Feeling good.",
    "Rough morning.",
    "Rough night.",
    "Great morning.",
    "Great night.",
    "Long day.",
    "Short day.",
    "Weird day.",
    "Quiet day.",
    "Busy day.",
    "Weekend was nice.",
    "Weekend was rough.",
    "Monday hit hard.",
    "Friday finally.",
    "Missed you.",
    "Been busy.",
    "Been thinking.",
    "Been resting.",
    "Been working a lot.",
    "Been reading a lot.",
    "Been walking a lot.",
    "Been eating well.",
    "Been eating badly.",
    "Slept well.",
    "Slept badly.",
    "Woke up sore.",
    "Woke up smiling.",
    "Kind of overwhelmed.",
    "Kind of proud.",
    "Kind of scared.",
    "Kind of excited.",
    "Sort of stuck.",
    "Sort of free.",
    "Feeling stronger.",
    "Feeling softer.",
    "Feeling small.",
    "Feeling seen.",
    "Feeling missed.",
    "Feeling loved.",
]

short_reactions = [
    "Say more if you'd like.",
    "That sounds like a lot. What's going on?",
    "That's a mood. Anything specific?",
    "Nice, want to share more?",
    "Hmm, tell me more.",
    "Alright. What's behind it?",
    "Oh? What's happening?",
    "Gotcha. Anything I can do?",
    "Fair enough. Any change coming?",
    "Take your time. Where does it start?",
]

for stmt in short_statements:
    if len(f4_final) >= 500:
        break
    r = random.choice(short_reactions)
    key = (stmt, r)
    if key in seen:
        continue
    seen.add(key)
    f4_final.append((stmt, r))

# Enforce under 30 words
f4_final = enforce_under_30(f4_final)
random.shuffle(f4_final)
f4_final = f4_final[:500]

# -----------------------------------------------------------------------------
# Family 1 cleanup: enforce under 30 words + top up to 500 if short
# -----------------------------------------------------------------------------
f1_final = enforce_under_30(f1_final)

# Top up F1 if short by combining more openers with more answers
if len(f1_final) < 500:
    all_q_pools = [cap_openers, what_are_you, who_made, name_qs, code_qs, math_cap_qs,
                   write_qs, lookup_qs, facts_qs, memory_qs, feelings_qs, limits_qs, lang_qs]
    all_a_pools = [cap_answers, what_are_you_ans, who_made_ans, name_ans, code_ans, math_cap_ans,
                   write_ans, lookup_ans, facts_ans, memory_ans, feelings_ans, limits_ans, lang_ans]
    seen_f1 = set((q, a) for q, a in f1_final)
    for i, qs in enumerate(all_q_pools):
        if len(f1_final) >= 500:
            break
        answers = all_a_pools[i]
        for q in qs:
            if len(f1_final) >= 500:
                break
            for a in answers:
                if len(f1_final) >= 500:
                    break
                key = (q, a)
                if key in seen_f1:
                    continue
                seen_f1.add(key)
                f1_final.append((q, a))

f1_final = enforce_under_30(f1_final)
random.shuffle(f1_final)
f1_final = f1_final[:500]

# -----------------------------------------------------------------------------
# Family 2 cleanup: enforce word cap
# -----------------------------------------------------------------------------
f2_final = enforce_under_30(f2_final)

# -----------------------------------------------------------------------------
# Combine, shuffle, verify, write
# -----------------------------------------------------------------------------

assert len(f1_final) == 500, f"F1 count {len(f1_final)}"
assert len(f2_final) == 500, f"F2 count {len(f2_final)}"
assert len(f3_final) == 500, f"F3 count {len(f3_final)}"
assert len(f4_final) == 500, f"F4 count {len(f4_final)}"

all_pairs = [
    ("F1", q, a) for q, a in f1_final
] + [
    ("F2", q, a) for q, a in f2_final
] + [
    ("F3", q, a) for q, a in f3_final
] + [
    ("F4", q, a) for q, a in f4_final
]

# Sanity: ensure ASCII, no emojis, no markdown special
def is_clean(s):
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True

for tag, q, a in all_pairs:
    assert is_clean(q), f"non-ascii q: {q!r}"
    assert is_clean(a), f"non-ascii a: {a!r}"
    assert len(a.split()) < 30, f"too long ({len(a.split())} words): {a!r}"

random.shuffle(all_pairs)
assert len(all_pairs) == 2000

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
    for tag, q, a in all_pairs:
        f.write(json.dumps({"user": q, "assistant": a}, ensure_ascii=True) + "\n")

# Report to stdout
counts = {"F1": 0, "F2": 0, "F3": 0, "F4": 0}
for tag, _, _ in all_pairs:
    counts[tag] += 1

print(f"Total lines written: {len(all_pairs)}")
print(f"Family counts: {counts}")
print(f"Output path: {OUT_PATH}")
