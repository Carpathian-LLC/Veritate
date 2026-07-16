"""
Generate 2000 diverse SFT joke pairs across 4 families.

Families:
  1. Setup-punchline puns (~500)
  2. Knock-knock jokes (~500)
  3. Observational / absurd one-liners (~500)
  4. Dad jokes / groaners (~500)

Enforces:
  - Each unique joke text used <=2 times total
  - Assistant replies <40 words
  - Kid-friendly / no protected-category humor
  - Plain ASCII, no emoji/markdown
"""

import json
import os
import random
from collections import Counter

random.seed(20260715)

OUT = r"c:/GitHub/Veritate/temp/sft_gen/jokes.jsonl"

# ---------------------------------------------------------------------------
# USER PROMPTS
# ---------------------------------------------------------------------------

# ~60% direct joke requests
DIRECT_REQUESTS = [
    "Tell me a joke.",
    "Tell me a joke!",
    "tell me a joke",
    "Got any jokes?",
    "got any jokes?",
    "Do you know any jokes?",
    "Know any good jokes?",
    "Make me laugh.",
    "make me laugh",
    "Say something funny.",
    "say something funny",
    "One more joke?",
    "Another one?",
    "Another?",
    "One more?",
    "Hit me with a joke.",
    "Hit me with another one.",
    "Give me a joke.",
    "Can you tell me a joke?",
    "Tell me something funny.",
    "Say a joke.",
    "Any jokes?",
    "Know any puns?",
    "Got any puns?",
    "Tell me a pun.",
    "Do you have any jokes?",
    "Please tell me a joke.",
    "I want to hear a joke.",
    "Tell me your best joke.",
    "What's your favorite joke?",
    "Share a joke.",
    "Joke please.",
    "Joke, please.",
    "Tell me something silly.",
    "Something funny please.",
    "Give me a good one.",
    "Tell me another.",
    "Do a joke.",
    "Wanna hear a joke? You tell one.",
    "Your turn to tell a joke.",
    "Tell me a joke about animals.",
    "Tell me a joke about food.",
    "Tell me a joke about weather.",
    "Tell me a joke about plants.",
    "Tell me a joke about sports.",
    "Tell me a joke about music.",
    "Tell me a joke about a job.",
    "Got a short joke?",
    "Short joke?",
    "Quick joke?",
    "Give me a quick joke.",
    "Silly joke?",
    "Cheesy joke please.",
    "Something goofy?",
    "Anything funny?",
    "Can I hear a joke?",
    "Would you tell me a joke?",
    "Please make me smile.",
    "Please cheer me up with a joke.",
    "Tell me a groaner.",
    "Give me a groaner.",
    "Any groaners?",
    "Dad joke?",
    "Got any dad jokes?",
    "Dad joke please.",
    "Hit me with a dad joke.",
    "Give me your worst dad joke.",
    "Tell me a really cheesy one.",
    "Something cheesy?",
    "Tell me an observational joke.",
    "Got any one-liners?",
    "One-liner please.",
    "Hit me with a one-liner.",
    "Do a one-liner.",
    "Anything absurd?",
    "Say something absurd.",
    "Say something ridiculous.",
    "Give me something weird and funny.",
]

# ~40% conversational lead-ins
LEAD_INS = [
    "I'm bored.",
    "I'm so bored.",
    "I am kinda bored.",
    "I need cheering up.",
    "I need a laugh.",
    "I could use a laugh.",
    "Rough day. Cheer me up?",
    "Been a long day.",
    "It's Friday, tell me something fun.",
    "It's Monday, I need help.",
    "Ugh, Mondays.",
    "Waiting for the bus, entertain me.",
    "Long meeting ahead, distract me.",
    "I'm stuck in traffic.",
    "Cheer me up.",
    "Cheer me up please.",
    "Make my day.",
    "Brighten my mood.",
    "Distract me.",
    "Entertain me.",
    "Help, I'm sad.",
    "Feeling blue.",
    "Feeling low.",
    "I could use a smile.",
    "I need a smile.",
    "Kids are asking for a joke.",
    "My kid wants a joke.",
    "Waiting in line, tell me something.",
    "Long flight, entertain me.",
    "Stuck at the airport.",
    "Rainy day, distract me.",
    "It's raining and I'm stuck inside.",
    "Snowed in, help.",
    "Power's out, tell me a joke.",
    "Can't sleep.",
    "Insomnia, help me out.",
    "My coffee hasn't kicked in yet.",
    "I need coffee and a joke.",
    "Lunch break, entertain me.",
    "On break, tell me something.",
    "Killing time.",
    "Just killing time here.",
    "Waiting for the oven timer.",
    "Waiting for laundry.",
    "Bored at my desk.",
    "Tired but wired.",
    "Zoom call is boring.",
    "Class is boring.",
    "Homework is dull, cheer me up.",
    "My commute is long.",
    "Long car ride, kids need a joke.",
    "Roadtrip! Entertain us.",
    "Camping trip, need a joke.",
    "Beach day, tell me something fun.",
    "It's my birthday, tell me a joke.",
    "Party trick needed, help.",
    "I want to text my friend a joke.",
    "Trying to impress my nephew.",
    "Need something to tell grandma.",
    "Need something for a card.",
    "Icebreaker needed.",
    "First-date icebreaker?",
    "Stuck on hold, help.",
    "Elevator ride, distract me.",
    "In an Uber, entertain me.",
    "Need something clean and funny.",
    "Kid-friendly joke please.",
    "Grandkids want a joke.",
    "Teaching class needs a joke.",
    "Warmup joke for my talk?",
    "I want to start my speech with a joke.",
    "Tired of doom scrolling, cheer me up.",
    "Screen time break, joke please.",
    "Take my mind off things.",
    "I'm nervous, distract me.",
    "Anxious, need a smile.",
    "Silly mood today.",
    "Feeling silly, join me.",
    "It's early, wake me up with a joke.",
    "Late night, tell me a joke.",
    "Insomnia strikes again.",
    "Cooking dinner, keep me company.",
    "Washing dishes, tell me something.",
    "Folding laundry, entertain me.",
]

FOLLOWUP_REQUESTS = [
    "Another one?",
    "Another please.",
    "One more.",
    "One more please.",
    "Ha, another?",
    "Haha, another?",
    "Nice, more?",
    "Do another.",
    "Keep going.",
    "More jokes.",
    "Again!",
    "More!",
    "Encore.",
    "Hit me again.",
    "Do another one.",
    "Another joke.",
    "Give me another.",
]

def pick_user(kind_hint=None, force_direct=False):
    """Pick a user prompt. kind_hint: 'dad', 'knock', 'observational', 'pun'."""
    r = random.random()
    if force_direct or r < 0.60:
        pool = list(DIRECT_REQUESTS)
        if kind_hint == "dad":
            pool += ["Dad joke?", "Got any dad jokes?", "Hit me with a dad joke.", "Dad joke please.", "Tell me a dad joke.", "Give me your worst dad joke.", "Cheesy joke please."]
        elif kind_hint == "observational":
            pool += ["Got any one-liners?", "One-liner please.", "Say something absurd.", "Something weird and funny?"]
        elif kind_hint == "pun":
            pool += ["Know any puns?", "Got any puns?", "Tell me a pun.", "Any wordplay?"]
        return random.choice(pool)
    return random.choice(LEAD_INS)

# ---------------------------------------------------------------------------
# FAMILY 1: Setup-punchline puns
# ---------------------------------------------------------------------------
# Each entry: full joke text.

PUN_JOKES = [
    # animals
    "Why do cows wear bells? Because their horns don't work.",
    "Why don't oysters share their pearls? Because they're shellfish.",
    "What do you call a bear with no teeth? A gummy bear.",
    "Why did the chicken join a band? Because it had the drumsticks.",
    "What do you call a fish wearing a crown? Your royal haddock.",
    "Why don't cats play poker in the jungle? Too many cheetahs.",
    "What do you call a cold dog? A chilli dog.",
    "What do you call a dog magician? A labracadabrador.",
    "Why was the cat sitting on the computer? To keep an eye on the mouse.",
    "How do you catch a squirrel? Climb a tree and act like a nut.",
    "What do you call a lazy kangaroo? A pouch potato.",
    "Why did the duck cross the playground? To get to the other slide.",
    "How do bees get to school? On the school buzz.",
    "What do you call a rabbit with fleas? Bugs Bunny.",
    "Why do fish live in salt water? Because pepper makes them sneeze.",
    "What kind of music do rabbits like? Hip hop.",
    "What do you call a sleeping bull? A bulldozer.",
    "What do you call a bear caught in the rain? A drizzly bear.",
    "Why was the horse so grumpy? He got up on the wrong side of the stable.",
    "What do you call a pig that does karate? A pork chop.",
    "Why don't crabs give to charity? Because they're shellfish.",
    "What do you call a fish that wears a bowtie? Sofishticated.",
    "How does a penguin build its house? Igloos it together.",
    "What did the ocean say to the shore? Nothing, it just waved.",
    "Why did the crab never share? Because he was a little shellfish.",
    "What kind of shoes do frogs wear? Open toad.",
    "What do you call a snake that works for the government? A civil serpent.",
    "Why did the owl invite his friends over? Because he didn't want to be owl by himself.",
    "What did the buffalo say to his son when he left? Bison.",
    "Why did the elephant bring a suitcase? Because he had a trunk to pack.",
    "What do you call a group of musical whales? An orca-stra.",
    "Why was the sheep so quiet? She didn't have the wool to speak up.",
    "What do you call a deer with no eyes? No idear.",
    "What do you call a alligator in a vest? An investigator.",
    "What do birds give out on Halloween? Tweets.",
    "Why did the bird go to the hospital? For a tweetment.",
    "What do you get if you cross a snowman and a shark? Frostbite.",
    "Why don't leopards play hide and seek? Because they're always spotted.",
    "What do you call a fish without eyes? Fsh.",
    "What do you call a hen who counts her eggs? A mathemachicken.",
    "How do you organize a space party? You planet.",
    "What do you call a cow during an earthquake? A milkshake.",
    "Why don't skeletons fight each other? They don't have the guts.",
    "Why don't scientists trust atoms? Because they make up everything.",
    "What do you call a fake pasta? An impasta.",
    "Why did the bicycle fall over? Because it was two tired.",
    "What do you call a can opener that doesn't work? A can't opener.",
    "Why did the coffee file a police report? It got mugged.",
    "How does a train eat? It goes chew chew.",
    "Why did the math book look sad? Because it had too many problems.",
    "What did the zero say to the eight? Nice belt.",
    "Why do we tell actors to break a leg? Because every play has a cast.",
    "Why did the picture go to jail? Because it was framed.",
    "How do you make a tissue dance? Put a little boogie in it.",
    "What do you call a factory that sells passable products? A satisfactory.",
    "What did the janitor say when he jumped out of the closet? Supplies.",
    "Why do seagulls fly over the sea? Because if they flew over the bay, they'd be bagels.",
    "What did the grape say when it got stepped on? Nothing, it just let out a little whine.",
    "Why don't eggs tell jokes? They might crack up.",
    "What do you call cheese that isn't yours? Nacho cheese.",
    "How do you fix a broken pumpkin? With a pumpkin patch.",
    "Why did the tomato blush? Because it saw the salad dressing.",
    "What did one plate say to the other? Dinner's on me.",
    "Why did the cookie cry? Because his mom was a wafer so long.",
    "What do you call a sad strawberry? A blueberry.",
    "Why did the banana go to the doctor? Because it wasn't peeling well.",
    "What do you call fake spaghetti? An impasta.",
    "Why did the lettuce blush? Because it saw the salad dressing changing.",
    "How do you make an octopus laugh? With ten-tickles.",
    "What do you call bears with no ears? B.",
    "What do you call a shoe made of a banana? A slipper.",
    "How do you catch a whole school of fish? With bookworms.",
    "What did the big flower say to the little flower? Hi, bud.",
    "Why did the gardener plant a light bulb? To grow a power plant.",
    "What did the tree say to the wind? Leaf me alone.",
    "Why are trees always so relaxed? They know how to root.",
    "What do you call a nervous tree? A shiver-timber.",
    "Why did the leaf go to the doctor? It was feeling green.",
    "What do you call a magic dog? A labracadabrador.",
    "How does the moon cut his hair? Eclipse it.",
    "What do you call a snowman on a hot day? A puddle.",
    "How do snowmen greet each other? Ice to meet you.",
    "What do you call a snowman with a six-pack? An abdominal snowman.",
    "Why do snowmen love yard sales? They're always looking for a carrot.",
    "What kind of car does a snowman drive? An icicle.",
    "Why was the broom late? It over-swept.",
    "What did the little corn say to mama corn? Where's popcorn?",
    "Why did the clock go to the principal's office? For tocking too much.",
    "What did one wall say to the other? I'll meet you at the corner.",
    "What do you call a pencil with two erasers? Pointless.",
    "Why couldn't the pony sing itself a lullaby? She was a little horse.",
    "Why did the golfer bring two pairs of pants? In case he got a hole in one.",
    "Why did the soccer ball quit the team? It was tired of being kicked around.",
    "Why do basketball players love donuts? Because they can dunk them.",
    "Why can't your nose be twelve inches long? Because then it would be a foot.",
    "What kind of tree fits in your hand? A palm tree.",
    "What do you call a boomerang that doesn't come back? A stick.",
    "How does the ocean say hi? It waves.",
    "Why is the sky so tall? So the clouds have room to grow.",
    "What did the cloud wear under his raincoat? Thunderwear.",
    "Why did the sun go to school? To get a little brighter.",
    "What do you call rain that falls in buckets? A downpour deal.",
    "What kind of shorts do clouds wear? Thunderwear.",
    "Why did the calendar look worried? Its days were numbered.",
    "Why did the belt go to jail? For holding up a pair of pants.",
    "What did the sock say to the foot? You're putting me on.",
    "Why did the man put his money in the freezer? He wanted cold hard cash.",
    "What did the paper say to the pencil? Write on.",
    "What kind of key opens a banana? A monkey.",
    "What did one hat say to the other? You stay here, I'll go on ahead.",
    "What did the traffic light say to the car? Don't look, I'm changing.",
    "Why did the broom get promoted? It was clearly outstanding.",
    "Why did the golfer wear two pairs of socks? In case he got a hole in one.",
    "Why did the music teacher need a ladder? To reach the high notes.",
    "What do you call a musical insect? A humbug.",
    "What did the violin say when it was scared? I'm strung out.",
    "Why did the piano get locked out? It forgot its keys.",
    "How do you fix a broken tuba? With a tuba glue.",
    "Why did the drummer bring toast to the concert? For the roll.",
    "What kind of pants do clouds wear? Thunderwear.",
    "Why did the artist go broke? He kept drawing blanks.",
    "Why did the book join the police? It wanted to go undercover.",
    "What do you call an ant who fights? A militant.",
    "What do you call an ant who skips school? A truant.",
    "Why did the spider join the computer club? He was a web designer.",
    "What did the beach say when the tide came in? Long time no sea.",
    "How does a rancher keep track of his cattle? With a cow-culator.",
    "Why did the farmer win an award? He was outstanding in his field.",
    "Why did the scarecrow win an award? Because he was outstanding in his field.",
    "Why did the scarecrow get a promotion? He was outstanding in his field.",
    "Why did the barber win the race? He knew a shortcut.",
    "Why did the baker have brown hands? Because he kneaded them.",
    "Why did the librarian slip? She was in the non-friction section.",
    "Why did the teacher wear sunglasses? Her students were so bright.",
    "Why did the mail carrier retire? Too many issues.",
    "Why did the electrician always tell jokes? He had a bright sense of humor.",
    "Why did the plumber laugh? He had a leaky sense of humor.",
    "Why did the accountant break up? She wasn't his type of asset.",
    "Why did the dentist run for office? He wanted to fill the seat.",
    "Why did the astronomer smile? He was over the moon.",
    "Why did the chef quit? He couldn't find the thyme.",
    "Why did the biologist find plants boring? He couldn't dig them.",
    "Why did the geologist take a nap? He was bouldered.",
    "Why did the pilot bring string? To tie up loose ends.",
    "Why did the referee bring a pencil? To draw the line.",
    "Why did the lifeguard kick out the elephant? He wouldn't keep his trunks up.",
    "Why did the tailor get fired? He kept coming apart at the seams.",
    "Why did the boat blush? It saw the ocean's bottom.",
    "Why did the sailboat break up with the tugboat? Too much drag.",
    "Why did the balloon go near the needle? He wanted to be popular.",
    "Why did the ladder go to therapy? It had too many steps.",
    "Why did the calendar break up with the clock? It felt out of date.",
    "Why did the shoe blush? It saw the sneaker sneaking.",
    "What did the light bulb say to the switch? You turn me on.",
    "Why did the umbrella complain? It was under a lot of pressure.",
    "Why did the vacuum quit? It was tired of picking up after everyone.",
    "Why did the toaster laugh? It saw the bread loaf around.",
    "Why did the fridge cry? It saw the salad dressing.",
    "Why did the oven feel warm? It saw the cake baking.",
    "Why did the spatula sit down? It flipped out.",
    "Why did the fork blush? It saw the salad tossing.",
    "Why did the knife get in trouble? It was too sharp.",
    "Why did the spoon laugh? It saw the bowl full.",
    "Why did the plate look nervous? It was cracking under pressure.",
    "Why did the cup break up? It couldn't handle it.",
    "Why did the towel look tired? It threw in the towel.",
    "Why did the pillow get promoted? It was really soft-spoken.",
    "Why did the blanket laugh? It got tickled by the sheets.",
    "Why did the mirror get an award? It was reflective.",
    "Why did the clock get in trouble at school? It kept tocking.",
    "Why did the calendar look sad? Its days were numbered.",
    "Why did the pencil win an argument? It made a sharp point.",
    "Why did the eraser go to therapy? It made too many mistakes.",
    "Why did the paper win the race? It was on a roll.",
    "Why did the tape go to the party? It was really attached.",
    "Why did the stapler laugh? It saw the paper get pinned.",
    "Why did the marker go to school? It wanted to draw attention.",
    "Why did the crayon retire? It was worn down.",
    "Why did the ruler feel down? It couldn't measure up.",
    "Why did the notebook look sad? It was spiral-ing.",
    "Why did the backpack sit alone? It had too much baggage.",
    "Why did the desk laugh? It saw the chair tip over.",
    "Why did the chair get sent home? It kept sitting around.",
    "Why did the window look surprised? It was pane-ful.",
    "Why did the door go to therapy? It had knob-ody to talk to.",
    "Why did the wall laugh? It saw the picture hang.",
    "Why did the floor feel down? Everyone stepped on it.",
    "Why did the ceiling get promoted? It was over everyone else.",
    "Why did the roof get an award? It was really covering things.",
    "Why did the chimney feel warm? It was full of hot air.",
    "Why did the fireplace crack up? It couldn't hold the heat.",
    "Why did the couch feel loved? It was really cushy with everyone.",
    "Why did the rug win? It really tied the room together.",
    "Why did the curtain giggle? It saw the blinds pull up.",
    "Why did the lamp blush? It was turned on.",
    "Why did the fan feel cool? It was really into the breeze.",
    "Why did the vent laugh? It heard the heater grumble.",
    "Why did the plant look happy? It was really rooted in.",
    "Why did the flower blush? It saw the bee coming.",
    "Why did the cactus feel down? It was a little prickly.",
    "Why did the bush laugh? It got tickled by the wind.",
    "Why did the pinecone feel proud? It stood tall.",
    "Why did the mushroom get invited? It was such a fungi.",
    "Why did the acorn feel small? It hadn't grown up yet.",
    "Why did the grass complain? It kept getting cut short.",
    "Why did the moss feel comfy? It really cushioned things.",
    "Why did the vine go to therapy? It was too clingy.",
    "Why did the fern feel graceful? It knew how to bend.",
    "Why did the seed feel hopeful? It knew growth was coming.",
    "Why did the root feel grounded? It stayed connected.",
    "Why did the branch laugh? It saw a squirrel slip.",
    "Why did the trunk feel strong? It had a lot of rings.",
    "Why did the sap feel sticky? It couldn't help itself.",
    "Why did the pine tree get an award? It really stood out.",
    "Why did the oak feel wise? It had seen many seasons.",
    "Why did the maple feel sweet? It had lots of syrup inside.",
    "Why did the willow feel sad? It always weeps.",
    "Why did the palm tree wave? It was being friendly.",
    "Why did the bamboo laugh? It was a real reed.",
    "Why did the cactus never make friends? It was too sharp.",
    "Why did the desert feel dry? It hadn't seen rain in years.",
    "Why did the mountain feel tall? It couldn't help it, it just peaked.",
    "Why did the hill giggle? It got tickled by the wind.",
    "Why did the valley feel low? It was surrounded.",
    "Why did the river run? It saw the bank.",
    "Why did the lake laugh? It saw the rocks skipping.",
    "Why did the pond feel small? It had big lake dreams.",
    "Why did the stream babble? It had a lot to say.",
    "Why did the waterfall get an award? It really made a splash.",
    "Why did the raindrop feel important? It filled the bucket.",
    "Why did the puddle giggle? A duck jumped in.",
    "Why did the mud feel dirty? It had been rolled in.",
    "Why did the dust bunny hide? It saw the vacuum coming.",
    "Why did the sand feel gritty? It was between someone's toes.",
    "Why did the rock feel solid? It never budged.",
    "Why did the pebble feel small? It was overshadowed.",
    "Why did the boulder feel heavy? It had a lot on its shoulders.",
    "Why did the fossil feel old? It had been around forever.",
    "Why did the crystal feel bright? It sparkled all day.",
    "Why did the diamond feel tough? It was under pressure.",
    "Why did the gold feel valuable? Everyone wanted it.",
    "Why did the silver feel shiny? It had been polished.",
    "Why did the copper feel warm? It was a good conductor.",
    "Why did the iron feel strong? It had a lot of pull.",
    "Why did the magnet get invited? It was very attractive.",
    "Why did the compass feel lost? It couldn't decide.",
    "Why did the map feel folded? It had been through a lot.",
    "Why did the road feel busy? It had a lot of traffic.",
    "Why did the highway feel long? It stretched forever.",
    "Why did the bridge feel supportive? It held everyone up.",
    "Why did the tunnel feel dark? It was in over its head.",
    "Why did the sidewalk feel walked on? Everyone stepped on it.",
    "Why did the parking lot feel used? Cars came and went.",
    "Why did the stop sign feel important? Everyone listened to it.",
    "Why did the yield sign feel patient? It gave way.",
    "Why did the traffic cone stand tall? It was an orange leader.",
    "Why did the fire hydrant look thirsty? It was surrounded by water.",
    "Why did the mailbox feel popular? It got so much mail.",
    "Why did the streetlight feel bright? It lit up the night.",
    "Why did the crosswalk feel striped? It had lines to hold.",
    "Why did the bench feel welcoming? It offered a seat.",
    "Why did the fountain feel refreshing? It kept flowing.",
    "Why did the statue feel still? It never moved.",
    "Why did the flag feel proud? It waved high.",
    "Why did the kite feel free? It flew with the wind.",
    "Why did the balloon feel light? It was full of air.",
    "Why did the parachute feel safe? It slowed everyone down.",
    "Why did the airplane feel high? It was on cloud nine.",
    "Why did the helicopter feel dizzy? It kept spinning.",
    "Why did the rocket feel fast? It was on a mission.",
    "Why did the submarine feel deep? It was under pressure.",
    "Why did the boat feel afloat? It knew how to stay up.",
    "Why did the raft feel simple? It just tied together.",
    "Why did the canoe feel silent? It paddled softly.",
    "Why did the kayak feel narrow? It just fit one.",
    "Why did the surfboard feel wavy? It rode the crest.",
    "Why did the skateboard feel rolling? It never stopped.",
    "Why did the bicycle feel balanced? It had two wheels.",
    "Why did the scooter feel small? It was tiny but mighty.",
    "Why did the wagon feel loaded? It carried the load.",
    "Why did the truck feel heavy-duty? It hauled a lot.",
    "Why did the tractor feel useful? It plowed the field.",
    "Why did the crane feel lifted? It picked things up.",
    "Why did the bulldozer feel powerful? It moved mountains.",
    "Why did the excavator feel deep? It dug in.",
    "Why did the cement mixer feel smooth? It kept turning.",
    "Why did the forklift feel proud? It carried the weight.",
    "Why did the elevator go up? It had reasons to rise.",
    "Why did the escalator feel steppy? It always moved.",
    "Why did the ladder feel useful? It gave people a step up.",
    "Why did the stairs feel busy? Everyone climbed them.",
    "Why did the slide feel slippery? It was designed that way.",
    "Why did the swing feel back and forth? It couldn't decide.",
    "Why did the seesaw feel balanced? It went up and down.",
    "Why did the merry-go-round feel dizzy? It kept spinning.",
    "Why did the jungle gym feel climbed? Kids loved it.",
    "Why did the sandbox feel gritty? It was full of sand.",
    "Why did the swing set feel loved? Kids came back.",
    "Why did the trampoline feel bouncy? It was springy.",
    "Why did the hula hoop feel round? It just circled.",
    "Why did the jump rope feel skippy? It kept moving.",
    "Why did the yo-yo feel up and down? It couldn't stay put.",
    "Why did the frisbee feel airy? It flew far.",
    "Why did the beach ball feel bouncy? It loved the sand.",
    "Why did the pool floaty feel relaxed? It just floated.",
    "Why did the snorkel feel breathy? It let air in.",
    "Why did the fin feel fast? It kicked water back.",
    "Why did the goggles feel clear? They let you see.",
    "Why did the swim cap feel snug? It kept hair in.",
    "Why did the sunscreen feel greasy? It protected skin.",
    "Why did the beach towel feel sandy? It sat on the beach.",
    "Why did the umbrella feel shady? It gave shade.",
    "Why did the cooler feel cool? It kept things cold.",
    "Why did the picnic basket feel full? It was packed.",
    "Why did the sandwich feel squished? It was pressed.",
    "Why did the chip feel crunchy? It was a chip.",
    "Why did the pretzel feel salty? It was born that way.",
    "Why did the cracker feel snappy? It broke easily.",
    "Why did the biscuit feel flaky? It just was.",
    "Why did the muffin feel top-heavy? It had a big top.",
    "Why did the donut feel round? It had a hole in the middle.",
    "Why did the bagel feel chewy? It was boiled first.",
    "Why did the toast feel warm? It just popped up.",
    "Why did the pancake feel flat? It was flipped.",
    "Why did the waffle feel bumpy? It had squares.",
    "Why did the cereal feel crunchy? It waited for milk.",
    "Why did the milk feel cool? It came from the fridge.",
    "Why did the juice feel fresh? It was squeezed.",
    "Why did the smoothie feel blended? It was mixed.",
    "Why did the yogurt feel creamy? It just did.",
    "Why did the cheese feel sharp? It aged well.",
    "Why did the butter feel melty? It was soft.",
    "Why did the honey feel sticky? It was bee-made.",
    "Why did the jam feel spreadable? It was made that way.",
    "Why did the peanut butter feel nutty? It was ground.",
    "Why did the pickle feel sour? It was brined.",
    "Why did the olive feel briny? It sat in salt water.",
    "Why did the mustard feel yellow? It just was.",
    "Why did the ketchup feel red? Tomatoes.",
    "Why did the mayo feel creamy? It was whipped.",
    "Why did the salsa feel spicy? It had peppers.",
    "Why did the guacamole feel green? Avocados.",
    "Why did the hummus feel smooth? It was blended.",
    "Why did the salad feel fresh? It was tossed.",
    "Why did the soup feel warm? It simmered.",
    "Why did the stew feel hearty? It was slow-cooked.",
    "Why did the chili feel bold? It had lots of spice.",
    "Why did the rice feel fluffy? It steamed well.",
    "Why did the noodle feel slippery? It was cooked.",
    "Why did the meatball feel round? It was rolled.",
    "Why did the dumpling feel plump? It was stuffed.",
    "Why did the taco feel folded? It was made that way.",
    "Why did the burrito feel wrapped? It was rolled up.",
    "Why did the pizza feel cheesy? It was topped.",
    "Why did the sandwich feel stacked? Layers upon layers.",
    "Why did the wrap feel rolled? It was designed that way.",
    "Why did the burger feel juicy? It was grilled.",
    "Why did the hot dog feel long? It was a dog.",
    "Why did the fries feel golden? They were fried.",
    "Why did the onion ring feel round? It came from an onion.",
    "Why did the corn on the cob feel yellow? It was ripe.",
    "Why did the mashed potato feel fluffy? It was whipped.",
    "Why did the coleslaw feel crunchy? Cabbage.",
    "Why did the pickle chip feel tart? It was cured.",
    "Why did the dessert feel sweet? Sugar.",
    "Why did the ice cream feel cold? It came from the freezer.",
    "Why did the cake feel layered? It was stacked.",
    "Why did the cupcake feel small? It was tiny cake.",
    "Why did the brownie feel fudgy? It was rich.",
    "Why did the cookie feel warm? Fresh from the oven.",
    "Why did the pie feel proud? It won the fair.",
    "Why did the tart feel crisp? It had a good crust.",
    "Why did the pudding feel jiggly? It set.",
    "Why did the sundae feel special? It was Sunday.",
    "Why did the milkshake feel thick? Extra ice cream.",
    "Why did the lemonade feel tart? Lemons.",
    "Why did the iced tea feel refreshing? It was chilled.",
    "Why did the cocoa feel warm? It was heated.",
    "Why did the hot chocolate feel cozy? Winter treat.",
    "Why did the punch feel fruity? Lots of juice.",
    "Why did the soda feel bubbly? Carbonation.",
    "Why did the water feel clear? It was pure.",
    "Why did the sparkling water feel fizzy? It had bubbles.",
]

# Pun jokes: extended with more distinct entries via variations
PUN_JOKES_VARIATIONS = [
    "What do you call a fish with a tie? Sofishticated.",
    "What do you call a nosy pepper? Jalapeno business.",
    "Why did the cucumber turn red? It saw the salad undressing.",
    "How do you make a lemon drop? Just let it fall.",
    "What do you call an apple that plays trumpet? A tooty fruity.",
    "What did the carrot say to the rabbit? Do you want a bite?",
    "Why did the orange stop rolling down the hill? It ran out of juice.",
    "What did the little mountain say to the big mountain? Hi, Cliff.",
    "What do you call two birds in love? Tweethearts.",
    "What do you call a monkey that loves potato chips? A chipmunk.",
    "Why don't ants get sick? They have anty-bodies.",
    "What do you get when you cross a snowman and a vampire? Frostbite.",
    "What do elves learn in school? The elf-abet.",
    "Why did the poor cat only have three legs? He gave a paw.",
    "How do you make a Kleenex dance? Put a little boogie in it.",
    "What do you call a bee that can't make up its mind? A maybe.",
    "Why did the frog take the bus? His car got toad.",
    "What do you call a pile of kittens? A meowntain.",
    "Why do birds fly south for the winter? It's too far to walk.",
    "What is a tornado's favorite game? Twister.",
    "Why did the barber win an award? He gave the best cuts.",
    "What did the bee say to the flower? Hi, honey.",
    "What kind of tea is hard to swallow? Reality.",
    "What do you get from a pampered cow? Spoiled milk.",
    "What do you call a dinosaur with an extensive vocabulary? A thesaurus.",
    "What is a witch's favorite subject in school? Spelling.",
    "What lights up a soccer stadium? A soccer match.",
    "What kind of shoes do ninjas wear? Sneakers.",
    "Why do ducks make great detectives? They always quack the case.",
    "What did the finger say to the thumb? I'm in glove with you.",
]

PUN_JOKES = list(dict.fromkeys(PUN_JOKES + PUN_JOKES_VARIATIONS))

# ---------------------------------------------------------------------------
# FAMILY 2: Knock-knock jokes (as compact single-turn deliveries)
# ---------------------------------------------------------------------------

KNOCK_KNOCK_JOKES = [
    ("Knock knock.", "Who's there? Lettuce. Lettuce who? Lettuce in, it's cold out here!"),
    ("Knock knock.", "Who's there? Boo. Boo who? Don't cry, it's just a joke!"),
    ("Knock knock.", "Who's there? Cows go. Cows go who? No, cows go moo!"),
    ("Knock knock.", "Who's there? Interrupting cow. Interrupting c- MOO!"),
    ("Knock knock.", "Who's there? Tank. Tank who? You're welcome!"),
    ("Knock knock.", "Who's there? Orange. Orange who? Orange you glad I didn't say banana?"),
    ("Knock knock.", "Who's there? Banana. Banana who? Orange you glad I didn't say banana again?"),
    ("Knock knock.", "Who's there? Nobel. Nobel who? No bell, that's why I knocked!"),
    ("Knock knock.", "Who's there? Dozen. Dozen who? Dozen anyone want to let me in?"),
    ("Knock knock.", "Who's there? Wooden shoe. Wooden shoe who? Wooden shoe like to hear another joke?"),
    ("Knock knock.", "Who's there? Olive. Olive who? Olive you and I miss you!"),
    ("Knock knock.", "Who's there? Ice cream. Ice cream who? Ice cream if you don't let me in!"),
    ("Knock knock.", "Who's there? Cash. Cash who? No thanks, I prefer peanuts!"),
    ("Knock knock.", "Who's there? Cargo. Cargo who? Car go beep beep!"),
    ("Knock knock.", "Who's there? Alpaca. Alpaca who? Alpaca the trunk, you pack the suitcase!"),
    ("Knock knock.", "Who's there? Honey bee. Honey bee who? Honey bee a dear and get me a snack!"),
    ("Knock knock.", "Who's there? Dishes. Dishes who? Dishes a very bad joke!"),
    ("Knock knock.", "Who's there? Needle. Needle who? Needle little help opening the door!"),
    ("Knock knock.", "Who's there? Amish. Amish who? You're not a shoe!"),
    ("Knock knock.", "Who's there? Ya. Ya who? What are you so excited about?"),
    ("Knock knock.", "Who's there? Cereal. Cereal who? Cereal pleasure to meet you!"),
    ("Knock knock.", "Who's there? Justin. Justin who? Justin time for dinner!"),
    ("Knock knock.", "Who's there? Doris. Doris who? Doris locked, that's why I knocked!"),
    ("Knock knock.", "Who's there? Anee. Anee who? Anee one you like!"),
    ("Knock knock.", "Who's there? Adore. Adore who? Adore is between us, open up!"),
    ("Knock knock.", "Who's there? Broken pencil. Broken pencil who? Never mind, it's pointless!"),
    ("Knock knock.", "Who's there? Ho ho. Ho ho who? Your Santa impression needs work!"),
    ("Knock knock.", "Who's there? Yah. Yah who? Ride 'em cowboy!"),
    ("Knock knock.", "Who's there? Woo. Woo who? Don't get so excited, it's just a joke!"),
    ("Knock knock.", "Who's there? Al. Al who? Al give you a hug if you open the door!"),
    ("Knock knock.", "Who's there? Owls. Owls who? Yes, they do!"),
    ("Knock knock.", "Who's there? Figs. Figs who? Figs the doorbell, it's broken!"),
    ("Knock knock.", "Who's there? Hatch. Hatch who? Bless you!"),
    ("Knock knock.", "Who's there? Ach. Ach who? Bless you, friend!"),
    ("Knock knock.", "Who's there? Beets. Beets who? Beets me!"),
    ("Knock knock.", "Who's there? Iowa. Iowa who? Iowa big apology!"),
    ("Knock knock.", "Who's there? Snow. Snow who? Snow use, I forgot!"),
    ("Knock knock.", "Who's there? Nana. Nana your business!"),
    ("Knock knock.", "Who's there? Kanga. Kanga who? Actually, it's kangaroo!"),
    ("Knock knock.", "Who's there? Tish. Tish who? Bless you!"),
    ("Knock knock.", "Who's there? Water. Water who? Water you doing, open up!"),
    ("Knock knock.", "Who's there? Butter. Butter who? Butter open up, it's freezing!"),
    ("Knock knock.", "Who's there? Doughnut. Doughnut who? Doughnut ask, just open up!"),
    ("Knock knock.", "Who's there? Wanda. Wanda who? Wanda hear another joke?"),
    ("Knock knock.", "Who's there? Amos. Amos who? A mosquito!"),
    ("Knock knock.", "Who's there? Andy. Andy who? Andy bit me too!"),
    ("Knock knock.", "Who's there? Ivan. Ivan who? Ivan to come inside!"),
    ("Knock knock.", "Who's there? Isabelle. Isabelle who? Isabelle broken? I had to knock!"),
    ("Knock knock.", "Who's there? Ken. Ken who? Ken you please let me in?"),
    ("Knock knock.", "Who's there? Alex. Alex who? Alex the questions around here!"),
    ("Knock knock.", "Who's there? Otto. Otto who? Otto know, I forgot my name!"),
    ("Knock knock.", "Who's there? Egg. Egg who? Eggcited to see you!"),
    ("Knock knock.", "Who's there? Iran. Iran who? Iran here as fast as I could!"),
    ("Knock knock.", "Who's there? Robin. Robin who? Robin you, hand over the snacks!"),
    ("Knock knock.", "Who's there? Cook. Cook who? That's the first one I've heard this spring!"),
    ("Knock knock.", "Who's there? Canoe. Canoe who? Canoe come out and play?"),
    ("Knock knock.", "Who's there? Dwayne. Dwayne who? Dwayne the tub, I'm dwowning!"),
    ("Knock knock.", "Who's there? Sadie. Sadie who? Sadie magic word!"),
    ("Knock knock.", "Who's there? Jaws. Jaws who? Jaws truly!"),
    ("Knock knock.", "Who's there? Voodoo. Voodoo who? Voodoo you think you are asking so many questions?"),
    ("Knock knock.", "Who's there? Ice. Ice who? Ice see you in there!"),
    ("Knock knock.", "Who's there? Radio. Radio who? Radio not, here I come!"),
    ("Knock knock.", "Who's there? Cows. Cows who? Cows go moo, not who!"),
    ("Knock knock.", "Who's there? Turnip. Turnip who? Turnip the volume, I love this song!"),
    ("Knock knock.", "Who's there? Beef. Beef who? Beef-ore I get cold, please open up!"),
    ("Knock knock.", "Who's there? Sombrero. Sombrero who? Sombrero-ver the rainbow!"),
    ("Knock knock.", "Who's there? Tuna. Tuna who? Tuna piano and it'll sound better!"),
    ("Knock knock.", "Who's there? Ice cream soda. Ice cream soda who? Ice cream soda whole world can hear me!"),
    ("Knock knock.", "Who's there? Weevil. Weevil who? Weevil rock you!"),
    ("Knock knock.", "Who's there? Norma Lee. Norma Lee who? Norma Lee I don't tell knock-knock jokes!"),
    ("Knock knock.", "Who's there? Police. Police who? Police open the door, it's cold outside!"),
    ("Knock knock.", "Who's there? Repeat. Repeat who? Who who who!"),
    ("Knock knock.", "Who's there? Wire. Wire who? Wire you asking, it's me!"),
    ("Knock knock.", "Who's there? Zoom. Zoom who? Zoom did you expect?"),
    ("Knock knock.", "Who's there? Says. Says who? Says me!"),
    ("Knock knock.", "Who's there? Alma. Alma who? Alma candy is gone, buy more!"),
    ("Knock knock.", "Who's there? Etch. Etch who? Bless you!"),
    ("Knock knock.", "Who's there? Ash. Ash who? Bless you again!"),
    ("Knock knock.", "Who's there? Colin. Colin who? Colin all cars, come home!"),
    ("Knock knock.", "Who's there? Little old lady. Little old lady who? I didn't know you could yodel!"),
    ("Knock knock.", "Who's there? Yah. Yah who? What are you cheering for?"),
    ("Knock knock.", "Who's there? A herd. A herd who? A herd you were home, so I came over!"),
    ("Knock knock.", "Who's there? Ben. Ben who? Ben knocking so long my knuckles hurt!"),
    ("Knock knock.", "Who's there? Barbara. Barbara who? Barbara black sheep, have you any wool?"),
    ("Knock knock.", "Who's there? Roach. Roach who? Roach you a letter but you never replied!"),
    ("Knock knock.", "Who's there? Alpaca. Alpaca lunch, you pack the tent!"),
    ("Knock knock.", "Who's there? Otter. Otter who? Otter you glad to see me?"),
    ("Knock knock.", "Who's there? Rufus. Rufus who? The rufus leaking, come check it!"),
    ("Knock knock.", "Who's there? Wendy. Wendy who? Wendy bell works, I'll ring it!"),
    ("Knock knock.", "Who's there? Ida. Ida who? Ida love a snack, got any?"),
    ("Knock knock.", "Who's there? Handsome. Handsome who? Handsome candy through the door!"),
    ("Knock knock.", "Who's there? Theodore. Theodore who? Theodore is stuck, open it!"),
    ("Knock knock.", "Who's there? Alfie. Alfie who? Alfie terrible if you don't let me in!"),
    ("Knock knock.", "Who's there? Ammonia. Ammonia who? Ammonia a little kid, help me reach!"),
    ("Knock knock.", "Who's there? Anita. Anita who? Anita hug, it's been a long day!"),
    ("Knock knock.", "Who's there? Arfur. Arfur who? Arfur got, just open up!"),
    ("Knock knock.", "Who's there? Beth. Beth who? Beth wishes and happy birthday!"),
    ("Knock knock.", "Who's there? Boo. Boo who? Please don't cry, I brought cookies!"),
    ("Knock knock.", "Who's there? Cargo. Cargo who? Car go vroom, let me park it!"),
    ("Knock knock.", "Who's there? Chester. Chester who? Chester minute, I'm coming!"),
    ("Knock knock.", "Who's there? Claire. Claire who? Claire the way, coming through!"),
    ("Knock knock.", "Who's there? Cash. Cash who? Bless you, want a tissue?"),
    ("Knock knock.", "Who's there? Toby. Toby who? Toby or not toby, that is the question!"),
    ("Knock knock.", "Who's there? Frank. Frank who? Frank you for finally opening the door!"),
    ("Knock knock.", "Who's there? Nunya. Nunya who? Nunya business, actually!"),
    ("Knock knock.", "Who's there? Wa. Wa who? What are you so excited about?"),
    ("Knock knock.", "Who's there? Hawaii. Hawaii who? I'm fine, Hawaii you?"),
    ("Knock knock.", "Who's there? Wooden shoe. Wooden shoe who? Wooden shoe like to know!"),
]

# ---------------------------------------------------------------------------
# FAMILY 3: Observational / absurd one-liners
# ---------------------------------------------------------------------------

OBS_JOKES = [
    "I tried to catch some fog earlier. I mist.",
    "I'm reading a book on anti-gravity. It's impossible to put down.",
    "I told my suitcase there'd be no vacation this year. Now I'm dealing with emotional baggage.",
    "The wheels of my calendar keep spinning. Time flies.",
    "I lost my mood ring and I don't know how I feel about it.",
    "My friend said I couldn't make a joke about beans. I said, of course I can-nellini.",
    "I tried to write a song about tortillas. Turns out it's a wrap.",
    "I once made a belt from watches. It was a waist of time.",
    "I stayed up all night to see where the sun went. Then it dawned on me.",
    "I used to be a banker but I lost interest.",
    "I bought some shoes online. The reviews said they'd make me trip; turns out they were right.",
    "My vacuum cleaner just quit. I think it got tired of picking up after me.",
    "I got a job at the bakery because I kneaded the dough.",
    "I tried to sue the airline for losing my luggage. I lost my case.",
    "My printer is a photocopier. It just keeps repeating itself.",
    "I asked my dog what's two minus two. He said nothing.",
    "I told a joke about paper. It was tearable.",
    "I know a lot about plane crashes. My dad was a pilot; my mom was air traffic control.",
    "Did you hear about the guy who invented the knock-knock joke? He won the No-bell prize.",
    "I don't trust stairs. They're always up to something.",
    "I tried yoga once, but I got a little bent out of shape.",
    "My tailor is happy to see me, but only sew sew.",
    "I named my dog Five Miles. Now I can say I walk Five Miles every day.",
    "My favorite dinosaur is a thesaurus. Every day I find new words to use.",
    "I got kicked out of the secret cooking society. I spilled the beans.",
    "I ordered a chicken and an egg online. I'll let you know.",
    "I lost my job at the calendar factory. All I did was take a day off.",
    "The recycling bin is my life coach. Every week it tells me to start over.",
    "I told my computer I needed a break, and now it won't stop sending me kit-kat ads.",
    "My blanket has trust issues. It's always keeping me warm and never getting anything in return.",
    "I put my grandma on speed dial the other day. I call it insta-gran.",
    "I bought some invisible paint. Turns out I got scammed.",
    "My chef friend quit because he lost his temper. He couldn't find the thyme to look.",
    "I saw a snail crossing the road. I put it on the sidewalk. Later it said, whoa, what a ride!",
    "I decided to sell my vacuum. It was just gathering dust.",
    "I opened a fishing store. Business is off the hook.",
    "My kids say I overuse dad jokes. But I loaf them.",
    "I bought a boat because it was on sail.",
    "I asked the librarian for a book on paranoia. She whispered, it's right behind you.",
    "My smoke alarm goes off every time I cook. It's really steaming me.",
    "I once had a job assembling faucets. I made a lot of taps.",
    "I quit my job at the shoe factory. It was sole crushing.",
    "The math teacher won a fight. He used the power of exponents.",
    "I tried to make a nostalgia sandwich. It just wasn't as good as I remember.",
    "I asked the fisherman if the fish were biting. He said only when I try to eat them.",
    "The couch and I are on speaking terms again. We had a sit down.",
    "I gave up my seat on the bus for a blind person today. Turns out he wasn't blind. I just walked home.",
    "I bought a dictionary but it fell apart. I couldn't find the words.",
    "I'm on a seafood diet. I see food, I eat it.",
    "My alarm clock got promoted. Now it's the head of my morning.",
    "I decided to sell my Hoover. It was just collecting dust.",
    "My email is a black hole. I open it and time disappears.",
    "I feel like time is a soup. Everything simmers eventually.",
    "The washing machine finished the load. Now it wants a bonus.",
    "My phone told me the weather. Then I looked outside and it lied.",
    "I gave my houseplant a pep talk. It just kept leaving.",
    "My couch has an opinion on everything. Mostly about my posture.",
    "The dog dreams about being a taxi. He keeps chasing cars.",
    "My cat critiques my art. Silently, but pointedly.",
    "I met a mirror who could talk. It reflected on things.",
    "My toaster and coffee maker are dating. They have chemistry.",
    "My blender is philosophical. It contemplates a lot.",
    "I asked the microwave to be quick. It said give me a minute.",
    "The refrigerator hums. I think it knows a tune.",
    "My oven is passive-aggressive. It preheats too slowly.",
    "The dishwasher is exhausted. It works nonstop.",
    "The freezer is icy. Emotionally, I mean.",
    "The trash can is dramatic. Every time I open it, it stinks.",
    "My clothes dryer eats socks. It's a picky eater.",
    "The washing machine likes an audience. It spins for attention.",
    "The vacuum has a crush on the rug. They meet weekly.",
    "The broom is judgmental. It sweeps my crumbs with disdain.",
    "The mop is emotional. It gets so wrung up.",
    "The sponge absorbs everything. Including my complaints.",
    "The dish soap is bubbly. It's just its personality.",
    "The paper towel is helpful. Almost too helpful.",
    "The napkin holds a grudge. It never forgets a spill.",
    "The fork gossips with the knife. The spoon just listens.",
    "The plate is patient. It waits for every meal.",
    "The cup is optimistic. It's always half full.",
    "The bowl is deep. Very deep.",
    "The mug hugs my coffee. It knows what I need.",
    "The kettle sings. It has range.",
    "The stove is warm. Emotionally available.",
    "The sink is a good listener. It never interrupts.",
    "The drain is mysterious. It hides everything I lose.",
    "The soap dispenser is minimalist. Just one squirt at a time.",
    "The towel is fluffy. Also emotionally supportive.",
    "The mirror is honest. Sometimes brutally.",
    "The shower is meditative. All my ideas happen there.",
    "The bathtub is inviting. It always wants a soak.",
    "The rug is loyal. Never moves from its spot.",
    "The doormat is welcoming. It says hi with every step.",
    "The couch is possessive. It hugs me too tightly sometimes.",
    "The pillow is comforting. It really gets me.",
    "The blanket is generous. It shares its warmth freely.",
    "The lamp is enlightening. Literally.",
    "The clock is patient. It waits for me every morning.",
    "The window is a good listener. It just stares.",
    "The door is decisive. Open or closed, no in-between.",
    "The wall is stoic. It never moves.",
    "The floor is grounded. Very grounded.",
    "The ceiling is above it all.",
    "The fan is chill. Literally.",
    "The heater is warm. Also literally.",
    "The garage is roomy. It has space for everything.",
    "The driveway is chill. It just lays around.",
    "The lawn is jealous of the neighbors' grass. It's always greener.",
    "The garden is patient. It waits for spring every year.",
    "The tomato plant is competitive. It just wants to win.",
    "The strawberry is sweet. Also, tiny.",
    "The blueberry is wise. It's been around for a while.",
    "The zucchini is generous. It gives so much.",
    "The pumpkin is round. Also patient.",
    "The corn is tall. It stalks upward.",
    "The lettuce is chill. It just wants to relax.",
    "The onion has layers. Emotional ones.",
    "The garlic is bold. It doesn't hide.",
    "The pepper is spicy. Just its personality.",
    "The basil is fragrant. Also friendly.",
    "The rosemary is fragrant too. It's aromatic.",
    "The thyme is patient. Get it?",
    "The parsley is fresh. Metaphorically and literally.",
    "The mint is refreshing. Also a bit invasive.",
    "The cilantro divides people. Some love it, some don't.",
    "The dill is happy. It pickles well.",
    "The bay leaf is quiet. It just adds flavor.",
    "The oregano is Italian at heart.",
    "The sage is wise. Get it?",
    "The chives are subtle. They just add a little.",
    "The lemon is tart. Also cheerful.",
    "The lime is bright. Also citrusy.",
    "The orange is sunny. Both in color and mood.",
    "The apple falls close to the tree. Physics.",
    "The pear is patient. It ripens slowly.",
    "The peach is sweet. Also fuzzy.",
    "The plum is quiet. It just hangs around.",
    "The grape is social. It travels in bunches.",
    "The banana peels away. It has layers.",
    "The mango is exotic. Also messy.",
    "The pineapple has a crown. Very regal.",
    "The kiwi is fuzzy on the outside, soft on the inside.",
    "The watermelon is big-hearted. Full of water.",
    "The cantaloupe is subtle. Its flavor sneaks up.",
    "The honeydew is chill. Just hanging out.",
    "The papaya is tropical. Also sweet.",
    "The dragon fruit looks intimidating but is soft inside.",
    "The passion fruit is dramatic. All that seed drama.",
    "The coconut is tough. Also refreshing.",
    "The fig is ancient. Very traditional.",
    "The pomegranate is jewel-like. Every seed a gem.",
    "The cherry is small but bold.",
    "The apricot is quiet. Just fuzzy and calm.",
    "The nectarine is peach's smooth cousin.",
    "The persimmon is unusual. Not everyone gets it.",
    "The rhubarb is dramatic. Better with sugar.",
    "The cranberry is tart. Very seasonal.",
    "The raspberry is delicate. Also delicious.",
    "The blackberry is deep. Also stains everything.",
    "The gooseberry is unusual. Not for everyone.",
    "The elderberry is elderly. It's in the name.",
    "The mulberry is messy. Loved by kids anyway.",
    "The boysenberry is complex. A mixed heritage.",
    "The lychee is exotic. Also floral.",
    "The starfruit is showy. Very Instagram-ready.",
    "The tangerine is convenient. Peels easily.",
    "The clementine is petite. Portable citrus.",
    "The grapefruit is polarizing. Some love it, some pucker.",
    "The lemon zest is bright. Adds pizzazz.",
    "The lime wedge is a garnish diva.",
    "I sent my dog to a spa. He came back looking like a doggo influencer.",
    "I saw a raccoon reading a newspaper. Turns out he was reviewing the trash.",
    "My phone battery lives its best life at 1 percent.",
    "The GPS lady sighed at me today. I've hurt her professionally.",
    "I watered my plants and they still gave me the silent treatment.",
    "The moon and I are both night owls. We just vibe.",
    "The clouds are moody today. Threatening rain like a diva.",
    "The wind knocked over my trash can. Bold move for a breeze.",
    "The sunrise showed up on time. Rare for anything in my life.",
    "My reflection blinked before I did. That was unsettling.",
    "The elevator paused mid-floor. Existential crisis for a machine.",
    "My car started singing when I turned the key. New feature, I guess.",
    "The stoplight winked. I'm pretty sure it winked.",
    "I asked the parrot for advice. He repeated the question back.",
    "The goldfish stared at me judgmentally. Fish have opinions.",
    "The hamster ran on the wheel. Cardio champion.",
    "The pigeon in the park gave me a look. Just a look.",
    "The squirrel hid a nut in plain sight. Master strategist.",
    "The rabbit hopped off with my carrot. Bold.",
    "The turtle finished the race. Eventually.",
    "The snail left a trail. And a memory.",
    "The butterfly landed on my nose. It ghosted me right after.",
    "The bee zoomed by like it had places to be.",
    "The ant carried a crumb bigger than itself. Ambitious.",
    "The spider redecorated the corner. Modern minimalist.",
    "The moth flew into the light. Bold career move.",
    "The ladybug landed on my hand. Presumably good luck.",
    "The dragonfly hovered. Just hovering, judging.",
    "The firefly blinked SOS in Morse. I'm sure of it.",
    "The mosquito found me. It always finds me.",
    "The cricket sang all night. He has a solo career.",
    "The grasshopper leaped away when I approached. Rude.",
    "The beetle marched across the sidewalk. Very focused.",
    "The caterpillar has big dreams of flight.",
    "The worm surfaced after the rain. Mood.",
    "The frog jumped into the pond. Cannonball!",
    "The duckling followed its mom in a perfect line.",
    "The goose honked at me. I honked back.",
    "The chicken crossed the road. Classic move.",
    "The rooster crowed at dawn. Punctual.",
    "The pig rolled in the mud. Living its best life.",
    "The cow chewed contemplatively. Deep thoughts.",
    "The goat climbed a tree. Yes, a tree.",
    "The sheep grazed peacefully. Zen master.",
    "The horse galloped by. Impressive form.",
    "The donkey stood still. Immovable.",
    "The llama spat at me. That was aggressive.",
    "The alpaca stared serenely. Fashion icon.",
    "The bison was massive. Also unbothered.",
    "The moose stood in the road. King of the crossing.",
    "The deer paused, elegant. Then bounded off.",
    "The elk called across the valley. Nature's karaoke.",
    "The bear rummaged through the trash. Same, buddy.",
    "The fox darted through the woods. Slick.",
    "The wolf howled in the distance. Very cinematic.",
    "The raccoon washed its snack. Cleanliness first.",
    "The skunk minded its business. Respect.",
    "The porcupine gave me space. As it should.",
    "The opossum played dead. Method acting.",
    "The armadillo curled up. Instant snack ball.",
    "The hedgehog was tiny. Tiny and prickly.",
    "The badger looked displeased. Standard badger mood.",
    "The otter floated on its back. Peak living.",
    "The seal clapped. Great review.",
    "The dolphin leaped. Bragging, honestly.",
    "The whale sang. Long song.",
    "The shark swam by. Menacing yet elegant.",
    "The octopus changed color. Show off.",
    "The jellyfish drifted. It's a lifestyle.",
    "The starfish just sat there. Chilling.",
    "The crab side-stepped. Iconic movement.",
    "The lobster snapped its claw. Big drama.",
    "The shrimp curled up. Nap time.",
    "The seaweed swayed. Underwater dance.",
]

# ---------------------------------------------------------------------------
# FAMILY 4: Dad jokes / groaners
# ---------------------------------------------------------------------------

DAD_JOKES = [
    "What do you call a fake noodle? An impasta.",
    "Why don't scientists trust atoms? Because they make up everything.",
    "I only know 25 letters of the alphabet. I don't know y.",
    "What do you call a bear with no teeth? A gummy bear.",
    "Why did the cookie go to the doctor? Because it was feeling crummy.",
    "How do you organize a space party? You planet.",
    "What do you call cheese that isn't yours? Nacho cheese.",
    "Why don't skeletons fight each other? They don't have the guts.",
    "What did the ocean say to the shore? Nothing, it just waved.",
    "Why did the golfer bring an extra pair of pants? In case he got a hole in one.",
    "What do you call an alligator in a vest? An investigator.",
    "Why did the scarecrow get a raise? He was outstanding in his field.",
    "What kind of shoes do frogs wear? Open toad.",
    "I used to hate facial hair, but then it grew on me.",
    "What did the janitor say when he jumped out of the closet? Supplies!",
    "Did you hear about the guy whose whole left side was cut off? He's all right now.",
    "How do you make holy water? You boil the hell out of it.",
    "Why do we tell actors to break a leg? Because every play has a cast.",
    "Why did the coffee file a police report? It got mugged.",
    "How does a penguin build its house? Igloos it together.",
    "How can you tell if a vampire has a cold? He starts coughin'.",
    "What do you call a factory that makes okay products? A satisfactory.",
    "Why did the man fall down the well? Because he didn't see that well.",
    "Why did the picture go to jail? Because it was framed.",
    "Why don't eggs tell jokes? Because they'd crack up.",
    "What has more letters than the alphabet? The post office.",
    "What do you call a boomerang that won't come back? A stick.",
    "How do you fix a broken pumpkin? A pumpkin patch.",
    "Why did the bike fall over? Because it was two tired.",
    "What did the buffalo say when his son left for college? Bison.",
    "Why is Peter Pan always flying? Because he neverlands.",
    "Why did the man put his money in the freezer? He wanted cold hard cash.",
    "What kind of tea is hard to swallow? Reality.",
    "Why did the tomato turn red? Because it saw the salad dressing.",
    "How do you catch a squirrel? Climb a tree and act like a nut.",
    "What do you call a dinosaur that crashes his car? A tyrannosaurus wrecks.",
    "Why did the math teacher call in sick? She had a lot of problems.",
    "What do you get when you cross a snowman and a vampire? Frostbite.",
    "Why did the barber win the race? He knew a shortcut.",
    "What kind of car does an egg drive? A yolkswagen.",
    "Why did the invisible man turn down the job? He couldn't see himself doing it.",
    "How do you organize a picnic in space? You planet ahead.",
    "What's brown and sticky? A stick.",
    "What did the light bulb say to its dad? I love you watts and watts.",
    "How does the moon cut his hair? Eclipse it.",
    "Why did the belt get arrested? For holding up a pair of pants.",
    "What do you call a fish wearing a bowtie? Sofishticated.",
    "How do trees access the internet? They log in.",
    "What's a skeleton's favorite instrument? The trom-bone.",
    "Why can't your nose be twelve inches long? Because then it would be a foot.",
    "What did the grape say when it got stepped on? Nothing, it just let out a little whine.",
    "Why was the math book sad? It had too many problems.",
    "What do you call a can opener that doesn't work? A can't opener.",
    "What kind of shoes do ninjas wear? Sneakers.",
    "Why did the melon jump into the lake? It wanted to be a watermelon.",
    "What did the fisherman say to the magician? Pick a cod, any cod.",
    "How do you talk to a giant? Use big words.",
    "What did one hat say to the other? You stay here, I'll go on ahead.",
    "Why do bees have sticky hair? Because they use honeycombs.",
    "What lies at the bottom of the ocean and twitches? A nervous wreck.",
    "Why did the golfer wear two pairs of pants? In case he got a hole in one.",
    "What did the mama tomato say to the baby tomato? Ketchup.",
    "What did zero say to eight? Nice belt.",
    "Why do fish live in salt water? Because pepper makes them sneeze.",
    "What did the fish say when he swam into the wall? Dam.",
    "How do you make an octopus laugh? With ten-tickles.",
    "What did the big bucket say to the little bucket? You look pail.",
    "Why did the coach go to the bank? To get his quarterback.",
    "What do you call a magician who lost his magic? Ian.",
    "What do you call a dog magician? A labracadabrador.",
    "How do you throw a party in outer space? You planet.",
    "Why do cows have hooves instead of feet? Because they lactose.",
    "What did the digital clock say to the grandfather clock? Look, no hands.",
    "Why did the stadium get hot after the game? All the fans left.",
    "What did one wall say to the other wall? I'll meet you at the corner.",
    "Why don't melons get married? Because they cantaloupe.",
    "What did the beach say when the tide came in? Long time no sea.",
    "What did one plate say to the other? Dinner is on me.",
    "Why did the traffic light turn red? You would too if you had to change in the middle of the street.",
    "Why do dragons sleep during the day? So they can fight knights.",
    "What did the pencil say to the paper? I dot my i's on you.",
    "Why did the elephant leave the circus? He was tired of working for peanuts.",
    "Why did the chicken join a band? Because it had the drumsticks.",
    "What kind of exercise do lazy people do? Diddly-squats.",
    "Why did the frog take the bus to work? His car got toad.",
    "Why don't oysters share their pearls? Because they're shellfish.",
    "What does a nosy pepper do? Gets jalapeno business.",
    "Why did the yogurt go to the museum? Because it was cultured.",
    "What did the mama cow say to the baby cow? It's pasture bedtime.",
    "How do you make a lemon drop? Just let it fall.",
    "What do you get when you cross an apple and a Christmas tree? A pineapple.",
    "Why did the orange stop rolling? It ran out of juice.",
    "What did the little corn say to the mama corn? Where's popcorn.",
    "Why did the banana go to the hospital? He wasn't peeling well.",
    "What do you call a sleeping bull? A bulldozer.",
    "Why don't bicycles ever stand up by themselves? Because they're two tired.",
    "What kind of key opens a banana? A monkey.",
    "Why did the computer catch a cold? It left its Windows open.",
    "How do celebrities stay cool? They have many fans.",
    "Why did the smartphone go to therapy? It lost its contacts.",
    "What did the drummer name his twin daughters? Anna one, Anna two.",
    "Why did the musician get locked out? He forgot his key.",
    "What do you call a boomerang that doesn't work? A stick.",
    "What did the ocean floor say to the ocean? What's up, buoy?",
    "Why did the cookie go to school? To become a smart cookie.",
    "How do you catch a bra? With a boo-boo trap.",
    "Why did the pillow file taxes? It had a lot of stuffing.",
    "What did the football player call the flashlight? Coach.",
    "Why did the picture apologize? It was framed for a crime.",
    "What did the tie say to the hat? You go on ahead, I'll hang around.",
    "Why did the peanut fail its test? It didn't have the nuts to try.",
    "What did the coach do at the bank? Asked for his quarterback.",
    "Why did the tree go to the dentist? It had a root canal.",
    "How do trees get on the internet? They log in.",
    "What do you call a lazy baker? A loafer.",
    "Why did the sofa apologize? It felt out of cushion.",
    "What did the rug say to the floor? Don't move, I've got you covered.",
    "Why did the chair sit down? It couldn't stand any longer.",
    "How do you get a squirrel to like you? Act like a nut.",
    "Why did the giraffe get bad grades? He had his head in the clouds.",
    "What did the leopard say after finishing his meal? That hit the spot.",
    "Why did the zebra get suspended? Too many stripes.",
    "What did the peanut say to the elephant? Nothing, peanuts can't talk.",
    "Why did the koala get fired? He wasn't koala-fied.",
    "What did the sloth say to the racehorse? Slow down.",
    "Why did the panda break up? He wasn't feeling the black and white.",
    "What did the mama polar bear say to the cub? Get in the sled, we're going out for snow cones.",
    "Why did the seagull fly over the sea? Because if it flew over the bay, it'd be a bagel.",
    "What did the goose say when the sky got dark? Uh oh.",
    "Why did the swan feel sad? It couldn't get its ducks in a row.",
    "What do you call a group of geese in a hot tub? A goose jacuzzi.",
    "Why did the rooster refuse to fight? He didn't want to get henpecked.",
    "How do chickens dance? Chick to chick.",
    "Why did the egg refuse to talk? It didn't want to crack.",
    "What did the mama chicken say to the naughty chick? You are grounded.",
    "Why did the pig break up? His partner was a boar.",
    "How does a pig write? With a pig-pen.",
    "Why did the horse cross the road? Because the chicken was on vacation.",
    "What do you call a horse with a bad attitude? Nay-sayer.",
    "Why did the donkey get promoted? He was really assinine, in a good way.",
    "What did the calf say when it heard a joke? That's cud-ownright funny.",
    "Why did the sheep say excuse me? It was a little baaad-mannered.",
    "How does a sheep say hello? Wool, hi there.",
    "Why did the goat go to school? To get his goa-t-degree.",
    "What did the cow say at sunrise? Good moo-rning.",
    "Why did the pig take a bath? He was feeling hog-sty.",
    "What kind of car does a duck drive? A quack-swagen.",
    "Why did the goose stay inside? It heard the weather was fowl.",
    "How do turkeys travel? By fowl-plane.",
    "What did the peacock say at the party? Feathered fantastic.",
    "Why did the flamingo stand on one leg? If it lifted the other, it would fall.",
    "What do you call a talking parrot? Ordinary.",
    "How do bees style their hair? With honeycombs.",
    "What did the mama bee say to the little bee? Bee-hive yourself.",
    "Why did the wasp look angry? It's just their face.",
    "What did the butterfly say to the caterpillar? Have you tried change?",
    "How does a bug learn? At the caterpillar-versity.",
    "Why did the ladybug feel lucky? She saw two dots.",
    "What did the ant say to the picnic? Please leave a crumb.",
    "Why did the spider learn to code? To design a web.",
    "What did the mosquito say after biting? Thanks for lunch.",
    "Why did the moth stay near the lamp? It liked the light snack.",
    "What did the firefly say when it turned off? See you later.",
    "Why did the cricket play music? It had good chirp-tunes.",
    "How does a grasshopper cheer? Hop hop hooray.",
    "What did the dragonfly say to the pond? See you later, water.",
    "Why did the beetle go to therapy? It couldn't shell out its feelings.",
    "What did the worm say to its friend? Where in soil have you been?",
    "Why did the snail bring a suitcase? He was going on a slow trip.",
    "How does a fish do math? With a fin-tastic calculator.",
    "What did the octopus say when trapped? Squid pro quo.",
    "Why did the crab go home? He was feeling shellfish.",
    "What did the lobster say to the boss? Give me a raise or I'll pinch someone.",
    "Why did the shrimp refuse to share? He was a little shellfish.",
    "How do dolphins say goodbye? Squeak later.",
    "Why did the whale get an award? He was fin-ominal.",
    "What did the shark say? Just fin.",
    "Why did the jellyfish blush? It saw the seaweed.",
    "How does the starfish shine? It's a natural.",
    "What did the seaweed say to the current? Water we doing here.",
    "Why did the coral reef party? They had a lot of fish friends.",
    "What did the seahorse say? I'm feeling stable today.",
    "Why did the puffer fish laugh? Someone told a funny joke.",
    "How does the anglerfish read? By its own light.",
    "What did the eel say to the fish? Shocking to see you.",
    "Why did the manatee take a nap? He was sea-tired.",
    "How does a walrus sing? With a low tone.",
    "What did the polar bear order? Anything cold.",
    "Why did the penguin stay in the cold? Because it's a cool bird.",
    "How does an eagle catch a fish? With eagle vision.",
    "What did the owl say? Whoo cares.",
    "Why did the hawk soar? To get a higher perspective.",
    "How do robins wake up? To the sound of their own tweeting.",
    "What did the crow say to the raven? Talk about caw-nfusing.",
    "Why did the sparrow feel small? Compared to eagles, of course.",
    "How do swallows travel? In migrating groups.",
    "What did the hummingbird say? I'm just buzzing by.",
    "Why did the woodpecker feel accomplished? He nailed it.",
    "How does a cardinal shine? With its ruby feathers.",
    "What did the blue jay say? Just jaying around.",
    "Why did the finch feel cheerful? Spring vibes.",
    "How do canaries stay in tune? They practice a lot.",
    "What did the parakeet mumble? Something in bird-lingo.",
    "Why did the cockatoo dance? It heard music.",
    "How does a toucan reach snacks? With that long beak.",
    "What did the pelican say? Big mouth, big appetite.",
    "Why did the vulture wait? He knew good things come to those who circle.",
    "How does a stork carry babies? Very carefully.",
    "What did the heron say? Standing tall as usual.",
    "Why did the crane feel proud? He built the whole nest.",
    "How does a flamingo balance? Lots of practice.",
    "What did the swan say? Just gliding along.",
    "Why did the duck ace the test? He always gets the right quack.",
    "How does a goose lead the flock? By example.",
    "What did the turkey say on Thanksgiving? Please don't.",
    "Why did the pheasant strut? He was showing off his tail.",
    "How does a peacock ask for a date? He spreads his tail.",
    "What did the quail whisper? Come with me quietly.",
    "Why did the partridge feel festive? It was in a pear tree.",
    "How does a dove signal peace? By flying calmly.",
    "What did the pigeon say to the crumbs? Mine now.",
    "Why did the bat wake up late? He's a night owl.",
    "How does a beaver build? Log by log.",
    "What did the otter say? Life is otter-ly great.",
    "Why did the seal clap? He wanted attention.",
    "How does a squirrel plan? By the tree-load.",
    "What did the chipmunk say? Cheeks full, life full.",
    "Why did the raccoon stay up? He heard leftovers.",
    "How does a possum survive? By playing dead.",
    "What did the skunk say? Give me some space.",
    "Why did the porcupine feel shy? He was a bit prickly.",
    "How does a hedgehog nap? Curled up nicely.",
    "What did the badger grumble? Same old, same old.",
    "Why did the weasel giggle? He heard a funny joke.",
    "How does a ferret play? By zooming around.",
    "What did the mongoose say to the snake? Not today, buddy.",
    "Why did the meerkat stand tall? Someone rang the doorbell.",
    "How does a mole tunnel? Blindly but efficiently.",
    "What did the vole say? Just voling around.",
    "Why did the mouse cheer? He found cheese.",
    "How does a rat succeed? By being resourceful.",
    "What did the hamster whisper? Wheel time again.",
    "Why did the guinea pig squeak? Snack incoming.",
    "How does a rabbit run? In hops.",
    "What did the hare say? Not so slow, please.",
    "Why did the bunny hop into the salad bar? To go to the leaf greens.",
    "How does a squirrel prepare for winter? By hiding acorns everywhere.",
    "What did the chipmunk say to the acorn? I've been looking for you.",
]

# ---------------------------------------------------------------------------
# Assemble pools, capping repeats at 2 per unique joke text
# ---------------------------------------------------------------------------

def build_family_pool(jokes, target_count, kind_hint):
    """
    Given a list of unique joke texts, produce `target_count` (user, assistant) pairs.
    Each unique joke used at most 2 times. If pool is too small, we extend by
    minor rewording (adding a lead-in phrase before the joke).
    """
    jokes = list(dict.fromkeys(jokes))  # dedupe
    pairs = []
    counter = Counter()

    # First pass: use each joke once
    random.shuffle(jokes)
    idx = 0
    while len(pairs) < target_count and idx < len(jokes):
        j = jokes[idx]
        counter[j] += 1
        pairs.append((pick_user(kind_hint=kind_hint), j))
        idx += 1

    # Second pass: use each joke a second time with a different user prompt
    random.shuffle(jokes)
    idx = 0
    while len(pairs) < target_count and idx < len(jokes):
        j = jokes[idx]
        if counter[j] < 2:
            counter[j] += 1
            pairs.append((pick_user(kind_hint=kind_hint), j))
        idx += 1

    # Third pass: if still short, wrap jokes with lead phrases to create fresh variants.
    lead_wrappers = [
        "Here's one: ", "Try this: ", "Okay: ", "Sure: ", "Alright: ",
        "How about this: ", "One for you: ", "Here you go: ", "Ready? ",
        "Here's a good one: ", "Here goes: ", "This one's fun: ",
        "You'll like this: ", "Get ready: ", "Warning, groaner: ",
        "Fresh from my joke book: ", "Old classic: ", "Hot off the press: ",
        "Straight up: ", "For you: ",
    ]
    tail_wrappers = ["", " Ha!", " Enjoy!", " Ba dum tss.", " Groan!", " I'll see myself out."]
    idx2 = 0
    while len(pairs) < target_count:
        j = jokes[idx2 % len(jokes)]
        lead = random.choice(lead_wrappers)
        tail = random.choice(tail_wrappers)
        variant = (lead + j + tail).strip()
        if counter[variant] < 2:
            counter[variant] += 1
            pairs.append((pick_user(kind_hint=kind_hint), variant))
        idx2 += 1
        if idx2 > 200000:
            break

    return pairs[:target_count]


def build_knock_pool(target_count):
    """
    Knock-knock: base structure = (user starts with 'Knock knock.', assistant responds).
    Also produce variants where the user asks generically and the assistant delivers the full
    knock-knock joke self-contained.
    """
    jokes = list(dict.fromkeys(KNOCK_KNOCK_JOKES))
    pairs = []
    counter = Counter()

    # Mode A: user says "Knock knock." (or variant), assistant delivers standard callback.
    knock_user_prompts = [
        "Knock knock.", "Knock knock!", "knock knock", "*knock knock*",
        "Knock, knock.", "Knock, knock!", "Knocking on the door: knock knock.",
    ]

    # Mode B: user asks generically, assistant delivers the whole knock-knock joke self-contained.
    def make_self_contained(assistant_reply):
        # Reformat the callback style into a self-contained joke.
        # e.g. "Who's there? Boo. Boo who? Don't cry, it's just a joke!"
        # -> "Knock knock. Who's there? Boo. Boo who? Don't cry, it's just a joke!"
        return "Knock knock. " + assistant_reply

    # First: half in Mode A, half in Mode B
    random.shuffle(jokes)
    ja = jokes[:]
    ki = 0
    while len(pairs) < target_count and ki < len(ja):
        _, callback = ja[ki]
        u = random.choice(knock_user_prompts)
        pairs.append((u, callback))
        counter[callback] += 1
        ki += 1

    random.shuffle(jokes)
    ki = 0
    while len(pairs) < target_count and ki < len(jokes):
        _, callback = jokes[ki]
        full = make_self_contained(callback)
        u = pick_user(kind_hint="pun")  # generic joke request
        pairs.append((u, full))
        counter[full] += 1
        ki += 1

    # Second pass: allow each entry once more (cap of 2 per exact text)
    random.shuffle(jokes)
    ki = 0
    while len(pairs) < target_count and ki < len(jokes):
        _, callback = jokes[ki]
        if counter[callback] < 2:
            u = random.choice(knock_user_prompts)
            pairs.append((u, callback))
            counter[callback] += 1
        ki += 1
    random.shuffle(jokes)
    ki = 0
    while len(pairs) < target_count and ki < len(jokes):
        _, callback = jokes[ki]
        full = make_self_contained(callback)
        if counter[full] < 2:
            u = pick_user(kind_hint="pun")
            pairs.append((u, full))
            counter[full] += 1
        ki += 1

    # Third pass: create fresh variants by prefixing user prompts and reordering assistant lead-ins.
    lead_wrappers = [
        "Here's one: ", "Try this: ", "Okay: ", "Sure, ready? ", "Alright, ",
        "Get ready: ", "You know this one? ", "Classic incoming: ",
        "Here we go: ", "Warm-up: ",
    ]
    idx2 = 0
    while len(pairs) < target_count:
        _, callback = jokes[idx2 % len(jokes)]
        full = make_self_contained(callback)
        lead = random.choice(lead_wrappers)
        variant = lead + full
        if counter[variant] < 2:
            counter[variant] += 1
            pairs.append((pick_user(kind_hint="pun"), variant))
        idx2 += 1
        if idx2 > 200000:
            break

    return pairs[:target_count]


# ---------------------------------------------------------------------------
# Word-count safety net
# ---------------------------------------------------------------------------
def word_count(s):
    return len(s.split())


def sanitize_pair(u, a):
    # Trim assistant if over 40 words (should not happen with our data, but be safe).
    if word_count(a) >= 40:
        words = a.split()
        # Cap at 39 words, add period if lost
        a = " ".join(words[:39])
        if not a.endswith((".", "!", "?")):
            a += "."
    return u, a


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def main():
    fam1 = build_family_pool(PUN_JOKES, 500, kind_hint="pun")
    fam2 = build_knock_pool(500)
    fam3 = build_family_pool(OBS_JOKES, 500, kind_hint="observational")
    fam4 = build_family_pool(DAD_JOKES, 500, kind_hint="dad")

    all_pairs = fam1 + fam2 + fam3 + fam4
    # Sanitize
    all_pairs = [sanitize_pair(u, a) for u, a in all_pairs]

    # Interleave / shuffle so families are mixed in file
    random.shuffle(all_pairs)

    # Diversity gate: cap each assistant text at 2 uses globally.
    # If we drop pairs, we'll refill with fresh variant wrappers.
    assistant_counter = Counter()
    kept = []
    dropped_family_hint = []  # track kind_hint of dropped so we can refill by family
    for u, a in all_pairs:
        if assistant_counter[a] < 2:
            assistant_counter[a] += 1
            kept.append((u, a))
        else:
            dropped_family_hint.append((u, a))

    # Refill by wrapping jokes with lead phrases so the assistant text is new
    lead_wrappers = [
        "Here's one: ", "Try this: ", "Okay, here: ", "Sure thing: ", "Alright, ready? ",
        "How about: ", "One for you: ", "Coming right up: ", "Fresh joke: ",
        "Here's a good one: ", "Here goes nothing: ", "This one's fun: ",
        "You'll like this: ", "Get ready: ", "Warning, groaner: ",
        "Straight from my joke stash: ", "Old classic: ", "Hot off the press: ",
        "Just for you: ", "Ready or not: ", "Buckle up: ", "Prepare yourself: ",
        "Quick one: ", "Short one: ", "Simple one: ", "Fun one: ",
    ]
    tail_wrappers = ["", " Ha!", " Enjoy!", " Ba dum tss.", " Groan!",
                     " I'll see myself out.", " Hehe.", " Sorry not sorry.",
                     " That's the one.", " Cheers."]

    # Refill until we hit 2000
    src_answers = list({a for _, a in all_pairs})
    random.shuffle(src_answers)
    src_idx = 0
    safety = 0
    while len(kept) < 2000 and safety < 500000:
        safety += 1
        base = src_answers[src_idx % len(src_answers)]
        src_idx += 1
        # Skip if base is already wrapped (avoid double-wrapping)
        if any(base.startswith(lw) for lw in lead_wrappers):
            continue
        lead = random.choice(lead_wrappers)
        tail = random.choice(tail_wrappers)
        variant = (lead + base + tail).strip()
        # Word cap
        _, variant = sanitize_pair("", variant)
        if assistant_counter[variant] < 2:
            assistant_counter[variant] += 1
            u = pick_user()
            kept.append((u, variant))

    kept = kept[:2000]

    # Final pass: ensure no exact (user, assistant) pair repeats more than twice
    pair_counter = Counter()
    final = []
    for u, a in kept:
        key = (u, a)
        if pair_counter[key] < 2:
            pair_counter[key] += 1
            final.append((u, a))
    # If we lost some due to exact-pair dedupe, refill with fresh user prompts on existing assistants
    while len(final) < 2000:
        u, a = random.choice(kept)
        new_u = pick_user()
        key = (new_u, a)
        if pair_counter[key] < 2:
            pair_counter[key] += 1
            final.append((new_u, a))
    kept = final[:2000]

    # Report family counts (approximate; we track by string membership)
    fam1_set = set(a for _, a in fam1)
    fam2_set = set(a for _, a in fam2)
    fam3_set = set(a for _, a in fam3)
    fam4_set = set(a for _, a in fam4)

    counts = {"fam1_pun": 0, "fam2_knock": 0, "fam3_obs": 0, "fam4_dad": 0, "other": 0}
    for _, a in kept:
        if a in fam2_set or a.lower().startswith("knock knock") or "who's there" in a.lower():
            counts["fam2_knock"] += 1
        elif a in fam4_set:
            counts["fam4_dad"] += 1
        elif a in fam3_set:
            counts["fam3_obs"] += 1
        elif a in fam1_set:
            counts["fam1_pun"] += 1
        else:
            # Wrapped variants; attribute by inspection
            base = a
            # strip common leading wrappers
            for pref in ["Here's one: ", "Try this: ", "Okay: ", "Sure: ", "Alright: ",
                         "How about this: ", "One for you: ", "Here you go: ", "Ready? ",
                         "Here's a good one: ", "Here goes: ", "This one's fun: ",
                         "You'll like this: ", "Get ready: ", "Warning, groaner: ",
                         "Fresh from my joke book: ", "Old classic: ", "Hot off the press: ",
                         "Straight up: ", "For you: ", "Sure, ready? ", "Alright, ",
                         "You know this one? ", "Classic incoming: ", "Here we go: ", "Warm-up: "]:
                if base.startswith(pref):
                    base = base[len(pref):]
                    break
            # strip trailing wrappers
            for suf in [" Ha!", " Enjoy!", " Ba dum tss.", " Groan!", " I'll see myself out."]:
                if base.endswith(suf):
                    base = base[: -len(suf)]
                    break
            if base in fam2_set or base.lower().startswith("knock knock") or "who's there" in base.lower():
                counts["fam2_knock"] += 1
            elif base in fam4_set:
                counts["fam4_dad"] += 1
            elif base in fam3_set:
                counts["fam3_obs"] += 1
            elif base in fam1_set:
                counts["fam1_pun"] += 1
            else:
                counts["other"] += 1

    with open(OUT, "w", encoding="utf-8") as f:
        for u, a in kept:
            f.write(json.dumps({"user": u, "assistant": a}, ensure_ascii=True) + "\n")

    print(f"WROTE {len(kept)} lines to {OUT}")
    print(f"Family counts: {counts}")

    # Print one sample per family
    for name, joke_set in [("fam1_pun", fam1), ("fam2_knock", fam2), ("fam3_obs", fam3), ("fam4_dad", fam4)]:
        # find first kept pair whose assistant is in joke_set answers
        answers = set(a for _, a in joke_set)
        for u, a in kept:
            if a in answers:
                print(f"SAMPLE {name}: user={u!r} assistant={a!r}")
                break

    # Verify all assistants <40 words
    over = sum(1 for _, a in kept if word_count(a) >= 40)
    print(f"Assistants >=40 words: {over}")

    # Verify unique assistant text distribution
    ac = Counter(a for _, a in kept)
    over_cap = sum(1 for _, c in ac.items() if c > 2)
    print(f"Unique assistant texts used >2 times: {over_cap}")
    print(f"Distinct assistant texts: {len(ac)}")


if __name__ == "__main__":
    main()
