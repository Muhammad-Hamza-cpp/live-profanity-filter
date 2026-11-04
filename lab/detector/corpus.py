r"""Sentences for the TTS training data, each word tagged with its curse group (or None).

Three kinds, mixed:
  - curse sentences: a curse form in a natural slot, or dropped into a clean carrier sentence
  - negative sentences: a sound-alike / minced oath in the same kinds of slots (must stay audible)
  - clean sentences: carriers as they are

    .venv-lab\Scripts\python lab\detector\corpus.py      # print a sample
"""
import random
import re
import sys
from pathlib import Path

from lexicon import GROUPS, NEGATIVES

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from censor import is_profane, load_words  # noqa: E402

# Everyday speech in the registers the filter will hear: streams, games, videos, podcasts, chat.
CARRIERS = """\
Alright guys, welcome back to the channel.
Let's get into today's video.
I can't believe how long this update took.
Okay, give me one second to check the settings.
He was hiding behind the wall the whole time.
We need to push the objective right now.
Thanks for watching, and I'll see you tomorrow.
Make sure you grab some health before the next fight.
That was honestly the best round I've played all week.
The new map looks really different from the old one.
I think the server is lagging again.
Can you hear me okay, or is my mic too quiet?
So today we're making a really simple pasta.
Add a little bit of salt and let it boil.
The traffic this morning was absolutely terrible.
I left my phone in the car again.
My boss wants the report by Friday.
We're going to the beach this weekend.
Did you see the game last night?
The referee made a really strange call.
That movie was way longer than it needed to be.
I've been waiting for this package for two weeks.
The weather here has been really weird lately.
Let me know what you think in the comments.
Don't forget to like and subscribe.
We have three minutes left on the clock.
He's got a sniper on the roof, watch out.
I'm going to reload, cover me.
Okay, that's actually pretty cool.
I really didn't expect that to work.
The kids are finally asleep, so let's talk.
Honestly, I have no idea what I'm doing here.
Can somebody explain what just happened?
We're going to need a bigger boat.
My computer keeps crashing every time I open this.
I spent the whole afternoon fixing the car.
Turn left at the next light.
The coffee machine at work is broken again.
I told him three times already.
Why does this always happen to me?
She said she'd call back in an hour.
Look at the size of that thing.
We should have left an hour ago.
I swear this level is impossible.
Get back here, you coward.
Okay, okay, I'm calm, I'm calm.
That guy has been camping in that corner all game.
Run, run, run, they're right behind us.
Look out, there's another one on the left.
I can't feel my hands, it's so cold.
This is the third time today the internet went down.
We lost the match by one point.
He just walked right past me.
Let's try that one more time.
I need a break, I've been playing for six hours.
You know what, forget it.
Put the box down over there.
The battery is almost dead.
Who left the door open?
I'm not doing that again.
Come on, come on, come on.
That's exactly what I was talking about.
Wait, what did he just say?
You have got to be kidding me.
I paid twenty dollars for this.
Stop touching my stuff.
Why is it so loud in here?
That's not how any of this works.
He drove straight into the wall.
I forgot to save my progress.
The whole team just left the game.
They charged me twice for the same thing.
I've been on hold for forty minutes.
The printer is jammed again.
My neighbor's dog won't stop barking.
I stepped on a piece of glass.
The landlord still hasn't fixed the heating.
Somebody ate my lunch from the fridge.
We missed the last train home.
I locked myself out of the house.
The meeting could have been an email.
He took my parking spot again.
It's raining and I forgot my umbrella.
This chair is really uncomfortable.
I burned the toast again.
The cat knocked my drink off the table.
You never listen to a word I say.
We're running out of time here.
Can you turn the music down a little?
I think we took a wrong turn somewhere.
There's no way that just happened.
Let me tell you what happened at work today.
The doctor said I need more sleep.
My knee has been hurting all week.
We got stuck in the elevator for an hour.
I can't find my keys anywhere.
Does anyone know how to fix this?
Just give me a minute to think.
The ending of that show made no sense.
I bought the wrong size again.
Keep your voice down, the baby is sleeping.
They cancelled the flight at the last minute.
The food at that place was cold.
I studied all night and still failed the test.
Everyone stop talking for a second.
Who keeps moving my charger?
I'm so tired of this.
Let's just go home.
Pass me the remote, please.
That was a really close call.
He scored in the last ten seconds.
We should get pizza tonight.
My brother borrowed my car without asking.
This is going to take forever.
I need to call my mom back.
The water bill was huge this month.
Grab the ammo and follow me.
They're flanking us from the right side.
Heal me, heal me, I'm almost dead.
Don't shoot, it's me.
We're winning this round for sure.
Open the gate, we're coming through.
I've got eyes on two of them.
Fall back, fall back.
It's a trap, don't go in there.
""".splitlines()

# A curse word in a natural slot. {w} takes any form; most are fine grammatically or close enough
# that the voices read them naturally.
CURSE_TEMPLATES = """\
{w}!
Oh, {w}.
Oh {w}, I forgot.
{w}, that hurt.
What the {w} was that?
Are you {w} kidding me?
That's {w} ridiculous.
He's such a {w}.
Stop being a {w}.
You {w}.
Shut up, you {w}.
This {w} game is broken.
I hate this {w} level.
{w}, he got me again.
That was so {w}.
Well, {w}.
He literally said {w} on stream.
And then she just yelled {w} at me.
No, no, no, {w}!
{w}, {w}, {w}!
Oh my god, {w}.
Get out of my way, {w}.
I swear to god, {w}.
That {w} took my spot.
Look at this {w}.
What a {w}.
Holy {w}.
This is {w}.
Who the {w} does that?
Move, {w}!
I'm so {w} tired.
Did you just call me a {w}?
Yeah, {w} that.
{w} this traffic.
Oh, come on, {w}!
""".splitlines()

NEGATIVE_TEMPLATES = """\
{w}!
Oh, {w}.
I said {w}, not the other thing.
Did you hear him say {w}?
The word was {w}.
That's {w}.
Look at the {w}.
{w}, I forgot.
He kept talking about the {w}.
Oh {w}, that's annoying.
What the {w} was that?
It's a {w}, obviously.
You know, {w}.
I like the {w}.
Stop saying {w}.
Write down {w}.
""".splitlines()

PHRASE_TEMPLATES = """\
{w}!
Oh, {w}.
{w}, man.
He just said {w}.
And I was like, {w}.
{w}, seriously.
Okay, {w}.
""".splitlines()


# Clean words that fit the curse slots ("such a ___", "What the ___", "This ___ game").
FILLERS = """\
heck idiot genius legend noob clown moron nerd loser camper hacker cheater tryhard rat snake dog
pig jerk creep weirdo liar coward fool dummy baby child monster beast machine hero joke mess
disaster nightmare great weird stupid awesome crazy amazing terrible annoying random new old big
little dumb lucky silly freaking flipping darn dang goodness world deal point problem thing noise
guy dude buddy man kid player bot sniper hell-bent heckin cute nice cool sick wild boring broken
cheap slow fast ugly beautiful funny sad bad good happy angry tired hungry gosh golly jeez wow
whoa yikes ouch hey yes no okay sure right wait stop please thanks sorry seriously honestly
literally actually basically totally really very pretty super mega ultra
""".split()
FILLERS = [w for w in FILLERS if w != "hell-bent"]
CLEAN_PHRASES = [
    "what the heck", "who the heck", "oh my gosh", "son of a gun", "holy cow", "holy moly",
    "shut the door", "get the ball", "what the hey", "for goodness sake", "oh my goodness",
    "what a day", "come on man", "no way dude", "you little rascal", "get out of here",
    "for crying out loud", "holy smokes", "good grief", "give me a break", "piece of cake",
]


def _word_group():
    """Single curse word (normalised) -> its group, so a phrase tags each curse word correctly."""
    out = {}
    for group, (_, forms) in GROUPS.items():
        for form in forms:
            if len(form.split()) == 1:
                out.setdefault(_norm(form), group)
    return out


def _norm(word):
    return re.sub(r"[^a-z']", "", word.lower()).strip("'")


_PATTERNS = load_words(ROOT / "words.txt")
_WORD_GROUP = _word_group()
# Common words are heard far more often, so they get more examples (rest: 1).
COMMON = {"fuck": 4, "shit": 4, "ass": 3, "bitch": 3, "damn": 3, "hell": 3,
          "dick": 2, "crap": 2, "piss": 2, "bastard": 2}
_GROUP_NAMES = list(GROUPS)
_GROUP_WEIGHTS = [COMMON.get(g, 1) for g in _GROUP_NAMES]


def _tag_form(form, group):
    """Tag the words of one curse form. In a phrase only the curse words are tagged
    ('what the fuck' -> fuck), each with its own group ('fucking hell' -> fuck, hell)."""
    tags = []
    for w in form.split():
        if not is_profane(w, _PATTERNS):
            tags.append(None)
        else:
            tags.append(_WORD_GROUP.get(_norm(w), group))
    return tags


def _fill(template, form, tags):
    """Put the form into every {w} slot. Punctuation glued to a slot stays on its last word."""
    words, out_tags = [], []
    for tok in template.split():
        if "{w}" in tok:
            pre, post = tok.split("{w}", 1)
            fw = form.split()
            fw[0], fw[-1] = pre + fw[0], fw[-1] + post.replace("{w}", "")
            words += fw
            out_tags += tags
        else:
            words.append(tok)
            out_tags.append(None)
    return list(zip(words, out_tags))


def _insert(carrier, form, tags, rng):
    """Drop a form into a clean carrier sentence at a word boundary."""
    words = carrier.split()
    i = rng.randrange(len(words) + 1)
    fw = form.split()
    if i == 0:
        fw[0] = fw[0][:1].upper() + fw[0][1:]
        words[0] = words[0][:1].lower() + words[0][1:] if words[0] != "I" and not words[0].startswith("I'") else words[0]
    out = words[:i] + fw + words[i:]
    return list(zip(out, [None] * i + tags + [None] * (len(words) - i)))


def make(rng, kind=None):
    """One sentence: [(word, group or None)]."""
    kind = kind or rng.choices(["curse", "context", "negative", "clean"], weights=[0.4, 0.2, 0.2, 0.2])[0]
    if kind == "clean":
        return [(w, None) for w in rng.choice(CARRIERS).split()]
    if kind == "context":
        # The curse templates with a clean word in the slot, so "What the ...", "such a ..."
        # predict nothing by themselves: the detector has to hear the word.
        if rng.random() < 0.25:
            form = rng.choice(CLEAN_PHRASES)
            return _fill(rng.choice(PHRASE_TEMPLATES), form, [None] * len(form.split()))
        form = rng.choice(FILLERS if rng.random() < 0.7 else NEGATIVES)
        return _fill(rng.choice(CURSE_TEMPLATES), form, [None] * len(form.split()))
    if kind == "negative":
        form = rng.choice(NEGATIVES)
        tags = [None] * len(form.split())
        if rng.random() < 0.6:
            return _fill(rng.choice(NEGATIVE_TEMPLATES), form, tags)
        return _insert(rng.choice(CARRIERS), form, tags, rng)
    group = rng.choices(_GROUP_NAMES, weights=_GROUP_WEIGHTS)[0]
    form = rng.choice(GROUPS[group][1])
    tags = _tag_form(form, group)
    r = rng.random()
    if len(tags) > 1 and r < 0.5:
        return _fill(rng.choice(PHRASE_TEMPLATES), form, tags)
    if r < 0.6:
        return _fill(rng.choice(CURSE_TEMPLATES), form, tags)
    return _insert(rng.choice(CARRIERS), form, tags, rng)


if __name__ == "__main__":
    rng = random.Random(0)
    for _ in range(30):
        print(" ".join(f"[{w}|{g}]" if g else w for w, g in make(rng)))
