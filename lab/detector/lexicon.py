r"""Training vocabulary for the mute detector.

GROUPS: what the detector learns to mute. Each group is one output of the detector head, so the
app can switch groups on and off without retraining. Forms are written the way a transcript spells
them (and the way the TTS voices are given them); multi-word phrases are fine, only the curse word
inside is labelled.

NEGATIVES: words that sound like a curse word, or stand in for one, and must stay audible. The
detector learns where the boundary is from these as much as from the curse words themselves.

    .venv-lab\Scripts\python lab\detector\lexicon.py     # consistency check against words.txt
"""

# group: (category, forms)
GROUPS = {
    # --- strong ---
    "fuck": ("strong", [
        "fuck", "fucks", "fucked", "fucker", "fuckers", "fucking", "fuckin", "fuckup", "fuckups",
        "fuckface", "fuckhead", "fuckwit", "fucktard", "fuckboy", "fuckery", "fuckable", "fuckoff",
        "motherfucker", "motherfuckers", "motherfucking", "motherfuckin", "clusterfuck", "mindfuck",
        "dumbfuck", "unfuckingbelievable", "absofuckinglutely", "fuckwad", "fuckton", "fuckall",
        "what the fuck", "fuck you", "fuck off", "fuck this", "fucking hell", "for fuck's sake",
        "shut the fuck up", "get the fuck out", "who the fuck", "fuck me", "fuck yeah", "fuck it",
    ]),
    "shit": ("strong", [
        "shit", "shits", "shitty", "shitter", "shitting", "shitted", "shitless", "shite",
        "bullshit", "bullshitting", "bullshitter", "horseshit", "chickenshit", "dipshit", "apeshit",
        "batshit", "dogshit", "jackshit", "shithead", "shitheads", "shithole", "shitholes", "shitshow",
        "shitload", "shitstorm", "shitface", "shitfaced", "shitbag", "shitpost", "shitposting",
        "holy shit", "oh shit", "no shit", "piece of shit", "shit happens", "eat shit", "shitty day",
    ]),
    "bitch": ("strong", [
        "bitch", "bitches", "bitching", "bitchy", "bitched", "bitchin", "bitchass", "sonofabitch",
        "son of a bitch", "sons of bitches", "sumbitch", "life's a bitch", "you little bitch",
    ]),
    "ass": ("strong", [
        "ass", "asses", "asshole", "assholes", "dumbass", "dumbasses", "jackass", "jackasses",
        "smartass", "badass", "badasses", "asshat", "asswipe", "kickass", "fatass", "lardass",
        "hardass", "halfassed", "half-assed", "kissass", "candyass", "assclown", "asshattery",
        "pain in the ass", "kick your ass", "my ass", "get your ass over here", "cover your ass",
        "arse", "arses", "arsehole", "arseholes",
    ]),
    "dick": ("strong", [
        "dick", "dicks", "dickhead", "dickheads", "dickish", "dickwad", "dickface", "dickbag",
        "don't be a dick", "what a dick",
    ]),
    "cock": ("strong", [
        "cock", "cocks", "cocksucker", "cocksuckers", "cocksucking", "cockhead",
    ]),
    "cunt": ("strong", ["cunt", "cunts", "cunty"]),
    "prick": ("strong", ["prick", "pricks", "what a prick"]),
    "piss": ("strong", [
        "piss", "pissed", "pissing", "pisses", "pissy", "pisser", "pisshead", "piss off",
        "pissed off", "taking the piss",
    ]),
    "bastard": ("strong", ["bastard", "bastards", "you lucky bastard"]),
    "pussy": ("strong", ["pussy", "pussies"]),
    "tits": ("strong", ["tits", "titties", "titty", "tit"]),
    "slut": ("strong", ["slut", "sluts", "slutty"]),
    "whore": ("strong", ["whore", "whores", "whorish", "manwhore", "attention whore"]),
    "jizz": ("strong", ["jizz", "jizzed"]),
    "dildo": ("strong", ["dildo", "dildos"]),
    "skank": ("strong", ["skank", "skanks", "skanky"]),
    "douche": ("strong", ["douche", "douches", "douchebag", "douchebags", "douchey"]),

    # --- British / Commonwealth ---
    "twat": ("british", ["twat", "twats"]),
    "wank": ("british", ["wank", "wanker", "wankers", "wanking", "wanky"]),
    "bollocks": ("british", ["bollock", "bollocks", "bollocking", "dog's bollocks"]),
    "bugger": ("british", ["bugger", "buggered", "buggery", "buggering", "bugger off"]),
    "tosser": ("british", ["tosser", "tossers"]),
    "bloody": ("british", ["bloody", "bloody hell", "bloody idiot"]),
    "sod": ("british", ["sod off", "sod it", "sodding"]),

    # --- slurs ---
    "nigger": ("slur", ["nigger", "niggers", "nigga", "niggas"]),
    "faggot": ("slur", ["faggot", "faggots", "fag", "fags"]),
    "retard": ("slur", ["retard", "retards", "retarded"]),
    "tranny": ("slur", ["tranny", "trannies"]),
    "spic": ("slur", ["spic", "spics"]),
    "chink": ("slur", ["chink", "chinks"]),
    "kike": ("slur", ["kike", "kikes"]),
    "wetback": ("slur", ["wetback", "wetbacks"]),
    "gook": ("slur", ["gook", "gooks"]),
    "dyke": ("slur", ["dyke", "dykes"]),
    "beaner": ("slur", ["beaner", "beaners"]),
    "towelhead": ("slur", ["towelhead", "towelheads", "raghead", "ragheads"]),
    "paki": ("slur", ["paki", "pakis"]),

    # --- mild ---
    "damn": ("mild", [
        "damn", "damned", "damnit", "dammit", "goddamn", "goddamned", "goddammit", "goddamnit",
        "damn it", "god damn it", "damn right",
    ]),
    "hell": ("mild", [
        "hell", "hellhole", "what the hell", "go to hell", "hell no", "hell yeah", "the hell",
        "like hell", "who the hell",
    ]),
    "crap": ("mild", ["crap", "crappy", "crapper", "crapload", "crapped", "craps", "oh crap"]),
}

# Must stay audible. Sound-alikes first, then minced oaths people use *instead* of swearing.
NEGATIVES = [
    # fuck
    "duck", "ducking", "truck", "luck", "lucky", "buck", "muck", "puck", "tuck", "stuck", "struck",
    "fudge", "fun", "funny", "fund", "funk", "fuss", "folk", "fork", "frock", "flock", "Fokker",
    "Fukuoka", "fluke", "mother", "motherhood", "fuchsia",
    # shit
    "shift", "shifting", "sheet", "sheets", "shut", "shot", "shoot", "shirt", "ship", "sit", "spit",
    "shiitake", "Schitt's", "shed", "shiver", "split", "chit",
    # bitch
    "beach", "beaches", "pitch", "pitched", "witch", "ditch", "rich", "itch", "bench", "britches",
    "Bitcoin", "bit", "bitter",
    # ass
    "as", "has", "assume", "assess", "assets", "assist", "assassin", "class", "classes", "pass",
    "passing", "glass", "grass", "mass", "bass", "brass", "sass", "sassy", "embassy", "harass",
    "massive", "Cass", "lass", "molasses", "Assam", "ask", "cassette", "passive",
    # arse
    "arson", "Arsenal", "parse", "sparse", "farce", "erase",
    # dick
    "dictionary", "dictate", "Dickens", "dig", "thick", "tick", "kick", "sick", "pick", "Rick",
    "Nick", "deck", "dock", "duck",
    # cock
    "cockpit", "cocktail", "cockroach", "peacock", "Hancock", "Hitchcock", "cockatoo", "rock",
    "sock", "clock", "lock", "cog", "cork", "cook", "cockney",
    # cunt
    "count", "country", "can't", "hunt", "punt", "front", "Kant", "cant", "Canute", "cut",
    # prick
    "brick", "trick", "price", "pickle", "prickly", "prick up", "tricky",
    # piss
    "kiss", "miss", "hiss", "piece", "peace", "pizza", "Pisa", "pistachio", "piston", "this",
    "kissed", "missed",
    # bastard
    "mustard", "custard", "dastardly", "bastion", "Bastille", "busted",
    # pussy
    "pushy", "pussycat", "puss in boots", "Pusey", "push", "fussy", "hussy",
    # tits
    "title", "titles", "titan", "titanic", "tip", "tight", "tilt", "twit", "teeth", "titmouse",
    "blue tit", "tittle", "kits", "bits",
    # slut, whore, skank, jizz, dildo, douche
    "slot", "slate", "smut", "slit", "slight", "horror", "hoarse", "horse", "war", "more", "four",
    "horde", "hoard", "her", "skunk", "stank", "shank", "skate", "jazz", "fizz", "jism", "Bilbo",
    "dill", "touche", "touché", "dutch", "douse", "Dusseldorf",
    # British
    "twin", "twist", "what", "twit", "bank", "tank", "thank", "wink", "Wonka", "Yankee", "swank",
    "bullocks", "ballots", "Pollock", "burger", "bigger", "booger", "butter", "bug", "toss",
    "tossed", "dossier", "bloodied", "blood", "muddy", "sob", "sawed", "sodium", "sad", "sought",
    # slurs
    "Niger", "Nigeria", "niggle", "niggling", "trigger", "digger", "snicker", "figure", "nigh",
    "flag", "fig", "fang", "fact", "regard", "tardy", "Tran", "trainee", "spick and span", "spike",
    "spin", "chunk", "chin", "kink", "Mike", "like", "kite", "back", "wet", "book", "look", "gecko",
    "dike", "Dyck", "Nike", "beater", "bean", "towel", "rag", "Pakistan", "Pakistani", "pack",
    "packing",
    # mild
    "dam", "dams", "Amsterdam", "damp", "Dan", "ma'am", "Damon", "damage", "condemn",
    "hello", "shell", "held", "help", "well", "Helen", "hail", "hall", "heal", "hello there",
    "Michelle", "Hellenic", "Hal", "clap", "trap", "wrap", "scrap", "crab", "crop", "crepe",
    "cramp", "crack", "crackle", "Kraft",
    # minced oaths / euphemisms (deliberately clean)
    "freaking", "frigging", "fricking", "frick", "frack", "effing", "eff", "fudge", "shoot",
    "shucks", "sugar", "darn", "darned", "dang", "dagnabbit", "heck", "gosh", "golly", "jeez",
    "crud", "crikey", "blimey", "bloomin", "rats", "son of a gun", "son of a biscuit",
    "what the heck", "holy cow", "oh snap", "flipping", "fiddlesticks", "poop", "butt", "jerk",
    "idiot", "stupid", "dummy", "screw you", "sucks", "suck",
]
NEGATIVES = list(dict.fromkeys(NEGATIVES))
# Same sound and spelling as a curse word but clean in context ("chink in the armor", "bloody
# nose", "fire retardant") would contradict the curse labels, so they are left out for now.

CATEGORIES = sorted({c for c, _ in GROUPS.values()})


def all_forms():
    """[(form, group)] for every curse form."""
    return [(f, g) for g, (_, forms) in GROUPS.items() for f in forms]


def check():
    """Every curse form must be caught by words.txt, and no negative may be (unless it really is
    ambiguous), so the ASR modes, the bench scoring and the detector all agree on what a curse is."""
    import re
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from censor import is_profane, load_words
    patterns = load_words(root / "words.txt")
    problems = 0
    for form, group in all_forms():
        if not any(is_profane(w, patterns) for w in form.split()):
            print(f"not in words.txt: {form!r} ({group})")
            problems += 1
    # A phrase containing a curse word on purpose ("chink in the armor") is a known ambiguity:
    # same sound, different meaning. Only flag negatives whose words all look clean on their own.
    for neg in NEGATIVES:
        hits = [w for w in re.split(r"[\s-]+", neg) if is_profane(w, patterns)]
        if hits and len(neg.split()) == 1:
            print(f"negative matches words.txt: {neg!r}")
            problems += 1
    forms = all_forms()
    print(f"{len(GROUPS)} groups, {len(forms)} curse forms, {len(NEGATIVES)} negatives, "
          f"categories {CATEGORIES} -> {problems} problem(s)")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
