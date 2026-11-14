"""Light text helpers for the data analyses: tokens, question type, lexical overlap, negation,
and a rule-based guess of how an unanswerable question was made from its source sentence."""
import re

QTYPES = ["what", "who", "when", "where", "which", "how many", "how much", "how", "why", "other"]
NEGATION = {"not", "no", "never", "none", "nor", "without", "neither", "cannot"}
STOP = set("""a an the of in on at to for from by with and or is are was were be been being do does did
what who whom whose when where which why how many much this that these those it its as into than then
there their they he she his her i you we our your has have had can could would should will may might
s about after before during between over under up down out""".split())
# Antonym pairs typical of SQuAD 2.0 rewrites (both directions are checked)
ANTONYMS = [("first", "last"), ("largest", "smallest"), ("biggest", "smallest"), ("most", "least"),
            ("more", "less"), ("more", "fewer"), ("highest", "lowest"), ("high", "low"), ("best", "worst"),
            ("earliest", "latest"), ("early", "late"), ("before", "after"), ("increase", "decrease"),
            ("increased", "decreased"), ("rise", "fall"), ("gain", "lose"), ("win", "lose"), ("won", "lost"),
            ("start", "end"), ("began", "ended"), ("open", "close"), ("north", "south"), ("east", "west"),
            ("northern", "southern"), ("eastern", "western"), ("maximum", "minimum"), ("top", "bottom"),
            ("major", "minor"), ("strong", "weak"), ("old", "new"), ("oldest", "newest"), ("long", "short"),
            ("longest", "shortest"), ("inside", "outside"), ("upper", "lower"), ("positive", "negative"),
            ("accept", "reject"), ("allowed", "banned"), ("legal", "illegal"), ("public", "private")]
_ANT = {}
for a, b in ANTONYMS:
    _ANT.setdefault(a, set()).add(b)
    _ANT.setdefault(b, set()).add(a)


def words(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower())


def content_words(s):
    return [w for w in words(s) if w not in STOP]


def jaccard(a, b):
    """Jaccard similarity of the content-word sets of two texts."""
    a, b = set(content_words(a)), set(content_words(b))
    return len(a & b) / len(a | b) if a | b else 0.0


def qtype(q):
    w = words(q)
    text = " ".join(w)
    for t in ("how many", "how much"):
        if t in text:
            return t
    for t in ("what", "who", "when", "where", "which", "why", "how"):
        if t in w or (t == "who" and ("whom" in w or "whose" in w)):
            return t
    return "other"


def overlap(q, ctx):
    """Share of the question's content words that appear in the context."""
    qw = content_words(q)
    cw = set(words(ctx))
    return sum(w in cw for w in qw) / len(qw) if qw else 0.0


def negations(s):
    """Negation tokens of s; contractions such as "didn't" count as "not"."""
    out = set()
    for w in words(s):
        if w in NEGATION:
            out.add(w)
        elif w.endswith("n't"):
            out.add("not")
    return out


def negated(q):
    return bool(negations(q))


def _entities(s):
    """Numbers and capitalised words that are not sentence-initial (a cheap proper-noun proxy)."""
    toks = re.findall(r"[A-Za-z0-9][\w'-]*", s)
    return {t.lower() for i, t in enumerate(toks) if t[0].isdigit() or (i > 0 and t[0].isupper())}


def unanswerable_type(question, source_sentence):
    """Guess how an unanswerable question was derived from the sentence holding its plausible answer:
    NEG (negation added/removed), ANT (antonym swapped in), ENT (an entity / number of the question is
    not in the sentence), else OTHER (impossible condition, mutual exclusion, other)."""
    qw, sw = set(words(question)), set(words(source_sentence))
    if negations(question) != negations(source_sentence):
        return "NEG"
    if any(w not in sw and _ANT.get(w, set()) & sw for w in qw):
        return "ANT"
    if _entities(question) - set(words(source_sentence)):
        return "ENT"
    return "OTHER"
