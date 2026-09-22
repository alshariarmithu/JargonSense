"""Stage 3 -- preprocessing: sentence segmentation, stemming, lemmatization.

Run a demonstration from the repository root:

    python -m src.preprocessing.normalize

--------------------------------------------------------------------------
Why this stage is a measurement and not a formality
--------------------------------------------------------------------------
Stemming and lemmatization collapse inflected forms onto one root:

    killed, kills, killing   ->   kill
    stopped, stops           ->   stop

Usually that helps.  A small corpus has few examples of each word, and merging
inflections gives the model denser evidence per feature.

For *this* project it might destroy the thing being studied.  "Killed the
process" is ordinary technical description; collapsing it into the emotional
register of "kill" hides exactly the confusion the explainability stage is
meant to expose.

So the project does not assume an answer.  ``normalize()`` offers all three
options behind one switch, and ``tools/run_ablation.py`` trains the same model
under each and reports the difference.  Whichever wins, the comparison is a
result rather than an assumption.

--------------------------------------------------------------------------
What the three modes do
--------------------------------------------------------------------------
============  ==========================================================
``"none"``    Tokenise only.  Words keep their inflections.
``"lemma"``   Dictionary root, POS-aware.  "stopped" -> "stop".
``"stem"``    Porter stemming.  Cruder, and can produce non-words
              ("finally" -> "final"), but merges more aggressively.
============  ==========================================================

All three run through **the same tokeniser**, so the only difference between
them is the word-normalisation step.  That is what makes the ablation a fair
comparison: nothing else changes.

--------------------------------------------------------------------------
Two things this module deliberately does not do
--------------------------------------------------------------------------
1. **No stopword removal, in any mode.**  Negation words ("not", "no",
   "never") appear in every standard stopword list, and negation flips
   sentiment.  Removing them would break the corpus in a way no amount of
   tuning could repair.

2. **Placeholders are never touched.**  ``clean()`` has already replaced URLs,
   code spans and emoticons with uppercase markers (``URL``, ``CODE``,
   ``EMO_POS``, ...).  Stemming those would fold them into ordinary words --
   ``CODE`` would become ``code`` and merge with the English noun -- and the
   TF-IDF vocabulary would be quietly corrupted.
"""

from __future__ import annotations

import json
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer

from src.extraction.clean_text import PLACEHOLDERS, clean, tokenize
from src.paths import ABLATION

MODES = ("none", "lemma", "stem")
STEMMERS = ("porter", "snowball")

# Written by tools/run_ablation.py; read by every later stage.
CHOSEN_MODE_FILE = ABLATION / "chosen_mode.json"

# NLTK ships its models as separate downloads.  Map "where nltk looks" ->
# "what to download if it is not there".
_NLTK_DATA = {
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/wordnet": "wordnet",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}

_PLACEHOLDER_SET = set(PLACEHOLDERS)


def ensure_nltk_data() -> None:
    """Download the NLTK models this module needs, once, if they are missing."""
    for lookup_path, package in _NLTK_DATA.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(package, quiet=True)


_PORTER = PorterStemmer()
_SNOWBALL = SnowballStemmer("english")
_LEMMATIZER = WordNetLemmatizer()


# --------------------------------------------------------------------------
# sentence segmentation
# --------------------------------------------------------------------------
def segment(text) -> list[str]:
    """Split a document into sentences.

    Used for the per-document sentence statistics reported in Stage 3.  It does
    **not** feed the models -- they consume whole documents -- so segmentation
    is a measurement tool here rather than a transformation.
    """
    if text is None or text != text:  # NaN is the only value unequal to itself
        return []
    stripped = str(text).strip()
    if not stripped:
        return []
    ensure_nltk_data()
    return nltk.sent_tokenize(stripped)


def sentence_stats(texts) -> dict:
    """Sentence-level statistics over a collection of documents, for the report."""
    counts = [len(segment(t)) for t in texts]
    counts = [c for c in counts if c > 0]
    if not counts:
        return {"documents": 0, "sentences": 0,
                "mean_sentences_per_document": 0.0, "max_sentences": 0}
    return {
        "documents": len(counts),
        "sentences": sum(counts),
        "mean_sentences_per_document": round(sum(counts) / len(counts), 2),
        "max_sentences": max(counts),
    }


# --------------------------------------------------------------------------
# word normalisation
# --------------------------------------------------------------------------
def _wordnet_pos(treebank_tag: str) -> str:
    """Translate an NLTK part-of-speech tag into the letter WordNet expects.

    This mapping is the difference between lemmatization working and silently
    doing nothing.  WordNetLemmatizer assumes every word is a noun unless told
    otherwise, so ``lemmatize("stopped")`` returns ``"stopped"`` while
    ``lemmatize("stopped", "v")`` returns ``"stop"`` -- and the verbs are
    exactly the words this project cares about.
    """
    if treebank_tag.startswith("J"):
        return "a"  # adjective
    if treebank_tag.startswith("V"):
        return "v"  # verb
    if treebank_tag.startswith("R"):
        return "r"  # adverb
    return "n"      # noun, and the fallback for everything else


@lru_cache(maxsize=100_000)
def _stem_word(word: str, stemmer: str) -> str:
    if stemmer == "snowball":
        return _SNOWBALL.stem(word)
    return _PORTER.stem(word)


@lru_cache(maxsize=100_000)
def _lemmatize_word(word: str, pos: str) -> str:
    return _LEMMATIZER.lemmatize(word, pos)


def normalize_tokens(text, mode: str = "none", stemmer: str = "porter") -> list[str]:
    """Tokenise one document and apply the chosen word-normalisation mode.

    Parameters
    ----------
    text
        A document, normally already passed through ``clean()``.  ``None`` and
        NaN are treated as empty, so this is safe to hand to ``Series.apply``.
    mode
        ``"none"``, ``"lemma"`` or ``"stem"``.
    stemmer
        ``"porter"`` (default) or ``"snowball"``.  Ignored unless
        ``mode="stem"``.

    Returns
    -------
    list[str]
        Tokens, with placeholders such as ``URL`` and ``CODE`` passed through
        untouched.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if stemmer not in STEMMERS:
        raise ValueError(f"stemmer must be one of {STEMMERS}, got {stemmer!r}")

    if text is None or text != text:
        return []

    tokens = tokenize(str(text))
    if mode == "none" or not tokens:
        return tokens

    if mode == "stem":
        return [
            token if token in _PLACEHOLDER_SET else _stem_word(token, stemmer)
            for token in tokens
        ]

    # mode == "lemma": tag the whole sentence at once, because a tagger needs
    # surrounding words to tell a noun from a verb.
    ensure_nltk_data()
    tagged = nltk.pos_tag(tokens)
    return [
        token if token in _PLACEHOLDER_SET
        else _lemmatize_word(token, _wordnet_pos(tag))
        for token, tag in tagged
    ]


@lru_cache(maxsize=50_000)
def _normalize_cached(text: str, mode: str, stemmer: str) -> str:
    return " ".join(normalize_tokens(text, mode=mode, stemmer=stemmer))


def normalize(text, mode: str = "none", stemmer: str = "porter") -> str:
    """Tokenise and word-normalise one document, returned as a string.

    This is the form sklearn wants.  Because every mode goes through the same
    tokeniser, and the tokeniser is the same ``TOKEN_PATTERN`` the vectoriser
    uses, joining the tokens back with spaces is lossless -- the vectoriser
    will split them into exactly the same list again.

    Usage in a pipeline::

        from functools import partial
        TfidfVectorizer(preprocessor=partial(normalize, mode="lemma"),
                        token_pattern=TOKEN_PATTERN)

    Results are cached per document.  Cross-validation hands the same document
    to this function once per fold and once per parameter combination, and
    lemmatization has to POS-tag the whole sentence every time -- roughly six
    seconds per pass over the training split.  Without the cache a modest grid
    search would spend most of its time re-tagging text it has already seen.
    """
    if text is None or text != text:
        return ""
    return _normalize_cached(str(text), mode, stemmer)


def compare_modes(text) -> dict[str, str]:
    """Return the same document under all three modes, for report tables."""
    return {mode: normalize(text, mode=mode) for mode in MODES}


# --------------------------------------------------------------------------
# Stage 2 + Stage 3, composed
# --------------------------------------------------------------------------
def prepare(text, mode: str = "none") -> str:
    """Clean and normalise one raw document -- the full text pipeline.

    This is the single entry point every model uses, so that a model trained
    in Stage 5 and a sentence typed into the demo in Stage 6 go through
    exactly the same transformations::

        from functools import partial
        TfidfVectorizer(preprocessor=partial(prepare, mode="lemma"),
                        token_pattern=TOKEN_PATTERN)
    """
    return normalize(clean(text), mode=mode)


def load_chosen_mode(default: str = "none") -> str:
    """Return the normalisation mode selected by the Stage 3 ablation.

    The choice lives in a file written by ``tools/run_ablation.py`` rather
    than being hard-coded here, so that the value used by every later stage is
    demonstrably the one the experiment produced.  Falls back to ``default``
    if the ablation has not been run yet.
    """
    if not CHOSEN_MODE_FILE.exists():
        return default
    chosen = json.loads(CHOSEN_MODE_FILE.read_text(encoding="utf-8"))["mode"]
    if chosen not in MODES:
        raise ValueError(f"{CHOSEN_MODE_FILE} holds an unknown mode: {chosen!r}")
    return chosen


# --------------------------------------------------------------------------
# demonstration
# --------------------------------------------------------------------------
# The sentences this project's argument rests on.  Printing them side by side
# makes the trade-off visible without reading any code.
DEMO_SENTENCES = [
    "the process was killed",
    "kill the process before restarting the server",
    "the build failed with a fatal error on line NUM",
    "garbage collection destroys unused objects",
    "this library is broken and the docs are useless",
    "see URL and call CODE for details",
]


def main() -> None:
    print("Stage 3 -- word normalisation, three modes\n")
    width = max(len(s) for s in DEMO_SENTENCES)
    for sentence in DEMO_SENTENCES:
        modes = compare_modes(sentence)
        print(f"  original : {sentence}")
        print(f"  lemma    : {modes['lemma']}")
        print(f"  stem     : {modes['stem']}")
        print()

    print("Lemmatization needs context -- this is why whole sentences are")
    print("tagged at once rather than word by word:\n")
    print(f"  {'word':10} {'alone':>10} {'in a sentence':>16}")
    for word in ("killed", "crashed", "failed", "destroys"):
        alone = normalize(word, mode="lemma")
        in_context = normalize_tokens(f"it {word} the server", mode="lemma")[1]
        print(f"  {word:10} {alone:>10} {in_context:>16}")

    print("\nStemming is cruder and can produce non-words:")
    for word in ("was", "this", "library", "garbage", "finally"):
        print(f"  {word:10} -> {normalize(word, mode='stem')}")

    print("\nPlaceholders survive every mode:")
    print(f"  {normalize('see URL and call CODE EMO_POS', mode='stem')}")

    print("\nSentence segmentation:")
    example = "First sentence. Second one! And a third?"
    print(f"  {example!r}")
    for i, sentence in enumerate(segment(example), 1):
        print(f"    {i}. {sentence}")


if __name__ == "__main__":
    main()
