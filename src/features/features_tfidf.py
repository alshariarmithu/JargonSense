"""TF-IDF feature construction for the retained sentiment pipeline.

The project keeps one representation: word unigrams, bigrams, and trigrams.
The vectorizer accepts raw text so the saved sklearn pipeline owns all text
cleaning and normalization required at prediction time.
"""

from __future__ import annotations

import json
from functools import partial

from sklearn.feature_extraction.text import TfidfVectorizer

from src.extraction.clean_text import TOKEN_PATTERN
from src.paths import FEATURES
from src.preprocessing.normalize import load_chosen_mode, prepare

CONFIG = "tfidf13"
NGRAM_RANGE = (1, 3)
DEFAULT_SETTINGS = {
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}
TUNED_SETTINGS_FILE = FEATURES / "tfidf_settings.json"


def load_tuned_settings() -> dict:
    """Return the retained configuration's tuned settings or defaults."""
    if not TUNED_SETTINGS_FILE.exists():
        return dict(DEFAULT_SETTINGS)
    settings = json.loads(TUNED_SETTINGS_FILE.read_text(encoding="utf-8"))
    return {**DEFAULT_SETTINGS, **settings.get(CONFIG, settings)}


def build_vectorizer(mode: str | None = None, **overrides) -> TfidfVectorizer:
    """Build the `(1,3)` TF-IDF vectorizer used by the final model."""
    if mode is None:
        mode = load_chosen_mode()
    settings = {**load_tuned_settings(), **overrides}
    return TfidfVectorizer(
        preprocessor=partial(prepare, mode=mode),
        token_pattern=TOKEN_PATTERN,
        ngram_range=NGRAM_RANGE,
        **settings,
    )
