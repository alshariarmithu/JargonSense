"""Tests for src/features/features_tfidf.py (Stage 4.1).

Run from the repository root:  python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.extraction.clean_text import LABELS
from src.features.features_tfidf import (
    CONFIGS,
    DEFAULT_SETTINGS,
    build_vectorizer,
    compound_phrase_report,
    load_tuned_settings,
    select_settings,
    top_ngrams_per_class,
    validation_standard_error,
    vocabulary_stats,
)

# A small corpus with an obvious signal, so assertions are about behaviour
# rather than about the real data.
TEXTS = [
    "thanks this works great",
    "thanks a lot this is excellent",
    "this is terrible and i hate it",
    "horrible broken library i hate this",
    "how do i use this function",
    "what does this method return",
    "the process was killed by the handler",
    "kill the process then restart it",
]
POLARITY = [
    "positive", "positive",
    "negative", "negative",
    "neutral", "neutral",
    "neutral", "neutral",
]


def fit(config="tfidf12", **overrides):
    vectorizer = build_vectorizer(config, domain="se", mode="none",
                                  min_df=1, **overrides)
    matrix = vectorizer.fit_transform(TEXTS)
    return vectorizer, matrix


# --------------------------------------------------------------------------
# the contract Stage 6 depends on
# --------------------------------------------------------------------------
def test_vectorizer_accepts_raw_text():
    """The vectoriser must clean text itself, not receive it pre-cleaned.

    LIME, SHAP and the demo all hand the model raw strings.  If cleaning
    happened outside the vectoriser, they would be explaining a model that
    never existed.
    """
    vectorizer = build_vectorizer("tfidf11", domain="se", mode="none",
                                  min_df=1, max_df=1.0)
    vectorizer.fit([
        "Kill the PROCESS via <code>foo()</code> at http://x.com",
        "another post entirely",
    ])
    vocabulary = vectorizer.vocabulary_
    assert "CODE" in vocabulary       # code span was replaced
    assert "URL" in vocabulary        # url was replaced
    assert "process" in vocabulary    # and the text was lowercased
    assert "PROCESS" not in vocabulary


def test_configs_differ_only_in_ngram_range():
    for name, expected in CONFIGS.items():
        assert build_vectorizer(name, mode="none").ngram_range == expected


def test_unigram_config_cannot_contain_phrases():
    """tfidf11 is the control: it structurally cannot learn word pairs."""
    vectorizer, _ = fit("tfidf11")
    assert all(" " not in term for term in vectorizer.vocabulary_)


def test_bigram_config_captures_word_pairs():
    vectorizer, _ = fit("tfidf12")
    assert "the process" in vectorizer.vocabulary_


def test_trigram_config_captures_three_word_phrases():
    vectorizer, _ = fit("tfidf13")
    assert "kill the process" in vectorizer.vocabulary_


def test_unknown_config_is_rejected():
    with pytest.raises(ValueError, match="config must be one of"):
        build_vectorizer("tfidf99")


# --------------------------------------------------------------------------
# the fix that made the top-terms table meaningful
# --------------------------------------------------------------------------
def test_top_ngrams_rank_sentiment_words_above_stopwords():
    """Ranking by raw mean TF-IDF returns "the" and "i" for every class.

    Distinctiveness subtracts a term's weight in the other classes, so a word
    used everywhere cancels out.  This test fails if the ranking ever reverts
    to plain mean TF-IDF.
    """
    vectorizer, matrix = fit("tfidf11")
    top = top_ngrams_per_class(vectorizer, matrix, POLARITY, top_n=5)

    positive = top[top["polarity"] == "positive"]["ngram"].tolist()
    negative = top[top["polarity"] == "negative"]["ngram"].tolist()

    assert "thanks" in positive
    assert "hate" in negative
    assert "this" not in positive + negative   # appears in every document


def test_distinctiveness_is_positive_for_class_specific_terms():
    vectorizer, matrix = fit("tfidf11")
    top = top_ngrams_per_class(vectorizer, matrix, POLARITY, top_n=3)
    assert (top["distinctiveness"] > 0).all()


def test_top_ngrams_covers_every_class():
    vectorizer, matrix = fit("tfidf11")
    top = top_ngrams_per_class(vectorizer, matrix, POLARITY, top_n=3)
    assert set(top["polarity"]) == set(LABELS)


# --------------------------------------------------------------------------
# compound phrases
# --------------------------------------------------------------------------
def test_compound_report_marks_phrases_absent_from_the_unigram_vocabulary():
    vectorizers, matrices = {}, {}
    for config in CONFIGS:
        vectorizers[config], matrices[config] = fit(config)

    report = compound_phrase_report(
        vectorizers, matrices, POLARITY, phrases=["kill the process", "the process"])
    report = report.set_index("phrase")

    # No multi-word phrase can be in the unigram vocabulary.
    assert not report.loc["the process", "in_tfidf11"]
    assert not report.loc["kill the process", "in_tfidf11"]
    # A bigram reaches tfidf12; a trigram needs tfidf13.
    assert report.loc["the process", "in_tfidf12"]
    assert not report.loc["kill the process", "in_tfidf12"]
    assert report.loc["kill the process", "in_tfidf13"]


def test_compound_report_counts_documents_per_class():
    vectorizers, matrices = {}, {}
    for config in CONFIGS:
        vectorizers[config], matrices[config] = fit(config)

    report = compound_phrase_report(
        vectorizers, matrices, POLARITY, phrases=["the process"])
    row = report.iloc[0]
    # "the process" appears in the two neutral SE-jargon documents only.
    assert row["n_neutral"] == 2
    assert row["n_positive"] == 0
    assert row["n_negative"] == 0


# --------------------------------------------------------------------------
# the selection rule
# --------------------------------------------------------------------------
def test_standard_error_shrinks_with_a_bigger_validation_set():
    tuning = pd.DataFrame({"val_macro_f1": [0.79, 0.78, 0.80]})
    assert (validation_standard_error(tuning, 650)
            > validation_standard_error(tuning, 6500))


def test_selection_prefers_the_smaller_vocabulary_within_one_standard_error():
    """The one-standard-error rule: when scores tie, ship the simpler model."""
    subset = pd.DataFrame({
        "val_macro_f1": [0.7960, 0.7900],
        "vocabulary_size": [130_078, 4_813],
        "min_df": [1, 5],
        "max_features": ["None", "None"],
    })
    pick = select_settings(subset, standard_error=0.0161)
    assert pick["vocabulary_size"] == 4_813, "should not chase a within-noise win"


def test_selection_keeps_the_best_when_the_margin_is_real():
    subset = pd.DataFrame({
        "val_macro_f1": [0.8500, 0.7000],
        "vocabulary_size": [130_078, 4_813],
        "min_df": [1, 5],
        "max_features": ["None", "None"],
    })
    pick = select_settings(subset, standard_error=0.0161)
    assert pick["vocabulary_size"] == 130_078, "a large margin must still win"


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------
def test_vocabulary_stats_reports_shape_and_sparsity():
    vectorizer, matrix = fit("tfidf11")
    stats = vocabulary_stats("tfidf11", vectorizer, matrix)
    assert stats["vocabulary_size"] == matrix.shape[1]
    assert 0.0 <= stats["sparsity"] <= 1.0
    assert stats["nonzero_entries"] == matrix.nnz


def test_load_tuned_settings_returns_usable_arguments():
    for config in CONFIGS:
        settings = load_tuned_settings(config)
        assert "min_df" in settings and "max_df" in settings
        # Must be accepted by the vectoriser it configures.
        build_vectorizer(config, mode="none", **settings)


def test_tuned_settings_fall_back_to_defaults(tmp_path, monkeypatch):
    import src.features.features_tfidf as ft
    monkeypatch.setattr(ft, "TUNED_SETTINGS_FILE", tmp_path / "absent.json")
    assert ft.load_tuned_settings("tfidf12") == DEFAULT_SETTINGS
