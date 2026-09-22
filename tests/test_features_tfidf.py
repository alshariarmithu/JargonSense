"""Tests for the retained `(1,3)` TF-IDF representation."""

import pytest

import src.features.features_tfidf as features
from src.extraction.clean_text import TOKEN_PATTERN


def test_vectorizer_uses_unigrams_bigrams_and_trigrams():
    vectorizer = features.build_vectorizer(mode="none", min_df=1, max_df=1.0)
    matrix = vectorizer.fit_transform([
        "kill the process", "thanks this works", "how does this work"
    ])
    names = set(vectorizer.get_feature_names_out())
    assert vectorizer.ngram_range == (1, 3)
    assert "kill" in names
    assert "kill the" in names
    assert "kill the process" in names
    assert matrix.shape[0] == 3


def test_vectorizer_accepts_raw_text_and_uses_shared_token_pattern():
    vectorizer = features.build_vectorizer(mode="none", min_df=1, max_df=1.0)
    vectorizer.fit(["Call <code>foo()</code>!", "Thanks, this works!"])
    assert vectorizer.token_pattern == TOKEN_PATTERN
    assert "CODE" in vectorizer.vocabulary_


def test_tuned_settings_are_usable():
    settings = features.load_tuned_settings()
    vectorizer = features.build_vectorizer(mode="none", **settings)
    assert vectorizer.min_df >= 1


def test_tuned_settings_fall_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "TUNED_SETTINGS_FILE", tmp_path / "absent.json")
    assert features.load_tuned_settings() == features.DEFAULT_SETTINGS

