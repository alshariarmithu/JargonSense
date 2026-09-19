"""Tests for src/preprocessing/normalize.py (Stage 3).

Run from the repository root:  python -m pytest -q
"""

import pytest

from src.extraction.clean_text import PLACEHOLDERS, TOKEN_PATTERN, tokenize
from src.preprocessing.normalize import (
    MODES,
    compare_modes,
    load_chosen_mode,
    prepare,
    normalize,
    normalize_tokens,
    segment,
    sentence_stats,
)


# --------------------------------------------------------------------------
# the behaviour the whole ablation depends on
# --------------------------------------------------------------------------
def test_modes_differ_on_inflected_verbs():
    """If all three modes agreed, the Stage 3 ablation would be pointless."""
    sentence = "the process was killed"
    assert normalize(sentence, mode="none") == "the process was killed"
    assert normalize(sentence, mode="lemma") == "the process be kill"
    assert normalize(sentence, mode="stem") == "the process wa kill"


def test_none_mode_preserves_inflection():
    """'killed' must stay 'killed' -- it is the word the project studies."""
    assert "killed" in normalize_tokens("the process was killed", mode="none")
    assert "kill" not in normalize_tokens("the process was killed", mode="none")


def test_stemming_collapses_kill_and_killed():
    """The collapse that motivates measuring instead of assuming."""
    killed = normalize_tokens("it killed the server", mode="stem")
    kill = normalize_tokens("it kill the server", mode="stem")
    assert killed == kill


def test_lemmatization_uses_pos_tags():
    """Without POS tags WordNet leaves verbs alone and lemmatization is a no-op.

    ``WordNetLemmatizer().lemmatize("stopped")`` returns "stopped"; only
    ``lemmatize("stopped", "v")`` returns "stop".  This test fails if the POS
    mapping is ever dropped.
    """
    tokens = normalize_tokens("the server stopped responding", mode="lemma")
    assert "stop" in tokens
    assert "stopped" not in tokens


# --------------------------------------------------------------------------
# placeholders must survive -- see the module docstring
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_placeholders_are_never_normalized(mode):
    text = " ".join(PLACEHOLDERS)
    assert normalize_tokens(text, mode=mode) == list(PLACEHOLDERS)


def test_code_placeholder_does_not_become_the_english_word():
    """Stemming CODE -> 'code' would merge it with the ordinary noun."""
    tokens = normalize_tokens("see URL and call CODE for details", mode="stem")
    assert "CODE" in tokens
    assert "code" not in tokens


# --------------------------------------------------------------------------
# negation survives -- removing it would break the corpus
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_negation_words_are_kept(mode):
    tokens = normalize_tokens("it does not work and never did", mode=mode)
    assert "not" in tokens
    assert "never" in tokens


# --------------------------------------------------------------------------
# the ablation is only fair if nothing but the mode changes
# --------------------------------------------------------------------------
def test_all_modes_produce_the_same_token_count():
    """Modes must differ in token *spelling*, never in token count.

    If one mode dropped or added tokens, the ablation would be comparing two
    things at once and the comparison would mean nothing.
    """
    sentence = "the build failed with a fatal error on line 42 !"
    counts = {m: len(normalize_tokens(sentence, mode=m)) for m in MODES}
    assert len(set(counts.values())) == 1, counts


def test_normalize_is_idempotent_through_the_vectorizer_tokenizer():
    """Joining tokens with spaces must survive re-tokenisation by TF-IDF.

    ``normalize()`` returns a string that ``TfidfVectorizer(token_pattern=
    TOKEN_PATTERN)`` will split again.  If the join were lossy, the model
    would not see the tokens this module produced.
    """
    sentence = "the build failed with a fatal error on line 42 !"
    for mode in MODES:
        tokens = normalize_tokens(sentence, mode=mode)
        assert tokenize(normalize(sentence, mode=mode)) == tokens


# --------------------------------------------------------------------------
# sentence segmentation
# --------------------------------------------------------------------------
def test_segment_splits_on_all_three_terminators():
    assert segment("First one. Second one! And a third?") == [
        "First one.",
        "Second one!",
        "And a third?",
    ]


def test_segment_handles_empty_and_missing_input():
    assert segment("") == []
    assert segment("   ") == []
    assert segment(None) == []
    assert segment(float("nan")) == []


def test_sentence_stats_counts_documents_and_sentences():
    stats = sentence_stats(["One. Two.", "Only one.", ""])
    assert stats["documents"] == 2          # the empty string is not counted
    assert stats["sentences"] == 3
    assert stats["max_sentences"] == 2


# --------------------------------------------------------------------------
# input handling and argument validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_missing_input_is_safe(mode):
    """Safe to hand straight to ``Series.apply`` on a column with gaps."""
    assert normalize_tokens(None, mode=mode) == []
    assert normalize_tokens(float("nan"), mode=mode) == []
    assert normalize("", mode=mode) == ""


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        normalize("hello", mode="lemmatise")


def test_unknown_stemmer_is_rejected():
    with pytest.raises(ValueError, match="stemmer must be one of"):
        normalize("hello", mode="stem", stemmer="lancaster")


def test_snowball_is_available_as_an_alternate_stemmer():
    porter = normalize("the running processes crashed", mode="stem", stemmer="porter")
    snowball = normalize("the running processes crashed", mode="stem", stemmer="snowball")
    # Both must stem; they need not agree on every word.
    assert "run" in porter and "run" in snowball


def test_compare_modes_returns_every_mode():
    result = compare_modes("the process was killed")
    assert set(result) == set(MODES)


# --------------------------------------------------------------------------
# Stage 2 + Stage 3 composed
# --------------------------------------------------------------------------
def test_prepare_applies_cleaning_then_normalisation():
    """One entry point, so a trained model and the demo see the same text."""
    raw = "Kill the PROCESS -- see <code>foo()</code> at http://example.com"
    assert prepare(raw, domain="se", mode="none") == "kill the process see CODE at URL"


def test_prepare_normalisation_mode_is_applied():
    raw = "The server stopped responding"
    assert "stop" in prepare(raw, domain="se", mode="lemma").split()
    assert "stopped" in prepare(raw, domain="se", mode="none").split()


def test_prepare_handles_missing_input():
    assert prepare(None, domain="se") == ""
    assert prepare(float("nan"), domain="se") == ""


def test_load_chosen_mode_returns_a_valid_mode():
    """Later stages read the ablation's decision rather than hard-coding one."""
    assert load_chosen_mode() in MODES


def test_load_chosen_mode_falls_back_when_not_yet_run(tmp_path, monkeypatch):
    import src.preprocessing.normalize as nm
    monkeypatch.setattr(nm, "CHOSEN_MODE_FILE", tmp_path / "absent.json")
    assert nm.load_chosen_mode() == "none"
    assert nm.load_chosen_mode(default="stem") == "stem"


def test_load_chosen_mode_rejects_a_corrupt_file(tmp_path, monkeypatch):
    import src.preprocessing.normalize as nm
    bad = tmp_path / "chosen_mode.json"
    bad.write_text('{"mode": "lancaster"}', encoding="utf-8")
    monkeypatch.setattr(nm, "CHOSEN_MODE_FILE", bad)
    with pytest.raises(ValueError, match="unknown mode"):
        nm.load_chosen_mode()
