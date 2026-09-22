"""Tests for src/extraction/clean_text.py (Stage SE-2.2).

Run from the repository root:  python -m pytest -q
"""

import pytest

from src.extraction.clean_text import (
    TOKEN_PATTERN,
    collapse_whitespace,
    describe_rules,
    clean,
    clean_tokens,
    tokenize,
)


# --------------------------------------------------------------------------
# the sentences the work-division doc names explicitly
# --------------------------------------------------------------------------
def test_se_jargon_sentence_keeps_its_vocabulary():
    # "killed" must survive as "killed", not be stemmed to "kill".
    assert clean("The process was killed") == "the process was killed"


def test_se_fatal_error_keeps_the_line_number():
    # Digits are never masked: a line number is part of the error text.
    assert clean("Fatal error on line 3") == "fatal error on line 3"


# --------------------------------------------------------------------------
# negation -- the single most important thing not to break
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("I can't sleep", "i can not sleep"),
        ("It doesn't work", "it does not work"),
        ("It won't stop", "it will not stop"),
        # curly apostrophe, as copy-pasted from a browser
        ("It isn’t working", "it is not working"),
    ],
)
def test_negation_contractions_expand(raw, expected):
    assert clean(raw) == expected


def test_negation_words_are_never_removed():
    out = clean("No more crashes since I upgraded")
    assert out.startswith("no more crashes")


def test_words_ending_in_nt_are_left_alone():
    # Regression: an optional apostrophe in the n't rule turned "want" into
    # "wa not" and "deployment" into "deployme not".
    assert clean("I want a different deployment") == (
        "i want a different deployment"
    )


# --------------------------------------------------------------------------
# ordering constraints
# --------------------------------------------------------------------------
def test_urls_are_replaced_before_emoticons():
    # "http://" contains ":/", a sad face.  Wrong order gives "httpEMO_NEG/...".
    out = clean("see http://stackoverflow.com/q/1 for details")
    assert out == "see URL for details"
    assert "EMO_NEG" not in out


def test_placeholders_stay_uppercase_and_distinct_from_real_words():
    # "url" and "user" occur as ordinary words in SE text; if lowercasing ran
    # after substitution the two would be indistinguishable.
    out = clean("the user hit www.example.com")
    assert out == "the user hit URL"


def test_mentions_become_user():
    assert clean("@talnicolas thanks") == "USER thanks"


# --------------------------------------------------------------------------
# the code rules
# --------------------------------------------------------------------------
def test_code_spans_are_masked():
    raw = "call foo(bar) and check org.apache.commons"
    assert clean(raw) == "call CODE and check CODE"


def test_digits_are_never_masked():
    # There is no NUM rule: "1/4" and "3" are ordinary text in SE prose.
    raw = "1/4 of the batch failed after 3 weeks"
    assert clean(raw) == raw.lower()


def test_single_dot_does_not_trigger_the_code_rule():
    # A missing space after a full stop must not be read as a dotted path.
    assert clean("it crashed.the log says so") == (
        "it crashed.the log says so"
    )


# --------------------------------------------------------------------------
# shared normalisation
# --------------------------------------------------------------------------
def test_html_entities_are_decoded_including_double_escaping():
    assert clean("a &amp;lt; b") == "a < b"


def test_emoticons_become_polarity_tokens():
    out = clean("works great :-) but slow :(")
    assert out == "works great EMO_POS but slow EMO_NEG"


def test_repeated_characters_and_punctuation_collapse():
    assert clean("sooooo bad!!!!") == "soo bad!!"


@pytest.mark.parametrize("bad", [None, float("nan"), "", "   \n\t  "])
def test_empty_and_missing_input_give_the_empty_string(bad):
    assert clean(bad) == ""


# --------------------------------------------------------------------------
# tokeniser contract
# --------------------------------------------------------------------------
def test_tokenizer_keeps_punctuation_and_placeholder_tokens():
    assert clean_tokens("Broken?! call foo(bar) :-(") == [
        "broken", "?", "!", "call", "CODE", "EMO_NEG",
    ]


def test_token_pattern_is_the_one_the_tokenizer_uses():
    # A3 passes TOKEN_PATTERN to TfidfVectorizer and A6 passes it to LIME, so
    # the constant and the function must not drift apart.
    import re

    text = clean("kill the process! 42 times")
    assert re.findall(TOKEN_PATTERN, text) == tokenize(text)


# --------------------------------------------------------------------------
# collapse_whitespace: used by the corpus builder before clean() runs
# --------------------------------------------------------------------------
def test_collapse_whitespace_removes_line_breaks_and_double_spaces():
    assert collapse_whitespace("a.  b\n\n\nc") == "a. b c"


def test_collapse_whitespace_trims_the_ends():
    assert collapse_whitespace("  \n padded \t ") == "padded"


@pytest.mark.parametrize("empty", [None, float("nan"), ""])
def test_collapse_whitespace_tolerates_missing_text(empty):
    # The builder maps it over a column that may hold NaN.
    assert collapse_whitespace(empty) == ""


def test_collapse_whitespace_changes_no_token():
    # The point of doing it at build time: it must be a no-op on the output of
    # clean(), which squeezes the same characters at step 9.
    raw = "Crashed after  400 iterations\n\nno fix :-(   really?!"
    assert clean(collapse_whitespace(raw)) == clean(raw)


# --------------------------------------------------------------------------
# report table
# --------------------------------------------------------------------------
def test_rule_table_is_complete_and_ordered():
    rows = describe_rules()
    steps = [r[0] for r in rows]
    assert steps == list(range(1, len(rows) + 1))
    assert any(scope == "se" for _, scope, *_ in rows)
