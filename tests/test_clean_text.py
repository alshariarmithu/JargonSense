"""Tests for src/extraction/clean_text.py (Stage SE-2.2).

Run from the repository root:  python -m pytest -q
"""

import pytest

from src.extraction.clean_text import (
    DOMAINS,
    TOKEN_PATTERN,
    collapse_whitespace,
    describe_rules,
    clean,
    clean_tokens,
    tokenize,
)


# --------------------------------------------------------------------------
# the four sentences the work-division doc names explicitly
# --------------------------------------------------------------------------
def test_se_jargon_sentence_keeps_its_vocabulary():
    # "killed" must survive as "killed", not be stemmed to "kill".
    assert clean("The process was killed", "se") == "the process was killed"


def test_se_fatal_error_keeps_the_line_number():
    # Numbers are only masked in the health domain.
    assert clean("Fatal error on line 3", "se") == "fatal error on line 3"


def test_health_relief_sentence_keeps_stopped():
    assert clean("The nausea finally stopped", "health") == (
        "the nausea finally stopped"
    )


def test_health_dosage_becomes_a_placeholder():
    assert clean("I take 600mg three times a day", "health") == (
        "i take DOSE three times a day"
    )


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
    assert clean(raw, "health") == expected


def test_negation_words_are_never_removed():
    out = clean("No more panic attacks since I started this", "health")
    assert out.startswith("no more panic attacks")


def test_words_ending_in_nt_are_left_alone():
    # Regression: an optional apostrophe in the n't rule turned "want" into
    # "wa not" and "treatment" into "treatme not".
    assert clean("I want a different treatment", "health") == (
        "i want a different treatment"
    )


# --------------------------------------------------------------------------
# ordering constraints
# --------------------------------------------------------------------------
def test_urls_are_replaced_before_emoticons():
    # "http://" contains ":/", a sad face.  Wrong order gives "httpEMO_NEG/...".
    out = clean("see http://stackoverflow.com/q/1 for details", "se")
    assert out == "see URL for details"
    assert "EMO_NEG" not in out


def test_placeholders_stay_uppercase_and_distinct_from_real_words():
    # "url" and "user" occur as ordinary words in SE text; if lowercasing ran
    # after substitution the two would be indistinguishable.
    out = clean("the user hit www.example.com", "se")
    assert out == "the user hit URL"


def test_mentions_become_user():
    assert clean("@talnicolas thanks", "se") == "USER thanks"


# --------------------------------------------------------------------------
# domain-specific rules fire only in their own domain
# --------------------------------------------------------------------------
def test_code_spans_are_masked_only_in_se():
    raw = "call foo(bar) and check org.apache.commons"
    assert clean(raw, "se") == "call CODE and check CODE"
    assert clean(raw, "health") == raw.lower()


def test_dose_and_num_are_masked_only_in_health():
    raw = "1/4 packet twice a day for 3 weeks"
    assert clean(raw, "health") == "DOSE twice a day for NUM weeks"
    assert clean(raw, "se") == raw.lower()


def test_single_dot_does_not_trigger_the_code_rule():
    # A missing space after a full stop must not be read as a dotted path.
    assert clean("it crashed.the log says so", "se") == (
        "it crashed.the log says so"
    )


def test_text_with_no_domain_pattern_is_identical_across_domains():
    """Guards Section 3.3: preprocessing may not diverge without cause."""
    raw = "This medication ruined my week and I am furious about it"
    assert clean(raw, "se") == clean(raw, "health")


# --------------------------------------------------------------------------
# shared normalisation
# --------------------------------------------------------------------------
def test_html_entities_are_decoded_including_double_escaping():
    assert clean("a &amp;lt; b", "se") == "a < b"


def test_emoticons_become_polarity_tokens():
    out = clean("works great :-) but slow :(", "se")
    assert out == "works great EMO_POS but slow EMO_NEG"


def test_repeated_characters_and_punctuation_collapse():
    assert clean("sooooo bad!!!!", "health") == "soo bad!!"


@pytest.mark.parametrize("bad", [None, float("nan"), "", "   \n\t  "])
def test_empty_and_missing_input_give_the_empty_string(bad):
    assert clean(bad, "se") == ""


def test_unknown_domain_raises():
    with pytest.raises(ValueError):
        clean("anything", "finance")


# --------------------------------------------------------------------------
# tokeniser contract
# --------------------------------------------------------------------------
def test_tokenizer_keeps_punctuation_and_placeholder_tokens():
    assert clean_tokens("Broken?! take 10 mg :-(", "health") == [
        "broken", "?", "!", "take", "DOSE", "EMO_NEG",
    ]


def test_token_pattern_is_the_one_the_tokenizer_uses():
    # A3 passes TOKEN_PATTERN to TfidfVectorizer and A6 passes it to LIME, so
    # the constant and the function must not drift apart.
    import re

    text = clean("kill the process! 42 times", "se")
    assert re.findall(TOKEN_PATTERN, text) == tokenize(text)


# --------------------------------------------------------------------------
# report table
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# collapse_whitespace: used by the corpus builders before clean() runs
# --------------------------------------------------------------------------
def test_collapse_whitespace_removes_line_breaks_and_double_spaces():
    # The two shapes Druglib reviews actually contain.
    assert collapse_whitespace("a.  b\n\n\nc") == "a. b c"


def test_collapse_whitespace_trims_the_ends():
    assert collapse_whitespace("  \n padded \t ") == "padded"


@pytest.mark.parametrize("empty", [None, float("nan"), ""])
def test_collapse_whitespace_tolerates_missing_text(empty):
    # Builders map it over a column that may hold NaN.
    assert collapse_whitespace(empty) == ""


@pytest.mark.parametrize("domain", DOMAINS)
def test_collapse_whitespace_changes_no_token(domain):
    # The point of doing it at build time: it must be a no-op on the output of
    # clean(), which squeezes the same characters at step 9.
    raw = "Stopped after  400 mg\n\nno relief :-(   really?!"
    assert clean(collapse_whitespace(raw), domain) == clean(raw, domain)


@pytest.mark.parametrize("domain", DOMAINS)
def test_rule_table_is_complete_and_ordered(domain):
    rows = describe_rules(domain)
    steps = [r[0] for r in rows]
    assert steps == list(range(1, len(rows) + 1))
    assert any(scope == domain for _, scope, *_ in rows)
