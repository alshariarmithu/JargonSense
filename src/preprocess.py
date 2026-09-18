"""Domain-aware text preprocessing (Task A2).

One `preprocess()` is used by every feature extractor and model in both
domains.  The module is deliberately free of I/O and of pandas so that it can
be imported from tests, notebooks, the CLI and sklearn pipelines alike.

Public API
----------
preprocess(text, domain) -> str     normalised text
tokenize(text)           -> list    tokens of already-normalised text
TOKEN_PATTERN            -> str     regex shared by tokenize(),
                                    TfidfVectorizer(token_pattern=...) and
                                    LimeTextExplainer(split_expression=...)
describe_rules(domain)   -> list    the rule table, for the report

Rules, in application order
---------------------------
shared, identical in both domains:
  1. html.unescape            &amp;lt;  ->  <
  2. lowercase                                              see note [1]
  3. urls           -> URL                                  see note [2]
  4. @mentions      -> USER
  5. domain rules             the only permitted divergence, see below
  6. emoticons      -> EMO_POS / EMO_NEG
  7. negation expansion       can't -> can not,  n't -> " not"
  8. collapse runs            sooooo -> soo,  !!!! -> !!
  9. squeeze whitespace

domain="se":
  code tags <code>..</code>, backtick spans, calls foo(bar), and dotted
  paths a.b.c  ->  CODE

domain="health":
  dosage expressions 600mg / 10 mg / 1/4 packet  ->  DOSE
  every remaining bare number                    ->  NUM

Deliberately NOT done: stopword removal and stemming.  "kill"/"killed" and
"stop"/"stopped" are the phenomenon this project measures, and collapsing them
would erase it.  Negation words are never removed for the same reason -- in the
health corpus ("no more panic attacks") negation *is* the entire effect.

Notes
-----
[1] The work-division doc lists lowercasing as step 4, after URL and USER have
    been inserted.  It is moved to step 2 here so that every placeholder stays
    uppercase.  The literal words "url" and "user" are extremely common in
    StackOverflow text; lowercase placeholders would be indistinguishable from
    them and would silently corrupt the TF-IDF vocabulary.

[2] URL replacement must run before emoticon replacement, because "http://"
    contains ":/", which is a sad-face emoticon.  Reversing the two turns every
    URL into "httpEMO_NEG/...".

Both deviations are order-only; the set of rules is unchanged, and they apply
identically to both domains, so no cross-domain comparison is affected.
"""

from __future__ import annotations

import html
import re
from typing import NamedTuple

DOMAINS = ("se", "health")

#: Label order used everywhere in the project.
LABELS = ["negative", "neutral", "positive"]

#: Random seed used everywhere in the project.
SEED = 42

#: Placeholder tokens introduced by the rules below.  All uppercase so they can
#: never collide with a real word.
PLACEHOLDERS = ("URL", "USER", "CODE", "DOSE", "NUM", "EMO_POS", "EMO_NEG")

#: Single source of truth for tokenisation.  Matches words (including the
#: underscore of EMO_POS / EMO_NEG), bare numbers, and the two punctuation marks
#: that carry sentiment.
TOKEN_PATTERN = r"[A-Za-z_]+|\d+|[!?]"
_TOKEN_RE = re.compile(TOKEN_PATTERN)


class Rule(NamedTuple):
    """One regex substitution, named so the report table can be generated."""

    name: str
    pattern: re.Pattern
    repl: str

    def apply(self, text: str) -> str:
        return self.pattern.sub(self.repl, text)


def _rule(name: str, pattern: str, repl: str) -> Rule:
    return Rule(name, re.compile(pattern), repl)


# --------------------------------------------------------------------------
# emoticons
# --------------------------------------------------------------------------
# Longest first, so ":-)" is tried before ":)".  The trailing lookahead stops
# the two-character forms from firing inside words and paths: it blocks
# "c:\dir" -> "cEMO_NEGir" and "10:30" style collisions.
_EMO_POS = [
    ":-))", ":))", ":-)", ":-]", ":-d", ":-p", ";-)", "^_^",
    ":)", ":]", ":d", ":p", ";)", ";p", "=)", "=d", "<3",
]
_EMO_NEG = [
    ":-((", ":((", ":'(", ":-(", ":-[", ":-/", ":-\\", ">:(", "</3",
    ":(", ":[", ":/", ":\\", "=(",
]


def _emoticon_rule(name: str, faces: list[str], repl: str) -> Rule:
    alt = "|".join(re.escape(f) for f in sorted(faces, key=len, reverse=True))
    return _rule(name, rf"(?:{alt})(?![A-Za-z0-9])", repl)


# --------------------------------------------------------------------------
# rule tables
# --------------------------------------------------------------------------
# steps 3-4: run before the domain rules
_PRE_DOMAIN: list[Rule] = [
    _rule("urls -> URL", r"https?://\S+|www\.\S+", "URL"),
    _rule("mentions -> USER", r"@\w[\w.-]*", "USER"),
]

# step 5: the only permitted divergence between domains
_DOMAIN: dict[str, list[Rule]] = {
    "se": [
        _rule("code tags -> CODE", r"<code>.*?</code>", "CODE"),
        _rule("backtick spans -> CODE", r"`+[^`]*`+", "CODE"),
        _rule("calls foo(bar) -> CODE", r"\b\w[\w]*\([^()]*\)", "CODE"),
        # Two or more dots required.  A single dot would swallow ordinary prose
        # with a missing space after the full stop ("it crashed.the log says").
        _rule("dotted paths a.b.c -> CODE", r"\b\w+(?:\.\w+){2,}\b", "CODE"),
    ],
    "health": [
        _rule(
            "dosage with unit -> DOSE",
            r"\b\d+(?:[.,]\d+)?\s*"
            r"(?:mgs?|mcg|ug|g|kg|ml|cc|l|iu|units?|tabs?|tablets?|pills?"
            r"|capsules?|caps?|puffs?|drops?|sprays?|patch(?:es)?|%)\b",
            "DOSE",
        ),
        _rule(
            "fractional dose -> DOSE",
            r"\b\d+\s*/\s*\d+\s*"
            r"(?:packets?|tablets?|tabs?|pills?|cups?|teaspoons?|tsps?"
            r"|tbsps?|doses?|patch(?:es)?)\b",
            "DOSE",
        ),
        # Everything numeric that is left: bare counts, fractions, decimals.
        _rule("bare numbers -> NUM", r"\b\d+(?:[./,]\d+)*\b", "NUM"),
    ],
}

# steps 6-9: run after the domain rules
_POST_DOMAIN: list[Rule] = [
    _emoticon_rule("positive emoticons -> EMO_POS", _EMO_POS, "EMO_POS"),
    _emoticon_rule("negative emoticons -> EMO_NEG", _EMO_NEG, "EMO_NEG"),
    _rule("normalise apostrophes", r"[\u2018\u2019\u02bc\u00b4`]", "'"),
    _rule("can't -> can not", r"\bcan'?t\b", "can not"),
    _rule("won't -> will not", r"\bwon'?t\b", "will not"),
    _rule("shan't -> shall not", r"\bshan'?t\b", "shall not"),
    _rule("ain't -> is not", r"\bain'?t\b", "is not"),
    # The apostrophe is mandatory here: "n'?t" would rewrite "want" as "wa not".
    _rule("n't -> not", r"n't\b", " not"),
    _rule("collapse repeated letters", r"([a-z])\1{2,}", r"\1\1"),
    _rule("collapse repeated punctuation", r"([!?.,])\1{2,}", r"\1\1"),
    _rule("squeeze whitespace", r"\s+", " "),
]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def preprocess(text, domain: str) -> str:
    """Normalise one document.

    Parameters
    ----------
    text
        Raw document.  ``None`` and NaN are treated as the empty string, so the
        function is safe to hand straight to ``Series.apply``.
    domain
        ``"se"`` or ``"health"``.  Controls step 5 only; every other rule is
        identical across domains.

    Returns
    -------
    str
        Lowercased text with placeholders in uppercase.
    """
    if domain not in _DOMAIN:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")

    if text is None or text != text:  # NaN is the only value unequal to itself
        return ""

    out = str(text)
    # Parts of the Senti4SD export are escaped twice ("&amp;lt;").
    for _ in range(2):
        unescaped = html.unescape(out)
        if unescaped == out:
            break
        out = unescaped
    out = out.lower()

    for rule in _PRE_DOMAIN + _DOMAIN[domain] + _POST_DOMAIN:
        out = rule.apply(out)

    return out.strip()


def tokenize(text: str) -> list[str]:
    """Split *already preprocessed* text using ``TOKEN_PATTERN``.

    Keeps ``!``, ``?`` and the emoticon placeholders as standalone tokens.
    """
    return _TOKEN_RE.findall(text or "")


def preprocess_tokens(text, domain: str) -> list[str]:
    """Convenience: ``tokenize(preprocess(text, domain))``."""
    return tokenize(preprocess(text, domain))


def describe_rules(domain: str) -> list[tuple[int, str, str, str, str]]:
    """Return the rule table as ``(step, scope, name, pattern, replacement)``.

    The report's preprocessing table is generated from this, so the
    documentation cannot drift away from the code.
    """
    if domain not in _DOMAIN:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")

    rows: list[tuple[int, str, str, str, str]] = [
        (1, "shared", "html.unescape", "-", "-"),
        (2, "shared", "lowercase", "-", "-"),
    ]
    step = 3
    for scope, rules in (
        ("shared", _PRE_DOMAIN),
        (domain, _DOMAIN[domain]),
        ("shared", _POST_DOMAIN),
    ):
        for rule in rules:
            rows.append((step, scope, rule.name, rule.pattern.pattern, rule.repl))
            step += 1
    return rows
