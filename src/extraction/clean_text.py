from __future__ import annotations

import html
import re
from typing import NamedTuple

LABELS = ["negative", "neutral", "positive"]

SEED = 42

PLACEHOLDERS = ("URL", "USER", "CODE", "EMO_POS", "EMO_NEG")

TOKEN_PATTERN = r"[A-Za-z_]+|\d+|[!?]"
_TOKEN_RE = re.compile(TOKEN_PATTERN)

#
WHITESPACE_PATTERN = r"\s+"
_WHITESPACE_RE = re.compile(WHITESPACE_PATTERN)


class Rule(NamedTuple):
    """One regex substitution, named so the report table can be generated."""

    name: str
    pattern: re.Pattern
    repl: str

    def apply(self, text: str) -> str:
        return self.pattern.sub(self.repl, text)


def _rule(name: str, pattern: str, repl: str) -> Rule:
    return Rule(name, re.compile(pattern), repl)


# emotion -icons

_EMO_POS = [
    ":-))",
    ":))",
    ":-)",
    ":-]",
    ":-d",
    ":-p",
    ";-)",
    "^_^",
    ":)",
    ":]",
    ":d",
    ":p",
    ";)",
    ";p",
    "=)",
    "=d",
    "<3",
]
_EMO_NEG = [
    ":-((",
    ":((",
    ":'(",
    ":-(",
    ":-[",
    ":-/",
    ":-\\",
    ">:(",
    "</3",
    ":(",
    ":[",
    ":/",
    ":\\",
    "=(",
]


def _emoticon_rule(name: str, faces: list[str], repl: str) -> Rule:
    alt = "|".join(re.escape(f) for f in sorted(faces, key=len, reverse=True))
    return _rule(name, rf"(?:{alt})(?![A-Za-z0-9])", repl)


# rule tables
# steps 3-4: generic web noise, run before the code rules
_PRE_CODE: list[Rule] = [
    _rule("urls -> URL", r"https?://\S+|www\.\S+", "URL"),
    _rule("mentions -> USER", r"@\w[\w.-]*", "USER"),
]

# step 5: the StackOverflow-specific rules -- code spans become one CODE token
_CODE: list[Rule] = [
    _rule("code tags -> CODE", r"<code>.*?</code>", "CODE"),
    _rule("backtick spans -> CODE", r"`+[^`]*`+", "CODE"),
    _rule("calls foo(bar) -> CODE", r"\b\w[\w]*\([^()]*\)", "CODE"),
    _rule("dotted paths a.b.c -> CODE", r"\b\w+(?:\.\w+){2,}\b", "CODE"),
]

# steps 6-9: run after the code rules
_POST_CODE: list[Rule] = [
    _emoticon_rule("positive emoticons -> EMO_POS", _EMO_POS, "EMO_POS"),
    _emoticon_rule("negative emoticons -> EMO_NEG", _EMO_NEG, "EMO_NEG"),
    _rule("normalise apostrophes", r"[\u2018\u2019\u02bc\u00b4`]", "'"),
    _rule("can't -> can not", r"\bcan'?t\b", "can not"),
    _rule("won't -> will not", r"\bwon'?t\b", "will not"),
    _rule("shan't -> shall not", r"\bshan'?t\b", "shall not"),
    _rule("ain't -> is not", r"\bain'?t\b", "is not"),
    _rule("n't -> not", r"n't\b", " not"),
    _rule("collapse repeated letters", r"([a-z])\1{2,}", r"\1\1"),
    _rule("collapse repeated punctuation", r"([!?.,])\1{2,}", r"\1\1"),
    _rule("squeeze whitespace", WHITESPACE_PATTERN, " "),
]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def clean(text) -> str:
    """Normalise one document.

    Parameters
    ----------
    text
        Raw document.  ``None`` and NaN are treated as the empty string, so the
        function is safe to hand straight to ``Series.apply``.

    Returns
    -------
    str
        Lowercased text with placeholders in uppercase.
    """
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

    for rule in _PRE_CODE + _CODE + _POST_CODE:
        out = rule.apply(out)

    return out.strip()


def collapse_whitespace(text) -> str:
    if text is None or text != text:
        return ""
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def clean_tokens(text) -> list[str]:
    """Convenience: ``tokenize(clean(text))``."""
    return tokenize(clean(text))


def describe_rules() -> list[tuple[int, str, str, str, str]]:
    """Return the rule table as ``(step, scope, name, pattern, replacement)``.

    The report's preprocessing table is generated from this, so the
    documentation cannot drift away from the code.
    """
    rows: list[tuple[int, str, str, str, str]] = [
        (1, "generic", "html.unescape", "-", "-"),
        (2, "generic", "lowercase", "-", "-"),
    ]
    step = 3
    for scope, rules in (
        ("generic", _PRE_CODE),
        ("se", _CODE),
        ("generic", _POST_CODE),
    ):
        for rule in rules:
            rows.append((step, scope, rule.name, rule.pattern.pattern, rule.repl))
            step += 1
    return rows
