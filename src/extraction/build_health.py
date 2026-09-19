"""
Outputs
data/processed/health/clean.csv              id;text;polarity
data/processed/health/bias_probe.csv         id;rating;benefits;side_effects
data/processed/health/neutral_candidates.csv to hand-label -> neutral_labelled.csv
data/processed/health/validation_sample.csv  to hand-label, for kappa
data/processed/health/build_log.json         every count above
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.extraction.clean_text import LABELS, SEED, collapse_whitespace, clean

from src.paths import ROOT
RAW_DIR = ROOT / "data/raw/druglib"
OUT_DIR = ROOT / "data/processed/health"
SE_LOG = ROOT / "data/processed/se/build_log.json"

TEXT_COL = "commentsReview"
MIN_WORDS = 15
NEUTRAL_CANDIDATES = 1200
VALIDATION_SAMPLE = 100
GATE1_MIN_NEUTRAL = 300


BANDS = {"negative": range(1, 4), "positive": range(8, 11)}

_BOILERPLATE = re.compile(
    r"^(?:thanks?|thank you|see above|as above|same as above|none|no comments?"
    r"|n/?a|nothing|no|don'?t know(?: yet)?|not sure|too soon to tell)[.!]*$"
)

_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'd|me|my|mine|myself|we|our|us)\b")

#: Copy-pasted drug monograph text: clinical register, third person.
_CLINICAL = re.compile(
    r"\b(?:indicated for|contraindicated|contraindication|adverse reactions?"
    r"|in the management of|in the treatment of|patients? should|monitor"
    r"|assess(?:ment)? for|prophylaxis|administration|therapy is|efficacy"
    r"|mg/kg|dosage should|is used to treat)\b"
)

# Signals used to prefilter neutral candidates (B1.3 step 3).
_IMPERATIVE = re.compile(
    r"\b(?:take|takes|took|taking|use|used|using|apply|applied|applying"
    r"|inject|injected|swallow|inhale|dose|dosage|prescribed)\b"
)
_SCHEDULE = re.compile(
    r"\b(?:once|twice|three times|four times|times a day|times daily|per day"
    r"|every day|daily|nightly|at night|in the morning|at bedtime|as needed"
    r"|every \w+ hours|before meals|with food)\b"
)
_AFFECT = re.compile(
    r"\b(?:love|loved|loves|hate|hated|terrible|awful|horrible|dreadful|great"
    r"|excellent|amazing|wonderful|fantastic|worst|best|recommend|miracle"
    r"|useless|disappointed|disappointing|happy|thrilled|angry|furious"
    r"|lifesaver|god ?send|never again)\b|!"
)



# loading

def load_raw() -> pd.DataFrame:
    parts = sorted(RAW_DIR.glob("druglib_*.csv"))
    if not parts:
        raise FileNotFoundError(
            f"no druglib_*.csv in {RAW_DIR} -- run `python -m src.acquisition.download` first."
        )
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    return df.rename(columns={"reviewID": "id"})


def write_bias_probe(df: pd.DataFrame) -> int:
    """B1.1: benefits/side-effects fields are positive/negative by construction.

    They are useless as training data and are set aside for the bias probe in
    B5.3, where side-effect text from rows rated 8-10 is classified: a patient
    who liked the drug describing its side effects is a clean, hand-label-free
    measurement of lexical bias.
    """
    probe = (
        df[["id", "rating", "benefitsReview", "sideEffectsReview"]]
        .rename(
            columns={"benefitsReview": "benefits", "sideEffectsReview": "side_effects"}
        )
        .assign(
            benefits=lambda d: d["benefits"].map(collapse_whitespace),
            side_effects=lambda d: d["side_effects"].map(collapse_whitespace),
        )
    )
    probe.to_csv(OUT_DIR / "bias_probe.csv", sep=";", index=False, encoding="utf-8")
    return len(probe)


# --------------------------------------------------------------------------
# filtering (B1.2)
# --------------------------------------------------------------------------
def filter_comments(
    df: pd.DataFrame, log: dict, inspect: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the B1.2 filters.

    Returns ``(strict, relaxed)``: the corpus as B1.2 specifies it, and the same
    corpus with the minimum-length filter left off.  The second is only used to
    draw neutral candidates, and only if ``--neutral-min-words`` lowers the
    threshold.
    """
    df = df.rename(columns={TEXT_COL: "text"})
    df["text"] = df["text"].map(collapse_whitespace)
    lowered = df["text"].str.lower()
    df["n_words"] = df["text"].str.split().str.len()

    quality_drops = {
        "missing_text": df["text"].str.len() == 0,
        "boilerplate": lowered.str.match(_BOILERPLATE),
        "drug_monograph": ~lowered.str.contains(_FIRST_PERSON)
        & lowered.str.contains(_CLINICAL),
    }

    keep = pd.Series(True, index=df.index)
    for name, mask in quality_drops.items():
        fired = mask & keep  # count each row against the first filter that hits
        log[f"dropped_{name}"] = int(fired.sum())
        if inspect and fired.any():
            print(f"\n--- dropped by {name} ({fired.sum()}) ---")
            for text in df.loc[fired, "text"].head(4):
                print("   ", text[:160].replace("\n", " "))
        keep &= ~mask

    short = df["n_words"] < MIN_WORDS
    log[f"dropped_under_{MIN_WORDS}_words"] = int((short & keep).sum())
    if inspect and (short & keep).any():
        print(f"\n--- dropped by under_{MIN_WORDS}_words ({(short & keep).sum()}) ---")
        for text in df.loc[short & keep, "text"].head(4):
            print("   ", text[:160].replace("\n", " "))

    relaxed = df[keep].drop_duplicates(subset="text", keep="first")
    strict = relaxed[relaxed["n_words"] >= MIN_WORDS]
    log["dropped_duplicate_text"] = int(keep.sum() - len(relaxed))
    log["rows_after_filtering"] = len(strict)
    log["rows_after_filtering_no_length_rule"] = len(relaxed)
    return strict.reset_index(drop=True), relaxed.reset_index(drop=True)


# --------------------------------------------------------------------------
# labelling (B1.3)
# --------------------------------------------------------------------------
def assign_bands(df: pd.DataFrame, log: dict | None = None) -> pd.DataFrame:
    def band(rating: int) -> str:
        for label, span in BANDS.items():
            if rating in span:
                return label
        return "discard"

    df = df.copy()
    df["polarity"] = df["rating"].map(band)
    if log is not None:
        log["band_counts"] = df["polarity"].value_counts().to_dict()
    return df


def score_neutrality(df: pd.DataFrame) -> pd.DataFrame:
    """Score each review on absence-of-affect signals, to cut labelling cost.

    Dosage, imperative and schedule signals count toward neutrality; any overt
    affect word or exclamation mark disqualifies the row outright.
    """
    normalised = df["text"].map(lambda t: clean(t, "health"))
    return df.assign(
        score=normalised.str.contains("DOSE").astype(int)
        + normalised.str.contains(_IMPERATIVE).astype(int)
        + normalised.str.contains(_SCHEDULE).astype(int),
        has_affect=normalised.str.contains(_AFFECT),
    )


def neutral_candidates(scored: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick the rows most worth hand-labelling as neutral.

    Rows rated 4-7 are offered first: the binning discards them anyway, so
    labelling them costs no other class any data.  The rating is kept in the
    output for context but does *not* decide the label -- a review of any rating
    can be free of affect.
    """
    out = scored[(scored["score"] >= 2) & ~scored["has_affect"]].copy()
    out["free_to_label"] = (out["polarity"] == "discard").astype(int)
    out = out.sort_values(["free_to_label", "score"], ascending=False)
    return out.head(n)[["id", "text", "rating", "polarity", "n_words", "score"]]


def load_hand_labels(log: dict) -> pd.DataFrame:
    """Read neutral_labelled.csv if the hand-labelling pass has been done."""
    path = OUT_DIR / "neutral_labelled.csv"
    if not path.exists():
        log["neutral_labelled_rows"] = 0
        return pd.DataFrame(columns=["id", "text", "polarity"])

    labelled = pd.read_csv(path, sep=";")
    labelled["polarity"] = labelled["polarity"].astype(str).str.strip().str.lower()
    labelled = labelled[labelled["polarity"].isin(LABELS)]
    log["neutral_labelled_rows"] = len(labelled)
    log["neutral_labelled_counts"] = labelled["polarity"].value_counts().to_dict()
    return labelled[["id", "text", "polarity"]]


# --------------------------------------------------------------------------
# balancing (B1.5)
# --------------------------------------------------------------------------
def target_proportions(log: dict) -> dict[str, float]:
    """SE class proportions from build_se.py, so the two corpora are comparable."""
    if SE_LOG.exists():
        props = json.loads(SE_LOG.read_text("utf-8"))["class_proportions"]
        log["target_proportions_source"] = "data/processed/se/build_log.json"
        return props
    log["target_proportions_source"] = "uniform fallback (run build_se.py first)"
    return {label: 1 / len(LABELS) for label in LABELS}


def balance(df: pd.DataFrame, props: dict[str, float], log: dict) -> pd.DataFrame:
    """Downsample to the largest size that honours *props* exactly.

    The scarcest class sets the ceiling, so the achievable total is
    ``min(available[c] / props[c])``.  Reporting that rather than silently
    dropping a class keeps the class-balance asymmetry between the two corpora
    visible, which Section 3.3 requires.
    """
    available = df["polarity"].value_counts().reindex(LABELS).fillna(0).astype(int)
    log["available_per_class"] = available.to_dict()

    present = [c for c in LABELS if available[c] > 0]
    total = int(min(available[c] / props[c] for c in present)) if present else 0
    log["binding_class"] = (
        min(present, key=lambda c: available[c] / props[c]) if present else None
    )

    # How many of the missing class would be needed to reach the target shape,
    # given what the other classes can supply.  This is the real labelling bill.
    for label in LABELS:
        if available[label] == 0:
            others = [c for c in present]
            cap = int(min(available[c] / props[c] for c in others)) if others else 0
            log[f"{label}_rows_needed_for_target"] = int(round(props[label] * cap))

    frames = [
        df[df["polarity"] == label].sample(
            n=min(int(round(props[label] * total)), int(available[label])),
            random_state=SEED,
        )
        for label in LABELS
        if available[label] > 0
    ]
    if not frames:
        return df.iloc[:0]

    out = pd.concat(frames).sample(frac=1, random_state=SEED).reset_index(drop=True)
    log["sampled_per_class"] = (
        out["polarity"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()
    )
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def build(
    inspect: bool = False,
    neutral_min_words: int = MIN_WORDS,
    n_candidates: int = NEUTRAL_CANDIDATES,
) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: dict[str, object] = {"neutral_min_words": neutral_min_words}

    raw = load_raw()
    log["rows_loaded"] = len(raw)
    log["bias_probe_rows"] = write_bias_probe(raw)

    strict, relaxed = filter_comments(raw, log, inspect)
    banded = assign_bands(strict, log)

    # Neutral candidates may be drawn from shorter reviews than the other
    # classes, but only when explicitly asked for; see the module docstring.
    pool = relaxed[relaxed["n_words"] >= neutral_min_words]
    scored = score_neutrality(assign_bands(pool))
    candidates = neutral_candidates(scored, n_candidates)
    candidates.assign(polarity="").to_csv(
        OUT_DIR / "neutral_candidates.csv", sep=";", index=False, encoding="utf-8"
    )
    log["neutral_candidates_written"] = len(candidates)
    log["neutral_candidates_available"] = int(
        ((scored["score"] >= 2) & ~scored["has_affect"]).sum()
    )
    # What a lower threshold would buy, for the report's justification table.
    relaxed_scored = score_neutrality(assign_bands(relaxed[relaxed["n_words"] >= 8]))
    log["neutral_candidates_available_at_8_words"] = int(
        ((relaxed_scored["score"] >= 2) & ~relaxed_scored["has_affect"]).sum()
    )

    # B1.4: hand-label these against the rating-derived label to check binning.
    banded_only = banded[banded["polarity"] != "discard"]
    banded_only.sample(
        n=min(VALIDATION_SAMPLE, len(banded_only)), random_state=SEED
    )[["id", "text", "rating", "polarity"]].rename(
        columns={"polarity": "rating_derived"}
    ).assign(my_label="").to_csv(
        OUT_DIR / "validation_sample.csv", sep=";", index=False, encoding="utf-8"
    )

    labelled = pd.concat(
        [banded_only[["id", "text", "polarity"]], load_hand_labels(log)],
        ignore_index=True,
    ).drop_duplicates(subset="id", keep="last")

    props = target_proportions(log)
    log["target_proportions"] = props
    final = balance(labelled, props, log)

    final[["id", "text", "polarity"]].to_csv(
        OUT_DIR / "clean.csv", sep=";", index=False, encoding="utf-8"
    )
    log["rows_final"] = len(final)
    if len(final):
        log["mean_words"] = round(final["text"].str.split().str.len().mean(), 2)
        log["mean_words_per_class"] = (
            final.assign(n=final["text"].str.split().str.len())
            .groupby("polarity")["n"]
            .mean()
            .round(2)
            .to_dict()
        )
    (OUT_DIR / "build_log.json").write_text(json.dumps(log, indent=2), "utf-8")

    _report(log, final)
    return final


def _report(log: dict, final: pd.DataFrame) -> None:
    print(f"\n[HEALTH] {log['rows_loaded']} rows loaded")
    print(f"[HEALTH] bias probe: {log['bias_probe_rows']} rows set aside")
    print("[HEALTH] filters:")
    for key, value in log.items():
        if key.startswith("dropped_"):
            print(f"[HEALTH]   {key[8:]:<22} {value:>5}")
    print(f"[HEALTH] after filtering: {log['rows_after_filtering']}")
    print(f"[HEALTH] rating bands: {log['band_counts']}")
    print(
        f"[HEALTH] neutral candidates: {log['neutral_candidates_available']} available "
        f"at {log['neutral_min_words']} words, "
        f"{log['neutral_candidates_available_at_8_words']} at 8 words"
    )
    print(f"[HEALTH] neutral hand-labels found: {log['neutral_labelled_rows']}")
    print(f"[HEALTH] target proportions ({log['target_proportions_source']}):")
    print(f"[HEALTH]   {log['target_proportions']}")
    print(
        f"[HEALTH] available: {log['available_per_class']}  "
        f"binding class: {log['binding_class']}"
    )

    if len(final):
        for label in LABELS:
            n = int((final["polarity"] == label).sum())
            print(f"[HEALTH]   {label:<9} {n:>5}  ({n / len(final):.1%})")
        print(f"[HEALTH] mean words per class: {log['mean_words_per_class']}")
    print(f"[HEALTH] wrote {OUT_DIR / 'clean.csv'}  ({len(final)} rows)")

    neutral = int(log.get("neutral_labelled_rows", 0))
    if neutral < GATE1_MIN_NEUTRAL:
        needed = log.get("neutral_rows_needed_for_target")
        print(
            f"\n[HEALTH] GATE 1 NOT MET: {neutral} neutral rows, "
            f"{GATE1_MIN_NEUTRAL} required.\n"
            f"[HEALTH] 1. Hand-label neutral_candidates.csv (fill the empty\n"
            f"[HEALTH]    `polarity` column), save as neutral_labelled.csv,\n"
            f"[HEALTH]    and re-run this script.\n"
            f"[HEALTH] 2. Hand-label validation_sample.csv and compare `my_label`\n"
            f"[HEALTH]    against `rating_derived`. Below 80% agreement the binning\n"
            f"[HEALTH]    is wrong -- try 1-2 / 9-10 (B1.4)."
        )
        if needed:
            print(
                f"[HEALTH] Matching SE's class shape needs {needed} neutral rows,\n"
                f"[HEALTH]   not the 400 the plan budgets, because negative is\n"
                f"[HEALTH]   capped at {log['available_per_class']['negative']}. "
                f"Either label {needed},\n"
                f"[HEALTH]   or accept a smaller corpus and say so in the report."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="print example rows removed by each filter, to tune the rules",
    )
    parser.add_argument(
        "--neutral-min-words",
        type=int,
        default=MIN_WORDS,
        help=(
            f"minimum length for neutral candidates (default {MIN_WORDS}, same as "
            "every other class). Lowering it buys neutrals at the cost of a "
            "length confound -- see the module docstring."
        ),
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=NEUTRAL_CANDIDATES,
        help=f"how many neutral candidates to write (default {NEUTRAL_CANDIDATES})",
    )
    args = parser.parse_args()
    build(
        inspect=args.inspect,
        neutral_min_words=args.neutral_min_words,
        n_candidates=args.candidates,
    )
