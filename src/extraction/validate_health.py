"""Score the health-corpus label validation (Task B1.4).

Reads the hand-labelled validation sample and answers the two questions that
GATE 1 turns on:

  1. does the rating-derived label agree with human judgement at >= 80%?
  2. if not, does the stricter 1-2 / 9-10 binning do better?

It also computes Cohen's kappa against a second annotator's file when one is
supplied.  Working alone there is no second annotator, so inter-annotator kappa
is simply unavailable -- the script says so rather than inventing a number, and
that gap belongs in the report's Limitations section.

    python -m src.extraction.validate_health
    python -m src.extraction.validate_health --annotator2 path/to/their_labels.csv

Input:  data/processed/health/validation_sample.csv with `my_label` filled in
Output: data/processed/health/validation_log.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.extraction.clean_text import LABELS

from src.paths import ROOT
OUT_DIR = ROOT / "data/processed/health"
SAMPLE = OUT_DIR / "validation_sample.csv"

AGREEMENT_THRESHOLD = 0.80
KAPPA_THRESHOLD = 0.75

#: The binning in use, and the fallback B1.4 step 3 prescribes if it fails.
SCHEMES = {
    "current_1-3_8-10": {"negative": range(1, 4), "positive": range(8, 11)},
    "strict_1-2_9-10": {"negative": range(1, 3), "positive": range(9, 11)},
}


def cohens_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two annotators over the project's label set.

    Written out rather than imported so this script runs on pandas alone.
    """
    if not len(a):
        return float("nan")
    observed = (a.to_numpy() == b.to_numpy()).mean()
    expected = sum(
        (a == label).mean() * (b == label).mean() for label in LABELS
    )
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def band(rating: int, scheme: dict[str, range]) -> str:
    for label, span in scheme.items():
        if rating in span:
            return label
    return "discard"


def load_labels(path: Path, column: str) -> pd.DataFrame:
    """Read a hand-labelled file, keeping only rows with a usable label."""
    df = pd.read_csv(path, sep=";")
    if column not in df.columns:
        raise ValueError(f"{path.name} has no `{column}` column")
    df[column] = df[column].astype(str).str.strip().str.lower()
    return df[df[column].isin(LABELS)]


def validate(annotator2: Path | None = None) -> dict:
    if not SAMPLE.exists():
        raise FileNotFoundError(
            f"{SAMPLE} not found -- run `python -m src.extraction.build_health` first."
        )

    raw = pd.read_csv(SAMPLE, sep=";")
    mine = load_labels(SAMPLE, "my_label")

    log: dict[str, object] = {
        "sample_rows": len(raw),
        "rows_hand_labelled": len(mine),
    }

    if not len(mine):
        log["status"] = "not labelled yet"
        _report(log)
        return log

    log["my_label_counts"] = mine["my_label"].value_counts().to_dict()

    # Question 1 and 2: rating-derived label vs my judgement, per scheme.
    log["schemes"] = {}
    for name, scheme in SCHEMES.items():
        derived = mine["rating"].map(lambda r: band(r, scheme))
        # A scheme's discard band has no label to compare against.
        compared = mine[derived != "discard"]
        derived = derived[derived != "discard"]
        agreement = (
            float((derived.to_numpy() == compared["my_label"].to_numpy()).mean())
            if len(compared)
            else float("nan")
        )
        log["schemes"][name] = {
            "rows_compared": len(compared),
            "rows_in_discard_band": int(len(mine) - len(compared)),
            "agreement": round(agreement, 4),
            "kappa": round(cohens_kappa(derived, compared["my_label"]), 4),
            "meets_threshold": bool(agreement >= AGREEMENT_THRESHOLD),
        }

    # Where the disagreements are, so the binning can be diagnosed not guessed.
    current = mine["rating"].map(lambda r: band(r, SCHEMES["current_1-3_8-10"]))
    wrong = mine[(current != "discard") & (current != mine["my_label"])]
    log["disagreements"] = len(wrong)
    log["disagreement_by_rating"] = (
        wrong["rating"].value_counts().sort_index().to_dict()
    )
    log["disagreement_directions"] = (
        (current[wrong.index] + " -> " + wrong["my_label"]).value_counts().to_dict()
    )

    # Question 3: inter-annotator kappa, only if a second annotator exists.
    if annotator2 is None:
        log["inter_annotator_kappa"] = None
        log["inter_annotator_note"] = (
            "single annotator: Cohen's kappa between annotators is not "
            "computable. Report rating-agreement only and state this in "
            "Limitations."
        )
    else:
        theirs = load_labels(annotator2, "my_label").set_index("id")["my_label"]
        both = mine.set_index("id").join(theirs.rename("their_label"), how="inner")
        log["inter_annotator_rows"] = len(both)
        log["inter_annotator_kappa"] = round(
            cohens_kappa(both["my_label"], both["their_label"]), 4
        )

    OUT_DIR.joinpath("validation_log.json").write_text(
        json.dumps(log, indent=2, default=str), "utf-8"
    )
    _report(log)
    return log


def _report(log: dict) -> None:
    print(f"[VALIDATE] sample rows: {log['sample_rows']}")
    print(f"[VALIDATE] hand-labelled: {log['rows_hand_labelled']}")

    if not log["rows_hand_labelled"]:
        print(
            "\n[VALIDATE] Nothing to score yet. Open\n"
            "[VALIDATE]   data/processed/health/validation_sample.csv\n"
            "[VALIDATE] and fill the `my_label` column with negative / neutral /\n"
            "[VALIDATE] positive, judging the text only -- do not look at the\n"
            "[VALIDATE] `rating` or `rating_derived` columns while labelling, or\n"
            "[VALIDATE] the agreement figure is worthless."
        )
        return

    print(f"[VALIDATE] my labels: {log['my_label_counts']}")
    print("\n[VALIDATE] rating-derived label vs my judgement:")
    for name, r in log["schemes"].items():
        flag = "PASS" if r["meets_threshold"] else "FAIL"
        print(
            f"[VALIDATE]   {name:<18} {r['agreement']:.1%} agreement  "
            f"kappa={r['kappa']:.3f}  n={r['rows_compared']}  [{flag}]"
        )

    print(f"\n[VALIDATE] {log['disagreements']} disagreements under the current scheme")
    if log["disagreement_directions"]:
        print(f"[VALIDATE]   directions: {log['disagreement_directions']}")
        print(f"[VALIDATE]   by rating:  {log['disagreement_by_rating']}")

    current = log["schemes"]["current_1-3_8-10"]
    strict = log["schemes"]["strict_1-2_9-10"]
    print()
    if current["meets_threshold"]:
        print("[VALIDATE] GATE 1 label check: PASS on the current 1-3 / 8-10 binning.")
    elif strict["meets_threshold"]:
        print(
            "[VALIDATE] GATE 1 label check: current binning FAILS but strict\n"
            "[VALIDATE]   1-2 / 9-10 passes. Change BANDS in src/build_health.py\n"
            "[VALIDATE]   to that scheme, re-run the build, and report both numbers."
        )
    else:
        print(
            "[VALIDATE] GATE 1 label check: FAIL on both schemes.\n"
            "[VALIDATE]   Take the documented fallback (B1.4 step 4): switch to the\n"
            "[VALIDATE]   Drugs.com corpus, UCI id 462, sampled to ~4,000 rows."
        )

    if log.get("inter_annotator_kappa") is None:
        print(f"[VALIDATE] {log['inter_annotator_note']}")
    else:
        kappa = log["inter_annotator_kappa"]
        flag = "PASS" if kappa >= KAPPA_THRESHOLD else "FAIL"
        print(
            f"[VALIDATE] inter-annotator kappa = {kappa:.3f} over "
            f"{log['inter_annotator_rows']} rows  [{flag}, need >= {KAPPA_THRESHOLD}]"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotator2",
        type=Path,
        default=None,
        help="a second annotator's labels (same format), for Cohen's kappa",
    )
    validate(annotator2=parser.parse_args().annotator2)
