"""Build the SE corpus from the Senti4SD gold standard (Task A1).

Reads the gold-standard workbook, standardises it to the project's
``id;text;polarity`` contract, and reports the numbers the report needs:
the filter log, Fleiss' kappa over the three raters, and the class
distribution that the health corpus is balanced against in build_health.py.

    python -m src.build_se

Outputs
-------
data/processed/se/clean.csv      id;text;polarity
data/processed/se/build_log.json filter counts, kappa, class distribution

The workbook is preferred over Senti4SD's train/test partition CSVs: it holds
the same 4,423 items but also the r1/r2/r3 rater columns, without which the
agreement statistic cannot be computed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.preprocess import LABELS, collapse_whitespace

ROOT = Path(__file__).resolve().parents[1]
XLSX = (
    ROOT
    / "data/raw/senti4sd/Senti4SD_GoldStandard_and_DSM"
    / "Senti4SD_GoldStandard_EmotionPolarity.xlsx"
)
SHEET = "Annotations and Gold Labels"
OUT_DIR = ROOT / "data/processed/se"

RATERS = ["r1", "r2", "r3"]

#: The rater columns are free text: mixed case plus a handful of typos.
_RATER_FIXES = {"postive": "positive", "poitive": "positive", "netural": "neutral"}


def _normalise_label(value) -> str | None:
    """Lowercase a rater's label and repair the known typos."""
    if value is None or value != value:
        return None
    text = str(value).strip().lower()
    text = _RATER_FIXES.get(text, text)
    return text if text in LABELS else None


def fleiss_kappa(ratings: pd.DataFrame) -> tuple[float, int]:
    """Fleiss' kappa over the rater columns of *ratings*.

    Items where any rater is missing or unparseable are dropped, since Fleiss'
    kappa assumes a fixed number of raters per item.  Returns the kappa and the
    number of items it was computed on.
    """
    from statsmodels.stats.inter_rater import fleiss_kappa as _fk

    clean = ratings.apply(lambda col: col.map(_normalise_label)).dropna()
    counts = pd.DataFrame(
        {label: (clean == label).sum(axis=1) for label in LABELS},
        index=clean.index,
    )
    return float(_fk(counts.to_numpy())), len(counts)


def build() -> pd.DataFrame:
    if not XLSX.exists():
        raise FileNotFoundError(
            f"{XLSX} not found -- run `python data/download.py` first."
        )

    raw = pd.read_excel(XLSX, sheet_name=SHEET)
    log: dict[str, object] = {"rows_loaded": len(raw)}

    df = raw.rename(
        columns={
            "Label": "id",
            "Text": "text",
            "Final Label (majority voting)": "polarity",
        }
    )[["id", "text", "polarity", *RATERS]]

    # Agreement is measured on the full annotated set, before any filtering.
    kappa, kappa_n = fleiss_kappa(df[RATERS])
    log["fleiss_kappa"] = round(kappa, 4)
    log["fleiss_kappa_items"] = kappa_n

    df["id"] = df["id"].astype(str).str.strip()
    # Same whitespace normalisation as build_health.py, so a stored row is one
    # line in both corpora.  The Senti4SD export happens to be free of internal
    # runs already; this keeps it that way if the export ever changes.
    df["text"] = df["text"].map(collapse_whitespace)
    df["polarity"] = df["polarity"].astype(str).str.strip().str.lower()

    before = len(df)
    df = df[df["text"].str.len() > 0]
    log["dropped_empty_text"] = before - len(df)

    before = len(df)
    df = df[df["polarity"].isin(LABELS)]
    log["dropped_bad_label"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    log["dropped_duplicate_text"] = before - len(df)

    if df["id"].duplicated().any():
        raise ValueError("ids are not unique after cleaning")

    df = df[["id", "text", "polarity"]].reset_index(drop=True)
    log["rows_final"] = len(df)
    log["class_counts"] = df["polarity"].value_counts().reindex(LABELS).to_dict()
    log["class_proportions"] = {
        k: round(v / len(df), 4) for k, v in log["class_counts"].items()
    }
    log["mean_words"] = round(df["text"].str.split().str.len().mean(), 2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "clean.csv", sep=";", index=False, encoding="utf-8")
    (OUT_DIR / "build_log.json").write_text(json.dumps(log, indent=2), "utf-8")

    print(f"[SE] {log['rows_loaded']} rows loaded")
    print(f"[SE] dropped: empty={log['dropped_empty_text']} "
          f"bad_label={log['dropped_bad_label']} "
          f"duplicate={log['dropped_duplicate_text']}")
    print(f"[SE] Fleiss' kappa = {kappa:.4f} over {kappa_n} items, 3 raters")
    print(f"[SE] mean length = {log['mean_words']} words")
    for label in LABELS:
        n = log["class_counts"][label]
        print(f"[SE]   {label:<9} {n:>5}  ({n / len(df):.1%})")
    print(f"[SE] wrote {OUT_DIR / 'clean.csv'}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    build()
