"""
Creates stratified 70/15/15 train/val/test splits for the Senti4SD corpus.

Outputs:
data/processed/se/train.csv
data/processed/se/val.csv
data/processed/se/test.csv
data/processed/se/splits_log.json
"""

import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.extraction.clean_text import LABELS, SEED

from src.paths import SE_DIR

def create_splits() -> None:
    clean_path = SE_DIR / "clean.csv"

    if not clean_path.exists():
        raise FileNotFoundError(f"Input file not found: {clean_path}. Run build_se.py first.")

    df = pd.read_csv(clean_path, sep=";")

    if "polarity" not in df.columns:
        raise ValueError(f"Expected 'polarity' column in {clean_path}")

    # First split: 70% train, 30% temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["polarity"]
    )

    # Second split: divide the 30% temp evenly into 15% val and 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["polarity"]
    )

    # Save splits
    train_df.to_csv(SE_DIR / "train.csv", sep=";", index=False, encoding="utf-8")
    val_df.to_csv(SE_DIR / "val.csv", sep=";", index=False, encoding="utf-8")
    test_df.to_csv(SE_DIR / "test.csv", sep=";", index=False, encoding="utf-8")

    # Generate splits log
    log = {
        "seed": SEED,
        "total_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "distributions": {
            "overall": df["polarity"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(),
            "train": train_df["polarity"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(),
            "val": val_df["polarity"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(),
            "test": test_df["polarity"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()
        }
    }

    (SE_DIR / "splits_log.json").write_text(json.dumps(log, indent=2), "utf-8")

    print(f"[SE] Created splits in {SE_DIR}:")
    print(f"[SE]   Train: {len(train_df)} rows ({len(train_df)/len(df):.1%})")
    print(f"[SE]   Val:   {len(val_df)} rows ({len(val_df)/len(df):.1%})")
    print(f"[SE]   Test:  {len(test_df)} rows ({len(test_df)/len(df):.1%})")

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the frozen train, val, and test splits."""
    if not (SE_DIR / "train.csv").exists():
        raise FileNotFoundError(f"Splits not found in {SE_DIR}. Run create_splits first.")

    train_df = pd.read_csv(SE_DIR / "train.csv", sep=";")
    val_df = pd.read_csv(SE_DIR / "val.csv", sep=";")
    test_df = pd.read_csv(SE_DIR / "test.csv", sep=";")

    return train_df, val_df, test_df

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    create_splits()
