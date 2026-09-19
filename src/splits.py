"""
Creates stratified 70/15/15 train/val/test splits for a given domain.

Outputs:
data/processed/{domain}/train.csv
data/processed/{domain}/val.csv
data/processed/{domain}/test.csv
data/processed/{domain}/splits_log.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import DOMAINS, LABELS, SEED

ROOT = Path(__file__).resolve().parents[1]

def create_splits(domain: str) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"Domain must be one of {DOMAINS}")

    data_dir = ROOT / f"data/processed/{domain}"
    clean_path = data_dir / "clean.csv"
    
    if not clean_path.exists():
        raise FileNotFoundError(f"Input file not found: {clean_path}. Run build_{domain}.py first.")
    
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
    train_df.to_csv(data_dir / "train.csv", sep=";", index=False, encoding="utf-8")
    val_df.to_csv(data_dir / "val.csv", sep=";", index=False, encoding="utf-8")
    test_df.to_csv(data_dir / "test.csv", sep=";", index=False, encoding="utf-8")
    
    # Generate splits log
    log = {
        "domain": domain,
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
    
    (data_dir / "splits_log.json").write_text(json.dumps(log, indent=2), "utf-8")
    
    print(f"[{domain.upper()}] Created splits in {data_dir}:")
    print(f"[{domain.upper()}]   Train: {len(train_df)} rows ({len(train_df)/len(df):.1%})")
    print(f"[{domain.upper()}]   Val:   {len(val_df)} rows ({len(val_df)/len(df):.1%})")
    print(f"[{domain.upper()}]   Test:  {len(test_df)} rows ({len(test_df)/len(df):.1%})")

def load_splits(domain: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the frozen train, val, and test splits for the given domain."""
    if domain not in DOMAINS:
        raise ValueError(f"Domain must be one of {DOMAINS}")
    
    data_dir = ROOT / f"data/processed/{domain}"
    
    if not (data_dir / "train.csv").exists():
        raise FileNotFoundError(f"Splits not found for {domain}. Run create_splits first.")
        
    train_df = pd.read_csv(data_dir / "train.csv", sep=";")
    val_df = pd.read_csv(data_dir / "val.csv", sep=";")
    test_df = pd.read_csv(data_dir / "test.csv", sep=";")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=DOMAINS, help="Domain to split")
    args = parser.parse_args()
    create_splits(args.domain)
