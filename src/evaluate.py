"""
Shared evaluation module for downstream models.
"""

from typing import Sequence
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from src.preprocess import LABELS

def evaluate_predictions(y_true: Sequence[str], y_pred: Sequence[str], title: str = "Evaluation Results") -> dict:
    """
    Computes accuracy, macro-F1, and per-class metrics.
    Prints a formatted report and returns the metrics dictionary.
    """
    print(f"\n--- {title} ---")
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}\n")
    
    report = classification_report(y_true, y_pred, labels=LABELS, digits=4, zero_division=0)
    print(report)
    
    report_dict = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    
    # Ensure our specific metrics are at the top level for easy access
    report_dict["accuracy"] = acc
    report_dict["macro_f1"] = macro_f1
    
    return report_dict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth.")
    parser.add_argument("--true", required=True, help="Path to ground truth CSV (must contain 'polarity' column)")
    parser.add_argument("--pred", required=True, help="Path to predictions CSV (must contain 'prediction' column)")
    args = parser.parse_args()
    
    df_true = pd.read_csv(args.true, sep=";")
    df_pred = pd.read_csv(args.pred, sep=";")
    
    if len(df_true) != len(df_pred):
        raise ValueError(f"Mismatched row counts: true={len(df_true)}, pred={len(df_pred)}")
        
    evaluate_predictions(df_true["polarity"], df_pred["prediction"])
