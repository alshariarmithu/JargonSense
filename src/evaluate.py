"""
Shared evaluation module for downstream models.
"""

import json
from pathlib import Path
from typing import Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

from src.preprocess import LABELS

ROOT = Path(__file__).resolve().parents[1]

def evaluate(name: str, domain: str, y_true: Sequence[str], y_pred: Sequence[str], out_dir: str = "results/metrics") -> dict:
    """
    Computes accuracy, macro-F1, weighted-F1, per-class P/R/F1, and confusion matrices.
    Saves metrics to JSON and confusion matrix to a PNG file.
    """
    out_path = ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = ROOT / "results/figures"
    fig_path.mkdir(parents=True, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)
    
    report_dict = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    
    cm_counts = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize='true')
    
    # Calculate domain-specific confusion rates
    se_neutral_to_negative = 0.0
    health_positive_to_negative = 0.0
    
    if "neutral" in LABELS and "negative" in LABELS:
        neutral_idx = LABELS.index("neutral")
        negative_idx = LABELS.index("negative")
        if np.sum(cm_counts[neutral_idx, :]) > 0:
            se_neutral_to_negative = cm_norm[neutral_idx, negative_idx]
            
    if "positive" in LABELS and "negative" in LABELS:
        positive_idx = LABELS.index("positive")
        negative_idx = LABELS.index("negative")
        if np.sum(cm_counts[positive_idx, :]) > 0:
            health_positive_to_negative = cm_norm[positive_idx, negative_idx]

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            label: report_dict[label] for label in LABELS if label in report_dict
        },
        "confusion_matrix_counts": cm_counts.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "se_neutral_to_negative_rate": float(se_neutral_to_negative),
        "health_positive_to_negative_rate": float(health_positive_to_negative)
    }
    
    # Save to JSON
    json_path = out_path / f"{domain}_{name}.json"
    json_path.write_text(json.dumps(metrics, indent=2), "utf-8")
    
    # Save Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f"Confusion Matrix: {domain.upper()} - {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(fig_path / f"cm_{domain}_{name}.png", dpi=150)
    plt.close()
    
    return metrics

def append_to_results_table(name: str, domain: str, metrics: dict, table_path: str = "results/metrics/all_results.csv"):
    """
    Appends the evaluation metrics to the master results CSV.
    """
    file_path = ROOT / table_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    row = {
        "domain": domain,
        "model_name": name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }
    
    # Add per-class f1 if available
    for label in LABELS:
        if label in metrics["per_class"]:
            row[f"f1_{label}"] = metrics["per_class"][label]["f1-score"]
            
    row["se_neutral_to_negative_rate"] = metrics.get("se_neutral_to_negative_rate", 0.0)
    row["health_positive_to_negative_rate"] = metrics.get("health_positive_to_negative_rate", 0.0)
    
    df_row = pd.DataFrame([row])
    
    if file_path.exists():
        df_row.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(file_path, index=False)

def mcnemar_test(y_true: Sequence[str], y_pred1: Sequence[str], y_pred2: Sequence[str]) -> dict:
    """
    Performs McNemar's test to compare two models' predictions.
    Null hypothesis: the two models have the same error rate.
    """
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    correct1 = (y_true == y_pred1)
    correct2 = (y_true == y_pred2)
    
    a = np.sum(correct1 & correct2)
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    d = np.sum(~correct1 & ~correct2)
    
    full_table = [[a, b], [c, d]]
    
    if b + c == 0:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "contingency_table": full_table,
            "significant_05": False
        }
    
    result = mcnemar(full_table, exact=False, correction=True)
    
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "contingency_table": full_table,
        "significant_05": bool(result.pvalue < 0.05)
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth.")
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--domain", required=True, help="Domain (se or health)")
    parser.add_argument("--true", required=True, help="Path to ground truth CSV (must contain 'polarity' column)")
    parser.add_argument("--pred", required=True, help="Path to predictions CSV (must contain 'prediction' column)")
    args = parser.parse_args()
    
    df_true = pd.read_csv(args.true, sep=";")
    df_pred = pd.read_csv(args.pred, sep=";")
    
    if len(df_true) != len(df_pred):
        raise ValueError(f"Mismatched row counts: true={len(df_true)}, pred={len(df_pred)}")
        
    metrics = evaluate(args.name, args.domain, df_true["polarity"], df_pred["prediction"])
    append_to_results_table(args.name, args.domain, metrics)
    print(f"Evaluated {args.name} on {args.domain}. Macro-F1: {metrics['macro_f1']:.4f}")
