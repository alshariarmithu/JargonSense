import numpy as np
import pandas as pd
import pytest
from src.evaluation.evaluate import evaluate, mcnemar_test, append_to_results_table
from src.extraction.clean_text import LABELS

def test_evaluate_accuracy_macro():
    y_true = ["positive", "positive", "negative", "neutral", "neutral"]
    y_pred = ["positive", "neutral", "negative", "neutral", "negative"]
    # Labels: negative, neutral, positive
    # True labels counts:
    # negative: 1
    # neutral: 2
    # positive: 2
    
    # y_pred counts:
    # negative: 2
    # neutral: 2
    # positive: 1
    
    # Correct:
    # y_true[0] == y_pred[0] -> True
    # y_true[1] == y_pred[1] -> False (positive != neutral)
    # y_true[2] == y_pred[2] -> True
    # y_true[3] == y_pred[3] -> True
    # y_true[4] == y_pred[4] -> False (neutral != negative)
    # Total correct: 3 / 5 = 0.6
    
    # F1 per class:
    # negative: precision = 1/2, recall = 1/1 -> f1 = 2 * (1/2) * 1 / (3/2) = 2/3 = 0.6667
    # neutral: precision = 1/2, recall = 1/2 -> f1 = 2 * (1/2) * (1/2) / (1) = 0.5
    # positive: precision = 1/1, recall = 1/2 -> f1 = 2 * 1 * (1/2) / (3/2) = 2/3 = 0.6667
    # macro_f1 = (0.6667 + 0.5 + 0.6667) / 3 = 1.8333 / 3 = 0.6111
    
    metrics = evaluate("test_model", "se", y_true, y_pred, out_dir="results/metrics_test", fig_dir="results/figures_test")
    
    assert np.isclose(metrics["accuracy"], 0.6)
    assert np.isclose(metrics["macro_f1"], (2/3 + 0.5 + 2/3)/3)
    assert metrics["confusion_matrix_counts"] == [
        [1, 0, 0], # negative
        [1, 1, 0], # neutral
        [0, 1, 1]  # positive
    ]

def test_se_neutral_to_negative_rate():
    # SE confusion rate: fraction of gold-neutral predicted negative
    y_true = ["neutral", "neutral", "neutral", "negative"]
    y_pred = ["negative", "negative", "positive", "negative"]
    # gold-neutral: 3
    # predicted negative for those: 2
    # fraction: 2/3
    
    metrics = evaluate("test_se", "se", y_true, y_pred, out_dir="results/metrics_test", fig_dir="results/figures_test")
    assert np.isclose(metrics["se_neutral_to_negative_rate"], 2/3)

def test_results_table_replaces_rather_than_duplicates(tmp_path):
    """Re-running an experiment must update its row, not add a second one.

    Without this, every re-run leaves a stale copy behind and the master
    results table silently fills with contradictory duplicates.
    """
    # An absolute path overrides the module's ROOT-relative default.
    table = tmp_path / "all_results.csv"
    relative = str(table)

    y_true = ["positive", "negative"]
    first = evaluate("m", "se", y_true, ["positive", "negative"],
                     out_dir="results/metrics_test", fig_dir="results/figures_test")
    second = evaluate("m", "se", y_true, ["negative", "negative"],
                      out_dir="results/metrics_test", fig_dir="results/figures_test")

    append_to_results_table("other", "se", first, table_path=relative)
    append_to_results_table("m", "se", first, table_path=relative)
    append_to_results_table("m", "se", second, table_path=relative)

    written = pd.read_csv(table)
    assert len(written) == 2, "re-running a model must not add a second row"
    # The row keeps its original position...
    assert written["model_name"].tolist() == ["other", "m"]
    # ...and holds the newest numbers, not the first ones.
    assert np.isclose(written.loc[1, "accuracy"], second["accuracy"])


def test_mcnemar_identical():
    # Identical predictions should have p-value ~1.0
    y_true = ["positive", "negative", "neutral", "positive"]
    y_pred1 = ["positive", "negative", "neutral", "negative"]
    y_pred2 = ["positive", "negative", "neutral", "negative"]
    
    res = mcnemar_test(y_true, y_pred1, y_pred2)
    assert res["pvalue"] == 1.0
    assert not res["significant_05"]

def test_mcnemar_different():
    # Hand-computed McNemar test
    y_true = ["A"] * 20
    # M1 correct on all 20
    y_pred1 = ["A"] * 20
    
    # M2 correct on first 10, wrong on next 10
    y_pred2 = ["A"] * 10 + ["B"] * 10
    
    # M1 correct, M2 correct -> a = 10
    # M1 correct, M2 wrong -> b = 10
    # M1 wrong, M2 correct -> c = 0
    # M1 wrong, M2 wrong -> d = 0
    
    res = mcnemar_test(y_true, y_pred1, y_pred2)
    # statistic = (abs(b - c) - 1)**2 / (b + c) 
    # = (abs(10 - 0) - 1)**2 / 10 = 81 / 10 = 8.1
    assert np.isclose(res["statistic"], 8.1)
