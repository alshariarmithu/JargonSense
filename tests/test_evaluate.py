import numpy as np
import pytest
from src.evaluation.evaluate import evaluate, mcnemar_test
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
    
    metrics = evaluate("test_model", "se", y_true, y_pred, out_dir="results/metrics_test")
    
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
    
    metrics = evaluate("test_se", "se", y_true, y_pred, out_dir="results/metrics_test")
    assert np.isclose(metrics["se_neutral_to_negative_rate"], 2/3)
    assert metrics["health_positive_to_negative_rate"] == 0.0

def test_health_positive_to_negative_rate():
    # HEALTH confusion rate: fraction of gold-positive predicted negative
    y_true = ["positive", "positive", "positive", "positive", "neutral"]
    y_pred = ["negative", "negative", "negative", "positive", "positive"]
    # gold-positive: 4
    # predicted negative: 3
    # fraction: 3/4 = 0.75
    
    metrics = evaluate("test_health", "health", y_true, y_pred, out_dir="results/metrics_test")
    assert np.isclose(metrics["health_positive_to_negative_rate"], 0.75)

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
