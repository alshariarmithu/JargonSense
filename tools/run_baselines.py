import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.splits import load_splits
from src.evaluate import evaluate, append_to_results_table

import nltk
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def run_majority_baseline(domain: str):
    train_df, _, test_df = load_splits(domain)
    
    # Find majority class in train
    majority_class = train_df["polarity"].mode()[0]
    
    y_true = test_df["polarity"].tolist()
    y_pred = [majority_class] * len(y_true)
    
    metrics = evaluate(f"majority", domain, y_true, y_pred)
    append_to_results_table(f"majority", domain, metrics)
    print(f"[{domain.upper()}] Majority baseline complete. Macro-F1: {metrics['macro_f1']:.4f}")

def run_vader_baseline(domain: str):
    _, _, test_df = load_splits(domain)
    
    sia = SentimentIntensityAnalyzer()
    
    y_true = test_df["polarity"].tolist()
    texts = test_df["text"].tolist()
    
    y_pred = []
    for text in texts:
        scores = sia.polarity_scores(str(text))
        comp = scores['compound']
        if comp >= 0.05:
            y_pred.append("positive")
        elif comp <= -0.05:
            y_pred.append("negative")
        else:
            y_pred.append("neutral")
            
    metrics = evaluate(f"vader", domain, y_true, y_pred)
    append_to_results_table(f"vader", domain, metrics)
    print(f"[{domain.upper()}] VADER baseline complete. Macro-F1: {metrics['macro_f1']:.4f}")

if __name__ == "__main__":
    for domain in ["se", "health"]:
        run_majority_baseline(domain)
        run_vader_baseline(domain)
