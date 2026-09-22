"""Evaluate the retained pipeline on the frozen SE test split."""

from __future__ import annotations

import joblib
import pandas as pd

from src.evaluation.evaluate import append_to_results_table, evaluate
from src.modeling.models_tfidf import MODEL_NAME
from src.paths import MODELS, RESULTS
from src.preprocessing.splits import load_splits


def main() -> None:
    _, _, test_df = load_splits()
    model_path = MODELS / "se" / f"{MODEL_NAME}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found; run `python -m src.modeling.models_tfidf` first"
        )

    model = joblib.load(model_path)
    predictions = model.predict(test_df["text"])
    metrics = evaluate(MODEL_NAME, "se", test_df["polarity"], predictions)
    append_to_results_table(MODEL_NAME, "se", metrics)

    row = {
        "model": MODEL_NAME,
        "representation": "TF-IDF (1,3)",
        "classifier": "calibrated Linear SVM",
        "kind": "discriminative",
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "neutral_to_negative": round(metrics["se_neutral_to_negative_rate"], 4),
        "negative_recall": round(metrics["per_class"]["negative"]["recall"], 4),
    }
    output = RESULTS / "metrics" / "master_results_se.csv"
    pd.DataFrame([row]).to_csv(output, index=False)
    print(pd.Series(row).to_string())
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
