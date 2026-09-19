"""Run the retained pipeline on the 60-sentence SE stress set."""

from __future__ import annotations

import joblib
import pandas as pd

from src.modeling.models_tfidf import MODEL_NAME
from src.paths import MODELS, RESULTS, STRESS_TEST

STRESS_RESULTS = RESULTS / "stress_test"


def load_stress_set() -> pd.DataFrame:
    frame = pd.read_csv(STRESS_TEST / "se_stress.csv", sep=";")
    frame["lexicon_word"] = frame["lexicon_word"].fillna("")
    frame["has_jargon"] = frame["lexicon_word"] != ""
    return frame


def score_model(model, stress: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    predicted = model.predict(stress["text"])
    frame = stress.assign(predicted=predicted)
    frame["correct"] = frame["predicted"] == frame["intended_label"]
    neutral = frame[frame["intended_label"] == "neutral"]
    negative = frame[frame["intended_label"] == "negative"]
    with_jargon = negative[negative["has_jargon"]]
    without_jargon = negative[~negative["has_jargon"]]
    pairs = frame[frame["pair_id"].notna()]
    both_correct = pairs.groupby("pair_id").apply(
        lambda group: bool(group["correct"].all()), include_groups=False
    )
    metrics = {
        "model": MODEL_NAME,
        "false_alarm_rate": round(float((neutral["predicted"] == "negative").mean()), 4),
        "negative_recall": round(float((negative["predicted"] == "negative").mean()), 4),
        "recall_with_jargon": round(float((with_jargon["predicted"] == "negative").mean()), 4),
        "recall_without_jargon": round(float((without_jargon["predicted"] == "negative").mean()), 4),
        "minimal_pair_accuracy": round(float(both_correct.mean()), 4),
        "accuracy": round(float(frame["correct"].mean()), 4),
    }
    return metrics, frame


def main() -> None:
    model_path = MODELS / "se" / f"{MODEL_NAME}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found; run `python -m src.modeling.models_tfidf` first"
        )
    metrics, predictions = score_model(joblib.load(model_path), load_stress_set())
    STRESS_RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(STRESS_RESULTS / "stress_results_se.csv", index=False)
    predictions.to_csv(
        STRESS_RESULTS / f"stress_predictions_{MODEL_NAME}.csv", index=False
    )
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
