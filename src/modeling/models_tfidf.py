"""Train the retained TF-IDF (1,3) + calibrated Linear SVM pipeline.

Run from the repository root with ``python -m src.modeling.models_tfidf``.
The saved pipeline accepts raw text and exposes both ``predict`` and
``predict_proba`` for evaluation and explanation tools.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluation.evaluate import evaluate
from src.extraction.clean_text import SEED
from src.features.features_tfidf import build_vectorizer
from src.paths import MODELS
from src.preprocessing.splits import load_splits

MODEL_NAME = "tfidf13_svm"
CV_FOLDS = 5
SCORING = "f1_macro"
PARAM_GRID = {
    "clf__estimator__C": [0.01, 0.1, 1, 10],
    "clf__estimator__class_weight": [None, "balanced"],
}


def make_classifier() -> CalibratedClassifierCV:
    """Return a calibrated linear SVM with probability output."""
    return CalibratedClassifierCV(LinearSVC(random_state=SEED), cv=CV_FOLDS)


def build_model(domain: str = "se", mode: str | None = None) -> Pipeline:
    """Build the complete raw-text prediction pipeline."""
    return Pipeline([
        ("tfidf", build_vectorizer(domain=domain, mode=mode)),
        ("clf", make_classifier()),
    ])


def train(domain: str = "se", tune: bool = True) -> tuple[Pipeline, dict]:
    """Train, validate, save, and return the retained pipeline."""
    train_df, val_df, _ = load_splits(domain)
    pipeline = build_model(domain=domain)
    started = time.perf_counter()

    if tune:
        search = GridSearchCV(
            pipeline,
            PARAM_GRID,
            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42),
            scoring=SCORING,
            n_jobs=1,
            refit=True,
        )
        search.fit(train_df["text"], train_df["polarity"])
        model = search.best_estimator_
        best_params = search.best_params_
        cv_score = float(search.best_score_)
    else:
        pipeline.fit(train_df["text"], train_df["polarity"])
        model = pipeline
        best_params = {}
        cv_score = float("nan")

    metrics = evaluate(
        f"{MODEL_NAME}_val",
        domain,
        val_df["polarity"],
        model.predict(val_df["text"]),
        out_dir="results/models",
        fig_dir="results/models/figures",
    )
    model_dir = MODELS / domain
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{MODEL_NAME}.joblib")

    summary = {
        "model": MODEL_NAME,
        "cv_macro_f1": cv_score,
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "best_params": best_params,
        "fit_seconds": round(time.perf_counter() - started, 1),
    }
    return model, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip grid search")
    args = parser.parse_args()
    _, summary = train(tune=not args.quick)
    print(json.dumps(summary, indent=2))
    print(f"saved {MODELS / 'se' / f'{MODEL_NAME}.joblib'}")


if __name__ == "__main__":
    main()
