"""Stage 5.1 -- the nine TF-IDF models.

Run from the repository root:

    python -m src.modeling.models_tfidf              # all nine
    python -m src.modeling.models_tfidf --quick      # skip tuning, defaults only

--------------------------------------------------------------------------
Three feature sets x three classifiers
--------------------------------------------------------------------------
The classifiers are the comparison the proposal commits to:

============  ==========================================================
``mnb``       Multinomial Naive Bayes -- **generative**.  Models what the
              words of a negative post look like, then asks which class
              would most likely have produced this post.
``lr``        Logistic Regression -- **discriminative**.  Skips the
              generative story and models the class boundary directly.
``svm``       Linear SVM -- margin-based.  Draws the widest separating
              gap it can.  Neither generative nor probabilistic.
============  ==========================================================

--------------------------------------------------------------------------
Why the SVM is wrapped
--------------------------------------------------------------------------
``LinearSVC`` has no ``predict_proba``: it returns a distance from the
boundary, not a probability.  LIME, SHAP and the demo all need probabilities,
so it is wrapped in ``CalibratedClassifierCV``, which fits a small calibration
model on held-out folds to turn those distances into probabilities.

This makes the SVM the slowest of the three by some margin -- each fit is
really five fits -- which is worth knowing before wondering why it is slow.

--------------------------------------------------------------------------
Every model is a complete pipeline
--------------------------------------------------------------------------
A saved ``.joblib`` holds vectoriser *and* classifier, and accepts raw text::

    model = joblib.load("models/se/tfidf12_lr.joblib")
    model.predict(["Kill the process before restarting"])

Stage 6 depends on this.  LIME generates thousands of variants of a sentence
and asks the model to score them; the demo passes whatever a user types.
Neither can run the cleaning step first.

--------------------------------------------------------------------------
The test split is not touched here
--------------------------------------------------------------------------
Tuning uses 5-fold cross-validation **inside the training split**, and models
are compared on validation.  The test split is read once, in Stage 6.3.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.extraction.clean_text import SEED
from src.evaluation.evaluate import evaluate
from src.features.features_tfidf import CONFIGS, build_vectorizer, load_tuned_settings
from src.paths import MODELS, RESULTS
from src.preprocessing.normalize import load_chosen_mode
from src.preprocessing.splits import load_splits

MODEL_RESULTS = RESULTS / "models"

CV_FOLDS = 5
SCORING = "f1_macro"


def make_classifier(name: str):
    """Return an untrained classifier and the grid to search for it."""
    if name == "mnb":
        # alpha is additive smoothing: how much probability mass to reserve
        # for words never seen with a class during training.
        return MultinomialNB(), {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}

    if name == "lr":
        # C is inverse regularisation strength -- small C means a simpler
        # model.  Stage 4 showed a train/validation gap of 0.13-0.20, so this
        # is the knob that matters most here.
        return (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {
                "clf__C": [0.01, 0.1, 1, 10, 100],
                "clf__class_weight": [None, "balanced"],
            },
        )

    if name == "svm":
        # CalibratedClassifierCV wraps LinearSVC to provide predict_proba.
        # Grid keys reach through the wrapper with `estimator__`.
        return (
            CalibratedClassifierCV(
                LinearSVC(random_state=SEED), cv=CV_FOLDS),
            {
                "clf__estimator__C": [0.01, 0.1, 1, 10],
                "clf__estimator__class_weight": [None, "balanced"],
            },
        )

    raise ValueError(f"unknown classifier {name!r}")


CLASSIFIERS = ("mnb", "lr", "svm")


def build_model(config: str, classifier: str, domain: str = "se",
                mode: str | None = None) -> Pipeline:
    """Vectoriser + classifier as one pipeline that accepts raw text."""
    vectorizer = build_vectorizer(
        config, domain=domain, mode=mode, **load_tuned_settings(config))
    estimator, _ = make_classifier(classifier)
    return Pipeline([("tfidf", vectorizer), ("clf", estimator)])


def train_one(config: str, classifier: str, train_df, val_df,
              domain: str = "se", tune: bool = True) -> dict:
    """Tune one model by cross-validation, then score it on validation."""
    pipeline = build_model(config, classifier, domain=domain)
    _, grid = make_classifier(classifier)

    started = time.perf_counter()

    if tune:
        search = GridSearchCV(
            pipeline, grid,
            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
            scoring=SCORING, n_jobs=-1, refit=True,
        )
        search.fit(train_df["text"], train_df["polarity"])
        model = search.best_estimator_
        best_params = {k.replace("clf__", ""): v
                       for k, v in search.best_params_.items()}
        cv_score = float(search.best_score_)
    else:
        pipeline.fit(train_df["text"], train_df["polarity"])
        model = pipeline
        best_params, cv_score = {}, float("nan")

    elapsed = time.perf_counter() - started

    name = f"{config}_{classifier}"
    metrics = evaluate(
        f"{name}_val", domain, val_df["polarity"], model.predict(val_df["text"]),
        out_dir="results/models", fig_dir="results/models/figures",
    )

    model_dir = MODELS / domain
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{name}.joblib")

    return {
        "model": name,
        "config": config,
        "classifier": classifier,
        "kind": "generative" if classifier == "mnb" else "discriminative",
        "best_params": json.dumps(best_params),
        "cv_macro_f1": round(cv_score, 4),
        "val_accuracy": round(metrics["accuracy"], 4),
        "val_macro_f1": round(metrics["macro_f1"], 4),
        "val_neutral_to_negative": round(metrics["se_neutral_to_negative_rate"], 4),
        # Guards against a fake win on the line above.  A model that almost
        # never predicts "negative" scores a wonderful neutral->negative rate
        # while being useless; low recall on real negatives exposes that.
        "val_negative_recall": round(
            metrics["per_class"]["negative"]["recall"], 4),
        "fit_seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="se", choices=["se", "health"])
    parser.add_argument("--quick", action="store_true",
                        help="skip the grid search and use library defaults")
    args = parser.parse_args()

    domain = args.domain
    train_df, val_df, _ = load_splits(domain)
    MODEL_RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Stage 5.1 -- TF-IDF models on {domain.upper()}")
    print(f"  normalisation  : {load_chosen_mode()} (Stage 3 ablation)")
    print(f"  train / val    : {len(train_df):,} / {len(val_df):,}")
    print(f"  tuning         : {'off (--quick)' if args.quick else f'{CV_FOLDS}-fold CV on train, scoring {SCORING}'}")
    print(f"  test split     : untouched, spent once in Stage 6.3\n")

    rows = []
    for config in CONFIGS:
        for classifier in CLASSIFIERS:
            print(f"  {config}_{classifier:4} ...", end="", flush=True)
            row = train_one(config, classifier, train_df, val_df,
                            domain=domain, tune=not args.quick)
            rows.append(row)
            print(f" val macro-F1 {row['val_macro_f1']:.4f}"
                  f"   ({row['fit_seconds']:.0f}s)")

    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    table.to_csv(MODEL_RESULTS / f"tfidf_models_{domain}.csv", index=False)

    # ------------------------------------------------------------- report
    print("\n" + "-" * 90)
    print(f"{'model':14} {'kind':15} {'CV':>7} {'val F1':>8} {'val acc':>8}"
          f" {'neut->neg':>10} {'neg recall':>11}")
    print("-" * 90)
    for _, row in table.iterrows():
        print(f"{row['model']:14} {row['kind']:15} {row['cv_macro_f1']:>7.4f}"
              f" {row['val_macro_f1']:>8.4f} {row['val_accuracy']:>8.4f}"
              f" {row['val_neutral_to_negative']:>9.1%}"
              f" {row['val_negative_recall']:>10.1%}")
    print("-" * 90)

    best = table.iloc[0]
    print(f"\nBest by validation macro-F1: {best['model']}  "
          f"({best['val_macro_f1']:.4f}), params {best['best_params']}")

    # Generative vs discriminative -- proposal Expected Outcome #2.
    by_kind = table.groupby("kind")["val_macro_f1"].agg(["mean", "max"])
    print("\nGenerative vs discriminative (proposal outcome #2)")
    for kind, stats in by_kind.iterrows():
        print(f"  {kind:15} mean {stats['mean']:.4f}   best {stats['max']:.4f}")

    # The project's own failure metric, which need not agree with macro-F1.
    kindest = table.loc[table["val_neutral_to_negative"].idxmin()]
    print(f"\nLowest neutral->negative rate: {kindest['model']} at "
          f"{kindest['val_neutral_to_negative']:.1%}")
    if kindest["model"] != best["model"]:
        print(f"  -- not the best-scoring model ({best['model']} at "
              f"{best['val_neutral_to_negative']:.1%}).")

    # Is that low rate real caution, or just a reluctance to say "negative"?
    if kindest["val_negative_recall"] < best["val_negative_recall"] - 0.10:
        print(f"\n  ** BUT READ THE LAST COLUMN **")
        print(f"  {kindest['model']} only finds "
              f"{kindest['val_negative_recall']:.0%} of genuinely negative posts,")
        print(f"  against {best['val_negative_recall']:.0%} for {best['model']}. Its low")
        print("  neutral->negative rate is not restraint about jargon -- it is")
        print("  reluctance to predict 'negative' at all. A model that never")
        print("  says negative scores 0% on that metric and is worthless.")
        print("  Report the two columns together, never the first alone.")

    baseline_f1 = 0.7079  # VADER, from Stage 6.2
    lift = best["val_macro_f1"] - baseline_f1
    print(f"\nBest model vs the VADER baseline: {lift:+.4f} macro-F1")

    print(f"\nwrote {MODEL_RESULTS / f'tfidf_models_{domain}.csv'}")
    print(f"wrote {len(rows)} pipelines to {MODELS / domain}")


if __name__ == "__main__":
    main()
