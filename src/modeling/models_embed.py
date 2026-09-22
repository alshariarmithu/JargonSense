"""Stage 5.2 -- the six embedding models.

Run from the repository root:

    python -m src.modeling.models_embed

--------------------------------------------------------------------------
Two embeddings x three classifiers
--------------------------------------------------------------------------
The same three classifiers as the TF-IDF half, with one forced substitution.

**MultinomialNB cannot be used here.**  It models word *counts*, so it
requires non-negative features.  Embedding dimensions are freely positive and
negative -- half of every GloVe vector is below zero -- so the generative
model has to become ``GaussianNB``, which fits a normal distribution per
dimension per class instead.

That is not a workaround to hide in a footnote.  It is a real constraint:
choosing a dense representation changes which generative models are even
available, and GaussianNB's assumption (dimensions independent and normally
distributed within a class) fits embeddings poorly.  Expect it to do badly,
and report *why* rather than just reporting the number.

--------------------------------------------------------------------------
Why scaling appears here and not on the TF-IDF side
--------------------------------------------------------------------------
TF-IDF values are already bounded and comparable.  Embedding dimensions are
not: they have different ranges and variances, which distorts both the
regularisation penalty in Logistic Regression and the margin in an SVM.  A
``StandardScaler`` sits between the vectoriser and those two classifiers.

GaussianNB is left unscaled -- it estimates a mean and variance per dimension
anyway, so standardising first changes nothing.

--------------------------------------------------------------------------
Tuning strategy
--------------------------------------------------------------------------
The grid search runs on **pre-computed document vectors**, not through the
full pipeline.  Embedding the training split is expensive and identical for
every parameter combination, so doing it inside cross-validation would repeat
the same work dozens of times -- and with ``n_jobs=-1`` every worker process
would load its own copy of 400,000 GloVe vectors.

The winning parameters are then used to fit one complete raw-text pipeline,
which is what gets saved.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.extraction.clean_text import SEED
from src.evaluation.evaluate import evaluate
from src.features.features_embed import EMBEDDINGS, build_vectorizer
from src.modeling.models_tfidf import CV_FOLDS, MODEL_RESULTS, SCORING
from src.paths import MODELS
from src.preprocessing.normalize import load_chosen_mode
from src.preprocessing.splits import load_splits

CLASSIFIERS = ("gnb", "lr", "svm")


def make_classifier(name: str):
    """Return the classifier steps and the grid to search over them."""
    if name == "gnb":
        # No scaler: GaussianNB estimates a mean and variance per dimension,
        # so standardising first would change nothing.
        return (
            [("clf", GaussianNB())],
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        )

    if name == "lr":
        return (
            [("scale", StandardScaler()),
             ("clf", LogisticRegression(max_iter=2000, random_state=SEED))],
            {"clf__C": [0.01, 0.1, 1, 10],
             "clf__class_weight": [None, "balanced"]},
        )

    if name == "svm":
        return (
            [("scale", StandardScaler()),
             ("clf", CalibratedClassifierCV(
                 LinearSVC(random_state=SEED), cv=CV_FOLDS))],
            {"clf__estimator__C": [0.01, 0.1, 1, 10],
             "clf__estimator__class_weight": [None, "balanced"]},
        )

    raise ValueError(f"unknown classifier {name!r}")


def build_model(embedding: str, classifier: str,
                mode: str | None = None) -> Pipeline:
    """Vectoriser + optional scaler + classifier, accepting raw text."""
    vectorizer = build_vectorizer(embedding, mode=mode)
    steps, _ = make_classifier(classifier)
    return Pipeline([("embed", vectorizer), *steps])


def train_one(embedding: str, classifier: str, train_df, val_df,
              domain: str = "se", tune: bool = True) -> dict:
    """Tune on pre-computed vectors, then fit and save the full pipeline."""
    started = time.perf_counter()

    # Embed once.  Every parameter combination sees the same vectors.
    vectorizer = build_vectorizer(embedding)
    vectorizer.fit(train_df["text"])
    x_train = vectorizer.transform(train_df["text"])

    steps, grid = make_classifier(classifier)
    head = Pipeline(steps)

    if tune:
        search = GridSearchCV(
            head, grid,
            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED),
            scoring=SCORING, n_jobs=-1, refit=False,
        )
        search.fit(x_train, train_df["polarity"])
        best_params = search.best_params_
        cv_score = float(search.best_score_)
    else:
        best_params, cv_score = {}, float("nan")

    # Refit as one complete raw-text pipeline, which is what gets saved.
    model = build_model(embedding, classifier)
    if best_params:
        model.set_params(**best_params)
    model.fit(train_df["text"], train_df["polarity"])

    elapsed = time.perf_counter() - started

    name = f"{embedding}_{classifier}"
    metrics = evaluate(
        f"{name}_val", domain, val_df["polarity"], model.predict(val_df["text"]),
        out_dir="results/models", fig_dir="results/models/figures",
    )

    model_dir = MODELS / domain
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{name}.joblib")

    return {
        "model": name,
        "config": embedding,
        "classifier": classifier,
        "kind": "generative" if classifier == "gnb" else "discriminative",
        "best_params": json.dumps(
            {k.replace("clf__", ""): v for k, v in best_params.items()}),
        "cv_macro_f1": round(cv_score, 4),
        "val_accuracy": round(metrics["accuracy"], 4),
        "val_macro_f1": round(metrics["macro_f1"], 4),
        "val_neutral_to_negative": round(metrics["se_neutral_to_negative_rate"], 4),
        "val_negative_recall": round(metrics["per_class"]["negative"]["recall"], 4),
        "fit_seconds": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="se", choices=["se"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    domain = args.domain
    train_df, val_df, _ = load_splits()
    MODEL_RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Stage 5.2 -- embedding models on {domain.upper()}")
    print(f"  normalisation : {load_chosen_mode()} (Stage 3 ablation)")
    print(f"  train / val   : {len(train_df):,} / {len(val_df):,}")
    print(f"  note          : MultinomialNB is impossible on embeddings")
    print(f"                  (negative values) -- GaussianNB replaces it\n")

    rows = []
    for embedding in EMBEDDINGS:
        for classifier in CLASSIFIERS:
            print(f"  {embedding}_{classifier:4} ...", end="", flush=True)
            row = train_one(embedding, classifier, train_df, val_df,
                            domain=domain, tune=not args.quick)
            rows.append(row)
            print(f" val macro-F1 {row['val_macro_f1']:.4f}"
                  f"   ({row['fit_seconds']:.0f}s)")

    table = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    table.to_csv(MODEL_RESULTS / f"embed_models_{domain}.csv", index=False)

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
    print(f"\nBest embedding model: {best['model']} ({best['val_macro_f1']:.4f})"
          f"  params {best['best_params']}")

    by_embedding = table.groupby("config")["val_macro_f1"].max()
    print("\nPretrained vs self-trained (proposal outcome #1)")
    for embedding, score in by_embedding.items():
        label = "GloVe (general English)" if embedding == "glove" \
            else "Word2Vec (this corpus)"
        print(f"  {label:26} best {score:.4f}")

    gnb = table[table["classifier"] == "gnb"]["val_macro_f1"].max()
    others = table[table["classifier"] != "gnb"]["val_macro_f1"].max()
    print(f"\nGaussianNB best {gnb:.4f} vs discriminative best {others:.4f}")
    print("  GaussianNB assumes every dimension is independent and normally")
    print("  distributed within a class. Embedding dimensions are neither,")
    print("  which is why the generative model suffers more here than it did")
    print("  on TF-IDF. Worth a paragraph, not a footnote.")

    print(f"\nwrote {MODEL_RESULTS / f'embed_models_{domain}.csv'}")
    print(f"wrote {len(rows)} pipelines to {MODELS / domain}")


if __name__ == "__main__":
    main()
