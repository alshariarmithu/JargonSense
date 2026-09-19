"""Tests for src/modeling/models_tfidf.py (Stage 5.1).

Run from the repository root:  python -m pytest -q
"""

import joblib
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.extraction.clean_text import LABELS
from src.modeling.models_tfidf import CLASSIFIERS, build_model, make_classifier

# Enough documents per class for CalibratedClassifierCV's internal 5 folds.
TEXTS = [
    "thanks this works great", "excellent library thanks a lot",
    "great work thanks", "this is excellent thanks",
    "thanks this is great", "excellent thanks",
    "this is terrible i hate it", "horrible broken library i hate this",
    "terrible and broken i hate it", "i hate this horrible thing",
    "this is terrible", "horrible broken thing",
    "how do i use this function", "what does this method return",
    "how do i call this", "what is the return type",
    "how does this work", "what does this do",
]
POLARITY = (["positive"] * 6) + (["negative"] * 6) + (["neutral"] * 6)


def tiny_model(classifier, config="tfidf11"):
    """A model small enough to train inside a test."""
    model = build_model(config, classifier, domain="se", mode="none")
    model.named_steps["tfidf"].set_params(min_df=1, max_df=1.0)
    return model.fit(TEXTS, POLARITY)


# --------------------------------------------------------------------------
# the classifiers the proposal commits to
# --------------------------------------------------------------------------
def test_naive_bayes_is_the_generative_model():
    estimator, grid = make_classifier("mnb")
    assert isinstance(estimator, MultinomialNB)
    assert "clf__alpha" in grid


def test_logistic_regression_tunes_regularisation():
    """Stage 4 found a train/validation gap of 0.13-0.20, so C matters."""
    estimator, grid = make_classifier("lr")
    assert isinstance(estimator, LogisticRegression)
    assert grid["clf__C"] == [0.01, 0.1, 1, 10, 100]
    assert None in grid["clf__class_weight"]


def test_svm_is_wrapped_for_probabilities():
    """LinearSVC has no predict_proba, which LIME, SHAP and the demo need."""
    estimator, grid = make_classifier("svm")
    assert isinstance(estimator, CalibratedClassifierCV)
    # The grid must reach through the wrapper to the LinearSVC inside.
    assert all(key.startswith("clf__estimator__") for key in grid)


def test_unknown_classifier_is_rejected():
    with pytest.raises(ValueError, match="unknown classifier"):
        make_classifier("randomforest")


# --------------------------------------------------------------------------
# the pipeline contract Stage 6 depends on
# --------------------------------------------------------------------------
def test_model_is_a_two_step_pipeline():
    model = build_model("tfidf12", "lr")
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["tfidf", "clf"]


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_every_model_predicts_from_raw_text(classifier):
    """No caller may be required to clean text first.

    LIME perturbs raw sentences, and the demo passes whatever a user types.
    If cleaning lived outside the model they would be explaining something
    the model never saw.
    """
    model = tiny_model(classifier)
    prediction = model.predict(["Kill the PROCESS via <code>foo()</code>!"])
    assert prediction[0] in LABELS


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_every_model_exposes_calibrated_probabilities(classifier):
    model = tiny_model(classifier)
    probabilities = model.predict_proba(["thanks this is great", "i hate this"])
    assert probabilities.shape == (2, len(LABELS))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0).all()


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_class_order_matches_the_project_label_order(classifier):
    """LIME and SHAP index into predict_proba by position, not by name."""
    model = tiny_model(classifier)
    assert list(model.classes_) == LABELS


@pytest.mark.parametrize("classifier", CLASSIFIERS)
def test_model_survives_a_save_and_load_roundtrip(tmp_path, classifier):
    model = tiny_model(classifier)
    path = tmp_path / f"{classifier}.joblib"
    joblib.dump(model, path)

    reloaded = joblib.load(path)
    sentence = ["thanks this works great"]
    assert reloaded.predict(sentence) == model.predict(sentence)
    assert np.allclose(reloaded.predict_proba(sentence),
                       model.predict_proba(sentence))


def test_model_learns_the_obvious_signal():
    """A sanity check that the pipeline is wired up, not just runnable."""
    model = tiny_model("lr")
    assert model.predict(["thanks this is excellent"])[0] == "positive"
    assert model.predict(["i hate this horrible broken thing"])[0] == "negative"


def test_naive_bayes_receives_non_negative_features():
    """MultinomialNB requires non-negative input.

    TF-IDF satisfies this, which is why MNB works here and why the embedding
    models in Stage 5.2 must use GaussianNB instead.
    """
    model = tiny_model("mnb")
    matrix = model.named_steps["tfidf"].transform(TEXTS)
    assert matrix.min() >= 0
