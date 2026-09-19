"""Contract tests for the retained TF-IDF + calibrated SVM pipeline."""

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.extraction.clean_text import LABELS
from src.features.features_tfidf import NGRAM_RANGE
from src.modeling.models_tfidf import MODEL_NAME, build_model, make_classifier

TEXTS = [
    "thanks this works great", "excellent library thanks", "great work thanks",
    "this is excellent", "thanks this is great", "excellent thanks",
    "this is terrible i hate it", "horrible broken library", "terrible and broken",
    "i hate this thing", "this is terrible", "horrible broken thing",
    "how do i use this function", "what does this method return", "how do i call this",
    "what is the return type", "how does this work", "what does this do",
]
POLARITY = ["positive"] * 6 + ["negative"] * 6 + ["neutral"] * 6


def tiny_model():
    model = build_model(mode="none")
    model.named_steps["tfidf"].set_params(min_df=1, max_df=1.0)
    return model.fit(TEXTS, POLARITY)


def test_only_retained_model_name():
    assert MODEL_NAME == "tfidf13_svm"


def test_pipeline_shape_and_configuration():
    model = build_model(mode="none")
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["tfidf", "clf"]
    assert model.named_steps["tfidf"].ngram_range == NGRAM_RANGE == (1, 3)
    assert isinstance(model.named_steps["clf"], CalibratedClassifierCV)


def test_classifier_exposes_probability_output():
    assert hasattr(make_classifier(), "predict_proba")


def test_pipeline_predicts_raw_text_and_probabilities():
    model = tiny_model()
    text = ["Kill the PROCESS via <code>foo()</code>!"]
    assert model.predict(text)[0] in LABELS
    probabilities = model.predict_proba(text)
    assert probabilities.shape == (1, len(LABELS))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert list(model.classes_) == LABELS


def test_pipeline_survives_save_load_roundtrip(tmp_path):
    model = tiny_model()
    path = tmp_path / f"{MODEL_NAME}.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    text = ["thanks this works great"]
    assert reloaded.predict(text) == model.predict(text)
    assert np.allclose(reloaded.predict_proba(text), model.predict_proba(text))


def test_pipeline_learns_obvious_sentiment():
    model = tiny_model()
    assert model.predict(["thanks this is excellent"])[0] == "positive"
    assert model.predict(["i hate this horrible broken thing"])[0] == "negative"
