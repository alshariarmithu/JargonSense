"""Web interface for the SE sentiment project.

Run from the repository root:

    python -m app.server

then open http://127.0.0.1:5000

--------------------------------------------------------------------------
Two ways of explaining a prediction
--------------------------------------------------------------------------
For a **linear** model on TF-IDF features the explanation is exact and free.
The decision score for a class is a dot product, so each feature's
contribution is just ``coefficient x tfidf_value``.  Nothing is sampled and
nothing is approximated -- this is the same quantity SHAP computes for linear
models, available instantly.

For the calibrated SVM and the embedding models there is no such shortcut, so
LIME is used: it perturbs the sentence a few hundred times and fits a local
surrogate.  That takes a second or two, which is why the interface shows a
spinner for those models and defaults to a linear one.

--------------------------------------------------------------------------
Attributing n-gram weights back to words
--------------------------------------------------------------------------
A (1,3) model has features like ``kill the process``, but the interface
colours individual words.  Each active n-gram's contribution is therefore
split evenly across the word positions it covers.  A word's displayed weight
is the sum of every n-gram containing it -- so "kill" in a trigram model
carries part of the weight of ``kill``, ``kill the`` and ``kill the process``.
"""

from __future__ import annotations

import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.extraction.clean_text import LABELS, tokenize
from src.paths import RESULTS, MODELS, STRESS_TEST
from src.preprocessing.normalize import prepare

app = Flask(__name__, static_folder="static", static_url_path="")

DOMAIN = "se"
LIME_SAMPLES = 400          # enough to be stable, fast enough to feel live
DEFAULT_MODEL = "tfidf13_lr"

_models: dict = {}
_metrics: pd.DataFrame | None = None


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_everything() -> None:
    """Load every saved pipeline once, at startup."""
    global _metrics

    paths = sorted((MODELS / DOMAIN).glob("*.joblib"))
    for path in paths:
        print(f"  loading {path.stem} ...", flush=True)
        _models[path.stem] = joblib.load(path)

    master = RESULTS / "metrics" / f"master_results_{DOMAIN}.csv"
    _metrics = pd.read_csv(master) if master.exists() else pd.DataFrame()
    print(f"  {len(_models)} models ready")


@lru_cache(maxsize=1)
def vader():
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def vader_predict(text: str) -> dict:
    """The general-purpose baseline, shaped like a model response."""
    scores = vader().polarity_scores(text)
    compound = scores["compound"]
    label = ("positive" if compound >= 0.05
             else "negative" if compound <= -0.05 else "neutral")

    # VADER gives one signed score, not a distribution.  Spread it over the
    # three classes so the interface can draw comparable bars, and say so.
    strength = min(abs(compound), 1.0)
    if label == "negative":
        probabilities = [strength, 1 - strength, 0.0]
    elif label == "positive":
        probabilities = [0.0, 1 - strength, strength]
    else:
        probabilities = [0.0, 1.0, 0.0]

    lexicon = vader().lexicon
    tokens = []
    for token in tokenize(text.lower()):
        valence = lexicon.get(token, 0.0)
        tokens.append({"token": token, "weight": round(-valence / 4.0, 4)})

    return {
        "model": "vader",
        "label": label,
        "probabilities": [round(p, 4) for p in probabilities],
        "tokens": tokens,
        "method": "lexicon lookup",
        "note": "VADER returns one signed score, not a distribution; "
                "the bars are derived from it.",
    }


# --------------------------------------------------------------------------
# explanation
# --------------------------------------------------------------------------
def is_linear_tfidf(model) -> bool:
    """Can this model be explained exactly, without sampling?"""
    steps = getattr(model, "named_steps", {})
    return ("tfidf" in steps
            and isinstance(steps.get("clf"), (LogisticRegression, MultinomialNB)))


def exact_weights(model, text: str, class_index: int) -> list[dict]:
    """Per-word contribution from the model's own coefficients.

    Contributions are centred across classes, so a weight answers "how much
    does this word push toward *this* class rather than the others", which is
    what the colouring needs.
    """
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]

    matrix = vectorizer.transform([text])
    names = vectorizer.get_feature_names_out()
    coefficients = np.asarray(classifier.coef_)
    centred = coefficients - coefficients.mean(axis=0, keepdims=True)

    words = tokenize(prepare(text, domain=DOMAIN))
    per_word = np.zeros(len(words))

    for column in matrix.nonzero()[1]:
        contribution = float(matrix[0, column]) * float(centred[class_index, column])
        if contribution == 0.0:
            continue
        gram = names[column].split()
        positions = [i for i in range(len(words) - len(gram) + 1)
                     if words[i:i + len(gram)] == gram]
        if not positions:
            continue
        share = contribution / (len(gram) * len(positions))
        for start in positions:
            for offset in range(len(gram)):
                per_word[start + offset] += share

    return [{"token": word, "weight": round(float(weight), 5)}
            for word, weight in zip(words, per_word)]


def lime_weights(model, text: str, class_index: int) -> list[dict]:
    """Sampled explanation for models with no usable coefficients."""
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=LABELS, split_expression=tokenize,
                                  random_state=42, bow=True)
    explanation = explainer.explain_instance(
        text, model.predict_proba, num_features=15,
        num_samples=LIME_SAMPLES, labels=[class_index])
    weights = dict(explanation.as_list(label=class_index))
    return [{"token": token, "weight": round(float(weights.get(token, 0.0)), 5)}
            for token in tokenize(text)]


def predict(model_name: str, text: str) -> dict:
    if model_name == "vader":
        return vader_predict(text)

    model = _models[model_name]
    probabilities = model.predict_proba([text])[0]
    label = model.predict([text])[0]

    # Explain the predicted class, not always "negative" -- the user wants to
    # know why they got the answer they got.
    class_index = list(model.classes_).index(label)

    if is_linear_tfidf(model):
        tokens = exact_weights(model, text, class_index)
        method = "exact (model coefficients)"
    else:
        tokens = lime_weights(model, text, class_index)
        method = f"LIME ({LIME_SAMPLES} samples)"

    return {
        "model": model_name,
        "label": label,
        "probabilities": [round(float(p), 4) for p in probabilities],
        "explained_class": label,
        "tokens": tokens,
        "method": method,
    }


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/models")
def api_models():
    """Every model, richest first, annotated with its test score."""
    lookup = {}
    if _metrics is not None and len(_metrics):
        lookup = _metrics.set_index("model").to_dict(orient="index")

    rows = [{
        "name": "vader",
        "representation": "lexicon",
        "kind": "baseline",
        "macro_f1": 0.7079,
        "exact": False,
        "fast": True,
    }]
    for name in sorted(_models):
        info = lookup.get(name, {})
        rows.append({
            "name": name,
            "representation": info.get("representation", "?"),
            "kind": info.get("kind", "?"),
            "macro_f1": info.get("macro_f1"),
            "neutral_to_negative": info.get("neutral_to_negative"),
            "exact": is_linear_tfidf(_models[name]),
            "fast": is_linear_tfidf(_models[name]),
        })
    rows.sort(key=lambda r: (r["macro_f1"] is None, -(r["macro_f1"] or 0)))
    return jsonify({"models": rows, "default": DEFAULT_MODEL, "labels": LABELS})


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip()
    name = payload.get("model", DEFAULT_MODEL)

    if not text:
        return jsonify({"error": "Type a sentence first."}), 400
    if name != "vader" and name not in _models:
        return jsonify({"error": f"Unknown model {name!r}."}), 400

    return jsonify(predict(name, text))


@app.post("/api/compare")
def api_compare():
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip()
    names = payload.get("models") or ["vader", "glove_lr", "tfidf13_lr"]

    if not text:
        return jsonify({"error": "Type a sentence first."}), 400

    results = [predict(name, text) for name in names
               if name == "vader" or name in _models]
    return jsonify({"text": text, "results": results})


@app.get("/api/results")
def api_results():
    if _metrics is None or not len(_metrics):
        return jsonify({"rows": []})
    return jsonify({
        "rows": _metrics.to_dict(orient="records"),
        "baselines": {"majority": 0.1852, "vader": 0.7079},
    })


@app.get("/api/stress")
def api_stress():
    summary = pd.read_csv(RESULTS / "stress_test" / f"stress_results_{DOMAIN}.csv")
    sentences = pd.read_csv(STRESS_TEST / f"{DOMAIN}_stress.csv", sep=";")
    sentences["lexicon_word"] = sentences["lexicon_word"].fillna("")
    sentences["pair_id"] = sentences["pair_id"].fillna(0).astype(int)

    predictions_path = (RESULTS / "stress_test" /
                        "stress_predictions_glove_gnb.csv")
    if predictions_path.exists():
        detail = pd.read_csv(predictions_path)
        sentences = sentences.merge(
            detail[["id", "predicted", "correct"]], on="id", how="left")

    family = {"glove": "GloVe (general English)", "w2v": "Word2Vec (this corpus)"}
    trained = summary[summary["model"] != "vader"].copy()
    trained["family"] = trained["model"].map(
        lambda m: family.get(m.split("_")[0], "TF-IDF (this corpus)"))
    by_family = trained.groupby("family")["false_alarm_rate"].mean().to_dict()
    by_family["VADER (general lexicon)"] = float(
        summary[summary["model"] == "vader"]["false_alarm_rate"].iloc[0])

    return jsonify({
        "summary": summary.to_dict(orient="records"),
        "sentences": sentences.fillna("").to_dict(orient="records"),
        "false_alarm_by_family": by_family,
    })


@app.get("/api/findings")
def api_findings():
    explanations = RESULTS / "explanations"
    features = RESULTS / "features"

    def read(path, **kwargs):
        return (pd.read_csv(path, **kwargs).fillna("").to_dict(orient="records")
                if path.exists() else [])

    agreement = []
    path = explanations / f"lime_vs_shap_{DOMAIN}.csv"
    if path.exists():
        frame = pd.read_csv(path)
        agreement = frame[["token",
                           "mean_weight_toward_negative_lime",
                           "mean_weight_toward_negative_shap"]].to_dict("records")

    ablation = read(RESULTS / "ablation" / f"normalization_{DOMAIN}.csv")
    return jsonify({
        "neighbours": read(features / f"neighbours_{DOMAIN}.csv"),
        "compound_phrases": read(features / f"compound_phrases_{DOMAIN}.csv"),
        "oov": read(features / f"oov_{DOMAIN}.csv"),
        "ablation": ablation,
        "lime_tokens": read(explanations / f"lime_tokens_{DOMAIN}.csv")[:25],
        "shap_tokens": read(explanations / f"shap_tokens_{DOMAIN}.csv")[:25],
        "agreement": agreement,
        "error_analysis": read(
            RESULTS / "error_analysis" / "neutral_as_negative_tfidf13_svm.csv"),
    })


EXAMPLES = [
    {"kind": "SE jargon, neutral",
     "text": "Kill the process before restarting the server."},
    {"kind": "SE jargon, neutral",
     "text": "The build failed with a fatal error on line 42."},
    {"kind": "SE jargon, neutral",
     "text": "Garbage collection destroys unused objects automatically."},
    {"kind": "SE jargon, neutral",
     "text": "A deadlock occurs when two threads wait on each other."},
    {"kind": "Genuine complaint",
     "text": "The documentation is worthless and the maintainers ignore every issue."},
    {"kind": "Genuine complaint",
     "text": "I regret choosing this library, it has been nothing but trouble."},
    {"kind": "Genuine praise",
     "text": "Thanks, this works perfectly and saved me hours of debugging."},
    {"kind": "Neutral question",
     "text": "How do I use this function with a custom comparator?"},
]


@app.get("/api/examples")
def api_examples():
    return jsonify({"examples": EXAMPLES})


def find_free_port(preferred: int) -> int:
    """Return ``preferred`` if it can be bound, otherwise the next free port.

    Windows reserves blocks of ports for Hyper-V and WSL, and 5000 is often
    inside one -- binding it fails with "access forbidden" rather than
    "address in use", which is confusing.  Probing first avoids that.
    """
    import socket

    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {preferred} and {preferred + 19}.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    port = find_free_port(args.port)

    print("Loading models (GloVe is shared, this takes a moment) ...")
    load_everything()
    print(f"\n  Interface ready -> http://{args.host}:{port}\n")
    app.run(host=args.host, port=port, debug=False)


if __name__ == "__main__":
    main()
