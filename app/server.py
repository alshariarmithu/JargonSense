"""Web interface for the retained SE sentiment pipeline.

Run ``python -m app.server`` from the repository root. Predictions use the
saved ``tfidf13_svm`` pipeline and LIME supplies local token explanations.
"""

from __future__ import annotations

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from src.extraction.clean_text import LABELS, tokenize
from src.paths import RESULTS, MODELS, STRESS_TEST
from src.modeling.models_tfidf import MODEL_NAME

app = Flask(__name__, static_folder="static", static_url_path="")

DOMAIN = "se"
LIME_SAMPLES = 400          # enough to be stable, fast enough to feel live
DEFAULT_MODEL = MODEL_NAME

_models: dict = {}
_metrics: pd.DataFrame | None = None


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_everything() -> None:
    """Load every saved pipeline once, at startup."""
    global _metrics

    path = MODELS / DOMAIN / f"{MODEL_NAME}.joblib"
    if path.exists():
        print(f"  loading {path.stem} ...", flush=True)
        _models[path.stem] = joblib.load(path)

    master = RESULTS / "metrics" / f"master_results_{DOMAIN}.csv"
    _metrics = pd.read_csv(master) if master.exists() else pd.DataFrame()
    print(f"  {len(_models)} models ready")


# --------------------------------------------------------------------------
# explanation
# --------------------------------------------------------------------------
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
    model = _models[model_name]
    probabilities = model.predict_proba([text])[0]
    label = model.predict([text])[0]

    # Explain the predicted class, not always "negative" -- the user wants to
    # know why they got the answer they got.
    class_index = list(model.classes_).index(label)

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

    rows = []
    for name in sorted(_models):
        info = lookup.get(name, {})
        rows.append({
            "name": name,
            "representation": info.get("representation", "?"),
            "kind": info.get("kind", "?"),
            "macro_f1": info.get("macro_f1"),
            "neutral_to_negative": info.get("neutral_to_negative"),
            "exact": False,
            "fast": False,
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
    if name not in _models:
        return jsonify({"error": f"Unknown model {name!r}."}), 400

    return jsonify(predict(name, text))


@app.get("/api/results")
def api_results():
    if _metrics is None or not len(_metrics):
        return jsonify({"rows": []})
    return jsonify({
        "rows": _metrics.to_dict(orient="records"),
        "baselines": {},
    })


@app.get("/api/stress")
def api_stress():
    summary = pd.read_csv(RESULTS / "stress_test" / f"stress_results_{DOMAIN}.csv")
    sentences = pd.read_csv(STRESS_TEST / f"{DOMAIN}_stress.csv", sep=";")
    sentences["lexicon_word"] = sentences["lexicon_word"].fillna("")
    sentences["pair_id"] = sentences["pair_id"].fillna(0).astype(int)

    predictions_path = (RESULTS / "stress_test" /
                        f"stress_predictions_{MODEL_NAME}.csv")
    if predictions_path.exists():
        detail = pd.read_csv(predictions_path)
        sentences = sentences.merge(
            detail[["id", "predicted", "correct"]], on="id", how="left")

    by_family = {
        "TF-IDF (1,3) + calibrated Linear SVM":
            float(summary.iloc[0]["false_alarm_rate"])
    } if len(summary) else {}

    return jsonify({
        "summary": summary.to_dict(orient="records"),
        "sentences": sentences.fillna("").to_dict(orient="records"),
        "false_alarm_by_family": by_family,
    })


# Each label must be distinct: the interface renders one card per example, and
# four cards reading "SE jargon" are four identical buttons to the reader.
EXAMPLES = [
    {"label": "kill", "kind": "SE jargon", "expected": "neutral",
     "text": "Kill the process before restarting the server."},
    {"label": "fatal error", "kind": "SE jargon", "expected": "neutral",
     "text": "The build failed with a fatal error on line 42."},
    {"label": "destroys", "kind": "SE jargon", "expected": "neutral",
     "text": "Garbage collection destroys unused objects automatically."},
    {"label": "deadlock", "kind": "SE jargon", "expected": "neutral",
     "text": "A deadlock occurs when two threads wait on each other."},
    {"label": "real complaint", "kind": "Genuine sentiment", "expected": "negative",
     "text": "The documentation is worthless and the maintainers ignore every issue."},
    {"label": "regret", "kind": "Genuine sentiment", "expected": "negative",
     "text": "I regret choosing this library, it has been nothing but trouble."},
    {"label": "thanks", "kind": "Genuine sentiment", "expected": "positive",
     "text": "Thanks, this works perfectly and saved me hours of debugging."},
    {"label": "plain question", "kind": "Ordinary prose", "expected": "neutral",
     "text": "How do I use this function with a custom comparator?"},

    # Held-out probes.  Every jargon word below occurs ZERO times in the 3,031
    # training documents -- zombie, reap, poison, watchdog, starvation, orphan,
    # panic, abort, deadlock, scheduler.  Nothing here comes from Senti4SD or
    # from the stress set either, so these are the only examples in the app the
    # pipeline has no exposure to whatsoever.  They are the honest demo: the
    # first four show the model staying neutral on vocabulary it never learned,
    # and the last four show where it still fails.
    {"label": "zombie", "kind": "Held-out jargon", "expected": "neutral",
     "text": "The parent process must reap its zombie children or the table fills up."},
    {"label": "poison pill", "kind": "Held-out jargon", "expected": "neutral",
     "text": "A poison pill message forces the consumer thread to terminate cleanly."},
    {"label": "watchdog", "kind": "Held-out jargon", "expected": "neutral",
     "text": "The watchdog timer aborts any request that hangs for over thirty seconds."},
    {"label": "kernel panic", "kind": "Held-out jargon", "expected": "neutral",
     "text": "The kernel panic dumped a core file to the crash directory."},

    {"label": "abort · calm", "kind": "Held-out sentiment", "expected": "neutral",
     "text": "The service aborted the transaction and rolled back cleanly."},
    {"label": "abort · angry", "kind": "Held-out sentiment", "expected": "negative",
     "text": "The service aborted my transaction again and lost two hours of work."},
    {"label": "corruption", "kind": "Held-out sentiment", "expected": "negative",
     "text": "Another silent data corruption bug, and the team still refuses to write tests."},
    {"label": "praise", "kind": "Held-out sentiment", "expected": "positive",
     "text": "This patch fixed the deadlock immediately, brilliant work on the lock ordering."},
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

    print("Loading retained pipeline ...")
    load_everything()
    print(f"\n  Interface ready -> http://{args.host}:{port}\n")
    app.run(host=args.host, port=port, debug=False)


if __name__ == "__main__":
    main()
