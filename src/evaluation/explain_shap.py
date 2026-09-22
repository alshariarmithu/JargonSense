"""Generate model-agnostic SHAP explanations for the retained pipeline."""

from __future__ import annotations

import argparse

import joblib
import shap

from src.evaluation.explain_lime import choose_instances, load_lexicon
from src.extraction.clean_text import LABELS
from src.modeling.models_tfidf import MODEL_NAME
from src.paths import EXPLANATIONS, MODELS
from src.preprocessing.splits import load_splits


def text_shap(model, texts, max_evals: int = 300):
    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(model.predict_proba, masker, output_names=LABELS)
    return explainer(list(texts), max_evals=max_evals, silent=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-evals", type=int, default=300)
    args = parser.parse_args()

    _, _, test_df = load_splits()
    model = joblib.load(MODELS / "se" / f"{MODEL_NAME}.joblib")
    chosen = choose_instances(
        test_df, model.predict(test_df["text"]), load_lexicon()
    )
    values = text_shap(model, chosen["text"].tolist(), args.max_evals)
    out_dir = EXPLANATIONS / "shap" / "se"
    out_dir.mkdir(parents=True, exist_ok=True)

    negative_index = LABELS.index("negative")
    for position, (_, row) in enumerate(chosen.iterrows()):
        html = shap.plots.text(values[position, :, negative_index], display=False)
        name = f"{row['id']}_{row['group'].replace(' ', '_')}.html"
        (out_dir / name).write_text(html, encoding="utf-8")
    print(f"wrote {len(chosen)} SHAP explanations to {out_dir}")


if __name__ == "__main__":
    main()
