"""Stage 6.6 -- SHAP explanations.

Run from the repository root:

    python -m src.evaluation.explain_shap

--------------------------------------------------------------------------
SHAP next to LIME
--------------------------------------------------------------------------
Both answer "which words drove this prediction", but they get there
differently, and the difference is worth a paragraph in the report.

LIME fits a small linear model to random perturbations near one sentence.
Its answer is an approximation, and a different random seed gives a slightly
different answer.

SHAP distributes the prediction among the input words according to a rule
from cooperative game theory -- each word's contribution is its average
marginal effect across orderings.  For a **linear** model on TF-IDF features
this can be computed exactly and instantly, with no sampling and no seed.

That is why this module uses two explainers:

============================  ==============================================
``LinearExplainer``           Exact, for the Logistic Regression model.
                              Fast enough to run over the whole test split.
``Explainer`` + Text masker   Model-agnostic, for the calibrated SVM.  Works
                              on any pipeline, but has to sample, so it is
                              slow and is used only on the 20 fixed posts.
============================  ==============================================

--------------------------------------------------------------------------
Comparable with LIME by construction
--------------------------------------------------------------------------
The same 20 posts chosen in Stage 6.5 are explained here, and the token
aggregation is written in the same column format, so the two methods can be
compared directly rather than by eye.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import shap

from src.extraction.clean_text import LABELS, SEED, tokenize
from src.evaluation.explain_lime import choose_instances, load_lexicon
from src.paths import EXPLANATIONS, MODELS, RESULTS
from src.preprocessing.splits import load_splits

SHAP_DIR = EXPLANATIONS / "shap"
NEGATIVE_INDEX = LABELS.index("negative")

BACKGROUND_SIZE = 150


def linear_shap_tokens(model, texts) -> pd.DataFrame:
    """Exact SHAP values for a linear model on TF-IDF features.

    ``LinearExplainer`` works on the *transformed* matrix, so the feature
    indices have to be mapped back to n-gram names by hand.  In exchange the
    values are exact rather than sampled.
    """
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    names = np.array(vectorizer.get_feature_names_out())

    matrix = vectorizer.transform(texts)
    background = shap.sample(matrix, min(BACKGROUND_SIZE, matrix.shape[0]),
                             random_state=SEED)

    explainer = shap.LinearExplainer(classifier, background)
    values = explainer.shap_values(matrix)

    # Multi-class returns one matrix per class, or a 3-D array.
    if isinstance(values, list):
        negative = values[NEGATIVE_INDEX]
    else:
        negative = values[:, :, NEGATIVE_INDEX]

    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    dense = np.asarray(matrix.todense())

    for row in range(negative.shape[0]):
        present = np.nonzero(dense[row])[0]
        for column in present:
            totals[names[column]] += float(negative[row, column])
            counts[names[column]] += 1

    rows = [{
        "token": token,
        "occurrences": counts[token],
        "total_weight_toward_negative": round(totals[token], 4),
        "mean_weight_toward_negative": round(totals[token] / counts[token], 5),
    } for token in totals]
    return pd.DataFrame(rows).sort_values(
        "mean_weight_toward_negative", ascending=False)


def text_shap(model, texts, max_evals: int = 300):
    """Model-agnostic SHAP for a pipeline that is not a plain linear model."""
    masker = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(model.predict_proba, masker, output_names=LABELS)
    return explainer(list(texts), max_evals=max_evals, silent=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--linear-model", default="tfidf13_lr",
                        help="model explained exactly with LinearExplainer")
    parser.add_argument("--agnostic-model", default="tfidf13_svm",
                        help="model explained with the Text masker")
    parser.add_argument("--domain", default="se")
    parser.add_argument("--sample", type=int, default=250)
    args = parser.parse_args()

    domain = args.domain
    _, _, test_df = load_splits(domain)
    jargon = load_lexicon()

    out_dir = SHAP_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    explanations = RESULTS / "explanations"
    explanations.mkdir(parents=True, exist_ok=True)

    print(f"Stage 6.6 -- SHAP on {domain.upper()}")
    print(f"  exact (linear)   : {args.linear_model}")
    print(f"  agnostic (text)  : {args.agnostic_model}\n")

    # ------------------------------------------------- exact, whole test set
    linear_model = joblib.load(MODELS / domain / f"{args.linear_model}.joblib")
    neutral = test_df[test_df["polarity"] == "neutral"]
    sample = neutral.sample(min(args.sample, len(neutral)), random_state=SEED)

    print(f"  exact SHAP over {len(sample)} gold-neutral posts ...")
    tokens = linear_shap_tokens(linear_model, sample["text"].tolist())
    tokens.to_csv(explanations / f"shap_tokens_{domain}.csv", index=False)

    frequent = tokens[tokens["occurrences"] >= 3]
    print("\n  Tokens pushing hardest toward 'negative' on neutral posts")
    print(f"    {'token':20} {'n':>4} {'mean SHAP':>11}  jargon?")
    for _, row in frequent.head(15).iterrows():
        mark = "  <-- lexicon" if row["token"] in jargon else ""
        print(f"    {row['token']:20} {row['occurrences']:>4}"
              f" {row['mean_weight_toward_negative']:>11.4f}{mark}")

    # ------------------------------------------------------- jargon bias score
    lexicon_rows = frequent[frequent["token"].isin(jargon)]
    lexicon_rows.to_csv(explanations / f"shap_jargon_{domain}.csv", index=False)

    print(f"\n  Domain Bias Score -- mean SHAP toward negative for jargon words")
    if len(lexicon_rows):
        bias = lexicon_rows["mean_weight_toward_negative"].mean()
        overall = frequent["mean_weight_toward_negative"].mean()
        print(f"    jargon words ({len(lexicon_rows)}) : {bias:+.5f}")
        print(f"    all words     ({len(frequent)}) : {overall:+.5f}")
        print(f"    ratio                  : {bias / overall:.2f}x"
              if overall else "")
        for _, row in lexicon_rows.head(12).iterrows():
            print(f"      {row['token']:18} {row['occurrences']:>3}x"
                  f"  {row['mean_weight_toward_negative']:+.5f}")
    else:
        print("    No lexicon word occurs >=3 times in the neutral test posts.")
        print("    Consistent with Stage 4 and Stage 6.5: the jargon is not in")
        print("    this corpus, so the bias score cannot be computed from it.")
        print("    Stage 6.7's stress set exists precisely for this reason.")

    # ---------------------------------------------- the 20 fixed instances
    agnostic_model = joblib.load(MODELS / domain / f"{args.agnostic_model}.joblib")
    y_pred = agnostic_model.predict(test_df["text"])
    chosen = choose_instances(test_df, y_pred, jargon)

    print(f"\n  model-agnostic SHAP on the {len(chosen)} fixed posts ...")
    values = text_shap(agnostic_model, chosen["text"].tolist())

    for position, (_, row) in enumerate(chosen.iterrows()):
        html = shap.plots.text(values[position, :, NEGATIVE_INDEX], display=False)
        path = out_dir / f"{row['id']}_{row['group'].replace(' ', '_')}.html"
        path.write_text(html, encoding="utf-8")
    print(f"  wrote {len(chosen)} HTML explanations to {out_dir}")

    # ------------------------------------------------ LIME vs SHAP agreement
    lime_path = explanations / f"lime_tokens_{domain}.csv"
    if lime_path.exists():
        lime_tokens = pd.read_csv(lime_path)
        merged = lime_tokens.merge(tokens, on="token", suffixes=("_lime", "_shap"))
        merged = merged[(merged["occurrences_lime"] >= 3)
                        & (merged["occurrences_shap"] >= 3)]
        if len(merged) > 5:
            correlation = merged["mean_weight_toward_negative_lime"].corr(
                merged["mean_weight_toward_negative_shap"], method="spearman")
            merged.to_csv(explanations / f"lime_vs_shap_{domain}.csv", index=False)
            print(f"\n  LIME vs SHAP agreement on {len(merged)} shared tokens")
            print(f"    Spearman rank correlation: {correlation:.3f}")
            print("    The two methods use different mathematics, so agreement")
            print("    is evidence that both are describing the model rather")
            print("    than their own sampling noise.")

    print(f"\nwrote {explanations}")


if __name__ == "__main__":
    main()
