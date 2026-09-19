"""Stage 6.5 -- LIME explanations.

Run from the repository root:

    python -m src.evaluation.explain_lime                 # default settings
    python -m src.evaluation.explain_lime --sample 150    # smaller aggregate

--------------------------------------------------------------------------
What LIME does, and why the answer is only approximate
--------------------------------------------------------------------------
LIME takes one sentence, generates thousands of copies with random words
deleted, asks the model to score each, and fits a tiny linear model to the
result.  The weights of that linear model are the explanation: "removing this
word moved the prediction toward neutral by this much".

Two consequences the report must state:

1. It is a **local approximation**.  The weights describe the model's
   behaviour near one sentence, not its overall logic.
2. It is **random**.  Different seeds give slightly different weights, which
   is why ``random_state`` is fixed here and why the proposal lists LIME's
   stability as a limitation.

--------------------------------------------------------------------------
The question this stage answers
--------------------------------------------------------------------------
Proposal Expected Outcome #3: *do models rely on domain-appropriate cues or
on general-language bias?*

The test is the jargon table.  For every word in ``data/lexicons/se_jargon.txt``,
measure its average LIME weight toward "negative" **on posts whose gold label
is neutral**.  A large positive number means the model reads ordinary
technical vocabulary as hostility -- which is the project's central claim,
stated as a number per word.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

from src.extraction.clean_text import LABELS, SEED, tokenize
from src.paths import EXPLANATIONS, LEXICONS, MODELS, RESULTS
from src.preprocessing.splits import load_splits

LIME_DIR = EXPLANATIONS / "lime"
NEGATIVE_INDEX = LABELS.index("negative")

# 2000 is LIME's own default.  Lowering it speeds the aggregate run up but
# makes individual explanations noisier; the trade-off is measured in 6.5.3.
NUM_SAMPLES = 1000
NUM_FEATURES = 10


def load_lexicon() -> set[str]:
    path = LEXICONS / "se_jargon.txt"
    return {line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def make_explainer(seed: int = SEED) -> LimeTextExplainer:
    """An explainer that splits text exactly as the model's vectoriser does.

    ``split_expression`` is handed the project's ``tokenize`` **function**,
    not ``TOKEN_PATTERN`` itself.  That distinction matters: LIME treats a
    string here as a pattern to *split on* and keeps everything between the
    matches, so passing the token pattern produces "tokens" like ``,`` and
    ``'`` -- punctuation the vectoriser never turns into a feature.  The
    explanation then describes a model that does not exist.

    Passing a callable makes LIME use its output as the token list directly,
    so LIME perturbs exactly the units the model was trained on.
    """
    return LimeTextExplainer(
        class_names=LABELS,
        split_expression=tokenize,
        random_state=seed,
        bow=True,
    )


def choose_instances(test_df, y_pred, jargon: set[str], per_group: int = 5):
    """The 20 posts explained in detail, chosen to cover four situations.

    Fixed and shared with SHAP in Stage 6.6, so the two methods can be
    compared on identical inputs.
    """
    frame = test_df.copy()
    frame["predicted"] = y_pred
    frame["correct"] = frame["polarity"] == frame["predicted"]
    frame["has_jargon"] = frame["text"].map(
        lambda t: bool(set(re.findall(r"[a-z_]+", t.lower())) & jargon))

    groups = {
        "correct negative": frame[frame["correct"] & (frame["polarity"] == "negative")],
        "correct neutral": frame[frame["correct"] & (frame["polarity"] == "neutral")],
        "error": frame[(frame["polarity"] == "neutral")
                       & (frame["predicted"] == "negative")],
        "jargon-heavy": frame[frame["has_jargon"] & (frame["polarity"] == "neutral")],
    }

    chosen = []
    for group, subset in groups.items():
        picked = subset.head(per_group).copy()
        picked["group"] = group
        chosen.append(picked)
    return pd.concat(chosen).reset_index(drop=True)


def explain_one(model, explainer, text: str):
    return explainer.explain_instance(
        text, model.predict_proba,
        num_features=NUM_FEATURES, num_samples=NUM_SAMPLES,
        labels=list(range(len(LABELS))),
    )


def aggregate_weights(model, texts, labels, seed: int = SEED) -> pd.DataFrame:
    """Average LIME weight toward 'negative' for every token, over a sample."""
    explainer = make_explainer(seed)
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for index, text in enumerate(texts, start=1):
        explanation = explain_one(model, explainer, text)
        for token, weight in explanation.as_list(label=NEGATIVE_INDEX):
            totals[token.lower()] += weight
            counts[token.lower()] += 1
        if index % 25 == 0:
            print(f"    {index}/{len(texts)}", end="\r", flush=True)

    rows = [{
        "token": token,
        "occurrences": counts[token],
        "total_weight_toward_negative": round(totals[token], 4),
        "mean_weight_toward_negative": round(totals[token] / counts[token], 5),
    } for token in totals]
    return pd.DataFrame(rows).sort_values(
        "mean_weight_toward_negative", ascending=False)


def stability(model, texts, seeds=(1, 2, 3, 4, 5), top_k: int = 5) -> dict:
    """How much the top tokens change when only the random seed changes.

    Reported as mean Jaccard overlap between seed pairs.  1.0 would mean LIME
    is deterministic; anything much below that limits how strongly individual
    explanations can be interpreted.
    """
    overlaps = []
    for text in texts:
        token_sets = []
        for seed in seeds:
            explanation = explain_one(model, make_explainer(seed), text)
            ranked = sorted(explanation.as_list(label=NEGATIVE_INDEX),
                            key=lambda pair: -abs(pair[1]))
            token_sets.append({t.lower() for t, _ in ranked[:top_k]})
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                union = token_sets[i] | token_sets[j]
                if union:
                    overlaps.append(len(token_sets[i] & token_sets[j]) / len(union))
    return {
        "top_k": top_k,
        "seeds": len(seeds),
        "mean_jaccard": round(float(np.mean(overlaps)), 4),
        "std_jaccard": round(float(np.std(overlaps)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="tfidf13_svm")
    parser.add_argument("--domain", default="se")
    parser.add_argument("--sample", type=int, default=150,
                        help="test posts used for the aggregate table")
    args = parser.parse_args()

    domain = args.domain
    _, _, test_df = load_splits(domain)
    jargon = load_lexicon()

    model = joblib.load(MODELS / domain / f"{args.model}.joblib")
    out_dir = LIME_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage 6.5 -- LIME on {args.model}")
    print(f"  num_samples per explanation : {NUM_SAMPLES}")
    print(f"  aggregate sample            : {args.sample} test posts\n")

    y_pred = model.predict(test_df["text"])

    # ------------------------------------------------- the 20 fixed instances
    chosen = choose_instances(test_df, y_pred, jargon)
    chosen.to_csv(RESULTS / "explanations" / f"chosen_instances_{domain}.csv",
                  index=False)

    explainer = make_explainer()
    print(f"  explaining {len(chosen)} chosen posts ...")
    for _, row in chosen.iterrows():
        explanation = explain_one(model, explainer, row["text"])
        path = out_dir / f"{row['id']}_{row['group'].replace(' ', '_')}.html"
        explanation.save_to_file(str(path))
    print(f"  wrote {len(chosen)} HTML explanations to {out_dir}")

    # ------------------------------------------------------------- aggregate
    neutral = test_df[test_df["polarity"] == "neutral"]
    sample = neutral.sample(min(args.sample, len(neutral)), random_state=SEED)
    print(f"\n  aggregating over {len(sample)} gold-neutral posts ...")
    weights = aggregate_weights(model, sample["text"].tolist(),
                                sample["polarity"].tolist())
    weights.to_csv(RESULTS / "explanations" / f"lime_tokens_{domain}.csv",
                   index=False)

    print("\n  Tokens pushing hardest toward 'negative' on neutral posts")
    print(f"    {'token':18} {'n':>4} {'mean weight':>12}  jargon?")
    for _, row in weights[weights["occurrences"] >= 3].head(15).iterrows():
        mark = "  <-- lexicon" if row["token"] in jargon else ""
        print(f"    {row['token']:18} {row['occurrences']:>4}"
              f" {row['mean_weight_toward_negative']:>12.4f}{mark}")

    # -------------------------------------------------------- jargon table
    lexicon_rows = weights[weights["token"].isin(jargon)
                           & (weights["occurrences"] >= 3)]
    lexicon_rows.to_csv(RESULTS / "explanations" / f"lime_jargon_{domain}.csv",
                        index=False)

    print(f"\n  Jargon lexicon words appearing >=3 times: {len(lexicon_rows)}")
    if len(lexicon_rows):
        mean_jargon = lexicon_rows["mean_weight_toward_negative"].mean()
        mean_all = weights[weights["occurrences"] >= 3][
            "mean_weight_toward_negative"].mean()
        print(f"    mean weight toward negative, jargon words : {mean_jargon:+.4f}")
        print(f"    mean weight toward negative, all words    : {mean_all:+.4f}")
        for _, row in lexicon_rows.iterrows():
            print(f"      {row['token']:16} {row['occurrences']:>3}x"
                  f"  {row['mean_weight_toward_negative']:+.4f}")
    else:
        print("    Too few jargon words survive the >=3 occurrence filter to")
        print("    support a per-word table. This follows from Stage 4: the")
        print("    vocabulary is largely absent from the corpus. The stress")
        print("    set in Stage 6.7 is where this question gets answered.")

    # -------------------------------------------------------------- stability
    print("\n  stability across 5 random seeds ...")
    stats = stability(model, chosen["text"].head(8).tolist())
    print(f"    mean Jaccard overlap of top-{stats['top_k']} tokens: "
          f"{stats['mean_jaccard']:.3f} (sd {stats['std_jaccard']:.3f})")
    print("    1.0 would mean LIME is deterministic. Anything well below")
    print("    that limits how firmly a single explanation can be read.")

    pd.DataFrame([stats]).to_csv(
        RESULTS / "explanations" / f"lime_stability_{domain}.csv", index=False)

    print(f"\nwrote {RESULTS / 'explanations'}")


if __name__ == "__main__":
    main()
