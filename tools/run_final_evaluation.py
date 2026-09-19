"""Stage 6.3-6.4 -- the one and only run on the test split.

Run from the repository root:

    python -m tools.run_final_evaluation

--------------------------------------------------------------------------
This script spends the test set
--------------------------------------------------------------------------
Everything up to now -- the normalisation ablation, the vectoriser tuning,
the classifier grids, the choice of best model -- was decided on training and
validation data.  The test split has not been read.

It is read here, once, by every saved model.  Re-running this script after
changing anything upstream means the reported numbers are no longer an honest
estimate of unseen performance, because choices will have been influenced by
test results.  If that happens, say so in the report.

--------------------------------------------------------------------------
What it produces
--------------------------------------------------------------------------
    results/metrics/all_results.csv        every model, one row
    results/metrics/{domain}_{model}.json  full metrics per model
    results/figures/cm_{domain}_{model}.png
    results/figures/summary_*.png          three comparison charts
    results/error_analysis/                misclassified test posts, tagged
"""

from __future__ import annotations

import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.extraction.clean_text import LABELS
from src.evaluation.evaluate import (append_to_results_table, evaluate,
                                     mcnemar_test)
from src.features.features_tfidf import CONFIGS
from src.features.features_embed import EMBEDDINGS
from src.modeling.models_tfidf import CLASSIFIERS as TFIDF_CLASSIFIERS
from src.modeling.models_embed import CLASSIFIERS as EMBED_CLASSIFIERS
from src.paths import FIGURES, LEXICONS, MODELS, RESULTS
from src.preprocessing.splits import load_splits

ERROR_ANALYSIS = RESULTS / "error_analysis"

REPRESENTATION_OF = {
    **{c: "TF-IDF" for c in CONFIGS},
    "glove": "GloVe",
    "w2v": "Word2Vec",
}


def load_lexicon() -> set[str]:
    """The SE jargon words, one per line, comments stripped."""
    path = LEXICONS / "se_jargon.txt"
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line.lower())
    return words


def model_names() -> list[str]:
    """Every model Stage 5 saved, TF-IDF first."""
    names = [f"{c}_{clf}" for c in CONFIGS for clf in TFIDF_CLASSIFIERS]
    names += [f"{e}_{clf}" for e in EMBEDDINGS for clf in EMBED_CLASSIFIERS]
    return names


def tag_error(text: str, jargon: set[str]) -> str:
    """Classify why one neutral post was predicted negative.

    Deliberately coarse.  The categories separate "the model was misled by
    technical vocabulary" -- the project's claim -- from the ordinary reasons
    any sentiment model gets things wrong.
    """
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z_]+", lowered))

    if tokens & jargon:
        return "contains SE jargon"
    if "?" in text and len(text.split()) < 30:
        return "short question"
    if any(marker in lowered for marker in
           ("but ", "however", "although", "though")):
        return "mixed or contrastive"
    return "other"


def main() -> None:
    domain = "se"
    _, _, test_df = load_splits(domain)
    jargon = load_lexicon()
    ERROR_ANALYSIS.mkdir(parents=True, exist_ok=True)

    print(f"Stage 6.3 -- final evaluation on {domain.upper()}")
    print(f"  test split : {len(test_df):,} documents, read once")
    print(f"  models     : {len(model_names())} saved pipelines + 2 baselines\n")

    y_true = test_df["polarity"].tolist()
    predictions = {}
    rows = []

    for name in model_names():
        path = MODELS / domain / f"{name}.joblib"
        if not path.exists():
            print(f"  {name:14} MISSING -- run Stage 5 first")
            continue

        model = joblib.load(path)
        y_pred = model.predict(test_df["text"]).tolist()
        predictions[name] = y_pred

        metrics = evaluate(name, domain, y_true, y_pred)
        append_to_results_table(name, domain, metrics)

        representation = REPRESENTATION_OF[name.split("_")[0]]
        classifier = name.split("_")[1]
        rows.append({
            "model": name,
            "representation": representation,
            "classifier": classifier,
            "kind": "generative" if classifier in ("mnb", "gnb")
                    else "discriminative",
            "accuracy": round(metrics["accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "neutral_to_negative": round(metrics["se_neutral_to_negative_rate"], 4),
            "negative_recall": round(metrics["per_class"]["negative"]["recall"], 4),
        })
        print(f"  {name:14} test macro-F1 {metrics['macro_f1']:.4f}")

    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    table.to_csv(RESULTS / "metrics" / f"master_results_{domain}.csv", index=False)

    # ----------------------------------------------------------- master table
    print("\n" + "=" * 92)
    print(f"{'model':14} {'representation':15} {'kind':15} {'test F1':>9}"
          f" {'accuracy':>9} {'neut->neg':>10} {'neg rec':>9}")
    print("=" * 92)
    for _, row in table.iterrows():
        print(f"{row['model']:14} {row['representation']:15} {row['kind']:15}"
              f" {row['macro_f1']:>9.4f} {row['accuracy']:>9.4f}"
              f" {row['neutral_to_negative']:>9.1%} {row['negative_recall']:>8.1%}")
    print("=" * 92)

    baselines = {"majority": 0.1852, "vader": 0.7079}
    best = table.iloc[0]
    print(f"\nBest overall: {best['model']}  test macro-F1 {best['macro_f1']:.4f}")
    print(f"  vs majority baseline {baselines['majority']:.4f}"
          f"  ({best['macro_f1'] - baselines['majority']:+.4f})")
    print(f"  vs VADER baseline    {baselines['vader']:.4f}"
          f"  ({best['macro_f1'] - baselines['vader']:+.4f})")

    # --------------------------------------------------- proposal outcome #1
    print("\nOutcome #1 -- which representation handles SE vocabulary best?")
    by_rep = table.groupby("representation")["macro_f1"].agg(["max", "mean"])
    for rep, stats in by_rep.sort_values("max", ascending=False).iterrows():
        print(f"  {rep:10} best {stats['max']:.4f}   mean {stats['mean']:.4f}")

    # --------------------------------------------------- proposal outcome #2
    print("\nOutcome #2 -- generative vs discriminative")
    by_kind = table.groupby("kind")["macro_f1"].agg(["max", "mean"])
    for kind, stats in by_kind.iterrows():
        print(f"  {kind:15} best {stats['max']:.4f}   mean {stats['mean']:.4f}")

    # ------------------------------------------------------ significance test
    best_tfidf = table[table["representation"] == "TF-IDF"].iloc[0]["model"]
    best_embed = table[table["representation"] != "TF-IDF"].iloc[0]["model"]
    result = mcnemar_test(y_true, predictions[best_tfidf], predictions[best_embed])
    print(f"\nMcNemar, {best_tfidf} vs {best_embed}:"
          f" p = {result['pvalue']:.2e}"
          f" ({'significant' if result['significant_05'] else 'not significant'})")

    # ------------------------------------------------------- error analysis
    best_model = best["model"]
    y_pred = predictions[best_model]
    errors = pd.DataFrame({
        "id": test_df["id"].values,
        "text": test_df["text"].values,
        "gold": y_true,
        "predicted": y_pred,
    })
    confusions = errors[(errors["gold"] == "neutral")
                        & (errors["predicted"] == "negative")].copy()
    confusions["tag"] = confusions["text"].map(lambda t: tag_error(t, jargon))
    confusions.to_csv(
        ERROR_ANALYSIS / f"neutral_as_negative_{best_model}.csv", index=False)

    print(f"\nError analysis -- {best_model}, gold neutral predicted negative")
    print(f"  {len(confusions)} of {(errors['gold'] == 'neutral').sum()}"
          f" neutral test posts ({best['neutral_to_negative']:.1%})")
    for tag, count in confusions["tag"].value_counts().items():
        print(f"    {tag:22} {count:>3}")

    print("\n  Five examples:")
    for _, row in confusions.head(5).iterrows():
        snippet = " ".join(row["text"].split())[:88]
        print(f"    [{row['tag']}] {snippet}")

    # --------------------------------------------------------------- charts
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    grouped = table.groupby("representation")["macro_f1"].max().sort_values()
    axes[0].barh(grouped.index, grouped.values, color="#4C72B0")
    axes[0].axvline(baselines["vader"], color="#C44E52", linestyle="--",
                    label=f"VADER {baselines['vader']:.3f}")
    axes[0].set_xlabel("test macro-F1")
    axes[0].set_title("Best model per representation")
    axes[0].legend(fontsize=8)

    pivot = table.pivot_table(index="representation", columns="kind",
                              values="macro_f1", aggfunc="mean")
    pivot.plot(kind="bar", ax=axes[1], color=["#DD8452", "#55A868"], rot=0)
    axes[1].set_ylabel("mean test macro-F1")
    axes[1].set_title("Generative vs discriminative")
    axes[1].legend(fontsize=8)

    axes[2].scatter(table["neutral_to_negative"], table["negative_recall"],
                    s=60, color="#4C72B0")
    for _, row in table.iterrows():
        axes[2].annotate(row["model"],
                         (row["neutral_to_negative"], row["negative_recall"]),
                         fontsize=6, xytext=(3, 3), textcoords="offset points")
    axes[2].set_xlabel("neutral misread as negative  (lower is better)")
    axes[2].set_ylabel("recall on real negatives  (higher is better)")
    axes[2].set_title("The trade-off; ideal is top-left")

    plt.tight_layout()
    plt.savefig(FIGURES / f"summary_{domain}.png", dpi=150)
    plt.close()

    print(f"\nwrote {RESULTS / 'metrics' / f'master_results_{domain}.csv'}")
    print(f"wrote {FIGURES / f'summary_{domain}.png'}")
    print(f"wrote {ERROR_ANALYSIS}")


if __name__ == "__main__":
    main()
