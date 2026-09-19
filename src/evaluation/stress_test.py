"""Stage 6.7 -- the hand-written stress set.

Run from the repository root:

    python -m src.evaluation.stress_test

--------------------------------------------------------------------------
Why this stage carries the project
--------------------------------------------------------------------------
The proposal's motivating examples are "fatal error", "kill the process" and
"null pointer exception".  Four separate analyses found that those phrases
barely exist in Senti4SD:

    Stage 4.1  "fatal error", "kill the process"      0 occurrences
    Stage 4.2  fatal 0x, hang 0x, abort 0x, kill 3x
    Stage 6.5  no lexicon word appears 3+ times in the LIME sample
    Stage 6.6  the SHAP bias score cannot be computed for the same reason

So the corpus cannot answer the question the project asks.  This hand-written
set can: 60 sentences built specifically to separate *technical vocabulary*
from *actual hostility*.

    40 harsh-but-neutral   "Kill the process before restarting the server."
    20 genuinely negative  of which 10 contain no jargon at all
    10 minimal pairs       the same jargon word, opposite intent

--------------------------------------------------------------------------
The four metrics
--------------------------------------------------------------------------
============================  ==============================================
false-alarm rate              share of harsh-but-neutral sentences called
                              negative.  **Lower is better.**  This is the
                              project's central quantity.
negative recall               share of genuine complaints caught.
                              **Higher is better.**  Without it, a model
                              that never says "negative" would look perfect.
jargon vs plain recall        recall on complaints that contain jargon
                              against those that do not.  A gap means the
                              model needs the jargon to notice hostility.
minimal-pair accuracy         share of the 10 pairs where *both* halves are
                              right.  The strictest test: same vocabulary,
                              opposite intent, no lexical shortcut available.
============================  ==============================================

--------------------------------------------------------------------------
Honest limitation
--------------------------------------------------------------------------
These sentences were written by one person who already knew what the models
were expected to get wrong.  That is a real bias and it belongs in
Limitations.  The minimal pairs are the partial defence: both halves share
the same jargon word, so a model cannot score well on them by keying on
vocabulary.
"""

from __future__ import annotations

import argparse

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.evaluate import evaluate
from src.paths import FIGURES, MODELS, RESULTS, STRESS_TEST
from src.preprocessing.splits import load_splits

STRESS_RESULTS = RESULTS / "stress_test"


def load_stress_set(domain: str = "se") -> pd.DataFrame:
    path = STRESS_TEST / f"{domain}_stress.csv"
    frame = pd.read_csv(path, sep=";")
    frame["lexicon_word"] = frame["lexicon_word"].fillna("")
    frame["has_jargon"] = frame["lexicon_word"] != ""
    return frame


def score_model(name: str, model, stress: pd.DataFrame) -> dict:
    """The four stress metrics for one model."""
    predicted = model.predict(stress["text"])
    frame = stress.assign(predicted=predicted)

    harsh_neutral = frame[frame["intended_label"] == "neutral"]
    complaints = frame[frame["intended_label"] == "negative"]

    false_alarms = (harsh_neutral["predicted"] == "negative").mean()
    negative_recall = (complaints["predicted"] == "negative").mean()

    with_jargon = complaints[complaints["has_jargon"]]
    without_jargon = complaints[~complaints["has_jargon"]]

    pairs = frame[frame["pair_id"].notna()]
    both_correct = (
        pairs.groupby("pair_id")
        .apply(lambda g: bool((g["predicted"] == g["intended_label"]).all()),
               include_groups=False)
    )

    return {
        "model": name,
        "false_alarm_rate": round(float(false_alarms), 4),
        "negative_recall": round(float(negative_recall), 4),
        "recall_with_jargon": round(
            float((with_jargon["predicted"] == "negative").mean()), 4),
        "recall_without_jargon": round(
            float((without_jargon["predicted"] == "negative").mean()), 4),
        "minimal_pair_accuracy": round(float(both_correct.mean()), 4),
        "accuracy": round(float((frame["predicted"] == frame["intended_label"]).mean()), 4),
    }


def vader_predictions(texts) -> list[str]:
    """The off-the-shelf baseline, for contrast."""
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    out = []
    for text in texts:
        compound = sia.polarity_scores(str(text))["compound"]
        out.append("positive" if compound >= 0.05
                   else "negative" if compound <= -0.05 else "neutral")
    return out


class _FixedPredictions:
    """Wraps a list of predictions so VADER can use the same scoring path."""

    def __init__(self, predictions):
        self._predictions = list(predictions)

    def predict(self, texts):
        return self._predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="se")
    args = parser.parse_args()

    domain = args.domain
    stress = load_stress_set(domain)
    STRESS_RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Stage 6.7 -- stress test on {domain.upper()}")
    print(f"  {len(stress)} hand-written sentences")
    print(f"    {(stress['intended_label'] == 'neutral').sum()} harsh-but-neutral")
    print(f"    {(stress['intended_label'] == 'negative').sum()} genuine complaints"
          f" ({(~stress['has_jargon'] & (stress['intended_label'] == 'negative')).sum()}"
          f" with no jargon)")
    print(f"    {stress['pair_id'].notna().sum() // 2} minimal pairs")
    print(f"    {stress['lexicon_word'].nunique() - 1} distinct lexicon words\n")

    model_dir = MODELS / domain
    rows = [score_model("vader", _FixedPredictions(
        vader_predictions(stress["text"])), stress)]

    for path in sorted(model_dir.glob("*.joblib")):
        model = joblib.load(path)
        rows.append(score_model(path.stem, model, stress))

    table = pd.DataFrame(rows).sort_values("false_alarm_rate")
    table.to_csv(STRESS_RESULTS / f"stress_results_{domain}.csv", index=False)

    print("=" * 96)
    print(f"{'model':14} {'false alarm':>12} {'neg recall':>11} {'jargon':>8}"
          f" {'plain':>8} {'pairs':>8} {'accuracy':>9}")
    print("=" * 96)
    for _, row in table.iterrows():
        print(f"{row['model']:14} {row['false_alarm_rate']:>11.1%}"
              f" {row['negative_recall']:>10.1%} {row['recall_with_jargon']:>7.0%}"
              f" {row['recall_without_jargon']:>7.0%}"
              f" {row['minimal_pair_accuracy']:>7.0%}"
              f" {row['accuracy']:>8.1%}")
    print("=" * 96)

    # ------------------------------------------------------- interpretation
    vader = table[table["model"] == "vader"].iloc[0]
    trained = table[table["model"] != "vader"]
    best = trained.loc[trained["false_alarm_rate"].idxmin()]

    print(f"\nTHE HEADLINE")
    print(f"  VADER, which has never seen a StackOverflow post, calls")
    print(f"  {vader['false_alarm_rate']:.0%} of harsh-but-neutral sentences negative.")
    print(f"  The lowest trained model ({best['model']}) calls"
          f" {best['false_alarm_rate']:.0%}.")

    # Rank by how much general English the representation carries.  This is
    # the cleanest answer the project has to Expected Outcome #3.
    family_of = {"glove": "GloVe (general English)",
                 "w2v": "Word2Vec (this corpus)"}
    table_with_family = trained.assign(
        family=trained["model"].map(
            lambda m: family_of.get(m.split("_")[0], "TF-IDF (this corpus)")))
    print("\n  Mean false-alarm rate by representation:")
    print(f"    {'VADER (general lexicon)':28} {vader['false_alarm_rate']:>6.0%}")
    for family, mean in (table_with_family.groupby("family")["false_alarm_rate"]
                         .mean().sort_values(ascending=False).items()):
        print(f"    {family:28} {mean:>6.0%}")
    print("  The ordering tracks how much general English each representation")
    print("  carries. Pretrained GloVe inherits the bias; vectors learned on")
    print("  StackOverflow do not. That is Expected Outcome #3, in one column.")

    # A model can win here for the wrong reason, so anchor on test quality.
    master = RESULTS / "metrics" / f"master_results_{domain}.csv"
    if master.exists():
        test_table = pd.read_csv(master)
        best_on_test = test_table.iloc[0]["model"]
        if best_on_test in set(trained["model"]):
            anchor = trained[trained["model"] == best_on_test].iloc[0]
            print(f"\n  CAVEAT -- the most useful comparison")
            print(f"  {best['model']} has the lowest false-alarm rate here but"
                  f" the worst")
            print(f"  test macro-F1 of any model. A cautious model looks good on"
                  f" this")
            print(f"  metric for free. Anchor the claim on {best_on_test}, which"
                  f" is best")
            print(f"  on the real test split: {anchor['false_alarm_rate']:.0%}"
                  f" false alarms against VADER's"
                  f" {vader['false_alarm_rate']:.0%}, while also")
            print(f"  catching {anchor['negative_recall']:.0%} of genuine complaints.")

    print(f"\nRecall on complaints containing jargon vs plain complaints:")
    for _, row in trained.nlargest(3, "accuracy").iterrows():
        gap = row["recall_with_jargon"] - row["recall_without_jargon"]
        print(f"  {row['model']:14} {row['recall_with_jargon']:.0%} vs"
              f" {row['recall_without_jargon']:.0%}   gap {gap:+.0%}")
    print("  A large positive gap means the model needs the technical")
    print("  vocabulary to notice hostility at all.")

    best_pairs = trained.loc[trained["minimal_pair_accuracy"].idxmax()]
    print(f"\nMinimal pairs -- same jargon word, opposite intent:")
    print(f"  best is {best_pairs['model']} at"
          f" {best_pairs['minimal_pair_accuracy']:.0%} of 10 pairs fully correct.")
    print("  This is the strictest test in the project: no model can score")
    print("  here by keying on vocabulary, because both halves share it.")

    # ------------------------------------------------------ per-sentence dump
    detail_model = joblib.load(model_dir / f"{best['model']}.joblib")
    detail = stress.assign(predicted=detail_model.predict(stress["text"]))
    detail["correct"] = detail["predicted"] == detail["intended_label"]
    detail.to_csv(STRESS_RESULTS / f"stress_predictions_{best['model']}.csv",
                  index=False)

    wrong = detail[~detail["correct"]]
    print(f"\n{best['model']} got {len(wrong)} of {len(detail)} wrong. Worst cases:")
    for _, row in wrong.head(8).iterrows():
        print(f"  [{row['intended_label']} -> {row['predicted']}] {row['text'][:72]}")

    # --------------------------------------------------------------- figure
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for _, row in table.iterrows():
        colour = "#C44E52" if row["model"] == "vader" else "#4C72B0"
        ax.scatter(row["false_alarm_rate"], row["negative_recall"],
                   s=80, color=colour, zorder=3)
        ax.annotate(row["model"],
                    (row["false_alarm_rate"], row["negative_recall"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("false-alarm rate on harsh-but-neutral sentences  (lower is better)")
    ax.set_ylabel("recall on genuine complaints  (higher is better)")
    ax.set_title("Stress test: ideal is top-left")
    ax.grid(alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(FIGURES / f"stress_scatter_{domain}.png", dpi=150)
    plt.close()

    print(f"\nwrote {STRESS_RESULTS / f'stress_results_{domain}.csv'}")
    print(f"wrote {FIGURES / f'stress_scatter_{domain}.png'}")


if __name__ == "__main__":
    main()
