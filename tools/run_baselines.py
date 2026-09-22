"""Stage 6 -- the two reference points every real model must beat.

Run from the repository root:

    python -m tools.run_baselines

Two baselines, chosen because they fail in *different* ways:

* **majority** -- always predicts the most common training class.  It shows
  what accuracy is worth nothing: a model can look decent on accuracy alone
  while never predicting two of the three classes.  Its macro-F1 is the floor.

* **VADER** -- a general-purpose sentiment lexicon built for social media.
  It is the domain-shift demonstration.  VADER has never seen a StackOverflow
  post, so every time it reads "kill the process" as hostility it is making
  exactly the mistake this project is about.

Both write a row to ``results/metrics/all_results.csv`` through the shared
``evaluate()``, so they sit in the same table as everything built later.
"""

import argparse

import nltk

from src.preprocessing.splits import load_splits
from src.evaluation.evaluate import evaluate, append_to_results_table

# VADER ships as an NLTK data file rather than with the library itself.
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# VADER's documented cut-offs for turning its continuous compound score into
# three classes.  Not tuned on our data -- tuning them would quietly turn the
# baseline into a fitted model and stop it being a fair reference point.
POSITIVE_CUTOFF = 0.05
NEGATIVE_CUTOFF = -0.05


def run_majority(domain: str) -> dict:
    """Predict the most frequent training class for every test row."""
    train_df, _, test_df = load_splits()

    majority_class = train_df["polarity"].mode()[0]
    y_true = test_df["polarity"].tolist()
    y_pred = [majority_class] * len(y_true)

    metrics = evaluate("majority", domain, y_true, y_pred)
    append_to_results_table("majority", domain, metrics)

    print(f"[{domain.upper()}] majority (always '{majority_class}')")
    print(f"[{domain.upper()}]   accuracy {metrics['accuracy']:.4f}"
          f"   macro-F1 {metrics['macro_f1']:.4f}")
    return metrics


def vader_label(sia: SentimentIntensityAnalyzer, text: str) -> str:
    """Map VADER's compound score onto our three labels."""
    compound = sia.polarity_scores(str(text))["compound"]
    if compound >= POSITIVE_CUTOFF:
        return "positive"
    if compound <= NEGATIVE_CUTOFF:
        return "negative"
    return "neutral"


def run_vader(domain: str) -> dict:
    """Score the test split with the off-the-shelf VADER lexicon."""
    _, _, test_df = load_splits()

    sia = SentimentIntensityAnalyzer()
    y_true = test_df["polarity"].tolist()
    y_pred = [vader_label(sia, t) for t in test_df["text"]]

    metrics = evaluate("vader", domain, y_true, y_pred)
    append_to_results_table("vader", domain, metrics)

    print(f"[{domain.upper()}] vader (general-purpose lexicon)")
    print(f"[{domain.upper()}]   accuracy {metrics['accuracy']:.4f}"
          f"   macro-F1 {metrics['macro_f1']:.4f}")

    # The headline number for this project: how often a model that knows
    # nothing about the domain mistakes domain vocabulary for hostility.
    rate = metrics["se_neutral_to_negative_rate"]
    print(f"[SE]   gold-neutral posts called negative: {rate:.1%}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default="se", choices=["se"],
                        help="corpus label used in output filenames")
    args = parser.parse_args()

    run_majority(args.domain)
    run_vader(args.domain)


if __name__ == "__main__":
    main()
