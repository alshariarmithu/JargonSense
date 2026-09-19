"""Stage 3.2 -- decide whether to stem, lemmatize, or neither.

Run from the repository root:

    python -m tools.run_ablation

--------------------------------------------------------------------------
The question
--------------------------------------------------------------------------
Stemming and lemmatization shrink the vocabulary by merging inflected forms
(``killed``, ``kills``, ``killing`` -> ``kill``).  On a corpus this small that
should help: fewer features, more evidence behind each one.

But the inflections may *be* the signal.  "Killed the process" is routine
technical description, and folding it into the emotional register of "kill" is
exactly the confusion this project exists to measure.  A preprocessing step
that erases the phenomenon would raise the score while destroying the study.

Rather than argue about it, this script measures it.

--------------------------------------------------------------------------
The experiment
--------------------------------------------------------------------------
One model -- TF-IDF (1,2) + Logistic Regression at library defaults -- trained
three times on the training split, changing **nothing but the normalisation
mode**, and scored on the **validation** split.

The test split is not touched.  It is spent once, in Stage 6.

Two numbers decide the outcome, and they can disagree:

* **macro-F1** -- ordinary accuracy, averaged fairly over the three classes.
* **neutral -> negative rate** -- how often a genuinely neutral post is called
  negative.  This is the project's headline failure, so a mode that improves
  macro-F1 while making this worse has bought accuracy with the very thing
  being studied.

The selection rule is macro-F1, stated up front so it cannot be chosen after
seeing the numbers.  If the winner is also worst on the confusion rate, the
script says so loudly -- that trade-off belongs in the report either way.

--------------------------------------------------------------------------
Output
--------------------------------------------------------------------------
    results/ablation/normalization_se.csv   the comparison table
    results/ablation/chosen_mode.json       the decision, read by Stage 4+
"""

import json
from functools import partial

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.extraction.clean_text import SEED, TOKEN_PATTERN
from src.evaluation.evaluate import evaluate
from src.paths import ABLATION
from src.preprocessing.normalize import MODES, prepare
from src.preprocessing.splits import load_splits

# Held identical across all three runs -- only `mode` varies.
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95

SELECTION_METRIC = "macro_f1"


def build_pipeline(domain: str, mode: str) -> Pipeline:
    """The same model every time, differing only in normalisation mode."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=partial(prepare, domain=domain, mode=mode),
            token_pattern=TOKEN_PATTERN,
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
            max_df=MAX_DF,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def run_one(domain: str, mode: str, train_df, val_df) -> dict:
    """Train under one mode and score it on the validation split."""
    pipeline = build_pipeline(domain, mode)
    pipeline.fit(train_df["text"], train_df["polarity"])

    y_pred = pipeline.predict(val_df["text"])
    metrics = evaluate(
        f"ablation_{mode}", domain, val_df["polarity"], y_pred,
        out_dir="results/ablation", fig_dir="results/ablation",
    )

    return {
        "mode": mode,
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "accuracy": round(metrics["accuracy"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "neutral_to_negative_rate": round(metrics["se_neutral_to_negative_rate"], 4),
    }


def main() -> None:
    domain = "se"
    train_df, val_df, _ = load_splits(domain)

    print(f"Stage 3.2 -- normalisation ablation on {domain.upper()}")
    print(f"  model      TF-IDF {NGRAM_RANGE} + LogisticRegression (defaults)")
    print(f"  train      {len(train_df):,} documents")
    print(f"  scored on  {len(val_df):,} validation documents "
          f"(the test split is untouched)")
    print(f"  selected by {SELECTION_METRIC}, decided before seeing results\n")

    rows = []
    for mode in MODES:
        print(f"  running mode={mode} ...", end="", flush=True)
        row = run_one(domain, mode, train_df, val_df)
        rows.append(row)
        print(f" macro-F1 {row['macro_f1']:.4f}")

    table = pd.DataFrame(rows)

    ABLATION.mkdir(parents=True, exist_ok=True)
    csv_path = ABLATION / f"normalization_{domain}.csv"
    table.to_csv(csv_path, index=False)

    # ---------------------------------------------------------------- report
    print("\n" + "-" * 74)
    print(f"{'mode':8} {'vocabulary':>11} {'accuracy':>10} {'macro-F1':>10}"
          f" {'neutral->negative':>19}")
    print("-" * 74)
    for row in rows:
        print(f"{row['mode']:8} {row['vocabulary_size']:>11,}"
              f" {row['accuracy']:>10.4f} {row['macro_f1']:>10.4f}"
              f" {row['neutral_to_negative_rate']:>18.1%}")
    print("-" * 74)

    winner = max(rows, key=lambda r: r[SELECTION_METRIC])
    runner_up = sorted(rows, key=lambda r: r[SELECTION_METRIC])[-2]
    best_confusion = min(rows, key=lambda r: r["neutral_to_negative_rate"])
    worst_confusion = max(rows, key=lambda r: r["neutral_to_negative_rate"])
    n_val = len(val_df)

    print(f"\nSelected: mode = '{winner['mode']}' (highest {SELECTION_METRIC})")

    # How big is the win, really?  A macro-F1 gap means little until it is
    # expressed as documents on a 650-row validation split.
    margin = winner[SELECTION_METRIC] - runner_up[SELECTION_METRIC]
    docs = abs(winner["accuracy"] - runner_up["accuracy"]) * n_val
    print(f"  margin over '{runner_up['mode']}': {margin:+.4f} macro-F1"
          f"  (~{docs:.0f} of {n_val} validation documents)")

    if docs < 10:
        print("\n  ** READ THIS BEFORE WRITING IT UP **")
        print("  The three modes are separated by only a handful of documents.")
        print("  Treat this as 'normalisation does not matter much on this")
        print("  corpus', not as evidence that one mode is genuinely better.")
        print("  The honest reason to prefer 'none' is that it keeps the")
        print("  inflections the project studies, and it is the simplest and")
        print("  fastest option -- not that it scored highest.")

    # The trade-off that matters: does the best-scoring mode also handle the
    # project's central failure best?
    print(f"\n  lowest neutral->negative rate: '{best_confusion['mode']}' "
          f"at {best_confusion['neutral_to_negative_rate']:.1%}")
    if best_confusion["mode"] != winner["mode"]:
        print(f"  -- which is NOT the selected mode "
              f"('{winner['mode']}' at "
              f"{winner['neutral_to_negative_rate']:.1%}).")
        print("  Accuracy and the project's failure metric disagree here.")
        print("  That disagreement is a finding; put it in the report.")
    if worst_confusion["mode"] == winner["mode"]:
        print("  ** The selected mode also has the WORST neutral->negative")
        print("     rate: it bought accuracy with the very failure being")
        print("     studied. Discuss this explicitly. **")

    decision = {
        "mode": winner["mode"],
        "selected_by": SELECTION_METRIC,
        "domain": domain,
        "scored_on": "validation",
        "model": f"tfidf{NGRAM_RANGE[0]}{NGRAM_RANGE[1]}_lr_defaults",
        "results": rows,
    }
    (ABLATION / "chosen_mode.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8")

    print(f"\nwrote {csv_path}")
    print(f"wrote {ABLATION / 'chosen_mode.json'}")
    print("\nEvery later stage reads chosen_mode.json, so the whole project")
    print("uses the mode this experiment selected rather than a hard-coded guess.")


if __name__ == "__main__":
    main()
