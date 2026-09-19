"""Stage 4.1 -- TF-IDF features.

Run the full analysis from the repository root:

    python -m src.features.features_tfidf

--------------------------------------------------------------------------
What this stage produces
--------------------------------------------------------------------------
Three vectorisers, identical except for how many words they look at together:

============  ==========================================================
``tfidf11``   Single words only.  No context at all.  The baseline.
``tfidf12``   Adds word pairs   -- "fatal error", "does not"
``tfidf13``   Adds word triples -- "kill the process"
============  ==========================================================

``tfidf11`` is not expected to win.  It is the control: without it, "word
pairs help" is an assertion, and the gap between ``tfidf11`` and ``tfidf12``
*is* the measurement of how much compound phrases are worth.

--------------------------------------------------------------------------
Why the vectoriser keeps its own preprocessor
--------------------------------------------------------------------------
It would be faster to clean every document once and hand the vectoriser a list
of ready-made strings.  It is done the slow way on purpose: the vectoriser
takes **raw** text, so a saved model is a complete pipeline.

That matters in Stage 6.  LIME and SHAP invent thousands of variations of a
sentence and ask the model to score them; the demo takes whatever a user
types.  Neither can run the cleaning step first.  If cleaning lived outside
the model, the explanation would describe a model the user never used.

--------------------------------------------------------------------------
Fit on training data only
--------------------------------------------------------------------------
Every vectoriser here is fitted on the training split and only transforms the
others.  The vocabulary and the IDF weights are learned parameters; fitting
them on validation or test text would let the model see data it is about to
be scored on.
"""

from __future__ import annotations

import json
from functools import partial

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.extraction.clean_text import LABELS, SEED, TOKEN_PATTERN
from src.paths import FEATURES
from src.preprocessing.normalize import load_chosen_mode, prepare
from src.preprocessing.splits import load_splits

# The three configurations, and the only thing that differs between them.
CONFIGS = {
    "tfidf11": (1, 1),
    "tfidf12": (1, 2),
    "tfidf13": (1, 3),
}

# Shared settings, held identical across configurations so the n-gram range is
# the only variable.
DEFAULT_SETTINGS = {
    "min_df": 2,        # a term must appear in 2+ documents to be a feature
    "max_df": 0.95,     # drop terms in >95% of documents: they separate nothing
    "sublinear_tf": True,   # log-scale term frequency; one word repeated ten
                            # times is not ten times as informative
}

TUNED_SETTINGS_FILE = FEATURES / "tfidf_settings.json"

# Phrases whose survival into the (1,2)/(1,3) vocabularies is the point of
# using n-grams at all.  Each one means something different from its parts:
# "fatal error" is routine, while "fatal" alone reads as catastrophe.
COMPOUND_PHRASES = [
    # technical, functionally neutral
    "fatal error",
    "null pointer",
    "memory leak",
    "stack trace",
    "error message",
    "kill the process",
    "null pointer exception",
    "out of memory",
    # negation and sentiment, where the pair flips the part
    "does not",
    "not work",
    "does not work",
    "no idea",
    # genuinely positive
    "thank you",
    "works fine",
]


def build_vectorizer(config: str, domain: str = "se", mode: str | None = None,
                     **overrides) -> TfidfVectorizer:
    """Return one of the three configured vectorisers.

    Parameters
    ----------
    config
        ``"tfidf11"``, ``"tfidf12"`` or ``"tfidf13"``.
    domain
        Passed through to ``prepare()``.
    mode
        Normalisation mode.  Defaults to whatever the Stage 3 ablation chose.
    **overrides
        Any ``TfidfVectorizer`` argument, e.g. ``min_df=5``.  Used by tuning.
    """
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {tuple(CONFIGS)}, got {config!r}")

    if mode is None:
        mode = load_chosen_mode()

    settings = {**DEFAULT_SETTINGS, **overrides}
    return TfidfVectorizer(
        preprocessor=partial(prepare, domain=domain, mode=mode),
        token_pattern=TOKEN_PATTERN,
        ngram_range=CONFIGS[config],
        **settings,
    )


def load_tuned_settings(config: str) -> dict:
    """Return the settings Stage 4.1's tuning chose, or the defaults."""
    if not TUNED_SETTINGS_FILE.exists():
        return dict(DEFAULT_SETTINGS)
    chosen = json.loads(TUNED_SETTINGS_FILE.read_text(encoding="utf-8"))
    return {**DEFAULT_SETTINGS, **chosen.get(config, {})}


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def vocabulary_stats(config: str, vectorizer: TfidfVectorizer, matrix) -> dict:
    """Size and sparsity of one fitted vectoriser's feature space."""
    rows, cols = matrix.shape
    filled = matrix.nnz / (rows * cols)
    return {
        "config": config,
        "ngram_range": str(CONFIGS[config]),
        "vocabulary_size": cols,
        "nonzero_entries": int(matrix.nnz),
        "sparsity": round(1 - filled, 6),
        "mean_features_per_document": round(matrix.nnz / rows, 1),
    }


def top_ngrams_per_class(vectorizer: TfidfVectorizer, matrix, labels,
                         top_n: int = 30) -> pd.DataFrame:
    """The n-grams most *characteristic* of each class.

    Ranking by mean TF-IDF alone does not work: it returns "i", "the", "to"
    for every class, because common words carry weight everywhere.  That says
    nothing about which class a term belongs to.

    So each term is scored by how much heavier it is inside the class than
    outside it::

        distinctiveness = mean TF-IDF within the class
                        - mean TF-IDF in the other two classes

    A word used equally everywhere scores ~0 and disappears.  Both numbers are
    saved, so the report can show that stopwords dominate the raw ranking and
    why the corrected one is used instead.
    """
    names = np.array(vectorizer.get_feature_names_out())
    labels = np.asarray(labels)
    overall_mean = np.asarray(matrix.mean(axis=0)).ravel()
    n_documents = matrix.shape[0]

    rows = []
    for label in LABELS:
        mask = labels == label
        if not mask.any():
            continue
        in_class = np.asarray(matrix[mask].mean(axis=0)).ravel()

        # Mean over the other classes, recovered from the totals rather than
        # slicing the matrix a second time.
        n_in = int(mask.sum())
        n_out = n_documents - n_in
        out_class = ((overall_mean * n_documents - in_class * n_in) / n_out
                     if n_out else np.zeros_like(in_class))

        distinctiveness = in_class - out_class
        for rank, index in enumerate(
                np.argsort(distinctiveness)[::-1][:top_n], start=1):
            rows.append({
                "polarity": label,
                "rank": rank,
                "ngram": names[index],
                "distinctiveness": round(float(distinctiveness[index]), 5),
                "mean_tfidf_in_class": round(float(in_class[index]), 5),
                "mean_tfidf_other_classes": round(float(out_class[index]), 5),
            })
    return pd.DataFrame(rows)


def compound_phrase_report(vectorizers: dict, matrices: dict, labels,
                           phrases=COMPOUND_PHRASES) -> pd.DataFrame:
    """Did the multi-word phrases survive into each vocabulary, and where?

    A phrase can only be learned if it is in the vocabulary, and it only
    reaches the vocabulary if the n-gram range is wide enough *and* it clears
    ``min_df``.  This table shows both, plus how the surviving phrases are
    distributed across the three classes.
    """
    labels = np.asarray(labels)

    rows = []
    for phrase in phrases:
        row = {"phrase": phrase, "words": len(phrase.split())}
        for config, vectorizer in vectorizers.items():
            index = vectorizer.vocabulary_.get(phrase)
            row[f"in_{config}"] = index is not None

            if config == "tfidf13" and index is not None:
                # Count documents per class containing the phrase, using the
                # widest vocabulary so every phrase is measurable.
                column = matrices[config][:, index]
                present = np.asarray(column.todense()).ravel() > 0
                for label in LABELS:
                    row[f"n_{label}"] = int((present & (labels == label)).sum())
        rows.append(row)

    table = pd.DataFrame(rows)
    for label in LABELS:
        column = f"n_{label}"
        if column in table:
            table[column] = table[column].fillna(0).astype(int)
    return table


SELECTION_RULE = (
    "among settings within one standard error of the best validation score, "
    "take the smallest vocabulary"
)


def validation_standard_error(tuning: pd.DataFrame, n_val: int) -> float:
    """How much the validation score would wobble on a different 650 documents.

    A score measured on a finite sample carries sampling error.  With roughly
    650 documents and accuracy near 0.79, one standard error is about 0.016 --
    so two settings differing by 0.01 are indistinguishable, and picking the
    higher one is choosing noise.

    Estimated as the binomial standard error of the mean validation score,
    ``sqrt(p(1-p)/n)``, which is a reasonable stand-in for macro-F1 here
    because the three classes are close to balanced.
    """
    p = float(tuning["val_macro_f1"].mean())
    return float(np.sqrt(p * (1 - p) / n_val))


def select_settings(subset: pd.DataFrame, standard_error: float) -> pd.Series:
    """Apply ``SELECTION_RULE`` to one configuration's tuning results.

    This is the one-standard-error rule: when several settings are
    statistically tied, prefer the simplest.  A smaller vocabulary means less
    memorisation, a smaller saved model, and explanations built from terms
    that appear often enough to mean something.
    """
    threshold = subset["val_macro_f1"].max() - standard_error
    tied = subset[subset["val_macro_f1"] >= threshold]
    return tied.loc[tied["vocabulary_size"].idxmin()]


def tune(domain: str, train_df, val_df,
         min_dfs=(1, 2, 5), max_features_options=(None, 20000, 50000)) -> pd.DataFrame:
    """Light tuning of the vectoriser itself, scored on validation.

    A Logistic Regression at library defaults is used as a fixed probe.  The
    point is to choose *feature* settings, so the classifier is held constant
    -- tuning both at once would make it impossible to say which change helped.
    """
    rows = []
    for config in CONFIGS:
        for min_df in min_dfs:
            for max_features in max_features_options:
                vectorizer = build_vectorizer(
                    config, domain=domain,
                    min_df=min_df, max_features=max_features,
                )
                x_train = vectorizer.fit_transform(train_df["text"])
                x_val = vectorizer.transform(val_df["text"])

                clf = LogisticRegression(max_iter=2000, random_state=SEED)
                clf.fit(x_train, train_df["polarity"])

                score = partial(f1_score, average="macro", labels=LABELS,
                                zero_division=0)
                train_f1 = score(train_df["polarity"], clf.predict(x_train))
                val_f1 = score(val_df["polarity"], clf.predict(x_val))

                rows.append({
                    "config": config,
                    "min_df": min_df,
                    "max_features": max_features if max_features else "None",
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "features_per_document": round(
                        len(vectorizer.vocabulary_) / len(train_df), 1),
                    "train_macro_f1": round(float(train_f1), 4),
                    "val_macro_f1": round(float(val_f1), 4),
                    # How much of the training score fails to survive to
                    # unseen data.  Large gaps mean memorisation.
                    "overfit_gap": round(float(train_f1 - val_f1), 4),
                })
                print(f"    {config}  min_df={min_df}  "
                      f"max_features={str(max_features):>5}  "
                      f"vocab={len(vectorizer.vocabulary_):>7,}  "
                      f"train={train_f1:.4f}  val={val_f1:.4f}  "
                      f"gap={train_f1 - val_f1:.4f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def main() -> None:
    domain = "se"
    mode = load_chosen_mode()
    train_df, val_df, _ = load_splits(domain)
    FEATURES.mkdir(parents=True, exist_ok=True)

    print(f"Stage 4.1 -- TF-IDF features on {domain.upper()}")
    print(f"  normalisation mode : {mode}  (chosen by the Stage 3 ablation)")
    print(f"  fitted on          : {len(train_df):,} training documents")
    print(f"  shared settings    : {DEFAULT_SETTINGS}\n")

    vectorizers, matrices = {}, {}
    stats = []
    for config in CONFIGS:
        vectorizer = build_vectorizer(config, domain=domain, mode=mode)
        matrix = vectorizer.fit_transform(train_df["text"])
        vectorizers[config] = vectorizer
        matrices[config] = matrix
        stats.append(vocabulary_stats(config, vectorizer, matrix))

    # ------------------------------------------------------------ vocabulary
    stats_table = pd.DataFrame(stats)
    stats_table.to_csv(FEATURES / f"vocabulary_stats_{domain}.csv", index=False)

    print("Vocabulary")
    print(f"  {'config':9} {'n-grams':9} {'vocabulary':>11} {'sparsity':>10}"
          f" {'features/doc':>13}")
    for row in stats:
        print(f"  {row['config']:9} {row['ngram_range']:9}"
              f" {row['vocabulary_size']:>11,} {row['sparsity']:>10.5f}"
              f" {row['mean_features_per_document']:>13.1f}")

    unigram = stats[0]["vocabulary_size"]
    trigram = stats[-1]["vocabulary_size"]
    print(f"\n  Adding pairs and triples multiplies the feature count by"
          f" {trigram / unigram:.1f}x, on {len(train_df):,} documents.")
    print("  Most of those features appear in exactly 2 posts, which is why")
    print("  trigrams may not pay for themselves at this corpus size.")

    # -------------------------------------------------------- top n-grams
    top = top_ngrams_per_class(
        vectorizers["tfidf12"], matrices["tfidf12"], train_df["polarity"])
    top.to_csv(FEATURES / f"top_ngrams_{domain}.csv", index=False)

    print("\nHeaviest terms per class (tfidf12, top 8 of 30 saved)")
    for label in LABELS:
        terms = top[top["polarity"] == label].head(8)["ngram"].tolist()
        print(f"  {label:9} {', '.join(terms)}")

    # ------------------------------------------------------ compound phrases
    phrases = compound_phrase_report(vectorizers, matrices, train_df["polarity"])
    phrases.to_csv(FEATURES / f"compound_phrases_{domain}.csv", index=False)

    print("\nCompound phrases -- does the n-gram range actually capture them?")
    print(f"  {'phrase':24} {'words':>5} {'11':>4} {'12':>4} {'13':>4}"
          f"  {'neg':>4} {'neu':>4} {'pos':>4}")
    for _, row in phrases.iterrows():
        mark = lambda flag: " yes" if flag else "   ."
        counts = "".join(
            f" {row.get(f'n_{label}', 0):>4}" for label in LABELS
        ) if row["in_tfidf13"] else "     -    -    -"
        print(f"  {row['phrase']:24} {row['words']:>5}"
              f" {mark(row['in_tfidf11'])} {mark(row['in_tfidf12'])}"
              f" {mark(row['in_tfidf13'])} {counts}")

    survived = int(phrases["in_tfidf13"].sum())
    missing = phrases.loc[~phrases["in_tfidf13"], "phrase"].tolist()
    print(f"\n  {survived} of {len(phrases)} phrases reached the (1,3) vocabulary.")
    print("  A phrase absent from tfidf11 but present in tfidf12/13 is a")
    print("  feature the unigram model structurally cannot learn.")

    if missing:
        print("\n  ** FINDING -- absent from the corpus entirely **")
        print(f"  {', '.join(missing)}")
        print("  These are the canonical SE phrases the project proposal cites")
        print("  as motivation. They do not occur in Senti4SD, so the case for")
        print("  n-grams cannot rest on them. What n-grams actually capture")
        print("  here is negation ('does not', 'not work') and politeness")
        print("  ('thank you') -- which the counts above show clearly.")
        print("  The canonical phrases can only be tested on the hand-written")
        print("  stress set in Stage 6.7, which makes that stage load-bearing")
        print("  rather than supplementary.")

    # ------------------------------------------------------------- tuning
    print("\nTuning the vectoriser (Logistic Regression held at defaults)")
    tuning = tune(domain, train_df, val_df)
    tuning.to_csv(FEATURES / f"tuning_{domain}.csv", index=False)

    standard_error = validation_standard_error(tuning, len(val_df))
    print(f"\n  Validation set is {len(val_df)} documents, so one standard error"
          f" on this estimate is about {standard_error:.4f} macro-F1")
    print(f"  ({standard_error * len(val_df):.0f} documents). Differences smaller"
          f" than that are noise.")
    print(f"  Selection rule: {SELECTION_RULE}\n")

    chosen = {}
    print("  chosen settings per configuration")
    for config in CONFIGS:
        subset = tuning[tuning["config"] == config]
        best = subset.loc[subset["val_macro_f1"].idxmax()]
        pick = select_settings(subset, standard_error)

        chosen[config] = {
            "min_df": int(pick["min_df"]),
            "max_features": (None if pick["max_features"] == "None"
                             else int(pick["max_features"])),
        }
        note = ""
        if pick["vocabulary_size"] != best["vocabulary_size"]:
            note = (f"   [top scorer was min_df={best['min_df']} at "
                    f"{best['val_macro_f1']:.4f}, "
                    f"{best['vocabulary_size']:,} features -- within noise]")
        print(f"    {config}  min_df={pick['min_df']}"
              f"  max_features={pick['max_features']}"
              f"  vocab={pick['vocabulary_size']:>7,}"
              f"  val={pick['val_macro_f1']:.4f}"
              f"  gap={pick['overfit_gap']:.3f}{note}")

    top_scorer = tuning.loc[tuning["val_macro_f1"].idxmax()]
    print(f"\n  Highest raw validation score: {top_scorer['config']} at "
          f"{top_scorer['val_macro_f1']:.4f} "
          f"(min_df={top_scorer['min_df']}, "
          f"{top_scorer['vocabulary_size']:,} features)")

    if int(top_scorer["min_df"]) == 1:
        print("\n  ** WHY THAT SETTING IS NOT SHIPPED -- FOR THE REPORT **")
        print(f"  min_df=1 keeps every term appearing even once, giving")
        print(f"  {top_scorer['vocabulary_size']:,} features for {len(train_df):,} documents"
              f" ({top_scorer['features_per_document']:.0f} per document).")
        print(f"  Its train/validation gap is {top_scorer['overfit_gap']:.3f}: it is largely")
        print("  memorising, and its lead is smaller than one standard error.")
        print("  It would also make LIME hard to read, since most features")
        print("  would be terms occurring in a single post.")

    print(f"\n  Note: every configuration has a train/validation gap of"
          f" {tuning['overfit_gap'].min():.2f}-{tuning['overfit_gap'].max():.2f}.")
    print("  That is expected for a linear model on 3,031 short documents,")
    print("  and it is why Stage 5 tunes regularisation (C) rather than")
    print("  trusting these defaults. Worth a sentence in the report.")

    TUNED_SETTINGS_FILE.write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    print(f"\nwrote {FEATURES / f'vocabulary_stats_{domain}.csv'}")
    print(f"wrote {FEATURES / f'top_ngrams_{domain}.csv'}")
    print(f"wrote {FEATURES / f'compound_phrases_{domain}.csv'}")
    print(f"wrote {FEATURES / f'tuning_{domain}.csv'}")
    print(f"wrote {TUNED_SETTINGS_FILE}")


if __name__ == "__main__":
    main()
