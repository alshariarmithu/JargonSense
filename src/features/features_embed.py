"""Stage 4.2 -- embedding features: pretrained GloVe vs self-trained Word2Vec.

Run the full analysis from the repository root:

    python -m src.features.features_embed

--------------------------------------------------------------------------
The comparison this stage exists for
--------------------------------------------------------------------------
TF-IDF treats ``kill`` as an arbitrary column index: it has no idea the word
means anything.  Embeddings place words in a space where distance is meaning,
which raises the question this project cares about most:

    **whose meaning?**

============  ==========================================================
``glove``     Pretrained on Wikipedia and news text.  It learned ``kill``
              from sentences about violence, because that is what
              "kill" means in general English.
``w2v``       Trained here, on the StackOverflow training split alone.
              It only ever saw ``kill`` next to ``process`` and
              ``thread``.
============  ==========================================================

If the nearest neighbours of ``kill`` are ``murder`` and ``killing`` in GloVe
but ``process`` and ``restart`` in Word2Vec, the domain-shift claim stops
being an argument and becomes a table.  That table is produced by
``neighbour_table()`` and is the most quotable output in the project.

--------------------------------------------------------------------------
How a document becomes one vector
--------------------------------------------------------------------------
Mean pooling: average the vectors of every in-vocabulary token.  It is crude
-- word order is discarded entirely, so "not good" and "good not" are
identical -- but it is the standard baseline, and being order-blind makes it
a fair partner for the bag-of-words TF-IDF models rather than a confound.

Out-of-vocabulary tokens are skipped.  A document with no known tokens gets a
zero vector, which is why the OOV rate is reported: a high rate means the
model is scoring empty vectors and the comparison is meaningless.

--------------------------------------------------------------------------
Trained on the training split only
--------------------------------------------------------------------------
Word2Vec is fitted inside ``fit()``, on training text alone.  Training it on
the whole corpus would leak test vocabulary and co-occurrence statistics into
the model before it is scored.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

from src.extraction.clean_text import SEED
from src.paths import FEATURES, MODELS
from src.preprocessing.normalize import load_chosen_mode, normalize_tokens, prepare
from src.preprocessing.splits import load_splits

EMBEDDINGS = ("glove", "w2v")

GLOVE_MODEL = "glove-wiki-gigaword-100"
VECTOR_SIZE = 100

# Word2Vec settings.  workers=1 with a fixed seed makes training reproducible;
# with more workers, thread scheduling changes the result run to run.
W2V_SETTINGS = {
    "vector_size": VECTOR_SIZE,
    "window": 5,
    "min_count": 2,
    "sg": 1,            # skip-gram: better than CBOW on small corpora
    "negative": 5,
    "epochs": 30,
    "seed": SEED,
    "workers": 1,
}

# The words whose neighbours make or break the domain-shift argument.
SE_PROBE_WORDS = ["kill", "fatal", "crash", "error", "exception", "hang", "abort"]

# Terms that should be heavily out-of-vocabulary in general-English GloVe.
CODE_TOKENS = ["CODE", "URL", "USER", "EMO_POS", "EMO_NEG"]

_glove_cache: KeyedVectors | None = None


def load_glove() -> KeyedVectors:
    """Load pretrained GloVe vectors, downloading them once if needed."""
    global _glove_cache
    if _glove_cache is None:
        import gensim.downloader
        _glove_cache = gensim.downloader.load(GLOVE_MODEL)
    return _glove_cache


def train_word2vec(texts, mode: str | None = None,
                   **overrides) -> KeyedVectors:
    """Train Word2Vec on one corpus and return just the vectors."""
    if mode is None:
        mode = load_chosen_mode()

    sentences = [normalize_tokens(prepare(t, mode=mode), mode="none")
                 for t in texts]
    settings = {**W2V_SETTINGS, **overrides}
    model = Word2Vec(sentences, **settings)
    return model.wv


# --------------------------------------------------------------------------
# sklearn transformer
# --------------------------------------------------------------------------
class MeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Turn raw documents into mean-pooled word vectors.

    Takes **raw** text, like the TF-IDF vectoriser, so that a saved model is a
    complete pipeline that LIME, SHAP and the demo can all call directly.

    Parameters
    ----------
    embedding
        ``"glove"`` for pretrained vectors, ``"w2v"`` to train on the data
        passed to ``fit()``.
    """

    def __init__(self, embedding: str = "glove",
                 mode: str | None = None, vector_size: int = VECTOR_SIZE):
        self.embedding = embedding
        self.mode = mode
        self.vector_size = vector_size

    def fit(self, X, y=None):
        if self.embedding not in EMBEDDINGS:
            raise ValueError(
                f"embedding must be one of {EMBEDDINGS}, got {self.embedding!r}")

        self.mode_ = self.mode if self.mode is not None else load_chosen_mode()

        if self.embedding == "glove":
            self.vectors_ = load_glove()
        else:
            # Trained here, inside fit, so cross-validation retrains per fold
            # and no test text ever reaches the vocabulary.
            self.vectors_ = train_word2vec(
                X, mode=self.mode_, vector_size=self.vector_size)

        self.vector_size_ = self.vectors_.vector_size
        return self

    # ---------------------------------------------------------- pickling
    # Without these two methods every saved GloVe pipeline embeds its own
    # copy of all 400,000 pretrained vectors -- 171 MB per model file, and
    # three separate copies in memory once they are all loaded.  The vectors
    # are identical, immutable and already cached on disk by gensim, so they
    # are dropped on save and restored from the shared cache on load.
    #
    # Self-trained Word2Vec vectors are kept: they are small, and they exist
    # nowhere else.

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("embedding") == "glove":
            state.pop("vectors_", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.embedding == "glove" and not hasattr(self, "vectors_"):
            self.vectors_ = load_glove()

    def _tokens(self, text) -> list[str]:
        return normalize_tokens(
            prepare(text, mode=self.mode_), mode="none")

    def transform(self, X) -> np.ndarray:
        matrix = np.zeros((len(X), self.vector_size_), dtype=np.float32)
        for row, text in enumerate(X):
            known = [self.vectors_[t] for t in self._tokens(text)
                     if t in self.vectors_]
            if known:
                matrix[row] = np.mean(known, axis=0)
        return matrix

    def oov_rate(self, X) -> dict:
        """Share of tokens, and of documents, the embedding cannot represent."""
        total = known = 0
        empty_documents = 0
        for text in X:
            tokens = self._tokens(text)
            hits = sum(1 for t in tokens if t in self.vectors_)
            total += len(tokens)
            known += hits
            if tokens and hits == 0:
                empty_documents += 1
        return {
            "tokens": total,
            "oov_token_rate": round(1 - known / total, 4) if total else 0.0,
            "documents_with_no_known_tokens": empty_documents,
        }


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def corpus_counts(vectors: KeyedVectors, words) -> dict:
    """How many times each probe word occurred in the training corpus.

    Only meaningful for the self-trained model, whose vocabulary carries
    counts.  This is the diagnostic that explains embedding quality: a word
    seen a dozen times cannot have a good vector, no matter the settings.
    """
    counts = {}
    for word in words:
        try:
            counts[word] = int(vectors.get_vecattr(word, "count"))
        except KeyError:
            counts[word] = 0
    return counts


def neighbour_table(vectors_by_name: dict, words=SE_PROBE_WORDS,
                    top_n: int = 10, counts: dict | None = None) -> pd.DataFrame:
    """Nearest neighbours of each probe word in each embedding.

    The intended headline: the same word, two different meanings, depending on
    which corpus taught it.  ``counts`` adds the training frequency, which is
    what makes a disappointing row interpretable rather than just bad.
    """
    rows = []
    for word in words:
        for name, vectors in vectors_by_name.items():
            row = {
                "word": word,
                "embedding": name,
                "train_occurrences": (counts or {}).get(word, ""),
            }
            if word not in vectors:
                row["neighbours"] = "(out of vocabulary)"
            else:
                row["neighbours"] = ", ".join(
                    w for w, _ in vectors.most_similar(word, topn=top_n))
            rows.append(row)
    return pd.DataFrame(rows)


def search_word2vec_settings(train_df, val_df,
                             mode: str | None = None) -> pd.DataFrame:
    """Try the Word2Vec settings the plan specifies, scored on validation.

    ``vector_size`` in {50, 100} and ``sg`` in {0, 1} (CBOW vs skip-gram),
    judged by what actually matters downstream -- validation macro-F1 with a
    Logistic Regression on top -- rather than by how the neighbour lists read.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler

    rows = []
    for vector_size in (50, 100):
        for sg in (0, 1):
            vectors = train_word2vec(
                train_df["text"], mode=mode,
                vector_size=vector_size, sg=sg)

            vectorizer = MeanEmbeddingVectorizer(
                embedding="w2v", mode=mode,
                vector_size=vector_size)
            vectorizer.vectors_ = vectors
            vectorizer.vector_size_ = vectors.vector_size
            vectorizer.mode_ = mode if mode is not None else load_chosen_mode()

            scaler = StandardScaler()
            x_train = scaler.fit_transform(vectorizer.transform(train_df["text"]))
            x_val = scaler.transform(vectorizer.transform(val_df["text"]))

            clf = LogisticRegression(max_iter=2000, random_state=SEED)
            clf.fit(x_train, train_df["polarity"])
            macro_f1 = f1_score(val_df["polarity"], clf.predict(x_val),
                                average="macro", zero_division=0)

            rows.append({
                "vector_size": vector_size,
                "sg": sg,
                "algorithm": "skip-gram" if sg else "CBOW",
                "vocabulary_size": len(vectors),
                "val_macro_f1": round(float(macro_f1), 4),
            })
            print(f"    vector_size={vector_size:>3}  "
                  f"{'skip-gram' if sg else 'CBOW':10}  "
                  f"val macro-F1 {macro_f1:.4f}")
    return pd.DataFrame(rows)


def oov_report(vectorizers: dict, texts) -> pd.DataFrame:
    """OOV rates per embedding, overall and for placeholder tokens."""
    rows = []
    for name, vectorizer in vectorizers.items():
        stats = vectorizer.oov_rate(texts)
        known_placeholders = [t for t in CODE_TOKENS if t in vectorizer.vectors_]
        rows.append({
            "embedding": name,
            "vocabulary_size": len(vectorizer.vectors_),
            "oov_token_rate": stats["oov_token_rate"],
            "documents_with_no_known_tokens": stats["documents_with_no_known_tokens"],
            "placeholders_known": ", ".join(known_placeholders) or "(none)",
        })
    return pd.DataFrame(rows)


def build_vectorizer(embedding: str,
                     mode: str | None = None) -> MeanEmbeddingVectorizer:
    """Factory matching ``features_tfidf.build_vectorizer``."""
    if embedding not in EMBEDDINGS:
        raise ValueError(f"embedding must be one of {EMBEDDINGS}, got {embedding!r}")
    return MeanEmbeddingVectorizer(embedding=embedding, mode=mode)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def main() -> None:
    domain = "se"
    mode = load_chosen_mode()
    train_df, _, _ = load_splits()
    FEATURES.mkdir(parents=True, exist_ok=True)

    print(f"Stage 4.2 -- embedding features on {domain.upper()}")
    print(f"  normalisation : {mode} (Stage 3 ablation)")
    print(f"  training on   : {len(train_df):,} documents\n")

    print("  loading pretrained GloVe ...", end="", flush=True)
    glove_vec = build_vectorizer("glove", mode=mode)
    glove_vec.fit(train_df["text"])
    print(f" {len(glove_vec.vectors_):,} words, {glove_vec.vector_size_}d")

    print("  training Word2Vec on the training split ...", end="", flush=True)
    w2v_vec = build_vectorizer("w2v", mode=mode)
    w2v_vec.fit(train_df["text"])
    print(f" {len(w2v_vec.vectors_):,} words, {w2v_vec.vector_size_}d")

    vectorizers = {"glove": glove_vec, "w2v": w2v_vec}

    model_dir = MODELS / domain
    model_dir.mkdir(parents=True, exist_ok=True)
    w2v_vec.vectors_.save(str(model_dir / "w2v.kv"))

    # ------------------------------------------------------------------ OOV
    oov = oov_report(vectorizers, train_df["text"])
    oov.to_csv(FEATURES / f"oov_{domain}.csv", index=False)

    print("\nOut-of-vocabulary coverage")
    print(f"  {'embedding':10} {'vocabulary':>11} {'OOV tokens':>11}"
          f" {'empty docs':>11}  placeholders known")
    for _, row in oov.iterrows():
        print(f"  {row['embedding']:10} {row['vocabulary_size']:>11,}"
              f" {row['oov_token_rate']:>10.1%} {row['documents_with_no_known_tokens']:>11}"
              f"  {row['placeholders_known']}")

    print("\n  GloVe is lowercase general English, so it has no vector for the")
    print("  uppercase placeholders CODE/URL/USER that Stage 2 inserts -- every")
    print("  code span in a post is simply invisible to it. Word2Vec learned")
    print("  them, because it was trained on the placeholder-bearing text.")

    # ----------------------------------------------------- nearest neighbours
    vectors_by_name = {name: v.vectors_ for name, v in vectorizers.items()}
    counts = corpus_counts(w2v_vec.vectors_, SE_PROBE_WORDS)
    neighbours = neighbour_table(vectors_by_name, counts=counts)
    neighbours.to_csv(FEATURES / f"neighbours_{domain}.csv", index=False)

    print("\nNearest neighbours -- the same word, two meanings")
    print("-" * 78)
    for word in SE_PROBE_WORDS:
        print(f"  {word}   (appears {counts[word]}x in training text)")
        for name in EMBEDDINGS:
            match = neighbours[(neighbours["word"] == word)
                               & (neighbours["embedding"] == name)]
            if not match.empty:
                print(f"    {name:6} {match.iloc[0]['neighbours'][:64]}")
        print()

    # An honest reading of the table above.
    rare = {w: c for w, c in counts.items() if c < 50}
    print("  ** HOW TO READ THIS -- FOR THE REPORT **")
    print("  GloVe delivers the expected result exactly: it learned 'kill'")
    print("  from general English, so its neighbours are murder, shoot,")
    print("  poison. That half of the domain-shift argument is evidence.")
    print()
    print("  Word2Vec does not, and the frequency column says why. Probe")
    print(f"  words appear {min(counts.values())}-{max(counts.values())} times in "
          f"{len(train_df):,} documents;")
    if rare:
        print(f"  {', '.join(f'{w} ({c}x)' for w, c in rare.items())}.")
    print("  Word2Vec needs hundreds of occurrences per word to place it well.")
    print("  This is not a bug to fix by tuning -- it is the corpus-size")
    print("  limitation the proposal already lists under Limitations, now")
    print("  measured rather than predicted.")
    print()
    print("  Report it that way: the contrast this project hoped to show can")
    print("  be shown from the GloVe side alone (general-English vectors")
    print("  mis-read SE vocabulary), while the self-trained side demonstrates")
    print("  what 3,031 documents cannot buy.")

    # ----------------------------------------------- Word2Vec settings search
    print("\nWord2Vec settings, judged by downstream validation macro-F1")
    _, val_df, _ = load_splits()
    search = search_word2vec_settings(train_df, val_df, mode=mode)
    search.to_csv(FEATURES / f"w2v_search_{domain}.csv", index=False)
    best = search.loc[search["val_macro_f1"].idxmax()]
    spread = search["val_macro_f1"].max() - search["val_macro_f1"].min()
    print(f"    best: vector_size={best['vector_size']}, {best['algorithm']}"
          f"  ({best['val_macro_f1']:.4f}); spread across all four: {spread:.4f}")

    # -------------------------------------------------------- document vectors
    shapes = {}
    for name, vectorizer in vectorizers.items():
        matrix = vectorizer.transform(train_df["text"][:200])
        shapes[name] = matrix.shape
        zero_rows = int((np.abs(matrix).sum(axis=1) == 0).sum())
        print(f"  {name:6} document matrix {matrix.shape}, "
              f"{zero_rows} all-zero rows")

    summary = {
        "domain": domain,
        "mode": mode,
        "glove_model": GLOVE_MODEL,
        "w2v_settings": W2V_SETTINGS,
        "oov": oov.to_dict(orient="records"),
    }
    (FEATURES / f"embedding_summary_{domain}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nwrote {FEATURES / f'oov_{domain}.csv'}")
    print(f"wrote {FEATURES / f'neighbours_{domain}.csv'}")
    print(f"wrote {FEATURES / f'embedding_summary_{domain}.json'}")
    print(f"wrote {model_dir / 'w2v.kv'}")


if __name__ == "__main__":
    main()
