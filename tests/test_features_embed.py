"""Tests for src/features/features_embed.py (Stage 4.2).

Run from the repository root:  python -m pytest -q

GloVe is 400,000 vectors and slow to load, so these tests build small
``KeyedVectors`` by hand instead.  What is being checked is pooling, OOV
handling and the pipeline contract, none of which depend on real vectors.
"""

import numpy as np
import pytest
from gensim.models import KeyedVectors

from src.features.features_embed import (
    EMBEDDINGS,
    MeanEmbeddingVectorizer,
    build_vectorizer,
    corpus_counts,
    neighbour_table,
    train_word2vec,
)

TEXTS = [
    "kill the process then restart the server",
    "the process was killed by the handler",
    "restart the server and the process",
    "the handler killed the server process",
]


def fake_vectors() -> KeyedVectors:
    """Three known words in an obvious 3-dimensional space."""
    vectors = KeyedVectors(vector_size=3)
    vectors.add_vectors(
        ["kill", "process", "server"],
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    )
    return vectors


def vectorizer_with(vectors: KeyedVectors) -> MeanEmbeddingVectorizer:
    """A fitted vectoriser wrapping supplied vectors, skipping training."""
    vectorizer = MeanEmbeddingVectorizer(embedding="w2v", domain="se", mode="none")
    vectorizer.vectors_ = vectors
    vectorizer.vector_size_ = vectors.vector_size
    vectorizer.mode_ = "none"
    return vectorizer


# --------------------------------------------------------------------------
# mean pooling
# --------------------------------------------------------------------------
def test_document_vector_is_the_mean_of_known_tokens():
    vectorizer = vectorizer_with(fake_vectors())
    # "kill" + "process": unknown words are skipped, so the mean is of two.
    matrix = vectorizer.transform(["kill the process"])
    assert np.allclose(matrix[0], [0.5, 0.5, 0.0])


def test_unknown_tokens_are_skipped_not_counted_as_zero():
    """Averaging in zeros for unknown words would shrink every vector."""
    vectorizer = vectorizer_with(fake_vectors())
    known_only = vectorizer.transform(["kill server"])
    with_unknown = vectorizer.transform(["kill zzzz server qqqq"])
    assert np.allclose(known_only, with_unknown)


def test_document_with_no_known_tokens_becomes_a_zero_vector():
    vectorizer = vectorizer_with(fake_vectors())
    matrix = vectorizer.transform(["zzzz qqqq"])
    assert np.allclose(matrix[0], 0.0)


def test_transform_shape_is_documents_by_dimensions():
    vectorizer = vectorizer_with(fake_vectors())
    assert vectorizer.transform(TEXTS).shape == (len(TEXTS), 3)


def test_mean_pooling_discards_word_order():
    """A documented limitation: "not good" and "good not" are identical.

    Recorded as a test so the report's claim about it stays true.
    """
    vectorizer = vectorizer_with(fake_vectors())
    forward = vectorizer.transform(["kill process server"])
    backward = vectorizer.transform(["server process kill"])
    assert np.allclose(forward, backward)


# --------------------------------------------------------------------------
# the pipeline contract
# --------------------------------------------------------------------------
def test_vectorizer_accepts_raw_text():
    """Like the TF-IDF side, cleaning happens inside the model."""
    vectorizer = vectorizer_with(fake_vectors())
    matrix = vectorizer.transform(["Kill the PROCESS via <code>foo()</code>!"])
    # "Kill" lowercased to a known word, the code span replaced and unknown.
    assert matrix[0][0] > 0


def test_unknown_embedding_is_rejected():
    with pytest.raises(ValueError, match="embedding must be one of"):
        build_vectorizer("fasttext")
    with pytest.raises(ValueError, match="embedding must be one of"):
        MeanEmbeddingVectorizer(embedding="fasttext").fit(TEXTS)


@pytest.mark.parametrize("embedding", EMBEDDINGS)
def test_build_vectorizer_accepts_every_supported_embedding(embedding):
    assert build_vectorizer(embedding).embedding == embedding


# --------------------------------------------------------------------------
# self-trained Word2Vec
# --------------------------------------------------------------------------
def test_word2vec_trains_on_the_supplied_text_only():
    """Fitting inside fit() is what stops test vocabulary leaking in."""
    vectors = train_word2vec(TEXTS, domain="se", mode="none", min_count=1, epochs=5)
    assert "process" in vectors
    assert "nausea" not in vectors    # never appeared in TEXTS


def test_word2vec_is_reproducible():
    first = train_word2vec(TEXTS, domain="se", mode="none", min_count=1, epochs=5)
    second = train_word2vec(TEXTS, domain="se", mode="none", min_count=1, epochs=5)
    assert np.allclose(first["process"], second["process"])


def test_word2vec_respects_the_requested_dimension():
    vectors = train_word2vec(TEXTS, domain="se", mode="none",
                             min_count=1, epochs=5, vector_size=25)
    assert vectors.vector_size == 25


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def test_oov_rate_counts_tokens_and_empty_documents():
    vectorizer = vectorizer_with(fake_vectors())
    stats = vectorizer.oov_rate(["kill process", "zzzz qqqq"])
    assert stats["tokens"] == 4
    assert stats["oov_token_rate"] == 0.5
    assert stats["documents_with_no_known_tokens"] == 1


def test_corpus_counts_reports_zero_for_absent_words():
    """The diagnostic that explains why rare words get poor vectors."""
    vectors = train_word2vec(TEXTS, domain="se", mode="none", min_count=1, epochs=5)
    counts = corpus_counts(vectors, ["process", "abort"])
    assert counts["process"] > 0
    assert counts["abort"] == 0


def test_neighbour_table_marks_out_of_vocabulary_probes():
    table = neighbour_table({"w2v": fake_vectors()}, words=["kill", "abort"])
    by_word = table.set_index("word")["neighbours"]
    assert by_word["abort"] == "(out of vocabulary)"
    assert by_word["kill"] != "(out of vocabulary)"


def test_neighbour_table_covers_every_embedding():
    table = neighbour_table(
        {"a": fake_vectors(), "b": fake_vectors()}, words=["kill"])
    assert set(table["embedding"]) == {"a", "b"}
