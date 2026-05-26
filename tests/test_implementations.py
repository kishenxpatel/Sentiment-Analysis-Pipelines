"""
Unit tests for src.features and src.models implementations.

Run with:
    pytest tests/test_implementations.py -v
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from src.features import tfidf, ppmiFeature
from src.models import fit, predict


# ---------------------------------------------------------------------------
# TF-IDF tests
# ---------------------------------------------------------------------------

def test_tfidf_output_shape(synthetic_data):
    """tfidf() must return an array with the same shape as its input."""
    X = synthetic_data["X"]
    result = tfidf(X)
    assert result.shape == X.shape, (
        f"Expected shape {X.shape}, got {result.shape}"
    )


def test_tfidf_non_negative(synthetic_data):
    """All TF-IDF values must be >= 0."""
    X = synthetic_data["X"]
    result = tfidf(X)
    assert np.all(result >= 0), (
        f"tfidf() returned negative values; min={result.min():.6f}"
    )


# ---------------------------------------------------------------------------
# Naïve Bayes tests
# ---------------------------------------------------------------------------

def test_nb_predict_returns_correct_length(synthetic_data):
    """predict() must return the same number of predictions as test samples."""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    X_test = synthetic_data["X_test"]

    fit(None, y_train, X_train)
    preds = predict(X_test)

    assert len(preds) == len(X_test), (
        f"Expected {len(X_test)} predictions, got {len(preds)}"
    )


def test_nb_matches_sklearn(synthetic_data):
    """
    Scratch NB and sklearn MultinomialNB accuracy must be within 0.05 of
    each other on the same synthetic integer-count feature matrix.
    """
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    X_test  = synthetic_data["X_test"]
    y_test  = synthetic_data["y_test"]

    # Scratch NB
    fit(None, y_train, X_train)
    scratch_preds = predict(X_test)
    scratch_acc   = accuracy_score(y_test, scratch_preds)

    # sklearn MultinomialNB (same Laplace smoothing: alpha=1)
    sk_clf   = MultinomialNB(alpha=1.0)
    sk_clf.fit(X_train, y_train)
    sk_preds = sk_clf.predict(X_test)
    sk_acc   = accuracy_score(y_test, sk_preds)

    assert abs(scratch_acc - sk_acc) <= 0.05, (
        f"Accuracy gap too large: scratch={scratch_acc:.3f}, sklearn={sk_acc:.3f}"
    )


# ---------------------------------------------------------------------------
# PPMI tests
# ---------------------------------------------------------------------------

def test_ppmi_non_negative():
    """PPMI feature output must contain no negative values for both splits."""
    train = [
        "good film great story wonderful",
        "bad movie terrible acting dull",
        "wonderful plot excellent cast direction",
        "boring slow film weak script",
        "amazing performances great cinematography",
        "poor dialogue flat characters story",
    ]
    dev = [
        "good movie great acting",
        "bad film boring plot",
    ]

    train_vec, dev_vec = ppmiFeature(train, dev)

    assert np.all(train_vec >= 0), (
        f"PPMI train matrix has negative values; min={train_vec.min():.6f}"
    )
    assert np.all(dev_vec >= 0), (
        f"PPMI dev matrix has negative values; min={dev_vec.min():.6f}"
    )
