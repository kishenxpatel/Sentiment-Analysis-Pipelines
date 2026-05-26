"""
Shared pytest fixtures for the sentiment analysis test suite.
"""

import sys
import os

# Ensure the project root is on the Python path so `from src.X import Y` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

# Download required NLTK data quietly (needed by src.preprocessing)
import nltk
for _pkg in ("stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


@pytest.fixture
def synthetic_data():
    """
    Small synthetic dataset for fast unit testing.

    Returns a dict with:

    * ``X``       — float64 array of shape (50, 20), non-negative integer-like counts
    * ``y``       — int array of shape (50,), binary labels {0, 1}
    * ``X_train`` — first 40 rows of X
    * ``y_train`` — first 40 labels
    * ``X_test``  — last 10 rows of X
    * ``y_test``  — last 10 labels
    """
    rng = np.random.RandomState(42)
    X = rng.randint(0, 10, size=(50, 20)).astype(float)
    y = rng.randint(0, 2, size=50)

    return {
        "X":       X,
        "y":       y,
        "X_train": X[:40],
        "y_train": y[:40],
        "X_test":  X[40:],
        "y_test":  y[40:],
    }
