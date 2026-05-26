"""
Feature extraction module for the Cornell Movie Review sentiment analysis pipeline.

Implements a custom NumPy TF-IDF, PPMI normalisation, and three n-gram feature
sets (trigrams, bigrams, unigrams).  A dispatcher function ``extractFeature``
provides a uniform interface to all feature sets.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.preprocessing import stemmingTokenize, lemmaTokenize


# ---------------------------------------------------------------------------
# Normalisation functions
# ---------------------------------------------------------------------------

def tfidf(featureVector):
    """
    Compute a custom TF-IDF matrix from a raw term-count matrix.

    | TF  = count / (row_total + 1e-9)
    | IDF = log(N / (doc_freq + 1))

    Parameters
    ----------
    featureVector : numpy.ndarray of shape (n_docs, n_terms)
        Raw term-count matrix (non-negative integers).

    Returns
    -------
    numpy.ndarray of shape (n_docs, n_terms)
        TF-IDF weighted matrix.  Shape is identical to the input.
    """
    totalTerms = np.sum(featureVector, axis=1, keepdims=True)
    tf = featureVector / (totalTerms + 1e-9)

    totalDocs = featureVector.shape[0]
    totalDocsWithTerm = np.sum(featureVector > 0, axis=0)
    idf = np.log(totalDocs / (totalDocsWithTerm + 1))

    return tf * idf


def _ppmi(featureVector):
    """
    Compute the Positive PMI matrix from a raw count matrix.

    Negative PMI values are replaced with zero, yielding PPMI.

    Parameters
    ----------
    featureVector : numpy.ndarray of shape (n_docs, n_terms)
        Raw term-count matrix.

    Returns
    -------
    numpy.ndarray of shape (n_docs, n_terms)
        PPMI matrix with all values >= 0.
    """
    totalTexts = featureVector.shape[0]
    totalTerms = featureVector.shape[1]

    probIxJ = featureVector / (totalTexts + 1e-9)
    probI = np.sum(featureVector, axis=0) / totalTexts          # term marginals
    probJ = np.sum(featureVector, axis=1).reshape(-1, 1) / totalTerms  # doc marginals

    # Replace zeros with epsilon to prevent log(0)
    probIxJ[probIxJ == 0] = 1e-9
    probI[probI == 0] = 1e-9
    probJ[probJ == 0] = 1e-9

    ppmiMatrix = np.log2(probIxJ / (probI * probJ))
    ppmiMatrix[ppmiMatrix < 0] = 0  # PPMI: clip negatives to zero
    return ppmiMatrix


# ---------------------------------------------------------------------------
# Feature set builders
# ---------------------------------------------------------------------------

def trigramsFeature(train, dev, trainLengths, devLengths):
    """
    Extract trigram features with frequency normalisation.

    Applies Porter stemming, fits a ``CountVectorizer`` with
    ``ngram_range=(3, 3)`` and ``max_features=1000``, then normalises each
    count by document length.

    Parameters
    ----------
    train : list of str
        Raw training texts.
    dev : list of str
        Raw development / test texts.
    trainLengths : numpy.ndarray of shape (n_train,)
        Character lengths of each training document.
    devLengths : numpy.ndarray of shape (n_dev,)
        Character lengths of each development / test document.

    Returns
    -------
    trainVector : numpy.ndarray
        Frequency-normalised trigram matrix for training data.
    devVector : numpy.ndarray
        Frequency-normalised trigram matrix for dev / test data.
    """
    trainText = stemmingTokenize(train)
    devText = stemmingTokenize(dev)

    vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=1000)
    trainVector = vectorizer.fit_transform(trainText).toarray()
    devVector = vectorizer.transform(devText).toarray()

    trainVector = trainVector / (trainLengths * 3)[:, None]
    devVector = devVector / (devLengths * 3)[:, None]

    return trainVector, devVector


def bigramsFeature(train, dev):
    """
    Extract bigram features with custom TF-IDF normalisation.

    Applies Porter stemming, fits a ``CountVectorizer`` with
    ``ngram_range=(2, 2)`` and ``max_features=1000``, then applies
    the custom :func:`tfidf` normalisation.

    Parameters
    ----------
    train : list of str
        Raw training texts.
    dev : list of str
        Raw development / test texts.

    Returns
    -------
    trainVector : numpy.ndarray
        TF-IDF normalised bigram matrix for training data.
    devVector : numpy.ndarray
        TF-IDF normalised bigram matrix for dev / test data.
    """
    trainText = stemmingTokenize(train)
    devText = stemmingTokenize(dev)

    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=1000)
    trainV = vectorizer.fit_transform(trainText).toarray()
    devV = vectorizer.transform(devText).toarray()

    return tfidf(trainV), tfidf(devV)


def unigramsFeature(train, dev):
    """
    Extract unigram features with custom TF-IDF normalisation.

    Applies WordNet lemmatisation, fits a ``CountVectorizer`` with
    ``ngram_range=(1, 1)`` and no vocabulary cap, then applies the
    custom :func:`tfidf` normalisation.

    Parameters
    ----------
    train : list of str
        Raw training texts.
    dev : list of str
        Raw development / test texts.

    Returns
    -------
    trainVector : numpy.ndarray
        TF-IDF normalised unigram matrix for training data.
    devVector : numpy.ndarray
        TF-IDF normalised unigram matrix for dev / test data.
    """
    trainText = lemmaTokenize(train)
    devText = lemmaTokenize(dev)

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    trainV = vectorizer.fit_transform(trainText).toarray()
    devV = vectorizer.transform(devText).toarray()

    return tfidf(trainV), tfidf(devV)


def ppmiFeature(train, dev):
    """
    Extract bigram features with Positive Pointwise Mutual Information (PPMI).

    Applies WordNet lemmatisation, fits a ``CountVectorizer`` with
    ``ngram_range=(2, 2)`` and ``max_features=1000``, then applies PPMI
    normalisation (negative values replaced with 0).

    Parameters
    ----------
    train : list of str
        Raw training texts.
    dev : list of str
        Raw development / test texts.

    Returns
    -------
    trainVector : numpy.ndarray
        PPMI normalised bigram matrix for training data (all values >= 0).
    devVector : numpy.ndarray
        PPMI normalised bigram matrix for dev / test data (all values >= 0).
    """
    trainText = lemmaTokenize(train)
    devText = lemmaTokenize(dev)

    vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=1000)
    trainV = vectorizer.fit_transform(trainText).toarray().astype(float)
    devV = vectorizer.transform(devText).toarray().astype(float)

    return _ppmi(trainV), _ppmi(devV)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extractFeature(xTrain, xDev, feature, trainLengths=None, devLengths=None):
    """
    Dispatch to the appropriate feature extractor.

    Parameters
    ----------
    xTrain : list of str
        Raw training texts.
    xDev : list of str
        Raw development or test texts.
    feature : str
        Feature set identifier.  One of:
        ``"trigrams"``, ``"bigrams"``, ``"unigrams"``, ``"ppmi"``.
    trainLengths : numpy.ndarray, optional
        Character lengths of training documents.  Required for ``"trigrams"``.
    devLengths : numpy.ndarray, optional
        Character lengths of dev / test documents.  Required for ``"trigrams"``.

    Returns
    -------
    trainVector : numpy.ndarray
        Feature matrix for training data.
    devVector : numpy.ndarray
        Feature matrix for dev / test data.

    Raises
    ------
    ValueError
        If an unrecognised *feature* string is supplied.
    """
    if feature == "trigrams":
        return trigramsFeature(xTrain, xDev, trainLengths, devLengths)
    elif feature == "bigrams":
        return bigramsFeature(xTrain, xDev)
    elif feature == "unigrams":
        return unigramsFeature(xTrain, xDev)
    elif feature == "ppmi":
        return ppmiFeature(xTrain, xDev)
    else:
        raise ValueError(
            f"Unknown feature type '{feature}'. "
            "Choose from: trigrams, bigrams, unigrams, ppmi."
        )
