"""
Classification models for the Cornell Movie Review sentiment analysis pipeline.

Provides:

* A from-scratch Naïve Bayes classifier with Laplace smoothing (``fit`` / ``predict``).
* Wrapper functions for scikit-learn's MultinomialNB, LogisticRegression, and LinearSVC.

The scratch NB stores its learned parameters in module-level dicts so that
``predict`` can be called independently after ``fit``, matching the original
notebook's interface.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB as _MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as _LogisticRegression

# ---------------------------------------------------------------------------
# Module-level state for the scratch Naïve Bayes classifier
# ---------------------------------------------------------------------------

logClassPriors: dict = {}
logCondProbs: dict = {}
_alpha = 1  # Laplace smoothing coefficient


# ---------------------------------------------------------------------------
# Scratch Naïve Bayes
# ---------------------------------------------------------------------------

def fit(xTrain, yTrain, featureVector):
    """
    Train the from-scratch Naïve Bayes classifier.

    Uses Laplace smoothing (alpha = 1) and stores class log-priors and
    class-conditional log-probabilities in module-level dicts so that
    :func:`predict` can be called afterwards without passing model objects.

    Parameters
    ----------
    xTrain : list of str or None
        Raw training texts.  Not used internally; present for API consistency
        with the notebook interface.
    yTrain : array-like of int
        Training labels (e.g. 0 = negative, 1 = positive).
    featureVector : numpy.ndarray of shape (n_train, n_features)
        Pre-computed feature matrix corresponding to *xTrain*.
    """
    global logClassPriors, logCondProbs

    yTrain = np.array(yTrain)
    totalLabels = len(yTrain)
    classLabels = np.unique(yTrain)

    for label in classLabels:
        classIndex = (yTrain == label)
        classCount = int(np.sum(classIndex))

        # Class prior — log P(class)
        logClassPriors[label] = np.log(
            (classCount + _alpha) / (totalLabels + _alpha * len(classLabels))
        )

        # Class-conditional log-probabilities with Laplace smoothing
        num = featureVector[classIndex, :].sum(axis=0) + _alpha
        den = classCount + _alpha * featureVector.shape[1]
        logCondProbs[label] = np.log(num / den)


def predict(xTest):
    """
    Predict class labels for a feature matrix using the trained Naïve Bayes model.

    Must be called after :func:`fit`.

    Parameters
    ----------
    xTest : numpy.ndarray of shape (n_samples, n_features)
        Feature matrix for the data to classify.

    Returns
    -------
    list of int
        Predicted class label for each sample.
    """
    predictions = []
    for text in xTest:
        classLikelihoods = {
            label: logClassPriors[label] + np.sum(logCondProbs[label] * text)
            for label in logClassPriors
        }
        predictions.append(max(classLikelihoods, key=classLikelihoods.get))
    return predictions


# ---------------------------------------------------------------------------
# scikit-learn wrappers
# ---------------------------------------------------------------------------

def MNB(xTrainVector, yTrain, xTestVector):
    """
    Train and predict with scikit-learn's ``MultinomialNB``.

    Uses default parameters (``alpha=1.0``, ``fit_prior=True``).

    Parameters
    ----------
    xTrainVector : numpy.ndarray
        Training feature matrix (non-negative values required by MultinomialNB).
    yTrain : array-like
        Training labels.
    xTestVector : numpy.ndarray
        Test feature matrix.

    Returns
    -------
    numpy.ndarray
        Predicted labels for each test sample.
    """
    classifier = _MultinomialNB()
    classifier.fit(xTrainVector, yTrain)
    return classifier.predict(xTestVector)


def LR(xTrainVector, yTrain, xDevVector, **kwargs):
    """
    Train and predict with scikit-learn's ``LogisticRegression``.

    Parameters
    ----------
    xTrainVector : numpy.ndarray
        Training feature matrix.
    yTrain : array-like
        Training labels.
    xDevVector : numpy.ndarray
        Dev / test feature matrix.
    **kwargs
        Additional hyperparameters forwarded to ``LogisticRegression``
        (e.g. ``C=1.0``, ``max_iter=1000``).

    Returns
    -------
    numpy.ndarray
        Predicted labels for each dev / test sample.
    """
    classifier = _LogisticRegression(random_state=42, **kwargs)
    classifier.fit(xTrainVector, yTrain)
    return classifier.predict(xDevVector)


def SVM(xTrainVector, yTrain, xDevVector, **kwargs):
    """
    Train and predict with scikit-learn's ``LinearSVC``.

    Parameters
    ----------
    xTrainVector : numpy.ndarray
        Training feature matrix.
    yTrain : array-like
        Training labels.
    xDevVector : numpy.ndarray
        Dev / test feature matrix.
    **kwargs
        Additional hyperparameters forwarded to ``LinearSVC``
        (e.g. ``C=1.0``, ``max_iter=2000``).

    Returns
    -------
    numpy.ndarray
        Predicted labels for each dev / test sample.
    """
    classifier = LinearSVC(**kwargs)
    classifier.fit(xTrainVector, yTrain)
    return classifier.predict(xDevVector)
