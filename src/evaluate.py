"""
Evaluation utilities for the Cornell Movie Review sentiment analysis pipeline.

Provides:

* :func:`evaluateModel` — prints accuracy, precision, recall, F1, and the
  confusion matrix numerically, and returns a metrics dict.
* :func:`plotConfusionMatrix` — renders a ``ConfusionMatrixDisplay`` plot.
* :func:`log_to_wandb` — optional Weights & Biases experiment tracking.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluateModel(yTrue, predictions, model_name=""):
    """
    Print classification metrics and the raw confusion matrix.

    Parameters
    ----------
    yTrue : array-like
        Ground-truth labels.
    predictions : array-like
        Predicted labels returned by a classifier.
    model_name : str, optional
        Label included in the output header for readability.

    Returns
    -------
    dict
        Dictionary with keys ``"accuracy"``, ``"precision"``, ``"recall"``,
        and ``"f1"``, each mapped to its float value.
    """
    accuracy = accuracy_score(yTrue, predictions)
    precision = precision_score(yTrue, predictions)
    recall = recall_score(yTrue, predictions)
    f1 = f1_score(yTrue, predictions)
    matrix = confusion_matrix(yTrue, predictions)

    header = f" — {model_name}" if model_name else ""
    print(f"=== Evaluation{header} ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


def plotConfusionMatrix(yTrue, predictions, model_name=""):
    """
    Display an interactive confusion-matrix plot using ``ConfusionMatrixDisplay``.

    Parameters
    ----------
    yTrue : array-like
        Ground-truth labels.
    predictions : array-like
        Predicted labels returned by a classifier.
    model_name : str, optional
        Title shown on the axes.
    """
    matrix = confusion_matrix(yTrue, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    if model_name:
        disp.ax_.set_title(model_name)


def log_to_wandb(metrics_dict, config_dict, project="sentiment-analysis-nlp"):
    """
    Log metrics and run configuration to Weights & Biases (optional).

    If ``wandb`` is not installed this function prints a helpful installation
    message and returns without raising an error, so it is safe to include
    in notebooks without making wandb a hard dependency.

    Parameters
    ----------
    metrics_dict : dict
        Metric key-value pairs to log, e.g.
        ``{"accuracy": 0.84, "f1": 0.85}``.
    config_dict : dict
        Run configuration to record, e.g.
        ``{"model": "naive_bayes_scratch", "feature_set": "unigrams_tfidf"}``.
    project : str, optional
        Weights & Biases project name.  Defaults to ``"sentiment-analysis-nlp"``.
    """
    try:
        import wandb
        wandb.init(project=project, config=config_dict)
        wandb.log(metrics_dict)
        wandb.finish()
    except ImportError:
        print("wandb not installed. Run: pip install wandb")
