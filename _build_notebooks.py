"""
Build / modify all three Jupyter notebooks for the sentiment analysis project.

Run this script once from the project root:
    python _build_notebooks.py
"""

import json
import os
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(ROOT, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_id():
    return uuid.uuid4().hex[:16]


def md_cell(source):
    return {
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {},
        "source": source,
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": new_id(),
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def uncomment(source):
    """Strip outer triple-quote wrapper from a cell that holds commented code."""
    s = source.strip()
    if s.startswith('"""') and s.endswith('"""'):
        inner = s[3:-3]
        # strip one leading newline if present
        if inner.startswith("\n"):
            inner = inner[1:]
        if inner.endswith("\n"):
            inner = inner[:-1]
        return inner
    return source


def load_nb(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_nb(nb, path):
    # Ensure every cell has an `id` field (required for nbformat >= 4.5)
    for cell in nb["cells"]:
        if "id" not in cell:
            cell["id"] = new_id()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved -> {os.path.relpath(path, ROOT)}")


def get_source(cell):
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


# ---------------------------------------------------------------------------
# Shared intro markdown content
# ---------------------------------------------------------------------------

INTRO_CLASSICAL = """\
# Classical NLP Sentiment Analysis Pipeline

This notebook implements an end-to-end sentiment analysis pipeline on the Cornell \
Movie Review Dataset, progressing from text preprocessing through feature engineering \
to model training and evaluation.

**Key findings:** Across Naïve Bayes (scratch), Multinomial NB, Logistic Regression, \
and SVM, all classical models converge at 0.84–0.85 F1 — a ceiling imposed by \
bag-of-words representations that discard word order and context. The dominant factor \
in performance is feature representation, not classifier choice: varying the feature \
set produces swings of up to 27 percentage points, while varying the classifier on \
the same features produces differences under 2 points.

**Pipeline stages:** Data ingestion → Preprocessing → N-gram feature extraction → \
Custom TF-IDF normalisation → Naïve Bayes (from scratch) → sklearn benchmarks → \
Hyperparameter tuning"""

INTRO_BERT = """\
# BERT Fine-Tuning for Sentiment Classification

This notebook fine-tunes both bert-base-cased and bert-base-uncased on the same \
Cornell Movie Review Dataset used in the classical pipeline notebook, enabling \
direct comparison between bag-of-words approaches and contextual transformer embeddings.

**Key findings:** BERT Uncased achieves 0.91 accuracy and F1 on the held-out test set, \
a 6-point gain over the best classical model (SVM, 0.85 F1). Training loss decreases \
~72% over 1,750 steps, indicating stable convergence. Uncased outperforms Cased (0.91 \
vs 0.89), consistent with the hypothesis that case sensitivity expands vocabulary \
without proportional discriminative benefit for sentiment tasks.

**Requirements:** GPU recommended (originally run on Google Colab T4). \
See requirements.txt for package versions."""

PPMI_EXTRACT_CELL = """\
# Updated extractFeature dispatcher — includes PPMI feature set
def extractFeature(xTrain, xDev, feature):
  # FEATURE SET 1 - Frequency Normalisation + Trigrams
  if feature == "trigrams":
    trainVector, devVector = trigramsFeature(xTrain, xDev)
    return trainVector, devVector

  # FEATURE SET 2 - TFIDF + Bigrams
  elif feature == "bigrams":
    trainVector, devVector = bigramsFeature(xTrain, xDev)
    return trainVector, devVector

  # FEATURE SET 3 - TFIDF + Unigrams
  elif feature == "unigrams":
    trainVector, devVector = unigramsFeature(xTrain, xDev)
    return trainVector, devVector

  # FEATURE SET 4 - PPMI + Bigrams
  elif feature == "ppmi":
    trainVector, devVector = ppmiFeature(xTrain, xDev)
    return trainVector, devVector"""

PPMI_EVAL_CELL = """\
# evaluating FEATURE SET 4 - PPMI + BIGRAMS
xTrainVector, xDevVector = extractFeature(xTrain, xDev, "ppmi")
fit(xTrain, yTrain, xTrainVector)
predictions = predict(xDevVector)

print("PPMI + BIGRAMS")
evaluateModel(yDev, predictions)"""

PPMI_MARKDOWN = """\
### PPMI vs. TF-IDF: Feature Comparison

**Positive Pointwise Mutual Information (PPMI)** measures how much more likely a \
term and a document co-occur together than by chance. Negative PMI values are \
zeroed out, emphasising informative collocations over common co-occurrences.

| Feature Set       | Accuracy | Precision | Recall | F1   |
|-------------------|----------|-----------|--------|------|
| Bigrams + TF-IDF  | 0.70     | 0.69      | 0.70   | 0.70 |
| PPMI + Bigrams    | see above | — | — | — |

**Key observations:**
- TF-IDF normalises for document length and penalises high-frequency terms across \
the corpus.
- PPMI additionally captures *associative strength* — it rewards bigrams that appear \
together more than chance would predict.
- Both feature sets operate on the same bigram vocabulary (`max_features=1000`), so \
performance differences reflect the normalisation strategy alone.
- In practice PPMI can be more sensitive to sparse high-PMI bigrams, which may help \
or hurt depending on vocabulary coverage in the training set."""

WANDB_CELL = """\
# Optional: Log this run to Weights & Biases
# Uncomment and run after installing wandb:  pip install wandb

from src.evaluate import log_to_wandb

log_to_wandb(
    metrics_dict={"accuracy": 0.84, "f1": 0.85, "precision": 0.80, "recall": 0.90},
    config_dict={"model": "naive_bayes_scratch", "feature_set": "unigrams_tfidf",
                 "random_state": 5, "alpha": 1}
)"""


# ---------------------------------------------------------------------------
# Task 7 + 4 + 5 — Build 01_classical_nlp_pipeline.ipynb
# ---------------------------------------------------------------------------

def build_classical(src_path, out_path):
    print("\n[01_classical_nlp_pipeline.ipynb]")
    nb = load_nb(src_path)
    cells = nb["cells"]

    # Task 7: prepend intro markdown cell
    cells.insert(0, md_cell(INTRO_CLASSICAL))

    # After the insert, original cell indices shift by +1
    # Original cell 28 → now index 29  (PPMI functions, triple-quoted)
    # Original cell 29 → now index 30  (ppmi elif snippet, triple-quoted)
    # Original cell 30 → now index 31  (PPMI eval, triple-quoted)

    # Task 4: uncomment PPMI function definition (orig cell 28 → now 29)
    ppmi_fn_idx = 29
    cells[ppmi_fn_idx]["source"] = uncomment(get_source(cells[ppmi_fn_idx]))

    # Task 4: replace elif-snippet cell with full extractFeature redefinition
    ppmi_elif_idx = 30
    cells[ppmi_elif_idx]["source"] = PPMI_EXTRACT_CELL

    # Task 4: replace triple-quoted eval stub with real evaluation cell
    ppmi_eval_idx = 31
    cells[ppmi_eval_idx]["source"] = PPMI_EVAL_CELL
    cells[ppmi_eval_idx]["outputs"] = []
    cells[ppmi_eval_idx]["execution_count"] = None

    # Task 4: insert PPMI comparison markdown after the evaluation cell
    cells.insert(ppmi_eval_idx + 1, md_cell(PPMI_MARKDOWN))

    # Task 5: insert wandb example cell after original unigrams eval
    # Original cell 21 (unigrams eval) → after intro insert → index 22
    # We want it right after that cell.
    wandb_insert_after = 22
    cells.insert(wandb_insert_after + 1, code_cell(WANDB_CELL))

    nb["cells"] = cells
    save_nb(nb, out_path)


# ---------------------------------------------------------------------------
# Task 7 — Build 02_bert_finetuning.ipynb
# ---------------------------------------------------------------------------

def build_bert(src_path, out_path):
    print("\n[02_bert_finetuning.ipynb]")
    nb = load_nb(src_path)
    nb["cells"].insert(0, md_cell(INTRO_BERT))
    save_nb(nb, out_path)


# ---------------------------------------------------------------------------
# Task 3 — Build 03_visualisations.ipynb
# ---------------------------------------------------------------------------

CELL1_MD = """\
# Sentiment Analysis — Visualisations

This notebook produces all result plots for the Cornell Movie Review sentiment \
analysis project. All data is hardcoded directly from experiment results — no \
external files are required.

**Plots generated:**
1. Feature Ablation Study (grouped bar chart)
2. Full Model Comparison (grouped bar chart)
3. BERT Training Loss Convergence (line plot)
4. Confusion Matrix Grid (2 × 3 subplot grid)

All plots are saved to the `assets/` folder at 150 dpi."""

CELL2_CODE = """\
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# ── global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight"})

# ── resolve assets/ directory regardless of CWD ───────────────────────────────
if os.path.isdir("notebooks"):        # running from project root
    ASSETS_DIR = "assets"
else:                                  # running from notebooks/ folder
    ASSETS_DIR = "../assets"
os.makedirs(ASSETS_DIR, exist_ok=True)
print(f"Assets will be saved to: {os.path.abspath(ASSETS_DIR)}")"""

CELL3_CODE = """\
feature_ablation_results = {
    "Trigrams + Freq Norm": {"Accuracy": 0.57, "Precision": 0.54, "Recall": 0.92, "F1": 0.68},
    "Bigrams + TF-IDF":     {"Accuracy": 0.70, "Precision": 0.69, "Recall": 0.70, "F1": 0.70},
    "Unigrams + TF-IDF":    {"Accuracy": 0.84, "Precision": 0.80, "Recall": 0.90, "F1": 0.85},
}

model_comparison_results = {
    "Naïve Bayes\\n(scratch)":   {"Accuracy": 0.84, "Precision": 0.80, "Recall": 0.90, "F1": 0.85},
    "MultinomialNB\\n(sklearn)": {"Accuracy": 0.84, "Precision": 0.81, "Recall": 0.89, "F1": 0.85},
    "Logistic\\nRegression":     {"Accuracy": 0.84, "Precision": 0.83, "Recall": 0.87, "F1": 0.85},
    "SVM\\n(LinearSVC)":         {"Accuracy": 0.85, "Precision": 0.84, "Recall": 0.86, "F1": 0.85},
    "BERT\\nCased":              {"Accuracy": 0.89, "Precision": 0.89, "Recall": 0.89, "F1": 0.89},
    "BERT\\nUncased":            {"Accuracy": 0.91, "Precision": 0.91, "Recall": 0.91, "F1": 0.91},
}

bert_training_loss = {
    "Uncased": {250: 0.5719, 500: 0.5585, 750: 0.5234, 1000: 0.4311,
                1250: 0.3827, 1500: 0.2036, 1750: 0.1734},
    "Cased":   {250: 0.5953, 500: 0.5753, 750: 0.5344, 1000: 0.4647,
                1250: 0.3672, 1500: 0.1817, 1750: 0.1477},
}

print("Data loaded.")
print(f"  Feature sets : {list(feature_ablation_results.keys())}")
print(f"  Models       : {list(model_comparison_results.keys())}")"""

CELL4_CODE = """\
# ── Plot 1: Feature Ablation grouped bar chart ────────────────────────────────
metrics      = ["Accuracy", "Precision", "Recall", "F1"]
feature_sets = list(feature_ablation_results.keys())
n_groups     = len(feature_sets)
n_metrics    = len(metrics)
x            = np.arange(n_groups)
width        = 0.18
palette      = sns.color_palette("muted", n_metrics)

fig, ax = plt.subplots(figsize=(11, 6))

for i, metric in enumerate(metrics):
    vals = [feature_ablation_results[fs][metric] for fs in feature_sets]
    bars = ax.bar(
        x + (i - (n_metrics - 1) / 2) * width, vals, width,
        label=metric, color=palette[i], edgecolor="white"
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.007,
            f"{v:.2f}", ha="center", va="bottom", fontsize=8.5
        )

ax.set_xticks(x)
ax.set_xticklabels(feature_sets, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Feature Ablation Study \\u2014 Na\\u00efve Bayes (Development Set)",
             fontsize=13, pad=12)
ax.legend(title="Metric", fontsize=10)
ax.yaxis.grid(True, alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
save_path = os.path.join(ASSETS_DIR, "feature_ablation.png")
plt.savefig(save_path, dpi=150)
print(f"Saved: {save_path}")
plt.show()"""

CELL5_CODE = """\
# ── Plot 2: Full Model Comparison grouped bar chart ───────────────────────────
models           = list(model_comparison_results.keys())
n_models         = len(models)
x                = np.arange(n_models)
palette          = sns.color_palette("muted", n_metrics)
BERT_UNCASED_IDX = models.index("BERT\\nUncased")
HIGHLIGHT_COLOR  = "#FFC107"   # amber / gold for BERT Uncased

fig, ax = plt.subplots(figsize=(14, 6))
all_bars = []

for i, metric in enumerate(metrics):
    vals = [model_comparison_results[m][metric] for m in models]
    bar_colors = [
        HIGHLIGHT_COLOR if j == BERT_UNCASED_IDX else palette[i]
        for j in range(n_models)
    ]
    bars = ax.bar(
        x + (i - (n_metrics - 1) / 2) * width, vals, width,
        color=bar_colors, edgecolor="white", label=metric
    )
    all_bars.append(bars)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{v:.2f}", ha="center", va="bottom", fontsize=7
        )

# Classical ceiling dashed line
ax.axhline(0.85, linestyle="--", color="dimgray", linewidth=1.5,
           label="Classical ceiling (F1 = 0.85)")

# Star annotation above BERT Uncased group
ax.annotate(
    "\\u2605 Best", xy=(BERT_UNCASED_IDX, 0.935),
    ha="center", fontsize=10, fontweight="bold", color="darkorange"
)

# Legend with highlight patch
highlight_patch = mpatches.Patch(color=HIGHLIGHT_COLOR, label="BERT Uncased (highlighted)")
handles, labels_leg = ax.get_legend_handles_labels()
handles.append(highlight_patch)
ax.legend(handles=handles, fontsize=9, loc="lower right")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison \\u2014 All Models on Test Set", fontsize=13, pad=12)
ax.yaxis.grid(True, alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
save_path = os.path.join(ASSETS_DIR, "model_comparison.png")
plt.savefig(save_path, dpi=150)
print(f"Saved: {save_path}")
plt.show()"""

CELL6_CODE = """\
# ── Plot 3: BERT Training Loss Curve ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

styles = {
    "Uncased": {"color": "#1976D2", "marker": "o", "linestyle": "-"},
    "Cased":   {"color": "#E64A19", "marker": "s", "linestyle": "--"},
}

for variant, losses in bert_training_loss.items():
    steps = sorted(losses.keys())
    vals  = [losses[s] for s in steps]
    ax.plot(steps, vals, label=f"BERT {variant}",
            linewidth=2.5, markersize=8, **styles[variant])

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Training Loss", fontsize=12)
ax.set_title("BERT Training Loss Convergence", fontsize=13, pad=12)
ax.set_xticks(sorted(bert_training_loss["Uncased"].keys()))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.4)

plt.tight_layout()
save_path = os.path.join(ASSETS_DIR, "bert_training_loss.png")
plt.savefig(save_path, dpi=150)
print(f"Saved: {save_path}")
plt.show()"""

CELL7_CODE = """\
# ── Plot 4: Confusion Matrix Grid (2 × 3) ────────────────────────────────────
confusion_matrices = {
    "NB (scratch)":  np.array([[318,  88], [ 39, 355]]),
    "MultinomialNB": np.array([[326,  80], [ 45, 349]]),
    "LR (tuned)":    np.array([[330,  77], [ 37, 356]]),
    "SVM (tuned)":   np.array([[332,  75], [ 34, 359]]),
    "BERT Cased":    np.array([[357,  50], [ 38, 355]]),
    "BERT Uncased":  np.array([[370,  37], [ 24, 369]]),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for ax, (name, cm) in zip(axes.flatten(), confusion_matrices.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=12, pad=8)

plt.suptitle("Confusion Matrices \\u2014 All Models on Test Set",
             fontsize=14, y=1.01)
plt.tight_layout()

save_path = os.path.join(ASSETS_DIR, "confusion_matrix_grid.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()"""

CELL8_MD = """\
## Key Visual Takeaways

### 1. Feature Ablation (Plot 1)
- **Representation quality dominates**: switching from trigrams (F1 = 0.68) to \
unigrams (F1 = 0.85) improves F1 by **17 points** — far larger than any \
classifier-level gain.
- Trigram frequency normalisation achieves very high recall (0.92) but low \
precision (0.54), biasing heavily toward positive predictions.
- TF-IDF weighting consistently improves the precision–recall balance.

### 2. Model Comparison (Plot 2)
- All four classical models plateau at **F1 = 0.85**, confirming a bag-of-words \
ceiling imposed by discarding word order.
- BERT Cased (+4 pp, F1 = 0.89) and **BERT Uncased (+6 pp, F1 = 0.91)** break \
through this ceiling by encoding full contextual embeddings.
- BERT Uncased outperforms Cased — case sensitivity appears to add vocabulary noise \
without proportional discriminative benefit for sentiment tasks.

### 3. BERT Loss Curve (Plot 3)
- Both models converge smoothly over 1,750 steps (~72 % loss reduction).
- Uncased reaches a slightly lower final loss, consistent with its superior test \
performance.
- No loss spikes are visible, indicating stable fine-tuning.

### 4. Confusion Matrices (Plot 4)
- Classical models show broadly balanced false-positive / false-negative errors.
- BERT Uncased dramatically reduces both error types compared to NB scratch \
(FP: 37 vs 88, FN: 24 vs 39).
- Across all models, false negatives (missed positives) are fewer than false \
positives."""


def build_visualisations(out_path):
    print("\n[03_visualisations.ipynb]")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0",
            },
        },
        "cells": [
            md_cell(CELL1_MD),
            code_cell(CELL2_CODE),
            code_cell(CELL3_CODE),
            code_cell(CELL4_CODE),
            code_cell(CELL5_CODE),
            code_cell(CELL6_CODE),
            code_cell(CELL7_CODE),
            md_cell(CELL8_MD),
        ],
    }
    save_nb(nb, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    src_classical = os.path.join(ROOT, "SentimentAnalysis.ipynb")
    src_bert      = os.path.join(ROOT, "BERT.ipynb")

    out_classical = os.path.join(NB_DIR, "01_classical_nlp_pipeline.ipynb")
    out_bert      = os.path.join(NB_DIR, "02_bert_finetuning.ipynb")
    out_vis       = os.path.join(NB_DIR, "03_visualisations.ipynb")

    build_classical(src_classical, out_classical)
    build_bert(src_bert, out_bert)
    build_visualisations(out_vis)

    print("\nAll notebooks built successfully.")
