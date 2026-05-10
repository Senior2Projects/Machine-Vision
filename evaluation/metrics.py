"""
metrics.py
----------
All evaluation metrics implemented from scratch using NumPy only.

Metrics
-------
    accuracy            — overall classification accuracy
    confusion_matrix    — C×C matrix of predicted vs true labels
    precision           — per-class true positives / predicted positives
    recall              — per-class true positives / actual positives
    f1_score            — per-class harmonic mean of precision and recall
    macro_f1            — unweighted mean of per-class F1
    weighted_f1         — class-frequency-weighted mean of per-class F1
    classification_report — full summary table as string

Usage
-----
    from evaluation.metrics import evaluate, classification_report

    results = evaluate(y_true, y_pred, class_names=CLASS_NAMES)
    print(classification_report(results))
"""

import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Overall classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray  (N,)  ground-truth integer class labels
    y_pred : np.ndarray  (N,)  predicted integer class labels

    Returns
    -------
    float  fraction of correct predictions in [0, 1]
    """
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = None,
) -> np.ndarray:
    """
    Compute the confusion matrix.

    C[i, j] = number of samples with true label i predicted as label j.

    Parameters
    ----------
    y_true   : np.ndarray  (N,)  ground-truth labels
    y_pred   : np.ndarray  (N,)  predicted labels
    n_classes: int | None        inferred from data if None

    Returns
    -------
    np.ndarray  int64 (C, C)

    Notes
    -----
    Row = true class, Column = predicted class.
    Perfect classifier → diagonal matrix.
    """
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(
    cm: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-class precision, recall, and F1 from a confusion matrix.

    Formulas
    --------
    precision[i] = TP[i] / (TP[i] + FP[i])   = cm[i,i] / cm[:,i].sum()
    recall[i]    = TP[i] / (TP[i] + FN[i])   = cm[i,i] / cm[i,:].sum()
    f1[i]        = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    Parameters
    ----------
    cm  : np.ndarray  (C, C)  confusion matrix
    eps : float               division guard

    Returns
    -------
    precision : np.ndarray  (C,)
    recall    : np.ndarray  (C,)
    f1        : np.ndarray  (C,)
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp          # column sum minus diagonal
    fn = cm.sum(axis=1) - tp          # row sum minus diagonal

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def macro_f1(f1_per_class: np.ndarray) -> float:
    """
    Macro-averaged F1: unweighted mean across all classes.

    Parameters
    ----------
    f1_per_class : np.ndarray  (C,)

    Returns
    -------
    float
    """
    return float(np.mean(f1_per_class))


def weighted_f1(f1_per_class: np.ndarray, y_true: np.ndarray) -> float:
    """
    Weighted-averaged F1: weighted by class support (true frequency).

    Parameters
    ----------
    f1_per_class : np.ndarray  (C,)
    y_true       : np.ndarray  (N,)

    Returns
    -------
    float
    """
    n_classes = len(f1_per_class)
    support   = np.array([(y_true == c).sum() for c in range(n_classes)],
                         dtype=np.float64)
    weights   = support / support.sum()
    return float(np.sum(f1_per_class * weights))


# ---------------------------------------------------------------------------
# All-in-one evaluate
# ---------------------------------------------------------------------------

def evaluate(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list[str] = None,
    n_classes:   int       = None,
) -> dict:
    """
    Compute the full evaluation suite from scratch.

    Parameters
    ----------
    y_true      : np.ndarray  (N,)  ground-truth labels
    y_pred      : np.ndarray  (N,)  predicted labels
    class_names : list[str] | None  names for each class index
    n_classes   : int | None        inferred if None

    Returns
    -------
    dict with keys:
        accuracy      : float
        confusion_matrix : np.ndarray (C, C)
        precision     : np.ndarray (C,)
        recall        : np.ndarray (C,)
        f1            : np.ndarray (C,)
        macro_f1      : float
        weighted_f1   : float
        class_names   : list[str]
        support       : np.ndarray (C,)  number of true samples per class
    """
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    acc = accuracy(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, n_classes)
    prec, rec, f1 = precision_recall_f1(cm)
    mf1 = macro_f1(f1)
    wf1 = weighted_f1(f1, y_true)

    support = cm.sum(axis=1)

    return {
        "accuracy":        acc,
        "confusion_matrix": cm,
        "precision":       prec,
        "recall":          rec,
        "f1":              f1,
        "macro_f1":        mf1,
        "weighted_f1":     wf1,
        "class_names":     class_names,
        "support":         support,
    }


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def classification_report(results: dict) -> str:
    """
    Format evaluation results as a readable classification report.

    Parameters
    ----------
    results : dict  output of evaluate()

    Returns
    -------
    str  formatted report
    """
    names   = results["class_names"]
    prec    = results["precision"]
    rec     = results["recall"]
    f1      = results["f1"]
    support = results["support"]

    col_w = max(len(n) for n in names) + 2
    header = (f"{'class':<{col_w}}  {'precision':>10}  {'recall':>10}"
              f"  {'f1-score':>10}  {'support':>8}")
    sep    = "-" * len(header)

    lines = [sep, header, sep]
    for i, name in enumerate(names):
        lines.append(
            f"{name:<{col_w}}  {prec[i]:>10.4f}  {rec[i]:>10.4f}"
            f"  {f1[i]:>10.4f}  {int(support[i]):>8}"
        )

    lines += [
        sep,
        f"{'macro avg':<{col_w}}  {'':>10}  {'':>10}  "
        f"{results['macro_f1']:>10.4f}  {int(support.sum()):>8}",
        f"{'weighted avg':<{col_w}}  {'':>10}  {'':>10}  "
        f"{results['weighted_f1']:>10.4f}  {int(support.sum()):>8}",
        sep,
        f"\nOverall Accuracy: {results['accuracy']:.4f}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confusion matrix plotter
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    results:    dict,
    title:      str  = "Confusion Matrix",
    normalize:  bool = True,
    save_path:  str  = None,
) -> None:
    """
    Plot the confusion matrix using matplotlib.

    Parameters
    ----------
    results    : dict   output of evaluate()
    title      : str    plot title
    normalize  : bool   show row-normalized percentages (default True)
    save_path  : str    if provided, save figure to this path
    """
    import matplotlib.pyplot as plt

    cm     = results["confusion_matrix"].astype(np.float64)
    names  = results["class_names"]
    C      = len(names)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = cm / (row_sums + 1e-8)
        fmt      = ".2f"
        vmin, vmax = 0, 1
    else:
        cm_plot = cm
        fmt     = "d"
        vmin, vmax = 0, cm.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_plot, interpolation="nearest",
                   cmap="Blues", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    thresh = (cm_plot.max() + cm_plot.min()) / 2.0
    for i in range(C):
        for j in range(C):
            val = cm_plot[i, j]
            txt = f"{val:{fmt}}" if fmt == ".2f" else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if val > thresh else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# CLI — evaluate any model's saved predictions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick test with dummy data to verify metrics are correct.
    """
    CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    np.random.seed(42)

    N = 900
    y_true = np.random.randint(0, 6, N)
    y_pred = np.where(np.random.rand(N) > 0.3, y_true,
                      np.random.randint(0, 6, N))

    results = evaluate(y_true, y_pred, class_names=CLASS_NAMES)

    print(classification_report(results))
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    plot_confusion_matrix(results, title="Test — Dummy Predictions", normalize=True)