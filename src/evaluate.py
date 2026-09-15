"""Evaluation metrics and diagnostic plots for binary failure-risk classifiers."""

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute the core classification metrics for an imbalanced binary problem.

    Accuracy is intentionally omitted from the headline metrics: at a 3.4%
    positive rate, a model that always predicts "no failure" scores ~96.6%
    accuracy while being useless. Precision/recall/F1/ROC-AUC/PR-AUC are
    reported instead.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels (thresholded at 0.5).
        y_proba: Predicted probability of the positive (failure) class.

    Returns:
        Dict of metric name to value.
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, out_dir: Path = FIGURES_DIR
) -> None:
    """Save a confusion matrix plot for a model.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        model_name: Used in the title and output filename.
        out_dir: Directory to save the figure into.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Failure", "Failure"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion matrix: {model_name}")
    fig.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    fig.savefig(out_dir / f"confusion_matrix_{slug}.png", dpi=150)
    plt.close(fig)


def plot_pr_curve(
    results: Dict[str, Dict[str, np.ndarray]], out_dir: Path = FIGURES_DIR
) -> None:
    """Save a combined precision-recall curve comparing multiple models.

    PR curves are more informative than ROC curves here: with ~96.6% of
    rows negative, ROC-AUC can look deceptively high while precision at
    usable recall levels is still poor.

    Args:
        results: Mapping of model name to a dict with keys ``y_true`` and
            ``y_proba``.
        out_dir: Directory to save the figure into.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        PrecisionRecallDisplay.from_predictions(
            r["y_true"], r["y_proba"], name=name, ax=ax
        )
    baseline = np.mean(next(iter(results.values()))["y_true"])
    ax.axhline(baseline, color="gray", linestyle="--", label=f"No-skill ({baseline:.3f})")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Precision-Recall curves")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(
    results: Dict[str, Dict[str, np.ndarray]], out_dir: Path = FIGURES_DIR
) -> None:
    """Save a combined ROC curve comparing multiple models.

    Args:
        results: Mapping of model name to a dict with keys ``y_true`` and
            ``y_proba``.
        out_dir: Directory to save the figure into.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        RocCurveDisplay.from_predictions(r["y_true"], r["y_proba"], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ROC curves")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_metric_comparison(
    metrics_df: pd.DataFrame, out_dir: Path = FIGURES_DIR
) -> None:
    """Save a grouped bar chart comparing precision/recall/F1/PR-AUC across models.

    Args:
        metrics_df: Dataframe indexed by model name with metric columns
            (as produced by concatenating ``compute_metrics`` results).
        out_dir: Directory to save the figure into.
    """
    cols = ["precision", "recall", "f1", "pr_auc"]
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_df[cols].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def evaluate_by_failure_type(
    y_pred: np.ndarray,
    failure_type: pd.Series,
    target: pd.Series,
) -> pd.DataFrame:
    """Break down recall by specific failure type for rows that are true failures.

    Args:
        y_pred: Predicted binary labels, aligned with ``failure_type``/``target``.
        failure_type: The ``Failure Type`` column for the same rows.
        target: The binary ``Target`` column for the same rows (used to
            restrict to actual failures, since ``Failure Type`` is
            "No Failure" for negatives).

    Returns:
        Dataframe with columns ``support`` (number of true failures of that
        type) and ``recall`` (fraction correctly flagged as a failure),
        indexed by failure type.
    """
    df = pd.DataFrame(
        {"failure_type": failure_type.values, "target": target.values, "y_pred": y_pred}
    )
    df = df[df["target"] == 1]
    grouped = df.groupby("failure_type").apply(
        lambda g: pd.Series({"support": len(g), "recall": (g["y_pred"] == 1).mean()}),
        include_groups=False,
    )
    return grouped.sort_values("support", ascending=False)
