"""Tests for src/evaluate.py."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compute_metrics, evaluate_by_failure_type


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.1, 0.9, 1.0])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_all_wrong_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.9, 0.1, 0.1])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_compute_metrics_no_positive_predictions_no_division_error():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_evaluate_by_failure_type_recall_per_type():
    y_pred = np.array([1, 0, 1, 0, 0])
    target = pd.Series([1, 1, 1, 1, 0])
    failure_type = pd.Series(
        ["Power Failure", "Power Failure", "Tool Wear Failure", "Tool Wear Failure", "No Failure"]
    )
    result = evaluate_by_failure_type(y_pred, failure_type, target)
    assert result.loc["Power Failure", "support"] == 2
    assert result.loc["Power Failure", "recall"] == pytest.approx(0.5)
    assert result.loc["Tool Wear Failure", "support"] == 2
    assert result.loc["Tool Wear Failure", "recall"] == pytest.approx(0.5)
    # Negative rows (target == 0) must not leak into the breakdown.
    assert "No Failure" not in result.index
