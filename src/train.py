"""Train and compare failure-risk classifiers.

Trains four models, all with explicit handling for the ~3.4% failure rate:

1. Logistic Regression (``class_weight="balanced"``) — interpretable baseline.
2. Random Forest (``class_weight="balanced"``).
3. Random Forest + SMOTE oversampling (an imblearn pipeline) — included to
   compare class-weighting against synthetic minority oversampling, an
   approach this project's author has prior experience with.
4. XGBoost (``scale_pos_weight`` set to the negative:positive ratio).

Saves every fitted pipeline to ``models/``, writes a JSON metrics summary,
and generates comparison figures in ``figures/``.

Run as a script:

    python -m src.train
"""

import json

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE
from src.data import load_processed_data
from src.evaluate import (
    compute_metrics,
    evaluate_by_failure_type,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_pr_curve,
    plot_roc_curve,
)
from src.preprocessing import FEATURE_COLUMNS, build_preprocessor, split_data


def build_models(neg_pos_ratio: float) -> dict:
    """Construct the full preprocessing+model pipelines to train.

    Args:
        neg_pos_ratio: Ratio of negative to positive training examples,
            used as ``scale_pos_weight`` for XGBoost.

    Returns:
        Mapping of model name to an unfitted sklearn/imblearn ``Pipeline``.
    """
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Random Forest + SMOTE": ImbPipeline(
            [
                ("preprocess", build_preprocessor()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.1,
                        scale_pos_weight=neg_pos_ratio,
                        eval_metric="aucpr",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    """Train all models, evaluate on the held-out test set, and persist artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Keep failure-type labels aligned with the test split for the
    # per-failure-type breakdown below.
    failure_type_test = df.loc[X_test.index, "Failure Type"]

    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    models = build_models(neg_pos_ratio)

    all_metrics = {}
    curve_results = {}
    failure_breakdowns = {}

    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = compute_metrics(y_test.values, y_pred, y_proba)
        all_metrics[name] = metrics
        curve_results[name] = {"y_true": y_test.values, "y_proba": y_proba}

        plot_confusion_matrix(y_test.values, y_pred, name)

        breakdown = evaluate_by_failure_type(y_pred, failure_type_test, y_test)
        failure_breakdowns[name] = breakdown.to_dict(orient="index")

        slug = name.lower().replace(" ", "_").replace("+", "plus")
        joblib.dump(pipeline, MODELS_DIR / f"{slug}.joblib")

        print(f"  precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
              f"f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f} "
              f"pr_auc={metrics['pr_auc']:.3f}")

    metrics_df = pd.DataFrame(all_metrics).T
    plot_pr_curve(curve_results)
    plot_roc_curve(curve_results)
    plot_metric_comparison(metrics_df)

    best_model_name = metrics_df["pr_auc"].idxmax()
    best_slug = best_model_name.lower().replace(" ", "_").replace("+", "plus")
    joblib.dump(models[best_model_name], MODELS_DIR / "best_model.joblib")

    summary = {
        "metrics": all_metrics,
        "failure_type_breakdown": failure_breakdowns,
        "best_model": best_model_name,
        "best_model_selection_criterion": "highest PR-AUC on held-out test set",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_positive_rate": float(y_test.mean()),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest model by PR-AUC: {best_model_name}")
    print(f"Saved models and metrics.json to {MODELS_DIR}")
    print(f"Saved comparison figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
