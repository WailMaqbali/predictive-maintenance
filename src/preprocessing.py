"""Preprocessing pipeline and train/test splitting."""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE, TARGET_COL, TEST_SIZE, TYPE_COL

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Power [W]",
    "Temp difference [K]",
]
CATEGORICAL_FEATURES = [TYPE_COL]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessing transformer.

    Scales numeric sensor features with a ``StandardScaler`` (needed for the
    logistic regression baseline; harmless for tree-based models) and
    one-hot encodes the categorical product quality ``Type`` (L/M/H).

    Returns:
        An unfitted ``ColumnTransformer``.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> list:
    """Return the output feature names of a fitted preprocessor.

    Args:
        preprocessor: A ``ColumnTransformer`` that has already been fit.

    Returns:
        List of output column names in the order produced by ``transform``.
    """
    return list(preprocessor.get_feature_names_out())


def split_data(
    df: pd.DataFrame,
    feature_columns: list = FEATURE_COLUMNS,
    target_column: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Stratification on the target is important given the ~3.4% positive
    (failure) rate: a plain random split risks producing a test fold with
    very few failure examples and an unstable evaluation.

    Args:
        df: Processed dataframe (see ``src.data.load_processed_data``).
        feature_columns: Columns to use as model input.
        target_column: Name of the binary target column.
        test_size: Fraction of rows held out for testing.
        random_state: Seed for reproducibility.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    X = df[feature_columns]
    y = df[target_column]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
