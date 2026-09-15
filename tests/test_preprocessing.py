"""Tests for src/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import FEATURE_COLUMNS, build_preprocessor, split_data


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "Type": rng.choice(["L", "M", "H"], size=n),
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 2, n),
            "Rotational speed [rpm]": rng.normal(1500, 100, n),
            "Torque [Nm]": rng.normal(40, 10, n),
            "Tool wear [min]": rng.uniform(0, 250, n),
            "Power [W]": rng.normal(6000, 1000, n),
            "Temp difference [K]": rng.normal(10, 1, n),
        }
    )
    # Rare positive class, mirroring the real ~3.4% failure rate.
    target = np.zeros(n, dtype=int)
    target[: max(1, int(0.034 * n))] = 1
    df["Target"] = rng.permutation(target)
    return df


def test_split_data_is_stratified(synthetic_df):
    X_train, X_test, y_train, y_test = split_data(synthetic_df, test_size=0.2)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert train_rate == pytest.approx(test_rate, abs=0.02)


def test_split_data_sizes(synthetic_df):
    X_train, X_test, y_train, y_test = split_data(synthetic_df, test_size=0.2)
    assert len(X_train) + len(X_test) == len(synthetic_df)
    assert len(X_test) == pytest.approx(len(synthetic_df) * 0.2, abs=1)


def test_split_data_uses_only_expected_feature_columns(synthetic_df):
    X_train, X_test, y_train, y_test = split_data(synthetic_df)
    assert list(X_train.columns) == FEATURE_COLUMNS


def test_split_data_is_reproducible(synthetic_df):
    a = split_data(synthetic_df, random_state=1)
    b = split_data(synthetic_df, random_state=1)
    pd.testing.assert_frame_equal(a[0], b[0])


def test_build_preprocessor_output_shape(synthetic_df):
    X = synthetic_df[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    # 7 numeric features + one-hot(Type, drop first) with 3 categories -> 2 dummy cols.
    assert transformed.shape == (len(X), 9)


def test_build_preprocessor_scales_numeric_features(synthetic_df):
    X = synthetic_df[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)
    numeric_block = np.asarray(transformed)[:, :7]
    assert numeric_block.mean(axis=0) == pytest.approx(np.zeros(7), abs=1e-6)
    assert numeric_block.std(axis=0) == pytest.approx(np.ones(7), abs=1e-6)
