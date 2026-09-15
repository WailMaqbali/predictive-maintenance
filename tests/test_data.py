"""Tests for src/data.py."""

import math

import pandas as pd
import pytest

from src.data import engineer_features, flag_label_inconsistencies, load_raw_data


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Type": ["M", "L", "H"],
            "Air temperature [K]": [298.1, 298.2, 300.0],
            "Process temperature [K]": [308.6, 308.7, 310.0],
            "Rotational speed [rpm]": [1551, 1408, 1400],
            "Torque [Nm]": [42.8, 46.3, 50.0],
            "Tool wear [min]": [0, 3, 10],
            "Target": [0, 0, 1],
            "Failure Type": ["No Failure", "No Failure", "Power Failure"],
        }
    )


def test_load_raw_data_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Download it first"):
        load_raw_data(missing_path)


def test_engineer_features_adds_expected_columns(sample_df):
    result = engineer_features(sample_df)
    assert "Power [W]" in result.columns
    assert "Temp difference [K]" in result.columns


def test_engineer_features_power_matches_physics(sample_df):
    result = engineer_features(sample_df)
    row = result.iloc[0]
    expected_power = 42.8 * (1551 * 2 * math.pi / 60)
    assert row["Power [W]"] == pytest.approx(expected_power, rel=1e-9)


def test_engineer_features_temp_difference(sample_df):
    result = engineer_features(sample_df)
    assert result["Temp difference [K]"].tolist() == pytest.approx([10.5, 10.5, 10.0])


def test_engineer_features_does_not_mutate_input(sample_df):
    original_cols = list(sample_df.columns)
    engineer_features(sample_df)
    assert list(sample_df.columns) == original_cols


def test_flag_label_inconsistencies_detects_target_zero_mismatch():
    df = pd.DataFrame(
        {"Target": [0, 1], "Failure Type": ["Random Failures", "Power Failure"]}
    )
    result = flag_label_inconsistencies(df)
    assert result["Label inconsistent"].tolist() == [True, False]


def test_flag_label_inconsistencies_detects_target_one_mismatch():
    df = pd.DataFrame({"Target": [1, 0], "Failure Type": ["No Failure", "No Failure"]})
    result = flag_label_inconsistencies(df)
    assert result["Label inconsistent"].tolist() == [True, False]


def test_flag_label_inconsistencies_consistent_rows_not_flagged(sample_df):
    result = flag_label_inconsistencies(sample_df)
    assert not result["Label inconsistent"].any()
