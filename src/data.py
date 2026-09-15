"""Data loading and feature engineering for the predictive maintenance dataset."""

from pathlib import Path

import pandas as pd

from src.config import FAILURE_TYPE_COL, NO_FAILURE_LABEL, RAW_DATA_PATH, TARGET_COL


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw AI4I 2020 predictive maintenance CSV.

    Args:
        path: Path to the CSV file. Defaults to ``data/predictive_maintenance.csv``.

    Returns:
        The raw dataframe, unmodified.

    Raises:
        FileNotFoundError: If the CSV has not been downloaded yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it first, e.g.:\n"
            f"  curl -o {path} "
            f"https://raw.githubusercontent.com/michele-abruzzese/predictive_maintenance/"
            f"main/predictive_maintenance.csv"
        )
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically-meaningful engineered features.

    Adds:
        - ``Power [W]``: mechanical power, torque [Nm] * rotational speed
          [rpm] converted to rad/s (P = tau * omega). This is the standard
          physical relationship for a rotating shaft and is a stronger
          failure signal than either raw variable alone (the AI4I dataset's
          Power Failure mode is explicitly defined by power falling outside
          a normal operating band).
        - ``Temp difference [K]``: process temperature minus air temperature,
          relevant to the Heat Dissipation Failure mode, which the dataset
          documentation defines via too-small a temperature differential.

    Args:
        df: Dataframe containing the raw sensor columns.

    Returns:
        A copy of ``df`` with the engineered columns appended.
    """
    df = df.copy()
    omega_rad_per_s = df["Rotational speed [rpm]"] * (2 * 3.141592653589793 / 60)
    df["Power [W]"] = df["Torque [Nm]"] * omega_rad_per_s
    df["Temp difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    return df


def flag_label_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean column flagging known Target/Failure Type disagreements.

    The published AI4I 2020 dataset contains two known inconsistencies
    between the binary ``Target`` and the multi-class ``Failure Type``
    columns (documented in BUILD_LOG.md):

    - 18 rows labelled ``Failure Type == "Random Failures"`` have
      ``Target == 0`` (i.e. not counted as a machine failure).
    - 9 rows have ``Target == 1`` but ``Failure Type == "No Failure"``.

    These rows are kept (removing them would discard real data and the
    dataset's own documentation acknowledges them as injected noise), but
    are flagged so downstream analysis can inspect or exclude them
    explicitly.

    Args:
        df: Dataframe containing ``Target`` and ``Failure Type`` columns.

    Returns:
        A copy of ``df`` with an added boolean ``Label inconsistent`` column.
    """
    df = df.copy()
    random_failure_mismatch = (df[TARGET_COL] == 0) & (
        df[FAILURE_TYPE_COL] != NO_FAILURE_LABEL
    )
    unlabelled_failure_mismatch = (df[TARGET_COL] == 1) & (
        df[FAILURE_TYPE_COL] == NO_FAILURE_LABEL
    )
    df["Label inconsistent"] = random_failure_mismatch | unlabelled_failure_mismatch
    return df


def load_processed_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw data and apply feature engineering and label-consistency flags.

    Args:
        path: Path to the raw CSV file.

    Returns:
        A dataframe ready for train/test splitting.
    """
    df = load_raw_data(path)
    df = engineer_features(df)
    df = flag_label_inconsistencies(df)
    return df
