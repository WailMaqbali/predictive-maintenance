"""Project-wide constants and paths.

All paths are derived relative to this file so the project can be run
from any working directory without hardcoded absolute paths.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "predictive_maintenance.csv"

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

DATASET_URL = (
    "https://raw.githubusercontent.com/michele-abruzzese/predictive_maintenance/"
    "main/predictive_maintenance.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "Target"
FAILURE_TYPE_COL = "Failure Type"
TYPE_COL = "Type"

# Raw sensor/identifier columns as they appear in the source CSV.
ID_COLS = ["UDI", "Product ID"]
RAW_FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Failure Type values that are not real failure modes / are data-quality edge
# cases (rows where Target and Failure Type disagree). Handled explicitly in
# src/data.py rather than silently dropped.
NO_FAILURE_LABEL = "No Failure"
AMBIGUOUS_FAILURE_LABEL = "Random Failures"
