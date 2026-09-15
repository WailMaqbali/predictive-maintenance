"""FastAPI service exposing the trained failure-risk classifier.

Run locally:

    uvicorn src.serve:app --reload

Then POST sensor readings to /predict, e.g.:

    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
      "type": "M",
      "air_temperature_k": 298.1,
      "process_temperature_k": 308.6,
      "rotational_speed_rpm": 1551,
      "torque_nm": 42.8,
      "tool_wear_min": 0
    }'
"""

import math
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "best_model.joblib"

_model = None


def load_model():
    """Load the best trained pipeline from disk, raising a clear error if missing."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    global _model
    _model = load_model()
    yield


app = FastAPI(
    title="Predictive Maintenance Failure-Risk Classifier",
    description=(
        "Portfolio project: predicts probability of machine failure from "
        "milling machine sensor readings (AI4I 2020 dataset). Not validated "
        "for safety-critical deployment -- see MODEL_CARD.md."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class SensorReading(BaseModel):
    """A single snapshot of milling machine sensor readings."""

    type: Literal["L", "M", "H"] = Field(
        ..., description="Product quality variant: Low, Medium, or High."
    )
    air_temperature_k: float = Field(..., gt=0, description="Air temperature in Kelvin.")
    process_temperature_k: float = Field(
        ..., gt=0, description="Process temperature in Kelvin."
    )
    rotational_speed_rpm: float = Field(..., gt=0, description="Rotational speed in rpm.")
    torque_nm: float = Field(..., ge=0, description="Torque in Newton-metres.")
    tool_wear_min: float = Field(..., ge=0, description="Tool wear in minutes.")


class PredictionResponse(BaseModel):
    """Model prediction for a single sensor reading."""

    failure_probability: float
    predicted_label: Literal["Failure", "No Failure"]
    risk_tier: Literal["low", "medium", "high"]


def _sensor_reading_to_frame(reading: SensorReading) -> pd.DataFrame:
    """Convert an API request into the engineered feature frame the model expects."""
    omega_rad_per_s = reading.rotational_speed_rpm * (2 * math.pi / 60)
    power_w = reading.torque_nm * omega_rad_per_s
    temp_diff_k = reading.process_temperature_k - reading.air_temperature_k

    return pd.DataFrame(
        [
            {
                "Type": reading.type,
                "Air temperature [K]": reading.air_temperature_k,
                "Process temperature [K]": reading.process_temperature_k,
                "Rotational speed [rpm]": reading.rotational_speed_rpm,
                "Torque [Nm]": reading.torque_nm,
                "Tool wear [min]": reading.tool_wear_min,
                "Power [W]": power_w,
                "Temp difference [K]": temp_diff_k,
            }
        ]
    )


def _risk_tier(probability: float) -> str:
    """Bucket a raw probability into a coarse risk tier for easier triage."""
    if probability < 0.2:
        return "low"
    if probability < 0.6:
        return "medium"
    return "high"


@app.get("/health")
def health() -> dict:
    """Liveness check that also confirms the model is loaded."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading) -> PredictionResponse:
    """Predict failure probability for a single sensor reading.

    Args:
        reading: Sensor snapshot from the request body.

    Returns:
        Failure probability, thresholded label, and a coarse risk tier.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = _sensor_reading_to_frame(reading)
    probability = float(_model.predict_proba(X)[0, 1])
    label = "Failure" if probability >= 0.5 else "No Failure"

    return PredictionResponse(
        failure_probability=probability,
        predicted_label=label,
        risk_tier=_risk_tier(probability),
    )
