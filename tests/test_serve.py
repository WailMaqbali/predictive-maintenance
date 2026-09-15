"""Tests for src/serve.py.

Skipped automatically if no trained model is present (models/ is gitignored;
run `python -m src.train` to generate one before running these).
"""

import pytest
from fastapi.testclient import TestClient

from src.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(),
    reason="No trained model found -- run `python -m src.train` first.",
)


@pytest.fixture
def client():
    from src.serve import app

    with TestClient(app) as c:
        yield c


VALID_READING = {
    "type": "M",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_response_shape(client):
    response = client.post("/predict", json=VALID_READING)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["predicted_label"] in {"Failure", "No Failure"}
    assert body["risk_tier"] in {"low", "medium", "high"}


def test_predict_label_consistent_with_probability(client):
    response = client.post("/predict", json=VALID_READING)
    body = response.json()
    if body["failure_probability"] >= 0.5:
        assert body["predicted_label"] == "Failure"
    else:
        assert body["predicted_label"] == "No Failure"


def test_predict_high_stress_reading_flagged_higher_risk_than_nominal(client):
    nominal = client.post("/predict", json=VALID_READING).json()
    high_stress = client.post(
        "/predict",
        json={
            "type": "L",
            "air_temperature_k": 302.5,
            "process_temperature_k": 312.0,
            "rotational_speed_rpm": 1350,
            "torque_nm": 65.0,
            "tool_wear_min": 220,
        },
    ).json()
    assert high_stress["failure_probability"] > nominal["failure_probability"]


def test_predict_rejects_invalid_type(client):
    bad_reading = {**VALID_READING, "type": "X"}
    response = client.post("/predict", json=bad_reading)
    assert response.status_code == 422


def test_predict_rejects_negative_torque(client):
    bad_reading = {**VALID_READING, "torque_nm": -5}
    response = client.post("/predict", json=bad_reading)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete = {k: v for k, v in VALID_READING.items() if k != "torque_nm"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
