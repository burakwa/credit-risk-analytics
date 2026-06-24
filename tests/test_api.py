from fastapi.testclient import TestClient
from app.main import app
from ml.inference import predictor
import pytest
from unittest.mock import patch


@pytest.fixture
def client():
    # Use TestClient with lifespan support to ensure the model is loaded on startup
    with TestClient(app) as c:
        yield c


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Credit Risk Analytics API is running" in data["message"]
    assert "version" in data


def test_predict_success(client):
    payload = {
        "features": [
            {
                "age": 30,
                "sex": "male",
                "job": 2,
                "housing": "own",
                "saving_accounts": "little",
                "checking_account": "moderate",
                "credit_amount": 5000,
                "duration": 24,
                "purpose": "car"
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 1
    assert data["predictions"][0] in [0, 1]


def test_predict_proba_success(client):
    payload = {
        "features": [
            {
                "age": 30,
                "sex": "male",
                "job": 2,
                "housing": "own",
                "saving_accounts": "little",
                "checking_account": "moderate",
                "credit_amount": 5000,
                "duration": 24,
                "purpose": "car"
            }
        ]
    }
    response = client.post("/api/v1/predict-proba", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
    assert isinstance(data["probabilities"], list)
    assert len(data["probabilities"]) == 1
    assert len(data["probabilities"][0]) == 2
    assert abs(sum(data["probabilities"][0]) - 1.0) < 1e-5


def test_predict_validation_error(client):
    # Invalid age (under 18)
    payload = {
        "features": [
            {
                "age": 17,
                "sex": "male",
                "job": 2,
                "housing": "own",
                "saving_accounts": "little",
                "checking_account": "moderate",
                "credit_amount": 5000,
                "duration": 24,
                "purpose": "car"
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)


def test_predict_empty_features(client):
    payload = {
        "features": []
    }
    response = client.post("/api/v1/predict", json=payload)
    # The API schema expects a List[CreditRiskFeatures], an empty list passes schema validation
    # but the ML model prepare_data raises ValueError, which leads to 500
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data


@patch("ml.inference.predictor.predict")
def test_predict_model_not_found_error(mock_predict, client):
    mock_predict.side_effect = FileNotFoundError("Model pickle file missing")
    payload = {
        "features": [
            {
                "age": 30,
                "sex": "male",
                "job": 2,
                "housing": "own",
                "saving_accounts": "little",
                "checking_account": "moderate",
                "credit_amount": 5000,
                "duration": 24,
                "purpose": "car"
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 503
    assert "Model yüklenemedi" in response.json()["detail"]
