# tests/test_api.py

import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict():
    payload = {
        "brand": "Apple",
        "future_years": [2025, 2026, 2027]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "brand" in data
    assert data["brand"] == "Apple"
    assert "predictions" in data
    assert len(data["predictions"]) == 3
