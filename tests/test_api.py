from fastapi.testclient import TestClient

from src.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_source" in data
    assert data["status"] == "ok"