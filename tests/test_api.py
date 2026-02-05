from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200, "Expected status code 200"

    data = response.json()
    assert data["status"] == "ok", "Expected status=ok"
    assert "model_loaded" in data, "Expected model_loaded field"
    assert isinstance(data["model_loaded"], bool), "Expected model_loaded to be a bool"
