from fastapi.testclient import TestClient

from src.app.main import app

# Client di test di FastAPI, simula richieste senza avviare realmente il server.
client = TestClient(app)


# Controlla che l'endpoint /health sia raggiungibile e 
# restituisca le informazioni sullo stato del servizio. 
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_source" in data
    assert data["status"] == "ok"