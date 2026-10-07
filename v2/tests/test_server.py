from fastapi.testclient import TestClient
from src.server.main import app
import os

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

def test_samples_list():
    response = client.get("/samples_list")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_process_sample():
    response = client.post("/process", data={"sample_name": "indoor.jpg"})
    assert response.status_code == 200
    data = response.json()
    assert "vertices" in data
    assert "indices" in data
    assert "uvs" in data
    assert "texture" in data
    assert "K" in data
