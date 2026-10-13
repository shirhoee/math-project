import os
from fastapi.testclient import TestClient

os.environ["FAST_TEST"] = "1"
os.environ["CELERY_ALWAYS_EAGER"] = "1"

from src.api.server import app

client = TestClient(app)


def test_samples_endpoint():
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        sample = data[0]
        assert "id" in sample
        assert "filename" in sample
        assert "url" in sample

        # Test sample file fetching
        file_resp = client.get(sample["url"])
        assert file_resp.status_code == 200
        assert "image" in file_resp.headers.get("content-type", "")


def test_static_frontend_serving():
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "")
    assert "<!doctype html>" in response.text.lower()
