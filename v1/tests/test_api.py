import os
import io
import time

os.environ["FAST_TEST"] = "1"
os.environ["CELERY_ALWAYS_EAGER"] = "1"

from fastapi.testclient import TestClient
from src.api.server import app
from PIL import Image
import numpy as np

client = TestClient(app)

def test_render_workflow():
    os.environ["FAST_TEST"] = "1"
    os.environ["CELERY_ALWAYS_EAGER"] = "1"
    
    # Create dummy image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG")
    buf.seek(0)
    
    # 1. Submit Job
    response = client.post(
        "/api/v1/render",
        files={"file": ("dummy.jpg", buf, "image/jpeg")},
        data={"renderer": "Point Splatting", "motion_preset_str": "Normal (0.07)"}
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert job_id is not None
    
    # 2. Poll Status
    start = time.time()
    result = None
    while time.time() - start < 15:
        res = client.get(f"/api/v1/jobs/{job_id}/status")
        assert res.status_code == 200
        data = res.json()
        if data.get("result"):
            result = data["result"]
            break
        time.sleep(0.5)
        
    assert result is not None, "Job did not complete in time"
    
    # 3. Get Result
    res = client.get(f"/api/v1/jobs/{job_id}/result")
    assert res.status_code == 200
    payload = res.json()
    assert payload["renderer"] == "Point Splatting"
    assert "frames_b64" in payload
    assert len(payload["frames_b64"]) > 0
