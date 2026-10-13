import os
import pytest
from PIL import Image
import numpy as np

os.environ["CELERY_ALWAYS_EAGER"] = "1"
os.environ["FAST_TEST"] = "1"

from src.pipeline import submit_render_job, poll_job_status, get_job_result

@pytest.fixture
def dummy_image(tmp_path):
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    path = tmp_path / "dummy.jpg"
    img.save(path)
    return str(path)

def test_celery_pipeline_eager(dummy_image):
    # submit job, it will execute synchronously due to ALWAYS_EAGER
    cached = submit_render_job(
        job_id=999,
        img_path=dummy_image,
        renderer="Layers (MPI)",
        motion_preset_str="Normal (0.07)",
        z_near=1.0,
        z_far=4.0,
        n_layers=8,
        sharpen_amount=0.0
    )
    assert cached is None
    
    # check status
    status = poll_job_status(999)
    assert status["status"] is None  # Success clears status text
    assert status["progress"] == 100
    assert status["result"] is not None
    assert "frames_b64" in status["result"]
    
    # get result directly
    res = get_job_result(999)
    assert res is not None
    assert res["W"] == 100 or res["W"] <= 256  # fast test resize
