import os
import time
from streamlit.testing.v1 import AppTest
import numpy as np
from PIL import Image

def test_app_loads():
    os.environ["FAST_TEST"] = "1"
    os.environ["CACHE_DIR"] = "temp_cache"
    if os.path.exists("temp_cache"):
        import shutil
        shutil.rmtree("temp_cache")

    at = AppTest.from_file("../app.py", default_timeout=180)
    at.run()
    
    assert len(at.exception) == 0
    assert at.title[0].value == "Interactive 3D View Synthesis"
    
    def wait_for_load(at, sample_name):
        start = time.time()
        while time.time() - start < 30:
            at.run()
            if len(at.exception) == 0 and any(f"Loaded: `{sample_name}`" in string for sublist in [[w.value for w in at.markdown]] for string in sublist):
                break
            time.sleep(1.0)
        assert len(at.exception) == 0
        assert any(f"Loaded: `{sample_name}`" in string for sublist in [[w.value for w in at.markdown]] for string in sublist)
        
        # Verify all frames have the same size
        import base64
        import io
        from PIL import Image
        frames_b64 = at.session_state.job_result["frames_b64"]
        assert len(frames_b64) > 0
        
        first_img = Image.open(io.BytesIO(base64.b64decode(frames_b64[0].split(',')[1])))
        w, h = first_img.size
        for i in range(1, len(frames_b64)):
            img = Image.open(io.BytesIO(base64.b64decode(frames_b64[i].split(',')[1])))
            assert img.size == (w, h), f"Frame {i} has size {img.size}, expected {(w, h)}"

    wait_for_load(at, "01_sample_1.jpg")
    
    # Load sample 2
    at.button(key="load_btn_1").click().run()
    wait_for_load(at, "01_sample_2.jpg")
    
    # Load sample 1 again
    at.button(key="load_btn_0").click().run()
    wait_for_load(at, "01_sample_1.jpg")

def test_app_upload_progress():
    os.environ["FAST_TEST"] = "1"
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.run()
    
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    Image.fromarray(img).save("dummy.jpg")
    
    at.session_state.selected_image_path = "dummy.jpg"
    at.run()
    assert len(at.exception) == 0

def test_cache_key():
    from app import get_hash
    # Width change
    h1 = get_hash("app.py", "CodeV2_W1024_H512_Grid41x3_Preset0.07_Z1.0-4.0_L16")
    h2 = get_hash("app.py", "CodeV2_W512_H256_Grid41x3_Preset0.07_Z1.0-4.0_L16")
    assert h1 != h2
    # Preset change
    h3 = get_hash("app.py", "CodeV2_W1024_H512_Grid41x3_Preset0.04_Z1.0-4.0_L16")
    assert h1 != h3
