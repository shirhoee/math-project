import os
import time
from streamlit.testing.v1 import AppTest
import numpy as np
from PIL import Image

def test_app_loads():
    os.environ["FAST_TEST"] = "1"
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.run()
    
    assert len(at.exception) == 0
    assert at.title[0].value == "Interactive 3D View Synthesis"
    
    expanders = at.expander
    assert len(expanders) >= 1
    assert "Settings" in expanders[0].label
    
    start = time.time()
    while time.time() - start < 10:
        at.run()
        if len(at.exception) == 0 and any("Live View Matrices" in string for sublist in [[w.value for w in at.markdown]] for string in sublist):
            break
        time.sleep(0.5)
        
    assert len(at.exception) == 0
    assert any("Live View Matrices" in string for sublist in [[w.value for w in at.markdown]] for string in sublist)

def test_app_upload_progress():
    os.environ["FAST_TEST"] = "1"
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.run()
    
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    Image.fromarray(img).save("dummy.jpg")
    
    at.session_state.selected_image_path = "dummy.jpg"
    at.run()
    assert len(at.exception) == 0
