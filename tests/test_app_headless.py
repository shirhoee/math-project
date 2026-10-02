import sys
import os
sys.path.insert(0, os.getcwd())
from streamlit.testing.v1 import AppTest


def test_cache():
    app_path = os.path.join(os.getcwd(), "app.py")
    at = AppTest.from_file(app_path, default_timeout=120)
    at.run()
    assert not at.exception
    
    # Move slider 9 times (so total renders = 10)
    for i in range(1, 10):
        for slider in at.slider:
            if slider.label == "Yaw (deg)":
                slider.set_value(float(i * 2)).run()
                break
                
    texts = [t.value for t in at.text]
    print("TEXTS:", texts)
    # Check counters
    # In app.py: st.text(f"Depth Runs: {depth_runs}")
    assert any("Depth Runs: 1" in t for t in texts)
    assert any("Unprojection Runs: 1" in t for t in texts)
    # The render function is not cached, so it runs every time (10 times)
    # But wait, app.py prints them? No, let's just make sure depth and unprojection are 1.
    pass
