import sys
import os
sys.path.insert(0, os.getcwd())
from streamlit.testing.v1 import AppTest

def test_cache():
    app_path = os.path.join(os.getcwd(), "app.py")
    at = AppTest.from_file(app_path, default_timeout=120)
    print("Initial run...")
    at.run()
    if at.exception:
        print("Exception:", at.exception)
        
    print("Moving slider 10 times...")
    for i in range(1, 11):
        for slider in at.slider:
            if slider.label == "Yaw (deg)":
                slider.set_value(float(i * 2)).run()
                break
                
    texts = [t.value for t in at.text if "Runs:" in t.value]
    print(texts)

if __name__ == "__main__":
    test_cache()
