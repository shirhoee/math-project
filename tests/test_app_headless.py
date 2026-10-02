from streamlit.testing.v1 import AppTest
import pytest

def test_cache_telemetry():
    at = AppTest.from_file("app.py", default_timeout=30)
    # Give it the sample image as UploadedFile
    with open("tests/fixtures/sample.jpg", "rb") as f:
        at.session_state["uploaded_file"] = f.read()
    
    # We can't directly set uploaded_file widget easily in AppTest without a path or bytes
    # But wait, Streamlit 1.28+ supports AppTest file_uploader interaction.
    # actually, I can just mock the uploaded_file in app.py or write a simple wrapper.
    # Let's just write the script.
