import os
from streamlit.testing.v1 import AppTest

def test_app_loads():
    os.environ["FAST_TEST"] = "1"
    at = AppTest.from_file("../app.py", default_timeout=180)
    at.run()
    
    # Check no exceptions
    assert not at.exception
    
    # Check title
    assert at.title[0].value == "Interactive 3D View Synthesis"
    
    # Check tabs exist by looking at markdown or checking tabs
    # AppTest tab API: at.tabs
    tabs = at.tabs
    assert len(tabs) >= 4, "Expected at least 4 tabs"
    
    # We can check the labels if supported, or just that they exist.
    expanders = at.expander
    assert len(expanders) >= 1
    assert "Settings" in expanders[0].label or "Diagnostics" in expanders[0].label
    
    # Check that atlas is built (diagnostics text appears)
    assert any("Atlas Rendering" in string for sublist in [[w.value for w in at.markdown] + [w.value for w in at.text]] for string in sublist)
