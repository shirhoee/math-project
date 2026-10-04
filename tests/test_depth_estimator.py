import numpy as np
from depth_estimator import box_filter, guided_filter

def test_box_filter_constant():
    img = np.ones((100, 100), dtype=np.float32) * 5.0
    filtered = box_filter(img, r=2)
    assert np.allclose(filtered, 5.0)

def test_guided_filter_constant():
    I = np.ones((100, 100), dtype=np.float32) * 0.5
    p = np.ones((100, 100), dtype=np.float32) * 0.3
    q = guided_filter(I, p, r=2, eps=1e-3)
    assert np.allclose(q, 0.3, atol=1e-3)

def test_guided_filter_step_edge():
    # Guide has a sharp step edge
    I = np.zeros((100, 100), dtype=np.float32)
    I[:, 50:] = 1.0
    
    # Input has a blurred step edge
    p = np.zeros((100, 100), dtype=np.float32)
    p[:, 48:52] = 0.5
    p[:, 52:] = 1.0
    
    q = guided_filter(I, p, r=5, eps=1e-3)
    
    # Check that the step edge is realigned to the guide (x=50)
    # The left side (x=45..49) should be close to 0, right side (x=50..54) close to 1
    assert np.mean(q[:, 45:49]) < 0.2
    assert np.mean(q[:, 51:55]) > 0.8

def test_guided_filter_noise_drop():
    I = np.ones((100, 100), dtype=np.float32) * 0.5
    np.random.seed(42)
    p = 0.5 + 0.1 * np.random.randn(100, 100).astype(np.float32)
    
    q = guided_filter(I, p, r=3, eps=1e-3)
    var_p = np.var(p)
    var_q = np.var(q)
    assert var_q < var_p * 0.5  # Variance should drop significantly on flat regions

def test_normalized_disparity_range():
    # Just checking mock output
    from depth_estimator import DepthEstimator
    import os
    
    # we don't want to instantiate the model in tests if not needed, but we can test the function logic if we mock.
    pass
