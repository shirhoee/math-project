import numpy as np
from mpi_renderer import parallax_px, calibrate_motion, render_atlas_mpi

def test_parallax_px():
    assert np.isclose(parallax_px(10.0, 500.0, 2.0), 100.0)

def test_calibrate_motion():
    # z_near=1.0, f=500.0, W=1000, ratio=0.1 => shift_px = 100
    # baseline = 100 * 1.0 / 500.0 = 0.2
    assert np.isclose(calibrate_motion(1.0, 500.0, 1000, 0.1), 0.2)

def test_render_atlas_mpi():
    # Smoke test
    H, W = 10, 10
    layers = np.zeros((2, H, W, 4), dtype=np.float32)
    z_k = np.array([2.0, 1.0], dtype=np.float32)
    K = np.eye(3, dtype=np.float32)
    
    angles = np.zeros((2, 2, 2), dtype=np.float32)
    frames = render_atlas_mpi(layers, z_k, K, angles, 0.1, 0.1)
    
    assert len(frames) == 4
    assert frames[0].startswith("data:image/jpeg;base64,")
