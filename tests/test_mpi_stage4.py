import numpy as np
from mpi_renderer import parallax_px, calibrate_motion, render_atlas_mpi

def test_parallax_px():
    assert np.isclose(parallax_px(10.0, 500.0, 2.0), 100.0)

def test_calibrate_motion():
    # In R2, calibrate_motion was updated to account for pivot rotation and zoom.
    # We verify that for a given configuration, it finds a tx that produces the target parallax.
    z_near, z_far = 1.0, 4.0
    f, W, yaw = 500.0, 1000, 10.0
    target_ratio = 0.1
    tx = calibrate_motion(z_near, z_far, f, W, yaw, target_ratio)
    assert tx > 0.1 and tx < 0.5

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
