import numpy as np
import pytest
from math_engine import TransformEngine
import os
import sys

def test_disparity_direction():
    # Load the real disparity map from Depth Anything V2
    import os
    if not os.path.exists('tests/fixtures/sample_disparity.npy'):
        pytest.skip("Fixture sample_disparity.npy not found")
        
    disparity = np.load('tests/fixtures/sample_disparity.npy')
    
    person_box = disparity[300:500, 300:500]
    sky_box = disparity[0:100, 0:200]
    
    mean_near = np.mean(person_box)
    mean_far = np.mean(sky_box)
    
    assert mean_near > mean_far
    
    # Engine validation: near object must have a smaller physical Z depth
    engine = TransformEngine(disparity.shape[1], disparity.shape[0])
    z = engine.disparity_to_depth(disparity, z_near=1.0, z_far=4.0)
    
    z_person = z[300:500, 300:500]
    z_sky = z[0:100, 0:200]
    
    assert np.mean(z_person) < np.mean(z_sky)

def test_edge_masking_slanted_plane():
    engine = TransformEngine(10, 10)
    
    # Slanted plane: Z increases linearly very slowly to not trigger edge_tau
    z_slanted = np.linspace(1, 1.1, 100).reshape(10, 10).astype(np.float32)
    P_slanted = engine.unproject_to_3d(z_slanted)
    C_slanted = np.zeros((100, 3), dtype=np.uint8)
    
    # Should drop 0 points since dZ/Z is very small
    _, depth_slanted = engine.project_to_2d(P_slanted, C_slanted, splat_gain=0, s_max=0, edge_tau=0.05, Z_map_original=z_slanted, edge_mode="drop")
    holes_slanted = np.sum(np.isinf(depth_slanted))
    assert holes_slanted == 0

    # Step edge
    z_step = np.ones((10, 10), dtype=np.float32)
    z_step[:, 5:] = 4.0
    P_step = engine.unproject_to_3d(z_step)
    
    _, depth_step = engine.project_to_2d(P_step, C_slanted, splat_gain=0, s_max=0, edge_tau=0.05, Z_map_original=z_step, edge_mode="drop")
    holes_step = np.sum(np.isinf(depth_step))
    assert holes_step > 0 # Points at the discontinuity are dropped

def test_splatting():
    engine = TransformEngine(10, 10)
    
    # Sparse points: just one pixel at center
    z = np.full((10, 10), np.inf, dtype=np.float32)
    z[5, 5] = 1.0
    
    P = engine.unproject_to_3d(z)
    valid = ~np.isinf(z.flatten())
    P = P[valid]
    C = np.array([[255, 255, 255]], dtype=np.uint8)
    
    # Without splatting
    c1, d1 = engine.project_to_2d(P, C, splat_gain=0, s_max=0, edge_tau=None, Z_map_original=None)
    assert np.sum(d1 < np.inf) == 1
    
    # With splatting
    c2, d2 = engine.project_to_2d(P, C, splat_gain=1.0, s_max=1, Z_ref=1.1, edge_tau=None, Z_map_original=None)
    assert np.sum(d2 < np.inf) > 1 # Footprint is 3x3 so should be up to 9
    
    # Fill-only splat test: isolated foreground pixel doesn't dilate onto occupied background
    # Background at Z=4
    z_bg = np.full((10, 10), 4.0, dtype=np.float32)
    z_bg[5, 5] = 1.0 # Foreground at Z=1
    
    P_both = engine.unproject_to_3d(z_bg)
    C_both = np.zeros((100, 3), dtype=np.uint8)
    C_both[55] = [255, 0, 0] # Foreground is red
    
    c3, d3 = engine.project_to_2d(P_both, C_both, splat_gain=1.0, s_max=1, Z_ref=1.1, edge_tau=None, Z_map_original=None)
    
    # The pixels adjacent to (5,5) should be background (Z=4) because pass 1 filled them
    assert d3[5, 4] == 4.0
    assert np.array_equal(c3[5, 4], [0, 0, 0])

def test_hole_fill_background_bias():
    engine = TransformEngine(10, 10)
    c = np.zeros((10, 10, 3), dtype=np.uint8)
    d = np.full((10, 10), np.inf, dtype=np.float32)
    
    # Foreground at left (Z=1, Red)
    d[:, 0:4] = 1.0
    c[:, 0:4] = [255, 0, 0]
    
    # Background at right (Z=4, Blue)
    d[:, 5:10] = 4.0
    c[:, 5:10] = [0, 0, 255]
    
    # Hole at column 4
    
    c_filled, d_filled = engine.fill_holes_iterative(c, d)
    
    # Hole should be filled with background (Z=4, Blue)
    assert d_filled[5, 4] == 4.0
    assert np.array_equal(c_filled[5, 4], [0, 0, 255])
    
def test_identity_defaults():
    engine = TransformEngine(20, 20)
    Z = np.full((20, 20), 2.0, dtype=np.float32)
    C = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    
    P = engine.unproject_to_3d(Z)
    
    # Defaults: splat on, edge mask on
    c1, d1 = engine.project_to_2d(P, C.reshape(-1, 3), splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z)
    
    diff_frac = np.mean(c1 != C)
    assert diff_frac < 0.05 # Under 5% difference

@pytest.mark.skipif(not os.path.exists('sample.jpg'), reason="sample.jpg not found")
def test_e2e_regression():
    from depth_estimator import DepthEstimator
    from PIL import Image
    
    img = Image.open('sample.jpg').convert('RGB').resize((100, 100))
    C = np.array(img)
    H, W = 100, 100
    
    estimator = DepthEstimator()
    img.save('temp_test.jpg')
    disparity = estimator.estimate_depth('temp_test.jpg')
    
    engine = TransformEngine(W, H)
    Z = engine.disparity_to_depth(disparity)
    P = engine.unproject_to_3d(Z)
    
    R = engine.get_rotation_matrix(0, np.radians(15), 0)
    P_new = engine.apply_transform(P, R)
    
    c1, d1 = engine.project_to_2d(P_new, C.reshape(-1, 3), Z_map_original=Z)
    
    # Since this is a regression test without a strict golden image provided,
    # we just assert pipeline runs and outputs correct shape
    assert c1.shape == (H, W, 3)
    assert d1.shape == (H, W)
