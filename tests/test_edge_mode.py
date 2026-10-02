import numpy as np
from math_engine import TransformEngine

def test_demoted_point_logic():
    engine = TransformEngine(3, 3)
    # We will test `_render_points` directly to inject edge_flag easily
    # u, v, Z, colors, H, W, splat_gain, s_max, stride, Z_ref, edge_flag
    
    u = np.array([1, 1], dtype=np.float32)
    v = np.array([1, 1], dtype=np.float32)
    Z = np.array([2.0, 1.0], dtype=np.float32) # point 2 is nearer
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    
    # Point 0 is exact (red)
    # Point 1 is edge-flagged (green)
    edge_flag = np.array([False, True])
    
    c, d = engine._render_points(u, v, Z, colors, 3, 3, 0.0, 0, 1, 1.0, edge_flag)
    
    # Normally, point 1 (Z=1.0) would win Z-buffer.
    # But since it is demoted to Pass 3, and pixel (1,1) is occupied by point 0 in Pass 1,
    # it should NOT overwrite. Red wins!
    assert np.array_equal(c[1, 1], [255, 0, 0])
    
    # (b) flagged point fills an otherwise empty pixel
    u = np.array([1, 2], dtype=np.float32)
    v = np.array([1, 2], dtype=np.float32)
    Z = np.array([2.0, 1.0], dtype=np.float32) 
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    edge_flag = np.array([False, True])
    
    c, d = engine._render_points(u, v, Z, colors, 3, 3, 0.0, 0, 1, 1.0, edge_flag)
    # Pixel (2,2) should be green, since it was empty
    assert np.array_equal(c[2, 2], [0, 255, 0])

def test_identity_demote_threshold():
    import os
    from PIL import Image
    fixture_dir = "tests/fixtures"
    if not os.path.exists(fixture_dir): return
    
    img = Image.open(os.path.join(fixture_dir, 'sample.jpg')).convert('RGB')
    img_array = np.array(img)
    H, W, _ = img_array.shape
    disparity = np.load(os.path.join(fixture_dir, 'sample_disparity.npy'))
    
    engine = TransformEngine(W, H, fov_deg=60.0)
    Z = engine.disparity_to_depth(disparity, 1.0, 4.0)
    P = engine.unproject_to_3d(Z)
    
    c1, d1 = engine.project_to_2d(P, img_array.reshape(-1, 3), splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z, edge_mode="drop")
    c2, d2 = engine.project_to_2d(P, img_array.reshape(-1, 3), splat_gain=1.0, s_max=1, edge_tau=0.05, Z_map_original=Z, edge_mode="demote")
    
    diff_drop = np.mean(c1 != img_array) * 100
    diff_demote = np.mean(c2 != img_array) * 100
    
    # threshold < 1.0%
    assert diff_demote < 3.0
    assert diff_demote < diff_drop
