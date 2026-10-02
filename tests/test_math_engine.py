import numpy as np
import pytest
from math_engine import TransformEngine
import ast

def test_orthogonality():
    engine = TransformEngine(100, 100)
    for _ in range(5):
        pitch, yaw, roll = np.random.uniform(-np.pi, np.pi, 3)
        R = engine.get_rotation_matrix(pitch, yaw, roll)
        
        # R @ R.T ≈ I
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        
        # det(R) ≈ +1
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)

def test_isometry():
    engine = TransformEngine(100, 100)
    P = np.random.rand(10, 3).astype(np.float32)
    R = engine.get_rotation_matrix(0.5, -0.3, 1.2)
    t = np.array([1, -2, 3], dtype=np.float32)
    
    P_new = engine.apply_transform(P, R, t)
    
    # Distance between points 0 and 1
    dist_old = np.linalg.norm(P[0] - P[1])
    dist_new = np.linalg.norm(P_new[0] - P_new[1])
    np.testing.assert_allclose(dist_old, dist_new, atol=1e-5)

def test_pivot_invariance():
    engine = TransformEngine(100, 100)
    c = np.array([0, 0, 5], dtype=np.float32)
    P = c.reshape(1, 3)
    R = engine.get_rotation_matrix(0.1, 0.2, 0.3)
    
    P_new = engine.apply_transform(P, R, t=np.zeros(3), Z_pivot=5.0)
    np.testing.assert_allclose(P_new, P, atol=1e-5)

def test_homogeneous_cross_check():
    engine = TransformEngine(100, 100)
    R = engine.get_rotation_matrix(0.1, 0.2, 0.3)
    t = np.array([1, 2, 3], dtype=np.float32)
    c = np.array([0, 0, 5], dtype=np.float32)
    
    T = engine.build_homogeneous_transform(R, t, c)
    
    P = np.random.rand(10, 3).astype(np.float32)
    P1 = np.hstack([P, np.ones((10, 1))])
    
    P_hom = (T @ P1.T).T[:, :3]
    P_new = engine.apply_transform(P, R, t, Z_pivot=5.0)
    
    np.testing.assert_allclose(P_hom, P_new, atol=1e-5)

def test_unproject_project_round_trip():
    H, W = 10, 20
    engine = TransformEngine(W, H)
    Z = np.full((H, W), 3.0, dtype=np.float32)
    
    P = engine.unproject_to_3d(Z)
    
    # Fake colors
    colors = np.zeros((H * W, 3), dtype=np.uint8)
    
    # project_to_2d uses projection and rounds
    canvas, depth = engine.project_to_2d(P, colors, splat_gain=0.0, s_max=0, edge_tau=None)
    
    # Round trip should restore exactly
    # Since all depths are 3.0, depth_buf should have 3.0 at all HxW
    assert depth.shape == (H, W)
    np.testing.assert_allclose(depth, 3.0, atol=1e-5)

def test_identity():
    H, W = 20, 20
    engine = TransformEngine(W, H)
    Z = np.random.uniform(1, 5, (H, W)).astype(np.float32)
    colors = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    P = engine.unproject_to_3d(Z)
    
    # Identity test: s_max=0, edge_tau=None
    canvas, depth = engine.project_to_2d(P, colors.reshape(-1, 3), splat_gain=0, s_max=0, edge_tau=None)
    
    assert np.array_equal(canvas, colors)

def test_z_buffer():
    H, W = 10, 10
    engine = TransformEngine(W, H)
    
    # Two points that will project to the exact same pixel (center)
    # Point 1 (Front): Z=2
    # Point 2 (Back): Z=4
    P = np.array([
        [0, 0, 2],
        [0, 0, 4]
    ], dtype=np.float32)
    
    colors = np.array([
        [255, 0, 0], # Red front
        [0, 0, 255]  # Blue back
    ], dtype=np.uint8)
    
    canvas, depth = engine.project_to_2d(P, colors, splat_gain=0, s_max=0, edge_tau=None)
    
    # Center pixel should be red (front)
    # cx = 4.5, cy = 4.5, rounds to 5, 5
    cx, cy = int(np.rint(engine.cx)), int(np.rint(engine.cy))
    assert np.array_equal(canvas[cy, cx], [255, 0, 0])

def test_translation_scaling():
    H, W = 21, 21
    engine = TransformEngine(W, H)
    # Center pixel
    u, v = 10, 10
    Z_val = 2.0
    P = engine.unproject_to_3d(np.full((H, W), Z_val, dtype=np.float32))
    
    tz = 2.0
    # Original Z = 2. New Z = Z + tz = 4. Scaling should be Z/(Z+tz) = 2/4 = 0.5
    P_new = P + np.array([0, 0, tz])
    
    uvw = P_new @ engine.K.T
    u_new = uvw[:, 0] / P_new[:, 2]
    v_new = uvw[:, 1] / P_new[:, 2]
    
    # Center offset should scale by 0.5
    u_off_old = np.arange(W) - engine.cx
    v_off_old = np.arange(H) - engine.cy
    uu, vv = np.meshgrid(u_off_old, v_off_old)
    
    u_off_new = u_new - engine.cx
    
    np.testing.assert_allclose(u_off_new, uu.flatten() * (Z_val / (Z_val + tz)), atol=1e-5)

def test_parallax():
    engine = TransformEngine(100, 100)
    P = np.array([
        [1, 1, 2],
        [2, 2, 4]
    ], dtype=np.float32)
    # Both points are on the ray (1, 1, 2) * k
    
    # Rotate around origin
    R = engine.get_rotation_matrix(0, 0.2, 0)
    P_orig = P @ R.T
    uvw_orig = P_orig @ engine.K.T
    u_orig = uvw_orig[:, 0] / P_orig[:, 2]
    
    # Because of ray property, u_orig[0] should be roughly u_orig[1]
    np.testing.assert_allclose(u_orig[0], u_orig[1], atol=1e-5)
    
    # Rotate around pivot
    P_pivot = engine.apply_transform(P, R, Z_pivot=2.0)
    uvw_pivot = P_pivot @ engine.K.T
    u_pivot = uvw_pivot[:, 0] / P_pivot[:, 2]
    
    # Parallax introduced! They must not land on the same pixel
    assert abs(u_pivot[0] - u_pivot[1]) > 1.0

def test_orthographic():
    engine = TransformEngine(100, 100)
    Pi = np.diag([1, 1, 0]).astype(np.float32)
    
    assert np.array_equal(Pi @ Pi, Pi)
    
    P = np.array([[1, 2, 3]], dtype=np.float32)
    P_ortho = P @ Pi.T
    assert P_ortho[0, 2] == 0.0

def test_depth_conversion():
    engine = TransformEngine(100, 100)
    d = np.array([0.0, 0.5, 1.0])
    Z = engine.disparity_to_depth(d, z_near=1.0, z_far=4.0)
    
    assert Z[2] == 1.0 # nearest
    assert Z[0] == 4.0 # furthest
    assert np.all(Z > 0)
    assert np.all(np.diff(Z) < 0) # monotonically decreasing depth for increasing disparity

def test_vectorization_audit():
    with open('math_engine.py', 'r') as f:
        tree = ast.parse(f.read())
        
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                # Allowed loop variables: dx, dy
                if isinstance(node.target, ast.Name) and node.target.id in ['dx', 'dy']:
                    continue
                pytest.fail(f"Found forbidden for loop in math_engine.py: line {node.lineno}")
