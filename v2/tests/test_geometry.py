import numpy as np
from src.geometry.camera import intrinsics_from_fov
from src.geometry.projection import disparity_to_depth, unproject_to_3d
from src.geometry.transform import get_rotation_matrix, pivot_transform
from src.geometry.homography import plane_induced_homography
from src.geometry.math_utils import adjugate_inverse

def test_adjugate_inverse():
    np.random.seed(42)
    M = np.random.rand(3, 3).astype(np.float32)
    inv1 = adjugate_inverse(M)
    inv2 = np.linalg.inv(M)
    assert np.allclose(inv1, inv2, atol=1e-9)

def test_intrinsics_from_fov():
    K = intrinsics_from_fov(800, 600, 90.0)
    assert K.shape == (3, 3)
    assert K[0, 2] == 399.5
    assert K[1, 2] == 299.5
    assert np.allclose(K[0, 0], 400.0)

def test_disparity_to_depth():
    d = np.array([[0.0, 1.0], [0.5, 0.5]])
    z = disparity_to_depth(d, 1.0, 4.0)
    assert np.allclose(z[0, 0], 4.0)
    assert np.allclose(z[0, 1], 1.0)
    
def test_unproject_to_3d():
    K = intrinsics_from_fov(100, 100, 90.0)
    u = np.array([[50.0]])
    v = np.array([[50.0]])
    Z = np.array([[2.0]])
    P = unproject_to_3d(u, v, Z, K)
    assert P.shape == (1, 1, 3)
    # At cx, cy, X and Y should be near 0
    assert np.abs(P[0, 0, 0]) < 0.1
    assert np.abs(P[0, 0, 1]) < 0.1
    assert np.allclose(P[0, 0, 2], 2.0)

def test_get_rotation_matrix():
    R = get_rotation_matrix(0, 0, 0)
    assert np.allclose(R, np.eye(3))
    
def test_pivot_transform():
    P = np.array([[1.0, 2.0, 3.0]])
    R = np.eye(3, dtype=np.float32)
    t = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 0.0, 3.0])
    P_new = pivot_transform(P, R, t, c)
    assert np.allclose(P_new[0], [2.0, 2.0, 3.0])

def test_plane_induced_homography():
    K = np.eye(3, dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    d = 2.0
    H = plane_induced_homography(K, R, b, d)
    expected_M = np.eye(3, dtype=np.float32)
    expected_M[0, 2] = 0.5
    assert np.allclose(H, expected_M)
