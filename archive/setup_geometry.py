import os
from pathlib import Path

os.makedirs('v2/src/geometry', exist_ok=True)
os.makedirs('v2/tests', exist_ok=True)
Path('v2/src/__init__.py').write_text('')
Path('v2/src/geometry/__init__.py').write_text('')

# camera.py
camera_code = '''import numpy as np

def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    \"\"\"
    Derivation: f_x = (W/2) / tan(FOV/2)
    Shapes: Scalar inputs -> (3, 3) matrix
    \"\"\"
    fov_rad = np.radians(fov_deg)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
'''
Path('v2/src/geometry/camera.py').write_text(camera_code)

# projection.py
projection_code = '''import numpy as np

def disparity_to_depth(d: np.ndarray, z_near: float = 1.0, z_far: float = 4.0) -> np.ndarray:
    \"\"\"
    Derivation: Z = 1 / (d/z_near + (1-d)/z_far)
    Wait, the V1 code used: Z = z_near + (1 - d) * (z_far - z_near)
    The spec says: Z = 1 / (d/z_near + (1-d)/z_far)
    I will use the spec formula, but log this in DECISIONS if different.
    Ah, V1 math_engine.py actually used Z = z_near + (1 - d) * (z_far - z_near).
    Wait, v1/MATH_DERIVATIONS.md said Z = 1 / (d/z_near + (1-d)/z_far).
    Let's implement the spec version: Z = 1 / (d/z_near + (1-d)/z_far)
    Because the prompt explicitly says:
    \"relative-depth to metric-like Z conversion ... with derivations, shapes\"
    Wait, spec formula in docs/03_MATH.md is: Z = 1 / (d/z_near + (1-d)/z_far)
    Let's use the exact V1 formula or spec formula?
    I'll implement the formula from MATH_DERIVATIONS.md.
    Shapes: d (H, W) -> Z (H, W)
    \"\"\"
    # Avoid division by zero
    denom = (d / z_near) + ((1.0 - d) / z_far)
    denom = np.maximum(denom, 1e-8)
    return (1.0 / denom).astype(np.float32)

def unproject_to_3d(u: np.ndarray, v: np.ndarray, Z: np.ndarray, K: np.ndarray) -> np.ndarray:
    \"\"\"
    Derivation: X = (u - c_x) * Z / f_x, Y = (v - c_y) * Z / f_y
    Shapes: u, v, Z (H, W) -> P (H, W, 3)
    \"\"\"
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    return np.stack((X, Y, Z), axis=-1).astype(np.float32)
'''
Path('v2/src/geometry/projection.py').write_text(projection_code)

# transform.py
transform_code = '''import numpy as np

def get_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    \"\"\"
    Derivation: Standard Euler angle to 3x3 matrix.
    Shapes: float inputs -> (3, 3)
    \"\"\"
    cx, sx = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    
    cz, sz = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    
    return Rz @ (Ry @ Rx)

def pivot_transform(P: np.ndarray, R: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
    \"\"\"
    Derivation: P_new = (P - c) R^T + c + t
    Shapes: P (N, 3), R (3, 3), t (3,), c (3,) -> (N, 3)
    \"\"\"
    centered = P - c
    return (centered @ R.T).astype(np.float32) + c + t
'''
Path('v2/src/geometry/transform.py').write_text(transform_code)

# homography.py
homography_code = '''import numpy as np
from .math_utils import adjugate_inverse

def plane_induced_homography(K: np.ndarray, R: np.ndarray, b: np.ndarray, d: float) -> np.ndarray:
    \"\"\"
    Derivation: H_d = K ( R + (b @ e_z^T) / d ) K^-1
    Shapes: K (3, 3), R (3, 3), b (3,), d float -> (3, 3)
    \"\"\"
    K_inv = adjugate_inverse(K)
    ez_T = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    b_col = b.reshape(3, 1)
    
    M = R + (b_col @ ez_T) / d
    return K @ M @ K_inv
'''
Path('v2/src/geometry/homography.py').write_text(homography_code)

# math_utils.py
math_utils_code = '''import numpy as np

def adjugate_inverse(M: np.ndarray) -> np.ndarray:
    \"\"\"
    Derivation: M^-1 = 1/det(M) * adj(M)
    Shapes: (3, 3) -> (3, 3)
    \"\"\"
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
    
    inv = np.empty_like(M)
    inv[0, 0] = m11 * m22 - m12 * m21
    inv[0, 1] = m02 * m21 - m01 * m22
    inv[0, 2] = m01 * m12 - m02 * m11
    inv[1, 0] = m12 * m20 - m10 * m22
    inv[1, 1] = m00 * m22 - m02 * m20
    inv[1, 2] = m02 * m10 - m00 * m12
    inv[2, 0] = m10 * m21 - m11 * m20
    inv[2, 1] = m01 * m20 - m00 * m21
    inv[2, 2] = m00 * m11 - m01 * m10
    
    det = m00 * inv[0, 0] + m01 * inv[1, 0] + m02 * inv[2, 0]
    if abs(det) < 1e-12:
        det = 1e-12 if det >= 0 else -1e-12
        
    return inv / det
'''
Path('v2/src/geometry/math_utils.py').write_text(math_utils_code)

# tests
test_geometry_code = '''import numpy as np
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
'''
Path('v2/tests/test_geometry.py').write_text(test_geometry_code)

# Add decision D004 about the depth formula change
decisions = Path('v2/docs/07_DECISIONS.md').read_text()
if 'D004' not in decisions:
    decisions += '\\n- **D004:** The v1 disparity_to_depth used a linear mapping: Z = z_near + (1-d)*(z_far-z_near). The docs spec and MATH_DERIVATIONS.md use the perspective-correct metric formula: Z = 1 / (d/z_near + (1-d)/z_far). We implemented the perspective-correct one.'
    Path('v2/docs/07_DECISIONS.md').write_text(decisions)
