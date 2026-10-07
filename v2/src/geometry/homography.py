import numpy as np
from .math_utils import adjugate_inverse

def plane_induced_homography(K: np.ndarray, R: np.ndarray, b: np.ndarray, d: float) -> np.ndarray:
    """
    Derivation: H_d = K ( R + (b @ e_z^T) / d ) K^-1
    Shapes: K (3, 3), R (3, 3), b (3,), d float -> (3, 3)
    """
    K_inv = adjugate_inverse(K)
    ez_T = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    b_col = b.reshape(3, 1)
    
    M = R + (b_col @ ez_T) / d
    return K @ M @ K_inv
