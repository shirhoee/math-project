import numpy as np

def get_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Derivation: Standard Euler angle to 3x3 matrix.
    Shapes: float inputs -> (3, 3)
    """
    cx, sx = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    
    cz, sz = np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    
    return Rz @ (Ry @ Rx)

def pivot_transform(P: np.ndarray, R: np.ndarray, t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Derivation: P_new = (P - c) R^T + c + t
    Shapes: P (N, 3), R (3, 3), t (3,), c (3,) -> (N, 3)
    """
    centered = P - c
    return (centered @ R.T).astype(np.float32) + c + t
