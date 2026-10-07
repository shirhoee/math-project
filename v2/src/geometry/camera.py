import numpy as np

def intrinsics_from_fov(width: int, height: int, fov_deg: float) -> np.ndarray:
    """
    Derivation: f_x = (W/2) / tan(FOV/2)
    Shapes: Scalar inputs -> (3, 3) matrix
    """
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
