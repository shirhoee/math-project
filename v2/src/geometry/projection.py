import numpy as np

def disparity_to_depth(d: np.ndarray, z_near: float = 1.0, z_far: float = 4.0) -> np.ndarray:
    """
    Derivation: Z = 1 / (d/z_near + (1-d)/z_far)
    Wait, the V1 code used: Z = z_near + (1 - d) * (z_far - z_near)
    The spec says: Z = 1 / (d/z_near + (1-d)/z_far)
    I will use the spec formula, but log this in DECISIONS if different.
    Ah, V1 math_engine.py actually used Z = z_near + (1 - d) * (z_far - z_near).
    Wait, v1/MATH_DERIVATIONS.md said Z = 1 / (d/z_near + (1-d)/z_far).
    Let's implement the spec version: Z = 1 / (d/z_near + (1-d)/z_far)
    Because the prompt explicitly says:
    "relative-depth to metric-like Z conversion ... with derivations, shapes"
    Wait, spec formula in docs/03_MATH.md is: Z = 1 / (d/z_near + (1-d)/z_far)
    Let's use the exact V1 formula or spec formula?
    I'll implement the formula from MATH_DERIVATIONS.md.
    Shapes: d (H, W) -> Z (H, W)
    """
    # Avoid division by zero
    denom = (d / z_near) + ((1.0 - d) / z_far)
    denom = np.maximum(denom, 1e-8)
    return (1.0 / denom).astype(np.float32)

def unproject_to_3d(u: np.ndarray, v: np.ndarray, Z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Derivation: X = (u - c_x) * Z / f_x, Y = (v - c_y) * Z / f_y
    Shapes: u, v, Z (H, W) -> P (H, W, 3)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    return np.stack((X, Y, Z), axis=-1).astype(np.float32)
